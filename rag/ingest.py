#!/usr/bin/env python3
"""Bulk-import travel guides into the Milvus vector database.

Guide text files are embedded with the Alibaba Cloud Qwen embedding model,
computed concurrently across a thread pool, and inserted in batches.

WARNING - this importer is destructive and expensive. Every run drops the
existing ``travel_guides`` collection and re-embeds every guide file from
scratch, so all documents are re-billed against the paid embedding API. There is
no incremental mode; only run it when you intend to rebuild the collection.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from tqdm import tqdm

from paths import (
    CITY_CODE_MAPPING_FILE,
    MILVUS_DB_FILE,
    TRAVEL_GUIDES_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding API configuration. The API key is intentionally left empty here; it
# is filled in by the operator before running the import.
DASHSCOPE_API_KEY = ""
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSIONS = 1024
# The embedding endpoint rejects longer inputs, so guide text is truncated.
EMBEDDING_INPUT_CHAR_LIMIT = 8192

# Milvus configuration (local Milvus Lite file).
MILVUS_URI = str(MILVUS_DB_FILE)
COLLECTION_NAME = "travel_guides"

# Guide files are named "<city_code>_<city_name>_travel_guide.txt".
GUIDE_FILENAME_SUFFIX = "_travel_guide.txt"
GUIDE_FILE_GLOB = f"*{GUIDE_FILENAME_SUFFIX}"
FILENAME_SEPARATOR = "_"
MIN_FILENAME_PARTS = 2

# Field lengths of the collection schema.
CITY_CODE_MAX_LENGTH = 10
CITY_NAME_MAX_LENGTH = 50
PROVINCE_NAME_MAX_LENGTH = 20
CONTENT_MAX_LENGTH = 65535
PRIMARY_KEY_MAX_LENGTH = 100
VECTOR_FIELD_NAME = "embedding"
CONSISTENCY_LEVEL = "Bounded"
VECTOR_INDEX_PARAMS: dict[str, Any] = {
    "index_type": "AUTOINDEX",
    "metric_type": "IP",
    "params": {},
}
VECTOR_SEARCH_PARAMS: dict[str, Any] = {"metric_type": "IP", "params": {}}
# Column order used when inserting; must match the schema field order.
INSERT_FIELD_ORDER = ("city_code", "city_name", "province_name", "content", "embedding")

# Concurrency and batching.
MAX_WORKERS = 10
INSERT_BATCH_SIZE = 50

# Post-import smoke test.
SMOKE_TEST_LIMIT = 3
SMOKE_TEST_VECTOR_VALUE = 0.1


class TravelGuideImporter:
    """Embeds travel-guide files and loads them into Milvus.

    WARNING: :meth:`setup_milvus` drops and recreates the collection, so
    importing always rebuilds it from scratch.
    """

    def __init__(self) -> None:
        # Each worker thread builds its own OpenAI client for thread safety.
        self.city_mapping: dict[str, dict[str, Any]] = {}
        self.load_city_mapping()
        self._lock = threading.Lock()  # guards the progress counters

    def load_city_mapping(self) -> None:
        """Load the city-code to province mapping from the config file.

        Raises:
            FileNotFoundError: If the mapping file is missing.
            KeyError: If the mapping file lacks the expected top-level key.
        """
        with CITY_CODE_MAPPING_FILE.open('r', encoding='utf-8') as f:
            data = json.load(f)
            self.city_mapping = data['城市编码映射']
        print(f"已加载 {len(self.city_mapping)} 个城市映射")

    def _create_client(self) -> OpenAI:
        """Create an embedding API client (one per thread, for thread safety)."""
        return OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL
        )

    def get_embedding(self, text: str, client: OpenAI | None = None) -> list[float] | None:
        """Embed ``text`` with the configured embedding model.

        Args:
            text: Text to embed; truncated to the model's input limit.
            client: Client to reuse; a new one is created when omitted.

        Returns:
            The embedding vector, or ``None`` if the API call failed.
        """
        if client is None:
            client = self._create_client()

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text[:EMBEDDING_INPUT_CHAR_LIMIT],
                dimensions=EMBEDDING_DIMENSIONS,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取embedding失败: {e}")
            return None

    def setup_milvus(self) -> Collection:
        """Connect to Milvus and create a fresh collection with its index.

        WARNING: this is destructive. An existing collection named
        ``travel_guides`` is dropped, discarding every stored document and
        embedding, so all guides must be re-embedded against the paid API.

        Returns:
            The newly created, indexed collection.
        """
        connections.connect(uri=MILVUS_URI)

        # Destructive: the previous collection and all of its embeddings go away.
        if utility.has_collection(COLLECTION_NAME):
            logger.warning(
                f"Dropping existing collection {COLLECTION_NAME!r}: all stored "
                f"embeddings are discarded and every guide will be re-embedded "
                f"against the paid embedding API."
            )
            Collection(COLLECTION_NAME).drop()
            print(f"删除已存在的集合: {COLLECTION_NAME}")

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=PRIMARY_KEY_MAX_LENGTH),
            FieldSchema(name="city_code", dtype=DataType.VARCHAR, max_length=CITY_CODE_MAX_LENGTH),
            FieldSchema(name="city_name", dtype=DataType.VARCHAR, max_length=CITY_NAME_MAX_LENGTH),
            FieldSchema(name="province_name", dtype=DataType.VARCHAR, max_length=PROVINCE_NAME_MAX_LENGTH),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=CONTENT_MAX_LENGTH),
            FieldSchema(name=VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSIONS),
        ]

        schema = CollectionSchema(fields, description="旅游攻略向量数据库")
        collection = Collection(COLLECTION_NAME, schema, consistency_level=CONSISTENCY_LEVEL)

        collection.create_index(VECTOR_FIELD_NAME, VECTOR_INDEX_PARAMS)
        print(f"创建集合和索引: {COLLECTION_NAME}")

        return collection

    def parse_filename(self, filename: str) -> dict[str, str] | None:
        """Derive city information from a guide file name.

        Args:
            filename: Path or bare name of a ``*_travel_guide.txt`` file.

        Returns:
            Dict with ``city_code``, ``city_name`` and ``province_name``, or
            ``None`` if the name does not parse or the code is unknown.
        """
        basename = Path(filename).name
        # Expected form: <city_code>_<city_name>_travel_guide.txt
        parts = basename.replace(GUIDE_FILENAME_SUFFIX, '').split(FILENAME_SEPARATOR)
        if len(parts) >= MIN_FILENAME_PARTS:
            city_code = parts[0]
            city_name = FILENAME_SEPARATOR.join(parts[1:])

            # The province comes from the city-code mapping.
            if city_code in self.city_mapping:
                province_name = self.city_mapping[city_code]['province']
                return {
                    'city_code': city_code,
                    'city_name': city_name,
                    'province_name': province_name
                }

        return None

    def process_single_file(self, file_path: str) -> tuple[bool, Any]:
        """Read and embed one guide file. Runs on a worker thread.

        Args:
            file_path: Path to a ``*_travel_guide.txt`` file.

        Returns:
            ``(True, record)`` with the insertable record on success, or
            ``(False, message)`` describing why the file was skipped.
        """
        # Each thread uses its own client.
        client = self._create_client()

        try:
            city_info = self.parse_filename(file_path)
            if not city_info:
                return False, f"无法解析文件名: {file_path}"

            with Path(file_path).open('r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                return False, f"文件内容为空: {file_path}"

            embedding = self.get_embedding(content, client)
            if not embedding:
                return False, f"无法获取embedding: {file_path}"

            record = {
                'city_code': city_info['city_code'],
                'city_name': city_info['city_name'],
                'province_name': city_info['province_name'],
                'content': content,
                'embedding': embedding
            }

            return True, record

        except Exception as e:
            return False, f"处理文件失败 {file_path}: {e}"

    def insert_batch_data(self, collection: Collection, batch_records: list[dict]) -> None:
        """Insert a batch of records into Milvus.

        Args:
            collection: Target collection.
            batch_records: Records keyed by field name, as returned by
                :meth:`process_single_file`.
        """
        if not batch_records:
            return

        # Milvus expects column-oriented data in schema field order.
        columns = [
            [record[field_name] for record in batch_records] for field_name in INSERT_FIELD_ORDER
        ]

        collection.insert(columns)

    def process_travel_guides(self, data_dir: str | Path = TRAVEL_GUIDES_DIR) -> Collection:
        """Embed and import every guide file in ``data_dir``, concurrently.

        WARNING: destructive. This recreates the collection via
        :meth:`setup_milvus` before importing, dropping any existing data.

        Args:
            data_dir: Directory holding the ``*_travel_guide.txt`` files.
                Defaults to :data:`paths.TRAVEL_GUIDES_DIR`.

        Returns:
            The loaded collection.
        """
        guide_file_paths = sorted(str(path) for path in Path(data_dir).glob(GUIDE_FILE_GLOB))
        print(f"找到 {len(guide_file_paths)} 个旅游攻略文件")

        collection = self.setup_milvus()

        batch_size = INSERT_BATCH_SIZE
        batch_records: list[dict] = []
        processed_count = 0
        failed_count = 0

        print(f"开始使用 {MAX_WORKERS} 个线程并发处理...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file_path = {executor.submit(self.process_single_file, file_path): file_path
                             for file_path in guide_file_paths}

            with tqdm(total=len(guide_file_paths), desc="处理旅游攻略") as progress_bar:
                for future in as_completed(future_to_file_path):
                    file_path = future_to_file_path[future]
                    try:
                        success, record_or_error = future.result()

                        if success:
                            batch_records.append(record_or_error)

                            # Flush once a full batch has accumulated.
                            if len(batch_records) >= batch_size:
                                try:
                                    self.insert_batch_data(collection, batch_records)
                                    with self._lock:
                                        processed_count += len(batch_records)
                                    batch_records = []
                                    print(f"已处理 {processed_count} 条记录")
                                except Exception as e:
                                    print(f"插入数据失败: {e}")
                                    batch_records = []
                        else:
                            with self._lock:
                                failed_count += 1
                            print(f"处理失败: {record_or_error}")

                    except Exception as e:
                        with self._lock:
                            failed_count += 1
                        print(f"处理文件时出现异常 {file_path}: {e}")

                    progress_bar.update(1)

        # Flush whatever is left in the final partial batch.
        if batch_records:
            try:
                self.insert_batch_data(collection, batch_records)
                processed_count += len(batch_records)
                print(f"已处理 {processed_count} 条记录")
            except Exception as e:
                print(f"插入最后批次数据失败: {e}")

        collection.load()
        print(f"\n=== 处理完成 ===")
        print(f"成功处理: {processed_count} 条记录")
        print(f"失败: {failed_count} 个文件")
        print(f"集合中实际记录数: {collection.num_entities}")

        return collection


def main() -> None:
    """Rebuild the travel-guide collection, then run a smoke-test query.

    WARNING: destructive. This drops the existing ``travel_guides`` collection
    and re-embeds every guide file, which re-charges the paid embedding API for
    the whole corpus.

    Raises:
        Exception: Re-raised after logging whatever failed during the import.
    """
    importer = TravelGuideImporter()

    try:
        collection = importer.process_travel_guides()

        print("\n=== 数据验证 ===")
        print(f"集合名称: {collection.name}")
        print(f"数据总数: {collection.num_entities}")

        # Smoke-test query with a constant vector, just to prove search works.
        results = collection.search(
            data=[[SMOKE_TEST_VECTOR_VALUE] * EMBEDDING_DIMENSIONS],
            anns_field=VECTOR_FIELD_NAME,
            param=VECTOR_SEARCH_PARAMS,
            limit=SMOKE_TEST_LIMIT,
            output_fields=["city_name", "province_name"]
        )

        print("查询测试结果:")
        for hits in results:
            for hit in hits:
                print(f"- {hit.entity.get('province_name')}-{hit.entity.get('city_name')} (分数: {hit.score:.4f})")

    except Exception as e:
        print(f"导入过程出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
