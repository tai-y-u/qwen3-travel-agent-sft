#!/usr/bin/env python3
"""RAG microservice for the travel-guide knowledge base.

Exposes an HTTP API for querying travel guides stored in Milvus. Retrieval is
hybrid: a location-priority scalar filter first, then weighted Reciprocal Rank
Fusion over vector and keyword hits, then a score-threshold filter.

Routes (served on port 8010):
    GET  /health              service liveness and collection size
    POST /search              hybrid / vector / keyword / location search
    POST /search_by_location  scalar filter by province and/or city
    GET  /stats               collection totals and a sampled province histogram
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import jieba
from flask import Flask, jsonify, request
from openai import OpenAI
from pymilvus import Collection, connections

from paths import MILVUS_DB_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding API configuration. The API key is intentionally left empty here; it
# is filled in by the operator before starting the service.
DASHSCOPE_API_KEY = ""
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSIONS = 1024
# The embedding endpoint rejects longer inputs, so queries are truncated.
EMBEDDING_INPUT_CHAR_LIMIT = 8192

# Milvus configuration.
MILVUS_URI = str(MILVUS_DB_FILE)
COLLECTION_NAME = "travel_guides"
CONNECT_MAX_RETRIES = 5
CONNECT_INITIAL_RETRY_DELAY_SECONDS = 2
CONNECT_RETRY_BACKOFF_FACTOR = 2
DEFAULT_CONNECTION_ALIAS = "default"

# Fields fetched for every document-shaped result.
DOCUMENT_OUTPUT_FIELDS = ["city_code", "city_name", "province_name", "content"]
# Fields fetched when building the province/city lookup used for location matching.
LOCATION_OUTPUT_FIELDS = ["province_name", "city_name"]
# Vector index was built with inner product, so search must use the same metric.
VECTOR_SEARCH_PARAMS: dict[str, Any] = {"metric_type": "IP", "params": {}}
VECTOR_FIELD_NAME = "embedding"

# Retrieval tuning.
VECTOR_SIMILARITY_THRESHOLD = 0.35
KEYWORD_SCORE_THRESHOLD = 1
RRF_SCORE_THRESHOLD = 0.02
RRF_K = 60
# Rows pulled once to build the province/city vocabulary for location matching.
LOCATION_METADATA_LIMIT = 1000
# Rows sampled by /stats for the province histogram.
STATS_SAMPLE_LIMIT = 100
# Over-fetch factors: extra candidates are retrieved so threshold filtering has
# something left to return.
VECTOR_ONLY_OVERFETCH = 2
KEYWORD_ONLY_OVERFETCH = 2
HYBRID_OVERFETCH = 3

# Tokens shorter than this are dropped from keyword search.
MIN_KEYWORD_LENGTH = 2
# Milvus `like` treats '%' as a wildcard and has no escape for it, so tokens
# containing it cannot be matched literally.
MILVUS_LIKE_WILDCARD = "%"

SEARCH_TYPE_VECTOR = "vector"
SEARCH_TYPE_KEYWORD = "keyword"
SEARCH_TYPE_LOCATION = "location"
SEARCH_TYPE_HYBRID = "hybrid"

STRATEGY_LOCATION_PRIORITY = "location_priority"
STRATEGY_HYBRID_FALLBACK = "hybrid_fallback"
STRATEGY_ERROR_FALLBACK = "error_fallback"

# HTTP layer defaults.
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_LOCATION_SEARCH_LIMIT = 10
DEFAULT_VECTOR_WEIGHT = 1.0
DEFAULT_KEYWORD_WEIGHT = 1.5

SERVICE_HOST = "0.0.0.0"
SERVICE_PORT = 8010

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

app = Flask(__name__)


def _escape_milvus_string(value: str) -> str:
    """Escape a value for use inside a double-quoted Milvus expression literal.

    Args:
        value: Raw text to embed in an expression.

    Returns:
        The value with backslashes and double quotes escaped, so it cannot
        terminate the literal or inject further expression syntax.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_document(get_field: Callable[[str], Any]) -> dict[str, Any]:
    """Build the common document payload shared by every result shape.

    Args:
        get_field: Accessor for one output field of a Milvus row or hit entity.

    Returns:
        Dict with the document fields plus the combined ``location`` label.
    """
    province_name = get_field("province_name")
    city_name = get_field("city_name")
    return {
        "city_code": get_field("city_code"),
        "city_name": city_name,
        "province_name": province_name,
        "content": get_field("content"),
        "location": f"{province_name}-{city_name}",
    }


class RAGService:
    """Retrieval service over the ``travel_guides`` Milvus collection."""

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
        self.collection: Collection | None = None
        # Cached province/city vocabulary; see ``_get_location_index``.
        self._location_index: tuple[set[str], set[str]] | None = None
        self.setup_milvus()
        self.VECTOR_SIMILARITY_THRESHOLD = VECTOR_SIMILARITY_THRESHOLD
        self.KEYWORD_SCORE_THRESHOLD = KEYWORD_SCORE_THRESHOLD
        self.RRF_SCORE_THRESHOLD = RRF_SCORE_THRESHOLD
        # Load the Chinese segmentation dictionary up front so the first request
        # does not pay for it.
        jieba.initialize()

    def setup_milvus(self) -> None:
        """Connect to Milvus and load the collection, retrying with backoff.

        Raises:
            Exception: The last error from Milvus if every attempt fails.
        """
        max_retries = CONNECT_MAX_RETRIES
        retry_delay = CONNECT_INITIAL_RETRY_DELAY_SECONDS

        for attempt in range(max_retries):
            try:
                # Drop any half-open connection before reconnecting.
                try:
                    connections.disconnect(DEFAULT_CONNECTION_ALIAS)
                except Exception as e:
                    logger.debug(f"No existing Milvus connection to close: {e}")

                # Give the previous attempt time to settle before retrying.
                if attempt > 0:
                    logger.info(f"重试连接Milvus (第{attempt+1}次)...")
                    time.sleep(retry_delay)

                connections.connect(uri=MILVUS_URI)
                self.collection = Collection(COLLECTION_NAME)
                self.collection.load()
                # A fresh connection may expose different data.
                self.invalidate_location_cache()

                logger.info(f"成功连接Milvus集合: {COLLECTION_NAME}")
                logger.info(f"集合中的数据量: {self.collection.num_entities}")
                return

            except Exception as e:
                logger.warning(f"连接Milvus失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"连接Milvus失败，已重试{max_retries}次: {e}")
                    raise
                retry_delay *= CONNECT_RETRY_BACKOFF_FACTOR  # exponential backoff

    def ensure_connection(self) -> None:
        """Verify the Milvus connection is usable, reconnecting if it is not."""
        try:
            if self.collection is None:
                self.setup_milvus()
            else:
                # Cheap round trip that fails if the connection has dropped.
                _ = self.collection.num_entities
        except Exception as e:
            logger.warning(f"检测到连接问题，重新连接: {e}")
            self.setup_milvus()

    def get_embedding(self, text: str) -> list[float] | None:
        """Embed ``text`` with the configured embedding model.

        Args:
            text: Text to embed; truncated to the model's input limit.

        Returns:
            The embedding vector, or ``None`` if the API call failed.
        """
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text[:EMBEDDING_INPUT_CHAR_LIMIT],
                dimensions=EMBEDDING_DIMENSIONS,
                encoding_format="float",
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"获取embedding失败: {e}")
            return None

    def extract_keywords(self, text: str) -> list[str]:
        """Segment ``text`` into search keywords.

        Args:
            text: Raw query text.

        Returns:
            Segmented tokens with stop words and single characters removed.
        """
        words = jieba.cut_for_search(text)
        # Chinese stop-word list (data, not prose - do not translate).
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '里', '啊', '哦', '哈', '吧', '呢', '吗',"酒店"}
        keywords = [
            word.strip()
            for word in words
            if len(word.strip()) >= MIN_KEYWORD_LENGTH and word.strip() not in stop_words
        ]
        return keywords

    def invalidate_location_cache(self) -> None:
        """Discard the cached province/city vocabulary.

        Call this after the collection's contents change so the next location
        search rebuilds the vocabulary from Milvus.
        """
        self._location_index = None

    def _get_location_index(self) -> tuple[set[str], set[str]]:
        """Return the cached ``(provinces, cities)`` vocabulary from Milvus.

        The vocabulary is read once per connection and reused, because the
        underlying metadata query is a fixed scan of up to
        ``LOCATION_METADATA_LIMIT`` rows and the collection is read-only while
        the service runs.

        Returns:
            Sets of the distinct province and city names present in the
            collection.

        Raises:
            Exception: Propagates Milvus query failures to the caller.
        """
        if self._location_index is not None:
            return self._location_index

        self.ensure_connection()
        rows = self.collection.query(
            expr="",
            output_fields=LOCATION_OUTPUT_FIELDS,
            limit=LOCATION_METADATA_LIMIT,  # enough rows to cover every location
        )

        provinces: set[str] = set()
        cities: set[str] = set()
        for row in rows:
            if row.get("province_name"):
                provinces.add(row.get("province_name"))
            if row.get("city_name"):
                cities.add(row.get("city_name"))

        self._location_index = (provinces, cities)
        logger.info(
            f"Cached location vocabulary: {len(provinces)} provinces, {len(cities)} cities"
        )
        return self._location_index

    def extract_locations(self, query: str) -> dict[str, list[str]]:
        """Find province and city names mentioned in ``query``.

        Args:
            query: Free-text user query.

        Returns:
            Dict with ``provinces`` and ``cities`` lists of matched names; both
            empty if nothing matched or the lookup failed.
        """
        try:
            provinces, cities = self._get_location_index()

            found_provinces = []
            found_cities = []

            # Match either the full stored name or the name with its
            # administrative suffix (province/city/county) stripped off.
            for province in provinces:
                if province in query or province.replace('省', '').replace('市', '').replace('自治区', '') in query:
                    found_provinces.append(province)

            for city in cities:
                if city in query or city.replace('市', '').replace('县', '') in query:
                    found_cities.append(city)

            return {
                "provinces": found_provinces,
                "cities": found_cities,
            }

        except Exception as e:
            logger.error(f"位置提取失败: {e}")
            return {"provinces": [], "cities": []}

    def keyword_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search documents by keyword occurrence in their content.

        Args:
            query: Free-text user query, segmented into keywords.
            limit: Maximum number of documents to fetch from Milvus.

        Returns:
            Matching documents with a term-frequency ``keyword_score`` and the
            keywords that matched, sorted by score descending. Empty on failure.
        """
        try:
            self.ensure_connection()

            keywords = self.extract_keywords(query)
            if not keywords:
                return []

            # Build one `like` clause per keyword against the content field.
            # Values are escaped so a keyword can never terminate the string
            # literal and inject expression syntax. Milvus offers no escape for
            # its '%' wildcard, so such tokens cannot be matched literally and
            # are dropped rather than silently widened into a wildcard.
            keyword_exprs = []
            for keyword in keywords:
                if MILVUS_LIKE_WILDCARD in keyword:
                    logger.warning(
                        f"Dropping keyword containing the Milvus wildcard "
                        f"'{MILVUS_LIKE_WILDCARD}': {keyword!r}"
                    )
                    continue
                keyword_exprs.append(
                    f'content like "%{_escape_milvus_string(keyword)}%"'
                )

            if not keyword_exprs:
                logger.warning(f"No usable keywords left for query: {query!r}")
                return []

            # OR the clauses together: matching a single keyword is enough.
            expr = " or ".join(keyword_exprs)

            rows = self.collection.query(
                expr=expr,
                output_fields=DOCUMENT_OUTPUT_FIELDS,
                limit=limit,
            )

            scored_results = []
            for row in rows:
                content = row.get("content", "").lower()
                keyword_score = 0
                matched_keywords = []

                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in content:
                        # Plain term frequency: how often the keyword occurs.
                        keyword_score += content.count(keyword_lower)
                        matched_keywords.append(keyword)

                document = _format_document(row.get)
                location = document.pop("location")
                document["keyword_score"] = keyword_score
                document["matched_keywords"] = matched_keywords
                document["location"] = location
                scored_results.append(document)

            scored_results.sort(key=lambda item: item["keyword_score"], reverse=True)
            return scored_results

        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []

    def reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        k: int = RRF_K,
    ) -> list[dict[str, Any]]:
        """Fuse vector and keyword result lists with weighted RRF.

        Each list contributes ``weight / (k + rank + 1)`` per document; the
        contributions are summed and the documents re-sorted.

        Args:
            vector_results: Vector hits, best first.
            keyword_results: Keyword hits, best first.
            vector_weight: Weight applied to vector contributions.
            keyword_weight: Weight applied to keyword contributions.
            k: RRF smoothing constant.

        Returns:
            Documents sorted by fused ``rrf_score`` descending, annotated with
            per-list ranks, scores and a ``fusion_info`` breakdown.
        """
        rrf_score_by_id: defaultdict[str, float] = defaultdict(float)
        document_by_id: dict[str, dict[str, Any]] = {}

        # Vector hits, weighted by vector_weight.
        for rank, vector_result in enumerate(vector_results):
            document_id = f"{vector_result['city_code']}_{vector_result['city_name']}"
            vector_rrf_score = vector_weight * (1.0 / (k + rank + 1))
            rrf_score_by_id[document_id] += vector_rrf_score
            document_by_id[document_id] = vector_result
            document_by_id[document_id]['vector_rank'] = rank + 1
            document_by_id[document_id]['vector_score'] = vector_result.get('score', 0)
            document_by_id[document_id]['vector_rrf_score'] = vector_rrf_score

        # Keyword hits, weighted by keyword_weight.
        for rank, keyword_result in enumerate(keyword_results):
            document_id = f"{keyword_result['city_code']}_{keyword_result['city_name']}"
            keyword_rrf_score = keyword_weight * (1.0 / (k + rank + 1))
            rrf_score_by_id[document_id] += keyword_rrf_score
            if document_id not in document_by_id:
                document_by_id[document_id] = keyword_result
            document_by_id[document_id]['keyword_rank'] = rank + 1
            document_by_id[document_id]['keyword_score'] = keyword_result.get('keyword_score', 0)
            document_by_id[document_id]['keyword_rrf_score'] = keyword_rrf_score
            document_by_id[document_id]['matched_keywords'] = keyword_result.get('matched_keywords', [])

        ranked_id_scores = sorted(rrf_score_by_id.items(), key=lambda item: item[1], reverse=True)

        fused_results = []
        for document_id, total_rrf_score in ranked_id_scores:
            fused_result = document_by_id[document_id].copy()
            fused_result['rrf_score'] = total_rrf_score
            fused_result['search_type'] = SEARCH_TYPE_HYBRID
            # Weight breakdown, kept for debugging.
            fused_result['fusion_info'] = {
                'vector_weight': vector_weight,
                'keyword_weight': keyword_weight,
                'vector_rrf_contribution': fused_result.get('vector_rrf_score', 0),
                'keyword_rrf_contribution': fused_result.get('keyword_rrf_score', 0),
            }
            fused_results.append(fused_result)

        return fused_results

    def location_priority_search(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Search by any province/city mentioned in ``query``, if any.

        Args:
            query: Free-text user query.
            limit: Maximum number of documents to return.

        Returns:
            Dict with the ``search_strategy`` used, the ``matched_locations``
            found, and the ``results`` list (empty when no location matched or
            the location lookup returned nothing).
        """
        try:
            # 1. Find the locations named in the query.
            locations = self.extract_locations(query)
            found_provinces = locations["provinces"]
            found_cities = locations["cities"]

            # 2. When a location is named, search by location first.
            if found_provinces or found_cities:
                location_results = []

                for province in found_provinces:
                    province_results = self.search_by_location(province=province, limit=limit)
                    for result in province_results:
                        result["match_type"] = "province_match"
                        result["matched_location"] = province
                        result["location_score"] = 1.0
                    location_results.extend(province_results)

                for city in found_cities:
                    city_results = self.search_by_location(city=city, limit=limit)
                    for result in city_results:
                        result["match_type"] = "city_match"
                        result["matched_location"] = city
                        result["location_score"] = 1.0
                    location_results.extend(city_results)

                # Deduplicate on city_code + city_name.
                seen_document_ids = set()
                unique_results = []
                for result in location_results:
                    document_id = f"{result.get('city_code')}_{result.get('city_name')}"
                    if document_id not in seen_document_ids:
                        seen_document_ids.add(document_id)
                        unique_results.append(result)

                if unique_results:
                    return {
                        "search_strategy": STRATEGY_LOCATION_PRIORITY,
                        "matched_locations": {
                            "provinces": found_provinces,
                            "cities": found_cities,
                        },
                        "results": unique_results[:limit],
                    }

            # 3. No location found, or the location search came back empty.
            return {
                "search_strategy": STRATEGY_HYBRID_FALLBACK,
                "matched_locations": {
                    "provinces": found_provinces,
                    "cities": found_cities,
                },
                "results": [],
            }

        except Exception as e:
            logger.error(f"位置优先搜索失败: {e}")
            return {
                "search_strategy": STRATEGY_ERROR_FALLBACK,
                "matched_locations": {"provinces": [], "cities": []},
                "results": [],
            }

    def filter_by_threshold(
        self, results: list[dict[str, Any]], search_type: str
    ) -> list[dict[str, Any]]:
        """Drop results scoring below the threshold for ``search_type``.

        Args:
            results: Candidate results to filter.
            search_type: ``"vector"``, ``"keyword"`` or anything else (hybrid).

        Returns:
            The kept results; each inspected result gains a ``filter_reason``
            describing the comparison that was made.
        """
        if not results:
            return results

        filtered_results = []

        for result in results:
            should_keep = False

            if search_type == SEARCH_TYPE_VECTOR:
                vector_score = result.get("score", 0)
                if vector_score >= self.VECTOR_SIMILARITY_THRESHOLD:
                    should_keep = True
                    result["filter_reason"] = f"vector_score({vector_score:.3f}) >= threshold({self.VECTOR_SIMILARITY_THRESHOLD})"
                else:
                    result["filter_reason"] = f"vector_score({vector_score:.3f}) < threshold({self.VECTOR_SIMILARITY_THRESHOLD})"

            elif search_type == SEARCH_TYPE_KEYWORD:
                keyword_score = result.get("keyword_score", 0)
                if keyword_score >= self.KEYWORD_SCORE_THRESHOLD:
                    should_keep = True
                    result["filter_reason"] = f"keyword_score({keyword_score}) >= threshold({self.KEYWORD_SCORE_THRESHOLD})"
                else:
                    result["filter_reason"] = f"keyword_score({keyword_score}) < threshold({self.KEYWORD_SCORE_THRESHOLD})"

            else:  # hybrid results are filtered on their fused RRF score
                rrf_score = result.get("rrf_score", 0)
                if rrf_score >= self.RRF_SCORE_THRESHOLD:
                    should_keep = True
                    result["filter_reason"] = f"rrf_score({rrf_score:.4f}) >= threshold({self.RRF_SCORE_THRESHOLD})"
                else:
                    result["filter_reason"] = f"rrf_score({rrf_score:.4f}) < threshold({self.RRF_SCORE_THRESHOLD})"

            if should_keep:
                filtered_results.append(result)

        return filtered_results

    def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        search_type: str = SEARCH_TYPE_HYBRID,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        use_threshold: bool = True,
    ) -> list[dict[str, Any]]:
        """Run a search: location priority, hybrid fallback, threshold filter.

        Note:
            These defaults are the same constants ``POST /search`` falls back to,
            so calling this method directly gives the same results as the HTTP
            endpoint. They previously disagreed (``limit=1``,
            ``vector_weight=0.7``), which silently made programmatic callers
            behave differently from the API.

        Args:
            query: Free-text user query.
            limit: Maximum number of results to return.
            search_type: ``"vector"``, ``"keyword"``, ``"location"``, or
                ``"hybrid"`` (anything else is treated as hybrid).
            vector_weight: RRF weight for vector hits in hybrid mode.
            keyword_weight: RRF weight for keyword hits in hybrid mode.
            use_threshold: Whether to apply the score-threshold filter.

        Returns:
            Up to ``limit`` results; empty on failure.
        """
        try:
            if search_type == SEARCH_TYPE_VECTOR:
                # Over-fetch so threshold filtering still has candidates left.
                results = self.vector_search(query, limit * VECTOR_ONLY_OVERFETCH)
            elif search_type == SEARCH_TYPE_KEYWORD:
                results = self.keyword_search(query, limit * KEYWORD_ONLY_OVERFETCH)
            elif search_type == SEARCH_TYPE_LOCATION:
                location_result = self.location_priority_search(query, limit)
                return location_result["results"]
            else:
                location_result = self.location_priority_search(query, limit)

                if location_result["results"]:
                    # Location search hit; return it directly, unfiltered.
                    for result in location_result["results"]:
                        result["search_strategy"] = location_result["search_strategy"]
                        result["matched_locations"] = location_result["matched_locations"]
                    return location_result["results"]

                # Location search missed; fall back to hybrid retrieval.
                logger.info(f"位置搜索无结果，降级到混合检索: {query}")

                vector_results = self.vector_search(query, limit * HYBRID_OVERFETCH)
                keyword_results = self.keyword_search(query, limit * HYBRID_OVERFETCH)

                if not vector_results and not keyword_results:
                    return []
                if not vector_results:
                    results = keyword_results
                    search_type = SEARCH_TYPE_KEYWORD
                elif not keyword_results:
                    results = vector_results
                    search_type = SEARCH_TYPE_VECTOR
                else:
                    results = self.reciprocal_rank_fusion(
                        vector_results,
                        keyword_results,
                        vector_weight,
                        keyword_weight,
                    )

                # Record which strategy produced these results.
                for result in results:
                    result["search_strategy"] = STRATEGY_HYBRID_FALLBACK
                    result["matched_locations"] = location_result["matched_locations"]

            if use_threshold:
                results = self.filter_by_threshold(results, search_type)
                logger.info(f"阈值过滤后剩余结果数: {len(results)}")

            return results[:limit]

        except Exception as e:
            logger.error(f"智能搜索失败: {e}")
            return []

    def vector_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search by embedding similarity only.

        Args:
            query: Free-text user query, embedded before searching.
            limit: Maximum number of hits to return.

        Returns:
            Hits with their similarity ``score``; empty if embedding or search
            failed.
        """
        try:
            self.ensure_connection()

            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []

            hits_per_query = self.collection.search(
                data=[query_embedding],
                anns_field=VECTOR_FIELD_NAME,
                param=VECTOR_SEARCH_PARAMS,
                limit=limit,
                output_fields=DOCUMENT_OUTPUT_FIELDS,
            )

            formatted_results = []
            for hits in hits_per_query:
                for hit in hits:
                    document = _format_document(hit.entity.get)
                    document["score"] = float(hit.score)
                    document["search_type"] = SEARCH_TYPE_VECTOR
                    formatted_results.append(document)

            return formatted_results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    def search_by_location(
        self, province: str | None = None, city: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Fetch documents filtered by province and/or city.

        Args:
            province: Exact province name to match, if given.
            city: City name to match as a substring, if given.
            limit: Maximum number of documents to return.

        Returns:
            Matching documents; empty on failure.
        """
        try:
            self.ensure_connection()

            # Province matches exactly; city matches as a substring.
            filter_exprs = []
            if province:
                filter_exprs.append(f'province_name == "{_escape_milvus_string(province)}"')
            if city:
                filter_exprs.append(f'city_name like "%{_escape_milvus_string(city)}%"')

            expr = " and ".join(filter_exprs) if filter_exprs else ""

            rows = self.collection.query(
                expr=expr,
                output_fields=DOCUMENT_OUTPUT_FIELDS,
                limit=limit,
            )

            return [_format_document(row.get) for row in rows]

        except Exception as e:
            logger.error(f"按位置搜索失败: {e}")
            return []


# Global service instance, created on first request.
rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Return the shared :class:`RAGService`, creating it on first use."""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service


@app.route('/health', methods=['GET'])
def health_check():
    """Report service liveness and the size of the loaded collection."""
    try:
        service = get_rag_service()
        return jsonify({
            "status": "healthy",
            "service": "RAG Travel Guide API",
            "collection_size": service.collection.num_entities if service.collection else 0
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/search', methods=['POST'])
def search():
    """Search travel guides, with optional search type and RRF weights."""
    try:
        payload = request.get_json()
        if not payload or 'query' not in payload:
            return jsonify({
                "error": "缺少query参数",
                "status": STATUS_ERROR
            }), 400

        query = payload['query']
        limit = payload.get('limit', DEFAULT_SEARCH_LIMIT)
        search_type = payload.get('search_type', SEARCH_TYPE_HYBRID)

        vector_weight = payload.get('vector_weight', DEFAULT_VECTOR_WEIGHT)
        keyword_weight = payload.get('keyword_weight', DEFAULT_KEYWORD_WEIGHT)

        if not query.strip():
            return jsonify({
                "error": "查询内容不能为空",
                "status": STATUS_ERROR
            }), 400

        service = get_rag_service()
        results = service.search(query, limit, search_type, vector_weight, keyword_weight)

        # Surface the strategy and locations the retrieval layer actually used.
        search_strategy = SEARCH_TYPE_HYBRID
        matched_locations = {"provinces": [], "cities": []}

        if results:
            first_result = results[0]
            search_strategy = first_result.get("search_strategy", SEARCH_TYPE_HYBRID)
            matched_locations = first_result.get("matched_locations", {"provinces": [], "cities": []})

        return jsonify({
            "status": STATUS_SUCCESS,
            "query": query,
            "search_type": search_type,
            "search_strategy": search_strategy,
            "matched_locations": matched_locations,
            "weights": {
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight
            },
            "limit": limit,
            "results_count": len(results),
            "results": results
        })

    except Exception as e:
        logger.error(f"搜索接口错误: {e}")
        return jsonify({
            "error": str(e),
            "status": STATUS_ERROR
        }), 500


@app.route('/search_by_location', methods=['POST'])
def search_by_location():
    """Search travel guides by province and/or city."""
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({
                "error": "请提供搜索参数",
                "status": STATUS_ERROR
            }), 400

        province = payload.get('province')
        city = payload.get('city')
        limit = payload.get('limit', DEFAULT_LOCATION_SEARCH_LIMIT)

        if not province and not city:
            return jsonify({
                "error": "请至少提供省份或城市参数",
                "status": STATUS_ERROR
            }), 400

        service = get_rag_service()
        results = service.search_by_location(province, city, limit)

        return jsonify({
            "status": STATUS_SUCCESS,
            "filters": {
                "province": province,
                "city": city
            },
            "limit": limit,
            "results_count": len(results),
            "results": results
        })

    except Exception as e:
        logger.error(f"按位置搜索接口错误: {e}")
        return jsonify({
            "error": str(e),
            "status": STATUS_ERROR
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Report the collection total and a sampled province distribution."""
    try:
        service = get_rag_service()
        total_count = service.collection.num_entities

        # Province histogram over a small sample, not the whole collection.
        sample_rows = service.collection.query(
            expr="",
            output_fields=["province_name"],
            limit=STATS_SAMPLE_LIMIT
        )

        province_counts: dict[str, int] = {}
        for row in sample_rows:
            province = row.get("province_name")
            if province:
                province_counts[province] = province_counts.get(province, 0) + 1

        return jsonify({
            "status": STATUS_SUCCESS,
            "total_travel_guides": total_count,
            "sample_province_distribution": province_counts,
            "collection_name": COLLECTION_NAME
        })

    except Exception as e:
        logger.error(f"获取统计信息错误: {e}")
        return jsonify({
            "error": str(e),
            "status": STATUS_ERROR
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Return the list of available endpoints for unknown routes."""
    return jsonify({
        "error": "接口不存在",
        "status": STATUS_ERROR,
        "available_endpoints": [
            "GET /health - 健康检查",
            "POST /search - 混合搜索 (支持vector/keyword/hybrid模式)",
            "POST /search_by_location - 按位置搜索",
            "GET /stats - 统计信息"
        ]
    }), 404


if __name__ == '__main__':
    print("启动RAG旅游攻略查询服务...")
    print("可用接口:")
    print("- GET  /health - 健康检查")
    print("- POST /search - 混合搜索 (支持vector/keyword/hybrid模式)")
    print("- POST /search_by_location - 按位置搜索") 
    print("- GET  /stats - 统计信息")

    # Threaded mode so concurrent requests are not serialized.
    app.run(host=SERVICE_HOST, port=SERVICE_PORT, debug=False, use_reloader=False, threaded=True)
