#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Turn the seed dataset into multi-turn tool-calling conversations.

Each seed record is replayed through :class:`TravelAssistantFuncCall`: the
assistant's own tool calls are intercepted, deduplicated, and rewritten as
OpenAI-style ``assistant``/``tool`` message pairs. The result is a conversation
that can be used directly as a supervised fine-tuning sample.

Conversions run on a thread pool and are flushed to numbered batch files as they
complete, so a long run leaves usable output even if it is interrupted.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from agent.assistant import TravelAssistantFuncCall
from paths import PROCESSED_DIR, SEED_DIR, ensure_dir

logger = logging.getLogger(__name__)

DEFAULT_INPUT_FILE = SEED_DIR / "travel_assistant_train_dataset.json"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "converted_datasets"
# The command line uses a distinct default so an ad-hoc run cannot overwrite the
# batches produced by a library caller.
CLI_DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "converted_datasets_final"

DEFAULT_NUM_THREADS = 8
# Conversations are flushed to a new batch file once this many have accumulated.
BATCH_SAVE_SIZE = 20
# The batch counter also advances every this many processed records, which
# leaves gaps in the file numbering. Kept as-is: existing output relies on it.
BATCH_ROLLOVER_INTERVAL = 100
BATCH_FILENAME_TEMPLATE = "converted_batch_{batch_num:03d}.json"
JSON_INDENT = 2

# Hotel recommendations must be followed by a review lookup, so when both tools
# fire the calls are emitted as separate turns instead of one parallel block.
HOTEL_TOOL_NAMES = frozenset({"recommend_hotels", "get_hotel_reviews"})

CALL_ID_PREFIX = "call_"
CALL_ID_HEX_LENGTH = 8

# Workflow id of the travel-planning SOP, the only one allowed to continue after
# the assistant asks a clarifying question.
TRAVEL_PLANNING_WORKFLOW = 1


class DatasetConverter:
    def __init__(self, input_file: str | Path, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
        """Prepare a converter for one seed file.

        Args:
            input_file: JSON array of seed records to convert.
            output_dir: Directory the numbered batch files are written to; it is
                created if missing.
        """
        self.input_file = Path(input_file)
        self.output_dir = ensure_dir(Path(output_dir))

        # Thread lock plus shared progress state.
        self.lock = threading.Lock()
        self.processed_count = 0
        self.converted_data: list[dict[str, Any]] = []
        self.current_batch = 0
        self.total_count = 0
        self.progress_bar: tqdm | None = None

    def load_dataset(self) -> list[dict[str, Any]]:
        """Read the seed dataset.

        Returns:
            The seed records, or an empty list if the file cannot be read or
            parsed.
        """
        try:
            with self.input_file.open(encoding="utf-8") as f:
                seed_items = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("加载数据集失败: %s", exc)
            return []
        logger.info("成功加载 %d 条数据", len(seed_items))
        self.total_count = len(seed_items)
        return seed_items

    def convert_single_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Replay one seed record as a conversation.

        Args:
            item: Seed record, keyed by the Chinese field names used on disk.

        Returns:
            ``{"conversation": [...]}`` on success, or None if the record could
            not be replayed.
        """
        try:
            assistant = TravelAssistantFuncCall(
                user_name=item["用户名字"],
                user_city_id=item["用户所处城市"],
                travel_date_range=f"{item['出发日期']}~{item['出发日期']}",
                start_coordinates=item["起点坐标"]
            )

            # System prompt = the record's user profile followed by the 5-workflow SOP.
            user_info_text = f"""
## 用户信息
- 用户名: {item['用户名字']}
- 当前城市ID: {item['用户所处城市']}
- 出发日期: {item['出发日期']}
- 起点坐标: {item['起点坐标']}

请在处理用户请求时考虑这些信息，比如：
- 问路时如果没有明确起点，使用起点坐标{item['起点坐标']}
- 旅行规划时根据用户的出发日期范围提供建议
- 天气查询时根据旅行攻略中的天数来确定查询天数，使用时间段查询
- 路线查询时起点优先使用起点坐标
"""

            system_prompt = user_info_text + """
你是一个专业的旅行助手，严格按照以下工作流程处理用户请求：

## 工作流1: 旅行规划
**触发条件**: 用户想制定旅行计划、询问某地旅游攻略、景点推荐等
**处理流程**:
1. 首先检查是否有目的地信息
   - 如果没有目的地或目的地不明确，必须反问："请告诉我您想去哪个城市旅行？"
   - 如果用户问"附近有什么好玩的"，可以基于用户当前城市提供建议
   - 不要进行任何工具调用，直接反问
2. 如果有明确的目的地，检查是否有出行日期
   - 如果没有明确日期，可以基于当前日期推荐合适的出行时间
   - 如果完全没有时间信息，反问："请问您计划什么时候出行？"
3. 信息完整后，必须**按照顺序**调用以下两个工具：
   - `search_travel_guide`: 搜索目的地旅行攻略
   - `get_weather_info`: 查询出行日期的天气信息
4. 综合天气和攻略信息，制定详细的旅行计划，规划里面说明每一天的天气信息。不需要详细，只需要说明每天的温度，晴雨天等信息，无需根据天气改变景点顺序，如果search_travel_guide这个工具返回的结果为空，则不再调用天气工具，直接返回无相应旅行路线

## 工作流2: 问路/地图导航
**触发条件**: 用户询问路线、问路、导航、"怎么走"、"如何到达"等
**处理流程**:
1. 检查起点和终点信息，起点信息在起点坐标获得，无需询问，也无需询问城市，城市就是当前城市ID
   - 如果用户说"从X到Y"、"X到Y怎么走"，则X是起点，Y是终点，直接调用工具，只要说了大致的地点，比如说了火车站、医院、学校等信息，则无需追问，直接调用工具查询即可，无需追问具体在哪里
   - 如果用户说"怎么回家"，可以提醒用户提供具体地址
   - 如果完全没有重点的地点信息，才反问："请问您要去哪里？"能不追问尽量不追问，只要有信息则直接调用工具开始查询
2. 信息完整后，调用`query_route`工具获取路线
3. 为用户提供步行、公交、驾车等多种路线选择，如果query_route这个工具返回的结果为空，则直接返回查询不到对应路线

## 工作流3: 酒店查询
**触发条件**: 用户询问酒店推荐、酒店预订、住宿等
**处理流程**:
### 3A: 酒店推荐
1. 检查必要信息：目的地，只有在缺少城市的时候追问，其余时候都直接调用工具
   - 缺少目的地：反问"请问您要在哪个城市找酒店？"
   - 如果用户说"本地酒店"或"附近酒店"，使用当前城市

2. 信息收集完毕后，**必须按顺序执行**：
   - 第一步：调用`recommend_hotels`工具获取酒店推荐
   - 第二步：**立即**为推荐的酒店调用`get_hotel_reviews`工具获取评价
   - **重要**：不要在第一步后就返回结果，必须完成所有工具调用，如果第一步没有推荐结果，则直接返回"暂时没有找到符合您需求的酒店推荐"，无需进行第二步工具调用
3. 整合酒店信息和评价，生成完整的推荐结果（包含酒店详情+用户评价）

### 3B: 酒店评价查询
**触发条件**: 询问某酒店"怎么样"、"评价"、"好不好"等
1. 提取酒店名称
   - 如果没有明确酒店名称，反问："请问您想了解哪家酒店的评价？"，
2. 调用`get_hotel_reviews`获取评价信息

## 工作流4: 闲聊
**触发条件**: 旅行相关的一般性问题、打招呼等
**处理流程**:
1. 判断是否与旅行相关
   - 如果是旅行相关话题，直接回答，不调用工具
   - 如果完全无关（如数学、编程等），礼貌拒绝："抱歉，我是专门的旅行助手，只能回答旅行相关的问题。"

## 重要原则:
1. **信息不足时必须反问，不要猜测**
2. **一次只问一个关键信息，避免一次问太多**
3. **工具调用顺序很重要，旅行规划时必须先调用攻略，再根据返回情况决定是否调用天气**
4. **酒店推荐工作流特别重要**：
   - 先调用`recommend_hotels`获取酒店列表
   - 然后为推荐酒店调用`get_hotel_reviews`
   - 最后整合所有信息统一回复
   - 绝不在获得酒店推荐后立即回复，必须获取评价后才能给最终答案
5. **非旅行相关问题要礼貌拒绝**

## 反问示例:
- "请告诉我您想去哪个城市旅行？"
- "请问您计划什么时候出行？（请提供具体日期）"
- "请问您要去哪里？"
- "请问您要在哪个城市找酒店？"
- "请问您的预算范围是多少？"

严格按照以上流程处理用户请求，确保信息完整后再调用相应工具。"""

            messages = [
                {
                    "role": "system",
                    "content": system_prompt.strip()
                },
                {
                    "role": "user",
                    "content": item["用户问题"]
                }
            ]

            # First assistant turn, with duplicate tool calls collapsed.
            self._process_tools_fixed(assistant, item["用户问题"], messages)

            if item["是否追问"] == "是" and item["追问回答"]:
                # Find the most recent assistant message: no tool_calls on it
                # means the assistant asked a clarifying question.
                last_assistant_message = None
                for message in reversed(messages):
                    if message["role"] == "assistant":
                        last_assistant_message = message
                        break

                # Only travel planning continues after a clarifying question;
                # the other workflows end the conversation there.
                if (item["工作流"] == TRAVEL_PLANNING_WORKFLOW and
                        last_assistant_message and
                        "tool_calls" not in last_assistant_message):

                    messages.append({
                        "role": "user",
                        "content": item["追问回答"]
                    })

                    self._process_tools_fixed(assistant, item["追问回答"], messages)

            return {"conversation": messages}

        # Broad on purpose: one unconvertible record must not abort the run.
        except Exception as exc:
            logger.exception("转换失败: %s", exc)
            return None

    def _process_tools_fixed(
        self,
        assistant: TravelAssistantFuncCall,
        user_input: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Run one assistant turn and append its messages to ``messages``.

        ``assistant.call_function`` is temporarily swapped for a wrapper that
        records every call and short-circuits repeats of an identical
        (name, arguments) pair, which is what keeps the assistant from calling
        the same tool twice within a turn. The recorded calls are then written
        out either as one assistant message carrying every tool call, or — when
        more than one hotel tool fired — as one assistant/tool pair per call, in
        call order.

        Args:
            assistant: Assistant replaying the conversation. Mutated in place.
            user_input: The user message to respond to.
            messages: Conversation being built; appended to in place.
        """
        try:
            original_call_function = assistant.call_function

            # Keyed by tool name + arguments so repeats reuse the first result.
            tool_call_results: dict[str, dict[str, Any]] = {}

            def capture_tool_calls(function_name, arguments):
                call_id = f"{CALL_ID_PREFIX}{uuid.uuid4().hex[:CALL_ID_HEX_LENGTH]}"
                call_key = f"{function_name}_{json.dumps(arguments, sort_keys=True)}"

                if call_key not in tool_call_results:
                    result = original_call_function(function_name, arguments)
                    tool_call_results[call_key] = {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False)
                        },
                        "result": result
                    }

                return tool_call_results[call_key]["result"]

            assistant.call_function = capture_tool_calls

            final_response = assistant.process_user_input(user_input)

            if tool_call_results:
                # Dict insertion order is call order.
                recorded_calls = list(tool_call_results.values())

                hotel_calls = [
                    call for call in recorded_calls
                    if call["function"]["name"] in HOTEL_TOOL_NAMES
                ]

                if len(hotel_calls) > 1:
                    # Hotel workflow: one assistant/tool pair per call, so the
                    # review lookup is seen to follow the recommendation.
                    for call_data in recorded_calls:
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": call_data["id"],
                                "type": call_data["type"],
                                "function": call_data["function"]
                            }]
                        })
                        messages.append({
                            "role": "tool",
                            "content": call_data["result"],
                            "tool_call_id": call_data["id"]
                        })
                else:
                    # Every other workflow: all tool calls in a single turn,
                    # followed by all of their results.
                    tool_calls = []
                    for call_data in recorded_calls:
                        tool_calls.append({
                            "id": call_data["id"],
                            "type": call_data["type"],
                            "function": call_data["function"]
                        })

                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls
                    })

                    for call_data in recorded_calls:
                        messages.append({
                            "role": "tool",
                            "content": call_data["result"],
                            "tool_call_id": call_data["id"]
                        })

                # The assistant's answer, written from the tool results.
                messages.append({
                    "role": "assistant",
                    "content": final_response
                })
            else:
                # No tool calls: a plain reply or a clarifying question.
                messages.append({
                    "role": "assistant",
                    "content": final_response
                })

        # Broad on purpose: the turn is recorded as an error reply so the caller
        # still gets a well-formed conversation.
        except Exception as exc:
            logger.error("处理工具调用失败: %s", exc)
            messages.append({
                "role": "assistant",
                "content": f"抱歉，处理您的请求时出现了错误：{exc}"
            })

    def save_batch(self, conversations: list[dict[str, Any]], batch_num: int) -> None:
        """Write one batch of converted conversations to disk.

        Args:
            conversations: Conversations to write.
            batch_num: Batch number, used to build the filename.
        """
        filepath = self.output_dir / BATCH_FILENAME_TEMPLATE.format(batch_num=batch_num)
        try:
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False, indent=JSON_INDENT)
        except (OSError, TypeError) as exc:
            logger.error("保存批次 %d 失败: %s", batch_num, exc)
            return
        logger.info("保存批次 %d: %d 条数据 -> %s", batch_num, len(conversations), filepath)

    def update_progress(self) -> None:
        """Count one processed record and advance the progress bar."""
        with self.lock:
            self.processed_count += 1
            if self.progress_bar:
                self.progress_bar.update(1)

    def convert_worker(self, items: list[dict[str, Any]], thread_id: int) -> None:
        """Convert one chunk of seed records, flushing full batches as they fill.

        Args:
            items: Seed records assigned to this worker.
            thread_id: Worker index, used only in error messages.
        """
        for item in items:
            try:
                converted_item = self.convert_single_item(item)

                self.update_progress()

                with self.lock:
                    self.converted_data.extend([converted_item] if converted_item else [])

                    if len(self.converted_data) >= BATCH_SAVE_SIZE:
                        self.current_batch += 1
                        self.save_batch(self.converted_data.copy(), self.current_batch)
                        self.converted_data.clear()

                    # Start a fresh file number every BATCH_ROLLOVER_INTERVAL
                    # records, independently of the size-based flush above.
                    if self.processed_count % BATCH_ROLLOVER_INTERVAL == 0:
                        self.current_batch += 1

            # Broad on purpose: a failed record is counted and skipped.
            except Exception as exc:
                logger.error("线程 %d 处理失败: %s", thread_id, exc)
                self.update_progress()
                continue

    def convert_dataset(self, num_threads: int = DEFAULT_NUM_THREADS) -> None:
        """Convert the whole seed file using a pool of worker threads.

        The records are split into ``num_threads`` contiguous chunks, one per
        worker, and any conversations left over at the end are flushed as a
        final batch.

        Args:
            num_threads: Number of worker threads to run.
        """
        logger.info("开始转换数据集...")

        seed_items = self.load_dataset()
        if not seed_items:
            logger.warning("没有数据需要转换")
            return

        self.progress_bar = tqdm(total=self.total_count, desc="转换进度")

        try:
            # Split into one contiguous chunk per thread; the last chunk absorbs
            # the remainder.
            chunk_size = max(1, len(seed_items) // num_threads)
            thread_chunks = []

            for i in range(num_threads):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < num_threads - 1 else len(seed_items)
                if start_idx < len(seed_items):
                    thread_chunks.append(seed_items[start_idx:end_idx])

            logger.info("使用 %d 个线程并行转换...", num_threads)

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for thread_id, chunk in enumerate(thread_chunks):
                    if chunk:
                        futures.append(
                            executor.submit(self.convert_worker, chunk, thread_id)
                        )

                for future in futures:
                    future.result()

            with self.lock:
                if self.converted_data:
                    self.current_batch += 1
                    self.save_batch(self.converted_data, self.current_batch)

        finally:
            if self.progress_bar:
                self.progress_bar.close()

        print(f"转换完成！总共处理了 {self.processed_count} 条数据")
        print(f"输出目录: {self.output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the conversion step.

    Both paths stay positional, so the historic
    ``python -m pipeline.convert_to_conversations <input> [output_dir]``
    invocation keeps working.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="seed dataset to convert (default: %(default)s)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=CLI_DEFAULT_OUTPUT_DIR,
        help="directory for the converted batches (default: %(default)s)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help="number of worker threads (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Convert a seed dataset from the command line.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if not args.input_file.exists():
        logger.error("输入文件不存在: %s", args.input_file)
        raise SystemExit(1)

    converter = DatasetConverter(args.input_file, args.output_dir)
    converter.convert_dataset(num_threads=args.threads)


if __name__ == "__main__":
    main()
