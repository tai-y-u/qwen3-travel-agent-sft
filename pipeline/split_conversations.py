#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Split multi-turn conversations into one training sample per assistant turn.

Every conversation that contains at least one tool message is expanded into a
family of prefixes: the messages from the system prompt up to and including the
first assistant reply, then up to the second assistant reply, and so on through
the last one. The prefix ending at the final assistant reply is emitted three
times so that the fully-resolved answer carries more weight during training.

Conversations without any tool message are passed through untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

from paths import PROCESSED_DIR, ensure_dir

logger = logging.getLogger(__name__)

DEFAULT_INPUT_FILE = PROCESSED_DIR / "merged_train_final.json"
DEFAULT_OUTPUT_FILE = PROCESSED_DIR / "merged_train_final_multiturn_v2.json"

CONVERSATION_KEY = "conversation"
# How many copies of the prefix ending at the final assistant reply to emit.
FINAL_TURN_COPIES = 3
JSON_INDENT = 2


def split_conversations(input_file: str | Path, output_file: str | Path) -> list[dict[str, Any]]:
    """Expand each conversation in ``input_file`` and write the result.

    Args:
        input_file: JSON array of ``{"conversation": [...]}`` records.
        output_file: Path of the expanded JSON array to write.

    Returns:
        The expanded list of conversation records.

    Raises:
        OSError: If ``input_file`` cannot be read or ``output_file`` written.
        json.JSONDecodeError: If ``input_file`` is not valid JSON.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    logger.info("开始处理文件: %s", input_path)
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)
    logger.info("原始数据包含 %d 个对话", len(data))

    expanded_conversations: list[dict[str, Any]] = []

    for conversation_index, conversation_data in enumerate(data):
        if CONVERSATION_KEY not in conversation_data:
            logger.warning("警告: 第 %d 个对话没有conversation字段，跳过", conversation_index)
            continue

        conversation = conversation_data[CONVERSATION_KEY]

        # Conversations that never call a tool are single-turn already, so they
        # are kept verbatim instead of being split.
        has_tool_messages = any(message.get("role") == "tool" for message in conversation)
        if not has_tool_messages:
            logger.info("第 %d 个对话没有tool消息，保持原样不拆分", conversation_index)
            expanded_conversations.append(conversation_data)
            continue

        assistant_indices = [
            i for i, message in enumerate(conversation) if message.get("role") == "assistant"
        ]
        if not assistant_indices:
            logger.warning("警告: 第 %d 个对话没有assistant回复，跳过", conversation_index)
            continue

        logger.info("处理第 %d 个对话，找到 %d 个assistant回复", conversation_index, len(assistant_indices))

        for turn_index, end_pos in enumerate(assistant_indices):
            # Slice through the current assistant reply, inclusive.
            split_conversation_data = {CONVERSATION_KEY: conversation[: end_pos + 1]}
            expanded_conversations.append(split_conversation_data)

            if turn_index == len(assistant_indices) - 1:
                # Oversample the final answer: FINAL_TURN_COPIES in total.
                for _ in range(FINAL_TURN_COPIES - 1):
                    expanded_conversations.append(copy.deepcopy(split_conversation_data))
                logger.info("  - 最后一轮assistant回复额外复制了%d份", FINAL_TURN_COPIES - 1)

    logger.info("拆分完成，总共生成 %d 个对话片段", len(expanded_conversations))

    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(expanded_conversations, f, ensure_ascii=False, indent=JSON_INDENT)
    logger.info("结果已保存到: %s", output_path)

    print("\n统计信息:")
    print(f"原始对话数量: {len(data)}")
    print(f"拆分后对话数量: {len(expanded_conversations)}")
    if data:
        print(f"扩展倍数: {len(expanded_conversations) / len(data):.2f}")

    return expanded_conversations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the split step.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="merged conversations to expand (default: %(default)s)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="expanded conversations to write (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the split step from the command line.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    try:
        split_conversations(args.input_file, args.output_file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("❌ 处理过程中出现错误: %s", exc)
        raise SystemExit(1) from exc

    print("\n✅ 拆分任务完成！")


if __name__ == "__main__":
    main()
