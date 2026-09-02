#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Split the seed dataset into a training and a test set.

The split is stratified: rows are grouped by (workflow, whether a follow-up
question is needed) and each group is divided 80/20. The random seed is fixed so
the split is reproducible for a given input file.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from paths import SEED_DIR, ensure_dir

DEFAULT_INPUT = SEED_DIR / "travel_assistant_dataset_20250914_112354.json"
DEFAULT_TRAIN_OUTPUT = SEED_DIR / "travel_assistant_train_dataset.json"
DEFAULT_TEST_OUTPUT = SEED_DIR / "travel_assistant_test_dataset.json"

# Fixed so the existing train/test files stay reproducible.
RANDOM_SEED = 42
TRAIN_RATIO = 0.8


def group_key(item: dict[str, Any]) -> str:
    """Return the stratification bucket a row belongs to.

    Args:
        item: One dataset row.

    Returns:
        A human-readable label combining the workflow and whether the row needs a
        follow-up question.
    """
    workflow = item['工作流']
    needs_follow_up = item['是否追问']

    if workflow == 1:  # travel planning
        return '旅行规划（不需要反问）' if needs_follow_up == '否' else '旅行规划（需要反问）'
    if workflow == 2:  # asking for directions
        return '问路（不需要反问）' if needs_follow_up == '否' else '问路（需要反问）'
    if workflow == 3:  # hotel lookup
        return '查询酒店（不需要反问）' if needs_follow_up == '否' else '查询酒店（需要反问）'
    if workflow == 4:  # travel-related chat
        return '旅行相关'
    if workflow == 5:  # refusal
        return '拒答'
    return f'未知工作流{workflow}'


def split_dataset(
    input_path: Path = DEFAULT_INPUT,
    train_path: Path = DEFAULT_TRAIN_OUTPUT,
    test_path: Path = DEFAULT_TEST_OUTPUT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a dataset 80/20 per stratum and write both halves to disk.

    Args:
        input_path: Dataset to split.
        train_path: Where the training split is written.
        test_path: Where the test split is written.

    Returns:
        The ``(train_data, test_data)`` splits.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
    """
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    print("开始划分数据集...")
    print(f"原始数据总量: {len(data)}条")

    # Group rows by stratum.
    grouped_data: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        grouped_data[group_key(item)].append(item)

    print("\n原始数据统计:")
    print("=" * 50)
    for stratum, items in grouped_data.items():
        print(f"{stratum}: {len(items)}条")

    # Seed here, immediately before the first shuffle, so the split is reproducible.
    random.seed(RANDOM_SEED)

    train_data: list[dict[str, Any]] = []
    test_data: list[dict[str, Any]] = []
    train_stats: dict[str, int] = {}
    test_stats: dict[str, int] = {}

    for stratum, items in grouped_data.items():
        random.shuffle(items)

        train_size = int(len(items) * TRAIN_RATIO)
        train_items = items[:train_size]
        test_items = items[train_size:]

        train_data.extend(train_items)
        test_data.extend(test_items)

        train_stats[stratum] = len(train_items)
        test_stats[stratum] = len(test_items)

    # Interleave the strata within each split.
    random.shuffle(train_data)
    random.shuffle(test_data)

    ensure_dir(train_path.parent)
    with train_path.open('w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    ensure_dir(test_path.parent)
    with test_path.open('w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("\n数据集划分完成!")
    print("=" * 50)
    print(f"训练集: {len(train_data)}条 -> {train_path}")
    print(f"测试集: {len(test_data)}条 -> {test_path}")

    print("\n训练集统计:")
    print("-" * 30)
    for stratum in sorted(train_stats.keys()):
        print(f"{stratum}: {train_stats[stratum]}条")

    print("\n测试集统计:")
    print("-" * 30)
    for stratum in sorted(test_stats.keys()):
        print(f"{stratum}: {test_stats[stratum]}条")

    print("\n划分比例验证:")
    print("-" * 30)
    for stratum in sorted(train_stats.keys()):
        total = train_stats[stratum] + test_stats[stratum]
        train_percent = train_stats[stratum] / total * 100
        test_percent = test_stats[stratum] / total * 100
        print(
            f"{stratum}: 训练集{train_percent:.1f}% ({train_stats[stratum]}/{total}), "
            f"测试集{test_percent:.1f}% ({test_stats[stratum]}/{total})"
        )

    overall_total = len(train_data) + len(test_data)
    print(f"\n总计: 训练集{len(train_data)}条, 测试集{len(test_data)}条")
    print(
        f"总体比例: 训练集{len(train_data) / overall_total * 100:.1f}%, "
        f"测试集{len(test_data) / overall_total * 100:.1f}%"
    )

    return train_data, test_data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments, with paths defaulting to the files in ``SEED_DIR``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="dataset JSON file to split"
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT,
        help="where to write the training split",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=DEFAULT_TEST_OUTPUT,
        help="where to write the test split",
    )
    return parser.parse_args()


def main() -> None:
    """Split the dataset named on the command line."""
    args = parse_args()
    split_dataset(args.input, args.train_output, args.test_output)


if __name__ == "__main__":
    main()
