#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Merge the per-batch JSON files written by ``pipeline.convert_to_conversations``.

The converter emits one ``converted_batch_XXX.json`` file per batch of
conversations. This script concatenates every batch in a directory into a single
JSON array, which is the form the training code expects.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from paths import PROCESSED_DIR, ensure_dir

logger = logging.getLogger(__name__)

# Directories holding the converter's per-batch output, and the merged files
# built from them. All four live under ``data/processed``.
DEFAULT_TEST_INPUT_DIR = PROCESSED_DIR / "converted_test_final"
DEFAULT_TRAIN_INPUT_DIR = PROCESSED_DIR / "converted_train_final"
DEFAULT_TEST_OUTPUT_FILE = PROCESSED_DIR / "merged_test_final.json"
DEFAULT_TRAIN_OUTPUT_FILE = PROCESSED_DIR / "merged_train_final.json"

BATCH_GLOB = "*.json"
JSON_INDENT = 2


def merge_json_files(input_dir: str | Path, output_file: str | Path) -> list[Any]:
    """Concatenate every JSON array in ``input_dir`` into ``output_file``.

    Files are processed in sorted filename order, which keeps the merged output
    stable across runs. A file that does not contain a top-level array, or that
    cannot be read or parsed, is reported and skipped rather than aborting the
    merge.

    Args:
        input_dir: Directory containing the per-batch JSON files.
        output_file: Path of the merged JSON array to write.

    Returns:
        The merged list of items, in the order they were written.
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    merged_data: list[Any] = []

    json_files = sorted(input_path.glob(BATCH_GLOB))
    logger.info("Found %d JSON files in %s", len(json_files), input_path)

    for json_file in json_files:
        logger.info("Processing: %s", json_file)
        try:
            with json_file.open(encoding="utf-8") as f:
                batch_data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Error processing %s: %s", json_file, exc)
            continue

        if isinstance(batch_data, list):
            merged_data.extend(batch_data)
            logger.info("  - Added %d items", len(batch_data))
        else:
            logger.warning("  - %s does not contain an array, skipping", json_file)

    logger.info("Writing merged data to %s (total items: %d)", output_path, len(merged_data))
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=JSON_INDENT)

    logger.info("Successfully merged %d files into %s", len(json_files), output_path)
    return merged_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the merge step.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_INPUT_DIR,
        help="directory of converted test batches (default: %(default)s)",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=DEFAULT_TEST_OUTPUT_FILE,
        help="merged test dataset to write (default: %(default)s)",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_INPUT_DIR,
        help="directory of converted train batches (default: %(default)s)",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT_FILE,
        help="merged train dataset to write (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Merge the test batches and then the train batches.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    logger.info("=== Merging Test Files ===")
    merge_json_files(args.test_dir, args.test_output)

    logger.info("=== Merging Train Files ===")
    merge_json_files(args.train_dir, args.train_output)

    print("\n=== Summary ===")
    print(f"Test dataset merged into: {args.test_output}")
    print(f"Train dataset merged into: {args.train_output}")


if __name__ == "__main__":
    main()
