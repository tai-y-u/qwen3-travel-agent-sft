#!/usr/bin/env python3
"""Merge a trained LoRA adapter into its base model and save the result.

The merged checkpoint is what :mod:`training.infer` loads, since it needs no
PEFT wrapper at inference time.

The checkpoint defaults come from :mod:`paths`, which keeps them in one place
because all three are shared invariants: ``BASE_MODEL`` must be the same base
``training.train_lora`` trained against, ``ADAPTER_DIR`` is where that run wrote
its adapter, and ``MERGED_MODEL_DIR`` is where ``training.infer`` looks for the
result. Running train -> merge -> infer with no flags therefore works. Override
per machine with ``TRAVEL_AGENT_BASE_MODEL`` / ``TRAVEL_AGENT_ADAPTER_DIR`` /
``TRAVEL_AGENT_MERGED_MODEL``, or pass the matching CLI flags.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from paths import ADAPTER_DIR, BASE_MODEL, MERGED_MODEL_DIR

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse merge arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and save merged model"
    )
    parser.add_argument("--base_model", type=str, default=BASE_MODEL,
                        help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_DIR)
    parser.add_argument("--output_dir", type=str, default=MERGED_MODEL_DIR,
                        help="Where to save merged model")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Load only from local cache")
    parser.add_argument("--bf16", action="store_true",
                        help="Load model in bfloat16 for lower memory during merge")
    parser.add_argument("--fp16", action="store_true",
                        help="Load model in float16 for lower memory during merge")
    parser.add_argument("--device_map_auto", action="store_true",
                        help="Use device_map='auto' to shard across devices during merge")
    return parser.parse_args()


def main() -> None:
    """Load base model plus adapter, merge the weights, and save the result."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    logger.info("Loading tokenizer from base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        device_map="auto" if args.device_map_auto else None,
        torch_dtype=torch_dtype,
    )

    logger.info("Loading LoRA adapter: %s", args.adapter_path)
    lora_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        is_trainable=False,
    )

    logger.info("Merging LoRA weights into base model (this may take a while)...")
    merged_model = lora_model.merge_and_unload()

    logger.info("Saving merged model to: %s", args.output_dir)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Done. Merged model saved at: %s", args.output_dir)


if __name__ == "__main__":
    main()
