#!/usr/bin/env python3
"""LoRA fine-tuning entry point for the Qwen function-calling model.

This is the project's only trainer. It reuses :mod:`training.dataset` for
sample construction and assistant-only loss masking, wraps the base model with
PEFT LoRA adapters, and saves the adapter to ``--output_dir``.

``BASE_MODEL`` and ``ADAPTER_DIR`` come from :mod:`paths`, which owns them so
that ``training.merge_lora`` merges into the same base this trained against and
reads the adapter from where this wrote it. ``BASE_MODEL`` points at the remote
GPU box, since no checkpoint is kept in the repository; ``ADAPTER_DIR`` defaults
inside the checkout. Override with ``TRAVEL_AGENT_BASE_MODEL`` /
``TRAVEL_AGENT_ADAPTER_DIR``, or pass ``--model_name_or_path`` / ``--output_dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from paths import ADAPTER_DIR, BASE_MODEL
from training.dataset import DataCollatorForCausal, JsonlConversations

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


class ConsoleLossCallback(TrainerCallback):
    """Print (and optionally append to a file) one line per logging step."""

    def __init__(self, log_file: str = "") -> None:
        """Open the optional log file.

        Args:
            log_file: Path to append plain log lines to; disabled when empty.
        """
        super().__init__()
        self.log_file = log_file
        self._fh = None
        if self.log_file:
            path = Path(self.log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """Emit ``step``, ``loss`` and ``learning_rate`` for the current step."""
        if not logs:
            return
        step = state.global_step
        loss = logs.get("loss", logs.get("train_loss"))
        lr = logs.get("learning_rate")
        msg = f"step={step}"
        if loss is not None:
            msg += f" | loss={loss:.6f}"
        if lr is not None:
            msg += f" | lr={lr:.6e}"
        print(msg, flush=True)
        if self._fh is not None:
            self._fh.write(msg + "\n")
            self._fh.flush()

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """Close the log file handle, if one was opened."""
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError as exc:
                logger.warning("Failed to close log file %s: %s", self.log_file, exc)


def parse_args() -> argparse.Namespace:
    """Parse training arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for Qwen (no quantization, reuse dataset utilities)"
    )

    # Data & model
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to JSON/JSONL or JSON array dataset")
    parser.add_argument("--model_name_or_path", type=str, default=BASE_MODEL,
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default=ADAPTER_DIR,
                        help="Where to write the LoRA adapter; merge_lora reads the same "
                             "location by default")

    # Sequence & tokenizer
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--local_files_only", action="store_true",
                        help="Load tokenizer/model only from local cache")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    # Precision & memory
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Dataloader & seed
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--log_file", type=str, default="",
                        help="Optional file to append plain logs")

    # Loss masking: default supervises every assistant turn (incl. intermediate tool calls).
    # Use --only_last_assistant for data already exploded one-sample-per-turn by
    # conversation_splitter.py, otherwise earlier turns get counted repeatedly.
    parser.add_argument("--only_last_assistant", action="store_true",
                        help="Compute loss on the final assistant turn only")

    # Tools injection (applied only to samples that carry no tools of their own)
    parser.add_argument("--tools_file", type=str, default="",
                        help="Path to JSON file with a top-level list of tools "
                             "(OpenAI/Qwen schema)")
    parser.add_argument("--tools_json", type=str, default="",
                        help="Inline JSON string representing a list of tools")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default=DEFAULT_TARGET_MODULES)

    return parser.parse_args()


def build_lora_model(base_model: Any, args: argparse.Namespace) -> Any:
    """Wrap ``base_model`` with LoRA adapters and report trainable parameters.

    Args:
        base_model: The loaded causal LM.
        args: Parsed arguments carrying the LoRA hyperparameters.

    Returns:
        The PEFT-wrapped model.
    """
    target_modules = [
        module_name.strip()
        for module_name in args.target_modules.split(",")
        if module_name.strip()
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model


def load_default_tools(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    """Load the tool list injected into samples that carry no tools.

    Args:
        args: Parsed arguments; ``--tools_file`` wins over ``--tools_json``.

    Returns:
        The tool list, or ``None`` when neither option was given.

    Raises:
        RuntimeError: If the requested tools cannot be read or are not a list.
    """
    if args.tools_file:
        try:
            with Path(args.tools_file).open("r", encoding="utf-8") as f:
                loaded_tools = json.load(f)
            if not isinstance(loaded_tools, list):
                raise ValueError("tools_file must contain a JSON array of tool objects")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load tools_file: {args.tools_file}: {exc}") from exc
        return loaded_tools
    if args.tools_json:
        try:
            loaded_tools = json.loads(args.tools_json)
            if not isinstance(loaded_tools, list):
                raise ValueError("tools_json must be a JSON array of tool objects")
        except ValueError as exc:
            raise RuntimeError(f"Failed to parse tools_json: {exc}") from exc
        return loaded_tools
    return None


def main() -> None:
    """Run the LoRA fine-tuning job described by the command line arguments."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer: %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading base model: %s", args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    # Optional precision cast (no quantization)
    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)

    # Wrap with LoRA adapters
    model = build_lora_model(model, args)

    # Optional: global tools list, used only for samples that have no tools of their own
    default_tools = load_default_tools(args)

    # Dataset
    logger.info("Loading dataset: %s", args.train_file)
    masking_description = (
        "last assistant turn only" if args.only_last_assistant else "all assistant turns"
    )
    logger.info("Loss masking: %s", masking_description)
    train_dataset = JsonlConversations(
        args.train_file,
        tokenizer,
        args.max_seq_length,
        only_last_assistant=args.only_last_assistant,
        default_tools=default_tools,
    )

    data_collator = DataCollatorForCausal(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_torch",
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[ConsoleLossCallback(args.log_file)],
    )

    trainer.train()

    # Save adapter
    trainer.save_state()
    trainer.save_model(args.output_dir)

    logger.info("Training complete. Adapter saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
