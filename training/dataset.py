#!/usr/bin/env python3
"""SFT dataset plumbing for Qwen chat fine-tuning, plus a loss-mask visualizer.

The module has two faces:

* **Library** (imported by ``training.train_lora``): :class:`JsonlConversations`
  turns a JSON array / JSONL file of chat conversations into tokenized samples
  whose ``labels`` are masked so that only assistant tokens contribute to the
  loss, and :class:`DataCollatorForCausal` pads a list of those samples into a
  batch.
* **CLI** (``python -m training.dataset``): :func:`main` prints one or more
  samples together with their loss mask, so the masking can be eyeballed before
  a training run. Everything below the "command line interface" banner exists
  only for that CLI.

``DEFAULT_TOKENIZER_PATH`` points at a model checkpoint, which lives on the
remote GPU box rather than in this repository. Set it (or pass
``--model_name_or_path``) to match the machine you are running on.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from paths import ALL_TOOLS_FILE, PROCESSED_DIR, ensure_dir

logger = logging.getLogger(__name__)

# Label value that tells the loss function to skip a position.
LABEL_IGNORE_INDEX = -100

# Tokenizer checkpoint used when no --model_name_or_path is given. Model
# checkpoints are not part of the repository; adjust for the target machine.
DEFAULT_TOKENIZER_PATH = "qwen3-4b-instruct"

DEFAULT_DATA_FILE = PROCESSED_DIR / "merged_train_final.json"
DEFAULT_MAX_SEQ_LENGTH = 20000
DEFAULT_MAX_PRINT_TOKENS = 20000

# Decoded spans longer than this are elided when printed by the CLI.
MAX_SEGMENT_PREVIEW_CHARS = 512
SEGMENT_PREVIEW_ELLIPSIS = "..."

ANSI_CODES = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}

__all__ = ["JsonlConversations", "DataCollatorForCausal", "main"]


# =========================================================================
# Chat template handling and dataset construction (self-contained: the only
# third-party dependencies are torch and transformers)
# =========================================================================

def _supports_tools_kw(tokenizer: AutoTokenizer) -> bool:
    """Report whether ``tokenizer.apply_chat_template`` accepts a ``tools`` kwarg.

    Some chat templates can render tool schemas into the prompt, which is what
    produces function-calling-shaped training text.

    Args:
        tokenizer: Tokenizer whose chat template is inspected.

    Returns:
        ``True`` if the template call accepts ``tools``, ``False`` otherwise
        (including when the signature cannot be inspected).
    """
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError) as exc:
        logger.debug("Cannot inspect apply_chat_template signature: %s", exc)
        return False
    return any(parameter.name == "tools" for parameter in sig.parameters.values())


class JsonlConversations:
    """Conversation dataset tokenized with the Qwen chat template.

    Input: a JSON array or a JSONL file where every item holds a message list
    under ``messages`` or ``conversation``, and optionally a ``tools`` array of
    tool schemas that is passed to the chat template when the template supports
    it.

    ``__getitem__`` returns a dict of:

    * ``input_ids``: the templated token sequence.
    * ``attention_mask``: all ones — this class never pads, so that each sample
      keeps its own length and stays easy to inspect by hand.
    * ``labels``: a copy of ``input_ids`` with every non-assistant position set
      to :data:`LABEL_IGNORE_INDEX`, so only assistant spans produce gradient.

    Assistant spans are located by *prefix differencing*: templating
    ``messages[:i + 1]`` for increasing ``i`` and treating the growth in token
    count as the span of message ``i``.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        only_last_assistant: bool = False,
        default_tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Load and normalize the dataset file.

        Args:
            path: JSON array file, or JSONL with one JSON object per line.
            tokenizer: Tokenizer providing the chat template.
            max_seq_length: Sequences longer than this are truncated from the
                front, keeping the tail.
            only_last_assistant: Supervise only the final assistant message
                instead of every assistant message.
            default_tools: Tool schemas injected into samples that carry no
                ``tools`` of their own. Ignored when empty.

        Raises:
            ValueError: If the file yields no usable samples.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples: list[dict[str, Any]] = []
        self._default_tools: list[dict[str, Any]] | None = (
            default_tools if isinstance(default_tools, list) and len(default_tools) > 0 else None
        )
        self.only_last_assistant = only_last_assistant

        path = Path(path)
        content = path.read_text(encoding="utf-8")
        stripped = content.lstrip()
        parsed_any = False

        # Try a JSON array first.
        if stripped.startswith("["):
            try:
                records = json.loads(content)
            except json.JSONDecodeError as exc:
                logger.debug("JSON array parse failed for %s: %s", path, exc)
            else:
                if isinstance(records, list):
                    for raw_record in records:
                        normalized = self._normalize_obj(raw_record)
                        if normalized is not None:
                            self.samples.append(normalized)
                    parsed_any = len(self.samples) > 0

        # Fall back to JSONL.
        if not parsed_any:
            logger.info("Failed to parse JSON array, trying JSONL: %s", path)
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                normalized = self._normalize_obj(raw_record)
                if normalized is not None:
                    self.samples.append(normalized)

        if len(self.samples) == 0:
            raise ValueError(
                "No valid samples found. Expect JSON array or JSONL with objects "
                "containing a 'messages' or 'conversation' list."
            )

        self._use_tools_kw = _supports_tools_kw(self.tokenizer)

    def _normalize_obj(self, record: Any) -> dict[str, Any] | None:
        """Reduce a raw record to ``{"messages": [...], "tools": [...]}``.

        Args:
            record: One decoded record from the dataset file.

        Returns:
            The normalized record, or ``None`` if it carries no message list.
        """
        if not isinstance(record, dict):
            return None
        messages = None
        if isinstance(record.get("messages"), list):
            messages = record["messages"]
        elif isinstance(record.get("conversation"), list):
            messages = record["conversation"]
        if messages is None:
            return None
        normalized: dict[str, Any] = {"messages": messages}
        if isinstance(record.get("tools"), list):
            normalized["tools"] = record["tools"]
        elif self._default_tools is not None:
            # Inject the shared tool list so tool supervision is uniform across
            # samples that shipped without their own tools.
            normalized["tools"] = self._default_tools
        return normalized

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tokenize: bool,
        return_tensors: str | None = None,
    ) -> Any:
        """Render ``messages`` through the tokenizer's chat template.

        Tools are passed only when the template accepts them, and the generation
        prompt is always disabled so the supervision signal stays exact.

        Args:
            messages: Chat messages to render.
            tools: Tool schemas, or ``None``.
            tokenize: Return token ids instead of text.
            return_tensors: Optional tensor format forwarded to the tokenizer.

        Returns:
            A list of token ids when ``tokenize`` is true, otherwise the
            templated text.
        """
        kwargs: dict[str, Any] = {"tokenize": tokenize, "add_generation_prompt": False}
        if return_tensors is not None:
            kwargs["return_tensors"] = return_tensors
        if tools and self._use_tools_kw:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _apply_template_ids(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> list[int]:
        """Template ``messages`` to token ids, degrading gracefully.

        A few older tokenizer versions reject the extra keyword arguments; those
        raise ``TypeError`` and are retried with the minimal call.

        Args:
            messages: Chat messages to render.
            tools: Tool schemas, or ``None``.

        Returns:
            The templated token ids.
        """
        try:
            return self._apply_template(messages, tools, tokenize=True, return_tensors=None)
        except TypeError as exc:
            logger.debug("Falling back to minimal apply_chat_template call: %s", exc)
            return self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Build one training sample.

        Steps:
            1. Template the whole conversation once to get ``full_ids``.
            2. Prefix-difference: template ``messages[:i + 1]`` for each ``i``;
               when message ``i`` is an assistant turn, ``(prev_len, cur_len)``
               is its token span inside ``full_ids``.
            3. Truncate from the head when over length, keeping the tail, and
               truncate the mask the same way.
            4. Copy ``input_ids`` into ``labels`` and blank every position
               outside the supervised spans.

        Args:
            idx: Index of the sample.

        Returns:
            Dict with ``input_ids``, ``attention_mask`` and ``labels`` tensors.

        Raises:
            ValueError: If head truncation cuts into every assistant span, so no
                complete supervision target survives.
        """
        sample = self.samples[idx]
        messages: list[dict[str, Any]] = sample["messages"]
        tools: list[dict[str, Any]] | None = sample.get("tools")

        # One full templating pass gives the complete token sequence.
        full_ids = self._apply_template_ids(messages, tools)
        total_len = len(full_ids)

        # Locate assistant spans by prefix differencing.
        assistant_spans: list[tuple[int, int]] = []
        prev_len = 0
        for i, message in enumerate(messages):
            prefix_ids = self._apply_template_ids(messages[: i + 1], tools)
            cur_len = len(prefix_ids)
            if message.get("role") == "assistant":
                start, end = prev_len, cur_len
                if end > start:
                    assistant_spans.append((start, end))
            prev_len = cur_len

        # only_last_assistant decides which spans are supervised.
        assistant_mask = torch.zeros(total_len, dtype=torch.bool)
        if self.only_last_assistant:
            if len(assistant_spans) > 0:
                start, end = assistant_spans[-1]
                assistant_mask[start:end] = True
        else:
            for start, end in assistant_spans:
                assistant_mask[start:end] = True

        # Truncate to the maximum length. Cutting from the head keeps the latest
        # turns of a long conversation, which is what we want to supervise.
        if total_len > self.max_seq_length:
            overflow = total_len - self.max_seq_length

            # A cut that lands inside a supervised span would leave a partial
            # target: the model would be trained to emit the tail of a reply
            # whose beginning it never saw. Drop supervision for any such span
            # rather than teaching it to start mid-sentence.
            for start, end in assistant_spans:
                if start < overflow < end:
                    assistant_mask[overflow:end] = False

            full_ids = full_ids[overflow:]
            assistant_mask = assistant_mask[overflow:]

            if not bool(assistant_mask.any()):
                # Every supervised span was cut into, so there is nothing left to
                # learn from. Silently returning all-ignored labels would make the
                # loss NaN, so fail with something actionable instead.
                longest_span = max((end - start for start, end in assistant_spans), default=0)
                raise ValueError(
                    f"Sample {idx} has no fully-contained assistant span at "
                    f"max_seq_length={self.max_seq_length}: the sequence is "
                    f"{total_len} tokens and its longest assistant turn is "
                    f"{longest_span} tokens. Raise --max_seq_length to at least "
                    f"{longest_span + 1} to keep this sample."
                )

        # No padding here, so every sample keeps its own length.
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)  # no padding, so all ones
        labels = input_ids.clone()
        labels[~assistant_mask] = LABEL_IGNORE_INDEX  # ignored by the loss

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class DataCollatorForCausal:
    """Pad variable-length samples into a batch for causal LM training.

    ``input_ids`` are padded with the tokenizer's pad id, ``attention_mask``
    with ``0`` and ``labels`` with :data:`LABEL_IGNORE_INDEX` so padding keeps
    its "ignore me" meaning. Passing ``pad_to_multiple_of`` rounds the batch
    length up, which helps throughput on some hardware.

    The CLI visualizer in this module does not use the collator; it exists for
    training and for batched export.
    """

    def __init__(
        self, tokenizer: AutoTokenizer, pad_to_multiple_of: int | None = None
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate ``features`` into a padded batch.

        Args:
            features: Samples as returned by :meth:`JsonlConversations.__getitem__`.

        Returns:
            Dict of batched ``input_ids``, ``attention_mask`` and ``labels``.
        """
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=LABEL_IGNORE_INDEX,
        )

        if self.pad_to_multiple_of is not None:
            def _pad_to_multiple(tensor: torch.Tensor, value: int) -> torch.Tensor:
                length = tensor.size(1)
                multiple = self.pad_to_multiple_of
                pad_len = (multiple - length % multiple) % multiple
                if pad_len == 0:
                    return tensor
                pad_shape = (tensor.size(0), pad_len)
                pad_tensor = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
                return torch.cat([tensor, pad_tensor], dim=1)

            batch_input_ids = _pad_to_multiple(batch_input_ids, self.tokenizer.pad_token_id)
            batch_attention_mask = _pad_to_multiple(batch_attention_mask, 0)
            batch_labels = _pad_to_multiple(batch_labels, LABEL_IGNORE_INDEX)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }


# =========================================================================
# Command line interface: mask visualization and tensor export
# =========================================================================

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the mask visualizer.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Qwen SFT dataset: visualize assistant-only mask and converted "
            "tensors (standalone)"
        )
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(DEFAULT_DATA_FILE),
        help="Path to JSON/JSONL file",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
        help="Tokenizer source",
    )
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                        help="Maximum sequence length; longer samples are cut from the head")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to print")
    parser.add_argument("--start", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--show_tokens", action="store_true",
                        help="Show token strings rather than ids")
    parser.add_argument("--max_print_tokens", type=int, default=DEFAULT_MAX_PRINT_TOKENS,
                        help="Max tokens to print per sample (head)")
    parser.add_argument("--color", action="store_true",
                        help="Use ANSI colors to highlight loss tokens")
    parser.add_argument("--export_dir", type=str, default="",
                        help="Optional dir to export converted tensors per sample as JSON")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Load tokenizer only from local files (offline)")
    parser.add_argument("--show_ignored_segments", action="store_true",
                        help="Show decoded spans that do NOT contribute to loss")
    parser.add_argument("--show_full_decoded", action="store_true",
                        help="Show full decoded text per sample")
    parser.add_argument("--only_last_assistant", action="store_true",
                        help="Only mask the last assistant segment for loss")
    # Global tools injection for samples missing tools
    parser.add_argument(
        "--tools_file",
        type=str,
        default=str(ALL_TOOLS_FILE),
        help="Path to JSON file with a list of tools to inject when sample.tools is missing",
    )
    parser.add_argument("--tools_json", type=str, default="",
                        help="Inline JSON string (list) of tools to inject when sample.tools "
                             "is missing")
    return parser.parse_args()


def _ansi(color: str) -> str:
    """Return the ANSI escape for ``color``, or an empty string if unknown."""
    return ANSI_CODES.get(color, "")


def _highlight(tokens: list[str], loss_mask: torch.Tensor, use_color: bool) -> str:
    """Render tokens with the supervised ones marked.

    Args:
        tokens: Token strings.
        loss_mask: Boolean mask, same length as ``tokens``.
        use_color: Highlight in green when true, wrap in square brackets when false.

    Returns:
        A single space-joined line.

    Raises:
        ValueError: If ``tokens`` and ``loss_mask`` have different lengths.
    """
    if len(tokens) != loss_mask.numel():
        raise ValueError(
            f"tokens and loss_mask must have the same length, got "
            f"{len(tokens)} and {loss_mask.numel()}"
        )
    parts: list[str] = []
    for i, token in enumerate(tokens):
        if loss_mask[i].item():
            if use_color:
                parts.append(f"{_ansi('green')}{token}{_ansi('reset')}")
            else:
                parts.append(f"[{token}]")
        else:
            parts.append(token)
    return " ".join(parts)


def _contiguous_true_spans(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Split a boolean mask into maximal ``[start, end)`` runs of ``True``.

    Args:
        mask: Boolean mask.

    Returns:
        The runs, in order.
    """
    spans: list[tuple[int, int]] = []
    start = None
    for i, is_true in enumerate(mask.tolist()):
        if is_true and start is None:
            start = i
        elif not is_true and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, mask.numel()))
    return spans


def _export_sample_json(
    path: Path,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Write one sample's tensors to ``path`` as JSON for offline debugging."""
    data = {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _preview(text: str) -> str:
    """Shorten ``text`` to :data:`MAX_SEGMENT_PREVIEW_CHARS` characters."""
    if len(text) <= MAX_SEGMENT_PREVIEW_CHARS:
        return text
    keep = MAX_SEGMENT_PREVIEW_CHARS - len(SEGMENT_PREVIEW_ELLIPSIS)
    return text[:keep] + SEGMENT_PREVIEW_ELLIPSIS


def _print_decoded_spans(
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    spans: list[tuple[int, int]],
    header: str,
) -> None:
    """Print each span of ``input_ids`` decoded with special tokens kept."""
    print(header)
    for start, end in spans:
        text = tokenizer.decode(input_ids[start:end], skip_special_tokens=False)
        print(f"  [{start}:{end}] -> {_preview(text)!r}")


def _load_default_tools(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    """Load the tool list injected into samples that carry no tools.

    Args:
        args: Parsed CLI arguments; ``--tools_file`` wins over ``--tools_json``.

    Returns:
        The tool list, or ``None`` when neither option was given.

    Raises:
        RuntimeError: If the requested tools cannot be read or are not a list.
    """
    if args.tools_file:
        try:
            with Path(args.tools_file).open("r", encoding="utf-8") as f:
                tools = json.load(f)
            if not isinstance(tools, list):
                raise ValueError("tools_file must contain a JSON array of tool objects")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load tools_file: {args.tools_file}: {exc}") from exc
        return tools
    if args.tools_json:
        try:
            tools = json.loads(args.tools_json)
            if not isinstance(tools, list):
                raise ValueError("tools_json must be a JSON array of tool objects")
        except ValueError as exc:
            raise RuntimeError(f"Failed to parse tools_json: {exc}") from exc
        return tools
    return None


def main() -> None:
    """Print samples with their loss mask, and optionally export their tensors.

    Loads the tokenizer (optionally offline), builds the dataset, then for each
    selected sample prints the sequence length and supervised token count, the
    token ids (or token strings) alongside the loss mask, and the decoded text
    of every supervised span — special tokens included, so template boundaries
    are visible.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    default_tools = _load_default_tools(args)

    print(f"Loading dataset: {args.data_file}")
    dataset = JsonlConversations(
        args.data_file,
        tokenizer,
        args.max_seq_length,
        args.only_last_assistant,
        default_tools=default_tools,
    )

    total_samples = len(dataset)
    start = max(0, args.start)
    end = min(total_samples, start + args.num_samples)

    export_dir = ensure_dir(Path(args.export_dir)) if args.export_dir else None

    for idx in range(start, end):
        sample = dataset[idx]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample["labels"]

        # Supervised positions are exactly those whose label survived masking.
        loss_mask = labels.ne(LABEL_IGNORE_INDEX)

        # Print only the head of very long sequences.
        head_len = min(input_ids.numel(), args.max_print_tokens)
        head_input_ids = input_ids[:head_len]
        head_loss_mask = loss_mask[:head_len]

        print("\n" + "=" * 80)
        print(f"Sample #{idx} / {total_samples}")
        print(
            f"seq_len={input_ids.numel()}  "
            f"loss_tokens={int(loss_mask.sum().item())}  "
            f"masked_out={int((~loss_mask).sum().item())}"
        )

        # Show token strings or raw ids, marking what contributes to the loss.
        if args.show_tokens:
            tokens = tokenizer.convert_ids_to_tokens(head_input_ids.tolist())
            line = _highlight(tokens, head_loss_mask, args.color)
            print("Tokens (green or [brackets] = contribute to loss):")
            print(line)
        else:
            ids_line = " ".join(str(token_id) for token_id in head_input_ids.tolist())
            mask_line = " ".join("1" if is_loss else "." for is_loss in head_loss_mask.tolist())
            print("input_ids:")
            print(ids_line)
            print("loss_mask (1=loss, .=ignore):")
            print(mask_line)

        # Decoded supervised spans make the template/mask alignment checkable.
        spans = _contiguous_true_spans(loss_mask)
        if spans:
            _print_decoded_spans(
                tokenizer, input_ids, spans, "Loss segments (decoded with special tokens):"
            )
        else:
            print("No loss segments found (unexpected for assistant-only masking)")

        # Optional: the complement, which is usually easier to read.
        if args.show_ignored_segments:
            ignored_spans = _contiguous_true_spans(~loss_mask)
            if ignored_spans:
                _print_decoded_spans(
                    tokenizer,
                    input_ids,
                    ignored_spans,
                    "Ignored segments (decoded with special tokens):",
                )
            else:
                print("No ignored segments (all tokens contribute to loss)")

        # Optional: the whole sample as text.
        if args.show_full_decoded:
            print("Full decoded (skip_special_tokens=True):")
            print(tokenizer.decode(input_ids.tolist(), skip_special_tokens=True))
            print("Full decoded (skip_special_tokens=False):")
            print(tokenizer.decode(input_ids.tolist(), skip_special_tokens=False))

        if export_dir is not None:
            out_path = export_dir / f"sample_{idx:06d}.json"
            _export_sample_json(out_path, input_ids, attention_mask, labels)
            print(f"Exported tensors to: {out_path}")


if __name__ == "__main__":
    main()


# Example invocation (run from the repository root):
# python -u -m training.dataset \
#   --data_file data/processed/merged_train_final.json \
#   --model_name_or_path /path/to/qwen3-0_6b \
#   --tools_file configs/all_tools.json \
#   --only_last_assistant \
#   --num_samples 8 --start 0 --show_full_decoded
