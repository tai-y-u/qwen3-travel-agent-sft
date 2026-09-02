#!/usr/bin/env bash
# Train the LoRA adapter with loss on the final assistant turn only.
set -euo pipefail

# Resolve the repository root from this script's own location, so the runner
# works from any checkout and any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Overridable settings ---------------------------------------------------
# The training data ships with the repo. The base model points at the remote GPU
# box, since no checkpoint is kept in the checkout. Override from the
# environment, e.g.
#   MODEL_PATH=/models/qwen3-0_6b OUTPUT_DIR=/tmp/out ./run_train_last_assistant.sh
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/processed/merged_train_final_multiturn_v2.json}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Agent+SFT/qwen3-0_6b}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/Agent+SFT/qwen3-0_6b_lora_v2_last_assistant}"
# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
# Makes `training.*` and the library packages importable without `pip install -e .`.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Export the adapter location so a later `python -m training.merge_lora` reads
# the adapter this run wrote, without needing --adapter_path spelled out.
export TRAVEL_AGENT_ADAPTER_DIR="${OUTPUT_DIR}"
export TRAVEL_AGENT_BASE_MODEL="${MODEL_PATH}"

python3 -m training.train_lora \
  --train_file "${TRAIN_FILE}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --only_last_assistant \
  --max_seq_length 20000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --lr_scheduler_type cosine \
  --bf16 \
  --gradient_checkpointing \
  --local_files_only
