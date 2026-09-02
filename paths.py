"""Canonical filesystem locations for the project.

Every module resolves paths through this file instead of hardcoding absolute
paths. Locations are derived from this file's own position, so the project can be
checked out or moved anywhere without edits.

Override any location with the matching environment variable when running
somewhere the layout differs (for example a remote GPU box):

    TRAVEL_AGENT_DATA_DIR=/root/autodl-tmp/data python -m pipeline.merge_batches

Model checkpoints live outside the repository and are defined at the bottom of
this file under the same override convention.
"""

from __future__ import annotations

import os
from pathlib import Path

# paths.py lives at the repository root, so that is this file's own directory.
PROJECT_ROOT = Path(__file__).resolve().parent


def _dir(env_var: str, default: Path) -> Path:
    """Return the directory named by ``env_var``, falling back to ``default``."""
    override = os.environ.get(env_var)
    return Path(override).expanduser().resolve() if override else default


CONFIG_DIR = _dir("TRAVEL_AGENT_CONFIG_DIR", PROJECT_ROOT / "configs")
DATA_DIR = _dir("TRAVEL_AGENT_DATA_DIR", PROJECT_ROOT / "data")

RAW_DIR = DATA_DIR / "raw"
SEED_DIR = DATA_DIR / "seed"
PROCESSED_DIR = DATA_DIR / "processed"

TRAVEL_GUIDES_DIR = RAW_DIR / "travel_guides"

# Individual files referenced from more than one module.
ALL_TOOLS_FILE = CONFIG_DIR / "all_tools.json"
CITY_CODE_MAPPING_FILE = CONFIG_DIR / "city_code_mapping.json"
CHINA_CITIES_FILE = SEED_DIR / "china_cities_list.json"
GENERATION_PROGRESS_FILE = SEED_DIR / "generation_progress.json"
MILVUS_DB_FILE = _dir("TRAVEL_AGENT_MILVUS_DB", DATA_DIR / "milvus.db")


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if absent, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- Model checkpoints -------------------------------------------------------
# Checkpoints are not part of the repository; they live wherever the machine
# running the training keeps them. These stay plain strings rather than Paths
# because each may equally name a Hugging Face hub id ("Qwen/Qwen3-0.6B").
#
# Defined here rather than per-module because two of them are shared invariants:
# `merge_lora` must merge into the same BASE_MODEL that `train_lora` trained
# against, and `infer` must load exactly the MERGED_MODEL_DIR that `merge_lora`
# wrote. Previously both pairs were hand-copied strings in two files each.


def _model_ref(env_var: str, default: str) -> str:
    """Return the checkpoint path or hub id named by ``env_var``, else ``default``."""
    return os.environ.get(env_var) or default


# Shared by training.train_lora and training.merge_lora.
BASE_MODEL = _model_ref("TRAVEL_AGENT_BASE_MODEL", "/root/autodl-tmp/qwen3-0_6b")
# Shared by training.merge_lora (writes) and training.infer (reads).
MERGED_MODEL_DIR = _model_ref(
    "TRAVEL_AGENT_MERGED_MODEL", "/root/autodl-tmp/qwen3-0_6b_lora_v5_merged"
)
# Shared by training.train_lora (writes the adapter) and training.merge_lora
# (reads it). Defaults inside the checkout so the train -> merge -> infer chain
# works out of the box; point it at the GPU box's path to use an existing run.
ADAPTER_DIR = _model_ref("TRAVEL_AGENT_ADAPTER_DIR", str(PROJECT_ROOT / "qwen_lora_output"))

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "SEED_DIR",
    "PROCESSED_DIR",
    "TRAVEL_GUIDES_DIR",
    "ALL_TOOLS_FILE",
    "CITY_CODE_MAPPING_FILE",
    "CHINA_CITIES_FILE",
    "GENERATION_PROGRESS_FILE",
    "MILVUS_DB_FILE",
    "BASE_MODEL",
    "MERGED_MODEL_DIR",
    "ADAPTER_DIR",
    "ensure_dir",
]
