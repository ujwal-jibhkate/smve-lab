"""Project paths and encoding settings.

Centralizing these means the notebook, the CLI script, and the tests all agree
on where things live and what the default model settings are, instead of each
one hardcoding its own copy (like the notebook's old Colab Drive path).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RESULTS_DIR = ROOT_DIR / "results"


def resolve_device(device: str = "auto") -> str:
    """Pick the best available torch device, unless one is explicitly requested."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(frozen=True)
class EncodeConfig:
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"  # "auto" | "mps" | "cuda" | "cpu"
    use_fp16: bool = True
    batch_size: int = 32
    max_length: int = 512
