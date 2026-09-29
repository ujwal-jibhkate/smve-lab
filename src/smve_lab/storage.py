"""Persisting and reloading BGE-M3 multi-representation embeddings.

Each item produces three representations of different, ragged shapes:
  - dense:   fixed-size (1024,) vector           -> stacks into one (N, 1024) array
  - sparse:  variable-length {token_id: weight}  -> flattened CSR-style
             (sparse_indptr / sparse_ids / sparse_weights)
  - colbert: variable-length (num_tokens, 1024)  -> flattened
             (colbert_flat / colbert_offsets)

We save two layouts of the same data on purpose:
  1. One .npy file per array (`save_embeddings`'s first loop), individually
     mmap-able. This is what matters once you scale past a small corpus like
     scifact: colbert_flat has one 1024-dim row *per token*, not per document,
     so it's the array most likely to outgrow RAM. `load_separate(...,
     mmap_colbert=True)` opens it without reading it all into memory.
  2. A single combined `.npz` bundle, which is not mmap-able (it's a zip
     archive) but is convenient to copy or hand off as one file when you
     don't need mmap.

Both are written every time so you always have the portable copy and the
mmap-able copy without re-running encoding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARRAY_KEYS = [
    "dense",
    "sparse_indptr",
    "sparse_ids",
    "sparse_weights",
    "colbert_flat",
    "colbert_offsets",
]


def save_embeddings(
    emb: dict[str, np.ndarray], ids: list[str], save_dir: Path, name: str
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for key, arr in emb.items():
        np.save(save_dir / f"{name}_{key}.npy", arr)
    with open(save_dir / f"{name}_ids.json", "w") as f:
        json.dump(ids, f)

    np.savez(save_dir / f"{name}_all.npz", ids=np.array(ids), **emb)


def load_separate(
    save_dir: Path, name: str, mmap_colbert: bool = True
) -> tuple[dict[str, np.ndarray], list[str]]:
    save_dir = Path(save_dir)
    emb = {}
    for k in ARRAY_KEYS:
        mode = "r" if (mmap_colbert and k == "colbert_flat") else None
        emb[k] = np.load(save_dir / f"{name}_{k}.npy", mmap_mode=mode)
    with open(save_dir / f"{name}_ids.json") as f:
        ids = json.load(f)
    return emb, ids


def load_combined(save_dir: Path, name: str) -> tuple[dict[str, np.ndarray], list[str]]:
    z = np.load(Path(save_dir) / f"{name}_all.npz")  # lazy: arrays read on access
    ids = z["ids"].tolist()
    emb = {k: z[k] for k in z.files if k != "ids"}
    return emb, ids


def get_sparse(emb: dict[str, np.ndarray], i: int) -> dict[int, float]:
    s, e = emb["sparse_indptr"][i], emb["sparse_indptr"][i + 1]
    return dict(zip(emb["sparse_ids"][s:e].tolist(), emb["sparse_weights"][s:e].tolist()))


def get_colbert(emb: dict[str, np.ndarray], i: int) -> np.ndarray:
    s, e = emb["colbert_offsets"][i], emb["colbert_offsets"][i + 1]
    return emb["colbert_flat"][s:e]
