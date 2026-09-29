"""Loading BGE-M3 and running batched multi-representation encoding."""

from __future__ import annotations

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

from smve_lab.config import EncodeConfig, resolve_device


def load_model(cfg: EncodeConfig) -> BGEM3FlagModel:
    device = resolve_device(cfg.device)
    return BGEM3FlagModel(cfg.model_name, use_fp16=cfg.use_fp16, devices=device)


def encode_texts(
    model: BGEM3FlagModel, texts: list[str], cfg: EncodeConfig
) -> dict[str, np.ndarray]:
    """Encode texts into dense, sparse (lexical), and ColBERT vectors.

    Dense output is fixed-size per text and stacks cleanly. Sparse and ColBERT
    outputs are ragged (a variable-length dict / token matrix per text), so
    they're flattened here into CSR-style arrays (indptr/ids/weights for
    sparse, flat/offsets for ColBERT) that numpy can store and mmap as single
    contiguous arrays. See storage.py for how they're saved and re-sliced.
    """
    dense_list: list[np.ndarray] = []
    colbert_list: list[np.ndarray] = []
    sp_ids: list[int] = []
    sp_w: list[float] = []
    sp_lens: list[int] = []

    for start in tqdm(range(0, len(texts), cfg.batch_size), desc="encoding"):
        batch = texts[start : start + cfg.batch_size]
        out = model.encode(
            batch,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        dense_list.append(out["dense_vecs"])

        for d in out["lexical_weights"]:  # {token_id(str): weight}
            sp_ids.extend(int(k) for k in d.keys())
            sp_w.extend(float(v) for v in d.values())
            sp_lens.append(len(d))

        colbert_list.extend(c.astype(np.float16) for c in out["colbert_vecs"])

    colbert_lens = np.array([len(c) for c in colbert_list])
    return {
        "dense": np.vstack(dense_list).astype(np.float32),
        "sparse_indptr": np.concatenate([[0], np.cumsum(sp_lens)]).astype(np.int64),
        "sparse_ids": np.array(sp_ids, dtype=np.int32),
        "sparse_weights": np.array(sp_w, dtype=np.float32),
        "colbert_flat": np.concatenate(colbert_list),
        "colbert_offsets": np.concatenate([[0], np.cumsum(colbert_lens)]).astype(np.int64),
    }
