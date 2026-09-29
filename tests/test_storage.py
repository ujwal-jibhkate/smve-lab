"""Round-trip tests for the embedding save/load format. No model or GPU needed."""

import numpy as np
import pytest

from smve_lab.storage import get_colbert, get_sparse, load_combined, load_separate, save_embeddings


def _fake_embeddings() -> dict[str, np.ndarray]:
    return {
        "dense": np.random.rand(3, 8).astype(np.float32),
        "sparse_indptr": np.array([0, 2, 3, 5], dtype=np.int64),
        "sparse_ids": np.array([10, 20, 30, 40, 50], dtype=np.int32),
        "sparse_weights": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        "colbert_flat": np.random.rand(6, 4).astype(np.float16),
        "colbert_offsets": np.array([0, 2, 5, 6], dtype=np.int64),
    }


def test_round_trip(tmp_path):
    emb = _fake_embeddings()
    ids = ["a", "b", "c"]
    save_embeddings(emb, ids, tmp_path, "docs")

    emb_sep, ids_sep = load_separate(tmp_path, "docs")
    emb_comb, ids_comb = load_combined(tmp_path, "docs")

    assert ids_sep == ids_comb == ids
    assert np.array_equal(emb_sep["dense"], emb["dense"])
    assert np.array_equal(emb_comb["dense"], emb["dense"])
    assert get_sparse(emb_sep, 1) == pytest.approx({30: 0.3})
    assert get_sparse(emb_comb, 1) == pytest.approx({30: 0.3})
    assert np.array_equal(get_colbert(emb_sep, 0), get_colbert(emb_comb, 0))
    assert get_colbert(emb_sep, 0).shape == (2, 4)
