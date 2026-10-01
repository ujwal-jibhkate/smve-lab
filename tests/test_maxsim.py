"""The vectorized MaxSim must match the obvious per-pair loop exactly."""

import numpy as np

from smve_lab.maxsim import maxsim_scores, subset_ragged, top_k


def _ragged(rng, lengths, dim):
    flat = rng.standard_normal((sum(lengths), dim)).astype(np.float16)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return flat, offsets


def _naive(q_flat, q_off, d_flat, d_off):
    out = np.zeros((len(q_off) - 1, len(d_off) - 1), dtype=np.float32)
    for i in range(len(q_off) - 1):
        q = q_flat[q_off[i] : q_off[i + 1]].astype(np.float32)
        for j in range(len(d_off) - 1):
            d = d_flat[d_off[j] : d_off[j + 1]].astype(np.float32)
            out[i, j] = (q @ d.T).max(axis=1).sum()
    return out


def test_matches_naive_loop_across_chunks():
    rng = np.random.default_rng(0)
    q_flat, q_off = _ragged(rng, [3, 1, 7, 4], dim=16)
    d_flat, d_off = _ragged(rng, [5, 2, 9, 1, 6, 3, 8], dim=16)
    expected = _naive(q_flat, q_off, d_flat, d_off)
    # Tiny chunk forces many chunks, including single-doc ones.
    for chunk in (1, 7, 1000):
        for q_chunk in (1, 5, 1000):  # one query per pass ... all queries in one pass
            got = maxsim_scores(q_flat, q_off, d_flat, d_off, chunk_tokens=chunk, show_progress=False,
                                q_chunk_tokens=q_chunk)
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_subset_ragged():
    rng = np.random.default_rng(1)
    flat, off = _ragged(rng, [2, 3, 4], dim=4)
    sub_flat, sub_off = subset_ragged(flat, off, [2, 0])
    assert sub_off.tolist() == [0, 4, 6]
    assert np.array_equal(sub_flat[:4], flat[5:9])
    assert np.array_equal(sub_flat[4:], flat[0:2])


def test_top_k_is_sorted_best_first():
    scores = np.array([[0.1, 0.9, 0.5, 0.3]], dtype=np.float32)
    run = top_k(scores, ["q"], ["a", "b", "c", "d"], k=3)
    assert [d for d, _ in run["q"]] == ["b", "c", "d"]
