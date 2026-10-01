"""MUVERA's batched encoder vs. a naive loop, and BM25 vs. a hand computation."""

import math

import numpy as np
import pytest
import torch

from smve_lab.bm25 import BM25, tokenize
from smve_lab.muvera import MuveraFDE


def _ragged(rng, lengths, dim):
    flat = rng.standard_normal((sum(lengths), dim)).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return flat, offsets


def _naive_fde(m: MuveraFDE, X: np.ndarray, is_query: bool) -> np.ndarray:
    """Straight from the paper, one token / repetition / bucket at a time."""
    Xt = torch.from_numpy(X)
    out = np.zeros((m.reps, m.n_buckets, m.d_proj), dtype=np.float32)
    for r in range(m.reps):
        G = m.G[:, r * m.k_sim:(r + 1) * m.k_sim]
        P = m.P[:, r * m.d_proj:(r + 1) * m.d_proj]
        codes = [int(sum(int(b) << i for i, b in enumerate((x @ G > 0).tolist()))) for x in Xt]
        proj = (Xt @ P).numpy()
        for b in range(m.n_buckets):
            members = [t for t, c in enumerate(codes) if c == b]
            if members:
                v = proj[members].sum(0)
                out[r, b] = v if is_query else v / len(members)
            elif not is_query:
                ham = [bin(c ^ b).count("1") for c in codes]
                out[r, b] = proj[int(np.argmin(ham))]
    return out.reshape(-1)


@pytest.mark.parametrize("is_query", [True, False])
def test_muvera_matches_naive(is_query):
    rng = np.random.default_rng(0)
    m = MuveraFDE(dim=16, reps=3, k_sim=3, d_proj=4, seed=1)
    flat, offsets = _ragged(rng, [5, 1, 9, 3], dim=16)
    for max_tokens in (1, 6, 10_000):  # one item per chunk ... everything in one chunk
        got = m.encode(flat, offsets, is_query, max_tokens=max_tokens, show_progress=False)
        for i in range(len(offsets) - 1):
            expected = _naive_fde(m, flat[offsets[i]:offsets[i + 1]], is_query)
            np.testing.assert_allclose(got[i], expected, rtol=1e-5, atol=1e-5)


def test_muvera_dimension():
    m = MuveraFDE(dim=8, reps=20, k_sim=5, d_proj=16)
    assert m.out_dim == 20 * 32 * 16


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The proteins are binding") == ["protein", "bind"]


def test_bm25_matches_hand_computation():
    docs = ["apple banana apple", "banana cherry", "cherry cherry cherry date"]
    bm = BM25(k1=1.2, b=0.75).fit(docs)
    s = bm.scores(bm.encode_queries(["apple"]))[0]
    # "apple": df=1, N=3, tf=2 in doc 0, doc lengths 3, 2, 4 -> avg 3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 3))
    assert s[0] == pytest.approx(expected, rel=1e-5)
    assert s[1] == 0 and s[2] == 0
    assert bm.scores(bm.encode_queries(["unknownword"])).sum() == 0


def test_ragged_ranges():
    from smve_lab.inverted_index import ragged_ranges
    np.testing.assert_array_equal(ragged_ranges(np.array([10, 50, 7]), np.array([3, 2, 0])), [10, 11, 12, 50, 51])


def test_inverted_index_equals_sparse_product():
    import scipy.sparse as sp
    from smve_lab.inverted_index import InvertedIndex, top_k
    rng = np.random.default_rng(0)
    D = sp.random(50, 200, density=0.05, format="csr", random_state=1, dtype=np.float32)
    idx = InvertedIndex(D)
    for _ in range(5):
        terms = rng.choice(200, size=12, replace=False)
        qw = rng.random(12).astype(np.float32)
        q = sp.csr_matrix((qw, (np.zeros(12, int), terms)), shape=(1, 200))
        expected = (q @ D.T).toarray()[0]
        np.testing.assert_allclose(idx.score(terms, qw), expected, rtol=1e-5, atol=1e-6)
        assert idx.postings_read(terms) == int(np.diff(D.tocsc().indptr)[terms].sum())
        top = top_k(expected, 5)
        assert np.all(np.diff(expected[top]) <= 0)
