"""SMVE: the fixed and batched versions must agree with the original notebook code."""

import numpy as np
import torch

from smve_lab.smve import make_anchors, smve, smve_encode, smve_scores


def _original_smve(x, k, B, is_query):
    """The notebook version, kept verbatim (minus dtype handling) as ground truth."""
    projections = x @ B
    values, indices = torch.topk(projections, k, dim=-1)
    sparse = torch.zeros_like(projections)
    sparse.scatter_(dim=-1, index=indices, src=values)
    pooled = sparse.sum(dim=0)
    if is_query:
        return pooled
    mask = torch.zeros_like(projections)
    mask.scatter_(dim=-1, index=indices, src=torch.ones_like(values))
    return pooled / mask.sum(dim=0).clamp(min=1)


def _ragged(rng, lengths, dim):
    flat = rng.standard_normal((sum(lengths), dim)).astype(np.float16)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return flat, offsets


def test_anchors_are_unit_and_seeded():
    B = make_anchors(16, 40, seed=3)
    assert B.shape == (16, 40)
    torch.testing.assert_close(B.norm(dim=0), torch.ones(40))
    assert torch.equal(B, make_anchors(16, 40, seed=3))
    assert not torch.equal(B, make_anchors(16, 40, seed=4))


def test_smve_matches_original():
    rng = np.random.default_rng(0)
    B = make_anchors(16, 64, seed=0)
    x = rng.standard_normal((9, 16)).astype(np.float16)
    xt = torch.from_numpy(x.astype(np.float32))
    for is_query in (True, False):
        torch.testing.assert_close(smve(x, 5, B, is_query), _original_smve(xt, 5, B, is_query))


def test_batched_matches_single_across_chunks():
    rng = np.random.default_rng(1)
    B = make_anchors(16, 64, seed=0)
    flat, offsets = _ragged(rng, [3, 1, 7, 4, 10, 2], dim=16)
    for is_query in (True, False):
        expected = np.stack([
            smve(flat[offsets[i]:offsets[i + 1]], 5, B, is_query).numpy()
            for i in range(len(offsets) - 1)
        ])
        # max_elems=64 -> 1 token per chunk budget: forces one item per chunk.
        for max_elems in (64, 64 * 8, 10**6):
            got = smve_encode(flat, offsets, B, 5, is_query, max_elems=max_elems,
                              show_progress=False).toarray()
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_scores_are_dot_products():
    rng = np.random.default_rng(2)
    B = make_anchors(16, 64, seed=0)
    qf, qo = _ragged(rng, [3, 5], dim=16)
    df, do = _ragged(rng, [4, 6, 2], dim=16)
    q = smve_encode(qf, qo, B, 4, True, show_progress=False)
    d = smve_encode(df, do, B, 4, False, show_progress=False)
    np.testing.assert_allclose(smve_scores(q, d), q.toarray() @ d.toarray().T, rtol=1e-5)
