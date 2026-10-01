"""SMVE: compress a bag of token vectors into ONE sparse vector.

MaxSim compares every query token with every document token, so its cost grows
with (query tokens x corpus tokens) and it has to keep all 1.86M document token
vectors around. SMVE instead turns each text into a single sparse vector
once, and then scores with one sparse dot product:

  1. Projection:     P = X @ B          X: (n_tokens, d) token vectors
                                        B: (d, w) random unit-length "anchors"
     P[t, a] is how well token t points in the direction of anchor a.
  2. Sparsification: keep each token's top-k anchors, zero the rest.
     A token is now described by the k anchors it is closest to.
  3. Pooling:        add up the kept values per anchor.
       - query: SUM over tokens                 -> q[a]
       - doc:   MEAN over the tokens that hit a -> d[a]
  4. Score:          q . d = sum_a q[a] * d[a]

Repetitions (`reps` = R): split the anchors into R independent blocks of w
and take the top-k inside EACH block, so every token keeps R*k anchors, k per
block. Concatenating R independent SMVE vectors is exactly this, because R
blocks of w random columns are statistically the same as one matrix with R*w
random columns - only where the top-k is taken differs. It gives a matching
query/doc token pair R independent chances to share an anchor (fewer exact
zeros) and averages R independent estimates (less noise). R=1 is plain SMVE.

Why this approximates MaxSim: a query token contributes to anchor a only if a
is one of its nearest anchors, and d[a] is the average strength of the doc
tokens that sit near that same anchor - i.e. doc tokens that are *close to the
query token*. So "sum over query tokens of how well the doc matches them" is
kept, but "max over doc tokens" is replaced by "average over nearby doc tokens".

Two implementations live here:
  - `smve`:        one text at a time; the readable reference (and what the
                   tests check the fast version against).
  - `smve_encode`: the whole corpus in token chunks, returning a scipy CSR
                   matrix. Same numbers, much less memory and Python overhead.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


def make_anchors(dim: int, w: int, seed: int = 0) -> torch.Tensor:
    """(dim, w) float32 matrix whose columns are random unit vectors.

    Gaussian samples are rotationally symmetric, so normalizing them gives
    directions spread uniformly over the sphere. The seed matters: queries and
    documents MUST be encoded with the same B, and re-running with the same
    seed reproduces the same index.
    """
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(dim, w, generator=g, dtype=torch.float32)
    return B / B.norm(dim=0, keepdim=True)


def topk_blocks(P: torch.Tensor, k: int, reps: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k of each row within each of `reps` equal column blocks.

    P: (T, reps*w) projections. Returns values and GLOBAL column indices, both
    (T, reps*k). With reps=1 this is just torch.topk(P, k).
    """
    T, W = P.shape
    assert W % reps == 0, "anchor count must be divisible by reps"
    w = W // reps
    values, indices = torch.topk(P.view(T, reps, w), k, dim=-1)  # (T, reps, k)
    indices = indices + torch.arange(reps, device=P.device)[None, :, None] * w  # local -> global
    return values.reshape(T, reps * k), indices.reshape(T, reps * k)


def smve(token_embeddings, k: int, B: torch.Tensor, is_query: bool, reps: int = 1) -> torch.Tensor:
    """Encode one text's token vectors (n_tokens, d) into a (w,) SMVE vector.

    With reps > 1, B holds reps blocks of anchors (w = reps * block width).
    """
    x = torch.as_tensor(np.asarray(token_embeddings)).to(torch.float32)
    projections = x @ B  # (n_tokens, w)
    values, indices = topk_blocks(projections, k, reps)  # (n_tokens, reps*k)

    # Pool straight from the (n_tokens, k) top-k lists instead of scattering
    # them into two dense (n_tokens, w) matrices first.
    w = B.shape[1]
    pooled = torch.zeros(w).index_add_(0, indices.flatten(), values.flatten())
    if is_query:
        return pooled
    counts = torch.bincount(indices.flatten(), minlength=w).clamp(min=1)
    return pooled / counts


def smve_encode(
    flat: np.ndarray,
    offsets: np.ndarray,
    B: torch.Tensor,
    k: int,
    is_query: bool,
    max_elems: int = 2**26,
    device: str = "cpu",
    show_progress: bool = True,
    center: np.ndarray | None = None,
    reps: int = 1,
) -> sp.csr_matrix:
    """Encode every item of a ragged (flat, offsets) token array with SMVE.

    Documents are processed in chunks of whole items so that the (tokens x w)
    projection block stays under `max_elems` floats (2**26 = 256 MB). Within a
    chunk, each token's top-k (item, anchor, value) triplets are merged with
    `torch.unique`, so we never build a dense (items x w) matrix. Result is an
    (n_items, w) CSR matrix - most of its w entries are zero.

    `center` (optional, shape (d,)): a vector subtracted from every token
    before projecting, followed by re-normalizing to unit length. See
    `token_mean` for why this helps.

    `reps`: number of anchor blocks in B (see module docstring). B must have
    reps * (block width) columns, e.g. make_anchors(d, reps * w, seed).
    """
    n_items, w = len(offsets) - 1, B.shape[1]
    B = B.to(device)
    mu = None if center is None else torch.as_tensor(center, dtype=torch.float32, device=device)
    lens = np.diff(offsets)
    tokens_per_chunk = max(1, max_elems // w)

    rows, cols, vals = [], [], []
    pbar = tqdm(total=n_items, desc="SMVE " + ("queries" if is_query else "docs"),
                unit="item", disable=not show_progress)
    start = 0
    while start < n_items:
        end = int(np.searchsorted(offsets, offsets[start] + tokens_per_chunk, side="right")) - 1
        end = min(max(end, start + 1), n_items)
        t0, t1 = offsets[start], offsets[end]

        X = torch.from_numpy(np.asarray(flat[t0:t1], dtype=np.float32)).to(device)
        if mu is not None:
            X = torch.nn.functional.normalize(X - mu, dim=1)
        values, indices = topk_blocks(X @ B, k, reps)  # (T, reps*k)

        # Which item (0-based within this chunk) each token belongs to.
        item = torch.repeat_interleave(
            torch.arange(end - start, device=device), torch.from_numpy(lens[start:end]).to(device)
        )
        # One integer key per (item, anchor) pair; equal keys get pooled together.
        keys = (item[:, None] * w + indices).flatten()
        uniq, inverse = torch.unique(keys, return_inverse=True)
        pooled = torch.zeros(len(uniq), device=device).index_add_(0, inverse, values.flatten())
        if not is_query:
            pooled /= torch.bincount(inverse, minlength=len(uniq))

        uniq = uniq.cpu().numpy()
        rows.append((uniq // w + start).astype(np.int32))
        cols.append((uniq % w).astype(np.int32))
        vals.append(pooled.cpu().numpy())

        pbar.update(end - start)
        start = end
    pbar.close()

    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, w),
        dtype=np.float32,
    )


def token_mean(flat: np.ndarray, sample: int = 200_000, seed: int = 0) -> np.ndarray:
    """Mean token vector, estimated from a random sample of tokens.

    BGE-M3 ColBERT vectors are anisotropic: they all lean towards a shared
    direction (two random tokens have cosine ~0.29, not ~0). Random anchors
    near that direction then land in almost every token's top-k and carry no
    information. Subtracting this mean before projecting removes the shared
    component so anchors are chosen by what's *specific* to each token.
    Compute it from documents only and reuse it for queries.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(flat), size=min(sample, len(flat)), replace=False))
    return np.asarray(flat[idx], dtype=np.float32).mean(axis=0)


def smve_scores(q_vecs: sp.csr_matrix, d_vecs: sp.csr_matrix) -> np.ndarray:
    """(n_queries, n_docs) score matrix: one sparse-sparse matrix product.

    Only anchors that are non-zero in BOTH the query and the doc cost
    anything, which is what makes this so much cheaper than MaxSim.
    """
    return np.asarray((q_vecs @ d_vecs.T).todense(), dtype=np.float32)


def csr_nbytes(m: sp.csr_matrix) -> int:
    """Bytes needed to store a CSR matrix (values + column indices + row pointers)."""
    return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
