"""Exhaustive ColBERT-style late-interaction (MaxSim) scoring.

For a query q with token vectors q_1..q_n and a document d with token vectors
d_1..d_m:

    MaxSim(q, d) = sum_{i=1..n}  max_{j=1..m}  q_i . d_j

Every query token finds its best-matching document token; those best matches
are summed. BGE-M3's ColBERT vectors are already L2-normalized, so each dot
product is a cosine similarity in [-1, 1].

The naive way (see notebooks/Max_SIM_from_scratch.ipynb) loops over every
(query, doc) pair. That's 300 x 5183 = 1.5M small matmuls for scifact, which
is slow in Python. Instead we exploit the flat/offsets layout from storage.py:

  1. Take ALL query tokens at once:            Q  (Tq, dim)
  2. Take a chunk of documents' tokens at once: D  (Td, dim)
  3. One big matmul:                            S = Q @ D.T  (Tq, Td)
     S[i, j] = similarity of query token i with document token j.
  4. Max over each document's column segment -> (Tq, n_docs_in_chunk)
  5. Sum over each query's row segment       -> (n_queries, n_docs_in_chunk)

Steps 4-5 use `np.maximum.reduceat` / `np.add.reduceat`, which apply a
reduction over consecutive segments given their start offsets - exactly the
shape of our ragged token layout. The result is identical to the naive loop
(tests/test_maxsim.py checks this), just computed in a few large BLAS calls.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def subset_ragged(
    flat: np.ndarray, offsets: np.ndarray, indices: list[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pick items `indices` out of a flat/offsets ragged array, returning a new
    (flat, offsets) pair containing only those items, in that order."""
    pieces = [flat[offsets[i] : offsets[i + 1]] for i in indices]
    lens = np.array([len(p) for p in pieces])
    new_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return np.concatenate(pieces), new_offsets


def maxsim_scores(
    q_flat: np.ndarray,
    q_offsets: np.ndarray,
    d_flat: np.ndarray,
    d_offsets: np.ndarray,
    chunk_tokens: int = 32_768,
    show_progress: bool = True,
    q_chunk_tokens: int = 16_384,
) -> np.ndarray:
    """Score every query against every document with MaxSim.

    Args:
        q_flat, q_offsets: ragged query token vectors (see storage.py).
        d_flat, d_offsets: ragged document token vectors; d_flat may be mmapped.
        chunk_tokens: approximate number of document tokens per matmul.
        q_chunk_tokens: approximate number of query tokens per pass over the
            documents. The similarity block is (q_chunk_tokens x chunk_tokens)
            float32, ~2 GB at the defaults. Small query sets (e.g. SciFact's
            7.9k tokens) fit in one pass; long-query sets (ArguAna, ~280k
            tokens) take several passes over the documents.

    Returns:
        (n_queries, n_docs) float32 matrix of MaxSim scores.
    """
    n_q, n_d = len(q_offsets) - 1, len(d_offsets) - 1
    # reduceat silently misbehaves on empty segments, so rule them out up front.
    assert np.all(np.diff(q_offsets) > 0), "every query needs at least one token"
    assert np.all(np.diff(d_offsets) > 0), "every document needs at least one token"

    scores = np.empty((n_q, n_d), dtype=np.float32)
    q_batches = []
    qs = 0
    while qs < n_q:
        qe = int(np.searchsorted(q_offsets, q_offsets[qs] + q_chunk_tokens, side="right")) - 1
        qe = min(max(qe, qs + 1), n_q)
        q_batches.append((qs, qe))
        qs = qe

    pbar = tqdm(total=n_d * len(q_batches), desc="MaxSim", unit="doc", disable=not show_progress)
    for qs, qe in q_batches:
        Q = np.asarray(q_flat[q_offsets[qs]:q_offsets[qe]], dtype=np.float32)
        q_starts = q_offsets[qs:qe] - q_offsets[qs]
        start = 0
        while start < n_d:
            # Grow the chunk until it holds ~chunk_tokens tokens (at least one doc).
            end = int(np.searchsorted(d_offsets, d_offsets[start] + chunk_tokens, side="right")) - 1
            end = min(max(end, start + 1), n_d)

            t0, t1 = d_offsets[start], d_offsets[end]
            D = np.asarray(d_flat[t0:t1], dtype=np.float32)  # (Td, dim)
            S = Q @ D.T  # (Tq, Td): every query token vs every doc token

            doc_starts = d_offsets[start:end] - t0
            best_per_doc = np.maximum.reduceat(S, doc_starts, axis=1)  # (Tq, n_docs)
            scores[qs:qe, start:end] = np.add.reduceat(best_per_doc, q_starts, axis=0)

            pbar.update(end - start)
            start = end
    pbar.close()
    return scores


def top_k(
    scores: np.ndarray, query_ids: list[str], doc_ids: list[str], k: int
) -> dict[str, list[tuple[str, float]]]:
    """Turn a score matrix into a ranked run: {query_id: [(doc_id, score), ...]}
    sorted best-first, keeping the top k documents per query."""
    k = min(k, scores.shape[1])
    # argpartition finds the top k in O(n); we then sort only those k.
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    run = {}
    for row, qid in enumerate(query_ids):
        idx = part[row][np.argsort(-scores[row, part[row]], kind="stable")]
        run[qid] = [(doc_ids[j], float(scores[row, j])) for j in idx]
    return run
