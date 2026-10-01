"""A minimal inverted index for sparse vectors (BM25, BGE-M3 lexical, SMVE).

A sparse document matrix stored row by row (one row per document) answers
"which terms does document d contain?". Search needs the opposite question,
"which documents contain term t?", so we store it column by column:

    term t  ->  posting list: [doc ids containing t], [their weights]

That's an inverted index, and it is exactly the CSC ("compressed sparse
column") layout of the document matrix: indptr[t]:indptr[t+1] slices out
term t's postings.

Scoring a query (term-at-a-time): start every document at 0, and for each
query term t with weight q_t add q_t * w(t, d) to every document d in t's
posting list. Documents that share no term with the query are never touched.
Work = sum over query terms of their posting-list lengths ("postings read"),
which is the number that drives latency. Real engines (Lucene, PISA) add
dynamic pruning (WAND / MaxScore) to skip postings that can't reach the
top-k; that helps most when queries are short. This index is exhaustive (no
pruning), so its scores equal the plain sparse matrix product exactly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def ragged_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate range(s, s+l) for every (s, l) pair, without a Python loop.

    e.g. starts=[10, 50], lengths=[3, 2] -> [10, 11, 12, 50, 51]
    """
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    # Each output position = its range's start + its offset inside that range.
    range_id = np.repeat(np.arange(len(lengths)), lengths)
    offset = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return starts[range_id] + offset


class InvertedIndex:
    def __init__(self, doc_matrix: sp.csr_matrix):
        csc = sp.csc_matrix(doc_matrix, dtype=np.float32)
        csc.sort_indices()
        self.n_docs, self.n_terms = csc.shape
        self.indptr = csc.indptr.astype(np.int64)    # term t's postings live in [indptr[t], indptr[t+1])
        self.doc_ids = csc.indices.astype(np.int32)  # posting entries: which document
        self.weights = csc.data                      # posting entries: weight of term t in that document
        self.df = np.diff(self.indptr)               # posting-list length = document frequency

    def nbytes(self) -> int:
        return self.indptr.nbytes + self.doc_ids.nbytes + self.weights.nbytes

    def postings_read(self, terms: np.ndarray) -> int:
        return int(self.df[terms].sum())

    def score(self, terms: np.ndarray, q_weights: np.ndarray) -> np.ndarray:
        """Exhaustive term-at-a-time scoring -> one score per document."""
        lengths = self.df[terms]
        pos = ragged_ranges(self.indptr[terms], lengths)             # every posting entry to read
        contrib = self.weights[pos] * np.repeat(q_weights, lengths)  # q_t * w(t, d)
        return np.bincount(self.doc_ids[pos], weights=contrib, minlength=self.n_docs)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first."""
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]
