"""BM25, the classic lexical baseline, written from scratch.

For a query q and document d:

    BM25(q, d) = sum over query terms t of  IDF(t) * tf(t,d) * (k1 + 1)
                                            ---------------------------------------------
                                            tf(t,d) + k1 * (1 - b + b * len(d) / avg_len)

    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

  tf(t,d)  how often t occurs in d          N       number of documents
  df(t)    how many documents contain t     len(d)  document length in terms
  k1       how fast repeated terms saturate (1.2): the 5th "protein" adds less than the 1st
  b        how much long documents are penalised (0.75)

Everything except the query depends only on the corpus, so we precompute one
weight per (document, term) and BM25 becomes a sparse dot product:
score = (query term counts) . (document weights), the same machinery as SMVE.
Text is lowercased, split into words, stop words removed and stemmed (Snowball
English), roughly what Lucene/Pyserini do for the standard BEIR BM25 numbers.
"""

from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp
import Stemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"[a-z0-9]+")
_stem = Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]
    return _stem.stemWords(words)


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[str]) -> "BM25":
        tokenized = [tokenize(d) for d in docs]
        vocab: dict[str, int] = {}
        rows, cols, counts = [], [], []
        for i, toks in enumerate(tokenized):
            terms, tf = np.unique([vocab.setdefault(t, len(vocab)) for t in toks], return_counts=True)
            rows.extend([i] * len(terms))
            cols.extend(terms)
            counts.extend(tf)
        tf = sp.csr_matrix((np.array(counts, dtype=np.float32), (rows, cols)), shape=(len(docs), len(vocab)))

        n_docs = len(docs)
        doc_len = np.array([len(t) for t in tokenized], dtype=np.float32)
        df = np.bincount(tf.indices, minlength=len(vocab))
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Per-entry BM25 weight: idf * tf*(k1+1) / (tf + k1*(1 - b + b*len/avg))
        norm = self.k1 * (1 - self.b + self.b * doc_len / doc_len.mean())   # one value per document
        t = tf.data
        row_norm = np.repeat(norm, np.diff(tf.indptr))
        w = idf[tf.indices] * t * (self.k1 + 1) / (t + row_norm)
        self.doc_weights = sp.csr_matrix((w.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)
        self.vocab, self.idf = vocab, idf
        return self

    def encode_queries(self, queries: list[str]) -> sp.csr_matrix:
        """Query term counts (terms not in the corpus vocabulary are dropped: they match nothing)."""
        rows, cols = [], []
        for i, q in enumerate(queries):
            for t in tokenize(q):
                if t in self.vocab:
                    rows.append(i)
                    cols.append(self.vocab[t])
        m = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(len(queries), len(self.vocab)))
        m.sum_duplicates()
        return m

    def scores(self, q: sp.csr_matrix) -> np.ndarray:
        return np.asarray((q @ self.doc_weights.T).todense(), dtype=np.float32)

    def index_bytes(self) -> int:
        m = self.doc_weights
        return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
