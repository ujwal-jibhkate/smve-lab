"""Ranking metrics for retrieval evaluation, written from scratch.

Terminology used throughout:
  - run:   {query_id: [doc_id, ...]}  ranked best-first (what the system returned)
  - qrels: {query_id: {doc_id: relevance}}  ground-truth judgments; any doc
           not listed has relevance 0. In scifact every judgment is 1.

All metrics are computed per query, then averaged over queries (macro
average). This matches the BEIR / pytrec_eval conventions, so numbers are
comparable with published results:
  - nDCG uses linear gain (rel / log2(rank + 1)), not 2^rel - 1. For binary
    relevance like scifact the two are identical.
  - Recall@k divides by ALL relevant docs for the query, not by k.
  - MAP@k divides by ALL relevant docs for the query (pytrec_eval's map_cut).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dcg(relevances: list[float] | np.ndarray) -> float:
    """Discounted cumulative gain: sum_i rel_i / log2(i + 1), with rank i from 1."""
    rel = np.asarray(relevances, dtype=np.float64)
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))  # log2(rank + 1) for rank = 1..n
    return float(np.sum(rel / discounts))


def ndcg_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    """DCG of the top k returned docs, divided by the best DCG achievable (IDCG):
    the DCG you'd get if the k slots held the most relevant docs in order."""
    gains = [qrel.get(doc, 0) for doc in ranked[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def _n_relevant(qrel: dict[str, int]) -> int:
    return sum(1 for r in qrel.values() if r > 0)


def _hits(ranked: list[str], qrel: dict[str, int], k: int) -> list[bool]:
    return [qrel.get(doc, 0) > 0 for doc in ranked[:k]]


def precision_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    """Fraction of the top k slots that hold a relevant doc. Always divides by
    k, so if a query has only 1 relevant doc, P@10 can be at most 0.1."""
    return sum(_hits(ranked, qrel, k)) / k


def recall_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    """Fraction of all relevant docs that appear in the top k."""
    n_rel = _n_relevant(qrel)
    return sum(_hits(ranked, qrel, k)) / n_rel if n_rel else 0.0


def mrr_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    """1 / rank of the first relevant doc within the top k, else 0."""
    for rank, hit in enumerate(_hits(ranked, qrel, k), start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ap_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    """Average precision: mean of P@rank taken at each rank that holds a
    relevant doc, divided by the total number of relevant docs."""
    n_rel = _n_relevant(qrel)
    if n_rel == 0:
        return 0.0
    total, found = 0.0, 0
    for rank, hit in enumerate(_hits(ranked, qrel, k), start=1):
        if hit:
            found += 1
            total += found / rank
    return total / n_rel


def first_relevant_rank(ranked: list[str], qrel: dict[str, int]) -> int | None:
    """1-based rank of the first relevant doc in the run, or None if absent."""
    for rank, doc in enumerate(ranked, start=1):
        if qrel.get(doc, 0) > 0:
            return rank
    return None


METRICS = {
    "ndcg": ndcg_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr_at_k,
    "map": ap_at_k,
}


def evaluate_run(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ks: list[int],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Per-query metrics table: one row per judged query, columns like 'ndcg@10'.

    Only queries present in `qrels` are scored (unjudged queries can't be
    evaluated). A judged query missing from `run` scores 0 on everything.
    Averaging a column (df.mean()) gives the usual reported number.
    """
    metrics = metrics or list(METRICS)
    rows = []
    for qid, qrel in qrels.items():
        ranked = run.get(qid, [])
        row = {"query_id": qid, "n_relevant": _n_relevant(qrel)}
        for name in metrics:
            for k in ks:
                row[f"{name}@{k}"] = METRICS[name](ranked, qrel, k)
        row["first_relevant_rank"] = first_relevant_rank(ranked, qrel)
        rows.append(row)
    return pd.DataFrame(rows).set_index("query_id")
