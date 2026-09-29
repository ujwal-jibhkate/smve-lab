"""Shared evaluate-and-save step used by every retrieval method.

Any method (MaxSim, SMVE, ...) only has to produce a (n_queries, n_docs) score
matrix. This module turns it into rankings, metrics, files and plots in one
standard layout, so every run can be compared with every other run:

    <out_dir>/summary.json       mean metrics, timings, index size, settings
    <out_dir>/per_query.csv      every metric for every query
    <out_dir>/metrics_at_k.csv   mean metrics for k = 1..depth
    <out_dir>/run.trec           top-`depth` ranking per query (TREC format)
    <out_dir>/scores.npz         full score matrix + ids (optional)
    <out_dir>/plots/*.png        (optional)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from smve_lab.maxsim import top_k
from smve_lab.metrics import evaluate_run
from smve_lab.plots import (
    plot_first_relevant_rank,
    plot_metrics_vs_k,
    plot_per_query_hist,
    plot_score_separation,
)

KS = [1, 3, 5, 10, 20, 50, 100]
CURVE_METRICS = ["ndcg", "recall", "precision", "mrr"]


class Timer:
    """`with Timer() as t: ...` then read `t.seconds`."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0


def median_latency_ms(fn: Callable[[int], object], n_items: int, n: int = 10, warmup: int = 1) -> float:
    """Median wall-clock time of `fn(i)` over `n` different items, in ms.

    Batch timings tell you throughput; this tells you how long ONE query
    takes when it arrives on its own, which is what a user waits for.
    """
    idx = np.linspace(0, n_items - 1, n + warmup).astype(int)
    for i in idx[:warmup]:
        fn(int(i))
    times = []
    for i in idx[warmup:]:
        with Timer() as t:
            fn(int(i))
        times.append(t.seconds * 1000)
    return float(np.median(times))


def write_trec_run(run: dict[str, list[tuple[str, float]]], path: Path, tag: str) -> None:
    """TREC format: `query_id Q0 doc_id rank score tag`, one line per retrieved doc."""
    with open(path, "w") as f:
        for qid, ranked in run.items():
            for rank, (did, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {did} {rank} {score:.6f} {tag}\n")


def read_trec_run(path: Path) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            qid, _, did, *_ = line.split()
            run.setdefault(qid, []).append(did)
    return run


def evaluate_and_save(
    scores: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
    out_dir: Path,
    *,
    run_id: str,
    run_title: str,
    summary_extra: dict | None = None,
    depth: int = 100,
    score_norm: np.ndarray | None = None,
    save_scores: bool = True,
    make_plots: bool = True,
    verbose: bool = True,
) -> dict:
    """Rank, evaluate, save and plot one run. Returns the summary dict.

    score_norm: optional per-query divisor for the score-separation plot only
    (e.g. query token count, since both MaxSim and SMVE sum over query tokens
    and so grow with query length). Rankings are unaffected.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run = top_k(scores, query_ids, doc_ids, depth)
    ranked_ids = {qid: [d for d, _ in ranked] for qid, ranked in run.items()}
    ks = [k for k in KS if k <= depth]
    per_query = evaluate_run(ranked_ids, qrels, ks)
    means = per_query[[c for c in per_query.columns if "@" in c]].mean()

    curve_ks = list(range(1, depth + 1))
    curve_pq = evaluate_run(ranked_ids, qrels, curve_ks, metrics=CURVE_METRICS)
    curves = pd.DataFrame(
        {m: [curve_pq[f"{m}@{k}"].mean() for k in curve_ks] for m in CURVE_METRICS},
        index=pd.Index(curve_ks, name="k"),
    )

    summary = {
        "run": run_id,
        "title": run_title,
        "dataset": "BeIR/scifact",
        "n_queries": len(query_ids),
        "n_docs": len(doc_ids),
        "depth": depth,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(summary_extra or {}),
        "metrics": {k: round(float(v), 5) for k, v in means.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    per_query.to_csv(out_dir / "per_query.csv")
    curves.to_csv(out_dir / "metrics_at_k.csv")
    write_trec_run(run, out_dir / "run.trec", tag=run_id)
    if save_scores:
        np.savez(out_dir / "scores.npz", scores=scores,
                 query_ids=np.array(query_ids), doc_ids=np.array(doc_ids))

    if make_plots:
        plots = out_dir / "plots"
        plot_metrics_vs_k(curves, plots / "metrics_at_k.png", run_title)
        plot_per_query_hist(per_query["ndcg@10"], plots / "ndcg10_per_query.png", "nDCG@10", run_title)
        plot_first_relevant_rank(per_query["first_relevant_rank"], plots / "first_relevant_rank.png",
                                 depth, run_title)
        norm = np.ones(len(query_ids)) if score_norm is None else score_norm
        rel, nonrel = [], []
        for i, qid in enumerate(query_ids):
            for did, s in run[qid]:
                (rel if qrels[qid].get(did, 0) > 0 else nonrel).append(s / norm[i])
        plot_score_separation(np.array(rel), np.array(nonrel), plots / "score_separation.png",
                              f"{run_title}  ·  docs in each query's top {depth}",
                              "score / number of query tokens" if score_norm is not None else "score")

    if verbose:
        table = pd.DataFrame(
            {m: [means[f"{m}@{k}"] for k in ks] for m in ["ndcg", "recall", "precision", "mrr", "map"]},
            index=[f"@{k}" for k in ks],
        )
        print(f"\n{run_title} - mean over {len(query_ids)} queries:")
        print(table.round(4).to_string())
        print(f"saved to {out_dir}")
    return summary
