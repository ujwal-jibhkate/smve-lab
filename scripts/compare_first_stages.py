"""Stage 3a report: SMVE vs MUVERA as first stages, at equal budget.

A first stage's job is to put the relevant documents into a short candidate
list cheaply; a reranker (here exact MaxSim over the top 100) then orders them.
So for every SMVE and MUVERA setting on disk we report:

  - Recall@100                         did the relevant docs make the candidate list?
  - nDCG@10 of the first stage alone
  - nDCG@10 after MaxSim reranks its top 100   (computed exactly from the stored
                                                exhaustive MaxSim score matrix)

against two costs: index size and single-query latency. "Equal budget" =
best setting of each method whose cost is under the same limit. References:
exhaustive MaxSim, BM25, BGE-M3 dense / lexical / dense+lexical.

    uv run python scripts/compare_first_stages.py

Writes results/scifact_bgem3/first_stage/{all_runs.csv, summary.md, plots/frontiers.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smve_lab.evaluation import read_trec_run
from smve_lab.metrics import evaluate_run
from smve_lab.plots import plot_first_stage_frontiers
from smve_lab.datasets import DATASETS, info, load_qrels, results_dir

REFS = {"Exhaustive MaxSim": "colbert_maxsim", "BM25": "bm25", "BGE-M3 dense": "dense",
        "BGE-M3 lexical": "lexical", "BGE-M3 dense + lexical": "hybrid_dense_lexical"}
SIZE_BUDGETS_MB = [25, 50, 100, 200, 400]
LATENCY_BUDGETS_MS = [10, 30, 100, 250]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact", choices=list(DATASETS))
    args = p.parse_args()
    BASE = results_dir(args.dataset)
    OUT = BASE / "first_stage"
    ds = info(args.dataset)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    qrels = load_qrels(args.dataset)
    z = np.load(BASE / "colbert_maxsim" / "scores.npz")
    qi = {q: i for i, q in enumerate(z["query_ids"].tolist())}
    di = {d: j for j, d in enumerate(z["doc_ids"].tolist())}
    ms = z["scores"]

    def maxsim_rerank_ndcg(folder: Path, depth: int = 100) -> float:
        run = read_trec_run(folder / "run.trec")
        ranked = {q: sorted(r[:depth], key=lambda d: -ms[qi[q], di[d]]) + r[depth:] for q, r in run.items()}
        return float(evaluate_run(ranked, qrels, [10], metrics=["ndcg"])["ndcg@10"].mean())

    def row(folder: Path, family: str, name: str) -> dict:
        s = json.loads((folder / "summary.json").read_text())
        return {"family": family, "name": name, "run": folder.name, **s.get("params", {}),
                "index_mb": s["index"]["bytes"] / 1e6,
                "single_query_latency_ms": s["timing"]["single_query_latency_ms"],
                "index_build_seconds": s["timing"]["index_build_seconds"],
                "ndcg@10": s["metrics"]["ndcg@10"], "recall@100": s["metrics"]["recall@100"],
                "recall@20": s["metrics"]["recall@20"], "rerank_ndcg@10": maxsim_rerank_ndcg(folder)}

    rows = []
    for fam, sub in (("SMVE", "smve"), ("MUVERA", "muvera")):
        for f in sorted((BASE / sub).glob("*/summary.json")):
            if "seed0" in f.parent.name:  # one seed per setting, so families are comparable
                rows.append(row(f.parent, fam, f.parent.name))
    runs = pd.DataFrame(rows)
    refs = pd.DataFrame([row(BASE / folder, "reference", name) for name, folder in REFS.items()])
    pd.concat([runs, refs]).to_csv(OUT / "all_runs.csv", index=False)

    # Best setting of each family under each budget.
    def best_under(col: str, limit: float, metric: str) -> dict:
        out = {}
        for fam in ("SMVE", "MUVERA"):
            g = runs[(runs.family == fam) & (runs[col] <= limit)]
            if len(g):
                b = g.loc[g[metric].idxmax()]
                out[fam] = f"{b[metric]:.4f} ({b['run'].replace('_seed0', '')})"
            else:
                out[fam] = "–"
        return out

    def budget_table(col: str, limits: list, unit: str) -> str:
        lines = [f"| budget | metric | SMVE best (setting) | MUVERA best (setting) |", "|---|---|---|---|"]
        for lim in limits:
            for metric in ("recall@100", "rerank_ndcg@10", "ndcg@10"):
                b = best_under(col, lim, metric)
                lines.append(f"| ≤ {lim} {unit} | {metric} | {b['SMVE']} | {b['MUVERA']} |")
        return "\n".join(lines)

    ref_lines = ["| method | nDCG@10 | Recall@100 | nDCG@10 after MaxSim@100 | index | latency |",
                 "|---|---|---|---|---|---|"]
    for _, r in refs.iterrows():
        ref_lines.append(f"| {r['name']} | {r['ndcg@10']:.4f} | {r['recall@100']:.4f} | {r['rerank_ndcg@10']:.4f} "
                         f"| {r['index_mb']:,.1f} MB | {r['single_query_latency_ms']:.1f} ms |")
    top = runs.sort_values("rerank_ndcg@10", ascending=False).groupby("family").head(5)
    top_lines = ["| family | setting | nDCG@10 | Recall@100 | nDCG@10 after MaxSim@100 | index | latency |",
                 "|---|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        top_lines.append(f"| {r['family']} | {r['run'].replace('_seed0', '')} | {r['ndcg@10']:.4f} | "
                         f"{r['recall@100']:.4f} | {r['rerank_ndcg@10']:.4f} | {r['index_mb']:,.1f} MB | "
                         f"{r['single_query_latency_ms']:.1f} ms |")
    text = "\n".join([
        f"# First stages on {ds.display}: SMVE ({(runs.family == 'SMVE').sum()} settings) vs "
        f"MUVERA ({(runs.family == 'MUVERA').sum()} settings)\n",
        "All on CPU. MaxSim rerank = exact MaxSim over the first stage's top 100.\n",
        "## References\n", *ref_lines,
        "\n## Equal budget: index size\n", budget_table("index_mb", SIZE_BUDGETS_MB, "MB"),
        "\n## Equal budget: single-query latency\n", budget_table("single_query_latency_ms", LATENCY_BUDGETS_MS, "ms"),
        "\n## Top 5 settings per family (by nDCG@10 after MaxSim rerank)\n", *top_lines,
    ])
    (OUT / "summary.md").write_text(text + "\n")
    print(text)
    plot_first_stage_frontiers(runs, refs, OUT / "plots" / "frontiers.png", ds.display)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
