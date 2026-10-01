"""Step 1b report: tables and plots for the SMVE (w, k, R, centering) sweep.

Reads every run in results/scifact_bgem3/smve/*/summary.json (via runs.csv,
rebuilt here) plus the MaxSim, dense and dense+lexical summaries as reference
lines, and writes results/scifact_bgem3/smve/sweep/:

    runs.csv       one row per SMVE run (snapshot)
    meta.json      grid actually present, machine, date
    summary.md     best settings, seed spread, repetitions vs single matrix
    plots/         metric_vs_w_{not_centered,centered}.png, heatmap_ndcg10.png,
                   centering_effect.png, reps_equal_budget.png, tradeoff.png

    uv run python scripts/plot_smve_sweep.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_smve import SMVE_DIR, collect_runs  # noqa: E402

from smve_lab.config import RESULTS_DIR  # noqa: E402
from smve_lab.plots import (  # noqa: E402
    plot_sweep_centering_effect,
    plot_sweep_heatmaps,
    plot_sweep_metric_vs_w,
    plot_sweep_reps,
    plot_sweep_tradeoff,
)

BASE = RESULTS_DIR / "scifact_bgem3"
OUT = SMVE_DIR / "sweep"
REFS = {"MaxSim": "colbert_maxsim", "BGE-M3 dense + lexical": "hybrid_dense_lexical", "BGE-M3 dense": "dense"}
SHOW = ["w", "k", "reps", "center", "nnz_per_doc", "single_query_latency_ms", "index_bytes",
        "ndcg@10", "recall@10", "recall@100", "mrr@10"]


def load_refs() -> dict:
    refs = {}
    for name, folder in REFS.items():
        s = json.loads((BASE / folder / "summary.json").read_text())
        refs[name] = {**s["metrics"], "single_query_latency_ms": s["timing"]["single_query_latency_ms"],
                      "index_bytes": s["index"]["bytes"], "nnz_per_doc": None}
    return refs


def md(df: pd.DataFrame) -> str:
    df = df.copy()
    if "index_bytes" in df:
        df["index_bytes"] = (df["index_bytes"] / 1e6).round(1).astype(str) + " MB"
        df = df.rename(columns={"index_bytes": "index"})
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main() -> None:
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    runs["center"] = runs["center"].astype(bool)
    runs.to_csv(OUT / "runs.csv", index=False)
    refs = load_refs()
    maxsim = refs["MaxSim"]["ndcg@10"]

    main_grid = runs[(runs.seed == 0) & (runs.reps == 1)]
    (OUT / "meta.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_runs": len(runs),
        "w": sorted(runs.w.unique().tolist()), "k": sorted(runs.k.unique().tolist()),
        "reps": sorted(runs.reps.unique().tolist()), "seeds": sorted(runs.seed.unique().tolist()),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "torch_threads": torch.get_num_threads()},
        "note": "all timings on CPU; SciFact test split (300 queries)",
    }, indent=2))

    # --- Plots -----------------------------------------------------------------
    p = OUT / "plots"
    for center, tag in ((False, "not_centered"), (True, "centered")):
        plot_sweep_metric_vs_w(main_grid[main_grid.center == center], refs, p / f"metric_vs_w_{tag}.png",
                               f"SMVE on SciFact vs. w and k ({tag.replace('_', ' ')}, R = 1)")
    plot_sweep_heatmaps(main_grid, maxsim, p / "heatmap_ndcg10.png")
    plot_sweep_centering_effect(main_grid, p / "centering_effect.png")
    rep_runs = runs[runs.seed == 0]
    bases = sorted({(r.w, r.k) for r in rep_runs[rep_runs.reps > 1].itertuples()})
    plot_sweep_tradeoff(rep_runs, refs, p / "tradeoff.png")

    # --- Summary tables ------------------------------------------------------
    best = rep_runs.sort_values("ndcg@10", ascending=False).head(10)[SHOW]
    by_k = main_grid.loc[main_grid.groupby(["center", "k"])["ndcg@10"].idxmax(), SHOW]
    seeds = runs.groupby(["w", "k", "reps", "center"]).agg(
        seeds=("seed", "count"), ndcg10_mean=("ndcg@10", "mean"), ndcg10_std=("ndcg@10", "std"),
        recall100_mean=("recall@100", "mean"), recall100_std=("recall@100", "std")).query("seeds > 1").reset_index()
    rows = []
    for w, k in bases:
        for center in (False, True):
            for r in (2, 4, 8):
                a = rep_runs[(rep_runs.w == w) & (rep_runs.k == k) & (rep_runs.reps == r) & (rep_runs.center == center)]
                b = rep_runs[(rep_runs.w == w * r) & (rep_runs.k == k * r) & (rep_runs.reps == 1)
                             & (rep_runs.center == center)]
                if len(a) and len(b):
                    rows.append({"base": f"w={w} k={k}", "center": center, "R": r,
                                 "reps nDCG@10": a["ndcg@10"].iat[0], "single nDCG@10": b["ndcg@10"].iat[0],
                                 "Δ nDCG@10": a["ndcg@10"].iat[0] - b["ndcg@10"].iat[0],
                                 "reps R@100": a["recall@100"].iat[0], "single R@100": b["recall@100"].iat[0]})
    reps_vs_single = pd.DataFrame(rows)
    if len(reps_vs_single) and len(seeds):
        reps_vs_single.to_csv(OUT / "reps_vs_single.csv", index=False)
        plot_sweep_reps(reps_vs_single, float(seeds["ndcg10_std"].mean()), p / "reps_equal_budget.png")

    ref_df = pd.DataFrame([{"method": n, **{m: v[m] for m in ("ndcg@10", "recall@10", "recall@100", "mrr@10")}}
                           for n, v in refs.items()])
    text = [
        f"# SMVE sweep on SciFact ({len(runs)} runs)\n",
        "## Reference methods\n", md(ref_df),
        "\n## Top 10 settings by nDCG@10\n", md(best),
        "\n## Best w for each k (R = 1, seed 0)\n", md(by_k),
        "\n## Repetitions vs single matrix at equal budget\n",
        md(reps_vs_single) if len(reps_vs_single) else "(no repetition runs yet)",
        "\n## Seed spread\n", md(seeds) if len(seeds) else "(single seed only)",
    ]
    (OUT / "summary.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
