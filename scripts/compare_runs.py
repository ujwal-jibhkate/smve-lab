"""Compare SMVE runs against the MaxSim baseline: quality, per-query, cost.

Reads the folders written by evaluate_maxsim.py / evaluate_smve.py; it never
re-scores the full runs, so it's quick. The one thing it does compute is an
extra hybrid run - "SMVE top-100 -> MaxSim rerank" - because that is the
natural way to use a cheap first stage, and its timing must be measured.

    uv run python scripts/compare_runs.py
    uv run python scripts/compare_runs.py --candidates results/scifact_bgem3/smve/w16384_k32_center_seed0

Writes results/scifact_bgem3/comparison/:
    comparison.md       all tables, human-readable   <- start here
    comparison.json     the same numbers, machine-readable
    per_query.csv       nDCG@10 / Recall@100 of every run, per query
    plots/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from smve_lab.config import ARTIFACTS_DIR, RESULTS_DIR
from smve_lab.evaluation import Timer, read_trec_run
from smve_lab.maxsim import maxsim_scores, subset_ragged
from smve_lab.metrics import evaluate_run
from smve_lab.plots import (
    plot_cost,
    plot_curves_compare,
    plot_metric_bars,
    plot_per_query_delta,
    plot_tradeoff,
)
from smve_lab.scifact import load_qrels
from smve_lab.storage import load_separate

EMB_DIR = ARTIFACTS_DIR / "embeddings" / "scifact_bgem3"
BASE = RESULTS_DIR / "scifact_bgem3"
QUALITY_ROWS = ["ndcg@1", "ndcg@10", "ndcg@100", "recall@10", "recall@100",
                "precision@1", "precision@10", "mrr@10", "map@10"]
CURVE_KS = list(range(1, 101))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, default=BASE / "colbert_maxsim")
    p.add_argument("--candidates", type=Path, nargs="+",
                   default=[BASE / "smve" / "w65536_k32_seed0", BASE / "smve" / "w65536_k32_center_seed0"])
    p.add_argument("--no-rerank", action="store_true", help="skip the SMVE -> MaxSim rerank hybrid")
    p.add_argument("--rerank-from", default="SMVE",
                   help="rerank the best candidate whose label contains this text")
    p.add_argument("--rerank-depth", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=10_000, help="resamples for the confidence interval")
    p.add_argument("--out-dir", type=Path, default=BASE / "comparison")
    return p.parse_args()


def label_for(summary: dict) -> str:
    if "label" in summary:
        return summary["label"]
    if summary.get("method") == "maxsim":
        return "MaxSim"
    p = summary["params"]
    reps = p.get("reps", 1)
    return (f"SMVE w={p['w']:,} k={p['k']}" + (f" R={reps}" if reps > 1 else "")
            + (" centered" if p["center"] else ""))


def load_run(folder: Path) -> dict:
    summary = json.loads((folder / "summary.json").read_text())
    return {
        "summary": summary,
        "per_query": pd.read_csv(folder / "per_query.csv", index_col="query_id", dtype={"query_id": str}),
        "curves": pd.read_csv(folder / "metrics_at_k.csv", index_col="k"),
        "run": read_trec_run(folder / "run.trec"),
    }


def curves_from(ranked: dict[str, list[str]], qrels) -> pd.DataFrame:
    metrics = ["ndcg", "recall", "precision", "mrr"]
    pq = evaluate_run(ranked, qrels, CURVE_KS, metrics=metrics)
    return pd.DataFrame({m: [pq[f"{m}@{k}"].mean() for k in CURVE_KS] for m in metrics},
                        index=pd.Index(CURVE_KS, name="k"))


def rerank_with_maxsim(candidate_run: dict[str, list[str]], qrels, depth: int) -> tuple[dict, float]:
    """Re-score each query's top-`depth` SMVE docs with exact MaxSim and re-sort.

    Returns the new ranking and the measured MaxSim time (reading the
    candidates' token vectors + scoring), summed over all queries.
    """
    query_emb, all_qids = load_separate(EMB_DIR, "queries")
    doc_emb, doc_ids = load_separate(EMB_DIR, "docs", mmap_colbert=True)
    q_row = {q: i for i, q in enumerate(all_qids)}
    d_row = {d: i for i, d in enumerate(doc_ids)}
    qo, do = query_emb["colbert_offsets"], doc_emb["colbert_offsets"]

    reranked, total = {}, 0.0
    for qid, ranked in candidate_run.items():
        cands = ranked[:depth]
        i = q_row[qid]
        with Timer() as t:
            c_flat, c_off = subset_ragged(doc_emb["colbert_flat"], do, [d_row[d] for d in cands])
            s = maxsim_scores(query_emb["colbert_flat"][qo[i]:qo[i + 1]], np.array([0, qo[i + 1] - qo[i]]),
                              c_flat, c_off, show_progress=False)[0]
        total += t.seconds
        reranked[qid] = [cands[j] for j in np.argsort(-s, kind="stable")]
    return reranked, total


def paired_stats(base: pd.Series, cand: pd.Series, n_boot: int, seed: int = 0) -> dict:
    """How does a candidate differ from the baseline, query by query?

    - wins / ties / losses: count of queries where the candidate is better / equal / worse
    - mean delta + 95% bootstrap CI: resample queries with replacement, take the
      mean delta each time; the middle 95% of those means is the interval
    - Wilcoxon signed-rank p-value: is the median paired difference zero?
    """
    d = (cand - base.loc[cand.index]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    nonzero = d[d != 0]
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
        "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
    }


def fidelity(base_run: dict[str, list[str]], cand_run: dict[str, list[str]]) -> dict:
    """How closely does the candidate reproduce MaxSim's own ranking (ignoring qrels)?"""
    overlap10, cover10, top1_rank = [], [], []
    for qid, b in base_run.items():
        c = cand_run[qid]
        overlap10.append(len(set(b[:10]) & set(c[:10])) / 10)
        cover10.append(len(set(b[:10]) & set(c)) / 10)
        top1_rank.append(c.index(b[0]) + 1 if b[0] in c else np.nan)
    return {
        "top10_overlap": float(np.mean(overlap10)),
        "maxsim_top10_in_candidate_top100": float(np.mean(cover10)),
        "median_rank_of_maxsim_top1": float(np.nanmedian(top1_rank)),
        "maxsim_top1_missing_from_top100": int(np.isnan(top1_rank).sum()),
    }


def md_table(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    cols = ["", *df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [v if isinstance(v, str) else fmt.format(v) for v in row]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out = args.out_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    qrels = load_qrels(json.loads((args.baseline / "summary.json").read_text()).get("split", "test"))

    runs: dict[str, dict] = {}
    for folder in [args.baseline, *args.candidates]:
        r = load_run(folder)
        runs[label_for(r["summary"])] = r
    base_label = next(iter(runs))
    base = runs[base_label]
    n_q = base["summary"]["n_queries"]

    # Hybrid: best SMVE candidate as first stage, MaxSim over its top-100.
    if not args.no_rerank:
        pool = [l for l in list(runs)[1:] if args.rerank_from in l]
        best = max(pool, key=lambda l: runs[l]["summary"]["metrics"]["ndcg@10"])
        print(f"reranking the top {args.rerank_depth} of '{best}' with MaxSim...")
        reranked, rerank_secs = rerank_with_maxsim(runs[best]["run"], qrels, args.rerank_depth)
        pq = evaluate_run(reranked, qrels, [1, 3, 5, 10, 20, 50, 100])
        t = runs[best]["summary"]["timing"]
        hybrid = {
            "method": "hybrid",
            "timing": {
                "index_build_seconds": t["index_build_seconds"],
                "online_seconds": round(t["online_seconds"] + rerank_secs, 3),
                "rerank_seconds": round(rerank_secs, 3),
                # One query: SMVE single-query latency + that query's share of reranking.
                "single_query_latency_ms": round(t["single_query_latency_ms"] + 1000 * rerank_secs / n_q, 2),
            },
            # Needs both the SMVE index and the token vectors for reranking.
            "index": {"bytes": runs[best]["summary"]["index"]["bytes"] + base["summary"]["index"]["bytes"]},
            "metrics": pq[[c for c in pq.columns if "@" in c]].mean().to_dict(),
        }
        runs[f"{best} + MaxSim rerank@{args.rerank_depth}"] = {
            "summary": hybrid, "per_query": pq, "curves": curves_from(reranked, qrels), "run": reranked,
        }

    labels = list(runs)
    cands = labels[1:]

    # --- Quality -----------------------------------------------------------
    quality = pd.DataFrame({l: [runs[l]["summary"]["metrics"][m] for m in QUALITY_ROWS] for l in labels},
                           index=QUALITY_ROWS)
    rel = quality[cands].div(quality[base_label], axis=0).sub(1).mul(100)
    quality_md = quality.map(lambda v: f"{v:.4f}").copy()
    for c in cands:
        quality_md[c] = [f"{v:.4f} ({r:+.1f}%)" for v, r in zip(quality[c], rel[c])]

    # --- Paired per-query statistics ---------------------------------------
    stats = {c: {m: paired_stats(base["per_query"][m], runs[c]["per_query"][m], args.bootstrap)
                 for m in ("ndcg@10", "recall@100")} for c in cands}

    # --- Fidelity to MaxSim's ranking --------------------------------------
    fid = {c: fidelity(base["run"], runs[c]["run"]) for c in cands}

    # --- Cost ---------------------------------------------------------------
    cost_rows = {}
    for l in labels:
        t, idx = runs[l]["summary"]["timing"], runs[l]["summary"]["index"]
        cost_rows[l] = {
            "index build (offline)": t["index_build_seconds"],
            f"online, {n_q} queries": t["online_seconds"],
            "online per query (batched)": 1000 * t["online_seconds"] / n_q,
            "single-query latency": t["single_query_latency_ms"],
            "index size": idx["bytes"],
        }
    cost = pd.DataFrame(cost_rows).T
    b = cost.loc[base_label]
    speed = {c: {
        "online_speedup": b[f"online, {n_q} queries"] / cost.loc[c, f"online, {n_q} queries"],
        "latency_speedup": b["single-query latency"] / cost.loc[c, "single-query latency"],
        "index_size_ratio": b["index size"] / cost.loc[c, "index size"],
        # After how many queries does the one-off index build pay for itself?
        "break_even_queries_batched": cost.loc[c, "index build (offline)"]
            / max(1e-9, (b["online per query (batched)"] - cost.loc[c, "online per query (batched)"]) / 1000),
    } for c in cands}

    # --- Seed variance (from all SMVE runs on disk) -------------------------
    seeds_md = ""
    runs_csv = BASE / "smve" / "runs.csv"
    sweep = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame()
    if not sweep.empty:
        g = sweep.groupby(["w", "k", "reps", "center"])
        var = g["ndcg@10"].agg(["count", "mean", "std"]).query("count > 1")
        if not var.empty:
            var = var.assign(recall100_mean=g["recall@100"].mean(), recall100_std=g["recall@100"].std())
            var["count"] = var["count"].astype(str)
            seeds_md = md_table(var.reset_index().set_index(["w", "k", "reps", "center"]).rename(
                columns={"count": "seeds", "mean": "nDCG@10 mean", "std": "nDCG@10 std",
                         "recall100_mean": "R@100 mean", "recall100_std": "R@100 std"}), "{:.4f}")

    # --- Write --------------------------------------------------------------
    def fmt_cost(col: str, v: float) -> str:
        if col == "index size":
            return f"{v / 1e9:.2f} GB" if v >= 1e9 else f"{v / 1e6:.1f} MB"
        if col.startswith(("online per", "single")):
            return f"{v:.1f} ms"
        return f"{v:.2f} s"
    cost_md = cost.copy().astype(object)
    for col in cost.columns:
        cost_md[col] = [fmt_cost(col, v) for v in cost[col]]

    stats_df = pd.DataFrame({
        c: {
            "ΔnDCG@10 [95% CI]": f"{s['ndcg@10']['mean_delta']:+.4f} "
                                 f"[{s['ndcg@10']['ci95'][0]:+.4f}, {s['ndcg@10']['ci95'][1]:+.4f}]",
            "nDCG@10 better/tied/worse": f"{s['ndcg@10']['wins']} / {s['ndcg@10']['ties']} / {s['ndcg@10']['losses']}",
            "nDCG@10 Wilcoxon p": f"{s['ndcg@10']['wilcoxon_p']:.2g}",
            "ΔRecall@100 [95% CI]": f"{s['recall@100']['mean_delta']:+.4f} "
                                    f"[{s['recall@100']['ci95'][0]:+.4f}, {s['recall@100']['ci95'][1]:+.4f}]",
            "top-10 overlap with MaxSim": f"{fid[c]['top10_overlap']:.3f}",
            "MaxSim top-10 found in top-100": f"{fid[c]['maxsim_top10_in_candidate_top100']:.3f}",
            "median rank of MaxSim's #1": f"{fid[c]['median_rank_of_maxsim_top1']:.0f}",
            "online speed-up": f"{speed[c]['online_speedup']:.1f}×",
            "single-query speed-up": f"{speed[c]['latency_speedup']:.1f}×",
            "index smaller by": f"{speed[c]['index_size_ratio']:.1f}×",
            "break-even (queries)": f"{speed[c]['break_even_queries_batched']:,.0f}",
        } for c, s in stats.items()
    })

    md = [
        f"# Retrieval methods vs. {base_label} on SciFact ({n_q} test queries)\n",
        "## Quality (mean over queries; relative change vs. baseline in brackets)\n", md_table(quality_md),
        "\n## Cost\n",
        "Index build = one-off offline work per corpus. Online = encoding the queries + scoring them. "
        "Single-query latency = median time for one query on its own. All on CPU.\n",
        md_table(cost_md),
        f"\n## Paired comparison against {base_label}\n", md_table(stats_df),
    ]
    if seeds_md:
        md += ["\n## Seed variance (different random anchors)\n", seeds_md]
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    (out / "comparison.json").write_text(json.dumps({
        "baseline": base_label, "quality": quality.to_dict(), "cost": cost.to_dict(orient="index"),
        "paired": stats, "fidelity": fid, "speed": speed,
    }, indent=2, default=float))
    pd.DataFrame({f"{l} {m}": runs[l]["per_query"][m] for l in labels for m in ("ndcg@10", "recall@100")}
                 ).to_csv(out / "per_query.csv")

    # --- Plots --------------------------------------------------------------
    sub = f"SciFact, {n_q} queries"
    plots = out / "plots"
    bars = quality.loc[["ndcg@10", "recall@10", "mrr@10", "ndcg@100", "recall@100"]]
    bars.index = ["nDCG@10", "Recall@10", "MRR@10", "nDCG@100", "Recall@100"]
    plot_metric_bars(bars, plots / "quality_bars.png", sub)
    plot_curves_compare({l: runs[l]["curves"] for l in labels}, ["ndcg", "recall"], plots / "curves.png", sub)
    plot_per_query_delta({c: runs[c]["per_query"]["ndcg@10"] - base["per_query"]["ndcg@10"] for c in cands},
                         plots / "per_query_delta.png", "nDCG@10", base_label)
    cost_plot = cost.rename(columns={
        "index build (offline)": "Index build, offline (one-off)|s",
        f"online, {n_q} queries": f"Online: encode + score all {n_q} queries|s",
        "single-query latency": "Single-query latency|ms",
        "index size": "Index size|bytes",
    }).drop(columns=["online per query (batched)"])
    plot_cost(cost_plot, plots / "cost.png", sub)
    if not sweep.empty:
        single = sweep[sweep["seed"] == 0]
        bt = base["summary"]
        plot_tradeoff(single, {"ndcg@10": bt["metrics"]["ndcg@10"],
                               "single_query_latency_ms": bt["timing"]["single_query_latency_ms"],
                               "index_bytes": bt["index"]["bytes"]},
                      plots / "tradeoff.png", sub)

    print("\n".join(md))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
