"""Evaluate MUVERA (Fixed Dimensional Encodings) on SciFact, same metrics as everything else.

Each setting -> results/scifact_bgem3/muvera/r{R}_k{K}_p{P}[_center]_seed{S}/ ; all settings on
disk are collected into results/scifact_bgem3/muvera/runs.csv.

    # the paper's main setting (R=20, k_sim=5, d_proj=16 -> 10,240 dims)
    uv run python scripts/evaluate_muvera.py

    # a grid
    uv run python scripts/evaluate_muvera.py --reps 10 20 40 --k-sim 4 5 6 --d-proj 8 16 --light

Timing is split like SMVE: index_build = encode all documents once (offline);
online = encode the queries + dense dot products against the index.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from smve_lab.config import ARTIFACTS_DIR, RESULTS_DIR
from smve_lab.evaluation import Timer, evaluate_and_save, median_latency_ms
from smve_lab.maxsim import subset_ragged
from smve_lab.muvera import MuveraFDE
from smve_lab.scifact import load_qrels
from smve_lab.smve import token_mean
from smve_lab.storage import load_separate

EMB_DIR = ARTIFACTS_DIR / "embeddings" / "scifact_bgem3"
OUT = RESULTS_DIR / "scifact_bgem3" / "muvera"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, nargs="+", default=[20], help="R: independent repetitions")
    p.add_argument("--k-sim", type=int, nargs="+", default=[5], help="SimHash bits -> 2^k_sim buckets")
    p.add_argument("--d-proj", type=int, nargs="+", default=[16], help="projected size of each bucket block")
    p.add_argument("--center", nargs="+", default=["off"], choices=["off", "on"])
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--latency-queries", type=int, default=20)
    p.add_argument("--light", action="store_true", help="skip plots and scores.npz")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def collect_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(OUT.glob("*/summary.json")):
        s = json.loads(f.read_text())
        rows.append({"run": f.parent.name, **s["params"], **s["timing"], "index_bytes": s["index"]["bytes"],
                     "dims": s["index"]["dims"], **s["metrics"]})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "runs.csv", index=False)
    return df


def main() -> None:
    args = parse_args()
    qrels = load_qrels("test")
    q_emb, all_qids = load_separate(EMB_DIR, "queries")
    d_emb, doc_ids = load_separate(EMB_DIR, "docs", mmap_colbert=True)
    d_flat, d_off = d_emb["colbert_flat"], d_emb["colbert_offsets"]
    row = {q: i for i, q in enumerate(all_qids)}
    query_ids = list(qrels)
    q_flat, q_off = subset_ragged(q_emb["colbert_flat"], q_emb["colbert_offsets"], [row[q] for q in query_ids])
    dim = d_flat.shape[1]
    mu = None

    for R, k_sim, d_proj, center, seed in itertools.product(args.reps, args.k_sim, args.d_proj, args.center,
                                                             args.seed):
        center = center == "on"
        name = f"r{R}_k{k_sim}_p{d_proj}{'_center' if center else ''}_seed{seed}"
        if args.skip_existing and (OUT / name / "summary.json").exists():
            print(f"skip {name}")
            continue
        print(f"\n=== MUVERA {name} ===")
        if center and mu is None:
            mu = token_mean(d_flat)
        c = mu if center else None
        fde = MuveraFDE(dim, reps=R, k_sim=k_sim, d_proj=d_proj, seed=seed)

        with Timer() as t_docs:
            D = fde.encode(d_flat, d_off, is_query=False, center=c)
        with Timer() as t_q:
            Q = fde.encode(q_flat, q_off, is_query=True, center=c, show_progress=False)
        with Timer() as t_s:
            scores = Q @ D.T

        def one_query(i: int):
            q = fde.encode(q_flat[q_off[i]:q_off[i + 1]], q_off[i:i + 2] - q_off[i], is_query=True,
                           center=c, show_progress=False)
            return q @ D.T
        latency = median_latency_ms(one_query, len(query_ids), n=args.latency_queries)
        print(f"{fde.out_dim:,} dims · docs {t_docs.seconds:.1f}s · single query {latency:.1f} ms")

        evaluate_and_save(
            scores, query_ids, doc_ids, qrels, OUT / name,
            run_id=f"muvera_{name}",
            run_title=f"MUVERA R={R} k_sim={k_sim} d_proj={d_proj}{' centered' if center else ''} · SciFact",
            score_norm=np.diff(q_off), save_scores=not args.light, make_plots=not args.light,
            summary_extra={
                "method": "muvera", "split": "test",
                "label": f"MUVERA R={R} k={k_sim} p={d_proj}" + (" centered" if center else ""),
                "params": {"reps": R, "k_sim": k_sim, "d_proj": d_proj, "center": center, "seed": seed},
                "timing": {"index_build_seconds": round(t_docs.seconds, 3),
                           "query_encode_seconds": round(t_q.seconds, 4),
                           "scoring_seconds": round(t_s.seconds, 4),
                           "online_seconds": round(t_q.seconds + t_s.seconds, 4),
                           "single_query_latency_ms": round(latency, 3)},
                "index": {"description": "MUVERA FDEs, dense float32", "bytes": int(D.nbytes), "dims": fde.out_dim},
            },
        )
    runs = collect_runs()
    print(f"\n{len(runs)} MUVERA runs on disk -> {OUT / 'runs.csv'}")


if __name__ == "__main__":
    main()
