"""Evaluate SMVE retrieval on scifact, with the same metrics as the MaxSim baseline.

Each setting is encoded, scored, evaluated and saved to its own folder:

    results/scifact_bgem3/smve/w{W}_k{K}[_center]_seed{S}/

Every argument accepts several values; all combinations are run, and a table
of all SMVE runs found on disk is (re)written to results/scifact_bgem3/smve/runs.csv.

    # the notebook setting (w=65536, k=32)
    uv run python scripts/evaluate_smve.py

    # with mean-centering of token vectors
    uv run python scripts/evaluate_smve.py --center on

    # a sweep: 4 widths x 2 sparsities x centering off/on, light outputs
    uv run python scripts/evaluate_smve.py --w 1024 4096 16384 65536 --k 8 32 --center off on --light

Timing is split the way a search system would pay for it:
  - index_build: encode every document once, offline (+ estimating the mean
    token vector if centering)
  - online:      encode the queries + score them against the index
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smve_lab.config import ARTIFACTS_DIR, RESULTS_DIR
from smve_lab.evaluation import Timer, evaluate_and_save, median_latency_ms
from smve_lab.maxsim import subset_ragged
from smve_lab.scifact import load_qrels
from smve_lab.smve import csr_nbytes, make_anchors, smve_encode, smve_scores, token_mean
from smve_lab.storage import load_separate

EMB_DIR = ARTIFACTS_DIR / "embeddings" / "scifact_bgem3"
SMVE_DIR = RESULTS_DIR / "scifact_bgem3" / "smve"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--w", type=int, nargs="+", default=[65536], help="number of anchors (vector width)")
    p.add_argument("--k", type=int, nargs="+", default=[32], help="anchors kept per token")
    p.add_argument("--center", nargs="+", default=["off"], choices=["off", "on"],
                   help="subtract the mean token vector before projecting")
    p.add_argument("--seed", type=int, nargs="+", default=[0], help="anchor seed(s)")
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--depth", type=int, default=100)
    p.add_argument("--device", default="cpu", help="torch device for encoding (cpu is as fast as mps here)")
    p.add_argument("--latency-queries", type=int, default=20, help="single-query latency samples (0 = skip)")
    p.add_argument("--light", action="store_true", help="skip plots and scores.npz (for sweeps)")
    return p.parse_args()


def run_name(w: int, k: int, center: bool, seed: int) -> str:
    return f"w{w}_k{k}{'_center' if center else ''}_seed{seed}"


def collect_runs() -> pd.DataFrame:
    """One row per SMVE run on disk - the input for trade-off plots."""
    rows = []
    for f in sorted(SMVE_DIR.glob("*/summary.json")):
        s = json.loads(f.read_text())
        rows.append({"run": f.parent.name, **s["params"], **s["timing"],
                     "index_bytes": s["index"]["bytes"], **s["metrics"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(SMVE_DIR / "runs.csv", index=False)
    return df


def main() -> None:
    args = parse_args()

    qrels = load_qrels(args.split)
    query_emb, all_query_ids = load_separate(EMB_DIR, "queries")
    doc_emb, doc_ids = load_separate(EMB_DIR, "docs", mmap_colbert=True)
    d_flat, d_offsets = doc_emb["colbert_flat"], doc_emb["colbert_offsets"]

    qid_to_row = {qid: i for i, qid in enumerate(all_query_ids)}
    query_ids = list(qrels)
    q_flat, q_offsets = subset_ragged(
        query_emb["colbert_flat"], query_emb["colbert_offsets"], [qid_to_row[q] for q in query_ids]
    )
    dim = d_flat.shape[1]

    mu, mu_seconds = None, 0.0
    for w, k, center, seed in itertools.product(args.w, args.k, args.center, args.seed):
        center = center == "on"
        name = run_name(w, k, center, seed)
        print(f"\n=== SMVE {name} ===")

        # The mean token vector depends only on the corpus, so estimate it once.
        if center and mu is None:
            with Timer() as t:
                mu = token_mean(d_flat)
            mu_seconds = t.seconds
        c = mu if center else None
        B = make_anchors(dim, w, seed)

        # Offline: build the document index.
        with Timer() as t_docs:
            D = smve_encode(d_flat, d_offsets, B, k, is_query=False, device=args.device, center=c)
        # Online: encode queries, then score them against the index.
        with Timer() as t_q:
            Q = smve_encode(q_flat, q_offsets, B, k, is_query=True, device=args.device, center=c)
        with Timer() as t_s:
            scores = smve_scores(Q, D)
        print(f"docs {t_docs.seconds:.1f}s · queries {t_q.seconds:.2f}s · scoring {t_s.seconds:.3f}s")

        latency = None
        if args.latency_queries:
            def one_query(i: int):
                q = smve_encode(q_flat[q_offsets[i]:q_offsets[i + 1]], q_offsets[i:i + 2] - q_offsets[i],
                                B, k, is_query=True, device=args.device, show_progress=False, center=c)
                return smve_scores(q, D)
            latency = median_latency_ms(one_query, len(query_ids), n=args.latency_queries)
            print(f"single-query latency (median of {args.latency_queries}): {latency:.1f} ms")

        title = f"SMVE w={w} k={k}{' centered' if center else ''} · SciFact"
        evaluate_and_save(
            scores, query_ids, doc_ids, qrels, SMVE_DIR / name,
            run_id=f"smve_{name}",
            run_title=title,
            depth=args.depth,
            score_norm=np.diff(q_offsets),
            save_scores=not args.light,
            make_plots=not args.light,
            summary_extra={
                "method": "smve",
                "split": args.split,
                "params": {"w": w, "k": k, "center": center, "seed": seed},
                "timing": {
                    "index_build_seconds": round(t_docs.seconds + (mu_seconds if center else 0.0), 3),
                    "query_encode_seconds": round(t_q.seconds, 3),
                    "scoring_seconds": round(t_s.seconds, 4),
                    "online_seconds": round(t_q.seconds + t_s.seconds, 3),
                    "single_query_latency_ms": None if latency is None else round(latency, 2),
                },
                "index": {
                    "description": "SMVE doc vectors, scipy CSR (fp32 values, int32 indices)",
                    "bytes": csr_nbytes(D),
                    "nnz_per_doc": round(D.nnz / D.shape[0], 1),
                    "nnz_per_query": round(Q.nnz / Q.shape[0], 1),
                },
            },
        )

    runs = collect_runs()
    print(f"\n{len(runs)} SMVE runs on disk -> {SMVE_DIR / 'runs.csv'}")


if __name__ == "__main__":
    main()
