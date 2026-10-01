"""Single-vector BGE-M3 baselines: dense, lexical (sparse), and dense+lexical hybrid.

BGE-M3 produces three representations per text in ONE forward pass; we
already stored all three (see encode_scifact.py). MaxSim uses the ColBERT
token vectors. This script evaluates the other two, which are far cheaper:

  - dense:    one 1024-d unit vector per text.   score = q . d  (cosine)
  - lexical:  one sparse vector over the ~250k-token vocabulary, holding a
              learned weight per token that appears in the text.
              score = sum over shared tokens of q_weight * d_weight
              (same as FlagEmbedding's compute_lexical_matching_score)
  - hybrid:   dense + alpha * lexical, alpha = 0.3 as in the BGE-M3 paper.

These are the "does SMVE beat the simple options?" baselines: if a single
4 KB dense vector scores higher than SMVE, SMVE's case rests on reranking.

    uv run python scripts/evaluate_bgem3_single_vector.py

Outputs (same layout as every other run):
    results/scifact_bgem3/dense/
    results/scifact_bgem3/lexical/
    results/scifact_bgem3/hybrid_dense_lexical/
    results/scifact_bgem3/hybrid_dense_lexical/alpha_sensitivity.{csv,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from smve_lab.evaluation import Timer, evaluate_and_save, median_latency_ms
from smve_lab.maxsim import top_k
from smve_lab.metrics import evaluate_run
from smve_lab.plots import plot_alpha_sensitivity
from smve_lab.datasets import DATASETS, emb_dir, info, load_qrels, results_dir
from smve_lab.storage import load_separate

VOCAB_SIZE = 250_002  # XLM-RoBERTa vocabulary used by BGE-M3
ALPHA_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="scifact", choices=list(DATASETS))
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--alpha", type=float, default=0.3, help="lexical weight in the hybrid")
    p.add_argument("--depth", type=int, default=100)
    p.add_argument("--latency-queries", type=int, default=50)
    return p.parse_args()


def sparse_matrix(emb: dict[str, np.ndarray]) -> sp.csr_matrix:
    """The stored CSR-style arrays (indptr/ids/weights) are exactly a CSR matrix."""
    n = len(emb["sparse_indptr"]) - 1
    m = sp.csr_matrix((emb["sparse_weights"], emb["sparse_ids"], emb["sparse_indptr"]),
                      shape=(n, VOCAB_SIZE), dtype=np.float32)
    m.sum_duplicates()
    return m


def csr_nbytes(m: sp.csr_matrix) -> int:
    return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes


def main() -> None:
    args = parse_args()
    EMB_DIR, OUT, ds = emb_dir(args.dataset), results_dir(args.dataset), info(args.dataset)
    qrels = load_qrels(args.dataset, args.split)
    q_emb, all_qids = load_separate(EMB_DIR, "queries")
    d_emb, doc_ids = load_separate(EMB_DIR, "docs")

    row = {q: i for i, q in enumerate(all_qids)}
    query_ids = list(qrels)
    rows = np.array([row[q] for q in query_ids])

    Qd, Dd = q_emb["dense"][rows], d_emb["dense"]
    Qs, Ds = sparse_matrix(q_emb)[rows], sparse_matrix(d_emb)
    n = len(query_ids)

    # --- Score every query against every doc, timing each method ----------
    with Timer() as t_dense:
        dense = Qd @ Dd.T
    with Timer() as t_lex:
        lexical = np.asarray((Qs @ Ds.T).todense(), dtype=np.float32)
    with Timer() as t_hyb:
        hybrid = (Qd @ Dd.T) + args.alpha * np.asarray((Qs @ Ds.T).todense(), dtype=np.float32)

    lat = {}
    if args.latency_queries:
        lat["dense"] = median_latency_ms(lambda i: Qd[i:i + 1] @ Dd.T, n, n=args.latency_queries)
        lat["lexical"] = median_latency_ms(lambda i: (Qs[i] @ Ds.T).toarray(), n, n=args.latency_queries)
        lat["hybrid"] = median_latency_ms(
            lambda i: Qd[i:i + 1] @ Dd.T + args.alpha * (Qs[i] @ Ds.T).toarray(), n, n=args.latency_queries)

    runs = {
        "dense": (dense, t_dense.seconds, Dd.nbytes, "BGE-M3 dense",
                  "1024-d fp32 dense vector per doc"),
        "lexical": (lexical, t_lex.seconds, csr_nbytes(Ds), "BGE-M3 lexical",
                    "sparse token-weight vector per doc (CSR)"),
        "hybrid_dense_lexical": (hybrid, t_hyb.seconds, Dd.nbytes + csr_nbytes(Ds),
                                 f"BGE-M3 dense + {args.alpha:g}·lexical",
                                 "dense vectors + lexical CSR"),
    }
    lat_key = {"dense": "dense", "lexical": "lexical", "hybrid_dense_lexical": "hybrid"}
    for name, (scores, secs, nbytes, label, desc) in runs.items():
        evaluate_and_save(
            scores, query_ids, doc_ids, qrels, OUT / name,
            run_id=f"bgem3_{name}",
            run_title=f"{label} · {ds.display}",
            ignore_identical_ids=ds.ignore_identical_ids,
            depth=args.depth,
            summary_extra={
                "method": name,
                "label": label,
                "split": args.split,
                "dataset": args.dataset,
                "params": {"alpha": args.alpha} if name.startswith("hybrid") else {},
                "timing": {
                    # Vectors come out of the same BGE-M3 forward pass as the
                    # ColBERT vectors, so there is no extra index-build step.
                    "index_build_seconds": 0.0,
                    "query_encode_seconds": 0.0,
                    "scoring_seconds": round(secs, 4),
                    "online_seconds": round(secs, 4),
                    "single_query_latency_ms": round(lat.get(lat_key[name], float("nan")), 3),
                },
                "index": {"description": desc, "bytes": int(nbytes)},
            },
        )

    # --- How sensitive is the hybrid to alpha? -------------------------------
    # Sensitivity ANALYSIS on the test set, not tuning: the reported hybrid
    # keeps the paper's alpha=0.3 so it isn't fitted to these queries.
    dense_m = dense.copy()
    if ds.ignore_identical_ids:  # same self-match filter as evaluate_and_save
        col = {d: j for j, d in enumerate(doc_ids)}
        for i, q in enumerate(query_ids):
            if q in col:
                dense_m[i, col[q]] = -np.inf
    rows_out = []
    for a in ALPHA_GRID:
        run = top_k(dense_m + a * lexical, query_ids, doc_ids, args.depth)
        pq = evaluate_run({q: [d for d, _ in r] for q, r in run.items()}, qrels, [10, 100],
                          metrics=["ndcg", "recall", "mrr"])
        rows_out.append({"alpha": a, **pq[[c for c in pq.columns if "@" in c]].mean().to_dict()})
    sens = pd.DataFrame(rows_out)
    sens.to_csv(OUT / "hybrid_dense_lexical" / "alpha_sensitivity.csv", index=False)
    plot_alpha_sensitivity(sens, args.alpha, OUT / "hybrid_dense_lexical" / "plots" / "alpha_sensitivity.png",
                           f"BGE-M3 dense + α·lexical · {ds.display}")
    print("\nalpha sensitivity (analysis only):")
    print(sens.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
