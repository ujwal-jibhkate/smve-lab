"""Baseline: exhaustive BGE-M3 ColBERT MaxSim retrieval on scifact.

Scores every judged query against every document with MaxSim, ranks the
documents, and evaluates the ranking against the qrels with nDCG / Recall /
Precision / MRR / MAP at several cutoffs. This is the reference point that
SMVE experiments are compared against.

    uv run python scripts/evaluate_maxsim.py
    uv run python scripts/evaluate_maxsim.py --split train --chunk-tokens 16384

Outputs go to results/scifact_bgem3/colbert_maxsim/ by default; see
smve_lab/evaluation.py for the file layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smve_lab.evaluation import Timer, evaluate_and_save, median_latency_ms
from smve_lab.maxsim import maxsim_scores, subset_ragged
from smve_lab.datasets import DATASETS, emb_dir, info, load_qrels, results_dir
from smve_lab.storage import load_separate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="scifact", choices=list(DATASETS))
    p.add_argument("--split", default="test", choices=["test", "train"], help="qrels split (BEIR reports test)")
    p.add_argument("--depth", type=int, default=100, help="docs kept per query in the run")
    p.add_argument("--chunk-tokens", type=int, default=32_768, help="doc tokens per matmul; lower = less RAM")
    p.add_argument("--latency-queries", type=int, default=5, help="single-query latency samples (0 = skip)")
    p.add_argument("--out-dir", type=Path, default=None, help="default: results/{dataset}_bgem3/colbert_maxsim")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    EMB_DIR = emb_dir(args.dataset)
    ds = info(args.dataset)
    out_dir = args.out_dir or results_dir(args.dataset) / "colbert_maxsim"

    # 1. Ground truth: which docs are relevant for which query.
    qrels = load_qrels(args.dataset, args.split)
    print(f"qrels[{args.split}]: {len(qrels)} queries, {sum(len(v) for v in qrels.values())} judgments")

    # 2. Embeddings. Doc ColBERT vectors (~3.8 GB) stay memory-mapped on disk.
    query_emb, all_query_ids = load_separate(EMB_DIR, "queries")
    doc_emb, doc_ids = load_separate(EMB_DIR, "docs", mmap_colbert=True)
    d_flat, d_offsets = doc_emb["colbert_flat"], doc_emb["colbert_offsets"]

    # 3. Keep only queries that have judgments - unjudged ones can't be scored.
    qid_to_row = {qid: i for i, qid in enumerate(all_query_ids)}
    missing = [q for q in qrels if q not in qid_to_row]
    assert not missing, f"qrels queries with no embedding: {missing[:5]}"
    query_ids = list(qrels)
    q_flat, q_offsets = subset_ragged(
        query_emb["colbert_flat"], query_emb["colbert_offsets"], [qid_to_row[q] for q in query_ids]
    )
    print(f"scoring {len(query_ids)} queries ({len(q_flat):,} tokens) "
          f"x {len(doc_ids):,} docs ({len(d_flat):,} tokens)")

    # 4. MaxSim for every (query, doc) pair. There is no index to build: the
    #    "index" is the raw token vectors, so all the work happens at query time.
    with Timer() as t:
        scores = maxsim_scores(q_flat, q_offsets, d_flat, d_offsets, args.chunk_tokens)
    print(f"scored in {t.seconds:.1f}s")

    latency = None
    if args.latency_queries:
        def one_query(i: int):
            return maxsim_scores(q_flat[q_offsets[i]:q_offsets[i + 1]], q_offsets[i:i + 2] - q_offsets[i],
                                 d_flat, d_offsets, args.chunk_tokens, show_progress=False)
        latency = median_latency_ms(one_query, len(query_ids), n=args.latency_queries)
        print(f"single-query latency (median of {args.latency_queries}): {latency:.0f} ms")

    # 5. Rank, evaluate, store, plot.
    evaluate_and_save(
        scores, query_ids, doc_ids, qrels, out_dir,
        run_id="bgem3_colbert_maxsim",
        run_title=f"BGE-M3 ColBERT · exhaustive MaxSim · {ds.display}",
        ignore_identical_ids=ds.ignore_identical_ids,
        depth=args.depth,
        score_norm=np.diff(q_offsets),
        summary_extra={
            "method": "maxsim",
            "split": args.split,
            "dataset": args.dataset,
            "timing": {
                "index_build_seconds": 0.0,
                "query_encode_seconds": 0.0,
                "scoring_seconds": round(t.seconds, 3),
                "online_seconds": round(t.seconds, 3),
                "single_query_latency_ms": None if latency is None else round(latency, 2),
            },
            "index": {
                "description": "fp16 ColBERT token vectors + offsets",
                "bytes": int(d_flat.nbytes + d_offsets.nbytes),
                "n_vectors": int(len(d_flat)),
            },
        },
    )


if __name__ == "__main__":
    main()
