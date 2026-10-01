"""BM25 baseline on SciFact (hand-written, see smve_lab/bm25.py).

    uv run python scripts/evaluate_bm25.py

Writes results/scifact_bgem3/bm25/ in the standard layout.
"""

from __future__ import annotations

import argparse

from smve_lab.bm25 import BM25
from smve_lab.evaluation import Timer, evaluate_and_save, median_latency_ms
from smve_lab.datasets import DATASETS, info, load_qrels, load_texts, results_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact", choices=list(DATASETS))
    args = p.parse_args()
    ds = info(args.dataset)
    qrels = load_qrels(args.dataset)
    doc_ids, doc_texts, all_qids, q_texts = load_texts(args.dataset)
    q_text = dict(zip(all_qids, q_texts))
    query_ids = list(qrels)
    queries = [q_text[q] for q in query_ids]

    with Timer() as t_index:
        bm = BM25().fit(doc_texts)
    with Timer() as t_q:
        Q = bm.encode_queries(queries)
    with Timer() as t_s:
        scores = bm.scores(Q)
    latency = median_latency_ms(lambda i: bm.scores(bm.encode_queries([queries[i]])), len(queries), n=50)
    print(f"index {t_index.seconds:.1f}s · vocab {len(bm.vocab):,} · single query {latency:.2f} ms")

    evaluate_and_save(
        scores, query_ids, doc_ids, qrels, results_dir(args.dataset) / "bm25",
        run_id="bm25", run_title=f"BM25 (k1=1.2, b=0.75) · {ds.display}",
        ignore_identical_ids=ds.ignore_identical_ids,
        summary_extra={
            "method": "bm25", "label": "BM25", "split": "test", "dataset": args.dataset,
            "params": {"k1": bm.k1, "b": bm.b},
            "timing": {"index_build_seconds": round(t_index.seconds, 3),
                       "query_encode_seconds": round(t_q.seconds, 4),
                       "scoring_seconds": round(t_s.seconds, 4),
                       "online_seconds": round(t_q.seconds + t_s.seconds, 4),
                       "single_query_latency_ms": round(latency, 3)},
            "index": {"description": "BM25 weights per (doc, term), CSR", "bytes": bm.index_bytes(),
                      "nnz_per_doc": round(bm.doc_weights.nnz / len(doc_ids), 1),
                      "nnz_per_query": round(Q.nnz / len(query_ids), 1), "vocab": len(bm.vocab)},
        },
    )


if __name__ == "__main__":
    main()
