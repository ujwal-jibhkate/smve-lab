"""Stage 4a, part 1: build the per-query routing dataset.

The router makes ONE decision per query:

    base pipeline (always):  BGE-M3 dense + 0.3*lexical  ->  MaxSim rerank of top 10   (~8 ms)
    escalate?                cross-encoder rerank of the first stage's top 20            (+~1.6 s)

For every query we record
  - what each option would score: nDCG@10 for  first stage only / base / escalated
  - the label y = 1 if escalating strictly improves nDCG@10 over the base
  - features the router may use. They are computed ONLY from the first stage
    (scores and rankings that already exist at decision time), so routing adds
    no extra model calls: top scores, score gaps, dense-vs-lexical agreement,
    query length.

Queries: SciFact test (300) and train (809), NFCorpus test (323), ArguAna test (1406).
Cross-encoder scores are cached per dataset and resumable; SciFact test reuses the
stage-2 cache. Latency of each option is measured per dataset (median over queries).

    uv run python scripts/build_router_data.py

Writes results/router/{data.csv, latency.json, ce_cache_<dataset>.csv}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from smve_lab.config import RESULTS_DIR
from smve_lab.datasets import emb_dir, info, load_qrels, load_texts, results_dir
from smve_lab.evaluation import Timer
from smve_lab.maxsim import maxsim_scores, subset_ragged
from smve_lab.metrics import ndcg_at_k
from smve_lab.reranker import CrossEncoder
from smve_lab.storage import load_separate

OUT = RESULTS_DIR / "router"
SPLITS = [("scifact", "test"), ("scifact", "train"), ("nfcorpus", "test"), ("arguana", "test")]
ALPHA = 0.3        # lexical weight in the first stage (BGE-M3 paper)
POOL = 100         # first-stage candidates kept
MS_DEPTH = 10      # base: MaxSim rerank depth
CE_DEPTH = 20      # escalation: cross-encoder rerank depth
VOCAB = 250_002
LATENCY_QUERIES = 20


def sparse_rows(emb: dict, rows: np.ndarray) -> sp.csr_matrix:
    m = sp.csr_matrix((emb["sparse_weights"], emb["sparse_ids"], emb["sparse_indptr"]),
                      shape=(len(emb["sparse_indptr"]) - 1, VOCAB), dtype=np.float32)
    m.sum_duplicates()
    return m[rows]


def first_stage_features(dense_row: np.ndarray, lex_row: np.ndarray, hyb_row: np.ndarray) -> dict:
    """Signals that exist before any reranking: how confident and how consistent the first stage is."""
    top_h = np.argsort(-hyb_row)[:POOL]
    s = hyb_row[top_h]
    top_d = np.argsort(-dense_row)[:10]
    top_l = np.argsort(-lex_row)[:10]
    d_sorted = np.sort(dense_row)[::-1]
    l_sorted = np.sort(lex_row)[::-1]
    return {
        "hyb_top1": s[0], "hyb_margin12": s[0] - s[1], "hyb_gap1_10": s[0] - s[9], "hyb_std10": s[:10].std(),
        "dense_top1": d_sorted[0], "dense_margin12": d_sorted[0] - d_sorted[1], "dense_gap1_10": d_sorted[0] - d_sorted[9],
        "lex_top1": l_sorted[0], "lex_margin12": l_sorted[0] - l_sorted[1], "lex_gap1_10": l_sorted[0] - l_sorted[9],
        "dense_lex_jaccard10": len(set(top_d) & set(top_l)) / len(set(top_d) | set(top_l)),
        "dense_top1_in_lex10": float(top_d[0] in set(top_l)),
        "lex_top1_in_dense10": float(top_l[0] in set(top_d)),
        "hyb_top1_is_dense_top1": float(top_h[0] == top_d[0]),
    }, top_h


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ce = CrossEncoder(device="mps")
    rows, latency = [], {}

    for name, split in SPLITS:
        ds = info(name)
        print(f"\n=== {ds.display} {split} ===")
        qrels = load_qrels(name, split)
        doc_ids, doc_texts, all_qids, q_texts = load_texts(name)
        d_text, q_text = dict(zip(doc_ids, doc_texts)), dict(zip(all_qids, q_texts))
        q_emb, emb_qids = load_separate(emb_dir(name), "queries")
        d_emb, emb_dids = load_separate(emb_dir(name), "docs", mmap_colbert=True)
        assert emb_dids == doc_ids
        qrow = {q: i for i, q in enumerate(emb_qids)}
        query_ids = [q for q in qrels if q in qrow]
        rows_idx = np.array([qrow[q] for q in query_ids])
        dcol = {d: j for j, d in enumerate(doc_ids)}

        # First stage: dense + 0.3 * lexical over the whole corpus.
        dense = q_emb["dense"][rows_idx] @ d_emb["dense"].T
        lex = np.asarray((sparse_rows(q_emb, rows_idx) @ sparse_rows(d_emb, np.arange(len(doc_ids))).T).todense(),
                         dtype=np.float32)
        hyb = dense + ALPHA * lex
        if ds.ignore_identical_ids:
            for i, q in enumerate(query_ids):
                if q in dcol:
                    hyb[i, dcol[q]] = dense[i, dcol[q]] = lex[i, dcol[q]] = -np.inf

        q_flat, q_off = q_emb["colbert_flat"], q_emb["colbert_offsets"]
        d_flat, d_off = d_emb["colbert_flat"], d_emb["colbert_offsets"]

        def maxsim_rerank(i: int, cands: list[int]) -> np.ndarray:
            r = rows_idx[i]
            c_flat, c_off = subset_ragged(d_flat, d_off, cands)
            return maxsim_scores(q_flat[q_off[r]:q_off[r + 1]], np.array([0, q_off[r + 1] - q_off[r]]),
                                 c_flat, c_off, show_progress=False)[0]

        # Cross-encoder scores for every (query, top-20 doc) pair, cached.
        cache_path = OUT / f"ce_cache_{name}.csv"
        cache = {}
        for f in [cache_path] + ([results_dir(name) / "rerank" / "ce_scores.csv"] if name == "scifact" else []):
            if Path(f).exists():
                c = pd.read_csv(f, dtype={"query_id": str, "doc_id": str})
                cache.update({(a, b): s for a, b, s in zip(c.query_id, c.doc_id, c.ce_score)})

        feats, tops = [], []
        for i in range(len(query_ids)):
            f, top = first_stage_features(dense[i], lex[i], hyb[i])
            feats.append(f)
            tops.append(top)
        todo = [(q, doc_ids[j]) for q, top in zip(query_ids, tops) for j in top[:CE_DEPTH]
                if (q, doc_ids[j]) not in cache]
        print(f"cross-encoder pairs to score: {len(todo):,} (cached {len(query_ids) * CE_DEPTH - len(todo):,})")
        for s in tqdm(range(0, len(todo), 1024), desc="cross-encoder", unit="chunk"):
            part = todo[s:s + 1024]
            sc = ce.score([q_text[q] for q, _ in part], [d_text[d] for _, d in part])
            pd.DataFrame({"query_id": [q for q, _ in part], "doc_id": [d for _, d in part], "ce_score": sc}) \
                .to_csv(cache_path, mode="a", header=not cache_path.exists(), index=False)
            cache.update({k: v for k, v in zip(part, sc)})

        # Per-query outcomes of each option.
        for i, q in enumerate(tqdm(query_ids, desc="labels", unit="q")):
            top = list(tops[i])
            first = [doc_ids[j] for j in top]
            ms = maxsim_rerank(i, top[:MS_DEPTH])
            base = [first[j] for j in np.argsort(-ms, kind="stable")] + first[MS_DEPTH:]
            ce_s = np.array([cache[(q, d)] for d in first[:CE_DEPTH]])
            esc = [first[j] for j in np.argsort(-ce_s, kind="stable")] + first[CE_DEPTH:]
            n_first, n_base, n_esc = (ndcg_at_k(r, qrels[q], 10) for r in (first, base, esc))
            r = rows_idx[i]
            rows.append({"dataset": name, "split": split, "query_id": q, **feats[i],
                         "q_tokens": int(q_off[r + 1] - q_off[r]),
                         "q_lex_terms": int(q_emb["sparse_indptr"][r + 1] - q_emb["sparse_indptr"][r]),
                         "ndcg_first": n_first, "ndcg_base": n_base, "ndcg_escalate": n_esc,
                         "gain": n_esc - n_base, "y": int(n_esc > n_base)})

        # Latency of each option, one query at a time (median), on this dataset.
        sample = np.linspace(0, len(query_ids) - 1, LATENCY_QUERIES).astype(int)
        t_ms, t_ce = [], []
        for i in [sample[0]] + list(sample):  # first is a warm-up
            top = list(tops[i])
            with Timer() as t:
                maxsim_rerank(i, top[:MS_DEPTH])
            t_ms.append(t.seconds * 1000)
            with Timer() as t:
                ce.score([q_text[query_ids[i]]] * CE_DEPTH, [d_text[doc_ids[j]] for j in top[:CE_DEPTH]])
            t_ce.append(t.seconds * 1000)
        first_ms = json.loads((results_dir(name) / "hybrid_dense_lexical" / "summary.json").read_text()) \
            ["timing"]["single_query_latency_ms"]
        latency[f"{name}/{split}"] = {"first_stage_ms": first_ms, "maxsim10_ms": float(np.median(t_ms[1:])),
                                      "ce20_ms": float(np.median(t_ce[1:]))}
        latency[f"{name}/{split}"]["base_ms"] = first_ms + latency[f"{name}/{split}"]["maxsim10_ms"]
        latency[f"{name}/{split}"]["escalate_extra_ms"] = latency[f"{name}/{split}"]["ce20_ms"]
        g = pd.DataFrame([r for r in rows if r["dataset"] == name and r["split"] == split])
        print(f"nDCG@10 first {g.ndcg_first.mean():.4f} · base {g.ndcg_base.mean():.4f} · "
              f"always escalate {g.ndcg_escalate.mean():.4f} · oracle {np.maximum(g.ndcg_base, g.ndcg_escalate).mean():.4f}"
              f" · escalation helps on {g.y.mean():.0%} · latency {latency[f'{name}/{split}']}")

    pd.DataFrame(rows).to_csv(OUT / "data.csv", index=False)
    (OUT / "latency.json").write_text(json.dumps(latency, indent=2))
    print(f"\n{len(rows)} queries -> {OUT / 'data.csv'}")


if __name__ == "__main__":
    main()
