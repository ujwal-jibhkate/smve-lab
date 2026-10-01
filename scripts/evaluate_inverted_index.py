"""Stage 3b: real query latency with a hand-built inverted index.

Earlier SMVE latencies came from a general-purpose code path (build a scipy
sparse query, multiply by the whole doc matrix) that has a lot of per-call
overhead. Here every sparse method is served the way a search engine would:

    encode the query  ->  read the posting lists of its terms  ->  pick the top 100

and we time the three parts separately for every one of the 300 queries
(median and p95 reported), plus count how many posting entries each query
reads. Scores are checked to be identical to the earlier evaluation, so
quality numbers don't change; only the cost picture does.

Methods: SMVE at four representative settings, BM25, BGE-M3 lexical (inverted
index), and BGE-M3 dense / MUVERA (dense brute-force scan) for reference.
Query encoding = the method's own transform of the query; the BGE-M3 forward
pass that produced the token vectors is excluded for every method alike.

    uv run python scripts/evaluate_inverted_index.py

Writes results/scifact_bgem3/inverted_index/{summary.csv, per_query.csv, summary.md, plots/}.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from smve_lab.bm25 import BM25
from smve_lab.config import ARTIFACTS_DIR, RESULTS_DIR
from smve_lab.inverted_index import InvertedIndex, top_k
from smve_lab.maxsim import subset_ragged
from smve_lab.metrics import evaluate_run
from smve_lab.muvera import MuveraFDE
from smve_lab.plots import plot_index_latency
from smve_lab.scifact import load_qrels, load_scifact
from smve_lab.smve import make_anchors, smve_encode, smve_scores, token_mean, topk_blocks
from smve_lab.storage import load_separate

BASE = RESULTS_DIR / "scifact_bgem3"
OUT = BASE / "inverted_index"
EMB_DIR = ARTIFACTS_DIR / "embeddings" / "scifact_bgem3"
VOCAB_SIZE = 250_002
SMVE_SETTINGS = [  # (w, k, reps, center): small, mid, the notebook setting, and a repetitions frontier point
    (4096, 16, 1, False), (16384, 32, 1, True), (65536, 32, 1, True), (8192, 8, 4, False),
]
MUVERA_SETTING = (40, 4, 32, False)  # best quality under 10 ms in stage 3a
DEPTH = 100
WARMUP = 5


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t) * 1000


def run_queries(name: str, encode, score, n_q: int, postings=None) -> tuple[list, pd.DataFrame]:
    """Encode / score / top-k each query separately; returns rankings (doc indices) and per-query timings."""
    rows, ranked = [], []
    for i in list(range(WARMUP)) + list(range(n_q)):
        q, t_enc = timed(lambda: encode(i))
        s, t_score = timed(lambda: score(q))
        top, t_top = timed(lambda: top_k(s, DEPTH))
        rows.append({"method": name, "query": i, "encode_ms": t_enc, "score_ms": t_score, "topk_ms": t_top,
                     "postings_read": postings(q) if postings else np.nan,
                     "query_nnz": len(q[0]) if isinstance(q, tuple) else np.nan})
        ranked.append(top)
    rows, ranked = rows[WARMUP:], ranked[WARMUP:]  # drop warm-up repeats
    df = pd.DataFrame(rows)
    df["total_ms"] = df.encode_ms + df.score_ms + df.topk_ms
    return ranked, df


def main() -> None:
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(torch.get_num_threads())
    qrels = load_qrels("test")
    query_ids = list(qrels)
    n_q = len(query_ids)
    _, _, _, doc_ids, doc_texts, all_qids, q_texts = load_scifact()
    q_text = dict(zip(all_qids, q_texts))
    q_emb, emb_qids = load_separate(EMB_DIR, "queries")
    d_emb, _ = load_separate(EMB_DIR, "docs", mmap_colbert=True)
    row = {q: i for i, q in enumerate(emb_qids)}
    qrows = np.array([row[q] for q in query_ids])
    q_flat, q_off = subset_ragged(q_emb["colbert_flat"], q_emb["colbert_offsets"], qrows)
    d_flat, d_off = d_emb["colbert_flat"], d_emb["colbert_offsets"]
    mu = token_mean(d_flat)

    def evaluate(ranked) -> dict:
        run = {q: [doc_ids[j] for j in ranked[i]] for i, q in enumerate(query_ids)}
        pq = evaluate_run(run, qrels, [10, 100], metrics=["ndcg", "recall"])
        return {"ndcg@10": pq["ndcg@10"].mean(), "recall@100": pq["recall@100"].mean()}

    timings, summary = [], []

    def record(method, setting, family, ranked, df, index_bytes, old_latency, extra=None):
        timings.append(df)
        summary.append({"method": method, "setting": setting, "family": family, "index_mb": index_bytes / 1e6,
                        "query_nnz": df.query_nnz.mean(), "postings_read_mean": df.postings_read.mean(),
                        "postings_read_median": df.postings_read.median(),
                        "encode_ms": df.encode_ms.median(), "score_ms": df.score_ms.median(),
                        "topk_ms": df.topk_ms.median(), "total_ms": df.total_ms.median(),
                        "total_p95_ms": df.total_ms.quantile(0.95), "old_latency_ms": old_latency,
                        **evaluate(ranked), **(extra or {})})
        s = summary[-1]
        print(f"{method:38s} total {s['total_ms']:7.2f} ms (encode {s['encode_ms']:.2f} · score {s['score_ms']:.2f} "
              f"· top-k {s['topk_ms']:.2f}) · postings {s['postings_read_mean']:,.0f} · nDCG@10 {s['ndcg@10']:.4f}",
              flush=True)

    # --- SMVE ---------------------------------------------------------------------
    for w, k, reps, center in SMVE_SETTINGS:
        c = mu if center else None
        setting = f"w{w}_k{k}{f'_r{reps}' if reps > 1 else ''}{'_center' if center else ''}"
        print(f"\nencoding SMVE {setting} documents...")
        B = make_anchors(d_flat.shape[1], reps * w, seed=0)
        D = smve_encode(d_flat, d_off, B, k, is_query=False, center=c, reps=reps)
        index = InvertedIndex(D)
        mu_t = None if c is None else torch.from_numpy(c)

        def encode(i, B=B, k=k, reps=reps, mu_t=mu_t):
            x = torch.from_numpy(np.asarray(q_flat[q_off[i]:q_off[i + 1]], dtype=np.float32))
            if mu_t is not None:
                x = torch.nn.functional.normalize(x - mu_t, dim=1)
            vals, idx = topk_blocks(x @ B, k, reps)
            terms, inv = torch.unique(idx.flatten(), return_inverse=True)
            weights = torch.zeros(len(terms)).index_add_(0, inv, vals.flatten())  # query: SUM per anchor
            return terms.numpy(), weights.numpy()

        ranked, df = run_queries(f"SMVE {setting}", encode, lambda q, ix=index: ix.score(*q), n_q,
                                 lambda q, ix=index: ix.postings_read(q[0]))
        # Same scores as the earlier evaluation (spot-check 10 queries)?
        Q = smve_encode(q_flat, q_off, B, k, is_query=True, center=c, reps=reps, show_progress=False)
        ref = smve_scores(Q[:10], D)
        for i in range(10):
            np.testing.assert_allclose(index.score(*encode(i)), ref[i], rtol=1e-4, atol=1e-5)
        old = json.loads((BASE / "smve" / f"{setting}_seed0" / "summary.json").read_text())
        record(f"SMVE {setting}", setting, "SMVE", ranked, df, index.nbytes(),
               old["timing"]["single_query_latency_ms"])
        del D, index

    # --- BM25 -------------------------------------------------------------------------
    bm = BM25().fit(doc_texts)
    bm_index = InvertedIndex(bm.doc_weights)

    def bm_encode(i):
        q = bm.encode_queries([q_text[query_ids[i]]])
        return q.indices, q.data

    ranked, df = run_queries("BM25", bm_encode, lambda q: bm_index.score(*q), n_q, lambda q: bm_index.postings_read(q[0]))
    old = json.loads((BASE / "bm25" / "summary.json").read_text())["timing"]["single_query_latency_ms"]
    record("BM25", "k1=1.2 b=0.75", "lexical", ranked, df, bm_index.nbytes(), old)

    # --- BGE-M3 lexical ------------------------------------------------------------------
    D_lex = sp.csr_matrix((d_emb["sparse_weights"], d_emb["sparse_ids"], d_emb["sparse_indptr"]),
                          shape=(len(doc_ids), VOCAB_SIZE), dtype=np.float32)
    D_lex.sum_duplicates()
    lex_index = InvertedIndex(D_lex)
    qi, qw, qp = q_emb["sparse_ids"], q_emb["sparse_weights"], q_emb["sparse_indptr"]

    def lex_encode(i):
        r = qrows[i]
        terms, inv = np.unique(qi[qp[r]:qp[r + 1]], return_inverse=True)
        return terms, np.bincount(inv, weights=qw[qp[r]:qp[r + 1]]).astype(np.float32)

    ranked, df = run_queries("BGE-M3 lexical", lex_encode, lambda q: lex_index.score(*q), n_q,
                             lambda q: lex_index.postings_read(q[0]))
    old = json.loads((BASE / "lexical" / "summary.json").read_text())["timing"]["single_query_latency_ms"]
    record("BGE-M3 lexical", "learned token weights", "lexical", ranked, df, lex_index.nbytes(), old)

    # --- Dense brute force (reference) -------------------------------------------------------
    Dd, Qd = d_emb["dense"], q_emb["dense"][qrows]
    ranked, df = run_queries("BGE-M3 dense (brute force)", lambda i: Qd[i], lambda q: Dd @ q, n_q)
    old = json.loads((BASE / "dense" / "summary.json").read_text())["timing"]["single_query_latency_ms"]
    record("BGE-M3 dense (brute force)", "1024-d", "dense", ranked, df, Dd.nbytes, old)

    # --- MUVERA brute force (reference) ------------------------------------------------------
    R, ks, dp, center = MUVERA_SETTING
    fde = MuveraFDE(d_flat.shape[1], reps=R, k_sim=ks, d_proj=dp, seed=0)
    print("\nencoding MUVERA documents...")
    Dm = fde.encode(d_flat, d_off, is_query=False)
    ranked, df = run_queries(
        f"MUVERA r{R}_k{ks}_p{dp} (brute force)",
        lambda i: fde.encode(q_flat[q_off[i]:q_off[i + 1]], q_off[i:i + 2] - q_off[i], is_query=True,
                             show_progress=False)[0],
        lambda q: Dm @ q, n_q)
    old = json.loads((BASE / "muvera" / f"r{R}_k{ks}_p{dp}_seed0" / "summary.json").read_text())
    record(f"MUVERA r{R}_k{ks}_p{dp} (brute force)", f"{fde.out_dim} dims", "dense", ranked, df, Dm.nbytes,
           old["timing"]["single_query_latency_ms"])

    # --- Save ------------------------------------------------------------------------------------
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "summary.csv", index=False)
    pd.concat(timings).to_csv(OUT / "per_query.csv", index=False)
    cols = ["method", "index_mb", "query_nnz", "postings_read_mean", "encode_ms", "score_ms", "topk_ms",
            "total_ms", "total_p95_ms", "old_latency_ms", "ndcg@10", "recall@100"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in summ[cols].iterrows():
        lines.append("| " + " | ".join(
            f"{v:,.2f}" if isinstance(v, float) and abs(v) >= 1 else (f"{v:.4f}" if isinstance(v, float) else str(v))
            for v in r) + " |")
    text = "\n".join([
        "# Query latency with a hand-built inverted index (SciFact, 300 queries, CPU)\n",
        "Times are medians per query in ms; `old_latency_ms` = the earlier general-purpose code path. "
        "Scores identical to the earlier evaluation (checked).\n", *lines])
    (OUT / "summary.md").write_text(text + "\n")
    print("\n" + text)
    plot_index_latency(summ, pd.concat(timings), OUT / "plots" / "latency.png")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
