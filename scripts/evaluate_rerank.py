"""Step 2: rerank first-stage candidates with a cross-encoder vs. with MaxSim.

Two-stage retrieval: a cheap FIRST STAGE finds candidates, an expensive
RERANKER re-orders the top-d of them. We compare, for d in {10, 20, 50, 100}:

  first stages:  SMVE (w=65,536 k=32 centered)   and   BGE-M3 dense + lexical
  rerankers:     MaxSim (BGE-M3 ColBERT vectors, CPU)
                 cross-encoder bge-reranker-v2-m3 (fp16 on the Mac GPU / MPS)

on quality (nDCG@10, MRR@10, Recall@10/100), single-query latency, compute
(FLOPs per query) and storage. Documents below depth d keep their first-stage
order after the reranked ones, so Recall@100 only changes when d = 100 (and
then not at all, since reranking reorders the same 100 docs).

The cross-encoder scores for all top-100 pairs are computed once and cached
(resumable; ~1 hour on an M1 Pro), so each depth is a cheap re-sort.
Cross-encoder scores are per pair, so scoring the top-100 once and reusing the
scores for smaller d gives exactly the same result as scoring only the top-d.

    uv run python scripts/evaluate_rerank.py

Writes results/scifact_bgem3/rerank/: ce_scores.csv (cache), latency.json,
summary.csv, per_query.csv, report.md, meta.json, plots/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from compare_runs import paired_stats  # noqa: E402

from smve_lab.config import ARTIFACTS_DIR, RESULTS_DIR  # noqa: E402
from smve_lab.evaluation import Timer, read_trec_run  # noqa: E402
from smve_lab.maxsim import maxsim_scores, subset_ragged  # noqa: E402
from smve_lab.metrics import evaluate_run  # noqa: E402
from smve_lab.plots import plot_rerank_cost, plot_rerank_depth, plot_rerank_latency  # noqa: E402
from smve_lab.reranker import MODEL_NAME, CrossEncoder  # noqa: E402
from smve_lab.scifact import load_qrels, load_scifact  # noqa: E402
from smve_lab.storage import load_separate  # noqa: E402

BASE = RESULTS_DIR / "scifact_bgem3"
OUT = BASE / "rerank"
EMB_DIR = ARTIFACTS_DIR / "embeddings" / "scifact_bgem3"
FIRST_STAGES = {
    "SMVE": BASE / "smve" / "w65536_k32_center_seed0",
    "Dense + lexical": BASE / "hybrid_dense_lexical",
}
DEPTHS = [10, 20, 50, 100]
POOL = 100  # candidates kept from each first stage
DIM = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--latency-queries", type=int, default=20)
    p.add_argument("--remeasure-latency", action="store_true")
    p.add_argument("--limit-queries", type=int, default=0, help="debug: only the first N queries")
    p.add_argument("--out-dir", type=Path, default=OUT)
    return p.parse_args()


def score_all_pairs(ce: CrossEncoder, pairs: pd.DataFrame, q_text: dict, d_text: dict, cache: Path,
                    batch_size: int) -> pd.DataFrame:
    """Cross-encoder score + token count for every (query, doc) pair, cached and resumable."""
    done = pd.read_csv(cache, dtype={"query_id": str, "doc_id": str}) if cache.exists() else \
        pd.DataFrame(columns=["query_id", "doc_id", "ce_score", "n_tokens"])
    key = set(zip(done["query_id"], done["doc_id"]))
    todo = pairs[[k not in key for k in zip(pairs["query_id"], pairs["doc_id"])]]
    print(f"cross-encoder pairs: {len(pairs):,} total, {len(done):,} cached, {len(todo):,} to score")
    chunk = 1024  # write to disk after every chunk, so an interruption loses at most one chunk
    for s in tqdm(range(0, len(todo), chunk), desc="cross-encoder", unit="chunk"):
        part = todo.iloc[s:s + chunk]
        qs = [q_text[q] for q in part["query_id"]]
        ds = [d_text[d] for d in part["doc_id"]]
        out = part.assign(ce_score=ce.score(qs, ds, batch_size), n_tokens=ce.token_lengths(qs, ds))
        out.to_csv(cache, mode="a", header=not cache.exists(), index=False)
    return pd.read_csv(cache, dtype={"query_id": str, "doc_id": str})


def rerank(first: list[str], scores: dict[str, float], depth: int) -> list[str]:
    """Sort the top-`depth` by reranker score; keep the rest in first-stage order."""
    top = sorted(first[:depth], key=lambda d: -scores[d])
    return top + first[depth:]


def main() -> None:
    global OUT
    args = parse_args()
    OUT = args.out_dir
    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    qrels = load_qrels("test")
    if args.limit_queries:
        qrels = dict(list(qrels.items())[:args.limit_queries])
    query_ids = list(qrels)
    _, _, _, doc_ids, doc_texts, all_qids, q_texts = load_scifact()
    d_text = dict(zip(doc_ids, doc_texts))
    q_text = dict(zip(all_qids, q_texts))

    first_runs = {name: {q: r[:POOL] for q, r in read_trec_run(folder / "run.trec").items()}
                  for name, folder in FIRST_STAGES.items()}
    first_summ = {name: json.loads((folder / "summary.json").read_text()) for name, folder in FIRST_STAGES.items()}
    pairs = pd.DataFrame(sorted({(q, d) for run in first_runs.values() for q in query_ids for d in run[q]}),
                         columns=["query_id", "doc_id"])

    # --- Cross-encoder scores for every candidate pair (the slow part) ------
    ce = CrossEncoder(device=args.device)
    t0 = time.time()
    ce_df = score_all_pairs(ce, pairs, q_text, d_text, OUT / "ce_scores.csv", args.batch_size)
    print(f"cross-encoder scoring done ({time.time() - t0:.0f}s this run)")
    ce_score = {(r.query_id, r.doc_id): r.ce_score for r in ce_df.itertuples()}
    ce_tokens = {(r.query_id, r.doc_id): r.n_tokens for r in ce_df.itertuples()}

    # --- MaxSim scores (exact, from the stored exhaustive score matrix) -----
    z = np.load(BASE / "colbert_maxsim" / "scores.npz")
    ms_q = {q: i for i, q in enumerate(z["query_ids"].tolist())}
    ms_d = {d: j for j, d in enumerate(z["doc_ids"].tolist())}
    ms = z["scores"]

    q_emb, q_all = load_separate(EMB_DIR, "queries")
    d_emb, d_all = load_separate(EMB_DIR, "docs", mmap_colbert=True)
    q_row = {q: i for i, q in enumerate(q_all)}
    d_row = {d: i for i, d in enumerate(d_all)}
    qo, do = q_emb["colbert_offsets"], d_emb["colbert_offsets"]
    q_len = {q: int(qo[q_row[q] + 1] - qo[q_row[q]]) for q in query_ids}
    d_len = {d: int(do[d_row[d] + 1] - do[d_row[d]]) for d in d_all}

    # --- Latency: one query at a time, median over a sample of queries -------
    lat_path = OUT / "latency.json"
    if lat_path.exists() and not args.remeasure_latency:
        latency = json.loads(lat_path.read_text())
    else:
        sample = [query_ids[i] for i in np.linspace(0, len(query_ids) - 1, args.latency_queries).astype(int)]
        latency = {}
        for name, run in first_runs.items():
            for depth in DEPTHS:
                ce_t, ms_t = [], []
                for q in [sample[0]] + sample:  # first one is a warm-up, dropped below
                    docs = run[q][:depth]
                    with Timer() as t:
                        ce.score([q_text[q]] * len(docs), [d_text[d] for d in docs], args.batch_size)
                    ce_t.append(t.seconds * 1000)
                    i = q_row[q]
                    with Timer() as t:
                        c_flat, c_off = subset_ragged(d_emb["colbert_flat"], do, [d_row[d] for d in docs])
                        maxsim_scores(q_emb["colbert_flat"][qo[i]:qo[i + 1]], np.array([0, qo[i + 1] - qo[i]]),
                                      c_flat, c_off, show_progress=False)
                    ms_t.append(t.seconds * 1000)
                latency[f"{name}|cross-encoder|{depth}"] = float(np.median(ce_t[1:]))
                latency[f"{name}|MaxSim|{depth}"] = float(np.median(ms_t[1:]))
                print(f"latency {name} d={depth}: cross-encoder {latency[f'{name}|cross-encoder|{depth}']:.0f} ms, "
                      f"MaxSim {latency[f'{name}|MaxSim|{depth}']:.1f} ms")
        lat_path.write_text(json.dumps(latency, indent=2))

    # --- Cost models -----------------------------------------------------------
    maxsim_bytes = json.loads((BASE / "colbert_maxsim" / "summary.json").read_text())["index"]["bytes"]
    text_bytes = sum(len(t.encode("utf-8")) for t in doc_texts)
    n_tokens_corpus = int(do[-1])

    def first_stage_flops(name: str, q: str) -> float:
        if name == "Dense + lexical":
            return 2 * DIM * len(doc_ids)  # dense dot products; the sparse lexical part is negligible
        s = first_summ[name]
        w = s["params"]["w"]
        # project query tokens onto w anchors + sparse dot (query non-zeros x average posting length)
        postings = s["index"]["nnz_per_doc"] * len(doc_ids) / w
        return 2 * q_len[q] * DIM * w + 2 * s["index"]["nnz_per_query"] * postings

    def rerank_flops(reranker: str, q: str, docs: list[str]) -> float:
        if reranker == "cross-encoder":
            return float(ce.flops_per_pair(np.array([ce_tokens[(q, d)] for d in docs])).sum())
        return 2.0 * q_len[q] * sum(d_len[d] for d in docs) * DIM

    # --- Evaluate every (first stage, reranker, depth) -------------------------
    rows, per_query = [], {}
    ks = [10, 100]
    for name, run in first_runs.items():
        fs = first_summ[name]
        configs = [("none", 0)] + [(r, d) for r in ("MaxSim", "cross-encoder") for d in DEPTHS]
        for reranker, depth in configs:
            ranked = {}
            for q in query_ids:
                if reranker == "none":
                    ranked[q] = run[q]
                elif reranker == "MaxSim":
                    ranked[q] = rerank(run[q], {d: ms[ms_q[q], ms_d[d]] for d in run[q][:depth]}, depth)
                else:
                    ranked[q] = rerank(run[q], {d: ce_score[(q, d)] for d in run[q][:depth]}, depth)
            pq = evaluate_run(ranked, qrels, ks, metrics=["ndcg", "recall", "mrr"])
            label = f"{name}" if reranker == "none" else f"{name} + {reranker} @{depth}"
            per_query[label] = pq["ndcg@10"]
            rr_lat = 0.0 if reranker == "none" else latency[f"{name}|{reranker}|{depth}"]
            first_lat = fs["timing"]["single_query_latency_ms"]
            f_first = np.mean([first_stage_flops(name, q) for q in query_ids])
            f_rr = 0.0 if reranker == "none" else np.mean([rerank_flops(reranker, q, run[q][:depth])
                                                          for q in query_ids])
            storage = fs["index"]["bytes"] + {"none": 0, "MaxSim": maxsim_bytes,
                                              "cross-encoder": ce.weight_bytes + text_bytes}[reranker]
            rows.append({
                "config": label, "first_stage": name, "reranker": reranker, "depth": depth,
                **pq[[c for c in pq.columns if "@" in c]].mean().round(5).to_dict(),
                "latency_first_ms": first_lat, "latency_rerank_ms": round(rr_lat, 2),
                "latency_total_ms": round(first_lat + rr_lat, 2),
                "flops_first": f_first, "flops_rerank": f_rr, "flops_total": f_first + f_rr,
                "storage_bytes": storage,
                "rerank_device": {"none": "-", "MaxSim": "CPU", "cross-encoder": f"{args.device} fp16"}[reranker],
            })

    # References: exhaustive MaxSim and dense alone.
    refs = {}
    for label, folder in (("Exhaustive MaxSim", "colbert_maxsim"), ("BGE-M3 dense", "dense")):
        s = json.loads((BASE / folder / "summary.json").read_text())
        pq = pd.read_csv(BASE / folder / "per_query.csv", index_col="query_id", dtype={"query_id": str})
        per_query[label] = pq["ndcg@10"]
        flops = (np.mean([2.0 * q_len[q] * n_tokens_corpus * DIM for q in query_ids]) if folder == "colbert_maxsim"
                 else 2 * DIM * len(doc_ids))
        refs[label] = {"ndcg@10": s["metrics"]["ndcg@10"], "mrr@10": s["metrics"]["mrr@10"],
                       "recall@10": s["metrics"]["recall@10"], "recall@100": s["metrics"]["recall@100"],
                       "latency_total_ms": s["timing"]["single_query_latency_ms"],
                       "flops_total": flops, "storage_bytes": s["index"]["bytes"]}

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    pq_df = pd.DataFrame(per_query).loc[query_ids]  # align every column on the evaluated queries
    pq_df.to_csv(OUT / "per_query.csv")

    # Paired comparisons of the depth-100 pipelines against exhaustive MaxSim.
    paired = {}
    for label in [c for c in pq_df.columns if c.endswith("@100")]:
        paired[label] = paired_stats(pq_df["Exhaustive MaxSim"], pq_df[label], 10_000)

    (OUT / "meta.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cross_encoder": {"model": MODEL_NAME, "device": args.device, "dtype": "fp16", "max_length": ce.max_length,
                          "params": ce.n_params, "non_embedding_params": ce.n_params_non_embedding,
                          "weight_bytes": ce.weight_bytes, "mean_tokens_per_pair": float(ce_df["n_tokens"].mean())},
        "maxsim_rerank_device": "CPU", "first_stages": {k: str(v) for k, v in FIRST_STAGES.items()},
        "depths": DEPTHS, "latency_queries": args.latency_queries, "corpus_text_bytes": text_bytes,
        "references": refs, "paired_vs_exhaustive_maxsim_ndcg10": paired,
    }, indent=2, default=float))

    # --- Report + plots -----------------------------------------------------------
    def fmt_bytes(b):
        return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"

    tbl = summary.assign(latency=summary["latency_total_ms"].map(lambda v: f"{v:,.0f} ms" if v >= 10 else f"{v:.1f} ms"),
                         GFLOPs=(summary["flops_total"] / 1e9).map(lambda v: f"{v:,.2f}"),
                         storage=summary["storage_bytes"].map(fmt_bytes))
    cols = ["config", "ndcg@10", "mrr@10", "recall@10", "recall@100", "latency", "GFLOPs", "storage", "rerank_device"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in tbl[cols].iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    ref_lines = [f"| {k} | {v['ndcg@10']:.4f} | {v['latency_total_ms']:,.1f} ms | {v['flops_total'] / 1e9:,.2f} "
                 f"| {fmt_bytes(v['storage_bytes'])} |" for k, v in refs.items()]
    pair_lines = [f"| {k} | {v['mean_delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                  f"{v['wins']} / {v['ties']} / {v['losses']} | {v['wilcoxon_p']:.2g} |" for k, v in paired.items()]
    report = "\n".join([
        "# Reranking: cross-encoder vs MaxSim (SciFact, 300 test queries)\n",
        f"Cross-encoder: `{MODEL_NAME}`, {ce.n_params / 1e6:.0f}M params, fp16 on {args.device}. "
        "MaxSim rerank: BGE-M3 ColBERT vectors on CPU. Latency = one query on its own "
        f"(median of {args.latency_queries}), first stage + rerank. GFLOPs = per query.\n",
        "## All pipelines\n", *lines,
        "\n## References\n", "| method | nDCG@10 | latency | GFLOPs | storage |", "|---|---|---|---|---|", *ref_lines,
        "\n## Depth-100 pipelines vs exhaustive MaxSim (nDCG@10, paired)\n",
        "| pipeline | Δ [95% CI] | better / tied / worse | Wilcoxon p |", "|---|---|---|---|", *pair_lines,
    ])
    (OUT / "report.md").write_text(report + "\n")
    print(report)

    plot_rerank_depth(summary, refs, OUT / "plots" / "quality_vs_depth.png")
    plot_rerank_latency(summary, refs, OUT / "plots" / "quality_vs_latency.png")
    plot_rerank_cost(summary, refs, OUT / "plots" / "cost.png")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
