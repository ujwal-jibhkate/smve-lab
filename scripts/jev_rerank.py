"""Jev as the reranker, against the bge-reranker-v2-m3 cross-encoder.

Same setup as stage 4: first stage = BGE-M3 dense + 0.3 * lexical, rerank its top 20,
on the SciFact / NFCorpus / ArguAna test sets. The cross-encoder scores for exactly
these pairs are already cached from stage 4a, so the comparison is like for like.

Three ways of asking Jev (wording fixed before any results were seen):
  V1  one call per (query, candidate)  Noul "does the candidate directly address the query?"
  V2  one call per (query, candidate)  same, plus the dataset's task description
  V3  one call per query               Choice: which of the 20 candidates should rank first?

Task descriptions are the standard per-dataset instructions from E5-mistral
(Wang et al., 2023), not written for this experiment. Candidates are cut to
~300 words (the cross-encoder sees ~380 words: 512 tokens incl. the query).
Jev rounds probabilities, so ties keep the first stage's order.

    uv run python scripts/jev_rerank.py --pilot 50          # 50 queries per dataset
    uv run python scripts/jev_rerank.py                     # all test queries
    uv run python scripts/jev_rerank.py --latency-queries 10

Writes results/jev_rerank/{summary.csv, per_query.csv, report.md, latency.json, plots/}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare_runs import paired_stats  # noqa: E402
from jev_router import first_stage_tops, shorten  # noqa: E402

from smve_lab.config import RESULTS_DIR  # noqa: E402
from smve_lab.datasets import info, load_qrels, load_texts  # noqa: E402
from smve_lab.jev import Jev  # noqa: E402
from smve_lab.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from smve_lab.plots import plot_jev_rerank  # noqa: E402

OUT = RESULTS_DIR / "jev_rerank"
ROUTER = RESULTS_DIR / "router"
DATASETS = ["scifact", "nfcorpus", "arguana"]
DEPTH = 20
DOC_WORDS = 300
QUERY_WORDS = 250
PRICE = 0.042 / 1e6
TASK = {  # E5-mistral task instructions for these BEIR datasets
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim.",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question.",
    "arguana": "Given a claim, find documents that refute the claim.",
}
V1_Q = {"relevant": {
    "type": "noul",
    "instructions": "Does `candidate` directly and specifically address what `query` is looking for?",
    "criteria": {"true": "The candidate directly and specifically addresses the query.",
                 "false": "The candidate is only loosely related, on a neighbouring topic, or off-topic."}}}


def v2_q(ds: str) -> dict:
    return {"relevant": {
        "type": "noul",
        "instructions": {"task": TASK[ds],
                         "question": "Following `task`, is `candidate` a document that should be returned for `query`?"},
        "criteria": {"true": "The candidate is exactly the kind of document the task asks for, for this query.",
                     "false": "The candidate does not fulfil the task for this query, even if it is on a related topic."}}}


def v3_q(ds: str, n: int) -> dict:
    return {"first": {
        "type": "choice",
        "instructions": {"task": TASK[ds],
                         "question": "Following `task`, which of `candidates` should be ranked first for `query`?"},
        "criteria": {f"c{i:02d}": f"The document `candidates[{i}]`" for i in range(n)}}}


def rerank(first: list[str], scores: list[float]) -> list[str]:
    order = sorted(range(len(first)), key=lambda i: -scores[i])  # stable: ties keep first-stage order
    return [first[i] for i in order]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", type=int, default=0, help="only N queries per dataset (0 = all)")
    p.add_argument("--variants", nargs="+", default=["V1", "V2", "V3"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--latency-queries", type=int, default=0, help="live (uncached) latency probe per dataset")
    args = p.parse_args()
    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(ROUTER / "data.csv", dtype={"query_id": str})
    jev = Jev(OUT / "jev_cache.jsonl", workers=args.workers)
    per_query, latency = [], {}

    for ds in DATASETS:
        rows = data[(data.dataset == ds) & (data.split == "test")].reset_index(drop=True)
        if args.pilot:
            rows = rows.iloc[np.random.default_rng(0).choice(len(rows), min(args.pilot, len(rows)), replace=False)]
        qrels = load_qrels(ds)
        doc_ids, doc_texts, q_ids, q_texts = load_texts(ds)
        d_text, q_text = dict(zip(doc_ids, doc_texts)), dict(zip(q_ids, q_texts))
        tops = first_stage_tops(ds, rows, n=DEPTH)
        ce = {}  # SciFact test pairs live in the stage-2 cache, the rest in the router cache
        for f in [ROUTER / f"ce_cache_{ds}.csv", RESULTS_DIR / f"{ds}_bgem3" / "rerank" / "ce_scores.csv"]:
            if f.exists():
                c = pd.read_csv(f, dtype={"query_id": str, "doc_id": str})
                ce.update({(a, b): s for a, b, s in zip(c.query_id, c.doc_id, c.ce_score)})
        qids = list(rows.query_id)
        print(f"\n=== {info(ds).display}: {len(qids)} queries ===")

        def pair_items(q, questions):
            query = shorten(q_text[q], QUERY_WORDS)
            return [({"query": query, "candidate": shorten(d_text[d], DOC_WORDS)}, questions) for d in tops[("test", q)]]

        jev_scores, tokens = {}, {}
        if "V1" in args.variants or "V2" in args.variants:
            for v, qs in (("V1", V1_Q), ("V2", v2_q(ds))):
                if v not in args.variants:
                    continue
                items = [it for q in qids for it in pair_items(q, qs)]
                res = jev.ask_many(items, desc=f"{v} {ds}")
                for i, q in enumerate(qids):
                    chunk = res[i * DEPTH:(i + 1) * DEPTH]
                    jev_scores[(v, q)] = [r["answers"]["relevant"] for r in chunk]
                tokens[v] = sum(r["input_tokens"] for r in res)
        if "V3" in args.variants:
            items = []
            for q in qids:
                state = {"query": shorten(q_text[q], QUERY_WORDS),
                         "candidates": [shorten(d_text[d], DOC_WORDS) for d in tops[("test", q)]]}
                items.append((state, v3_q(ds, len(tops[("test", q)]))))
            res = jev.ask_many(items, desc=f"V3 {ds}")
            for q, r in zip(qids, res):
                probs = r["answers"]["first"]
                jev_scores[("V3", q)] = [probs.get(f"c{i:02d}", 0.0) for i in range(len(tops[("test", q)]))]
            tokens["V3"] = sum(r["input_tokens"] for r in res)

        for q, r in zip(qids, rows.itertuples()):
            first = tops[("test", q)]
            rankings = {"first stage": first, "cross-encoder": rerank(first, [ce[(q, d)] for d in first])}
            for v in args.variants:
                rankings[f"Jev {v}"] = rerank(first, jev_scores[(v, q)])
            for method, ranked in rankings.items():
                per_query.append({"dataset": ds, "query_id": q, "method": method,
                                  "ndcg@10": ndcg_at_k(ranked, qrels[q], 10), "mrr@10": mrr_at_k(ranked, qrels[q], 10),
                                  "recall@10": recall_at_k(ranked, qrels[q], 10)})
            per_query.append({"dataset": ds, "query_id": q, "method": "base (MaxSim@10)", "ndcg@10": r.ndcg_base,
                              "mrr@10": np.nan, "recall@10": np.nan})
        latency[ds] = {"tokens": tokens, "n_queries": len(qids)}

        # Live latency probe for one query at a time (bypasses the cache; small extra cost).
        if args.latency_queries:
            sample = qids[:args.latency_queries]
            lat = {}
            for v in [v for v in args.variants if v in ("V1", "V2")]:
                qs = V1_Q if v == "V1" else v2_q(ds)
                times = []
                for q in sample:
                    items = pair_items(q, qs)
                    t = time.perf_counter()
                    with ThreadPoolExecutor(DEPTH) as pool:
                        list(pool.map(lambda it: jev.ask(*it, use_cache=False), items))
                    times.append((time.perf_counter() - t) * 1000)
                lat[v] = float(np.median(times))
            if "V3" in args.variants:
                times = []
                for q in sample:
                    state = {"query": shorten(q_text[q], QUERY_WORDS),
                             "candidates": [shorten(d_text[d], DOC_WORDS) for d in tops[("test", q)]]}
                    t = time.perf_counter()
                    jev.ask(state, v3_q(ds, DEPTH), use_cache=False)
                    times.append((time.perf_counter() - t) * 1000)
                lat["V3"] = float(np.median(times))
            latency[ds]["jev_ms"] = lat
            print("latency (median ms, one query):", {k: round(v) for k, v in lat.items()})
        router_lat = json.loads((ROUTER / "latency.json").read_text())[f"{ds}/test"]
        latency[ds]["cross_encoder_ms"] = router_lat["ce20_ms"]
        latency[ds]["first_stage_ms"] = router_lat["first_stage_ms"]

    pq = pd.DataFrame(per_query)
    tag = "_pilot" if args.pilot else ""
    pq.to_csv(OUT / f"per_query{tag}.csv", index=False)
    lat_path = OUT / f"latency{tag}.json"
    if lat_path.exists() and not args.latency_queries:  # keep an earlier latency probe
        old = json.loads(lat_path.read_text())
        for ds in latency:
            if "jev_ms" in old.get(ds, {}):
                latency[ds]["jev_ms"] = old[ds]["jev_ms"]
    lat_path.write_text(json.dumps(latency, indent=2))

    summary = pq.groupby(["dataset", "method"], sort=False)[["ndcg@10", "mrr@10", "recall@10"]].mean().reset_index()
    summary.to_csv(OUT / f"summary{tag}.csv", index=False)

    lines = [f"# Jev as the reranker vs bge-reranker-v2-m3 ({'pilot' if args.pilot else 'full run'})\n",
             "First stage = BGE-M3 dense + 0.3·lexical; rerank its top 20. Cross-encoder on the Mac GPU (MPS, fp16); "
             "Jev via API (`jev-1.13.0`). Paired tests vs the cross-encoder on nDCG@10.\n"]
    for ds in DATASETS:
        s = summary[summary.dataset == ds]
        if s.empty:
            continue
        lines.append(f"\n## {info(ds).display} ({latency[ds]['n_queries']} queries)\n")
        lines.append("| method | nDCG@10 | MRR@10 | Recall@10 | Δ nDCG@10 vs cross-encoder [95% CI] | better/tied/worse | latency | tokens · cost |")
        lines.append("|---|---|---|---|---|---|---|---|")
        g = pq[pq.dataset == ds].pivot(index="query_id", columns="method", values="ndcg@10")
        for _, r in s.iterrows():
            m = r["method"]
            if m == "cross-encoder":
                comp, wtl = "–", "–"
            else:
                st = paired_stats(g["cross-encoder"], g[m], 5000)
                comp = f"{st['mean_delta']:+.4f} [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]"
                wtl = f"{st['wins']} / {st['ties']} / {st['losses']}"
            lat_ms = {"cross-encoder": latency[ds]["cross_encoder_ms"]}.get(m)
            if m.startswith("Jev"):
                lat_ms = latency[ds].get("jev_ms", {}).get(m.split()[1])
            lat_s = f"{lat_ms:,.0f} ms" if lat_ms else ("~3 ms" if m == "first stage" else ("~5–15 ms" if m.startswith("base") else "–"))
            tok = latency[ds]["tokens"].get(m.split()[1]) if m.startswith("Jev") else None
            cost = f"{tok:,} · ${tok * PRICE:.3f}" if tok else "–"
            mrr = "–" if np.isnan(r["mrr@10"]) else f"{r['mrr@10']:.4f}"
            rec = "–" if np.isnan(r["recall@10"]) else f"{r['recall@10']:.4f}"
            lines.append(f"| {m} | {r['ndcg@10']:.4f} | {mrr} | {rec} | {comp} | {wtl} | {lat_s} | {cost} |")
    text = "\n".join(lines)
    (OUT / f"report{tag}.md").write_text(text + "\n")
    print("\n" + text)
    if not args.pilot:
        plot_jev_rerank(summary, latency, OUT / "plots" / "jev_rerank.png")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
