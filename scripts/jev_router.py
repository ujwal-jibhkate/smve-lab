"""Stage 4b, part 1: ask Jev about every routing query.

Two versions of what Jev sees (the "state"):
  A  query only                -> can run while the first stage runs (no extra wait)
  B  query + top-3 candidates  -> knows more, but must wait for the first stage

Questions follow Jev's guidance (docs: "atomic questions, composed in code";
known weak spots: numbers, literal reading, irrelevant context): no scores or
numbers are shown, each question asks one specific thing with explicit
yes/no criteria, and candidates are shortened to their first ~120 words.

Each answer is P(yes). They become routing signals two ways (see
evaluate_jev_router.py): directly, as zero-shot routers needing no training,
and as features in a logistic regression next to the first-stage features.

    uv run python scripts/jev_router.py --pilot 50   # 50 queries per test set
    uv run python scripts/jev_router.py              # all 2,838 queries

Writes results/router/jev_answers.csv (and the response cache jev_cache.jsonl).
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from smve_lab.config import RESULTS_DIR
from smve_lab.datasets import load_texts
from smve_lab.jev import Jev

OUT = RESULTS_DIR / "router"
TOP_N = 3
WORDS = 120

QUERY_QUESTIONS = {  # state A and B
    "q_ambiguous": {
        "type": "noul",
        "instructions": "Is `query` vague or ambiguous, so that several quite different documents could each "
                        "reasonably be what the person is looking for?",
        "criteria": {"true": "The query is broad, vague, or open to several different readings.",
                     "false": "The query is specific about what it is looking for."},
    },
    "q_needs_reasoning": {
        "type": "noul",
        "instructions": "To decide whether a document is the right match for `query`, must a reader reason about "
                        "how the document relates to the query (for example whether it supports, contradicts, or "
                        "argues against it), instead of only checking that it is about the same topic?",
        "criteria": {"true": "Matching depends on the relationship between query and document, not just shared topic.",
                     "false": "A document about the same specific topic would be a match."},
    },
    "q_specific_terms": {
        "type": "noul",
        "instructions": "Does `query` contain specific technical terms, names, or entities that the right "
                        "document would very likely also contain word for word?",
        "criteria": {"true": "The query has distinctive specific terms or names.",
                     "false": "The query uses only general, everyday wording."},
    },
}
CANDIDATE_QUESTIONS = {  # state B only
    "c_top1_direct": {
        "type": "noul",
        "instructions": "Does `top_candidates[0]` directly and specifically address what `query` is looking for?",
        "criteria": {"true": "The first candidate directly and specifically addresses the query.",
                     "false": "The first candidate is only loosely related or on a neighbouring topic."},
    },
    "c_other_better": {
        "type": "noul",
        "instructions": "Does `top_candidates[1]` or `top_candidates[2]` address `query` more directly than "
                        "`top_candidates[0]` does?",
        "criteria": {"true": "A lower-ranked candidate is a clearly better match than the first one.",
                     "false": "The first candidate is at least as good a match as the others."},
    },
    "c_any_direct": {
        "type": "noul",
        "instructions": "Does at least one of `top_candidates` directly and specifically address what `query` "
                        "is looking for?",
        "criteria": {"true": "At least one candidate directly and specifically addresses the query.",
                     "false": "All candidates are only loosely related to the query."},
    },
}


def shorten(text: str, n: int = WORDS) -> str:
    words = re.split(r"\s+", text.strip())
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", type=int, default=0, help="only N queries per test set (0 = all queries)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    data = pd.read_csv(OUT / "data.csv", dtype={"query_id": str})
    if args.pilot:
        rng = np.random.default_rng(0)
        parts = []
        for (ds, split), g in data[data.split == "test"].groupby(["dataset", "split"]):
            parts.append(g.iloc[rng.choice(len(g), min(args.pilot, len(g)), replace=False)])
        data = pd.concat(parts)

    jev = Jev(OUT / "jev_cache.jsonl", workers=args.workers)
    rows, items_a, items_b = [], [], []
    for ds in data.dataset.unique():
        doc_ids, doc_texts, q_ids, q_texts = load_texts(ds)
        d_text, q_text = dict(zip(doc_ids, doc_texts)), dict(zip(q_ids, q_texts))
        # The candidates the router sees: the first stage's top 3 (same ranking as build_router_data.py).
        tops = first_stage_tops(ds, data[data.dataset == ds])
        for _, r in data[data.dataset == ds].iterrows():
            query = shorten(q_text[r.query_id], 200)
            state_a = {"query": query}
            state_b = {"query": query,
                       "top_candidates": [shorten(d_text[d]) for d in tops[(r.split, r.query_id)][:TOP_N]]}
            items_a.append((state_a, QUERY_QUESTIONS))
            items_b.append((state_b, {**QUERY_QUESTIONS, **CANDIDATE_QUESTIONS}))
            rows.append({"dataset": r.dataset, "split": r.split, "query_id": r.query_id})

    res_a = jev.ask_many(items_a, desc="Jev, query only")
    res_b = jev.ask_many(items_b, desc="Jev, query + top 3")
    for row, a, b in zip(rows, res_a, res_b):
        row.update({f"A_{k}": v for k, v in a["answers"].items()})
        row.update({f"B_{k}": v for k, v in b["answers"].items()})
        row.update({"A_latency_ms": a["latency_ms"], "B_latency_ms": b["latency_ms"],
                    "A_tokens": a["input_tokens"], "B_tokens": b["input_tokens"]})
    out = pd.DataFrame(rows)
    path = OUT / ("jev_answers_pilot.csv" if args.pilot else "jev_answers.csv")
    out.to_csv(path, index=False)
    tokens = out.A_tokens.sum() + out.B_tokens.sum()
    print(f"\n{len(out)} queries · {tokens:,} input tokens (~${tokens / 1e6 * 0.042:.3f}) · "
          f"median latency A {out.A_latency_ms.median():.0f} ms, B {out.B_latency_ms.median():.0f} ms -> {path}")


def first_stage_tops(ds: str, rows: pd.DataFrame, n: int = TOP_N) -> dict:
    """Top-n first-stage documents per (split, query): dense + 0.3 * lexical, self-matches removed."""
    import scipy.sparse as sp

    from build_router_data import ALPHA, VOCAB
    from smve_lab.datasets import emb_dir, info
    from smve_lab.storage import load_separate

    q_emb, emb_qids = load_separate(emb_dir(ds), "queries")
    d_emb, doc_ids = load_separate(emb_dir(ds), "docs", mmap_colbert=True)
    qrow = {q: i for i, q in enumerate(emb_qids)}
    idx = np.array([qrow[q] for q in rows.query_id])

    def sparse(emb, r):
        m = sp.csr_matrix((emb["sparse_weights"], emb["sparse_ids"], emb["sparse_indptr"]),
                          shape=(len(emb["sparse_indptr"]) - 1, VOCAB), dtype=np.float32)
        m.sum_duplicates()
        return m[r]

    hyb = q_emb["dense"][idx] @ d_emb["dense"].T + ALPHA * np.asarray(
        (sparse(q_emb, idx) @ sparse(d_emb, np.arange(len(doc_ids))).T).todense(), dtype=np.float32)
    col = {d: j for j, d in enumerate(doc_ids)}
    out = {}
    for i, (split, q) in enumerate(zip(rows.split, rows.query_id)):
        if info(ds).ignore_identical_ids and q in col:
            hyb[i, col[q]] = -np.inf
        out[(split, q)] = [doc_ids[j] for j in np.argsort(-hyb[i])[:n]]
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    main()
