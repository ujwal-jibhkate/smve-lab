"""Loading and preparing the BeIR/scifact retrieval benchmark."""

from __future__ import annotations

from datasets import load_dataset


def _add_full_text(batch: dict) -> dict:
    return {"full_text": [t + ". " + x for t, x in zip(batch["title"], batch["text"])]}


def load_scifact():
    """Return (corpus, queries, qrels, doc_ids, doc_texts, query_ids, query_texts).

    `full_text` is title + ". " + body, matching how BEIR benchmarks are
    normally embedded — dropping the title loses signal on short-abstract
    corpora like scifact.
    """
    corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
    queries = load_dataset("BeIR/scifact", "queries")["queries"]
    qrels = load_dataset("BeIR/scifact-qrels")

    corpus = corpus.map(_add_full_text, batched=True)
    queries = queries.map(_add_full_text, batched=True)

    doc_ids = [str(x) for x in corpus["_id"]]
    doc_texts = corpus["full_text"]
    query_ids = [str(x) for x in queries["_id"]]
    query_texts = queries["full_text"]

    return corpus, queries, qrels, doc_ids, doc_texts, query_ids, query_texts


def load_qrels(split: str = "test") -> dict[str, dict[str, int]]:
    """Return relevance judgments as {query_id: {doc_id: relevance}}.

    BEIR reports scifact on the "test" split (300 queries); "train" has 809
    more. Ids are cast to str to match the embedding id files.
    """
    ds = load_dataset("BeIR/scifact-qrels")[split]
    qrels: dict[str, dict[str, int]] = {}
    for qid, did, score in zip(ds["query-id"], ds["corpus-id"], ds["score"]):
        qrels.setdefault(str(qid), {})[str(did)] = int(score)
    return qrels
