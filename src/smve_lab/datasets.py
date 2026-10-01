"""BEIR datasets used in this project, behind one interface.

    texts = load_texts("arguana")   -> doc_ids, doc_texts, query_ids, query_texts
    qrels = load_qrels("arguana")   -> {query_id: {doc_id: relevance}}

Paths follow one pattern so every script can take --dataset:
    artifacts/embeddings/{name}_bgem3/   BGE-M3 vectors
    results/{name}_bgem3/                 evaluation outputs

Per-dataset quirks:
  - ignore_identical_ids: in ArguAna the query is itself an argument in the
    corpus, so a document with the query's own id would trivially rank first.
    BEIR's standard evaluation drops such self-matches; so do we.
  - SciFact keeps its original text joining (title + ". " + text, even for an
    empty title) so its stored embeddings stay valid. New datasets only add the
    title when there is one.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

from smve_lab.config import ARTIFACTS_DIR, RESULTS_DIR


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    display: str
    ignore_identical_ids: bool = False


DATASETS = {
    "scifact": DatasetInfo("scifact", "SciFact"),
    "nfcorpus": DatasetInfo("nfcorpus", "NFCorpus"),
    "arguana": DatasetInfo("arguana", "ArguAna", ignore_identical_ids=True),
}


def info(name: str) -> DatasetInfo:
    return DATASETS[name]


def emb_dir(name: str):
    return ARTIFACTS_DIR / "embeddings" / f"{name}_bgem3"


def results_dir(name: str):
    return RESULTS_DIR / f"{name}_bgem3"


def _join(title: str, text: str) -> str:
    title, text = (title or "").strip(), (text or "").strip()
    return f"{title}. {text}" if title else text


def load_texts(name: str) -> tuple[list[str], list[str], list[str], list[str]]:
    if name == "scifact":
        from smve_lab.scifact import load_scifact
        _, _, _, doc_ids, doc_texts, query_ids, query_texts = load_scifact()
        return doc_ids, list(doc_texts), query_ids, list(query_texts)
    corpus = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries")["queries"]
    doc_ids = [str(x) for x in corpus["_id"]]
    doc_texts = [_join(t, x) for t, x in zip(corpus["title"], corpus["text"])]
    query_ids = [str(x) for x in queries["_id"]]
    query_texts = [_join(t, x) for t, x in zip(queries["title"], queries["text"])]
    return doc_ids, doc_texts, query_ids, query_texts


def load_qrels(name: str, split: str = "test") -> dict[str, dict[str, int]]:
    ds = load_dataset(f"BeIR/{name}-qrels")[split]
    qrels: dict[str, dict[str, int]] = {}
    for qid, did, score in zip(ds["query-id"], ds["corpus-id"], ds["score"]):
        if int(score) > 0:
            qrels.setdefault(str(qid), {})[str(did)] = int(score)
    return qrels
