# smve-lab

An independent evaluation of **SMVE** (sparse multi-vector embeddings, a method from [TopK](https://topk.io/blog/20260311-smve-multi-vector-retrieval)) as a cheap replacement for exhaustive ColBERT-style **MaxSim**. It runs on BGE-M3 multi-vector output and compares SMVE against MUVERA, BM25, BGE-M3 dense/lexical/hybrid, and a cross-encoder reranker.

Datasets: SciFact, NFCorpus, ArguAna (BEIR). All methods are evaluated on the same stored BGE-M3 ColBERT vectors.

## Findings so far

- **Standalone, SMVE and MUVERA trail a single dense vector** on all three datasets (e.g. SciFact nDCG@10: MaxSim 0.699, dense 0.654, best SMVE ~0.60).
- **After exact MaxSim reranking of the top 100, every first stage lands within noise of exhaustive MaxSim.** BM25 → MaxSim is best or tied on all three datasets.
- **At equal storage SMVE wins over MUVERA; at equal latency MUVERA wins.**
- With a hand-built inverted index SMVE query latency drops from 87 ms to 19 ms, and most of what remains is the query projection.
- Best pipeline on SciFact: dense+lexical → `bge-reranker-v2-m3` @20 (nDCG@10 0.731, ~1.6 s/query on MPS). The difference from exhaustive MaxSim is not significant (p=0.08).

Full tables: [`results/cross_dataset/summary.md`](results/cross_dataset/summary.md). Notes on SMVE repetitions: [`docs/repetitions_explained.md`](docs/repetitions_explained.md).

## Layout

| path | contents |
|---|---|
| `src/smve_lab/` | MaxSim, SMVE, evaluation, dataset loading, plots, storage |
| `scripts/` | one script per method / experiment (`evaluate_*.py`, `compare_*.py`, `plot_*.py`) |
| `results/<dataset>_bgem3/` | per-method metrics, per-query results, plots |
| `results/cross_dataset/` | cross-dataset comparison |
| `tests/` | unit tests |

Large regenerable outputs (`scores.npz`, `run.trec`) and the `data/` and `artifacts/` folders (embeddings) are not tracked.

## Reproduce

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/encode_beir.py --dataset nfcorpus   # BGE-M3 embeddings
bash scripts/run_all_methods.sh nfcorpus                  # every first stage + report
uv run python scripts/cross_dataset_summary.py            # cross-dataset tables
```

SciFact uses `scripts/encode_scifact.py`. Check each script's `--help` for options.
