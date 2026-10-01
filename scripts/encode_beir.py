"""Encode any project dataset's corpus + queries with BGE-M3 (dense, lexical, ColBERT).

    uv run python scripts/encode_beir.py --dataset nfcorpus
    uv run python scripts/encode_beir.py --dataset arguana

Same model and settings as encode_scifact.py; saves the mmap-able per-array
.npy layout to artifacts/embeddings/{dataset}_bgem3/ (no combined .npz copy).
"""

from __future__ import annotations

import argparse
import time

from smve_lab.bge_m3 import encode_texts, load_model
from smve_lab.config import EncodeConfig, resolve_device
from smve_lab.datasets import DATASETS, emb_dir, load_texts
from smve_lab.storage import load_separate, save_embeddings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    args = p.parse_args()

    cfg = EncodeConfig()
    out = emb_dir(args.dataset)
    print(f"device: {resolve_device(cfg.device)} -> {out}")
    doc_ids, doc_texts, query_ids, query_texts = load_texts(args.dataset)
    print(f"{len(doc_ids)} docs, {len(query_ids)} queries")
    model = load_model(cfg)

    for name, ids, texts in (("docs", doc_ids, doc_texts), ("queries", query_ids, query_texts)):
        t0 = time.time()
        emb = encode_texts(model, texts, cfg)
        print(f"encoded {name} in {time.time() - t0:.1f}s · colbert {emb['colbert_flat'].nbytes / 1e9:.2f} GB")
        save_embeddings(emb, ids, out, name, combined=False)

    emb, ids = load_separate(out, "docs")
    assert ids == doc_ids and emb["dense"].shape[0] == len(doc_ids)
    print(f"OK, saved to {out}")


if __name__ == "__main__":
    main()
