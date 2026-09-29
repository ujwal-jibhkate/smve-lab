"""End-to-end batch job: encode the scifact corpus + queries with BGE-M3.

Runs as a plain terminal process (not inside the notebook) so a multi-minute
job survives a closed laptop lid or a restarted Jupyter kernel:

    uv run python scripts/encode_scifact.py
"""

from __future__ import annotations

import time

from smve_lab.bge_m3 import encode_texts, load_model
from smve_lab.config import ARTIFACTS_DIR, EncodeConfig, resolve_device
from smve_lab.scifact import load_scifact
from smve_lab.storage import (
    get_colbert,
    get_sparse,
    load_combined,
    load_separate,
    save_embeddings,
)

SAVE_DIR = ARTIFACTS_DIR / "embeddings" / "scifact_bgem3"


def main() -> None:
    cfg = EncodeConfig()
    print(f"device: {resolve_device(cfg.device)}")

    print("loading scifact...")
    _, _, _, doc_ids, doc_texts, query_ids, query_texts = load_scifact()
    print(f"{len(doc_ids)} docs, {len(query_ids)} queries")

    print(f"loading {cfg.model_name}...")
    model = load_model(cfg)

    t0 = time.time()
    doc_emb = encode_texts(model, doc_texts, cfg)
    print(f"encoded docs in {time.time() - t0:.1f}s")
    save_embeddings(doc_emb, doc_ids, SAVE_DIR, "docs")
    print("colbert size (GB):", doc_emb["colbert_flat"].nbytes / 1e9)

    t0 = time.time()
    query_emb = encode_texts(model, query_texts, cfg)
    print(f"encoded queries in {time.time() - t0:.1f}s")
    save_embeddings(query_emb, query_ids, SAVE_DIR, "queries")

    # Round-trip sanity check.
    emb2, ids2 = load_combined(SAVE_DIR, "docs")
    emb3, ids3 = load_separate(SAVE_DIR, "docs")
    assert ids2 == ids3 == doc_ids
    assert emb2["dense"].shape[0] == len(doc_ids)
    print("OK:", emb2["dense"].shape, len(get_sparse(emb2, 0)), get_colbert(emb2, 0).shape)
    print(f"saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
