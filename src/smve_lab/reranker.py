"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

A cross-encoder reads the query and one document TOGETHER as a single input
("[CLS] query [SEP] document [SEP]") through a full transformer and outputs one
relevance score. Unlike MaxSim or SMVE nothing is precomputed per document: all
the work happens at query time, once per (query, document) pair. That makes it
accurate but expensive, so it's only used to re-order a first stage's top-d.

We call the Hugging Face model directly instead of FlagReranker.compute_score:
that method runs its first batch twice to probe the batch size, which would
double the measured latency of a single query.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class CrossEncoder:
    def __init__(self, name: str = MODEL_NAME, device: str = "mps", fp16: bool = True, max_length: int = 512):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if fp16 and device != "cpu" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        cfg = self.model.config
        self.n_layers, self.hidden = cfg.num_hidden_layers, cfg.hidden_size
        emb = sum(p.numel() for n, p in self.model.named_parameters() if "embeddings" in n)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.n_params_non_embedding = self.n_params - emb
        self.weight_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

    # "longest_first" trims whichever text is longer. With a short query that is
    # always the document (same as truncating only the document), but it also
    # handles queries that are themselves near the limit (some ArguAna arguments).
    def _encode(self, queries: list[str], docs: list[str]):
        return self.tokenizer(queries, docs, truncation="longest_first", max_length=self.max_length,
                              padding=True, return_tensors="pt")

    def token_lengths(self, queries: list[str], docs: list[str]) -> np.ndarray:
        """Number of tokens in each (query, doc) input, after truncation."""
        enc = self.tokenizer(queries, docs, truncation="longest_first", max_length=self.max_length)
        return np.array([len(ids) for ids in enc["input_ids"]])

    @torch.no_grad()
    def score(self, queries: list[str], docs: list[str], batch_size: int = 32) -> np.ndarray:
        """Relevance logits for each (query, doc) pair. Pairs are sorted by length
        so each batch pads as little as possible, then put back in order."""
        lengths = [len(q) + len(d) for q, d in zip(queries, docs)]
        order = np.argsort(lengths)[::-1]
        out = np.empty(len(queries), dtype=np.float32)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = self._encode([queries[i] for i in idx], [docs[i] for i in idx]).to(self.device)
            out[idx] = self.model(**enc).logits.view(-1).float().cpu().numpy()  # .cpu() waits for the GPU
        return out

    def flops_per_pair(self, n_tokens: np.ndarray) -> np.ndarray:
        """Approximate forward FLOPs for inputs of n tokens.

        Weight matmuls: 2 FLOPs per non-embedding parameter per token.
        Attention:      QK^T and (softmax)V are each 2*n^2*h per layer -> 4*L*n^2*h.
        (Embedding lookups are table reads, not matmuls, so they're excluded.)
        """
        n = np.asarray(n_tokens, dtype=np.float64)
        return 2 * self.n_params_non_embedding * n + 4 * self.n_layers * n**2 * self.hidden
