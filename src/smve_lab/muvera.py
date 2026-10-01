"""MUVERA Fixed Dimensional Encodings (Dhulipala et al., 2024, arXiv:2405.19504).

MUVERA, like SMVE, turns a bag of token vectors into ONE vector whose dot
product approximates MaxSim (the paper calls MaxSim "Chamfer similarity").
The difference is how tokens are grouped:

  SMVE:   soft assignment. Each token joins its top-k of w random anchors.
  MUVERA: hard assignment. k_sim random hyperplanes cut the space into
          B = 2^k_sim regions ("buckets"). Each token lands in exactly ONE
          bucket, decided by which side of each hyperplane it falls on (SimHash).

Per repetition (R independent repetitions, concatenated):
  1. bucket(x) = the k_sim sign bits of (x . g_1, ..., x . g_ksim) read as an integer
  2. psi(x)    = a random +-1 projection of x down to d_proj numbers (shrinks storage)
  3. query block for bucket b = SUM of psi(q) over query tokens in b
     doc   block for bucket b = MEAN of psi(p) over doc tokens in b
     empty doc bucket ("fill empty clusters"): use the doc token whose bucket is
     closest to b in Hamming distance, so a query token in b still finds a match
  4. concatenate B blocks of d_proj -> B*d_proj numbers; R repetitions -> R*B*d_proj

Score = plain dense dot product between the query and doc encodings. Why it
approximates MaxSim: a query token's nearest doc tokens usually share its
bucket, and the dot product of the query sum with the doc mean in that bucket is
the query token's similarity to "its" doc tokens. Same idea as SMVE's pooling
(query sum, doc mean), with hard buckets instead of top-k anchors.

The paper's main setting is R=20, k_sim=5, d_proj=16 -> 20*32*16 = 10,240 dims.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class MuveraFDE:
    def __init__(self, dim: int, reps: int = 20, k_sim: int = 5, d_proj: int = 16, seed: int = 0):
        assert k_sim <= 8, "bucket ids are kept in uint8"
        g = torch.Generator().manual_seed(seed)
        self.reps, self.k_sim, self.d_proj = reps, k_sim, d_proj
        self.n_buckets = 2 ** k_sim
        self.out_dim = reps * self.n_buckets * d_proj
        self.G = torch.randn(dim, reps * k_sim, generator=g)  # SimHash hyperplanes, all reps at once
        signs = torch.randint(0, 2, (dim, reps * d_proj), generator=g).float() * 2 - 1
        self.P = signs / np.sqrt(d_proj)  # +-1/sqrt(d_proj) projection, all reps at once
        self._powers = 2 ** torch.arange(k_sim)

    def _codes_and_proj(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bucket id per (token, rep) and projected token per rep: (T, R) and (T, R, d_proj)."""
        bits = (X @ self.G > 0).view(len(X), self.reps, self.k_sim).long()
        codes = (bits * self._powers).sum(-1)  # (T, R)
        proj = (X @ self.P).view(len(X), self.reps, self.d_proj)
        return codes, proj

    def encode(self, flat: np.ndarray, offsets: np.ndarray, is_query: bool, max_tokens: int = 65_536,
               fill_empty: bool = True, center: np.ndarray | None = None,
               show_progress: bool = True) -> np.ndarray:
        """FDE for every item of a ragged (flat, offsets) token array -> (n_items, out_dim) float32."""
        n_items, R, B, dp = len(offsets) - 1, self.reps, self.n_buckets, self.d_proj
        out = np.zeros((n_items, R, B, dp), dtype=np.float32)
        mu = None if center is None else torch.as_tensor(center, dtype=torch.float32)
        lens = np.diff(offsets)
        pbar = tqdm(total=n_items, desc="MUVERA " + ("queries" if is_query else "docs"), unit="item",
                    disable=not show_progress)
        start = 0
        while start < n_items:
            end = int(np.searchsorted(offsets, offsets[start] + max_tokens, side="right")) - 1
            end = min(max(end, start + 1), n_items)
            X = torch.from_numpy(np.asarray(flat[offsets[start]:offsets[end]], dtype=np.float32))
            if mu is not None:
                X = torch.nn.functional.normalize(X - mu, dim=1)
            codes, proj = self._codes_and_proj(X)
            item = torch.repeat_interleave(torch.arange(end - start), torch.from_numpy(lens[start:end]))
            # flat slot per (item, rep, bucket); sum projected tokens into their slot
            slot = (item[:, None] * R + torch.arange(R)[None, :]) * B + codes  # (T, R)
            sums = torch.zeros((end - start) * R * B, dp).index_add_(0, slot.reshape(-1), proj.reshape(-1, dp))
            counts = torch.bincount(slot.reshape(-1), minlength=(end - start) * R * B).float()
            if not is_query:
                sums /= counts.clamp(min=1)[:, None]
            block = sums.view(end - start, R, B, dp).numpy()

            if fill_empty and not is_query:
                cnt = counts.view(end - start, R, B).numpy()
                codes_np = codes.numpy().astype(np.uint8)
                proj_np = proj.numpy()
                bucket_ids = np.arange(B, dtype=np.uint8)
                pos = 0
                for i in range(end - start):
                    n = lens[start + i]
                    c = codes_np[pos:pos + n]  # (n, R)
                    empty = cnt[i] == 0  # (R, B)
                    if empty.any():
                        ham = _POPCOUNT[c[:, :, None] ^ bucket_ids[None, None, :]]  # (n, R, B)
                        nearest = ham.argmin(axis=0)  # (R, B): token closest to each bucket
                        r_idx, b_idx = np.nonzero(empty)
                        block[i, r_idx, b_idx] = proj_np[pos + nearest[r_idx, b_idx], r_idx]
                    pos += n
            out[start:end] = block
            pbar.update(end - start)
            start = end
        pbar.close()
        return out.reshape(n_items, self.out_dim)
