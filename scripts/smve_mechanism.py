"""Step 1a: WHY does SMVE lose accuracy? Measure it on single token pairs.

For one query token x and one doc token y, SMVE's contribution to the score is

    S(x, y) = sum over anchors a in topk(x) AND topk(y) of  (x.b_a) * (y.b_a)

(with one token, "sum" for queries and "mean" for docs are both just the
value). MaxSim would use the true cosine x.y instead. Two things go wrong:

  A. coverage: the two top-k anchor sets don't overlap, so S = 0 exactly even
     though x and y may be very similar. Measured as the fraction of pairs
     with S == 0, per true-cosine level.
  B. variance: when they do overlap, how many anchors they share is random,
     so S scatters around its mean. Measured as the coefficient of variation
     (std / mean) of the non-zero S values, per true-cosine level.
  Overall fidelity: Spearman correlation between S and the true cosine.

Pairs come from four settings:
  - synthetic, d=128   (ColBERTv2-sized; TopK's example uses w=2048, k=8)
  - synthetic, d=1024  (BGE-M3-sized)
  - real BGE-M3 SciFact token pairs, as stored
  - the same real pairs with mean-centering (the --center option)
Synthetic pairs at exact cosine c: y = c*x + sqrt(1 - c^2)*u, u orthogonal to x.

For each setting we compare, at the same budget multiplier m in {1, 2, 4, 8}:
  - repetitions:   block width w, k per block, R = m blocks
  - single matrix: width m*w, top-(m*k), R = 1
Both use m*w anchors and keep m*k non-zeros per token.

    uv run python scripts/smve_mechanism.py

Writes results/smve_mechanism/{per_level.csv, summary.csv, meta.json, plots/}.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from smve_lab.config import ARTIFACTS_DIR, RESULTS_DIR
from smve_lab.plots import plot_mechanism_curves, plot_mechanism_summary
from smve_lab.smve import make_anchors, token_mean, topk_blocks
from smve_lab.storage import load_separate

OUT = RESULTS_DIR / "smve_mechanism"
LEVELS = np.round(np.arange(0.2, 0.91, 0.1), 2)
PAIRS_PER_LEVEL = 1000
BUDGETS = [1, 2, 4, 8]
HIGH_COS = 0.7  # "strong match" threshold used in the summary


def synthetic_pairs(d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, cs = [], [], []
    for c in LEVELS:
        x = rng.standard_normal((PAIRS_PER_LEVEL, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        u = rng.standard_normal((PAIRS_PER_LEVEL, d))
        u -= (u * x).sum(1, keepdims=True) * x  # remove the x component
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        xs.append(x)
        ys.append(c * x + np.sqrt(1 - c**2) * u)
        cs.append(np.full(PAIRS_PER_LEVEL, c))
    return np.vstack(xs).astype(np.float32), np.vstack(ys).astype(np.float32), np.concatenate(cs)


def real_pairs(tokens: np.ndarray, rng: np.random.Generator, half_width: float = 0.025):
    """Sample real token pairs whose true cosine falls within +-half_width of each level."""
    T = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
    G = T @ T.T
    iu = np.triu_indices(len(T), k=1)
    cos = G[iu]
    xs, ys, cs = [], [], []
    for c in LEVELS:
        hit = np.flatnonzero(np.abs(cos - c) <= half_width)
        pick = rng.choice(hit, size=min(PAIRS_PER_LEVEL, len(hit)), replace=False)
        xs.append(T[iu[0][pick]])
        ys.append(T[iu[1][pick]])
        cs.append(cos[pick])
    return np.vstack(xs), np.vstack(ys), np.concatenate(cs)


def pair_scores(X: np.ndarray, Y: np.ndarray, B: torch.Tensor, k: int, reps: int,
                chunk: int = 512) -> np.ndarray:
    """SMVE estimate S(x, y) for each row pair."""
    out = []
    for i in range(0, len(X), chunk):
        Px = torch.from_numpy(X[i:i + chunk]) @ B
        Py = torch.from_numpy(Y[i:i + chunk]) @ B
        vx, ix = topk_blocks(Px, k, reps)
        vy, iy = topk_blocks(Py, k, reps)
        sx = torch.zeros_like(Px).scatter_(1, ix, vx)
        sy = torch.zeros_like(Py).scatter_(1, iy, vy)
        out.append((sx * sy).sum(1).numpy())
    return np.concatenate(out)


def main() -> None:
    rng = np.random.default_rng(0)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    # Real BGE-M3 tokens: 6000 random SciFact document tokens.
    doc_emb, _ = load_separate(ARTIFACTS_DIR / "embeddings" / "scifact_bgem3", "docs")
    flat = doc_emb["colbert_flat"]
    idx = np.sort(rng.choice(len(flat), size=6000, replace=False))
    real_x, real_y, real_c = real_pairs(np.asarray(flat[idx], dtype=np.float32), rng)
    mu = token_mean(flat)

    def centered(A: np.ndarray) -> np.ndarray:
        A = A - mu
        return (A / np.linalg.norm(A, axis=1, keepdims=True)).astype(np.float32)

    settings = {
        "synthetic, d=128": (*synthetic_pairs(128, rng), 128, 2048, 8),
        "synthetic, d=1024": (*synthetic_pairs(1024, rng), 1024, 4096, 8),
        "BGE-M3 tokens": (real_x, real_y, real_c, 1024, 4096, 8),
        "BGE-M3 tokens, centered": (centered(real_x), centered(real_y), real_c, 1024, 4096, 8),
    }

    per_level, summary = [], []
    for name, (X, Y, true_cos, d, w, k) in settings.items():
        level = np.array([LEVELS[np.argmin(np.abs(LEVELS - c))] for c in true_cos])
        for m in BUDGETS:
            variants = [("repetitions", w, k, m)] + ([("single matrix", w * m, k * m, 1)] if m > 1 else [])
            for mode, bw, bk, R in variants:
                B = make_anchors(d, bw * R, seed=1000 + m)
                S = pair_scores(X, Y, B, bk, R)
                for c in LEVELS:
                    s = S[level == c]
                    nz = s[s != 0]
                    per_level.append({
                        "setting": name, "mode": mode, "budget": m, "w": bw, "k": bk, "reps": R,
                        "cosine": c, "n": len(s), "zero_fraction": float((s == 0).mean()),
                        "cv_nonzero": float(nz.std() / nz.mean()) if len(nz) > 1 else np.nan,
                        "mean_score": float(s.mean()),
                    })
                high = true_cos >= HIGH_COS - 0.025
                summary.append({
                    "setting": name, "mode": mode, "budget": m, "w": bw, "k": bk, "reps": R,
                    "spearman": float(spearmanr(S, true_cos).statistic),
                    "pearson": float(pearsonr(S, true_cos).statistic),
                    "zero_fraction_all": float((S == 0).mean()),
                    f"zero_fraction_cos>={HIGH_COS}": float((S[high] == 0).mean()),
                })
                print(f"{name:26s} {mode:13s} m={m} (w={bw}, k={bk}, R={R}): "
                      f"spearman={summary[-1]['spearman']:.3f}  "
                      f"zero@cos>={HIGH_COS}={summary[-1][f'zero_fraction_cos>={HIGH_COS}']:.3f}", flush=True)

    per_level = pd.DataFrame(per_level)
    summary = pd.DataFrame(summary)
    per_level.to_csv(OUT / "per_level.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "levels": LEVELS.tolist(), "pairs_per_level": PAIRS_PER_LEVEL, "budgets": BUDGETS,
        "real_tokens": "6000 random BGE-M3 ColBERT doc tokens from SciFact; pairs binned at level +-0.025",
        "settings": {n: {"d": v[3], "base_w": v[4], "base_k": v[5]} for n, v in settings.items()},
    }, indent=2))

    plot_mechanism_curves(per_level, "zero_fraction", OUT / "plots" / "A_zero_fraction.png",
                          "Failure mode A: pairs whose SMVE score is exactly 0",
                          "fraction of pairs with S = 0")
    plot_mechanism_curves(per_level, "cv_nonzero", OUT / "plots" / "B_variance.png",
                          "Failure mode B: spread of the non-zero scores",
                          "std / mean of non-zero S")
    plot_mechanism_summary(summary, HIGH_COS, OUT / "plots" / "summary_vs_budget.png")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
