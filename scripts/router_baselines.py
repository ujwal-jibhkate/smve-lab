"""Stage 4a, part 2: baseline routers. When is the cross-encoder worth calling?

Every router gives each query a score: "how likely is escalating to the
cross-encoder to help?". Escalating the top p% of queries by that score traces a
quality-vs-cost curve, from p = 0 (never: base pipeline only) to p = 100%
(always escalate). A better router reaches higher quality at the same cost.

Routers:
  oracle      knows the true gain (upper bound, not achievable)
  random      escalates a random p% (the straight line between never and always)
  heuristics  one first-stage signal each, e.g. "small gap between the top-2
              scores" = the first stage is unsure
  logistic    logistic regression on all first-stage features
  boosted     gradient-boosted trees on the same features

Two protocols, both reporting results only on queries the model did NOT train on:
  A  pooled 5-fold cross-validation over all 2,838 queries (same distribution)
  B  train on SciFact train only, test on SciFact test / NFCorpus / ArguAna
     (does a router learned on one dataset transfer to others?)

Metrics per test set:
  AUC        P(router scores a helped query above a not-helped one); 0.5 = random
  gain kept  (router area - random area) / (oracle area - random area) under the
             quality-vs-escalation curve; 0 = random, 1 = oracle
  nDCG@10 and average latency when escalating 10 / 20 / 30 % of queries
  calibration (ECE): when the router says 30 %, does it help 30 % of the time?

    uv run python scripts/router_baselines.py

Writes results/router/{baselines.md, baselines.json, oof_scores.csv, plots/}.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from smve_lab.config import RESULTS_DIR
from smve_lab.datasets import info
from smve_lab.plots import plot_router_calibration, plot_router_curves, plot_router_coefficients

OUT = RESULTS_DIR / "router"
FEATURES = ["hyb_top1", "hyb_margin12", "hyb_gap1_10", "hyb_std10", "dense_top1", "dense_margin12",
            "dense_gap1_10", "lex_top1", "lex_margin12", "lex_gap1_10", "dense_lex_jaccard10",
            "dense_top1_in_lex10", "lex_top1_in_dense10", "hyb_top1_is_dense_top1", "q_tokens", "q_lex_terms"]
HEURISTICS = {  # score = higher means "escalate"; each is one first-stage signal
    "heuristic: small top-2 gap": lambda d: -d["hyb_margin12"],
    "heuristic: dense/lexical disagree": lambda d: -d["dense_lex_jaccard10"],
}
TEST_SETS = [("scifact", "test"), ("nfcorpus", "test"), ("arguana", "test")]
BUDGETS = [0.1, 0.2, 0.3]


def models():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "boosted": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                  min_samples_leaf=30, random_state=0),
    }


def curve(gain: np.ndarray, score: np.ndarray, base: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean nDCG@10 when escalating the top n queries by score, for n = 0..N."""
    order = np.argsort(-score, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(gain[order])])
    frac = np.arange(len(gain) + 1) / len(gain)
    return frac, base + cum / len(gain)


def evaluate(df: pd.DataFrame, scores: dict[str, np.ndarray], lat: dict, rng: np.random.Generator) -> dict:
    gain = df["gain"].to_numpy()
    base = df["ndcg_base"].mean()
    _, oracle = curve(gain, gain, base)
    frac, rand = curve(gain, np.zeros_like(gain), base)
    rand = base + frac * gain.mean()  # expected value of random routing
    area = lambda y: np.trapezoid(y - base, frac)
    out = {"n": len(df), "helped_share": float(df["y"].mean()), "never": base,
           "always": float(df["ndcg_escalate"].mean()), "oracle_best": float(oracle.max()),
           "base_ms": lat["base_ms"], "escalate_ms": lat["escalate_extra_ms"], "routers": {}}
    for name, s in {"oracle": gain, "random": None, **scores}.items():
        r = {}
        y = rand if s is None else curve(gain, s, base)[1]
        if s is not None and name != "oracle" and 0 < df["y"].sum() < len(df):
            r["auc"] = float(roc_auc_score(df["y"], s))
            boots = []
            for _ in range(1000):
                i = rng.integers(0, len(df), len(df))
                if 0 < df["y"].iloc[i].sum() < len(i):
                    boots.append(roc_auc_score(df["y"].iloc[i], s[i]))
            r["auc_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        r["gain_kept"] = float((area(y) - area(rand)) / (area(oracle) - area(rand)))
        for b in BUDGETS:
            n = int(round(b * len(df)))
            r[f"ndcg@{int(b * 100)}%"] = float(y[n])
        r["latency_at_budget_ms"] = {f"{int(b * 100)}%": lat["base_ms"] + b * lat["escalate_extra_ms"] for b in BUDGETS}
        r["curve"] = {"frac": frac[:: max(1, len(frac) // 200)].tolist(), "ndcg": y[:: max(1, len(frac) // 200)].tolist()}
        out["routers"][name] = r
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main() -> None:
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(OUT / "data.csv", dtype={"query_id": str})
    lat_all = json.loads((OUT / "latency.json").read_text())
    X, y = data[FEATURES].to_numpy(), data["y"].to_numpy()
    rng = np.random.default_rng(0)

    # Protocol A: pooled 5-fold CV, stratified by dataset and label -> out-of-fold probabilities.
    strata = data["dataset"] + "_" + data["y"].astype(str)
    oof = {m: np.zeros(len(data)) for m in models()}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, strata):
        for name, model in models().items():
            oof[name][te] = model.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

    # Protocol B: train on SciFact train only.
    tr = (data.dataset == "scifact") & (data.split == "train")
    transfer = {name: model.fit(X[tr], y[tr]).predict_proba(X)[:, 1] for name, model in models().items()}
    coef = models()["logistic"].fit(X, y)[-1].coef_[0]

    results = {"A_pooled_cv": {}, "B_train_on_scifact": {}}
    for ds, split in TEST_SETS:
        m = ((data.dataset == ds) & (data.split == split)).to_numpy()
        df = data[m].reset_index(drop=True)
        lat = lat_all[f"{ds}/{split}"]
        heur = {k: f(df).to_numpy() for k, f in HEURISTICS.items()}
        results["A_pooled_cv"][ds] = evaluate(df, {**heur, **{k: v[m] for k, v in oof.items()}}, lat, rng)
        results["B_train_on_scifact"][ds] = evaluate(df, {**heur, **{k: v[m] for k, v in transfer.items()}}, lat, rng)
        for k in oof:
            results["A_pooled_cv"][ds]["routers"][k]["ece"] = ece(oof[k][m], df["y"].to_numpy())
            results["B_train_on_scifact"][ds]["routers"][k]["ece"] = ece(transfer[k][m], df["y"].to_numpy())

    pd.DataFrame({"dataset": data.dataset, "split": data.split, "query_id": data.query_id, "y": y,
                  "gain": data.gain, **{f"oof_{k}": v for k, v in oof.items()},
                  **{f"transfer_{k}": v for k, v in transfer.items()}}).to_csv(OUT / "oof_scores.csv", index=False)
    (OUT / "baselines.json").write_text(json.dumps(results, indent=2))

    # --- Report ------------------------------------------------------------------------------
    lines = ["# Stage 4a: baseline routers (when to escalate to the cross-encoder)\n",
             "Base = dense+lexical -> MaxSim@10; escalate = cross-encoder over the top 20. Results on held-out "
             "queries only. gain kept: 0 = random routing, 1 = oracle.\n"]
    for proto, title in (("A_pooled_cv", "A. pooled 5-fold cross-validation"),
                         ("B_train_on_scifact", "B. trained on SciFact train only (transfer)")):
        lines.append(f"\n## {title}\n")
        for ds, _ in TEST_SETS:
            r = results[proto][ds]
            lines.append(f"\n### {info(ds).display} ({r['n']} queries; escalation helps on {r['helped_share']:.0%})\n")
            lines.append(f"never {r['never']:.4f} at {r['base_ms']:.0f} ms · always {r['always']:.4f} at "
                         f"{r['base_ms'] + r['escalate_ms']:,.0f} ms · oracle best {r['oracle_best']:.4f}\n")
            lines.append("| router | AUC [95% CI] | gain kept | nDCG@10 @10% | @20% | @30% | ECE |")
            lines.append("|---|---|---|---|---|---|---|")
            for name, v in r["routers"].items():
                auc = f"{v['auc']:.3f} [{v['auc_ci95'][0]:.3f}, {v['auc_ci95'][1]:.3f}]" if "auc" in v else "–"
                lines.append(f"| {name} | {auc} | {v['gain_kept']:+.3f} | {v['ndcg@10%']:.4f} | {v['ndcg@20%']:.4f} | "
                             f"{v['ndcg@30%']:.4f} | {v.get('ece', float('nan')):.3f} |")
        lat = results[proto]["scifact"]["routers"]["oracle"]["latency_at_budget_ms"]
    lines.append("\nAverage latency at an escalation budget = base + budget × cross-encoder time, e.g. SciFact: "
                 + ", ".join(f"{k}: {v:,.0f} ms" for k, v in lat.items()) + ".")
    lines.append("\n## Logistic-regression coefficients (standardised features, all data)\n")
    lines.append("| feature | coefficient |\n|---|---|")
    for f, c in sorted(zip(FEATURES, coef), key=lambda t: -abs(t[1])):
        lines.append(f"| {f} | {c:+.3f} |")
    text = "\n".join(lines)
    (OUT / "baselines.md").write_text(text + "\n")
    print(text)

    plot_router_curves(results, data, lat_all, OUT / "plots" / "routing_curves.png")
    plot_router_calibration(data, oof, OUT / "plots" / "calibration.png")
    plot_router_coefficients(FEATURES, coef, OUT / "plots" / "logistic_coefficients.png")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
