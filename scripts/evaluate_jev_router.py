"""Stage 4b, part 2: does Jev make a better router than first-stage features?

Same queries, same folds, same metrics as router_baselines.py.

Routers compared:
  zero-shot Jev (no training at all). Directions fixed BEFORE looking at results:
      Jev B: "a lower candidate is better"   escalate when P(yes) is high
      Jev B: "top-1 is NOT a direct match"   escalate when 1 - P(top-1 direct) is high
      Jev A: "query needs reasoning"         escalate when P(yes) is high
      Jev A: "query is ambiguous"            escalate when P(yes) is high
  trained (pooled 5-fold CV, and SciFact-train -> others transfer):
      logistic / boosted on first-stage features   (the stage-4a baselines)
      logistic on Jev answers only                 (A: 3 answers, B: 6 answers)
      logistic / boosted on first-stage features + Jev B answers

Jev's own latency is charged to every query it routes:
  A (query only) runs alongside the base pipeline: adds max(0, Jev - base) ms
  B (query + top 3) must wait for the first stage: adds the full Jev latency

    uv run python scripts/evaluate_jev_router.py

Writes results/router/{jev_router.md, jev_router.json, plots/jev_*.png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from router_baselines import FEATURES, TEST_SETS, ece, evaluate, models  # noqa: E402

from smve_lab.config import RESULTS_DIR  # noqa: E402
from smve_lab.datasets import info  # noqa: E402
from smve_lab.plots import plot_jev_auc, plot_jev_calibration, plot_jev_curves  # noqa: E402

OUT = RESULTS_DIR / "router"
JEV_A = ["A_q_ambiguous", "A_q_needs_reasoning", "A_q_specific_terms"]
JEV_B = ["B_q_ambiguous", "B_q_needs_reasoning", "B_q_specific_terms",
         "B_c_top1_direct", "B_c_other_better", "B_c_any_direct"]
ZERO_SHOT = {  # name: (function of the merged frame -> escalation score, state used)
    "Jev B: lower candidate better": (lambda d: d["B_c_other_better"], "B"),
    "Jev B: top-1 not direct": (lambda d: 1 - d["B_c_top1_direct"], "B"),
    "Jev A: needs reasoning": (lambda d: d["A_q_needs_reasoning"], "A"),
    "Jev A: ambiguous": (lambda d: d["A_q_ambiguous"], "A"),
}
TRAINED = {  # name: (feature columns, model kind, state used for latency)
    "first-stage logistic": (FEATURES, "logistic", None),
    "first-stage boosted": (FEATURES, "boosted", None),
    "Jev A logistic": (JEV_A, "logistic", "A"),
    "Jev B logistic": (JEV_B, "logistic", "B"),
    "first-stage + Jev B logistic": (FEATURES + JEV_B, "logistic", "B"),
    "first-stage + Jev B boosted": (FEATURES + JEV_B, "boosted", "B"),
}


def main() -> None:
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(OUT / "data.csv", dtype={"query_id": str})
    jev = pd.read_csv(OUT / "jev_answers.csv", dtype={"query_id": str})
    data = data.merge(jev, on=["dataset", "split", "query_id"], how="left", validate="one_to_one")
    assert data[JEV_B].notna().all().all(), "missing Jev answers"
    lat_all = json.loads((OUT / "latency.json").read_text())
    jev_ms = {"A": float(data["A_latency_ms"].median()), "B": float(data["B_latency_ms"].median())}
    price = 0.042 / 1e6
    y = data["y"].to_numpy()
    rng = np.random.default_rng(0)

    # Trained routers: identical folds to router_baselines.py (same data order, same seed).
    strata = data["dataset"] + "_" + data["y"].astype(str)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(data, strata))
    tr_mask = ((data.dataset == "scifact") & (data.split == "train")).to_numpy()
    oof, transfer = {}, {}
    for name, (cols, kind, _) in TRAINED.items():
        X = data[cols].to_numpy()
        oof[name] = np.zeros(len(data))
        for tr, te in folds:
            oof[name][te] = models()[kind].fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        transfer[name] = models()[kind].fit(X[tr_mask], y[tr_mask]).predict_proba(X)[:, 1]

    def overhead(state, base_ms):
        if state is None:
            return 0.0
        return max(0.0, jev_ms["A"] - base_ms) if state == "A" else jev_ms["B"]

    results = {"jev_latency_ms": jev_ms, "cost_usd_all_queries": float((data.A_tokens.sum() + data.B_tokens.sum()) * price),
               "A_pooled_cv": {}, "B_train_on_scifact": {}}
    for ds, split in TEST_SETS:
        m = ((data.dataset == ds) & (data.split == split)).to_numpy()
        df = data[m].reset_index(drop=True)
        lat = lat_all[f"{ds}/{split}"]
        zs = {k: f(df).to_numpy() for k, (f, _) in ZERO_SHOT.items()}
        for proto, trained in (("A_pooled_cv", oof), ("B_train_on_scifact", transfer)):
            r = evaluate(df, {**zs, **{k: v[m] for k, v in trained.items()}}, lat, rng)
            for k in r["routers"]:
                state = ZERO_SHOT[k][1] if k in ZERO_SHOT else (TRAINED[k][2] if k in TRAINED else None)
                r["routers"][k]["overhead_ms"] = overhead(state, lat["base_ms"])
                if k in ZERO_SHOT or k in TRAINED:
                    p = zs[k] if k in ZERO_SHOT else trained[k][m]
                    r["routers"][k]["ece"] = ece(np.clip(p, 0, 1), df["y"].to_numpy())
            results[proto][ds] = r
    (OUT / "jev_router.json").write_text(json.dumps(results, indent=2))

    # --- Report ---------------------------------------------------------------------------
    lines = ["# Stage 4b: Jev as the router\n",
             f"Jev `jev-1.13.0`; median latency per call {jev_ms['A']:.0f} ms (query only) / {jev_ms['B']:.0f} ms "
             f"(query + top 3), measured from the laptop. Cost for all 2,838 queries × 2 calls: "
             f"${results['cost_usd_all_queries']:.3f}. Results on held-out queries only; zero-shot Jev routers are "
             "never trained. 'gain kept': 0 = random routing, 1 = oracle. Latency at a budget includes Jev's "
             "overhead (A: max(0, Jev − base), B: full Jev call).\n"]
    for proto, title in (("A_pooled_cv", "A. pooled 5-fold cross-validation"),
                         ("B_train_on_scifact", "B. trained on SciFact train only (transfer)")):
        lines.append(f"\n## {title}\n")
        for ds, _ in TEST_SETS:
            r = results[proto][ds]
            lines.append(f"\n### {info(ds).display} ({r['n']} queries; escalation helps on {r['helped_share']:.0%}) · "
                         f"never {r['never']:.4f} · always {r['always']:.4f} · oracle best {r['oracle_best']:.4f}\n")
            lines.append("| router | AUC [95% CI] | gain kept | nDCG@10 @20% | latency @20% | ECE |")
            lines.append("|---|---|---|---|---|---|")
            for name, v in r["routers"].items():
                if name == "oracle" or name.startswith("heuristic"):
                    continue
                auc = f"{v['auc']:.3f} [{v['auc_ci95'][0]:.3f}, {v['auc_ci95'][1]:.3f}]" if "auc" in v else "–"
                l20 = r["base_ms"] + v.get("overhead_ms", 0) + 0.2 * r["escalate_ms"]
                e = f"{v['ece']:.3f}" if "ece" in v else "–"
                lines.append(f"| {name} | {auc} | {v['gain_kept']:+.3f} | {v['ndcg@20%']:.4f} | {l20:,.0f} ms | {e} |")
    # Paired bootstrap on the same queries: is the AUC difference real?
    from sklearn.metrics import roc_auc_score
    zs_all = {k: f(data).to_numpy() for k, (f, _) in ZERO_SHOT.items()}
    pairs = [("scifact", "Jev B: lower candidate better", zs_all, "first-stage boosted", oof, "pooled CV"),
             ("scifact", "first-stage + Jev B boosted", transfer, "first-stage boosted", transfer, "SciFact-trained"),
             ("nfcorpus", "Jev A: needs reasoning", zs_all, "first-stage logistic", oof, "pooled CV"),
             ("arguana", "Jev B: lower candidate better", zs_all, "first-stage boosted", oof, "pooled CV")]
    lines.append("\n## Paired comparisons (bootstrap over the same queries, 5,000 resamples)\n")
    lines.append("| test set | router A | router B | protocol | ΔAUC (A − B) [95% CI] | p |\n|---|---|---|---|---|---|")
    brng = np.random.default_rng(1)
    for ds, a, sa, b, sb, proto in pairs:
        m = ((data.dataset == ds) & (data.split == "test")).to_numpy()
        yy, pa, pb = y[m], sa[a][m], sb[b][m]
        diffs = []
        for _ in range(5000):
            i = brng.integers(0, len(yy), len(yy))
            if 0 < yy[i].sum() < len(i):
                diffs.append(roc_auc_score(yy[i], pa[i]) - roc_auc_score(yy[i], pb[i]))
        diffs = np.array(diffs)
        obs = roc_auc_score(yy, pa) - roc_auc_score(yy, pb)
        pval = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        lines.append(f"| {info(ds).display} | {a} | {b} | {proto} | {obs:+.3f} "
                     f"[{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}] | {pval:.3f} |")
    text = "\n".join(lines)
    (OUT / "jev_router.md").write_text(text + "\n")
    print(text)

    show = ["oracle", "random", "first-stage boosted", "first-stage logistic", "Jev B: lower candidate better",
            "Jev B logistic", "first-stage + Jev B logistic"]
    plot_jev_curves(results, show, OUT / "plots" / "jev_routing_curves.png")
    plot_jev_auc(results, [k for k in list(ZERO_SHOT) + list(TRAINED)], OUT / "plots" / "jev_auc.png")
    plot_jev_calibration(data, {"Jev B: lower candidate better (raw P(yes))": data["B_c_other_better"].to_numpy(),
                                "first-stage + Jev B logistic (CV)": oof["first-stage + Jev B logistic"],
                                "first-stage boosted (CV)": oof["first-stage boosted"]},
                         OUT / "plots" / "jev_calibration.png")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
