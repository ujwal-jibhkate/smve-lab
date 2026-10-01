"""Stage 3c report: every method on every dataset, and the "headroom" question (Q2).

Headroom = how much better exhaustive MaxSim is than a cheap single-vector
method on a dataset. It's the most any MaxSim approximation (SMVE, MUVERA)
could add over that cheap method. Q2 asks whether SMVE's standing improves
when headroom is large.

To avoid picking the best setting on the test set, SMVE and MUVERA are each
reported at ONE fixed setting chosen beforehand on SciFact:
  SMVE   w=65536 k=32 centered   (the notebook setting, centered)
  MUVERA R=40 k_sim=4 d_proj=32  (best quality under 10 ms on SciFact)
The best setting per dataset is also shown, marked as optimistic.

    uv run python scripts/cross_dataset_summary.py

Writes results/cross_dataset/{table.csv, summary.md, plots/}.
"""

from __future__ import annotations

import pandas as pd

from smve_lab.config import RESULTS_DIR
from smve_lab.datasets import DATASETS, info, results_dir
from smve_lab.plots import plot_cross_dataset

OUT = RESULTS_DIR / "cross_dataset"
FIXED = {"SMVE": "w65536_k32_center_seed0", "MUVERA": "r40_k4_p32_seed0"}
REF_ORDER = ["Exhaustive MaxSim", "BM25", "BGE-M3 dense", "BGE-M3 lexical", "BGE-M3 dense + lexical"]


def main() -> None:
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    rows = []
    for name in DATASETS:
        f = results_dir(name) / "first_stage" / "all_runs.csv"
        if not f.exists():
            print(f"(skipping {name}: no first_stage/all_runs.csv yet)")
            continue
        df = pd.read_csv(f)
        disp = info(name).display
        for ref in REF_ORDER:
            r = df[(df.family == "reference") & (df.name == ref)].iloc[0]
            rows.append({"dataset": disp, "method": ref, "ndcg@10": r["ndcg@10"], "recall@100": r["recall@100"],
                         "rerank_ndcg@10": r["rerank_ndcg@10"], "latency_ms": r["single_query_latency_ms"],
                         "index_mb": r["index_mb"]})
        for fam, run in FIXED.items():
            g = df[df.family == fam]
            for label, r in ((f"{fam} (fixed setting)", g[g.run == run]),
                             (f"{fam} (best on this dataset, optimistic)", g.loc[[g["rerank_ndcg@10"].idxmax()]])):
                r = r.iloc[0]
                rows.append({"dataset": disp, "method": label, "ndcg@10": r["ndcg@10"], "recall@100": r["recall@100"],
                             "rerank_ndcg@10": r["rerank_ndcg@10"], "latency_ms": r["single_query_latency_ms"],
                             "index_mb": r["index_mb"], "setting": r["run"].replace("_seed0", "")})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table.csv", index=False)

    # Headroom per dataset and where SMVE / MUVERA land relative to it.
    head = []
    for disp, g in table.groupby("dataset", sort=False):
        v = g.set_index("method")["ndcg@10"]
        rr = g.set_index("method")["rerank_ndcg@10"]
        head.append({
            "dataset": disp, "MaxSim": v["Exhaustive MaxSim"],
            "headroom vs dense": v["Exhaustive MaxSim"] - v["BGE-M3 dense"],
            "headroom vs dense+lexical": v["Exhaustive MaxSim"] - v["BGE-M3 dense + lexical"],
            "headroom vs BM25": v["Exhaustive MaxSim"] - v["BM25"],
            "SMVE − dense (alone)": v["SMVE (fixed setting)"] - v["BGE-M3 dense"],
            "MUVERA − dense (alone)": v["MUVERA (fixed setting)"] - v["BGE-M3 dense"],
            "SMVE→MaxSim − dense+lex→MaxSim": rr["SMVE (fixed setting)"] - rr["BGE-M3 dense + lexical"],
            "MUVERA→MaxSim − dense+lex→MaxSim": rr["MUVERA (fixed setting)"] - rr["BGE-M3 dense + lexical"],
            "best first stage →MaxSim": rr.idxmax(),
        })
    head = pd.DataFrame(head)
    head.to_csv(OUT / "headroom.csv", index=False)

    def md(df: pd.DataFrame) -> str:
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(f"{x:+.4f}" if isinstance(x, float) and ("−" in c or "headroom" in c)
                                           else (f"{x:.4f}" if isinstance(x, float) else str(x))
                                           for c, x in zip(df.columns, r)) + " |")
        return "\n".join(lines)

    wide = {m: table.pivot_table(index="method", columns="dataset", values=m, sort=False)
            for m in ("ndcg@10", "recall@100", "rerank_ndcg@10")}
    text = "\n".join([
        f"# All methods across datasets ({', '.join(table.dataset.unique())})\n",
        "SMVE / MUVERA 'fixed setting' = chosen on SciFact beforehand; 'best' = picked on each dataset's test set "
        "(optimistic). '→MaxSim' = exact MaxSim rerank of the first stage's top 100.\n",
        "## nDCG@10 (first stage alone)\n", md(wide["ndcg@10"].reset_index()),
        "\n## Recall@100\n", md(wide["recall@100"].reset_index()),
        "\n## nDCG@10 after MaxSim rerank of the top 100\n", md(wide["rerank_ndcg@10"].reset_index()),
        "\n## Headroom (Q2)\n", md(head),
    ])
    (OUT / "summary.md").write_text(text + "\n")
    print(text)
    plot_cross_dataset(table, head, OUT / "plots" / "cross_dataset.png")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
