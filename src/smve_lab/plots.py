"""Evaluation plots with a consistent, minimal style.

Every function takes plain data (DataFrames / arrays) and a path, and writes
one PNG. They know nothing about MaxSim specifically, so SMVE runs can reuse
them unchanged and produce directly comparable figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# Fixed categorical order (colorblind-validated); assigned by entity, not rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METRIC_COLORS = {
    "ndcg": SERIES[0],
    "recall": SERIES[1],
    "precision": SERIES[2],
    "mrr": SERIES[3],
    "map": SERIES[4],
}
METRIC_LABELS = {
    "ndcg": "nDCG",
    "recall": "Recall",
    "precision": "Precision",
    "mrr": "MRR",
    "map": "MAP",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "text.color": TEXT,
            "axes.labelcolor": TEXT_2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
        }
    )


def _title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=TEXT, pad=26)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=10, color=TEXT_2, va="bottom")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_metrics_vs_k(curves: pd.DataFrame, path: Path, run_name: str) -> None:
    """Line chart of mean metric value at each cutoff k.

    `curves` has one row per k (index) and one column per metric name.
    """
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ks = curves.index.to_numpy()
    for metric in curves.columns:
        y = curves[metric].to_numpy()
        ax.plot(ks, y, color=METRIC_COLORS[metric], lw=2, label=METRIC_LABELS[metric])
        # Direct label at the line end so identity isn't color-alone.
        ax.annotate(
            f"{METRIC_LABELS[metric]}  {y[-1]:.3f}",
            (ks[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            color=TEXT,
        )
    for k in (10,):
        ax.axvline(k, color=TEXT_2, lw=1, ls=(0, (3, 3)), zorder=0)
        ax.text(k, 1.0, " k = 10", color=TEXT_2, fontsize=9, va="top")
    ax.set_xscale("log")
    ticks = [k for k in (1, 3, 5, 10, 20, 50, 100) if k <= ks.max()]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(1, ks.max())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("cutoff k (log scale)")
    ax.set_ylabel("mean over queries")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=len(curves.columns))
    _title(ax, "Retrieval quality vs. cutoff k", run_name)
    _save(fig, path)


def plot_per_query_hist(values: pd.Series, path: Path, metric: str, run_name: str) -> None:
    """Histogram of a per-query metric (e.g. nDCG@10) - shows how the mean is made up."""
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bins = np.linspace(0, 1, 21)
    ax.hist(values, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    mean = values.mean()
    ax.axvline(mean, color=TEXT, lw=1.5)
    ax.annotate(
        f"mean {mean:.3f}",
        (mean, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-6, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9.5,
        color=TEXT,
    )
    n_zero = int((values == 0).sum())
    n_one = int((values == 1).sum())
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    ax.set_ylabel("number of queries")
    _title(
        ax,
        f"Per-query {metric}",
        f"{run_name}  ·  {len(values)} queries  ·  {n_one} perfect, {n_zero} zero",
    )
    _save(fig, path)


def plot_first_relevant_rank(ranks: pd.Series, path: Path, depth: int, run_name: str) -> None:
    """Bar chart of where the first relevant doc lands - the failure profile."""
    _style()
    edges = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, depth)]
    labels = [str(a) if a == b else f"{a}–{b}" for a, b in edges] + [f">{depth}"]
    counts = [int(ranks.between(a, b).sum()) for a, b in edges] + [int(ranks.isna().sum())]
    total = len(ranks)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(labels))
    colors = [SERIES[0]] * (len(labels) - 1) + [TEXT_2]
    ax.bar(x, counts, width=0.72, color=colors, edgecolor=SURFACE, linewidth=2)
    for xi, c in zip(x, counts):
        ax.text(xi, c, f"{c}\n{c / total:.0%}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_xlabel("rank of the first relevant document")
    ax.set_ylabel("number of queries")
    _title(ax, "Where does the first relevant document appear?", run_name)
    _save(fig, path)


def plot_score_separation(
    relevant: np.ndarray, non_relevant: np.ndarray, path: Path, run_name: str, xlabel: str
) -> None:
    """Overlaid distributions of scores for relevant vs. non-relevant docs.

    The less these overlap, the easier it is for the scorer to rank the right
    doc on top.
    """
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    lo = min(relevant.min(), non_relevant.min())
    hi = max(relevant.max(), non_relevant.max())
    bins = np.linspace(lo, hi, 50)
    for data, color, label in (
        (non_relevant, SERIES[1], f"non-relevant (n={len(non_relevant):,})"),
        (relevant, SERIES[0], f"relevant (n={len(relevant):,})"),
    ):
        ax.hist(data, bins=bins, density=True, color=color, alpha=0.25)
        ax.hist(data, bins=bins, density=True, histtype="step", color=color, lw=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    _title(ax, "Score separation: relevant vs. non-relevant", run_name)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Comparison plots: several runs side by side. `runs` is an ordered mapping
# {label: ...}; the first entry is the baseline. Colors follow run order.
# ---------------------------------------------------------------------------

DIVERGING_POS = "#2a78d6"  # candidate better than baseline
DIVERGING_NEG = "#e34948"  # candidate worse than baseline


def _run_colors(labels: list[str]) -> dict[str, str]:
    return {label: SERIES[i] for i, label in enumerate(labels)}


def plot_metric_bars(means: pd.DataFrame, path: Path, subtitle: str) -> None:
    """Grouped bars: one group per metric (rows), one bar per run (columns)."""
    _style()
    colors = _run_colors(list(means.columns))
    n_metrics, n_runs = means.shape
    fig, ax = plt.subplots(figsize=(9.5 if n_runs <= 4 else 11, 4.8))
    width = 0.8 / n_runs
    x = np.arange(n_metrics)
    for j, run in enumerate(means.columns):
        xs = x - 0.4 + width * (j + 0.5)
        ax.bar(xs, means[run], width=width, color=colors[run], edgecolor=SURFACE, linewidth=2, label=run)
        for xi, v in zip(xs, means[run]):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", color=TEXT,
                    fontsize=8.5 if n_runs <= 4 else 6.5, rotation=0 if n_runs <= 4 else 90)
    ax.set_xticks(x, means.index)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean over queries")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=min(n_runs, 3))
    _title(ax, "Retrieval quality", subtitle)
    _save(fig, path)


def plot_curves_compare(curves: dict[str, pd.DataFrame], metrics: list[str], path: Path, subtitle: str) -> None:
    """Small multiples: one panel per metric, one line per run, over k."""
    _style()
    colors = _run_colors(list(curves))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.6), sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for run, df in curves.items():
            ax.plot(df.index, df[metric], color=colors[run], lw=2, label=run)
        ax.set_xscale("log")
        ticks = [k for k in (1, 3, 5, 10, 20, 50, 100) if k <= df.index.max()]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlim(1, df.index.max())
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("cutoff k (log scale)")
        ax.set_title(f"{METRIC_LABELS[metric]}@k", loc="left", fontsize=12, fontweight="bold", color=TEXT)
    axes[0].set_ylabel("mean over queries")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=min(len(curves), 2),
               bbox_to_anchor=(0.5, 0))
    fig.suptitle(f"Quality vs. cutoff  ·  {subtitle}", x=0.07, ha="left", fontsize=13,
                 fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    _save(fig, path)


def plot_per_query_delta(deltas: dict[str, pd.Series], path: Path, metric: str, baseline: str) -> None:
    """For each candidate: per-query (candidate - baseline), sorted. Blue = better."""
    _style()
    fig, axes = plt.subplots(len(deltas), 1, figsize=(9.5, 3.2 * len(deltas)), squeeze=False)
    for ax, (run, d) in zip(axes[:, 0], deltas.items()):
        d = d.sort_values(ascending=False).to_numpy()
        x = np.arange(len(d))
        ax.bar(x, d, width=1.0, color=np.where(d >= 0, DIVERGING_POS, DIVERGING_NEG), linewidth=0)
        ax.axhline(0, color=TEXT_2, lw=1)
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        ties = len(d) - wins - losses
        ax.set_title(f"{run}", loc="left", fontsize=12, fontweight="bold", color=TEXT, pad=24)
        ax.text(0, 1.03, f"better on {wins} · tied on {ties} · worse on {losses} queries  ·  "
                         f"mean Δ {d.mean():+.3f}", transform=ax.transAxes, fontsize=9.5, color=TEXT_2)
        ax.set_xlim(-1, len(d))
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(f"Δ {metric}")
    axes[-1, 0].set_xlabel(f"queries, sorted by Δ {metric} vs. {baseline}")
    fig.suptitle(f"Per-query change in {metric} vs. {baseline}", x=0.07, ha="left", fontsize=13,
                 fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)


def plot_cost(cost: pd.DataFrame, path: Path, subtitle: str) -> None:
    """One horizontal-bar panel per cost column (log x), runs as rows.

    `cost` columns: (panel title, unit) pairs are taken from its column names
    formatted like "Online, 300 queries|s".
    """
    _style()
    colors = _run_colors(list(cost.index))
    n = len(cost.columns)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 1.15 * len(cost) * n + 1.2))
    y = np.arange(len(cost))[::-1]
    for ax, col in zip(axes, cost.columns):
        title, unit = col.split("|")
        vals = cost[col].to_numpy(dtype=float)
        shown = np.where(vals > 0, vals, np.nan)
        ax.barh(y, shown, height=0.62, color=[colors[r] for r in cost.index], edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            label = "none" if v == 0 else _fmt(v, unit)
            ax.text(v if v > 0 else np.nanmin(shown), yi, f"  {label}", va="center", fontsize=9.5, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlim(np.nanmin(shown) / 2, np.nanmax(shown) * 12)
        ax.set_yticks(y, cost.index)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=TEXT)
    fig.suptitle(f"Cost  ·  {subtitle}  ·  log scale", x=0.07, ha="left", fontsize=13,
                 fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    _save(fig, path)


def _fmt(v: float, unit: str) -> str:
    if unit == "bytes":
        for u, s in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
            if v >= s:
                return f"{v / s:.1f} {u}"
        return f"{v:.0f} B"
    if unit == "s":
        return f"{v:.1f} s" if v >= 1 else f"{v * 1000:.0f} ms"
    if unit == "ms":
        return f"{v:.0f} ms" if v >= 10 else f"{v:.1f} ms"
    return f"{v:.3g} {unit}"


def plot_tradeoff(sweep: pd.DataFrame, baseline: dict, path: Path, subtitle: str) -> None:
    """nDCG@10 vs. cost for every SMVE setting, with the MaxSim point for reference.

    Two panels share the y axis (quality); x is single-query latency and index
    size (both log). Colour = width w (one hue, light->dark), filled marker =
    centered, hollow = not centered; label = k.
    """
    _style()
    widths = sorted(sweep["w"].unique())
    ramp = ["#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"][-len(widths):] if len(widths) <= 5 \
        else plt.cm.Blues(np.linspace(0.4, 0.95, len(widths)))
    wcolor = dict(zip(widths, ramp))
    panels = [("single_query_latency_ms", "single-query latency (ms, log)"),
              ("index_bytes", "index size (MB, log)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (col, xlabel) in zip(axes, panels):
        scale = 1e6 if col == "index_bytes" else 1
        for _, r in sweep.iterrows():
            c = wcolor[r["w"]]
            ax.scatter(r[col] / scale, r["ndcg@10"], s=70, zorder=3, linewidths=2, edgecolors=c,
                       facecolors=c if r["center"] else SURFACE)
            ax.annotate(f"k{int(r['k'])}", (r[col] / scale, r["ndcg@10"]), xytext=(7, -3),
                        textcoords="offset points", fontsize=8.5, color=TEXT_2)
        bx = baseline[col] / scale
        ax.scatter(bx, baseline["ndcg@10"], marker="*", s=260, color=SERIES[1], zorder=4,
                   edgecolors=SURFACE, linewidths=1.5)
        ax.annotate("MaxSim", (bx, baseline["ndcg@10"]), xytext=(-10, 8), textcoords="offset points",
                    ha="right", fontsize=9.5, color=TEXT)
        ax.axhline(baseline["ndcg@10"], color=SERIES[1], lw=1, ls=(0, (3, 3)), zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("nDCG@10")
    axes[0].set_ylim(0, max(0.8, baseline["ndcg@10"] + 0.08))
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=8, color=wcolor[w], label=f"w = {w:,}")
               for w in widths]
    handles += [plt.Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=TEXT_2,
                           markeredgecolor=TEXT_2, label="centered"),
                plt.Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=SURFACE,
                           markeredgecolor=TEXT_2, markeredgewidth=2, label="not centered")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0))
    fig.suptitle(f"Quality / cost trade-off  ·  {subtitle}", x=0.07, ha="left", fontsize=13,
                 fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    _save(fig, path)


def plot_alpha_sensitivity(sens: pd.DataFrame, chosen: float, path: Path, subtitle: str) -> None:
    """Hybrid quality as the lexical weight alpha varies (alpha=0 is dense only)."""
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = sens["alpha"].to_numpy()
    for metric, label in (("ndcg@10", "nDCG@10"), ("mrr@10", "MRR@10"), ("recall@100", "Recall@100")):
        m = metric.split("@")[0]
        y = sens[metric].to_numpy()
        ax.plot(x, y, color=METRIC_COLORS[m if m != "recall" else "recall"], lw=2, marker="o", markersize=5)
        ax.annotate(f"{label}  {y[-1]:.3f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=TEXT)
    ax.axvline(chosen, color=TEXT_2, lw=1, ls=(0, (3, 3)))
    ax.text(chosen, 0.02, f" α = {chosen:g} (paper default, used)", color=TEXT_2, fontsize=9,
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("α  (score = dense + α · lexical)")
    ax.set_ylabel("mean over queries")
    ax.set_xlim(0, x.max())
    _title(ax, "Hybrid sensitivity to the lexical weight", f"{subtitle}  ·  analysis only, not tuned")
    _save(fig, path)


# ---------------------------------------------------------------------------
# SMVE mechanism plots (scripts/smve_mechanism.py)
# ---------------------------------------------------------------------------

BUDGET_RAMP = {1: "#86b6ef", 2: "#3987e5", 4: "#1c5cab", 8: "#0d366b"}  # one hue, light -> dark


def plot_mechanism_curves(per_level: pd.DataFrame, metric: str, path: Path, title: str, ylabel: str) -> None:
    """One panel per setting; x = true cosine; colour = budget m; solid = repetitions,
    dashed = single matrix with the same budget."""
    _style()
    settings = list(dict.fromkeys(per_level["setting"]))
    fig, axes = plt.subplots(1, len(settings), figsize=(4.6 * len(settings), 4.4), sharey=True)
    for ax, name in zip(axes, settings):
        df = per_level[per_level["setting"] == name]
        for (mode, m), g in df.groupby(["mode", "budget"], sort=False):
            ls = "-" if mode == "repetitions" else (0, (4, 3))
            ax.plot(g["cosine"], g[metric], color=BUDGET_RAMP[m], lw=2, ls=ls,
                    marker="o" if mode == "repetitions" else None, markersize=4)
        ax.set_title(name, loc="left", fontsize=11.5, fontweight="bold", color=TEXT)
        ax.set_xlabel("true cosine of the token pair")
        ax.set_xlim(per_level["cosine"].min() - 0.02, per_level["cosine"].max() + 0.02)
    axes[0].set_ylabel(ylabel)
    if metric == "zero_fraction":
        axes[0].set_ylim(-0.02, 1.02)
    handles = [plt.Line2D([], [], color=BUDGET_RAMP[m], lw=2.5, label=f"budget ×{m}") for m in BUDGET_RAMP]
    handles += [plt.Line2D([], [], color=TEXT_2, lw=2, marker="o", markersize=4, label="repetitions (R = budget)"),
                plt.Line2D([], [], color=TEXT_2, lw=2, ls=(0, (4, 3)), label="single matrix, same budget")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0))
    fig.suptitle(title, x=0.04, ha="left", fontsize=13.5, fontweight="bold", color=TEXT)
    fig.text(0.04, 0.905, "budget ×m = m·w anchors and m·k non-zeros per token; "
             "base (w, k) = (2048, 8) for d=128, (4096, 8) for d=1024",
             fontsize=9.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    _save(fig, path)


def plot_mechanism_summary(summary: pd.DataFrame, high_cos: float, path: Path) -> None:
    """Rows: Spearman(S, true cosine) and zero-fraction of strong pairs; x = budget."""
    _style()
    settings = list(dict.fromkeys(summary["setting"]))
    rows = [("spearman", "Spearman(S, true cosine)", "higher is better"),
            (f"zero_fraction_cos>={high_cos}", f"fraction S = 0 when cosine ≥ {high_cos}", "lower is better")]
    mode_color = {"repetitions": SERIES[1], "single matrix": SERIES[0]}
    fig, axes = plt.subplots(2, len(settings), figsize=(4.4 * len(settings), 7.2), sharex=True)
    for j, name in enumerate(settings):
        df = summary[summary["setting"] == name]
        base = df[df["budget"] == 1]
        for i, (col, ylabel, note) in enumerate(rows):
            ax = axes[i, j]
            for mode, color in mode_color.items():
                g = pd.concat([base, df[df["mode"] == mode]]).drop_duplicates("budget").sort_values("budget")
                ax.plot(g["budget"], g[col], color=color, lw=2, marker="o", markersize=6, label=mode)
            ax.set_xscale("log", base=2)
            ax.set_xticks(BUDGETS_TICKS, [f"×{b}" for b in BUDGETS_TICKS])
            if i == 0:
                ax.set_title(name, loc="left", fontsize=11.5, fontweight="bold", color=TEXT)
            if j == 0:
                ax.set_ylabel(f"{ylabel}\n({note})")
            if i == 1:
                ax.set_xlabel("budget (anchors and non-zeros per token)")
                ax.set_ylim(-0.02, max(0.05, summary[col].max() * 1.1))
    for i in range(2):
        lo = min(ax.get_ylim()[0] for ax in axes[i])
        hi = max(ax.get_ylim()[1] for ax in axes[i])
        for ax in axes[i]:
            ax.set_ylim(lo, hi)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Repetitions vs. one wider matrix at the same budget", x=0.04, ha="left",
                 fontsize=13.5, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, path)


BUDGETS_TICKS = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# SMVE sweep plots (scripts/plot_smve_sweep.py). `refs` maps a reference
# method label -> dict of its metrics / costs (MaxSim, dense, hybrid).
# ---------------------------------------------------------------------------

K_RAMP = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]  # sequential blue, k small -> large
REF_STYLE = {  # reference methods: fixed colour + line style, never a k colour
    "MaxSim": (SERIES[1], (0, (5, 3))),
    "BGE-M3 dense + lexical": (SERIES[2], (0, (1, 2))),
    "BGE-M3 dense": (TEXT_2, (0, (3, 2, 1, 2))),
}


def _ref_lines(ax, refs: dict, metric: str) -> None:
    for name, vals in refs.items():
        color, ls = REF_STYLE[name]
        ax.axhline(vals[metric], color=color, lw=1.6, ls=ls, zorder=1, label=name)


def plot_sweep_metric_vs_w(runs: pd.DataFrame, refs: dict, path: Path, title: str) -> None:
    """2x2 small multiples: metric vs w (log2), one line per k."""
    _style()
    metrics = [("ndcg@10", "nDCG@10"), ("recall@100", "Recall@100"), ("mrr@10", "MRR@10"), ("recall@10", "Recall@10")]
    ks = sorted(runs["k"].unique())
    colors = dict(zip(ks, K_RAMP[-len(ks):] if len(ks) <= len(K_RAMP) else plt.cm.Blues(np.linspace(.3, 1, len(ks)))))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (m, label) in zip(axes.flat, metrics):
        for k in ks:
            g = runs[runs["k"] == k].sort_values("w")
            ax.plot(g["w"], g[m], color=colors[k], lw=2, marker="o", markersize=4.5, label=f"k = {k}")
        _ref_lines(ax, refs, m)
        ax.set_xscale("log", base=2)
        ws = sorted(runs["w"].unique())
        ax.set_xticks(ws, [f"{w // 1024}K" for w in ws])
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=TEXT)
        ax.set_xlabel("w (number of anchors)")
        lo = min(runs[m].min(), min(v[m] for v in refs.values()))
        ax.set_ylim(max(0, lo - 0.05), min(1.0, max(v[m] for v in refs.values()) + 0.05))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=len(ks) + len(refs),
               bbox_to_anchor=(0.5, 0), fontsize=9.5)
    fig.suptitle(title, x=0.04, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.text(0.04, 0.935, "horizontal lines = reference methods (MaxSim, BGE-M3 dense + lexical, BGE-M3 dense); "
             "exact numbers in summary.md", fontsize=9.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    _save(fig, path)


def _heatmap(ax, grid: pd.DataFrame, cmap, vmin, vmax, fmt_cell) -> None:
    im = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=TEXT_2)
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, fmt_cell(v), ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if lum < 0.5 else TEXT)
    ax.set_xticks(range(grid.shape[1]), [f"{w // 1024}K" for w in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [str(k) for k in grid.index])
    ax.set_xlabel("w (number of anchors)")
    ax.set_ylabel("k (anchors kept per token)")
    ax.grid(False)
    return im


def plot_sweep_heatmaps(runs: pd.DataFrame, maxsim_ndcg: float, path: Path) -> None:
    """nDCG@10 over (w, k), uncentered vs centered; each cell shows value and gap to MaxSim."""
    _style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#e8f1fc", "#86b6ef", "#2a78d6", "#104281"])
    cmap.set_bad(SURFACE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    vmin, vmax = runs["ndcg@10"].min(), runs["ndcg@10"].max()
    for ax, center in zip(axes, (False, True)):
        grid = runs[runs["center"] == center].pivot_table(index="k", columns="w", values="ndcg@10")
        im = _heatmap(ax, grid, cmap, vmin, vmax, lambda v: f"{v:.3f}\n{v - maxsim_ndcg:+.3f}")
        ax.set_title("centered" if center else "not centered", loc="left", fontsize=12,
                     fontweight="bold", color=TEXT)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("nDCG@10")
    cbar.outline.set_visible(False)
    fig.suptitle(f"SMVE nDCG@10 over (w, k)  ·  second line = gap to MaxSim ({maxsim_ndcg:.3f})",
                 x=0.04, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    _save(fig, path)


def plot_sweep_centering_effect(runs: pd.DataFrame, path: Path) -> None:
    """(centered - not centered) for nDCG@10 and Recall@100; diverging, grey = no change."""
    _style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [DIVERGING_NEG, "#f0efec", DIVERGING_POS])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (m, label) in zip(axes, (("ndcg@10", "Δ nDCG@10"), ("recall@100", "Δ Recall@100"))):
        on = runs[runs["center"]].pivot_table(index="k", columns="w", values=m)
        off = runs[~runs["center"]].pivot_table(index="k", columns="w", values=m)
        delta = on - off
        lim = np.nanmax(np.abs(delta.to_numpy()))
        im = _heatmap(ax, delta, cmap, -lim, lim, lambda v: f"{v:+.3f}")
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=TEXT)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.outline.set_visible(False)
    fig.suptitle("Effect of centering (centered − not centered)  ·  blue = centering helps, red = hurts",
                 x=0.04, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def plot_sweep_reps(pairs: pd.DataFrame, seed_std: float, path: Path) -> None:
    """Each point = one equal-budget pair: x = single wider matrix, y = repetitions.

    Points on the diagonal y = x mean a tie. The grey band is +-1 seed standard
    deviation: differences inside it are within run-to-run randomness.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    sizes = {2: 40, 4: 80, 8: 140}
    for ax, (xs, ys, label) in zip(axes, (("single nDCG@10", "reps nDCG@10", "nDCG@10"),
                                          ("single R@100", "reps R@100", "Recall@100"))):
        lo = min(pairs[xs].min(), pairs[ys].min()) - 0.02
        hi = max(pairs[xs].max(), pairs[ys].max()) + 0.02
        grid = np.linspace(lo, hi, 2)
        ax.fill_between(grid, grid - seed_std, grid + seed_std, color=GRID, alpha=0.9, zorder=0, linewidth=0)
        ax.plot(grid, grid, color=TEXT_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for center, color in ((False, SERIES[0]), (True, SERIES[2])):
            g = pairs[pairs["center"] == center]
            ax.scatter(g[xs], g[ys], s=g["R"].map(sizes), color=color, alpha=0.85,
                       edgecolors=SURFACE, linewidths=1, zorder=3)
        d = pairs[ys] - pairs[xs]
        ax.text(0.02, 0.97, f"mean difference {d.mean():+.4f}\nlargest |difference| {d.abs().max():.4f}\n"
                            f"seed std ≈ {seed_std:.3f}", transform=ax.transAxes, va="top", fontsize=9.5,
                color=TEXT_2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{label}, one wider matrix (R·w anchors, top R·k)")
        ax.set_ylabel(f"{label}, repetitions (R blocks of w, top k each)")
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=TEXT)
    handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[0], label="not centered"),
               plt.Line2D([], [], marker="o", ls="", color=SERIES[2], label="centered")]
    handles += [plt.Line2D([], [], marker="o", ls="", color=TEXT_2, markersize=np.sqrt(v) / 1.3, label=f"R = {r}")
                for r, v in sizes.items()]
    handles += [plt.Line2D([], [], color=TEXT_2, ls=(0, (4, 3)), label="tie (y = x)"),
                plt.Rectangle((0, 0), 1, 1, color=GRID, label="±1 seed std")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0))
    fig.suptitle("Repetitions vs. one wider matrix at the same budget (SciFact)", x=0.04, ha="left",
                 fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _save(fig, path)


def plot_sweep_tradeoff(runs: pd.DataFrame, refs: dict, path: Path) -> None:
    """nDCG@10 vs. three costs; every SMVE run + reference methods as labelled points."""
    _style()
    panels = [("nnz_per_doc", "non-zeros per document (log)", 1),
              ("single_query_latency_ms", "single-query latency, ms (log)", 1),
              ("index_bytes", "index size, MB (log)", 1e6)]
    center_color = {False: SERIES[0], True: SERIES[2]}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    for ax, (col, xlabel, scale) in zip(axes, panels):
        for (center, multi), g in runs.groupby(["center", runs["reps"] > 1]):
            ax.scatter(g[col] / scale, g["ndcg@10"], s=34 if not multi else 46, marker="^" if multi else "o",
                       color=center_color[center], alpha=0.85, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        for name, vals in refs.items():
            color, ls = REF_STYLE[name]
            ax.axhline(vals["ndcg@10"], color=color, lw=1.2, ls=ls, zorder=1)
            if col in vals and vals[col] is not None:
                ax.scatter(vals[col] / scale, vals["ndcg@10"], marker="*", s=220, color=color,
                           edgecolors=SURFACE, linewidths=1.2, zorder=4)
                below = name == "BGE-M3 dense"
                ax.annotate(name, (vals[col] / scale, vals["ndcg@10"]), xytext=(0, -15 if below else 9),
                            textcoords="offset points", ha="center", fontsize=8.5, color=TEXT)
        ax.set_xscale("log")
        from matplotlib.ticker import FuncFormatter, NullFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("nDCG@10")
    handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[0], label="SMVE, not centered"),
               plt.Line2D([], [], marker="o", ls="", color=SERIES[2], label="SMVE, centered"),
               plt.Line2D([], [], marker="^", ls="", color=TEXT_2, label="repetitions (R > 1)"),
               plt.Line2D([], [], marker="*", ls="", markersize=12, color=TEXT_2, label="reference method")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
    fig.suptitle("SMVE quality vs. cost, every sweep setting", x=0.04, ha="left", fontsize=14,
                 fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, path)


# ---------------------------------------------------------------------------
# Reranking plots (scripts/evaluate_rerank.py)
# colour = reranker, line style = first stage; references in ink / grey.
# ---------------------------------------------------------------------------

RERANKER_COLOR = {"MaxSim": SERIES[1], "cross-encoder": SERIES[6], "none": TEXT_2}
STAGE_LS = {"Dense + lexical": "-", "SMVE": (0, (5, 3))}
RERANK_REF_STYLE = {"Exhaustive MaxSim": (TEXT, (0, (2, 2))), "BGE-M3 dense": (TEXT_2, (0, (5, 2, 1, 2)))}


def _rerank_handles(stages) -> list:
    h = [plt.Line2D([], [], color=RERANKER_COLOR["cross-encoder"], lw=2.5, label="cross-encoder rerank"),
         plt.Line2D([], [], color=RERANKER_COLOR["MaxSim"], lw=2.5, label="MaxSim rerank")]
    h += [plt.Line2D([], [], color=TEXT_2, lw=2, ls=STAGE_LS[s], label=f"first stage: {s}") for s in stages]
    h += [plt.Line2D([], [], color=c, lw=1.5, ls=ls, label=n) for n, (c, ls) in RERANK_REF_STYLE.items()]
    return h


def plot_rerank_depth(summary: pd.DataFrame, refs: dict, path: Path) -> None:
    """Quality vs rerank depth; depth 'none' = first stage alone."""
    _style()
    stages = list(dict.fromkeys(summary["first_stage"]))
    depths = sorted(summary.loc[summary.depth > 0, "depth"].unique())
    xpos = {0: 0, **{d: i + 1 for i, d in enumerate(depths)}}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (m, label) in zip(axes, (("ndcg@10", "nDCG@10"), ("mrr@10", "MRR@10"))):
        for stage in stages:
            base = summary[(summary.first_stage == stage) & (summary.reranker == "none")]
            for rr in ("MaxSim", "cross-encoder"):
                g = pd.concat([base, summary[(summary.first_stage == stage) & (summary.reranker == rr)]])
                ax.plot([xpos[d] for d in g.depth], g[m], color=RERANKER_COLOR[rr], ls=STAGE_LS[stage], lw=2,
                        marker="o", markersize=5)
        for name, (c, ls) in RERANK_REF_STYLE.items():
            ax.axhline(refs[name][m], color=c, lw=1.4, ls=ls, zorder=1)
        ax.set_xticks(list(xpos.values()), ["none"] + [str(d) for d in depths])
        ax.set_xlabel("rerank depth d (top-d candidates re-scored)")
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.legend(handles=_rerank_handles(stages), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Reranking quality vs. depth  ·  SciFact", x=0.04, ha="left", fontsize=14,
                 fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    _save(fig, path)


def plot_rerank_latency(summary: pd.DataFrame, refs: dict, path: Path) -> None:
    """nDCG@10 vs single-query latency (log); each line walks d = 10 -> 100."""
    _style()
    from matplotlib.ticker import FuncFormatter, NullFormatter
    stages = list(dict.fromkeys(summary["first_stage"]))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for stage in stages:
        base = summary[(summary.first_stage == stage) & (summary.reranker == "none")]
        ax.scatter(base["latency_total_ms"], base["ndcg@10"], s=60, facecolors=SURFACE, edgecolors=TEXT_2,
                   linewidths=2, zorder=4)
        ax.annotate(f"{stage} alone", (base["latency_total_ms"].iat[0], base["ndcg@10"].iat[0]),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=TEXT_2)
        for rr in ("MaxSim", "cross-encoder"):
            g = summary[(summary.first_stage == stage) & (summary.reranker == rr)].sort_values("depth")
            ax.plot(g["latency_total_ms"], g["ndcg@10"], color=RERANKER_COLOR[rr], ls=STAGE_LS[stage], lw=2,
                    marker="o", markersize=5, zorder=3)
            for _, r in g.iterrows():
                ax.annotate(f"{r.depth}", (r.latency_total_ms, r["ndcg@10"]), xytext=(4, 5),
                            textcoords="offset points", fontsize=8, color=TEXT_2)
    for name, (c, ls) in RERANK_REF_STYLE.items():
        v = refs[name]
        ax.scatter(v["latency_total_ms"], v["ndcg@10"], marker="*", s=240, color=c, edgecolors=SURFACE, zorder=5)
        ax.annotate(name, (v["latency_total_ms"], v["ndcg@10"]), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color=TEXT)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("single-query latency, ms (log)  ·  numbers = rerank depth")
    ax.set_ylabel("nDCG@10")
    fig.legend(handles=_rerank_handles(stages)[:2 + len(stages)], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0))
    _title(ax, "Quality vs. latency", "cross-encoder on the Mac GPU (MPS, fp16); MaxSim and first stages on CPU")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, path)


def plot_rerank_cost(summary: pd.DataFrame, refs: dict, path: Path) -> None:
    """Per-query compute and total storage, for depth-20 and depth-100 pipelines plus references."""
    _style()
    sel = summary[summary.depth.isin([20, 100])].copy()
    sel = sel.sort_values(["reranker", "first_stage", "depth"])
    labels = list(sel["config"]) + list(refs)
    colors = [RERANKER_COLOR[r] for r in sel["reranker"]] + [RERANK_REF_STYLE[n][0] for n in refs]
    flops = list(sel["flops_total"]) + [v["flops_total"] for v in refs.values()]
    storage = list(sel["storage_bytes"]) + [v["storage_bytes"] for v in refs.values()]
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * len(labels) + 2), sharey=True)
    for ax, vals, title, unit in ((axes[0], flops, "Compute per query", "flops"),
                                  (axes[1], storage, "Storage (first-stage index + reranker)", "bytes")):
        ax.barh(y, vals, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            txt = (f"{v / 1e12:.1f} TFLOP" if v >= 1e12 else f"{v / 1e9:.2f} GFLOP") if unit == "flops" \
                else _fmt(v, "bytes")
            ax.text(v, yi, f"  {txt}", va="center", fontsize=8.5, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlim(min(vals) / 3, max(vals) * 30)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT)
    axes[0].set_yticks(y, labels)
    fig.suptitle("Cost of each pipeline (log scale)", x=0.04, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


# ---------------------------------------------------------------------------
# First-stage comparison (scripts/compare_first_stages.py)
# ---------------------------------------------------------------------------

FAMILY_COLOR = {"SMVE": SERIES[0], "MUVERA": SERIES[1]}
FS_REF_STYLE = {  # reference first stages: (colour, marker)
    "Exhaustive MaxSim": (TEXT, "*"), "BM25": (SERIES[5], "P"), "BGE-M3 dense": (TEXT_2, "s"),
    "BGE-M3 lexical": (SERIES[3], "X"), "BGE-M3 dense + lexical": (SERIES[2], "D"),
}


def _pareto(df: pd.DataFrame, cost: str, quality: str) -> pd.DataFrame:
    """Settings not beaten by another setting that is both cheaper and better."""
    d = df.sort_values([cost, quality], ascending=[True, False])
    keep, best = [], -np.inf
    for i, r in d.iterrows():
        if r[quality] > best:
            keep.append(i)
            best = r[quality]
    return d.loc[keep]


def plot_first_stage_frontiers(runs: pd.DataFrame, refs: pd.DataFrame, path: Path, dataset: str = "SciFact") -> None:
    """Rows = quality metric, columns = cost; dots = every setting, lines = Pareto frontier per family."""
    _style()
    from matplotlib.ticker import FuncFormatter, NullFormatter
    rows = [("recall@100", "Recall@100 (first stage)"), ("ndcg@10", "nDCG@10 (first stage alone)"),
            ("rerank_ndcg@10", "nDCG@10 after MaxSim rerank of top 100")]
    cols = [("index_mb", "index size, MB (log)"), ("single_query_latency_ms", "single-query latency, ms (log)")]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(13, 4.2 * len(rows)), sharex="col", sharey="row")
    for i, (q, qlabel) in enumerate(rows):
        for j, (c, clabel) in enumerate(cols):
            ax = axes[i, j]
            for fam, color in FAMILY_COLOR.items():
                g = runs[runs.family == fam]
                ax.scatter(g[c], g[q], s=16, color=color, alpha=0.35, linewidths=0, zorder=2)
                f = _pareto(g, c, q)
                ax.plot(f[c], f[q], color=color, lw=2.2, marker="o", markersize=4, zorder=3, label=f"{fam} (best at each cost)")
            for _, r in refs.iterrows():
                color, marker = FS_REF_STYLE[r["name"]]
                ax.scatter(r[c], r[q], marker=marker, s=150 if marker == "*" else 70, color=color,
                           edgecolors=SURFACE, linewidths=1, zorder=4, label=r["name"])
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if i == len(rows) - 1:
                ax.set_xlabel(clabel)
            if j == 0:
                ax.set_ylabel(qlabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0), fontsize=9.5)
    fig.suptitle(f"First stages on {dataset}: SMVE vs MUVERA vs references  ·  faint dots = every setting",
                 x=0.04, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    _save(fig, path)


# ---------------------------------------------------------------------------
# Inverted-index latency (scripts/evaluate_inverted_index.py)
# ---------------------------------------------------------------------------

def plot_index_latency(summary: pd.DataFrame, per_query: pd.DataFrame, path: Path) -> None:
    """(A) median latency split into encode / score / top-k; (B) work per query vs scoring time."""
    _style()
    from matplotlib.ticker import FuncFormatter, NullFormatter
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.5 * len(summary) + 3.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(summary))[::-1]
    parts = [("encode_ms", "encode query", SERIES[0]), ("score_ms", "read postings / scan", SERIES[1]),
             ("topk_ms", "pick top 100", SERIES[2])]
    left = np.zeros(len(summary))
    for col, label, color in parts:
        ax.barh(y, summary[col], left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        left += summary[col].to_numpy()
    for yi, tot, old in zip(y, summary["total_ms"], summary["old_latency_ms"]):
        ax.text(tot, yi, f"  {tot:.2f} ms   (was {old:.1f} ms)", va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(y, summary["method"])
    ax.set_xlim(0, summary["total_ms"].max() * 1.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("median single-query latency, ms (CPU)")
    ax.set_title("Where the time goes", loc="left", fontsize=12, fontweight="bold", color=TEXT)
    ax.legend(loc="lower right")

    ax = axes[1]
    sparse = per_query.dropna(subset=["postings_read"])
    for i, (m, g) in enumerate(sparse.groupby("method", sort=False)):
        ax.scatter(g["postings_read"], g["score_ms"], s=12, alpha=0.55, color=SERIES[i % len(SERIES)],
                   linewidths=0, label=m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.set_minor_formatter(NullFormatter())
    ax.set_xlabel("posting entries read by the query (log)")
    ax.set_ylabel("scoring time, ms (log)")
    ax.set_title("Work per query drives scoring time", loc="left", fontsize=12, fontweight="bold", color=TEXT)
    ax.legend(loc="upper left", fontsize=8.5, markerscale=2)
    fig.suptitle("Sparse retrieval with a hand-built inverted index  ·  SciFact, 300 queries, one at a time",
                 x=0.04, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path)


# ---------------------------------------------------------------------------
# Cross-dataset summary (scripts/cross_dataset_summary.py)
# ---------------------------------------------------------------------------

CROSS_METHODS = {  # fixed colour per method across both panels
    "Exhaustive MaxSim": TEXT, "BM25": SERIES[5], "BGE-M3 dense": TEXT_2,
    "BGE-M3 dense + lexical": SERIES[2], "SMVE (fixed setting)": SERIES[0], "MUVERA (fixed setting)": SERIES[1],
}


def plot_cross_dataset(table: pd.DataFrame, head: pd.DataFrame, path: Path) -> None:
    """(A) first stage alone, (B) first stage + MaxSim rerank of top 100; grouped by dataset."""
    _style()
    datasets = list(dict.fromkeys(table["dataset"]))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    panels = [(axes[0], "ndcg@10", list(CROSS_METHODS), "First stage alone"),
              (axes[1], "rerank_ndcg@10", [m for m in CROSS_METHODS if m != "Exhaustive MaxSim"],
               "First stage + MaxSim rerank of top 100  (black line = exhaustive MaxSim)")]
    for ax, col, methods, title in panels:
        width = 0.8 / len(methods)
        x = np.arange(len(datasets))
        for j, m in enumerate(methods):
            vals = [table[(table.dataset == d) & (table.method == m)][col].iloc[0] for d in datasets]
            xs = x - 0.4 + width * (j + 0.5)
            ax.bar(xs, vals, width=width, color=CROSS_METHODS[m], edgecolor=SURFACE, linewidth=1.5, label=m)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=TEXT, rotation=90)
        if col == "rerank_ndcg@10":
            for xi, d in zip(x, datasets):
                ms = table[(table.dataset == d) & (table.method == "Exhaustive MaxSim")]["ndcg@10"].iloc[0]
                ax.plot([xi - 0.42, xi + 0.42], [ms, ms], color=TEXT, lw=2)
        hr = head.set_index("dataset")["headroom vs dense"]
        ax.set_xticks(x, [f"{d}\nheadroom vs dense {hr[d]:+.3f}" for d in datasets])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, table[col].max() * 1.18)
        ax.set_ylabel("nDCG@10")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0), fontsize=9.5)
    fig.suptitle("All methods across datasets  ·  SMVE / MUVERA at one fixed setting chosen on SciFact",
                 x=0.04, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _save(fig, path)


# ---------------------------------------------------------------------------
# Router plots (scripts/router_baselines.py)
# ---------------------------------------------------------------------------

ROUTER_STYLE = {  # name: (colour, line style, width)
    "oracle": (TEXT, (0, (4, 2)), 1.6), "random": (TEXT_2, (0, (1, 2)), 1.6),
    "heuristic: small top-2 gap": (SERIES[3], "-", 1.6), "heuristic: dense/lexical disagree": (SERIES[4], "-", 1.6),
    "logistic": (SERIES[0], "-", 2.4), "boosted": (SERIES[1], "-", 2.4),
}


def plot_router_curves(results: dict, data: pd.DataFrame, lat_all: dict, path: Path) -> None:
    """Rows = protocol, columns = test set; nDCG@10 vs average latency as more queries are escalated."""
    _style()
    protos = [("A_pooled_cv", "pooled 5-fold CV"), ("B_train_on_scifact", "trained on SciFact train")]
    sets = list(results["A_pooled_cv"])
    fig, axes = plt.subplots(len(protos), len(sets), figsize=(5.2 * len(sets), 4.4 * len(protos)), squeeze=False)
    for i, (proto, ptitle) in enumerate(protos):
        for j, ds in enumerate(sets):
            ax, r = axes[i, j], results[proto][ds]
            for name, v in r["routers"].items():
                c, ls, lw = ROUTER_STYLE[name]
                x = r["base_ms"] + np.array(v["curve"]["frac"]) * r["escalate_ms"]
                ax.plot(x, v["curve"]["ndcg"], color=c, ls=ls, lw=lw, label=name)
            ax.scatter([r["base_ms"], r["base_ms"] + r["escalate_ms"]], [r["never"], r["always"]], color=TEXT,
                       s=28, zorder=5)
            from smve_lab.datasets import info as ds_info
            ax.set_title(f"{ds_info(ds).display}  ·  {ptitle}", loc="left",
                         fontsize=11, fontweight="bold", color=TEXT)
            if i == len(protos) - 1:
                ax.set_xlabel("average latency per query, ms")
            if j == 0:
                ax.set_ylabel("nDCG@10")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0), fontsize=9.5)
    fig.suptitle("Routing: escalate the queries a router flags first  ·  left end = never, right end = always",
                 x=0.04, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, path)


def plot_router_calibration(data: pd.DataFrame, oof: dict, path: Path, bins: int = 10) -> None:
    """Reliability diagram: predicted P(escalation helps) vs how often it actually helped."""
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot([0, 1], [0, 1], color=TEXT_2, lw=1.2, ls=(0, (4, 3)), label="perfectly calibrated")
    edges = np.linspace(0, 1, bins + 1)
    y = data["y"].to_numpy()
    for name, p in oof.items():
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        xs, ys, ns = [], [], []
        for b in range(bins):
            if (idx == b).sum() >= 5:
                xs.append(p[idx == b].mean()); ys.append(y[idx == b].mean()); ns.append((idx == b).sum())
        c = ROUTER_STYLE[name][0]
        ax.plot(xs, ys, color=c, lw=2, label=name)
        ax.scatter(xs, ys, s=np.array(ns) / max(ns) * 220 + 15, color=c, edgecolors=SURFACE, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability that escalating helps")
    ax.set_ylabel("observed share of queries where it helped")
    ax.legend(loc="upper left")
    _title(ax, "Calibration of the learned routers", "pooled 5-fold CV, all datasets  ·  dot size = queries in bin")
    _save(fig, path)


def plot_router_coefficients(features: list[str], coef: np.ndarray, path: Path) -> None:
    """Standardised logistic-regression weights: positive = pushes toward escalating."""
    _style()
    order = np.argsort(np.abs(coef))
    fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(features) + 1.6))
    y = np.arange(len(features))
    c = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in coef[order]]
    ax.barh(y, coef[order], color=c, height=0.65, edgecolor=SURFACE)
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_yticks(y, [features[i] for i in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("coefficient on standardised feature")
    _title(ax, "What the logistic router looks at", "blue = more likely to escalate, red = less likely")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Jev router plots (scripts/evaluate_jev_router.py)
# ---------------------------------------------------------------------------

JEV_STYLE = {  # name: (colour, line style, width)
    "oracle": (TEXT, (0, (4, 2)), 1.5), "random": (TEXT_2, (0, (1, 2)), 1.5),
    "first-stage boosted": (SERIES[1], "-", 2.0), "first-stage logistic": (SERIES[3], "-", 1.6),
    "Jev B: lower candidate better": (SERIES[6], (0, (6, 2)), 2.2), "Jev B logistic": (SERIES[4], "-", 1.8),
    "first-stage + Jev B logistic": (SERIES[0], "-", 2.6),
}


def plot_jev_curves(results: dict, show: list[str], path: Path) -> None:
    """nDCG@10 vs average latency (Jev overhead included) as more queries are escalated."""
    _style()
    from smve_lab.datasets import info as ds_info
    protos = [("A_pooled_cv", "pooled 5-fold CV"), ("B_train_on_scifact", "trained on SciFact train")]
    sets = list(results["A_pooled_cv"])
    fig, axes = plt.subplots(2, len(sets), figsize=(5.4 * len(sets), 9), squeeze=False)
    for i, (proto, ptitle) in enumerate(protos):
        for j, ds in enumerate(sets):
            ax, r = axes[i, j], results[proto][ds]
            for name in show:
                v = r["routers"][name]
                c, ls, lw = JEV_STYLE[name]
                x = r["base_ms"] + v.get("overhead_ms", 0) + np.array(v["curve"]["frac"]) * r["escalate_ms"]
                ax.plot(x, v["curve"]["ndcg"], color=c, ls=ls, lw=lw, label=name)
            ax.set_title(f"{ds_info(ds).display}  ·  {ptitle}", loc="left", fontsize=11, fontweight="bold", color=TEXT)
            if i == 1:
                ax.set_xlabel("average latency per query, ms (router overhead included)")
            if j == 0:
                ax.set_ylabel("nDCG@10")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0), fontsize=9.5)
    fig.suptitle("Routing with Jev vs first-stage features  ·  escalate the queries each router flags first",
                 x=0.04, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _save(fig, path)


def plot_jev_auc(results: dict, routers: list[str], path: Path) -> None:
    """AUC with 95% CI per router and test set; trained routers shown for both protocols."""
    _style()
    from smve_lab.datasets import info as ds_info
    sets = list(results["A_pooled_cv"])
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 0.42 * len(routers) + 2.2), sharey=True)
    y = np.arange(len(routers))[::-1]
    for ax, ds in zip(axes, sets):
        for proto, color, dy, lab in (("A_pooled_cv", SERIES[0], 0.14, "pooled CV"),
                                      ("B_train_on_scifact", SERIES[1], -0.14, "trained on SciFact")):
            v = [results[proto][ds]["routers"][k] for k in routers]
            auc = np.array([x["auc"] for x in v])
            lo = auc - np.array([x["auc_ci95"][0] for x in v])
            hi = np.array([x["auc_ci95"][1] for x in v]) - auc
            ax.errorbar(auc, y + dy, xerr=[lo, hi], fmt="o", color=color, ms=5, capsize=0, lw=1.4, label=lab)
        ax.axvline(0.5, color=TEXT_2, lw=1, ls=(0, (3, 3)))
        ax.set_title(ds_info(ds).display, loc="left", fontsize=12, fontweight="bold", color=TEXT)
        ax.set_xlabel("AUC (0.5 = random)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, routers)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("How well each router picks the queries where the cross-encoder helps  ·  zero-shot Jev rows are "
                 "untrained (same value in both protocols)", x=0.02, ha="left", fontsize=13, fontweight="bold",
                 color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def plot_jev_calibration(data: pd.DataFrame, probs: dict, path: Path, bins: int = 10) -> None:
    """Reliability per dataset: predicted probability vs observed share where escalation helped."""
    _style()
    from smve_lab.datasets import info as ds_info
    sets = [d for d in ("scifact", "nfcorpus", "arguana")]
    fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 5.2), sharey=True)
    edges = np.linspace(0, 1, bins + 1)
    colors = [SERIES[6], SERIES[0], SERIES[1]]
    for ax, ds in zip(axes, sets):
        m = ((data.dataset == ds) & (data.split == "test")).to_numpy()
        yv = data["y"].to_numpy()[m]
        ax.plot([0, 1], [0, 1], color=TEXT_2, lw=1.2, ls=(0, (4, 3)))
        for (name, p), c in zip(probs.items(), colors):
            pv = np.clip(p[m], 0, 1)
            idx = np.clip(np.digitize(pv, edges) - 1, 0, bins - 1)
            pts = [(pv[idx == b].mean(), yv[idx == b].mean(), (idx == b).sum()) for b in range(bins) if (idx == b).sum() >= 5]
            if pts:
                xs, ys, ns = zip(*pts)
                ax.plot(xs, ys, color=c, lw=2, label=name)
                ax.scatter(xs, ys, s=np.array(ns) / max(ns) * 160 + 12, color=c, edgecolors=SURFACE, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(ds_info(ds).display, loc="left", fontsize=12, fontweight="bold", color=TEXT)
        ax.set_xlabel("predicted probability")
    axes[0].set_ylabel("observed share where escalating helped")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0), fontsize=9.5)
    fig.suptitle("Calibration per dataset  ·  dashed = perfectly calibrated  ·  dot size = queries in bin",
                 x=0.04, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    _save(fig, path)


# ---------------------------------------------------------------------------
# Jev as the reranker (scripts/jev_rerank.py)
# ---------------------------------------------------------------------------

RERANK_METHOD_COLOR = {
    "first stage": TEXT_2, "base (MaxSim@10)": SERIES[3], "cross-encoder": SERIES[1],
    "Jev V1": SERIES[6], "Jev V2": SERIES[0], "Jev V3": SERIES[2],
}


def plot_jev_rerank(summary: pd.DataFrame, latency: dict, path: Path) -> None:
    """(A) nDCG@10 per dataset and method; (B) nDCG@10 vs single-query latency."""
    _style()
    from matplotlib.ticker import FuncFormatter, NullFormatter
    from smve_lab.datasets import info as ds_info
    datasets = list(dict.fromkeys(summary["dataset"]))
    methods = [m for m in RERANK_METHOD_COLOR if m in set(summary["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    width = 0.8 / len(methods)
    x = np.arange(len(datasets))
    for j, m in enumerate(methods):
        vals = [summary[(summary.dataset == d) & (summary.method == m)]["ndcg@10"].iloc[0] for d in datasets]
        xs = x - 0.4 + width * (j + 0.5)
        ax.bar(xs, vals, width=width, color=RERANK_METHOD_COLOR[m], edgecolor=SURFACE, linewidth=1.5, label=m)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90, color=TEXT)
    ax.set_xticks(x, [ds_info(d).display for d in datasets])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, summary["ndcg@10"].max() * 1.2)
    ax.set_ylabel("nDCG@10")
    ax.set_title("Reranking the first stage's top 20", loc="left", fontsize=12, fontweight="bold", color=TEXT)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3, fontsize=9)

    ax = axes[1]
    markers = {"scifact": "o", "nfcorpus": "s", "arguana": "D"}
    for d in datasets:
        for m in methods:
            if m == "cross-encoder":
                lat = latency[d]["cross_encoder_ms"]
            elif m.startswith("Jev"):
                lat = latency[d].get("jev_ms", {}).get(m.split()[1])
            elif m == "first stage":
                lat = latency[d]["first_stage_ms"]
            else:
                lat = None
            if lat:
                v = summary[(summary.dataset == d) & (summary.method == m)]["ndcg@10"].iloc[0]
                ax.scatter(lat, v, s=70, marker=markers[d], color=RERANK_METHOD_COLOR[m], edgecolors=SURFACE, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("single-query latency, ms (log)")
    ax.set_ylabel("nDCG@10")
    handles = [plt.Line2D([], [], marker=mk, ls="", color=TEXT_2, label=ds_info(d).display) for d, mk in markers.items()
               if d in datasets]
    ax.legend(handles=handles, loc="center left", fontsize=9)
    ax.set_title("Quality vs latency  ·  colour = method", loc="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.suptitle("Jev (API) vs bge-reranker-v2-m3 (local GPU) as the reranker", x=0.04, ha="left", fontsize=14,
                 fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path)
