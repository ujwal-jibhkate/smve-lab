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
