"""Compute Spearman rank correlation between DialogMTEB and general MTEB scores.

Two comparisons:
  - DialogMTEB(v1, eng)          vs MTEB(eng)
  - DialogMTEB(v1, multilingual) vs MTEB(multilingual)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    BENCHMARKS,
    OUT_DIR,
    RUN_MODELS,
    cache,
    get_complete_models,
    load_score_df,
    load_unique_tasks,
)

import mteb

MIN_OVERLAP = 5

MTEB_COUNTERPARTS = {
    "eng": "MTEB(eng)",
    "multilingual": "MTEB(Multilingual, v2)",
}

COLORS = {
    "eng": "#4e79a7",
    "multilingual": "#f28e2b",
}


def mean_scores(tasks: list, min_coverage: int | None = None) -> pd.Series:
    """Return per-model mean score over the given tasks (RUN_MODELS only)."""
    df = load_score_df(tasks)
    task_names = [t.metadata.name for t in tasks]
    available = [t for t in task_names if t in df.index]
    complete = get_complete_models(df, task_names, min_coverage)
    return df.loc[available, complete].mean(axis=0, skipna=True)


def load_general_mteb_means(benchmark_name: str) -> pd.Series | None:
    try:
        bench = mteb.get_benchmark(benchmark_name)
    except Exception as e:
        print(f"  Could not load {benchmark_name}: {e}")
        return None
    tasks = bench.tasks
    results = cache.load_results(tasks=tasks, models=RUN_MODELS)
    df = results.to_dataframe().set_index("task_name")
    task_names = [t.metadata.name for t in tasks]
    available = [t for t in task_names if t in df.index]
    if not available:
        print(f"  No cached results for {benchmark_name}")
        return None
    return df.loc[available].T.dropna(how="all").mean(axis=1, skipna=True)


def scatter_panel(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    dialog_label: str,
    mteb_label: str,
    color: str,
) -> tuple[float, float, float, float, int]:
    common = x.index.intersection(y.index)
    valid = x[common].notna() & y[common].notna()
    xv, yv = x[common][valid], y[common][valid]

    spearman_r, spearman_p = spearmanr(xv, yv)
    pearson_r, pearson_p = pearsonr(xv, yv)

    ax.scatter(
        yv * 100,
        xv * 100,
        alpha=0.65,
        s=35,
        color=color,
        linewidths=0.3,
        edgecolors="white",
    )

    m, b = np.polyfit(yv, xv, 1)
    xs = np.linspace(yv.min(), yv.max(), 100)
    ax.plot(xs * 100, (m * xs + b) * 100, "--", color="black", linewidth=1.0, alpha=0.6)

    # Annotate top-5 rank outliers
    dialog_rank = xv.rank(ascending=False)
    mteb_rank = yv.rank(ascending=False)
    for model in (dialog_rank - mteb_rank).abs().nlargest(5).index:
        label = model.split("/")[-1][:16]
        ax.annotate(
            label,
            (yv[model] * 100, xv[model] * 100),
            fontsize=5.5,
            xytext=(3, 2),
            textcoords="offset points",
            color="#444444",
        )

    ax.set_xlabel(f"{mteb_label} Mean (%)", fontsize=10)
    ax.set_ylabel(f"{dialog_label} Mean (%)", fontsize=10)
    ax.set_title(
        f"{dialog_label} vs {mteb_label}\nSpearman ρ = {spearman_r:.3f}  (n={len(xv)})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)

    return spearman_r, spearman_p, pearson_r, pearson_p, len(xv)


def main():
    results: list[dict] = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (suffix, dialog_bench) in zip(axes, BENCHMARKS.items()):
        mteb_bench = MTEB_COUNTERPARTS[suffix]
        print(f"\n--- {dialog_bench} vs {mteb_bench} ---")

        print(f"Loading {dialog_bench}...")
        tasks = load_unique_tasks(dialog_bench)
        task_names = [t.metadata.name for t in tasks]
        dialog_df = load_score_df(tasks)
        complete = get_complete_models(dialog_df, task_names)
        print(f"  {len(complete)} complete models")
        available = [t for t in task_names if t in dialog_df.index]
        dialog_means = dialog_df.loc[available, complete].mean(axis=0, skipna=True)

        print(f"Loading {mteb_bench}...")
        mteb_means = load_general_mteb_means(mteb_bench)
        if mteb_means is None:
            axes_idx = list(BENCHMARKS.keys()).index(suffix)
            axes[axes_idx].set_visible(False)
            continue

        common = dialog_means.index.intersection(mteb_means.index)
        print(f"  {len(common)} models with scores on both")
        if len(common) < MIN_OVERLAP:
            print(f"  Too few overlapping models (< {MIN_OVERLAP}), skipping.")
            ax.set_visible(False)
            continue

        sr, sp, pr, pp, n = scatter_panel(
            ax,
            dialog_means,
            mteb_means,
            dialog_label=f"DialogMTEB ({suffix})",
            mteb_label=mteb_bench,
            color=COLORS[suffix],
        )

        print(f"  Spearman ρ = {sr:.4f}  (p={sp:.2e})")
        print(f"  Pearson  r = {pr:.4f}  (p={pp:.2e})")

        results.append(
            {
                "comparison": f"DialogMTEB ({suffix}) vs {mteb_bench}",
                "n": n,
                "spearman_r": sr,
                "spearman_p": sp,
                "pearson_r": pr,
                "pearson_p": pp,
            }
        )

    fig.suptitle("DialogMTEB vs General MTEB — Rank Correlation", fontsize=12, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "plot_mteb_rank_correlation.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
    plt.close(fig)

    if not results:
        return

    # LaTeX table
    tex_lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Rank correlation between DialogMTEB and general MTEB. "
        "A lower value indicates dialog performance reflects distinct capabilities.}",
        "    \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    \\textbf{Comparison} & \\textbf{N} "
        "& \\textbf{Spearman $\\rho$} & \\textbf{Pearson $r$} & \\textbf{p-value} \\\\",
        "    \\midrule",
    ]
    for row in results:
        tex_lines.append(
            f"    {row['comparison']} & {row['n']} "
            f"& {row['spearman_r']:.3f} & {row['pearson_r']:.3f} "
            f"& {row['spearman_p']:.2e} \\\\"
        )
    tex_lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    \\label{tab:dialogmteb_mteb_corr}",
        "\\end{table}",
    ]
    tex_out = OUT_DIR / "dialogmteb_mteb_rank_correlation.tex"
    tex_out.write_text("\n".join(tex_lines))
    print(f"LaTeX table written to {tex_out}")


if __name__ == "__main__":
    main()
