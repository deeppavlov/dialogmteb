"""Compute Spearman rank correlation between DialogMTEB and general MTEB scores.

Tests whether models that excel on dialog tasks also excel on general MTEB,
or whether dialog understanding is a distinct capability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    OUT_DIR,
    cache,
    load_unique_tasks,
    load_score_df,
    get_complete_models,
    RUN_MODELS,
)

import mteb

MIN_OVERLAP = 5  # minimum models with scores on both benchmarks


def load_mteb_scores(tasks: list) -> pd.DataFrame:
    results = cache.load_results(tasks=tasks, models=RUN_MODELS)
    return results.to_dataframe().set_index("task_name")


def main():
    # DialogMTEB — only run.py models with complete results
    print("Loading DialogMTEB results...")
    dialog_tasks = load_unique_tasks()
    dialog_task_names = [t.metadata.name for t in dialog_tasks]
    dialog_df = load_score_df(dialog_tasks)

    complete = get_complete_models(dialog_df, dialog_task_names)
    print(f"  run.py models with all 28 tasks complete: {len(complete)}")

    available_tasks = [t for t in dialog_task_names if t in dialog_df.index]
    dialog_means = dialog_df.loc[available_tasks, complete].mean(axis=0, skipna=True)
    print(f"  Using {len(dialog_means)} models for correlation")

    # General MTEB
    print("Loading MTEB(eng) results...")
    try:
        mteb_bench = mteb.get_benchmark("MTEB(eng)")
        mteb_tasks = mteb_bench.tasks
        mteb_df = load_mteb_scores(mteb_tasks)
        mteb_task_names = [t.metadata.name for t in mteb_tasks]
        mteb_means = (
            mteb_df.loc[[t for t in mteb_task_names if t in mteb_df.index]]
            .T.dropna(how="all")
            .mean(axis=1, skipna=True)
        )
        print(f"  {len(mteb_means)} models with MTEB(eng) scores")
    except Exception as e:
        print(f"  Failed: {e}")
        return

    # Find overlapping models (run.py complete models that also have MTEB scores)
    common = dialog_means.index.intersection(mteb_means.index)
    print(f"  {len(common)} run.py models with scores on both benchmarks")

    if len(common) < MIN_OVERLAP:
        print(
            f"Too few overlapping models (< {MIN_OVERLAP}). "
            "Run more models or download results."
        )
        return

    x = dialog_means[common]
    y = mteb_means[common]

    # Drop any NaN
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]

    spearman_r, spearman_p = spearmanr(x, y)
    pearson_r, pearson_p = pearsonr(x, y)

    print("\n" + "=" * 55)
    print("DialogMTEB vs MTEB(eng) Rank Correlation")
    print("=" * 55)
    print(f"  Models compared : {len(x)}")
    print(f"  Spearman ρ      : {spearman_r:.4f}  (p={spearman_p:.2e})")
    print(f"  Pearson r       : {pearson_r:.4f}  (p={pearson_p:.2e})")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        y * 100,
        x * 100,
        alpha=0.6,
        s=30,
        color="#4e79a7",
        linewidths=0.3,
        edgecolors="white",
    )

    # Trend line
    m, b = np.polyfit(y, x, 1)
    xs = np.linspace(y.min(), y.max(), 100)
    ax.plot(xs * 100, (m * xs + b) * 100, "r--", linewidth=1.2, alpha=0.7)

    # Annotate outliers (largest rank difference)
    dialog_rank = x.rank(ascending=False)
    mteb_rank = y.rank(ascending=False)
    rank_diff = (dialog_rank - mteb_rank).abs()
    for model in rank_diff.nlargest(5).index:
        label = model.split("/")[-1][:18]
        ax.annotate(
            label,
            (y[model] * 100, x[model] * 100),
            fontsize=6,
            xytext=(3, 2),
            textcoords="offset points",
        )

    ax.set_xlabel("MTEB(eng) Mean Score (%)", fontsize=11)
    ax.set_ylabel("DialogMTEB Mean Score (%)", fontsize=11)
    ax.set_title(
        f"DialogMTEB vs MTEB(eng)\nSpearman ρ = {spearman_r:.3f}  (n={len(x)})",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)

    out = OUT_DIR / "plot_mteb_rank_correlation.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"\nScatter plot saved to {out}")

    # LaTeX table snippet
    tex_lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Rank correlation between DialogMTEB and MTEB(eng). "
        "A lower correlation indicates that dialog performance reflects distinct capabilities.}",
        "    \\begin{tabular}{lcc}",
        "    \\toprule",
        "    \\textbf{Metric} & \\textbf{Value} & \\textbf{p-value} \\\\",
        "    \\midrule",
        f"    Spearman $\\rho$ & {spearman_r:.3f} & {spearman_p:.2e} \\\\",
        f"    Pearson $r$ & {pearson_r:.3f} & {pearson_p:.2e} \\\\",
        f"    Models compared & \\multicolumn{{2}}{{c}}{{{len(x)}}} \\\\",
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
