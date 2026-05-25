"""Plot: model parameter count vs DialogMTEB mean score (efficiency frontier).

Shows eng and multilingual variants side by side.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    BENCHMARKS,
    OUT_DIR,
    fetch_model_meta,
    get_complete_models,
    load_score_df,
    load_unique_tasks,
)


def build_df(benchmark_name: str) -> pd.DataFrame:
    tasks = load_unique_tasks(benchmark_name)
    task_names = [t.metadata.name for t in tasks]
    score_df = load_score_df(tasks)
    complete = get_complete_models(score_df, task_names)
    available = [t for t in task_names if t in score_df.index]
    means = score_df.loc[available, complete].mean(axis=0, skipna=True)

    rows = []
    for model_name in means.index:
        meta = fetch_model_meta(model_name)
        rows.append(
            {
                "model": model_name,
                "mean": means[model_name] * 100,
                "n_parameters": meta["n_parameters"],
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["n_parameters"])
    df["log_params"] = np.log10(df["n_parameters"])
    return df


def plot_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    ax.scatter(
        df["log_params"],
        df["mean"],
        color="#4e79a7",
        alpha=0.75,
        s=50,
        linewidths=0.3,
        edgecolors="white",
    )

    # Pareto frontier
    df_sorted = df.sort_values("log_params")
    best_mean = -np.inf
    pareto_mask = []
    for _, row in df_sorted.iterrows():
        if row["mean"] > best_mean:
            best_mean = row["mean"]
            pareto_mask.append(True)
        else:
            pareto_mask.append(False)
    pareto = df_sorted[pareto_mask]
    ax.plot(
        pareto["log_params"],
        pareto["mean"],
        color="#333333",
        linewidth=1.2,
        linestyle="--",
        alpha=0.6,
        zorder=0,
    )

    # Annotate all Pareto frontier points
    for _, row in pareto.iterrows():
        label = row["model"].split("/")[-1]
        if len(label) > 20:
            label = label[:18] + ".."
        ax.annotate(
            label,
            (row["log_params"], row["mean"]),
            fontsize=6.5,
            xytext=(4, 2),
            textcoords="offset points",
        )

    xtick_labels = {6: "1M", 7: "10M", 8: "100M", 9: "1B", 10: "10B"}
    xticks = sorted(xtick_labels)
    ax.set_xticks(xticks)
    ax.set_xticklabels([xtick_labels[x] for x in xticks])
    ax.set_xlabel("Number of Parameters", fontsize=10)
    ax.set_ylabel("DialogMTEB Mean Score (%)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (suffix, benchmark_name) in zip(axes, BENCHMARKS.items()):
        print(f"Loading {benchmark_name}...")
        df = build_df(benchmark_name)
        print(f"  {len(df)} models with parameter info")
        plot_panel(ax, df, f"Model Size vs. DialogMTEB ({suffix})")

    fig.suptitle("Model Size vs. DialogMTEB Performance", fontsize=13)
    fig.tight_layout()

    out = OUT_DIR / "plot_size_vs_performance.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
