"""Plot: model parameter count vs DialogMTEB mean score (efficiency frontier)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    OUT_DIR,
    load_unique_tasks,
    load_score_df,
    model_mean_scores,
    fetch_model_meta,
    get_complete_models,
)


def main():
    tasks = load_unique_tasks()
    task_names = [t.metadata.name for t in tasks]
    score_df = load_score_df(tasks)

    complete = get_complete_models(score_df, task_names)
    print(f"run.py models with all 28 tasks complete: {len(complete)}")

    means = (
        score_df.loc[[t for t in task_names if t in score_df.index], complete]
        .mean(axis=0, skipna=True)
        .sort_values(ascending=False)
    )

    print("Fetching model meta...")
    rows = []
    for model_name in means.index:
        meta = fetch_model_meta(model_name)
        rows.append(
            {
                "model": model_name,
                "mean": means[model_name] * 100,
                "n_parameters": meta["n_parameters"],
                "open_weights": meta["open_weights"],
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["n_parameters"])
    df["log_params"] = np.log10(df["n_parameters"])

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = df["open_weights"].map(
        {True: "#4e79a7", False: "#e15759", None: "#bab0ac"}
    )
    ax.scatter(
        df["log_params"],
        df["mean"],
        c=colors,
        alpha=0.75,
        s=50,
        linewidths=0.3,
        edgecolors="white",
    )

    # Pareto frontier (highest mean for each param bucket)
    df_sorted = df.sort_values("log_params")
    pareto_mask = []
    best_mean = -np.inf
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

    # Annotate top-5 by mean
    for _, row in df.nlargest(5, "mean").iterrows():
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

    xticks = [6, 7, 8, 9, 10]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"$10^{{{x}}}$" for x in xticks])
    ax.set_xlabel("Number of Parameters", fontsize=11)
    ax.set_ylabel("DialogMTEB Mean Score (%)", fontsize=11)
    ax.set_title("Model Size vs. DialogMTEB Performance", fontsize=12)
    ax.grid(True, alpha=0.3)

    legend_elements = [
        mpatches.Patch(color="#4e79a7", label="Open weights"),
        mpatches.Patch(color="#e15759", label="Proprietary"),
        mpatches.Patch(color="#bab0ac", label="Unknown"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    out = OUT_DIR / "plot_size_vs_performance.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
