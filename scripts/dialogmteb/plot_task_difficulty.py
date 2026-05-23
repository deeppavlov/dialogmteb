"""Plot: task difficulty (mean score per task across all evaluated models)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    OUT_DIR,
    TYPE_COLORS,
    TYPE_ABBREV,
    load_unique_tasks,
    load_score_df,
)


def main():
    tasks = load_unique_tasks()
    task_names = [t.metadata.name for t in tasks]
    task_type_map = {t.metadata.name: t.metadata.type for t in tasks}

    score_df = load_score_df(tasks)

    # Mean score per task (% scale), only models that have that task
    task_means = {}
    task_stds = {}
    task_n = {}
    for task in task_names:
        if task in score_df.index:
            vals = score_df.loc[task].dropna().astype(float) * 100
            task_means[task] = vals.mean()
            task_stds[task] = vals.std()
            task_n[task] = len(vals)

    # Sort by mean score ascending (hardest first)
    sorted_tasks = sorted(task_means, key=task_means.get)

    fig, ax = plt.subplots(figsize=(9, 7))

    y_pos = np.arange(len(sorted_tasks))
    means = [task_means[t] for t in sorted_tasks]
    stds = [task_stds.get(t, 0) for t in sorted_tasks]
    colors = [
        TYPE_COLORS.get(task_type_map.get(t, ""), "#888888") for t in sorted_tasks
    ]

    ax.barh(
        y_pos,
        means,
        xerr=stds,
        color=colors,
        alpha=0.85,
        height=0.7,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_tasks, fontsize=8.5)
    ax.set_xlabel("Mean Score ± Std (%)", fontsize=11)
    ax.set_title("DialogMTEB Task Difficulty", fontsize=12)
    ax.set_xlim(0, 105)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    legend_elements = [
        mpatches.Patch(color=TYPE_COLORS[t], label=TYPE_ABBREV[t])
        for t in TYPE_ABBREV
        if t in set(task_type_map.values())
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    fig.tight_layout()
    out = OUT_DIR / "plot_task_difficulty.pdf"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"Saved to {out}")
    print("\nTask means (ascending):")
    for t in sorted_tasks:
        n = task_n.get(t, 0)
        print(f"  {task_means[t]:5.1f}% ± {task_stds.get(t, 0):4.1f}  (n={n:3d})  {t}")


if __name__ == "__main__":
    main()
