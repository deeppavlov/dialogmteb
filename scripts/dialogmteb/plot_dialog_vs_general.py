"""Plot: dialog-specific models vs similarly-sized general embedding models."""

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
    TYPE_COLORS,
    TYPE_ABBREV,
    TYPE_ORDER,
    load_unique_tasks,
    load_score_df,
    fetch_model_meta,
    compute_borda,
)

# Dialog-specific models to highlight
DIALOG_MODELS = {
    "TODBERT/TOD-BERT-MLM-V1": "TOD-BERT",
    "AndrewZeng/futuretod-base-v1.0": "FutureTOD",
}

# Parameter budget for "similar-size" comparison (±50% of ~110M BERT-class)
BERT_SIZE_RANGE = (50_000_000, 200_000_000)


def main():
    tasks = load_unique_tasks()
    task_names = [t.metadata.name for t in tasks]
    task_type_map = {t.metadata.name: t.metadata.type for t in tasks}
    score_df = load_score_df(tasks)

    mt = score_df.loc[[t for t in task_names if t in score_df.index]].T.dropna(
        how="all"
    )
    borda = compute_borda(score_df, task_names)
    mt["mean"] = mt[task_names].mean(axis=1, skipna=True) * 100

    # Per-type averages
    for ttype in TYPE_ORDER:
        cols = [
            t for t in task_names if task_type_map.get(t) == ttype and t in mt.columns
        ]
        if cols:
            mt[f"cat_{ttype}"] = mt[cols].mean(axis=1, skipna=True) * 100

    # Get dialog model rows
    dialog_rows = {}
    for model_name, label in DIALOG_MODELS.items():
        if model_name in mt.index:
            dialog_rows[label] = mt.loc[model_name]

    if not dialog_rows:
        print("No dialog model results found in cache.")
        return

    # Collect general models in similar parameter range
    print("Fetching model metas for general models comparison...")
    general_rows = {}
    checked = 0
    for model_name in borda.sort_values(ascending=False).index:
        if model_name in DIALOG_MODELS:
            continue
        if checked > 200:
            break
        meta = fetch_model_meta(model_name)
        n = meta["n_parameters"]
        if n is not None and BERT_SIZE_RANGE[0] <= n <= BERT_SIZE_RANGE[1]:
            if model_name in mt.index:
                short = model_name.split("/")[-1][:25]
                general_rows[short] = mt.loc[model_name]
        checked += 1

    if not general_rows:
        print("No similarly-sized general models found.")
        return

    # Build comparison dataframe
    all_rows = {**{f"[Dialog] {k}": v for k, v in dialog_rows.items()}, **general_rows}
    cmp_df = pd.DataFrame(all_rows).T
    cmp_df = cmp_df.sort_values("mean", ascending=False).head(20)

    present_types = [t for t in TYPE_ORDER if f"cat_{t}" in cmp_df.columns]
    n_types = len(present_types)

    fig, axes = plt.subplots(
        1, n_types + 1, figsize=(3 * (n_types + 1), 6), sharey=True
    )

    models = cmp_df.index.tolist()
    y_pos = np.arange(len(models))
    is_dialog = ["[Dialog]" in m for m in models]
    bar_colors = ["#e15759" if d else "#4e79a7" for d in is_dialog]

    axes[0].barh(y_pos, cmp_df["mean"], color=bar_colors, alpha=0.85, height=0.7)
    axes[0].set_title("Mean", fontsize=10)
    axes[0].set_xlabel("%", fontsize=9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models, fontsize=7.5)
    axes[0].axvline(
        cmp_df[cmp_df.index.str.startswith("[Dialog]")]["mean"].mean(),
        color="#e15759",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )
    axes[0].axvline(
        cmp_df[~cmp_df.index.str.startswith("[Dialog]")]["mean"].mean(),
        color="#4e79a7",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    for ax, ttype in zip(axes[1:], present_types):
        col = f"cat_{ttype}"
        vals = cmp_df[col].fillna(0)
        ax.barh(y_pos, vals, color=bar_colors, alpha=0.85, height=0.7)
        ax.set_title(TYPE_ABBREV[ttype], fontsize=10)
        ax.set_xlabel("%", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)

    legend_elements = [
        mpatches.Patch(color="#e15759", label="Dialog-specific"),
        mpatches.Patch(color="#4e79a7", label="General (BERT-size)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Dialog-Specific vs. General Models on DialogMTEB\n"
        f"(General = {BERT_SIZE_RANGE[0] // 1e6:.0f}–{BERT_SIZE_RANGE[1] // 1e6:.0f}M params)",
        fontsize=11,
    )

    fig.tight_layout()
    out = OUT_DIR / "plot_dialog_vs_general.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    print(
        f"\nComparison ({len(dialog_rows)} dialog, {len(general_rows)} general models shown):"
    )
    print(cmp_df[["mean"]].to_string())


if __name__ == "__main__":
    main()
