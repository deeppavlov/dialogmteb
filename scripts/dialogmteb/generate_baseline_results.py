"""Generate LaTeX table of baseline model results on DialogMTEB.

Extracts scores for:
  - mteb/baseline-random-encoder  (random lower bound)
  - mteb/baseline-bm25s           (lexical upper-bound baseline)

And shows them alongside per-task-type averages for context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    OUT_DIR,
    TYPE_ABBREV,
    TYPE_ORDER,
    load_unique_tasks,
    load_score_df,
)

BASELINES = {
    "mteb/baseline-random-encoder": "Random",
    "mteb/baseline-bm25s": "BM25s",
}


def main():
    tasks = load_unique_tasks()
    task_names = [t.metadata.name for t in tasks]
    task_type_map = {t.metadata.name: t.metadata.type for t in tasks}

    score_df = load_score_df(tasks, models=list(BASELINES.keys()))

    # Check which baselines are present
    found = {
        model: label for model, label in BASELINES.items() if model in score_df.columns
    }
    if not found:
        print("No baseline models found in result cache.")
        print("Available models:", list(score_df.columns)[:10], "...")
        return

    print(f"Found baselines: {list(found.values())}")

    # Build results dataframe: rows = tasks, cols = baselines
    available_tasks = [t for t in task_names if t in score_df.index]
    baseline_df = score_df.loc[available_tasks, list(found.keys())].copy() * 100
    baseline_df.columns = [found[m] for m in found]
    baseline_df.index.name = "task"

    # Per-task-type section averages
    type_rows = {}
    for ttype in TYPE_ORDER:
        group_tasks = [t for t in available_tasks if task_type_map.get(t) == ttype]
        if group_tasks:
            type_rows[ttype] = baseline_df.loc[group_tasks].mean(skipna=True)

    # Overall mean
    overall = baseline_df.mean(skipna=True)

    print("\nBaseline results (%):")
    print(baseline_df.to_string())
    print(f"\nOverall mean: {overall.to_dict()}")

    # Sort tasks by type then name
    type_order_idx = {t: i for i, t in enumerate(TYPE_ORDER)}
    sorted_tasks = sorted(
        available_tasks,
        key=lambda t: (type_order_idx.get(task_type_map.get(t, ""), 99), t),
    )

    n_baselines = len(found)
    col_spec = "ll" + "c" * n_baselines
    header_models = " & ".join(f"\\textbf{{{label}}}" for label in found.values())

    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Baseline model results on DialogMTEB (\\%). "
        "\\textit{Random} = random encoder (lower bound). "
        "\\textit{BM25s} = BM25 lexical retrieval (strong lexical baseline). "
        "Dashes indicate tasks where the baseline cannot be applied.}",
        "    \\resizebox{\\linewidth}{!}{",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    \\textbf{{Task}} & \\textbf{{Type}} & {header_models} \\\\",
        "    \\midrule",
    ]

    current_type = None
    for task in sorted_tasks:
        ttype = task_type_map.get(task, "")
        if ttype != current_type:
            current_type = ttype
            abbrev = TYPE_ABBREV.get(ttype, ttype)
            # Type-average row
            if ttype in type_rows:
                avg_cells = " & ".join(
                    f"{v:.1f}" if not np.isnan(v) else "--" for v in type_rows[ttype]
                )
                lines.append(
                    f"    \\multicolumn{{{2 + n_baselines}}}{{l}}"
                    f"{{\\textit{{{abbrev}}} — avg: {avg_cells}}} \\\\"
                )
            else:
                lines.append(
                    f"    \\multicolumn{{{2 + n_baselines}}}{{l}}"
                    f"{{\\textit{{{abbrev}}}}} \\\\"
                )

        name = task.replace("_", "\\_")
        abbrev = TYPE_ABBREV.get(ttype, ttype)
        scores_row = baseline_df.loc[task]
        cells = " & ".join(f"{v:.1f}" if not np.isnan(v) else "--" for v in scores_row)
        lines.append(f"    {name} & {abbrev} & {cells} \\\\")

    # Overall row
    lines.append("    \\midrule")
    overall_cells = " & ".join(
        f"\\textbf{{{v:.1f}}}" if not np.isnan(v) else "--" for v in overall
    )
    lines.append(f"    \\textbf{{Overall}} & & {overall_cells} \\\\")

    lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    }",
        "    \\label{tab:dialogmteb_baselines}",
        "\\end{table*}",
    ]

    out = OUT_DIR / "dialogmteb_baseline_results.tex"
    out.write_text("\n".join(lines))
    print(f"\nLaTeX table written to {out}")


if __name__ == "__main__":
    main()
