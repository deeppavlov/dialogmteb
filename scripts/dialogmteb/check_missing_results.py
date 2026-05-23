"""Report which run.py models have missing results on DialogMTEB tasks.

Outputs:
  - A console table showing coverage per model
  - dialogmteb_missing_results.tex  LaTeX table for the paper appendix
  - dialogmteb_missing_results.csv  Machine-readable version
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    OUT_DIR,
    TYPE_ABBREV,
    TYPE_ORDER,
    load_unique_tasks,
    load_score_df,
    RUN_MODELS,
)


def main():
    tasks = load_unique_tasks()
    task_names = [t.metadata.name for t in tasks]
    task_type_map = {t.metadata.name: t.metadata.type for t in tasks}
    n_tasks = len(task_names)

    score_df = load_score_df(tasks)
    available_tasks = [t for t in task_names if t in score_df.index]

    run_models = RUN_MODELS
    print(f"Models in run.py:  {len(run_models)}")
    print(f"Tasks in benchmark: {n_tasks}")
    print()

    rows = []
    for model in run_models:
        if model not in score_df.columns:
            coverage = 0
            missing = task_names[:]
        else:
            col = score_df.loc[available_tasks, model]
            coverage = int(col.notna().sum())
            missing = [t for t in available_tasks if pd.isna(col[t])]
            # also mark tasks not in score_df at all as missing
            missing += [t for t in task_names if t not in available_tasks]

        rows.append(
            {
                "model": model,
                "coverage": coverage,
                "missing_count": n_tasks - coverage,
                "missing_tasks": missing,
                "complete": coverage == n_tasks,
            }
        )

    rows.sort(key=lambda r: (-r["coverage"], r["model"]))

    # Console summary
    complete = [r for r in rows if r["complete"]]
    incomplete = [r for r in rows if not r["complete"]]

    print(f"Complete ({len(complete)}/{len(run_models)} models):")
    for r in complete:
        print(f"  ✓ {r['coverage']:2d}/{n_tasks}  {r['model']}")

    print(f"\nIncomplete ({len(incomplete)}/{len(run_models)} models):")
    for r in incomplete:
        missing_str = ", ".join(r["missing_tasks"]) if r["missing_tasks"] else "--"
        print(f"  ✗ {r['coverage']:2d}/{n_tasks}  {r['model']}")
        for t in r["missing_tasks"]:
            ttype = TYPE_ABBREV.get(task_type_map.get(t, ""), "?")
            print(f"           [{ttype}] {t}")

    # CSV
    csv_rows = []
    for r in rows:
        csv_rows.append(
            {
                "model": r["model"],
                "coverage": r["coverage"],
                "missing_count": r["missing_count"],
                "missing_tasks": "; ".join(r["missing_tasks"]),
                "complete": r["complete"],
            }
        )
    csv_out = OUT_DIR / "dialogmteb_missing_results.csv"
    pd.DataFrame(csv_rows).to_csv(csv_out, index=False)
    print(f"\nCSV written to {csv_out}")

    # LaTeX table
    # Sort tasks by type order for column headers
    type_order_idx = {t: i for i, t in enumerate(TYPE_ORDER)}
    sorted_tasks = sorted(
        available_tasks,
        key=lambda t: (type_order_idx.get(task_type_map.get(t, ""), 99), t),
    )

    short_tasks = [t[:18] + ".." if len(t) > 20 else t for t in sorted_tasks]
    short_tasks_tex = [s.replace("_", "\\_") for s in short_tasks]
    header = " & ".join(f"\\rotatebox{{90}}{{\\tiny {s}}}" for s in short_tasks_tex)

    col_spec = "l" + "c" * (len(sorted_tasks) + 1)
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Missing results for run.py models on DialogMTEB. "
        "\\ding{51} = result available, \\ding{55} = missing.}",
        "    \\resizebox{\\linewidth}{!}{",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    \\textbf{{Model}} & \\textbf{{Cov.}} & {header} \\\\",
        "    \\midrule",
    ]

    for r in rows:
        model_short = r["model"].split("/")[-1].replace("_", "\\_")
        if len(model_short) > 35:
            model_short = model_short[:32] + "..."
        cov = f"{r['coverage']}/{n_tasks}"

        if r["model"] not in score_df.columns:
            cells = ["\\ding{55}"] * len(sorted_tasks)
        else:
            col = score_df.loc[available_tasks, r["model"]]
            cells = []
            for t in sorted_tasks:
                if t in col.index and not pd.isna(col[t]):
                    cells.append("\\ding{51}")
                else:
                    cells.append("\\ding{55}")

        row_str = " & ".join(cells)
        bold_cov = f"\\textbf{{{cov}}}" if r["complete"] else cov
        lines.append(f"    {model_short} & {bold_cov} & {row_str} \\\\")

    lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    }",
        "    \\label{tab:dialogmteb_missing}",
        "\\end{table*}",
    ]

    tex_out = OUT_DIR / "dialogmteb_missing_results.tex"
    tex_out.write_text("\n".join(lines))
    print(f"LaTeX table written to {tex_out}")


if __name__ == "__main__":
    main()
