"""Generate LaTeX table of language-stratified DialogMTEB results.

Groups tasks by language coverage:
  English-only  (only 'eng')
  Multilingual  (2+ languages, includes English)
  Russian       (rus)
  Chinese       (cmn / cmo)
  Other
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    OUT_DIR,
    load_unique_tasks,
    load_score_df,
    compute_borda,
)

TOP_N = 30


def classify_language(languages: list[str]) -> str:
    langs = set(lang.split("-")[0] for lang in languages)
    if langs == {"eng"}:
        return "English"
    if "rus" in langs and len(langs) == 1:
        return "Russian"
    if langs <= {"cmn", "cmo"}:
        return "Chinese"
    return "Multilingual"


def format_score(val: float, best: float) -> str:
    if np.isnan(val):
        return "--"
    s = f"{val:.1f}"
    return f"\\textbf{{{s}}}" if abs(val - best) < 5e-4 else s


def main():
    tasks = load_unique_tasks()
    task_names = [t.metadata.name for t in tasks]

    lang_group: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        group = classify_language(task.metadata.languages or [])
        lang_group[group].append(task.metadata.name)

    print("Language groups:")
    for g, ts in lang_group.items():
        print(f"  {g} ({len(ts)}): {ts}")

    score_df = load_score_df(tasks)
    borda = compute_borda(score_df, task_names)

    mt = score_df.loc[[t for t in task_names if t in score_df.index]].T.dropna(
        how="all"
    )
    mt["borda"] = borda
    mt["mean_all"] = mt[task_names].mean(axis=1, skipna=True) * 100

    group_order = ["English", "Russian", "Chinese", "Multilingual"]
    for group in group_order:
        tasks_in_group = [t for t in lang_group.get(group, []) if t in mt.columns]
        if tasks_in_group:
            mt[f"lang_{group}"] = mt[tasks_in_group].mean(axis=1, skipna=True) * 100
        else:
            mt[f"lang_{group}"] = np.nan

    mt = mt.sort_values("borda", ascending=False).head(TOP_N)

    # Best per column
    lang_cols = [f"lang_{g}" for g in group_order]
    bests = {col: mt[col].max() for col in ["mean_all"] + lang_cols}

    # Task counts per group
    group_counts = {
        g: len([t for t in lang_group.get(g, []) if t in score_df.index])
        for g in group_order
    }

    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        f"    \\caption{{Language-stratified DialogMTEB results (top {TOP_N} models by Borda count). "
        "Scores are means (\\%) within each language group. "
        "\\textbf{Bold} = best in column.}}",
        "    \\resizebox{\\linewidth}{!}{",
        "    \\begin{tabular}{lccccc}",
        "    \\toprule",
        "    \\textbf{Model} & \\textbf{All} & \\textbf{English} & "
        "\\textbf{Russian} & \\textbf{Chinese} & \\textbf{Multilingual} \\\\",
        "    \\textcolor{gray}{\\# tasks} & "
        + " & ".join(
            [
                f"\\textcolor{{gray}}{{({len(task_names)})}}",
            ]
            + [f"\\textcolor{{gray}}{{({group_counts[g]})}}" for g in group_order]
        )
        + " \\\\",
        "    \\midrule",
    ]

    for rank, (model_name, row) in enumerate(mt.iterrows(), 1):
        display = model_name.split("/")[-1] if "/" in model_name else model_name
        if len(display) > 40:
            display = display[:37] + "..."
        display = display.replace("_", "\\_")

        cells = [format_score(row["mean_all"], bests["mean_all"])]
        for g in group_order:
            col = f"lang_{g}"
            cells.append(format_score(row[col], bests[col]))

        lines.append(f"    {display} & {' & '.join(cells)} \\\\")

    lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    }",
        "    \\label{tab:dialogmteb_lang_results}",
        "\\end{table*}",
    ]

    out = OUT_DIR / "dialogmteb_language_results.tex"
    out.write_text("\n".join(lines))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
