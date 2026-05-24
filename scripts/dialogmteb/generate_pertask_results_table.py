"""Generate per-task × all-models results table for DialogMTEB.

Uses longtable + pdflscape so the table spans multiple landscape pages.
Rows = models (sorted by Borda count), columns = tasks (grouped by type).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mteb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    BENCHMARKS,
    cache,
    RUN_MODELS,
    TYPE_ORDER as TASK_TYPE_ORDER,
    TYPE_ABBREV as TASK_TYPE_ABBREV,
    TYPE_COLORS,
)

# Map type name → xcolor-compatible hex (strip leading #)
_TYPE_XCOLOR: dict[str, str] = {
    k: v.lstrip("#") for k, v in TYPE_COLORS.items()
}


def _escape(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def _shorten(name: str, max_len: int = 22) -> str:
    """Shorten a task name for a rotated column header."""
    cuts = [
        "DialogMTEB", "MASSIVE", "WebLINX",
    ]
    for c in cuts:
        if name.startswith(c):
            name = name[len(c):].lstrip()
    name = name.replace("Classification", "Clf.").replace("multilingual", "ml")
    if len(name) > max_len:
        name = name[: max_len - 1] + "."
    return name


def _load_results(tasks: list) -> pd.DataFrame:
    results = cache.load_results(tasks=tasks, models=RUN_MODELS)
    return results.to_dataframe().set_index("task_name")


def _build_task_type_map(tasks: list) -> dict[str, str]:
    return {t.metadata.name: t.metadata.type for t in tasks}


def _borda(df: pd.DataFrame, task_names: list[str]) -> pd.Series:
    available = [t for t in task_names if t in df.index]
    scores = pd.Series(0.0, index=df.columns)
    for task in available:
        row = df.loc[task].dropna().astype(float)
        if row.empty:
            continue
        ranks = row.rank(ascending=False, method="min")
        scores[row.index] += len(row) - ranks + 1
    return scores


def generate_table(benchmark_name: str, suffix: str) -> str:
    benchmark = mteb.get_benchmark(benchmark_name)
    seen: set[str] = set()
    unique_tasks: list = []
    for task in benchmark.tasks:
        if task.metadata.name not in seen:
            seen.add(task.metadata.name)
            unique_tasks.append(task)

    task_type_map = _build_task_type_map(unique_tasks)
    df = _load_results(unique_tasks)

    # Order tasks by type group
    ordered_tasks: list[str] = []
    for ttype in TASK_TYPE_ORDER:
        for t in unique_tasks:
            name = t.metadata.name
            if task_type_map.get(name) == ttype and name in df.index:
                ordered_tasks.append(name)
    # Fallback: any task not yet included
    for t in unique_tasks:
        name = t.metadata.name
        if name in df.index and name not in ordered_tasks:
            ordered_tasks.append(name)

    # Sort models by borda count
    borda_scores = _borda(df, ordered_tasks)
    model_order = (
        borda_scores[df.columns.intersection(borda_scores.index)]
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Per-task best score (for bold highlighting)
    task_best: dict[str, float] = {
        t: float(df.loc[t].max()) for t in ordered_tasks
    }

    n_tasks = len(ordered_tasks)
    n_cols = 2 + n_tasks + 1  # model + rank + tasks... + mean

    # Column spec: left-aligned model, right-aligned rank, narrow centred scores, right mean
    col_spec = "l@{\\hspace{4pt}}r" + "@{\\hspace{1pt}}r" * n_tasks + "@{\\hspace{4pt}}r"

    # Rotated task headers coloured by type
    def task_header(t: str) -> str:
        ttype = task_type_map.get(t, "")
        color = _TYPE_XCOLOR.get(ttype, "000000")
        short = _escape(_shorten(t))
        return f"\\rotatebox{{90}}{{\\textcolor[HTML]{{{color}}}{{\\textit{{{short}}}}}}}"

    task_headers = " & ".join(task_header(t) for t in ordered_tasks)

    # Type-group subheader row (multicolumn spans)
    group_spans: list[tuple[str, int]] = []
    for ttype in TASK_TYPE_ORDER:
        cnt = sum(1 for t in ordered_tasks if task_type_map.get(t) == ttype)
        if cnt:
            group_spans.append((ttype, cnt))

    def group_header_row() -> str:
        # model col + rank col (blank), then spans
        parts = ["", ""]
        for ttype, cnt in group_spans:
            color = _TYPE_XCOLOR.get(ttype, "000000")
            abbrev = _escape(TASK_TYPE_ABBREV.get(ttype, ttype))
            parts.append(
                f"\\multicolumn{{{cnt}}}{{c}}{{\\textcolor[HTML]{{{color}}}{{\\textbf{{{abbrev}}}}}}}"
            )
        parts.append("")  # mean col
        return " & ".join(parts) + " \\\\"

    def cmidrule_row() -> str:
        rules = []
        col = 3  # 1-based: col1=model, col2=rank, col3..
        for _, cnt in group_spans:
            rules.append(f"\\cmidrule(lr){{{col}-{col + cnt - 1}}}")
            col += cnt
        return " ".join(rules)

    # Repeated header (for longtable continuation)
    header_block = "\n".join([
        "\\toprule",
        group_header_row(),
        cmidrule_row(),
        f"\\textbf{{Model}} & \\textbf{{Rk}} & {task_headers} & \\textbf{{Mean}} \\\\",
        "\\midrule",
    ])

    lines = [
        "% Requires: \\usepackage{pdflscape,longtable,booktabs,xcolor}",
        "\\begin{landscape}",
        "{\\tiny",
        "\\setlength{\\tabcolsep}{2pt}",
        f"\\begin{{longtable}}{{{col_spec}}}",
        (
            f"\\caption{{Per-task results on {benchmark_name} ({len(model_order)} models, "
            f"{n_tasks} tasks). Sorted by Borda count. "
            f"\\textbf{{Bold}} = best per task. "
            + "Task types: "
            + ", ".join(
                f"\\textcolor[HTML]{{{_TYPE_XCOLOR.get(t, '000000')}}}{{{_escape(TASK_TYPE_ABBREV.get(t, t))}}}"
                f" = {_escape(t)}"
                for t, _ in group_spans
            )
            + ".}}\\\\"
        ),
        f"\\label{{tab:pertask-results-{suffix}}}\\\\[2pt]",
        header_block,
        "\\endfirsthead",
        f"\\multicolumn{{{n_cols}}}{{c}}{{\\tablename\\ \\thetable{{}} --- continued from previous page}}\\\\[2pt]",
        header_block,
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{{n_cols}}}{{r}}{{\\textit{{Continued on next page}}}}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]

    for rank, model in enumerate(model_order, 1):
        display = model.split("/")[-1] if "/" in model else model
        if len(display) > 32:
            display = display[:30] + ".."
        display = _escape(display)

        cell_parts: list[str] = []
        raw_vals: list[float] = []
        for t in ordered_tasks:
            val = (
                df.loc[t, model]
                if t in df.index and model in df.columns
                else np.nan
            )
            if pd.isna(val):
                cell_parts.append("-")
            else:
                raw_vals.append(float(val))
                s = f"{float(val) * 100:.1f}"
                if abs(float(val) - task_best[t]) < 5e-4:
                    s = f"\\textbf{{{s}}}"
                cell_parts.append(s)

        mean_str = f"{np.mean(raw_vals) * 100:.1f}" if raw_vals else "-"
        score_str = " & ".join(cell_parts)
        lines.append(f"{display} & {rank} & {score_str} & {mean_str} \\\\")

    lines += [
        "\\end{longtable}",
        "}",  # end \tiny
        "\\end{landscape}",
    ]
    return "\n".join(lines)


def main() -> None:
    for suffix, benchmark_name in BENCHMARKS.items():
        print(f"\nGenerating {benchmark_name}...")
        table = generate_table(benchmark_name, suffix)
        out = Path(__file__).parent / f"dialogmteb_{suffix}_pertask_results_table.tex"
        out.write_text(table)
        print(f"  Written to {out}")


if __name__ == "__main__":
    main()
