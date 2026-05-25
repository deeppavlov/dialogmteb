"""Compute per-task evaluation time from cached result JSON files.

Outputs a LaTeX table and prints a summary.
Times are extracted from the 'evaluation_time' field in each result JSON.
Reports median time per task across all models that have it.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    OUT_DIR,
    RESULT_CACHE_PATH,
    TYPE_ABBREV,
    TYPE_ORDER,
    load_unique_tasks,
)


def load_eval_times(results_dir: Path, task_names: set[str]) -> dict[str, list[float]]:
    """Return {task_name: [eval_time_seconds, ...]} from all result JSONs."""
    times: dict[str, list[float]] = defaultdict(list)
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for rev_dir in model_dir.iterdir():
            if not rev_dir.is_dir():
                continue
            for jf in rev_dir.glob("*.json"):
                task = jf.stem
                if task not in task_names:
                    continue
                try:
                    data = json.loads(jf.read_text())
                    t = data.get("evaluation_time")
                    if t is not None and t > 0:
                        times[task].append(float(t))
                except Exception:
                    pass
    return times


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


def main():
    tasks = load_unique_tasks()
    task_names = {t.metadata.name for t in tasks}
    task_type_map = {t.metadata.name: t.metadata.type for t in tasks}

    results_dir = RESULT_CACHE_PATH / "results"
    print(f"Scanning {results_dir} for evaluation times...")
    times = load_eval_times(results_dir, task_names)

    rows = []
    for task in tasks:
        name = task.metadata.name
        task_times = times.get(name, [])
        rows.append(
            {
                "name": name,
                "task_type": task_type_map.get(name, ""),
                "n_measurements": len(task_times),
                "median_s": float(np.median(task_times)) if task_times else np.nan,
                "mean_s": float(np.mean(task_times)) if task_times else np.nan,
                "min_s": float(np.min(task_times)) if task_times else np.nan,
                "max_s": float(np.max(task_times)) if task_times else np.nan,
            }
        )

    df = pd.DataFrame(rows)

    # Sort by type order then name
    type_order_idx = {t: i for i, t in enumerate(TYPE_ORDER)}
    df["_order"] = df["task_type"].map(lambda t: type_order_idx.get(t, 99))
    df = df.sort_values(["_order", "name"]).drop(columns="_order")

    total_median = df["median_s"].sum()
    print(
        f"\nTotal estimated evaluation time (sum of per-task medians): {fmt_time(total_median)}"
    )
    print(f"\nPer-task breakdown:")
    for _, row in df.iterrows():
        t = fmt_time(row["median_s"]) if not np.isnan(row["median_s"]) else "--"
        print(f"  {row['name']:45s} {t:>8s}  (n={row['n_measurements']})")

    # LaTeX table
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\begin{tabular}{llccc}",
        "    \\toprule",
        "    \\textbf{Task} & \\textbf{Type} & \\textbf{Median} "
        "& \\textbf{Min} & \\textbf{Max} \\\\",
        "    \\midrule",
    ]

    current_type = None
    for _, row in df.iterrows():
        if row["task_type"] != current_type:
            current_type = row["task_type"]
            abbrev = TYPE_ABBREV.get(current_type, current_type)
            lines.append(f"    \\multicolumn{{5}}{{l}}{{\\textit{{{abbrev}}}}} \\\\")

        name = row["name"].replace("_", "\\_")
        med = fmt_time(row["median_s"]) if not np.isnan(row["median_s"]) else "--"
        mn = fmt_time(row["min_s"]) if not np.isnan(row["min_s"]) else "--"
        mx = fmt_time(row["max_s"]) if not np.isnan(row["max_s"]) else "--"
        abbrev = TYPE_ABBREV.get(row["task_type"], row["task_type"])
        lines.append(f"    {name} & {abbrev} & {med} & {mn} & {mx} \\\\")

    if not np.isnan(total_median):
        lines.append("    \\midrule")
        lines.append(
            f"    \\textbf{{Total}} & & \\textbf{{{fmt_time(total_median)}}} & & \\\\"
        )

    lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    \\caption{Evaluation time per DialogMTEB task. "
        "Times are measured on a single model inference pass and reported as the median "
        "across all evaluated models. Total = sum of medians.}",
        "    \\label{tab:dialogmteb_eval_time}",
        "\\end{table}",
    ]

    out = OUT_DIR / "dialogmteb_eval_time.tex"
    out.write_text("\n".join(lines))
    print(f"\nLaTeX table written to {out}")

    csv_out = OUT_DIR / "dialogmteb_eval_time.csv"
    df.to_csv(csv_out, index=False)
    print(f"CSV written to {csv_out}")


if __name__ == "__main__":
    main()
