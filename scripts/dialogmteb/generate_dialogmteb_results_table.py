"""Generate LaTeX results table for DialogMTEB benchmark.

Ranks models by Borda count, shows per-category averages.
Reads results via mteb.ResultCache.
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
)

# Dialog-specific models go in a separate category
DIALOG_MODELS = {
    "TODBERT/TOD-BERT-MLM-V1",
    "AndrewZeng/futuretod-base-v1.0",
}


def load_results(tasks: list) -> pd.DataFrame:
    """Load results via ResultCache and return a (task x model) dataframe."""
    results = cache.load_results(tasks=tasks, models=RUN_MODELS)
    df = results.to_dataframe()  # rows = tasks, cols = models + 'task_name'
    df = df.set_index("task_name")
    return df


def build_task_type_map(tasks: list) -> dict[str, str]:
    return {t.metadata.name: t.metadata.type for t in tasks}


def compute_borda_count(df: pd.DataFrame, task_names: list[str]) -> pd.Series:
    available = [t for t in task_names if t in df.index]
    borda = pd.Series(0.0, index=df.columns)
    for task in available:
        row = df.loc[task].dropna()
        if row.empty:
            continue
        ranks = row.rank(ascending=False, method="min")
        n = len(row)
        borda[row.index] += n - ranks + 1
    return borda


def compute_category_averages(
    df: pd.DataFrame,
    task_names: list[str],
    task_type_map: dict[str, str],
) -> dict[str, pd.Series]:
    groups: dict[str, list[str]] = defaultdict(list)
    for task in task_names:
        if task in df.index:
            groups[task_type_map.get(task, "Unknown")].append(task)

    return {
        task_type: df.loc[tasks].mean(axis=0)
        for task_type, tasks in groups.items()
        if tasks
    }


def format_score(value: float, best: float, cat_best: float | None = None) -> str:
    if pd.isna(value):
        return "-"
    val_str = f"{value * 100:.1f}"
    is_global_best = abs(value - best) < 5e-4
    is_cat_best = (
        cat_best is not None and not pd.isna(cat_best) and abs(value - cat_best) < 5e-4
    )
    prefix = "\\cellcolor{gray!20}" if is_cat_best and not is_global_best else ""
    if is_global_best:
        return f"\\textbf{{{val_str}}}"
    return f"{prefix}{val_str}"


def build_rankings(
    df: pd.DataFrame,
    task_names: list[str],
    task_type_map: dict[str, str],
    top_n: int,
) -> pd.DataFrame:
    available = [t for t in task_names if t in df.index]
    # Drop models with no results at all
    model_cols = df.columns.tolist()
    df_t = df.loc[available, model_cols].T  # models × tasks
    df_t = df_t.dropna(how="all")

    category_avgs = compute_category_averages(
        df.loc[available], available, task_type_map
    )
    borda = compute_borda_count(df.loc[available], available)

    df_t = df_t.copy()
    df_t["borda"] = borda
    df_t["mean"] = df_t[available].mean(axis=1)

    # Weighted mean: average of per-category averages (equal weight per type)
    weighted = pd.Series(0.0, index=df_t.index)
    valid_counts = pd.Series(0, index=df_t.index)
    for task_type in TASK_TYPE_ORDER:
        if task_type in category_avgs:
            vals = category_avgs[task_type]
            mask = ~vals.isna()
            weighted[vals.index[mask]] += vals[mask]
            valid_counts[vals.index[mask]] += 1
    df_t["weighted_mean"] = weighted / valid_counts.replace(0, np.nan)

    # Attach per-category averages as columns
    for task_type in TASK_TYPE_ORDER:
        if task_type in category_avgs:
            df_t[f"cat_{task_type}"] = category_avgs[task_type]

    df_t = df_t.sort_values("borda", ascending=False).head(top_n)
    return df_t, category_avgs


def generate_latex_table(
    df_top: pd.DataFrame,
    category_avgs: dict[str, pd.Series],
    task_names: list[str],
    task_type_map: dict[str, str],
    top_n: int,
    benchmark_name: str = "",
) -> str:
    available = [t for t in task_names if t in category_avgs or True]

    # Count tasks per type
    type_counts: dict[str, int] = defaultdict(int)
    for t in task_names:
        ttype = task_type_map.get(t, "Unknown")
        type_counts[ttype] += 1

    # Best scores among top-N for bold highlighting
    best: dict[str, float] = {
        "mean": df_top["mean"].max(),
        "weighted_mean": df_top["weighted_mean"].max(),
    }
    for task_type in TASK_TYPE_ORDER:
        col = f"cat_{task_type}"
        if col in df_top.columns:
            best[task_type] = df_top[col].max()

    # Column spec: model | rank | mean | w.mean | per-category...
    present_types = [t for t in TASK_TYPE_ORDER if f"cat_{t}" in df_top.columns]
    n_cat_cols = len(present_types)
    col_spec = "l" + "c" * (3 + n_cat_cols)

    abbrevs = " & ".join(TASK_TYPE_ABBREV.get(t, t) for t in present_types)
    type_counts_str = " & ".join(
        f"\\textcolor{{gray}}{{({type_counts.get(t, 0)})}}" for t in present_types
    )
    total = len([t for t in task_names if task_type_map.get(t)])

    lines = [
        "\\begin{table*}[!th]",
        "    \\centering",
        "    \\caption{",
        f"    Top {top_n} models on {benchmark_name or 'DialogMTEB'} ({total} tasks). Ranked by Borda count.",
        "    \\textbf{Mean}: average across all tasks.",
        "    \\textbf{W.Mean}: average of per-category means (equal category weight).",
        "    Task types: "
        + ", ".join(f"{TASK_TYPE_ABBREV.get(t, t)} = {t}" for t in present_types)
        + ".",
        "    Best score shown in \\textbf{bold}; best within each category shaded.",
        "    }",
        f"    \\label{{tab:dialogmteb-results}}",
        "    \\resizebox{\\textwidth}{!}{",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    {\\footnotesize",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    \\textbf{{Model}} & \\textbf{{Rank}} & \\textbf{{Mean}} & \\textbf{{W.Mean}} & {abbrevs} \\\\",
        f"    \\textcolor{{gray}}{{\\# tasks}} & & \\textcolor{{gray}}{{({total})}} & \\textcolor{{gray}}{{({total})}} & {type_counts_str} \\\\",
        "    \\midrule",
    ]

    for rank, (model_name, row) in enumerate(df_top.iterrows(), 1):
        display = model_name.split("/")[-1] if "/" in model_name else model_name
        if len(display) > 38:
            display = display[:35] + "..."
        display = display.replace("_", "\\_")

        borda = int(row["borda"])
        mean_val = row["mean"]
        wmean_val = row["weighted_mean"]

        mean_str = format_score(mean_val, best["mean"])
        wmean_str = format_score(wmean_val, best["weighted_mean"])

        cat_strs = []
        for task_type in present_types:
            col = f"cat_{task_type}"
            val = row.get(col, np.nan)
            cat_strs.append(format_score(val, best.get(task_type, val)))

        cat_str = " & ".join(cat_strs)
        lines.append(
            f"    {display} & {rank} ({borda}) & {mean_str} & {wmean_str} & {cat_str} \\\\"
        )

    lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    }",
        "    } % end resizebox",
        "\\end{table*}",
    ]

    return "\n".join(lines)


def run_benchmark(benchmark_name: str, suffix: str, top_n: int) -> None:
    print(f"\nLoading {benchmark_name}...")
    benchmark = mteb.get_benchmark(benchmark_name)

    seen: set[str] = set()
    unique_tasks = []
    for task in benchmark.tasks:
        if task.metadata.name not in seen:
            seen.add(task.metadata.name)
            unique_tasks.append(task)
    task_names = [t.metadata.name for t in unique_tasks]
    task_type_map = build_task_type_map(unique_tasks)
    print(f"  {len(unique_tasks)} unique tasks")

    df = load_results(unique_tasks)
    print(f"  {len(df.columns)} models with at least one result")

    df_top, category_avgs = build_rankings(df, task_names, task_type_map, top_n)
    table = generate_latex_table(
        df_top, category_avgs, task_names, task_type_map, top_n, benchmark_name
    )

    output_file = Path(__file__).parent / f"dialogmteb_{suffix}_results_table.tex"
    output_file.write_text(table)
    print(f"  Written to {output_file}")

    print(f"  Top 5:")
    for rank, (model, row) in enumerate(df_top.head(5).iterrows(), 1):
        print(f"    {rank}. {model}: mean={row['mean'] * 100:.1f}%")


def main(top_n: int = 30):
    for suffix, benchmark_name in BENCHMARKS.items():
        run_benchmark(benchmark_name, suffix, top_n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()
    main(top_n=args.top_n)
