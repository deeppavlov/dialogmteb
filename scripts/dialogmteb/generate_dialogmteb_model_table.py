"""Generate LaTeX table listing models evaluated on DialogMTEB with their parameters.

Shows top-N models (by Borda count) with: n_parameters, embed_dim,
max_tokens, open_weights, and their DialogMTEB mean score.

Run with --top-n N to change how many models appear (default 30).
Run with --all to include every model that has results.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mteb
from scripts.dialogmteb._common import (
    OUT_DIR,
    get_complete_models,
    load_score_df,
    load_unique_tasks,
)

_KEY_RE = re.compile(r"@\w+\{([^,\s]+)", re.MULTILINE)


def get_citation_key(model_name: str) -> str:
    try:
        cit = str(mteb.get_model_meta(model_name).citation or "")
        m = _KEY_RE.search(cit)
        return m.group(1) if m else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fmt_params(n: int | None) -> str:
    if n is None:
        return "--"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    return str(n)


def fmt_tokens(t: float | None) -> str:
    if t is None or (isinstance(t, float) and np.isnan(t)) or np.isinf(t):
        return "--"
    t = int(t)
    if t >= 1_000:
        return f"{t // 1_000}k"
    return str(t)


def fmt_dim(d: int | None) -> str:
    return str(d) if d is not None else "--"


def fmt_open(flag: bool | None) -> str:
    if flag is True:
        return "\\checkmark"
    if flag is False:
        return "\\texttimes{}"
    return "--"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def fetch_model_meta(model_name: str) -> dict:
    try:
        meta = mteb.get_model_meta(model_name)
        return {
            "n_parameters": meta.n_parameters,
            "embed_dim": meta.embed_dim,
            "max_tokens": meta.max_tokens,
            "open_weights": meta.open_weights,
        }
    except Exception:
        return {
            "n_parameters": None,
            "embed_dim": None,
            "max_tokens": None,
            "open_weights": None,
        }


# ---------------------------------------------------------------------------
# table generation
# ---------------------------------------------------------------------------


def get_fully_evaluated_models() -> list[str]:
    """Return RUN_MODELS that have results for every task in the (multilingual) benchmark."""
    tasks = load_unique_tasks()
    task_names = [t.metadata.name for t in tasks]
    score_df = load_score_df(tasks)
    complete = get_complete_models(score_df, task_names)
    skipped = sorted(set(score_df.columns) - set(complete))
    if skipped:
        print(f"Skipping {len(skipped)} models missing results on some tasks: {skipped}")
    print(f"{len(complete)} models fully evaluated on all {len(task_names)} tasks")
    return complete


def build_model_rows(top_n: int | None) -> list[dict]:
    models = sorted(get_fully_evaluated_models())
    if top_n is not None:
        models = models[:top_n]

    rows = []
    for model_name in models:
        meta = fetch_model_meta(model_name)
        display = model_name.replace("_", "\\_")
        cite_key = get_citation_key(model_name)
        if cite_key:
            display += f"~\\cite{{{cite_key}}}"
        rows.append(
            {
                "model_name": model_name,
                "display": display,
                "n_parameters": fmt_params(meta["n_parameters"]),
                "embed_dim": fmt_dim(meta["embed_dim"]),
                "max_tokens": fmt_tokens(meta["max_tokens"]),
            }
        )
    return rows


def generate_latex_table(rows: list[dict], top_n: int | None) -> str:
    title_n = f"top {top_n}" if top_n else "all"
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\resizebox{\\linewidth}{!}{",
        "    \\begin{tabular}{lrrr}",
        "    \\toprule",
        "    \\textbf{Model} & \\textbf{N.Param} & \\textbf{Dim} & \\textbf{Ctx} \\\\",
        "    \\midrule",
    ]

    for row in rows:
        lines.append(
            f"    {row['display']} & {row['n_parameters']} "
            f"& {row['embed_dim']} & {row['max_tokens']} \\\\"
        )

    lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    }",
        f"    \\caption{{DialogMTEB model overview ({title_n} models). "
        "N.Param = number of parameters, Dim = embedding dimension, "
        "Ctx = max context length.}",
        "    \\label{tab:dialogmteb_models}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(top_n: int | None = 30):
    print(f"Fetching model metadata ({top_n or 'all'} models)...")
    rows = build_model_rows(top_n)

    table = generate_latex_table(rows, top_n)
    out = OUT_DIR / "dialogmteb_model_table.tex"
    out.write_text(table)
    print(f"Written to {out}")

    csv_out = OUT_DIR / "dialogmteb_model_table.csv"
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"CSV written to {csv_out}")

    print(f"\nModels listed:")
    for r in rows[:10]:
        print(f"  {r['model_name']}: params={r['n_parameters']}, dim={r['embed_dim']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--top-n", type=int, default=30)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    main(top_n=None if args.all else args.top_n)
