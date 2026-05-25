"""Generate LaTeX overview table for DialogMTEB benchmark tasks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import bibtexparser
from bibtexparser.bparser import BibTexParser

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mteb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    BENCHMARK_NAME,
    TYPE_ORDER,
    TYPE_ABBREV as TYPE_DISPLAY,
)

CITATION_FIXES: dict[str, str] = {}


def extract_citation_key(bibtex_str: str) -> str:
    if not bibtex_str:
        return ""
    try:
        parser = BibTexParser()
        parser.ignore_nonstandard_types = True
        parser.homogenize_fields = False
        bib_db = bibtexparser.loads(bibtex_str, parser)
        if bib_db.entries:
            key = bib_db.entries[0]["ID"]
            return "" if key == "inproceedings" else key
    except Exception:
        match = re.search(r"@\w+\{([^,]+),", bibtex_str)
        if match:
            return match.group(1)
    return ""


def format_languages(languages: list[str]) -> str:
    if not languages:
        return "1"
    unique = list(dict.fromkeys(lang.split("-")[0] for lang in languages))
    n = len(unique)
    if n == 1:
        return unique[0]
    return str(n)


def format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}".rstrip("0").rstrip(".") + "k"
    return str(value)


def format_n_samples(n_samples: dict | None, eval_splits: list[str]) -> str:
    if not n_samples:
        return ""
    total = sum(n_samples.get(s, 0) for s in eval_splits)
    if total == 0:
        total = sum(n_samples.values())
    return format_count(total)


def load_tasks() -> list:
    benchmark = mteb.get_benchmark(BENCHMARK_NAME)
    seen = set()
    unique_tasks = []
    for task in benchmark.tasks:
        if task.metadata.name not in seen:
            seen.add(task.metadata.name)
            unique_tasks.append(task)
    return unique_tasks


def build_task_rows(tasks: list) -> list[dict]:
    rows = []
    for task in tasks:
        meta = task.metadata

        if meta.name in CITATION_FIXES:
            cit_key = CITATION_FIXES[meta.name]
        else:
            cit_key = extract_citation_key(meta.bibtex_citation)

        cite = f"~\\cite{{{cit_key}}}" if cit_key else ""
        lang_str = format_languages(meta.languages)
        n_samples_str = format_n_samples(meta.n_samples, meta.eval_splits)
        main_metric = (meta.main_score or "").replace("test.", "").replace("_", "\\_")
        task_type = meta.type

        rows.append(
            {
                "name": meta.name,
                "cite": cite,
                "task_type": task_type,
                "display_type": TYPE_DISPLAY.get(task_type, task_type),
                "languages": lang_str,
                "n_samples": n_samples_str,
                "main_metric": main_metric,
            }
        )
    return rows


def generate_overview_table(rows: list[dict]) -> str:
    # Sort by type order, then name
    type_order_map = {t: i for i, t in enumerate(TYPE_ORDER)}
    rows_sorted = sorted(
        rows, key=lambda r: (type_order_map.get(r["task_type"], 99), r["name"])
    )

    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\resizebox{\\linewidth}{!}{",
        "    \\begin{tabular}{llccc}",
        "    \\toprule",
        "    \\textbf{Dataset} & \\textbf{Type} & "
        "\\textbf{N. Samples} & \\textbf{N. Langs} & \\textbf{Metric} \\\\",
        "    \\midrule",
    ]

    current_type = None
    first_section = True
    for row in rows_sorted:
        if row["task_type"] != current_type:
            current_type = row["task_type"]
            if not first_section:
                lines.append("    \\hline")
            display_type = TYPE_DISPLAY.get(current_type, current_type)
            lines.append(
                f"    \\multicolumn{{5}}{{l}}{{\\textit{{{display_type}}}}} \\\\"
            )
            lines.append("    \\hline")
            first_section = False

        name = row["name"].replace("_", "\\_") + row["cite"]
        lines.append(
            f"    {name} & {row['display_type']} & "
            f"{row['n_samples']} & {row['languages']} & {row['main_metric']} \\\\"
        )

    lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    }",
        "    \\caption{DialogMTEB Task Overview. Tasks are grouped by type and show dataset size, language coverage, and main evaluation metric.}",
        "    \\label{tab:dialogmteb_overview}",
        "\\end{table*}",
    ]

    return "\n".join(lines)


def main():
    print(f"Loading {BENCHMARK_NAME} tasks...")
    tasks = load_tasks()
    print(f"Found {len(tasks)} unique tasks")

    rows = build_task_rows(tasks)

    print("Generating LaTeX table...")
    table = generate_overview_table(rows)

    output_file = Path(__file__).parent / "dialogmteb_task_overview.tex"
    output_file.write_text(table)
    print(f"Written to {output_file}")

    print("\nTask type breakdown:")
    from collections import Counter

    counts = Counter(r["task_type"] for r in rows)
    for t in TYPE_ORDER:
        if t in counts:
            print(f"  {t}: {counts[t]}")


if __name__ == "__main__":
    main()
