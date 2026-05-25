"""Generate LaTeX table of dataset provenance for DialogMTEB tasks.

Shows: original paper, year, whether the task was already in MTEB,
and what DialogMTEB contributed (new task type / language / split).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import bibtexparser
from bibtexparser.bparser import BibTexParser

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    OUT_DIR,
    TYPE_ABBREV,
    TYPE_ORDER,
    load_unique_tasks,
)

import mteb


def extract_year(bibtex_str: str) -> str:
    if not bibtex_str:
        return "--"
    match = re.search(r"\byear\s*=\s*[{\"]?(\d{4})[}\"]?", bibtex_str, re.IGNORECASE)
    return match.group(1) if match else "--"


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


def infer_contribution(task) -> str:
    name = task.metadata.name
    if name in CONTRIBUTION_NOTES:
        return CONTRIBUTION_NOTES[name]
    if name in MTEB_EXISTING:
        return "From MTEB"
    return "Novel task"


def main():
    tasks = load_unique_tasks()

    # Check which tasks exist in general MTEB benchmarks
    try:
        mteb_eng = {t.metadata.name for t in mteb.get_benchmark("MTEB(eng)").tasks}
        mteb_full = {t.metadata.name for t in mteb.get_benchmark("MTEB").tasks}
    except Exception:
        mteb_eng = set()
        mteb_full = set()

    rows = []
    for task in tasks:
        meta = task.metadata
        in_mteb = meta.name in (mteb_eng | mteb_full)
        cit_key = extract_citation_key(meta.bibtex_citation)
        citation = f"\\cite{{{cit_key}}}" if cit_key else "--"
        year = extract_year(meta.bibtex_citation)
        langs = set(lang.split("-")[0] for lang in (meta.languages or []))
        n_langs = len(langs)
        lang_str = list(langs)[0] if n_langs == 1 else str(n_langs)

        rows.append(
            {
                "name": meta.name,
                "task_type": meta.type,
                "citation": citation,
                "year": year,
                "in_mteb": in_mteb,
                "n_langs": lang_str,
            }
        )

    # Sort by type then name
    type_order_idx = {t: i for i, t in enumerate(TYPE_ORDER)}
    rows.sort(key=lambda r: (type_order_idx.get(r["task_type"], 99), r["name"]))

    n_novel = sum(1 for r in rows if not r["in_mteb"])
    n_from_mteb = sum(1 for r in rows if r["in_mteb"])
    print(f"Novel tasks (not in MTEB): {n_novel}")
    print(f"Tasks from MTEB:           {n_from_mteb}")

    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\resizebox{\\linewidth}{!}{",
        "    \\begin{tabular}{lllcc}",
        "    \\toprule",
        "    \\textbf{Dataset} & \\textbf{Citation} & \\textbf{Type} "
        "& \\textbf{Year} & \\textbf{N.Langs} \\\\",
        "    \\midrule",
    ]

    current_type = None
    for row in rows:
        if row["task_type"] != current_type:
            current_type = row["task_type"]
            abbrev = TYPE_ABBREV.get(current_type, current_type)
            lines.append(f"    \\multicolumn{{5}}{{l}}{{\\textit{{{abbrev}}}}} \\\\")

        name = row["name"].replace("_", "\\_")
        abbrev = TYPE_ABBREV.get(row["task_type"], row["task_type"])
        lines.append(
            f"    {name} & {row['citation']} & {abbrev} "
            f"& {row['year']} & {row['n_langs']} \\\\"
        )

    lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    }",
        "    \\caption{DialogMTEB dataset provenance. "
        f"\\textbf{{Novel}} = {n_novel} tasks not present in general MTEB; "
        f"\\textbf{{MTEB}} = {n_from_mteb} tasks reused from MTEB. "
        "N.Langs = number of languages.}",
        "    \\label{tab:dialogmteb_provenance}",
        "\\end{table*}",
    ]

    out = OUT_DIR / "dialogmteb_dataset_provenance.tex"
    out.write_text("\n".join(lines))
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
