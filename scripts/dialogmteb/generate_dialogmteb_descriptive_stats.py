"""Generate per-type LaTeX subtables with descriptive statistics for DialogMTEB.

Uses task.calculate_descriptive_statistics() for all figures.
Produces one subtable per task type with type-specific columns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import (
    OUT_DIR,
    BENCHMARK_NAME,
    TYPE_ABBREV,
    TYPE_ORDER,
    load_unique_tasks,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_count(n: int | float | None) -> str:
    if n is None:
        return "--"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}".rstrip("0").rstrip(".") + "k"
    return str(n)


def fmt_float(x: float | None, decimals: int = 1) -> str:
    return f"{x:.{decimals}f}" if x is not None else "--"


def fmt_langs(languages: list[str]) -> str:
    unique = list(dict.fromkeys(lang.split("-")[0] for lang in (languages or [])))
    return unique[0] if len(unique) == 1 else str(len(unique))


def _sum_field(stats: dict, splits: list[str], *keys) -> int | None:
    """Sum a nested field across eval splits, returning None if not found."""
    total = 0
    found = False
    for s in splits:
        if s not in stats:
            continue
        node = stats[s]
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if node is not None:
            total += node
            found = True
    return total if found else None


def _mean_field(stats: dict, splits: list[str], *keys) -> float | None:
    vals = []
    for s in splits:
        if s not in stats:
            continue
        node = stats[s]
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if node is not None:
            vals.append(node)
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Per-type row builders
# ---------------------------------------------------------------------------


def row_classification(task, stats: dict, splits: list[str]) -> dict:
    meta = task.metadata
    n_samples = _sum_field(stats, splits, "num_samples")
    total_chars = _sum_field(stats, splits, "text_statistics", "total_text_length")
    avg_len = _mean_field(stats, splits, "text_statistics", "average_text_length")
    n_classes_vals = [
        stats[s]["label_statistics"]["unique_labels"]
        for s in splits
        if s in stats
        and stats[s].get("label_statistics")
        and "unique_labels" in stats[s]["label_statistics"]
    ]
    n_classes = max(n_classes_vals) if n_classes_vals else None
    return {
        "name": meta.name,
        "n_samples": fmt_count(n_samples),
        "n_langs": fmt_langs(meta.languages),
        "domains": ", ".join(sorted(meta.domains)[:3]) if meta.domains else "--",
        "metric": (meta.main_score or "").replace("test.", "").replace("_", "\\_"),
        "total_chars": fmt_count(total_chars),
        "avg_len": fmt_float(avg_len, 0),
        "n_classes": fmt_count(n_classes),
    }


def row_multilabel(task, stats: dict, splits: list[str]) -> dict:
    meta = task.metadata
    n_samples = _sum_field(stats, splits, "num_samples")
    total_chars = _sum_field(stats, splits, "text_statistics", "total_text_length")
    avg_len = _mean_field(stats, splits, "text_statistics", "average_text_length")
    n_classes_vals = [
        stats[s]["label_statistics"]["unique_labels"]
        for s in splits
        if s in stats
        and stats[s].get("label_statistics")
        and "unique_labels" in stats[s]["label_statistics"]
    ]
    n_classes = max(n_classes_vals) if n_classes_vals else None
    avg_labels = _mean_field(
        stats, splits, "label_statistics", "average_label_per_text"
    )
    return {
        "name": meta.name,
        "n_samples": fmt_count(n_samples),
        "n_langs": fmt_langs(meta.languages),
        "domains": ", ".join(sorted(meta.domains)[:3]) if meta.domains else "--",
        "metric": (meta.main_score or "").replace("test.", "").replace("_", "\\_"),
        "total_chars": fmt_count(total_chars),
        "avg_len": fmt_float(avg_len, 0),
        "n_labels": fmt_count(n_classes),
        "avg_labels_per_text": fmt_float(avg_labels, 2),
    }


def row_pair_classification(task, stats: dict, splits: list[str]) -> dict:
    meta = task.metadata
    n_samples = _sum_field(stats, splits, "num_samples")
    total_chars = _sum_field(stats, splits, "number_of_characters")
    avg_t1 = _mean_field(stats, splits, "text1_statistics", "average_text_length")
    avg_t2 = _mean_field(stats, splits, "text2_statistics", "average_text_length")
    n_classes_vals = [
        stats[s]["labels_statistics"]["unique_labels"]
        for s in splits
        if s in stats
        and stats[s].get("labels_statistics")
        and "unique_labels" in stats[s]["labels_statistics"]
    ]
    n_classes = max(n_classes_vals) if n_classes_vals else None
    return {
        "name": meta.name,
        "n_pairs": fmt_count(n_samples),
        "n_langs": fmt_langs(meta.languages),
        "domains": ", ".join(sorted(meta.domains)[:3]) if meta.domains else "--",
        "metric": (meta.main_score or "").replace("test.", "").replace("_", "\\_"),
        "total_chars": fmt_count(total_chars),
        "avg_text1_len": fmt_float(avg_t1, 0),
        "avg_text2_len": fmt_float(avg_t2, 0),
        "n_classes": fmt_count(n_classes),
    }


def row_retrieval(task, stats: dict, splits: list[str]) -> dict:
    meta = task.metadata
    n_queries = _sum_field(stats, splits, "num_samples")
    total_chars = _sum_field(stats, splits, "number_of_characters")
    avg_q_len = _mean_field(
        stats, splits, "queries_text_statistics", "average_text_length"
    )
    avg_d_len = _mean_field(
        stats, splits, "documents_text_statistics", "average_text_length"
    )
    avg_rel = _mean_field(
        stats, splits, "relevant_docs_statistics", "average_relevant_docs_per_query"
    )
    return {
        "name": meta.name,
        "n_queries": fmt_count(n_queries),
        "n_langs": fmt_langs(meta.languages),
        "domains": ", ".join(sorted(meta.domains)[:3]) if meta.domains else "--",
        "metric": (meta.main_score or "").replace("test.", "").replace("_", "\\_"),
        "total_chars": fmt_count(total_chars),
        "avg_query_len": fmt_float(avg_q_len, 0),
        "avg_doc_len": fmt_float(avg_d_len, 0),
        "avg_rel_docs": fmt_float(avg_rel, 2),
    }


def row_reranking(task, stats: dict, splits: list[str]) -> dict:
    meta = task.metadata
    n_samples = _sum_field(stats, splits, "num_samples")
    total_chars = _sum_field(stats, splits, "number_of_characters")
    avg_q_len = _mean_field(
        stats, splits, "queries_text_statistics", "average_text_length"
    )
    avg_d_len = _mean_field(
        stats, splits, "documents_text_statistics", "average_text_length"
    )
    avg_rel = _mean_field(
        stats, splits, "relevant_docs_statistics", "average_relevant_docs_per_query"
    )
    avg_top = _mean_field(
        stats, splits, "top_ranked_statistics", "average_top_ranked_per_query"
    )
    return {
        "name": meta.name,
        "n_samples": fmt_count(n_samples),
        "n_langs": fmt_langs(meta.languages),
        "domains": ", ".join(sorted(meta.domains)[:3]) if meta.domains else "--",
        "metric": (meta.main_score or "").replace("test.", "").replace("_", "\\_"),
        "total_chars": fmt_count(total_chars),
        "avg_query_len": fmt_float(avg_q_len, 0),
        "avg_doc_len": fmt_float(avg_d_len, 0),
        "avg_rel_docs": fmt_float(avg_rel, 2),
        "avg_top_ranked": fmt_float(avg_top, 1),
    }


def row_sts(task, stats: dict, splits: list[str]) -> dict:
    meta = task.metadata
    n_pairs = _sum_field(stats, splits, "num_samples")
    total_chars = _sum_field(stats, splits, "number_of_characters")
    avg_t1 = _mean_field(stats, splits, "text1_statistics", "average_text_length")
    avg_t2 = _mean_field(stats, splits, "text2_statistics", "average_text_length")
    avg_score = _mean_field(stats, splits, "label_statistics", "avg_score")
    return {
        "name": meta.name,
        "n_pairs": fmt_count(n_pairs),
        "n_langs": fmt_langs(meta.languages),
        "domains": ", ".join(sorted(meta.domains)[:3]) if meta.domains else "--",
        "metric": (meta.main_score or "").replace("test.", "").replace("_", "\\_"),
        "total_chars": fmt_count(total_chars),
        "avg_text1_len": fmt_float(avg_t1, 0),
        "avg_text2_len": fmt_float(avg_t2, 0),
        "avg_score": fmt_float(avg_score, 3),
    }


ROW_BUILDERS = {
    "Classification": row_classification,
    "MultilabelClassification": row_multilabel,
    "PairClassification": row_pair_classification,
    "Retrieval": row_retrieval,
    "Reranking": row_reranking,
    "STS": row_sts,
}

# ---------------------------------------------------------------------------
# LaTeX subtable definitions per type
# ---------------------------------------------------------------------------

SUBTABLE_SPECS = {
    "Classification": {
        "col_spec": "llrcclrrr",
        "header": (
            "\\textbf{Dataset} & \\textbf{Type} & \\textbf{N.~Samples} "
            "& \\textbf{N.~Langs} & \\textbf{Domains} & \\textbf{Metric} "
            "& \\textbf{Total Chars} & \\textbf{Avg.~Len} & \\textbf{N.~Classes} \\\\"
        ),
        "row_keys": [
            "name",
            "type_abbrev",
            "n_samples",
            "n_langs",
            "domains",
            "metric",
            "total_chars",
            "avg_len",
            "n_classes",
        ],
    },
    "MultilabelClassification": {
        "col_spec": "llrcclrrrl",
        "header": (
            "\\textbf{Dataset} & \\textbf{Type} & \\textbf{N.~Samples} "
            "& \\textbf{N.~Langs} & \\textbf{Domains} & \\textbf{Metric} "
            "& \\textbf{Total Chars} & \\textbf{Avg.~Len} "
            "& \\textbf{N.~Labels} & \\textbf{Avg.~Labels/Text} \\\\"
        ),
        "row_keys": [
            "name",
            "type_abbrev",
            "n_samples",
            "n_langs",
            "domains",
            "metric",
            "total_chars",
            "avg_len",
            "n_labels",
            "avg_labels_per_text",
        ],
    },
    "PairClassification": {
        "col_spec": "llrcclrrr",
        "header": (
            "\\textbf{Dataset} & \\textbf{Type} & \\textbf{N.~Pairs} "
            "& \\textbf{N.~Langs} & \\textbf{Domains} & \\textbf{Metric} "
            "& \\textbf{Total Chars} & \\textbf{Avg.~Len\\textsubscript{1}} "
            "& \\textbf{Avg.~Len\\textsubscript{2}} \\\\"
        ),
        "row_keys": [
            "name",
            "type_abbrev",
            "n_pairs",
            "n_langs",
            "domains",
            "metric",
            "total_chars",
            "avg_text1_len",
            "avg_text2_len",
        ],
    },
    "Retrieval": {
        "col_spec": "llrcclrrr",
        "header": (
            "\\textbf{Dataset} & \\textbf{Type} & \\textbf{N.~Queries} "
            "& \\textbf{N.~Langs} & \\textbf{Domains} & \\textbf{Metric} "
            "& \\textbf{Total Chars} & \\textbf{Avg.~Query Len} "
            "& \\textbf{Avg.~Doc Len} & \\textbf{Avg.~Rel.~Docs} \\\\"
        ),
        "row_keys": [
            "name",
            "type_abbrev",
            "n_queries",
            "n_langs",
            "domains",
            "metric",
            "total_chars",
            "avg_query_len",
            "avg_doc_len",
            "avg_rel_docs",
        ],
        "col_spec": "llrcclrrrr",
    },
    "Reranking": {
        "col_spec": "llrcclrrrr",
        "header": (
            "\\textbf{Dataset} & \\textbf{Type} & \\textbf{N.~Samples} "
            "& \\textbf{N.~Langs} & \\textbf{Domains} & \\textbf{Metric} "
            "& \\textbf{Total Chars} & \\textbf{Avg.~Query Len} "
            "& \\textbf{Avg.~Doc Len} & \\textbf{Avg.~Rel.~Docs} & \\textbf{Avg.~Top-K} \\\\"
        ),
        "row_keys": [
            "name",
            "type_abbrev",
            "n_samples",
            "n_langs",
            "domains",
            "metric",
            "total_chars",
            "avg_query_len",
            "avg_doc_len",
            "avg_rel_docs",
            "avg_top_ranked",
        ],
        "col_spec": "llrcclrrrrr",
    },
    "STS": {
        "col_spec": "llrcclrrrr",
        "header": (
            "\\textbf{Dataset} & \\textbf{Type} & \\textbf{N.~Pairs} "
            "& \\textbf{N.~Langs} & \\textbf{Domains} & \\textbf{Metric} "
            "& \\textbf{Total Chars} & \\textbf{Avg.~Len\\textsubscript{1}} "
            "& \\textbf{Avg.~Len\\textsubscript{2}} & \\textbf{Avg.~Score} \\\\"
        ),
        "row_keys": [
            "name",
            "type_abbrev",
            "n_pairs",
            "n_langs",
            "domains",
            "metric",
            "total_chars",
            "avg_text1_len",
            "avg_text2_len",
            "avg_score",
        ],
    },
}

TYPE_CAPTIONS = {
    "Classification": "Classification tasks",
    "MultilabelClassification": "Multi-label classification tasks",
    "PairClassification": "Pair classification tasks",
    "Retrieval": "Retrieval tasks",
    "Reranking": "Reranking tasks",
    "STS": "Semantic textual similarity tasks",
}


# ---------------------------------------------------------------------------
# Build + render
# ---------------------------------------------------------------------------


def build_all_rows(tasks: list) -> dict[str, list[dict]]:
    """Return {task_type: [row_dict, ...]}."""
    by_type: dict[str, list[dict]] = {}
    for task in tasks:
        meta = task.metadata
        ttype = meta.type
        builder = ROW_BUILDERS.get(ttype)
        if builder is None:
            print(f"  WARNING: no builder for type {ttype} ({meta.name})")
            continue

        print(f"  {meta.name}...", end=" ", flush=True)
        try:
            stats = task.calculate_descriptive_statistics()
            row = builder(task, stats, meta.eval_splits)
            row["type_abbrev"] = TYPE_ABBREV.get(ttype, ttype)
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        print("ok")
        by_type.setdefault(ttype, []).append(row)

    return by_type


def render_subtable(ttype: str, rows: list[dict]) -> str:
    spec = SUBTABLE_SPECS[ttype]
    col_spec = spec["col_spec"]
    header = spec["header"]
    row_keys = spec["row_keys"]
    caption = TYPE_CAPTIONS.get(ttype, ttype)
    label = ttype.lower().replace("multilabel", "multilabel_")

    rows_sorted = sorted(rows, key=lambda r: r["name"])

    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        f"    \\caption{{DialogMTEB {caption}.}}",
        "    \\resizebox{\\linewidth}{!}{",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header}",
        "    \\midrule",
    ]
    for row in rows_sorted:
        cells = []
        for k in row_keys:
            val = str(row.get(k, "--"))
            if k == "name":
                val = val.replace("_", "\\_")
            cells.append(val)
        lines.append("    " + " & ".join(cells) + " \\\\")

    lines += [
        "    \\bottomrule",
        "    \\end{tabular}",
        "    }",
        f"    \\label{{tab:dialogmteb_stats_{label}}}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def main():
    print(f"Loading {BENCHMARK_NAME} tasks...")
    tasks = load_unique_tasks()
    print(f"  {len(tasks)} unique tasks\n")

    print("Computing descriptive statistics...")
    by_type = build_all_rows(tasks)

    # Render one subtable per type in TYPE_ORDER
    all_tables: list[str] = []
    all_rows: list[dict] = []
    for ttype in TYPE_ORDER:
        rows = by_type.get(ttype, [])
        if not rows:
            continue
        all_tables.append(render_subtable(ttype, rows))
        for r in rows:
            all_rows.append({"task_type": ttype, **r})

    tex_out = OUT_DIR / "dialogmteb_descriptive_stats.tex"
    tex_out.write_text("\n\n".join(all_tables))
    print(f"\nLaTeX subtables written to {tex_out}")

    csv_out = OUT_DIR / "dialogmteb_descriptive_stats.csv"
    pd.DataFrame(all_rows).to_csv(csv_out, index=False)
    print(f"CSV written to {csv_out}")


if __name__ == "__main__":
    main()
