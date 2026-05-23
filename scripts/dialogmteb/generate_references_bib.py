"""Generate a consolidated BibTeX file for all DialogMTEB tasks and run.py models.

Sources:
  - Task citations: task.metadata.bibtex_citation
  - Model citations: mteb.get_model_meta(name).citation

Deduplication is done by BibTeX key. Missing citations are written as
placeholder entries and listed in a warnings section at the top.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import OUT_DIR, load_unique_tasks, RUN_MODELS

import mteb

# -----------------------------------------------------------------------
# Hand-written citations for models not (yet) in the MTEB registry
# -----------------------------------------------------------------------
MANUAL_CITATIONS: dict[str, str] = {
    "princeton-nlp/sup-simcse-bert-base-uncased": r"""@inproceedings{gao-etal-2021-simcse,
  title     = {{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
  author    = {Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year      = {2021},
  pages     = {6894--6910},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2021.emnlp-main.552},
}""",
    "TODBERT/TOD-BERT-MLM-V1": r"""@inproceedings{wu-etal-2020-tod,
  title     = {{TOD-BERT}: Pre-trained Natural Language Understanding for Task-Oriented Dialogue},
  author    = {Wu, Chien-Sheng and Hoi, Steven C.H. and Socher, Richard and Xiong, Caiming},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year      = {2020},
  pages     = {917--929},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2020.emnlp-main.66},
}""",
    "AndrewZeng/futuretod-base-v1.0": r"""@inproceedings{zeng-nie-2023-futuretod,
  title     = {{FutureTOD}: Teaching Future Knowledge to Pre-trained Language Model for Task-Oriented Dialogue},
  author    = {Zeng, Weihao and Nie, Keqing and Yao, Junwei and Huang, Ruixue and Cheng, Qianlong and Wang, Pei and Xu, Weiran},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2023},
  pages     = {12780--12793},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2023.acl-long.716},
}""",
    "codefuse-ai/F2LLM-v2-0.6B": r"""@misc{f2llm2024,
  title  = {{F2LLM}: Fine-tuned Foundation Language Models for Embedding},
  author = {CodeFuse AI},
  year   = {2024},
  note   = {\url{https://huggingface.co/codefuse-ai}},
}""",
}

# Models that share a citation with another key already in MANUAL_CITATIONS
CITATION_ALIASES: dict[str, str] = {
    "princeton-nlp/unsup-simcse-roberta-base": "princeton-nlp/sup-simcse-bert-base-uncased",
    "princeton-nlp/unsup-simcse-bert-large-uncased": "princeton-nlp/sup-simcse-bert-base-uncased",
    "princeton-nlp/unsup-simcse-roberta-large": "princeton-nlp/sup-simcse-bert-base-uncased",
    "princeton-nlp/sup-simcse-bert-large-uncased": "princeton-nlp/sup-simcse-bert-base-uncased",
    "princeton-nlp/sup-simcse-roberta-base": "princeton-nlp/sup-simcse-bert-base-uncased",
    "princeton-nlp/sup-simcse-roberta-large": "princeton-nlp/sup-simcse-bert-base-uncased",
    "codefuse-ai/F2LLM-v2-80M": "codefuse-ai/F2LLM-v2-0.6B",
    "codefuse-ai/F2LLM-v2-160M": "codefuse-ai/F2LLM-v2-0.6B",
    "codefuse-ai/F2LLM-v2-330M": "codefuse-ai/F2LLM-v2-0.6B",
    "codefuse-ai/F2LLM-v2-4B": "codefuse-ai/F2LLM-v2-0.6B",
}


# -----------------------------------------------------------------------
# BibTeX utilities
# -----------------------------------------------------------------------

_KEY_RE = re.compile(r"@\w+\{([^,\s]+)\s*,", re.MULTILINE)


def extract_key(bibtex: str) -> str | None:
    m = _KEY_RE.search(bibtex)
    return m.group(1).strip() if m else None


def normalize_bibtex(bibtex: str) -> str:
    """Strip leading/trailing whitespace from a bibtex entry."""
    return bibtex.strip()


def collect_entries(
    raw_bibtex: str,
) -> list[tuple[str, str]]:
    """Split a string that may contain multiple @-entries into (key, entry) pairs."""
    entries = []
    for chunk in re.split(r"\n(?=@)", raw_bibtex.strip()):
        chunk = chunk.strip()
        if not chunk:
            continue
        key = extract_key(chunk)
        if key:
            entries.append((key, chunk))
    return entries


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    # Collect all bibtex: key → entry string (first-seen wins)
    seen: dict[str, str] = {}  # key → bibtex entry
    source_map: dict[str, list[str]] = {}  # key → [model/task names]
    warnings: list[str] = []

    # ── Tasks ────────────────────────────────────────────────────────────
    tasks = load_unique_tasks()
    print(f"Processing {len(tasks)} tasks...")
    for task in tasks:
        bibtex = task.metadata.bibtex_citation or ""
        if not bibtex.strip():
            warnings.append(f"[TASK] No citation: {task.metadata.name}")
            continue
        for key, entry in collect_entries(bibtex):
            source_map.setdefault(key, []).append(f"task:{task.metadata.name}")
            if key not in seen:
                seen[key] = normalize_bibtex(entry)

    # ── Models ───────────────────────────────────────────────────────────
    print(f"Processing {len(RUN_MODELS)} models from run.py...")

    # Resolve aliases first so we know which canonical entry to emit
    canonical: dict[str, str] = {}  # model → canonical model for lookup
    for model in RUN_MODELS:
        canonical[model] = CITATION_ALIASES.get(model, model)

    for model in RUN_MODELS:
        canon = canonical[model]
        bibtex: str | None = None

        # 1. Try manual citation for canonical model
        if canon in MANUAL_CITATIONS:
            bibtex = MANUAL_CITATIONS[canon]

        # 2. Try mteb registry
        if bibtex is None:
            try:
                meta = mteb.get_model_meta(model)
                if meta.citation:
                    bibtex = str(meta.citation).strip()
            except Exception:
                pass

        if not bibtex:
            warnings.append(f"[MODEL] No citation: {model}")
            continue

        for key, entry in collect_entries(bibtex):
            source_map.setdefault(key, []).append(f"model:{model}")
            if key not in seen:
                seen[key] = normalize_bibtex(entry)

    # ── Write .bib file ──────────────────────────────────────────────────
    out = OUT_DIR / "dialogmteb_references.bib"

    header_lines = [
        "% DialogMTEB References",
        "% Auto-generated by generate_references_bib.py",
        f"% Tasks: {len(tasks)}  |  Models: {len(RUN_MODELS)}",
        f"% Total unique entries: {len(seen)}",
        "%",
    ]
    if warnings:
        header_lines.append("% WARNINGS — entries with no citation found:")
        for w in warnings:
            header_lines.append(f"%   {w}")
    header_lines.append("")

    # Group entries: task citations first, then model citations
    task_keys = []
    model_keys = []
    other_keys = []
    for key in seen:
        sources = source_map.get(key, [])
        has_task = any(s.startswith("task:") for s in sources)
        has_model = any(s.startswith("model:") for s in sources)
        if has_task and not has_model:
            task_keys.append(key)
        elif has_model and not has_task:
            model_keys.append(key)
        else:
            other_keys.append(key)

    sections = []
    if task_keys:
        sections.append(
            (
                "% ── Task Citations ──────────────────────────────────────────────",
                sorted(task_keys),
            )
        )
    if model_keys:
        sections.append(
            (
                "% ── Model Citations ─────────────────────────────────────────────",
                sorted(model_keys),
            )
        )
    if other_keys:
        sections.append(
            (
                "% ── Shared Citations ────────────────────────────────────────────",
                sorted(other_keys),
            )
        )

    body_lines = []
    for section_header, keys in sections:
        body_lines.append(section_header)
        body_lines.append("")
        for key in keys:
            body_lines.append(seen[key])
            body_lines.append("")

    content = "\n".join(header_lines) + "\n" + "\n".join(body_lines)
    out.write_text(content)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nBibTeX file written to {out}")
    print(f"  Task-only entries : {len(task_keys)}")
    print(f"  Model-only entries: {len(model_keys)}")
    print(f"  Shared entries    : {len(other_keys)}")
    print(f"  Total unique keys : {len(seen)}")
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
