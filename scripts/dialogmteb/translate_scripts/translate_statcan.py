#!/usr/bin/env python3
"""Translate mteb/StatcanDialogueDatasetRetrieval to Spanish (and French) using offline
vLLM inference.

This is a **retrieval** dataset (not flat sentences): `english-corpus` (documents),
`english-queries` (dialogue turns), `english-qrels` (relevance judgments). The corpus is
structured StatCan dataset metadata -- a `Title/Date range/Dimensions/Subject/Survey/
Frequency` header followed by category trees (`ID: 1, Parent: None, Name: Canada`).
Documents range up to ~391,133 characters, but `dev` and `test` corpora are
byte-identical (same 5,907 docs), and across the whole corpus there are only ~91,612
unique `Name` values out of 592,259 occurrences (plus titles/subjects/surveys/
frequencies/dimension names) -- categories repeat constantly ("Canada", age groups...).

So instead of translating whole documents, this script:
1. Parses every corpus doc into an ordered list of segments (header fields, blank
   lines, dimension-block headers, `ID/Parent/Name` lines).
2. Collects the global deduplicated set of translatable strings across the whole
   corpus (titles, subjects, surveys, frequencies, dimension names, leaf names) and
   translates that pool once, per item (not per document).
3. Reconstructs every doc by substitution -- IDs, parent numbers, and date ranges are
   left untouched; the 6 header labels use a fixed EN->{lang} map. A handful of
   malformed lines (~9 out of ~600k, from embedded newlines in the source data) don't
   fit the parser and are left untranslated -- negligible.

`english-queries` (543 dev + 553 test rows, ~7,961 turns total -- informal, typo-laden
live chat between a citizen and a StatCan agent) is translated turn-by-turn.
`english-qrels` is just `query-id`/`corpus-id`/`score` relations with no text, so it's
copied through unchanged.

Usage (run one model per process -- see translate_common.py for why):

    python translate_statcan.py translate --model gemma
    python translate_statcan.py translate --model qwen
    python translate_statcan.py smoke-test --model gemma
    python translate_statcan.py assemble    # no vLLM needed, reads both checkpoints

Checkpointed to translations/statcan/*.jsonl -- safe to interrupt/resume. Existing
checkpoints from the earlier notebook-based run use the same file names/schema and are
picked up automatically.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from datasets import Dataset, DatasetDict, load_dataset  # noqa: E402

from translate_common import MODELS, OfflineTranslator, load_checkpoint, parse_engine_kwargs  # noqa: E402

LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
}

# Only Spanish was originally requested for StatCan -- keep that as the default scope,
# but French is fully supported (examples below) if you pass --langs fr or --langs fr es.
DEFAULT_LANGS = ["es"]

SPLITS = ["dev", "test"]
OUT_DIR = SCRIPT_DIR / "translations" / "statcan"
SAVE_DIR = SCRIPT_DIR / "translations" / "statcan_final"

# Fixed header-label translations (Date range's VALUE is an ISO date range -- not
# translated -- but the label itself still needs localizing).
HEADER_LABELS = {
    "es": {
        "Title": "Título",
        "Date range": "Período",
        "Dimensions": "Dimensiones",
        "Subject": "Tema",
        "Survey": "Encuesta",
        "Frequency": "Frecuencia",
    },
    "fr": {
        "Title": "Titre",
        "Date range": "Période",
        "Dimensions": "Dimensions",
        "Subject": "Sujet",
        "Survey": "Enquête",
        "Frequency": "Fréquence",
    },
}

VOCAB_EXAMPLES = {
    "es": [
        ("Geography", "Geografía"),
        ("Canada", "Canadá"),
        ("Age group", "Grupo de edad"),
        ("Labour Force Survey", "Encuesta de la Población Activa"),
    ],
    "fr": [
        ("Geography", "Géographie"),
        ("Canada", "Canada"),
        ("Age group", "Groupe d'âge"),
        ("Labour Force Survey", "Enquête sur la population active"),
    ],
}

VOCAB_SYSTEM_PROMPT = (
    "You are a professional translator localizing Statistics Canada open-data category "
    "names, dataset titles, and survey names for a data catalog. Translate the item from "
    "English into {lang_name}. Keep proper nouns as their correct {lang_name} exonym "
    "where one exists (e.g. 'Canada' -> its {lang_name} form; provinces/countries use "
    "their standard {lang_name} names). Keep numeric codes, acronyms in parentheses, and "
    "units of measure recognizable. Do not add, remove, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)

QUERY_EXAMPLES = {
    "es": [
        (
            "hi, i was wondering if you vae any statistics on video game sales, ot high school drop out rates?",
            "hola, me preguntaba si tienen alguna estadistica sobre ventas de videojuegos, o tasas de abandono escolar?",
        ),
        (
            "Data for High School dropouts is compiled by the Provincial Education Ministry.",
            "Los datos sobre abandono escolar los recopila el Ministerio de Educacion provincial.",
        ),
    ],
    "fr": [
        (
            "hi, i was wondering if you vae any statistics on video game sales, ot high school drop out rates?",
            "salut, je me demandais si vous aviez des statistiques sur les ventes de jeux video, ou les taux de decrochage scolaire?",
        ),
        (
            "Data for High School dropouts is compiled by the Provincial Education Ministry.",
            "Les donnees sur le decrochage scolaire sont compilees par le ministere provincial de l'Education.",
        ),
    ],
}

QUERY_SYSTEM_PROMPT = (
    "You are a professional translator localizing a live chat between a citizen and a "
    "Statistics Canada data-request agent. Translate the message from English into "
    "{lang_name}. Keep the same meaning, tone, and register -- including typos, "
    "hesitations, or awkward phrasing -- but produce something that reads naturally to a "
    "native {lang_name} speaker. Do not add, remove, or explain anything. "
    "Reply with ONLY the translated sentence: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(examples: dict, lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in examples.get(lang_code, [])]
    return "\n\n".join(lines)


def build_vocab_system_prompt(lang_code: str) -> str:
    return VOCAB_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(VOCAB_EXAMPLES, lang_code))


def build_query_system_prompt(lang_code: str) -> str:
    return QUERY_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(QUERY_EXAMPLES, lang_code))


# --- Corpus parsing / reconstruction -----------------------------------------------

HEADER_RE = re.compile(r"^(Title|Date range|Dimensions|Subject|Survey|Frequency):\s*(.*)$")
ID_LINE_RE = re.compile(r"^ID: (\S+), Parent: (\S+), Name: (.*)$")


def parse_doc(text: str) -> list[tuple]:
    segments = []
    for line in text.split("\n"):
        if not line.strip():
            segments.append(("blank", line))
            continue
        m = ID_LINE_RE.match(line)
        if m:
            segments.append(("id", m.group(1), m.group(2), m.group(3)))
            continue
        m = HEADER_RE.match(line.strip())
        if m:
            segments.append(("header", m.group(1), m.group(2)))
            continue
        if line.rstrip().endswith(":"):
            segments.append(("block", line.rstrip()[:-1]))
            continue
        # Malformed line (embedded newline in the source Name/Title field) -- passed
        # through untranslated. Negligible: ~9 lines across the whole corpus.
        segments.append(("raw", line))
    return segments


def format_header_line(label: str, value: str) -> str:
    # Matches the source exactly when value is empty: "Frequency:" not "Frequency: ".
    return f"{label}: {value}" if value else f"{label}:"


def reconstruct_doc(segments: list[tuple], lookup: dict[str, str], labels: dict[str, str]) -> str:
    lines = []
    for seg in segments:
        kind = seg[0]
        if kind in ("blank", "raw"):
            lines.append(seg[1])
        elif kind == "id":
            _, id_, parent, name = seg
            lines.append(f"ID: {id_}, Parent: {parent}, Name: {lookup.get(name, name)}")
        elif kind == "block":
            dim_name = seg[1]
            lines.append(f"{lookup.get(dim_name, dim_name)}:")
        elif kind == "header":
            _, label, value = seg
            translated_label = labels.get(label, label)
            if label == "Date range":
                lines.append(format_header_line(translated_label, value))
            elif label == "Dimensions":
                dims = [d.strip() for d in value.split(",") if d.strip()]
                translated_value = ", ".join(lookup.get(d, d) for d in dims)
                lines.append(format_header_line(translated_label, translated_value))
            else:
                lines.append(format_header_line(translated_label, lookup.get(value, value)))
    return "\n".join(lines)


def collect_vocab(parsed_docs: dict[str, list[tuple]]) -> list[str]:
    all_strings = set()
    for segs in parsed_docs.values():
        for seg in segs:
            if seg[0] == "header" and seg[1] in ("Title", "Subject", "Survey", "Frequency"):
                all_strings.add(seg[2])
            elif seg[0] == "block":
                all_strings.add(seg[1])
            elif seg[0] == "id":
                all_strings.add(seg[3])
    all_strings.discard("")
    return sorted(all_strings)


def load_statcan():
    raw_corpus = load_dataset("mteb/StatcanDialogueDatasetRetrieval", "english-corpus")
    raw_queries = load_dataset("mteb/StatcanDialogueDatasetRetrieval", "english-queries")
    raw_qrels = load_dataset("mteb/StatcanDialogueDatasetRetrieval", "english-qrels")

    dev_map = {r["_id"]: r["text"] for r in raw_corpus["dev"]}
    test_map = {r["_id"]: r["text"] for r in raw_corpus["test"]}
    assert dev_map == test_map, "expected dev/test corpus to be identical"
    print(f"corpus: {len(dev_map)} unique docs (shared by dev/test)")
    print(f"queries: dev={len(raw_queries['dev'])}, test={len(raw_queries['test'])}")
    print(f"qrels: dev={len(raw_qrels['dev'])}, test={len(raw_qrels['test'])}")

    return raw_corpus, raw_queries, raw_qrels, dev_map


def vocab_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"vocab_{lang_code}_{model_key}.jsonl"


def queries_checkpoint_path(split_name: str, lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"queries_{split_name}_{lang_code}_{model_key}.jsonl"


VOCAB_KEY_FIELDS = ("en",)
QUERY_KEY_FIELDS = ("query_id", "turn_idx")


def build_query_units(dataset) -> list[tuple[tuple, str, dict]]:
    return [
        ((row["_id"], turn_idx), turn["content"], {"role": turn["role"], "content": turn["content"]})
        for row in dataset
        for turn_idx, turn in enumerate(row["text"])
    ]


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _, dev_map = load_statcan()
    parsed_docs = {doc_id: parse_doc(text) for doc_id, text in dev_map.items()}
    all_strings = collect_vocab(parsed_docs)
    print(f"{len(all_strings)} unique vocabulary strings to translate")

    for lang_code in args.langs:
        vocab_units = [((s,), s) for s in all_strings]
        translator.translate_units(
            vocab_units, build_vocab_system_prompt(lang_code), vocab_checkpoint_path(lang_code, args.model),
            VOCAB_KEY_FIELDS,
        )

    for split_name in args.splits:
        units = build_query_units(raw_queries[split_name])
        for lang_code in args.langs:
            translator.translate_units(
                units, build_query_system_prompt(lang_code), queries_checkpoint_path(split_name, lang_code, args.model),
                QUERY_KEY_FIELDS,
            )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _, dev_map = load_statcan()
    parsed_docs = {doc_id: parse_doc(text) for doc_id, text in dev_map.items()}
    all_strings = collect_vocab(parsed_docs)
    sample_strings = all_strings[:20]

    for lang_code in args.langs:
        vocab_units = [((s,), s) for s in sample_strings]
        out_path = OUT_DIR / f"smoketest_vocab_{lang_code}_{args.model}.jsonl"
        result = translator.translate_units(vocab_units, build_vocab_system_prompt(lang_code), out_path, VOCAB_KEY_FIELDS)
        for s in sample_strings[:8]:
            print(f"[vocab/{lang_code}/{args.model}] {s!r} -> {result[(s,)]!r}")

        sample_query_units = build_query_units(raw_queries["dev"].select(range(3)))
        query_out = OUT_DIR / f"smoketest_queries_{lang_code}_{args.model}.jsonl"
        query_result = translator.translate_units(sample_query_units, build_query_system_prompt(lang_code), query_out, QUERY_KEY_FIELDS)
        for key, text, _ in sample_query_units:
            print(f"[queries/{lang_code}/{args.model}] {text[:80]!r} -> {query_result[key][:80]!r}")
        print()


def cmd_assemble(args):
    raw_corpus, raw_queries, raw_qrels, dev_map = load_statcan()
    parsed_docs = {doc_id: parse_doc(text) for doc_id, text in dev_map.items()}
    all_strings = collect_vocab(parsed_docs)
    raw_fallback_count = sum(1 for segs in parsed_docs.values() for s in segs if s[0] == "raw")
    print(f"parsed {len(parsed_docs)} docs; {raw_fallback_count} untranslatable fallback lines")

    final = {}
    translated_docs_by_key = {}
    for lang_code in args.langs:
        for model_key in MODELS:
            vocab_out = vocab_checkpoint_path(lang_code, model_key)
            if not vocab_out.exists():
                print(f"skipping {lang_code}/{model_key}: no vocab checkpoint at {vocab_out}")
                continue
            vocab_lookup_raw = load_checkpoint(vocab_out, VOCAB_KEY_FIELDS)
            missing_vocab = [s for s in all_strings if (s,) not in vocab_lookup_raw]
            if missing_vocab:
                print(
                    f"skipping {lang_code}/{model_key}: vocab checkpoint incomplete "
                    f"({len(vocab_lookup_raw)}/{len(all_strings)} translated, "
                    f"{len(missing_vocab)} missing e.g. {missing_vocab[:3]}) -- run `translate` first"
                )
                continue
            vocab_lookup = {k[0]: v for k, v in vocab_lookup_raw.items()}

            translated_docs = {
                doc_id: reconstruct_doc(segs, vocab_lookup, HEADER_LABELS[lang_code])
                for doc_id, segs in parsed_docs.items()
            }
            translated_docs_by_key[(lang_code, model_key)] = translated_docs

            queries_ok = True
            translated_queries = {}
            for split_name in args.splits:
                out_path = queries_checkpoint_path(split_name, lang_code, model_key)
                if not out_path.exists():
                    print(f"skipping queries {lang_code}/{model_key}/{split_name}: no checkpoint at {out_path}")
                    queries_ok = False
                    break
                expected_units = build_query_units(raw_queries[split_name])
                lookup = load_checkpoint(out_path, QUERY_KEY_FIELDS)
                missing = [key for key, _, _ in expected_units if key not in lookup]
                if missing:
                    print(
                        f"skipping queries {lang_code}/{model_key}/{split_name}: checkpoint incomplete "
                        f"({len(lookup)}/{len(expected_units)} translated, {len(missing)} missing "
                        f"e.g. {missing[:3]}) -- run `translate` for this model/split first"
                    )
                    queries_ok = False
                    break
                translated_queries[split_name] = lookup
            if not queries_ok:
                continue

            corpus_rows = [{"_id": doc_id, "text": text, "title": ""} for doc_id, text in translated_docs.items()]
            corpus_ds = Dataset.from_list(corpus_rows)

            def build_queries_dataset(split_name):
                lookup = translated_queries[split_name]
                rows = []
                for row in raw_queries[split_name]:
                    turns = [
                        {"role": turn["role"], "content": lookup[(row["_id"], i)]}
                        for i, turn in enumerate(row["text"])
                    ]
                    rows.append({"_id": row["_id"], "text": turns})
                return Dataset.from_list(rows)

            final[(lang_code, model_key, "corpus")] = DatasetDict({"dev": corpus_ds, "test": corpus_ds})
            final[(lang_code, model_key, "queries")] = DatasetDict({
                split_name: build_queries_dataset(split_name) for split_name in args.splits
            })
            final[(lang_code, model_key, "qrels")] = DatasetDict({
                split_name: raw_qrels[split_name] for split_name in args.splits
            })
            print(lang_code, model_key, "corpus", len(corpus_rows), "docs")

    if not final:
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check:
        lang_code, model_key = next(iter(translated_docs_by_key))
        translated_docs = translated_docs_by_key[(lang_code, model_key)]
        sample_ids = random.sample(list(dev_map), min(3, len(dev_map)))
        for doc_id in sample_ids:
            print("=" * 20, doc_id)
            print("EN:", dev_map[doc_id][:300])
            print(f"{lang_code.upper()} [{model_key}]:", translated_docs[doc_id][:300])
            print()

    for (lang_code, model_key, part), dd in final.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-{part}"))
    print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/statcan-mt"
    # for (lang_code, model_key, part), dd in final.items():
    #     dd.push_to_hub(repo_id, config_name=f"{lang_code}-{model_key}-{part}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p, model_required=True):
        p.add_argument("--model", choices=list(MODELS), required=model_required)
        p.add_argument("--langs", nargs="+", choices=list(LANGUAGES), default=DEFAULT_LANGS)

    def add_engine_kwargs(p):
        p.add_argument(
            "--engine-kwarg", action="append", default=[], metavar="KEY=VALUE",
            help="extra vllm.LLM(...) kwarg, repeatable, e.g. "
                 "--engine-kwarg tensor_parallel_size=2 --engine-kwarg gpu_memory_utilization=0.85",
        )

    p_translate = sub.add_parser("translate", help="Translate with one model")
    add_common(p_translate)
    add_engine_kwargs(p_translate)
    p_translate.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    p_translate.set_defaults(func=cmd_translate)

    p_smoke = sub.add_parser("smoke-test", help="Translate a small vocab/query sample and print it")
    add_common(p_smoke)
    add_engine_kwargs(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke_test)

    p_assemble = sub.add_parser("assemble", help="Build final datasets from existing checkpoints (no vLLM)")
    add_common(p_assemble, model_required=False)
    p_assemble.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    p_assemble.add_argument("--spot-check", action="store_true", default=True)
    p_assemble.add_argument("--no-spot-check", dest="spot_check", action="store_false")
    p_assemble.set_defaults(func=cmd_assemble)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
