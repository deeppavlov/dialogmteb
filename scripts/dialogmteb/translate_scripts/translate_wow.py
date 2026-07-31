#!/usr/bin/env python3
"""Translate DeepPavlov/wizard_of_wikipedia to French and Spanish using offline vLLM.

Wizard of Wikipedia is a knowledge-grounded conversation **retrieval** dataset:
`corpus` (Wikipedia knowledge passages), `queries` (persona-grounded chit-chat about a
topic), `qrels` (relevance judgments, no text -- copied through unchanged).

Corpus dedup: `test`/`valid` corpus doc ids are a confirmed subset of `train`'s, so the
corpus is translated once (train's 165,023 docs) and reused for all three splits.

Query dedup: each conversation is stored repeated across ids like `query_5_0`,
`query_5_1`, ..., `query_5_8` -- verified that every row sharing a `query_{conv}_*`
prefix has byte-identical persona/topic/text. So translation happens once per unique
conversation and gets replayed across every row in its group (~9x fewer translations
than the raw row count).

Usage (run one model per process -- see translate_common.py for why):

    python translate_wow.py translate --model gemma
    python translate_wow.py translate --model qwen
    python translate_wow.py translate --model gemma --max-corpus-docs 500  # test first
    python translate_wow.py smoke-test --model gemma
    python translate_wow.py assemble    # no vLLM needed, reads both checkpoints
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from datasets import Dataset, DatasetDict, load_dataset  # noqa: E402

from translate_common import MODELS, OfflineTranslator, load_checkpoint, parse_engine_kwargs  # noqa: E402

LANGUAGES = {
    "fr": "French",
    "es": "Spanish",
}

QUERY_SPLITS = ["valid", "test", "train"]  # smallest/fastest-feedback split first
OUT_DIR = SCRIPT_DIR / "translations" / "wow"
SAVE_DIR = SCRIPT_DIR / "translations" / "wow_final"

CORPUS_BATCH_SIZE = 300  # smaller than the default: individual docs can be very long (up to ~136k chars)

CORPUS_EXAMPLES = {
    "fr": [
        (
            "A pharmacy technician is a health care provider who performs pharmacy-related functions.",
            "Un technicien en pharmacie est un professionnel de sante qui exerce des fonctions liees a la pharmacie.",
        ),
    ],
    "es": [
        (
            "A pharmacy technician is a health care provider who performs pharmacy-related functions.",
            "Un tecnico de farmacia es un profesional de la salud que realiza funciones relacionadas con la farmacia.",
        ),
    ],
}

CORPUS_SYSTEM_PROMPT = (
    "You are a professional translator localizing Wikipedia knowledge-base passages for "
    "a conversational search system. Translate the text from English into {lang_name}, "
    "producing formal, accurate, encyclopedic {lang_name} prose. Keep proper nouns "
    "(people, places, works, organizations) as their correct {lang_name} form where a "
    "standard one exists, otherwise leave them as written. Preserve facts, numbers, "
    "and dates exactly. Do not add, remove, summarize, or explain anything -- translate "
    "the full passage. Reply with ONLY the translation: no quotes, no notes.\n\n"
    "Examples:\n{examples_block}"
)

QUERY_EXAMPLES = {
    "fr": [
        (
            "I think science fiction is an amazing genre for anything.",
            "Je trouve que la science-fiction est un genre formidable pour a peu pres tout.",
        ),
        ("Science fiction", "Science-fiction"),
        ("my mother met elvis.", "ma mere a rencontre elvis."),
    ],
    "es": [
        (
            "I think science fiction is an amazing genre for anything.",
            "Creo que la ciencia ficcion es un genero increible para casi cualquier cosa.",
        ),
        ("Science fiction", "Ciencia ficcion"),
        ("my mother met elvis.", "mi madre conocio a elvis."),
    ],
}

QUERY_SYSTEM_PROMPT = (
    "You are a professional translator localizing a casual chit-chat conversation "
    "between two people discussing a topic (persona-grounded dialogue). Translate the "
    "given text from English into {lang_name}. It may be one line of dialogue, a short "
    "persona trait sentence, or a short topic/subject name -- in every case, keep the "
    "same meaning, tone, and casual register, using natural {lang_name}. Keep proper "
    "nouns (people, places, titles of works) as their correct {lang_name} form where one "
    "exists. Do not add, remove, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(examples: dict, lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {es}" for en, es in examples.get(lang_code, [])]
    return "\n\n".join(lines)


def build_corpus_system_prompt(lang_code: str) -> str:
    return CORPUS_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(CORPUS_EXAMPLES, lang_code))


def build_query_system_prompt(lang_code: str) -> str:
    return QUERY_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(QUERY_EXAMPLES, lang_code))


CORPUS_KEY_FIELDS = ("doc_id", "kind")
QUERY_KEY_FIELDS = ("conv_id", "kind", "turn_idx")


def corpus_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"corpus_{lang_code}_{model_key}.jsonl"


def queries_checkpoint_path(split_name: str, lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"queries_{split_name}_{lang_code}_{model_key}.jsonl"


def conv_id_of(row_id: str) -> str:
    return row_id.rsplit("_", 1)[0]


def group_queries(dataset) -> dict[str, list]:
    groups: dict[str, list] = {}
    for row in dataset:
        groups.setdefault(conv_id_of(row["id"]), []).append(row)
    return groups


def verify_query_groups(query_groups: dict[str, dict]) -> None:
    inconsistent = []
    for split, groups in query_groups.items():
        for conv_id, rows in groups.items():
            sigs = {(r["persona"], r["topic"], tuple(t["content"] for t in r["text"])) for r in rows}
            if len(sigs) > 1:
                inconsistent.append((split, conv_id))
        print(f"queries/{split}: {sum(len(v) for v in groups.values())} rows -> {len(groups)} unique conversations")
    if inconsistent:
        print(
            f"WARNING: {len(inconsistent)} conversation groups are NOT identical across "
            f"their rows (e.g. {inconsistent[:5]}) -- those groups' translations will "
            f"only reflect their first row."
        )
    else:
        print("verified: every conversation group is identical across its rows -- safe to dedupe")


def load_wow():
    raw_corpus = load_dataset("DeepPavlov/wizard_of_wikipedia", "corpus")
    raw_queries = load_dataset("DeepPavlov/wizard_of_wikipedia", "queries")
    raw_qrels = load_dataset("DeepPavlov/wizard_of_wikipedia", "qrels")

    corpus_ids = {split: set(raw_corpus[split]["id"]) for split in raw_corpus}
    assert corpus_ids["test"] <= corpus_ids["train"], "expected test corpus ids to be a subset of train"
    assert corpus_ids["valid"] <= corpus_ids["train"], "expected valid corpus ids to be a subset of train"

    for split in raw_corpus:
        print(f"corpus/{split}: {len(raw_corpus[split])} docs")
    for split in raw_queries:
        print(f"queries/{split}: {len(raw_queries[split])} rows")

    return raw_corpus, raw_queries, raw_qrels


def build_corpus_units(raw_corpus, max_docs: int | None) -> list[tuple[tuple, str]]:
    docs = raw_corpus["train"]
    if max_docs is not None:
        docs = docs.select(range(min(max_docs, len(docs))))
    units = []
    for row in docs:
        units.append(((row["id"], "title"), row["title"]))
        units.append(((row["id"], "text"), row["text"]))
    return units


def build_query_units(groups: dict[str, list]) -> list[tuple[tuple, str]]:
    units = []
    for conv_id, rows in groups.items():
        rep = rows[0]  # verified identical across the group by verify_query_groups()
        units.append(((conv_id, "persona", 0), rep["persona"]))
        units.append(((conv_id, "topic", 0), rep["topic"]))
        for turn_idx, turn in enumerate(rep["text"]):
            units.append(((conv_id, "turn", turn_idx), turn["content"]))
    return units


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _ = load_wow()
    query_groups = {split: group_queries(raw_queries[split]) for split in raw_queries}
    verify_query_groups(query_groups)

    for lang_code in args.langs:
        units = build_corpus_units(raw_corpus, args.max_corpus_docs)
        translator.translate_units(
            units, build_corpus_system_prompt(lang_code), corpus_checkpoint_path(lang_code, args.model),
            CORPUS_KEY_FIELDS, batch_size=CORPUS_BATCH_SIZE,
        )

    for split_name in args.splits:
        units = build_query_units(query_groups[split_name])
        for lang_code in args.langs:
            translator.translate_units(
                units, build_query_system_prompt(lang_code), queries_checkpoint_path(split_name, lang_code, args.model),
                QUERY_KEY_FIELDS,
            )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _ = load_wow()
    query_groups = {split: group_queries(raw_queries[split]) for split in raw_queries}

    smoke_conv_ids = list(query_groups["valid"])[:3]
    smoke_groups = {cid: query_groups["valid"][cid] for cid in smoke_conv_ids}

    for lang_code in args.langs:
        corpus_units = build_corpus_units(raw_corpus, max_docs=5)
        corpus_out = OUT_DIR / f"smoketest_corpus_{lang_code}_{args.model}.jsonl"
        corpus_result = translator.translate_units(
            corpus_units, build_corpus_system_prompt(lang_code), corpus_out, CORPUS_KEY_FIELDS,
        )
        for row in raw_corpus["train"].select(range(2)):
            print(f"[corpus/{lang_code}/{args.model}] {row['title']!r} -> {corpus_result[(row['id'], 'title')]!r}")

        query_units = build_query_units(smoke_groups)
        query_out = OUT_DIR / f"smoketest_queries_{lang_code}_{args.model}.jsonl"
        query_result = translator.translate_units(
            query_units, build_query_system_prompt(lang_code), query_out, QUERY_KEY_FIELDS,
        )
        for conv_id in smoke_conv_ids[:1]:
            rep = smoke_groups[conv_id][0]
            print(f"[queries/{lang_code}/{args.model}] TOPIC: {rep['topic']!r} -> {query_result[(conv_id, 'topic', 0)]!r}")
            print(f"[queries/{lang_code}/{args.model}] PERSONA: {rep['persona']!r} -> {query_result[(conv_id, 'persona', 0)]!r}")
        print()


def cmd_assemble(args):
    raw_corpus, raw_queries, raw_qrels = load_wow()
    query_groups = {split: group_queries(raw_queries[split]) for split in raw_queries}

    final_corpus = {}
    for lang_code in args.langs:
        for model_key in MODELS:
            out_path = corpus_checkpoint_path(lang_code, model_key)
            if not out_path.exists():
                print(f"skipping corpus {lang_code}/{model_key}: no checkpoint at {out_path}")
                continue
            lookup = load_checkpoint(out_path, CORPUS_KEY_FIELDS)
            by_id = {}
            for row in raw_corpus["train"]:
                key_title, key_text = (row["id"], "title"), (row["id"], "text")
                if key_title not in lookup or key_text not in lookup:
                    continue  # not yet translated (e.g. --max-corpus-docs was used, or still running)
                by_id[row["id"]] = {
                    "id": row["id"],
                    "title": lookup[key_title],
                    "text": lookup[key_text],
                }
            n_docs_total = min(args.max_corpus_docs, len(raw_corpus["train"])) if getattr(args, "max_corpus_docs", None) else len(raw_corpus["train"])
            if len(by_id) < n_docs_total:
                print(f"note: corpus {lang_code}/{model_key} has {len(by_id)}/{n_docs_total} docs translated so far -- assembling the partial set")
            dd = DatasetDict()
            for split_name in raw_corpus:
                rows = [by_id[doc_id] for doc_id in raw_corpus[split_name]["id"] if doc_id in by_id]
                dd[split_name] = Dataset.from_list(rows)
            final_corpus[(lang_code, model_key)] = dd
            print("corpus", lang_code, model_key, {s: len(dd[s]) for s in dd})

    final_queries = {}
    for lang_code in args.langs:
        for model_key in MODELS:
            dd = DatasetDict()
            incomplete = False
            for split_name in args.splits:
                out_path = queries_checkpoint_path(split_name, lang_code, model_key)
                if not out_path.exists():
                    print(f"skipping queries {lang_code}/{model_key}/{split_name}: no checkpoint at {out_path}")
                    incomplete = True
                    break
                expected_units = build_query_units(query_groups[split_name])
                lookup = load_checkpoint(out_path, QUERY_KEY_FIELDS)
                missing = [key for key, _ in expected_units if key not in lookup]
                if missing:
                    print(
                        f"skipping queries {lang_code}/{model_key}/{split_name}: checkpoint incomplete "
                        f"({len(lookup)}/{len(expected_units)} translated, {len(missing)} missing e.g. {missing[:3]}) "
                        f"-- run `translate` for this model/split first"
                    )
                    incomplete = True
                    break
                rows = []
                for row in raw_queries[split_name]:
                    conv_id = conv_id_of(row["id"])
                    text = [
                        {"content": lookup[(conv_id, "turn", i)], "role": turn["role"]}
                        for i, turn in enumerate(row["text"])
                    ]
                    rows.append({
                        "id": row["id"],
                        "text": text,
                        "persona": lookup[(conv_id, "persona", 0)],
                        "topic": lookup[(conv_id, "topic", 0)],
                    })
                dd[split_name] = Dataset.from_list(rows)
            if not incomplete:
                final_queries[(lang_code, model_key)] = dd
                print("queries", lang_code, model_key, {s: len(dd[s]) for s in dd})

    if args.spot_check and final_corpus and final_queries:
        lang_code, model_key = next(iter(final_corpus))
        row = final_corpus[(lang_code, model_key)]["test"][0]
        print("CORPUS TITLE:", row["title"])
        print("CORPUS TEXT:", row["text"][:300])
        if (lang_code, model_key) in final_queries:
            qrow = final_queries[(lang_code, model_key)]["valid"][0]
            print("TOPIC:", qrow["topic"])
            print("PERSONA:", qrow["persona"])
            for turn in qrow["text"][:3]:
                print(f"  [{turn['role']}]", turn["content"][:150])

    for (lang_code, model_key), dd in final_corpus.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-corpus"))
    for (lang_code, model_key), dd in final_queries.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-queries"))
    if final_corpus or final_queries:
        raw_qrels.save_to_disk(str(SAVE_DIR / "qrels"))
        print("saved to", SAVE_DIR)
    else:
        print("nothing to assemble -- no complete checkpoints found")

    # repo_id = "<your-username>/wizard-of-wikipedia-mt"
    # for (lang_code, model_key), dd in final_corpus.items():
    #     dd.push_to_hub(repo_id, config_name=f"{lang_code}-{model_key}-corpus")
    # for (lang_code, model_key), dd in final_queries.items():
    #     dd.push_to_hub(repo_id, config_name=f"{lang_code}-{model_key}-queries")
    # raw_qrels.push_to_hub(repo_id, config_name="qrels")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p, model_required=True):
        p.add_argument("--model", choices=list(MODELS), required=model_required)
        p.add_argument("--langs", nargs="+", choices=list(LANGUAGES), default=list(LANGUAGES))

    def add_engine_kwargs(p):
        p.add_argument(
            "--engine-kwarg", action="append", default=[], metavar="KEY=VALUE",
            help="extra vllm.LLM(...) kwarg, repeatable, e.g. "
                 "--engine-kwarg tensor_parallel_size=2 --engine-kwarg gpu_memory_utilization=0.85",
        )

    p_translate = sub.add_parser("translate", help="Translate with one model")
    add_common(p_translate)
    add_engine_kwargs(p_translate)
    p_translate.add_argument("--splits", nargs="+", choices=QUERY_SPLITS, default=QUERY_SPLITS,
                              help="query splits to translate (corpus is always translated once from train)")
    p_translate.add_argument("--max-corpus-docs", type=int, default=None,
                              help="translate only the first N corpus docs (for testing before the full 165,023)")
    p_translate.set_defaults(func=cmd_translate)

    p_smoke = sub.add_parser("smoke-test", help="Translate a small corpus/query sample and print it")
    add_common(p_smoke)
    add_engine_kwargs(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke_test)

    p_assemble = sub.add_parser("assemble", help="Build final datasets from existing checkpoints (no vLLM)")
    add_common(p_assemble, model_required=False)
    p_assemble.add_argument("--splits", nargs="+", choices=QUERY_SPLITS, default=QUERY_SPLITS)
    p_assemble.add_argument("--spot-check", action="store_true", default=True)
    p_assemble.add_argument("--no-spot-check", dest="spot_check", action="store_false")
    p_assemble.set_defaults(func=cmd_assemble)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
