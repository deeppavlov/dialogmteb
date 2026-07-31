#!/usr/bin/env python3
"""Translate mteb/TopiOCQA_validation_top_250_only_w_correct-v2 to Spanish using
offline vLLM inference.

TopiOCQA is a topic-oriented conversational QA retrieval dataset (this variant:
validation split only). `corpus` (89,933 Wikipedia passages, `text` + `title`, e.g.
title "Chad-China relations; Diplomatic Offices"), `queries` (1,000 rows, `text` a
growing list of alternating question/answer turns -- same pattern as
DeepPavlov/daily_dialog and mteb/FaithDial, no role field, ids like `1-1`/`1-4`/`1-5`
that skip numbers so there's no reliable way to reconstruct one row's turns from
another's), qrels (config name is `default`, not `qrels` -- no text, copied through
unchanged).

Both `corpus` and `queries` are deduplicated by pooling all their text globally and
translating each unique string once: queries' 12,850 naive turn occurrences collapse
to 3,915 unique strings (~3.3x); corpus is 89,933 rows but only 88,457 unique texts
and 80,976 unique titles -- verified against the live data. `title` and `text` are
translated as separate pools (short heading-style phrase vs. full sentence).

Usage (run one model per process -- see translate_common.py for why):

    python translate_topiocqa.py translate --model gemma
    python translate_topiocqa.py smoke-test --model gemma
    python translate_topiocqa.py assemble    # no vLLM needed, reads whatever models were run

Checkpointed to translations/topiocqa/*.jsonl -- safe to interrupt/resume.
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
    "es": "Spanish",
    "fr": "French",
}
DEFAULT_LANGS = ["es"]

DATASET_NAME = "mteb/TopiOCQA_validation_top_250_only_w_correct-v2"
OUT_DIR = SCRIPT_DIR / "translations" / "topiocqa"
SAVE_DIR = SCRIPT_DIR / "translations" / "topiocqa_final"

CORPUS_EXAMPLES = {
    "es": [
        (
            "The Chinese embassy is located in N'Djamena. The Chadian embassy is located in Beijing.",
            "La embajada china esta ubicada en N'Djamena. La embajada chadiana esta ubicada en Pekin.",
        ),
        ("Chad-China relations; Diplomatic Offices", "Relaciones Chad-China; Oficinas diplomaticas"),
    ],
    "fr": [
        (
            "The Chinese embassy is located in N'Djamena. The Chadian embassy is located in Beijing.",
            "L'ambassade de Chine est situee a N'Djamena. L'ambassade du Tchad est situee a Pekin.",
        ),
        ("Chad-China relations; Diplomatic Offices", "Relations Tchad-Chine ; Bureaux diplomatiques"),
    ],
}

CORPUS_SYSTEM_PROMPT = (
    "You are a professional translator localizing Wikipedia passages and their "
    "article/section titles for a conversational search system. Translate the text "
    "from English into {lang_name}, producing accurate, natural {lang_name} prose "
    "(or, for titles, a natural short heading). Keep proper nouns as their correct "
    "{lang_name} form where a standard one exists, otherwise leave them as written. "
    "Preserve facts, numbers, and dates exactly. Do not add, remove, summarize, or "
    "explain anything. Reply with ONLY the translation: no quotes, no notes.\n\n"
    "Examples:\n{examples_block}"
)

QUERY_EXAMPLES = {
    "es": [
        ("when will the new dunkirk film be released on dvd", "cuando se lanzara la nueva pelicula de dunkirk en dvd"),
        ("18 December 2017", "18 de diciembre de 2017"),
        ("what is this film about?", "de que trata esta pelicula?"),
    ],
    "fr": [
        ("when will the new dunkirk film be released on dvd", "quand le nouveau film dunkirk sortira-t-il en dvd"),
        ("18 December 2017", "18 decembre 2017"),
        ("what is this film about?", "de quoi parle ce film ?"),
    ],
}

QUERY_SYSTEM_PROMPT = (
    "You are a professional translator localizing turns of a topic-oriented "
    "conversational search dialogue (a user asking follow-up questions, short factual "
    "answers in between). Translate the text from English into {lang_name}, keeping "
    "the same meaning and casual register, natural to a native {lang_name} speaker. "
    "Keep proper nouns as their correct {lang_name} form where one exists. Do not "
    "add, remove, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(examples: dict, lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in examples.get(lang_code, [])]
    return "\n\n".join(lines)


def build_corpus_system_prompt(lang_code: str) -> str:
    return CORPUS_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(CORPUS_EXAMPLES, lang_code))


def build_query_system_prompt(lang_code: str) -> str:
    return QUERY_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(QUERY_EXAMPLES, lang_code))


def load_topiocqa():
    raw_corpus = load_dataset(DATASET_NAME, "corpus")
    raw_queries = load_dataset(DATASET_NAME, "queries")
    raw_qrels = load_dataset(DATASET_NAME, "default")
    print(f"corpus: {len(raw_corpus['validation'])} docs")
    print(f"queries: {len(raw_queries['validation'])} rows")
    print(f"qrels: {len(raw_qrels['validation'])} rows")
    return raw_corpus, raw_queries, raw_qrels


def collect_corpus_texts(raw_corpus) -> list[str]:
    texts = set(raw_corpus["validation"]["text"])
    texts.discard("")
    return sorted(texts)


def collect_corpus_titles(raw_corpus) -> list[str]:
    titles = set(raw_corpus["validation"]["title"])
    titles.discard("")
    return sorted(titles)


def collect_query_texts(raw_queries) -> list[str]:
    texts = set()
    for row in raw_queries["validation"]:
        texts.update(row["text"])
    texts.discard("")
    return sorted(texts)


def corpus_text_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"corpus_text_{lang_code}_{model_key}.jsonl"


def corpus_title_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"corpus_title_{lang_code}_{model_key}.jsonl"


def queries_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"queries_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _ = load_topiocqa()
    corpus_texts = collect_corpus_texts(raw_corpus)
    corpus_titles = collect_corpus_titles(raw_corpus)
    query_texts = collect_query_texts(raw_queries)
    print(f"{len(corpus_texts)} unique corpus texts, {len(corpus_titles)} unique titles, {len(query_texts)} unique query turns")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in corpus_texts], build_corpus_system_prompt(lang_code),
            corpus_text_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )
        translator.translate_units(
            [((s,), s) for s in corpus_titles], build_corpus_system_prompt(lang_code),
            corpus_title_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )
        translator.translate_units(
            [((s,), s) for s in query_texts], build_query_system_prompt(lang_code),
            queries_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _ = load_topiocqa()
    c_sample = raw_corpus["validation"].select(range(5))
    q_sample = raw_queries["validation"].select(range(3))

    for lang_code in args.langs:
        c_texts = sorted(set(c_sample["text"]))
        c_out = OUT_DIR / f"smoketest_corpus_text_{lang_code}_{args.model}.jsonl"
        c_result = translator.translate_units([((s,), s) for s in c_texts], build_corpus_system_prompt(lang_code), c_out, KEY_FIELDS)
        for s in c_texts[:2]:
            print(f"[corpus.text/{lang_code}/{args.model}] {s[:80]!r} -> {c_result[(s,)][:80]!r}")

        c_titles = sorted({t for t in c_sample["title"] if t})
        if c_titles:
            t_out = OUT_DIR / f"smoketest_corpus_title_{lang_code}_{args.model}.jsonl"
            t_result = translator.translate_units([((s,), s) for s in c_titles], build_corpus_system_prompt(lang_code), t_out, KEY_FIELDS)
            for s in c_titles[:2]:
                print(f"[corpus.title/{lang_code}/{args.model}] {s!r} -> {t_result[(s,)]!r}")

        q_texts = sorted({u for row in q_sample for u in row["text"]})
        q_out = OUT_DIR / f"smoketest_queries_{lang_code}_{args.model}.jsonl"
        q_result = translator.translate_units([((s,), s) for s in q_texts], build_query_system_prompt(lang_code), q_out, KEY_FIELDS)
        for row in q_sample:
            print(f"[query/{lang_code}/{args.model}] turns[-1]: {row['text'][-1]!r} -> {q_result[(row['text'][-1],)]!r}")
        print()


def cmd_assemble(args):
    raw_corpus, raw_queries, raw_qrels = load_topiocqa()
    corpus_texts = collect_corpus_texts(raw_corpus)
    corpus_titles = collect_corpus_titles(raw_corpus)
    query_texts = collect_query_texts(raw_queries)

    final_corpus, final_queries = {}, {}
    for lang_code in args.langs:
        for model_key in MODELS:
            ct_path = corpus_text_checkpoint_path(lang_code, model_key)
            tt_path = corpus_title_checkpoint_path(lang_code, model_key)
            q_path = queries_checkpoint_path(lang_code, model_key)
            if not (ct_path.exists() and tt_path.exists() and q_path.exists()):
                print(f"skipping {lang_code}/{model_key}: missing checkpoint(s) among {ct_path}, {tt_path}, {q_path}")
                continue

            ct_lookup_raw = load_checkpoint(ct_path, KEY_FIELDS)
            tt_lookup_raw = load_checkpoint(tt_path, KEY_FIELDS)
            q_lookup_raw = load_checkpoint(q_path, KEY_FIELDS)
            missing_ct = [s for s in corpus_texts if (s,) not in ct_lookup_raw]
            missing_tt = [s for s in corpus_titles if (s,) not in tt_lookup_raw]
            missing_q = [s for s in query_texts if (s,) not in q_lookup_raw]
            if missing_ct or missing_tt or missing_q:
                print(
                    f"skipping {lang_code}/{model_key}: incomplete "
                    f"({len(corpus_texts) - len(missing_ct)}/{len(corpus_texts)} corpus texts, "
                    f"{len(corpus_titles) - len(missing_tt)}/{len(corpus_titles)} titles, "
                    f"{len(query_texts) - len(missing_q)}/{len(query_texts)} query turns translated) "
                    f"-- run `translate` first"
                )
                continue

            ct_lookup = {k[0]: v for k, v in ct_lookup_raw.items()}
            tt_lookup = {k[0]: v for k, v in tt_lookup_raw.items()}
            q_lookup = {k[0]: v for k, v in q_lookup_raw.items()}

            corpus_rows = [
                {
                    "_id": row["_id"],
                    "text": ct_lookup.get(row["text"], row["text"]),
                    "title": tt_lookup.get(row["title"], row["title"]) if row["title"] else row["title"],
                }
                for row in raw_corpus["validation"]
            ]
            final_corpus[(lang_code, model_key)] = DatasetDict({"validation": Dataset.from_list(corpus_rows)})

            query_rows = [
                {"_id": row["_id"], "text": [q_lookup.get(u, u) for u in row["text"]]}
                for row in raw_queries["validation"]
            ]
            final_queries[(lang_code, model_key)] = DatasetDict({"validation": Dataset.from_list(query_rows)})

            print(lang_code, model_key, "corpus", len(corpus_rows), "queries", len(query_rows))

    if not (final_corpus or final_queries):
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check and final_corpus and final_queries:
        lang_code, model_key = next(iter(final_corpus))
        row = final_corpus[(lang_code, model_key)]["validation"][0]
        print("CORPUS TEXT:", row["text"][:250])
        print("CORPUS TITLE:", row["title"])
        qrow = final_queries[(lang_code, model_key)]["validation"][1]
        for u in qrow["text"]:
            print("  ", u[:150])

    for (lang_code, model_key), dd in final_corpus.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-corpus"))
    for (lang_code, model_key), dd in final_queries.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-queries"))
    if final_corpus or final_queries:
        raw_qrels.save_to_disk(str(SAVE_DIR / "qrels"))
        print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/topiocqa-mt"
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
    p_translate.set_defaults(func=cmd_translate)

    p_smoke = sub.add_parser("smoke-test", help="Translate a small corpus/query sample and print it")
    add_common(p_smoke)
    add_engine_kwargs(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke_test)

    p_assemble = sub.add_parser("assemble", help="Build final datasets from existing checkpoints (no vLLM)")
    add_common(p_assemble, model_required=False)
    p_assemble.add_argument("--spot-check", action="store_true", default=True)
    p_assemble.add_argument("--no-spot-check", dest="spot_check", action="store_false")
    p_assemble.set_defaults(func=cmd_assemble)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
