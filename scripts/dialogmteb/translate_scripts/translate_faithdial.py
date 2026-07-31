#!/usr/bin/env python3
"""Translate mteb/FaithDial to Spanish using offline vLLM inference.

FaithDial is a knowledge-grounded dialogue retrieval dataset: `corpus` (3,539
Wikipedia-derived knowledge sentences, `test` split only), `queries` (2,042 rows,
`qrels` (relevance judgments, no text -- copied through unchanged).

`queries.text` has the same growing-prefix structure as DeepPavlov/daily_dialog (each
row's `text` is a growing list of conversation turns, no role field, no explicit
conversation-id) -- pooled and deduplicated globally rather than reconstructed from
sibling rows: 10,060 naive turn occurrences collapse to 5,227 unique strings (~1.9x).
`corpus.text` also repeats across rows (3,539 rows, 2,182 unique texts, ~1.6x) and is
deduplicated the same way. `corpus.title` is always empty in this dataset -- copied
through as-is, no translation needed.

Usage (run one model per process -- see translate_common.py for why):

    python translate_faithdial.py translate --model gemma
    python translate_faithdial.py smoke-test --model gemma
    python translate_faithdial.py assemble    # no vLLM needed, reads whatever models were run

Checkpointed to translations/faithdial/*.jsonl -- safe to interrupt/resume.
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

OUT_DIR = SCRIPT_DIR / "translations" / "faithdial"
SAVE_DIR = SCRIPT_DIR / "translations" / "faithdial_final"

CORPUS_EXAMPLES = {
    "es": [
        (
            "Dylan's Candy Bar is a chain of boutique candy shops and candy supplier currently located in New York City; East Hampton, New York; Los Angeles, Chicago and Miami Beach.",
            "Dylan's Candy Bar es una cadena de tiendas boutique de caramelos y proveedor de dulces ubicada actualmente en la ciudad de Nueva York; East Hampton, Nueva York; Los Angeles, Chicago y Miami Beach.",
        ),
    ],
    "fr": [
        (
            "Dylan's Candy Bar is a chain of boutique candy shops and candy supplier currently located in New York City; East Hampton, New York; Los Angeles, Chicago and Miami Beach.",
            "Dylan's Candy Bar est une chaine de boutiques de bonbons et fournisseur de confiseries actuellement situee a New York ; East Hampton, New York ; Los Angeles, Chicago et Miami Beach.",
        ),
    ],
}

CORPUS_SYSTEM_PROMPT = (
    "You are a professional translator localizing Wikipedia-derived knowledge "
    "sentences for a conversational search system. Translate the text from English "
    "into {lang_name}, producing formal, accurate, encyclopedic {lang_name} prose. "
    "Keep proper nouns (people, places, organizations) as their correct {lang_name} "
    "form where a standard one exists, otherwise leave them as written. Preserve "
    "facts, numbers, and dates exactly. Do not add, remove, summarize, or explain "
    "anything. Reply with ONLY the translation: no quotes, no notes.\n\n"
    "Examples:\n{examples_block}"
)

QUERY_EXAMPLES = {
    "es": [
        ("I love candy, what's a good brand?", "Me encantan los dulces, cual es una buena marca?"),
        (
            "I don't know how good they are, but Dylan's Candy Bar has a chain of candy shops in various cities.",
            "No se que tan buenos son, pero Dylan's Candy Bar tiene una cadena de tiendas de dulces en varias ciudades.",
        ),
    ],
    "fr": [
        ("I love candy, what's a good brand?", "J'adore les bonbons, quelle est une bonne marque ?"),
        (
            "I don't know how good they are, but Dylan's Candy Bar has a chain of candy shops in various cities.",
            "Je ne sais pas s'ils sont bons, mais Dylan's Candy Bar possede une chaine de boutiques de bonbons dans plusieurs villes.",
        ),
    ],
}

QUERY_SYSTEM_PROMPT = (
    "You are a professional translator localizing turns of a casual knowledge-grounded "
    "conversation (a user asking about a topic, an assistant replying with facts). "
    "Translate the text from English into {lang_name}, keeping the same meaning and "
    "casual conversational register, natural to a native {lang_name} speaker. Keep "
    "proper nouns as their correct {lang_name} form where one exists. Do not add, "
    "remove, or explain anything. "
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


def load_faithdial():
    raw_corpus = load_dataset("mteb/FaithDial", "corpus")
    raw_queries = load_dataset("mteb/FaithDial", "queries")
    raw_qrels = load_dataset("mteb/FaithDial", "qrels")
    print(f"corpus: {len(raw_corpus['test'])} docs")
    print(f"queries: {len(raw_queries['test'])} rows")
    return raw_corpus, raw_queries, raw_qrels


def collect_corpus_texts(raw_corpus) -> list[str]:
    texts = set(raw_corpus["test"]["text"])
    texts.discard("")
    return sorted(texts)


def collect_query_texts(raw_queries) -> list[str]:
    texts = set()
    for row in raw_queries["test"]:
        texts.update(row["text"])
    texts.discard("")
    return sorted(texts)


def corpus_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"corpus_{lang_code}_{model_key}.jsonl"


def queries_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"queries_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _ = load_faithdial()
    corpus_texts = collect_corpus_texts(raw_corpus)
    query_texts = collect_query_texts(raw_queries)
    print(f"{len(corpus_texts)} unique corpus texts, {len(query_texts)} unique query texts to translate")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in corpus_texts], build_corpus_system_prompt(lang_code),
            corpus_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )
        translator.translate_units(
            [((s,), s) for s in query_texts], build_query_system_prompt(lang_code),
            queries_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _ = load_faithdial()
    c_sample = raw_corpus["test"]["text"][:5]
    q_sample = raw_queries["test"].select(range(3))

    for lang_code in args.langs:
        c_out = OUT_DIR / f"smoketest_corpus_{lang_code}_{args.model}.jsonl"
        c_result = translator.translate_units([((s,), s) for s in c_sample], build_corpus_system_prompt(lang_code), c_out, KEY_FIELDS)
        for s in c_sample[:2]:
            print(f"[corpus/{lang_code}/{args.model}] {s[:80]!r} -> {c_result[(s,)][:80]!r}")

        q_texts = sorted({u for row in q_sample for u in row["text"]})
        q_out = OUT_DIR / f"smoketest_queries_{lang_code}_{args.model}.jsonl"
        q_result = translator.translate_units([((s,), s) for s in q_texts], build_query_system_prompt(lang_code), q_out, KEY_FIELDS)
        for row in q_sample:
            print(f"[query/{lang_code}/{args.model}] turns[-1]: {row['text'][-1]!r} -> {q_result[(row['text'][-1],)]!r}")
        print()


def cmd_assemble(args):
    raw_corpus, raw_queries, raw_qrels = load_faithdial()
    corpus_texts = collect_corpus_texts(raw_corpus)
    query_texts = collect_query_texts(raw_queries)

    final_corpus, final_queries = {}, {}
    for lang_code in args.langs:
        for model_key in MODELS:
            c_path = corpus_checkpoint_path(lang_code, model_key)
            q_path = queries_checkpoint_path(lang_code, model_key)
            if not c_path.exists() or not q_path.exists():
                print(f"skipping {lang_code}/{model_key}: missing checkpoint(s) at {c_path} / {q_path}")
                continue

            c_lookup_raw = load_checkpoint(c_path, KEY_FIELDS)
            q_lookup_raw = load_checkpoint(q_path, KEY_FIELDS)
            missing_c = [s for s in corpus_texts if (s,) not in c_lookup_raw]
            missing_q = [s for s in query_texts if (s,) not in q_lookup_raw]
            if missing_c or missing_q:
                print(
                    f"skipping {lang_code}/{model_key}: incomplete "
                    f"({len(corpus_texts) - len(missing_c)}/{len(corpus_texts)} corpus, "
                    f"{len(query_texts) - len(missing_q)}/{len(query_texts)} query texts translated) "
                    f"-- run `translate` first"
                )
                continue

            c_lookup = {k[0]: v for k, v in c_lookup_raw.items()}
            q_lookup = {k[0]: v for k, v in q_lookup_raw.items()}

            corpus_rows = [
                {"_id": row["_id"], "text": c_lookup.get(row["text"], row["text"]), "title": row["title"]}
                for row in raw_corpus["test"]
            ]
            final_corpus[(lang_code, model_key)] = DatasetDict({"test": Dataset.from_list(corpus_rows)})

            query_rows = [
                {"_id": row["_id"], "text": [q_lookup.get(u, u) for u in row["text"]]}
                for row in raw_queries["test"]
            ]
            final_queries[(lang_code, model_key)] = DatasetDict({"test": Dataset.from_list(query_rows)})

            print(lang_code, model_key, "corpus", len(corpus_rows), "queries", len(query_rows))

    if not (final_corpus or final_queries):
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check and final_corpus and final_queries:
        lang_code, model_key = next(iter(final_corpus))
        print("CORPUS:", final_corpus[(lang_code, model_key)]["test"][0]["text"][:250])
        qrow = final_queries[(lang_code, model_key)]["test"][1]
        for u in qrow["text"]:
            print("  ", u[:150])

    for (lang_code, model_key), dd in final_corpus.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-corpus"))
    for (lang_code, model_key), dd in final_queries.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-queries"))
    if final_corpus or final_queries:
        raw_qrels.save_to_disk(str(SAVE_DIR / "qrels"))
        print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/faithdial-mt"
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
