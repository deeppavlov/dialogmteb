#!/usr/bin/env python3
"""Translate DeepPavlov/iKAT_2023 to Spanish (and French) using offline vLLM inference.

iKAT 2023 (TREC Interactive Knowledge Assistance Track) is a small personalized
conversational-search retrieval dataset: `corpus` (144 train + 406 test ClueWeb22 web
passages -- travel guides, university pages, product pages, general web content),
`queries` (76 train + 280 test rows, each with `utterance` -- the raw, sometimes
ambiguous/pronoun-laden conversational turn, e.g. "I'd like to stay here." -- and
`text` -- the same turn manually rewritten to be self-contained, e.g. "I'd like to
stay in the Netherlands."), `qrels` (relevance judgments, no text -- copied through
unchanged).

Much smaller than the other DeepPavlov conversational-retrieval datasets (~550 corpus
docs, ~712 query strings total) -- no conversation-id/context-accumulation dedup
needed here (verified: corpus train/test ids don't overlap, and rows are already
almost entirely unique), just per-item translation of two pools: `corpus.text` (web
passages) and the union of `queries.utterance`/`queries.text` (conversational search
turns, both raw and resolved forms -- same register, so pooled together).

Usage (run one model per process -- see translate_common.py for why):

    python translate_ikat.py translate --model gemma
    python translate_ikat.py translate --model qwen
    python translate_ikat.py smoke-test --model gemma
    python translate_ikat.py assemble    # no vLLM needed, reads both checkpoints

Checkpointed to translations/ikat/*.jsonl -- safe to interrupt/resume.
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

SPLITS = ["train", "test"]
OUT_DIR = SCRIPT_DIR / "translations" / "ikat"
SAVE_DIR = SCRIPT_DIR / "translations" / "ikat_final"

CORPUS_EXAMPLES = {
    "es": [
        (
            "Water resource management is the activity of planning, developing, distributing and managing the optimum use of water resources.",
            "La gestion de los recursos hidricos es la actividad de planificar, desarrollar, distribuir y gestionar el uso optimo de los recursos hidricos.",
        ),
    ],
    "fr": [
        (
            "Water resource management is the activity of planning, developing, distributing and managing the optimum use of water resources.",
            "La gestion des ressources en eau est l'activite de planification, de developpement, de distribution et de gestion de l'utilisation optimale des ressources en eau.",
        ),
    ],
}

CORPUS_SYSTEM_PROMPT = (
    "You are a professional translator localizing general web page content (travel "
    "guides, university admissions pages, product pages, encyclopedic articles, and "
    "similar) for a conversational search system. Translate the passage from English "
    "into {lang_name}, producing natural, accurate {lang_name} prose in a register "
    "matching the source (informational, promotional, encyclopedic, etc. as "
    "appropriate). Keep proper nouns (people, places, organizations, product/brand "
    "names) as their correct {lang_name} form where a standard one exists, otherwise "
    "leave them as written. Preserve facts, numbers, and dates exactly. Do not add, "
    "remove, summarize, or explain anything -- translate the full passage. "
    "Reply with ONLY the translation: no quotes, no notes.\n\n"
    "Examples:\n{examples_block}"
)

QUERY_EXAMPLES = {
    "es": [
        ("I'd like to stay here.", "Me gustaria quedarme aqui."),
        ("I'd like to stay in the Netherlands.", "Me gustaria quedarme en los Paises Bajos."),
        ("Which are the best three with the highest ranking?", "Cuales son las tres mejores con la clasificacion mas alta?"),
    ],
    "fr": [
        ("I'd like to stay here.", "J'aimerais rester ici."),
        ("I'd like to stay in the Netherlands.", "J'aimerais rester aux Pays-Bas."),
        ("Which are the best three with the highest ranking?", "Quelles sont les trois meilleures avec le classement le plus eleve ?"),
    ],
}

QUERY_SYSTEM_PROMPT = (
    "You are a professional translator localizing turns of a personalized "
    "conversational search dialogue -- casual, sometimes ambiguous/pronoun-laden "
    "questions and statements about the user's preferences (travel, study, shopping, "
    "etc.). Translate the text from English into {lang_name}, keeping the same "
    "meaning and casual conversational register, natural to a native {lang_name} "
    "speaker. Keep proper nouns (people, places, organizations) as their correct "
    "{lang_name} form where one exists. Do not add, remove, or explain anything -- do "
    "not resolve ambiguous references or add information that isn't in the source. "
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


def load_ikat():
    raw_corpus = load_dataset("DeepPavlov/iKAT_2023", "corpus")
    raw_queries = load_dataset("DeepPavlov/iKAT_2023", "queries")
    raw_qrels = load_dataset("DeepPavlov/iKAT_2023", "qrels")
    for split in raw_corpus:
        print(f"corpus/{split}: {len(raw_corpus[split])} docs")
    for split in raw_queries:
        print(f"queries/{split}: {len(raw_queries[split])} rows")
    return raw_corpus, raw_queries, raw_qrels


def collect_corpus_texts(raw_corpus) -> list[str]:
    texts = set()
    for split in raw_corpus:
        texts.update(raw_corpus[split]["text"])
    texts.discard("")
    return sorted(texts)


def collect_query_texts(raw_queries) -> list[str]:
    texts = set()
    for split in raw_queries:
        texts.update(raw_queries[split]["utterance"])
        texts.update(raw_queries[split]["text"])
    texts.discard("")
    return sorted(texts)


def corpus_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"corpus_{lang_code}_{model_key}.jsonl"


def queries_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"queries_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _ = load_ikat()
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
    raw_corpus, raw_queries, _ = load_ikat()
    c_sample = raw_corpus["train"]["text"][:5]
    q_sample = raw_queries["train"].select(range(5))

    for lang_code in args.langs:
        c_out = OUT_DIR / f"smoketest_corpus_{lang_code}_{args.model}.jsonl"
        c_result = translator.translate_units([((s,), s) for s in c_sample], build_corpus_system_prompt(lang_code), c_out, KEY_FIELDS)
        for s in c_sample[:2]:
            print(f"[corpus/{lang_code}/{args.model}] {s[:80]!r} -> {c_result[(s,)][:80]!r}")

        q_texts = sorted(set(q_sample["utterance"]) | set(q_sample["text"]))
        q_out = OUT_DIR / f"smoketest_queries_{lang_code}_{args.model}.jsonl"
        q_result = translator.translate_units([((s,), s) for s in q_texts], build_query_system_prompt(lang_code), q_out, KEY_FIELDS)
        for row in q_sample:
            print(f"[query/{lang_code}/{args.model}] utterance: {row['utterance']!r} -> {q_result[(row['utterance'],)]!r}")
            print(f"[query/{lang_code}/{args.model}] text:      {row['text']!r} -> {q_result[(row['text'],)]!r}")
        print()


def cmd_assemble(args):
    raw_corpus, raw_queries, raw_qrels = load_ikat()
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

            dd_corpus = DatasetDict()
            for split_name in raw_corpus:
                rows = [{"_id": row["_id"], "text": c_lookup.get(row["text"], row["text"])} for row in raw_corpus[split_name]]
                dd_corpus[split_name] = Dataset.from_list(rows)
            final_corpus[(lang_code, model_key)] = dd_corpus

            dd_queries = DatasetDict()
            for split_name in raw_queries:
                rows = [
                    {
                        "_id": row["_id"],
                        "utterance": q_lookup.get(row["utterance"], row["utterance"]),
                        "text": q_lookup.get(row["text"], row["text"]),
                    }
                    for row in raw_queries[split_name]
                ]
                dd_queries[split_name] = Dataset.from_list(rows)
            final_queries[(lang_code, model_key)] = dd_queries

            print(lang_code, model_key, "corpus", {s: len(dd_corpus[s]) for s in dd_corpus}, "queries", {s: len(dd_queries[s]) for s in dd_queries})

    if not final_corpus:
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check:
        lang_code, model_key = next(iter(final_corpus))
        print("CORPUS:", final_corpus[(lang_code, model_key)]["test"][0]["text"][:250])
        qds = final_queries[(lang_code, model_key)]["test"]
        idxs = random.sample(range(len(qds)), min(3, len(qds)))
        for i in idxs:
            row = qds[i]
            print("UTTERANCE:", row["utterance"])
            print("TEXT:     ", row["text"])
            print()

    for (lang_code, model_key), dd in final_corpus.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-corpus"))
    for (lang_code, model_key), dd in final_queries.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-queries"))
    raw_qrels.save_to_disk(str(SAVE_DIR / "qrels"))
    print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/ikat-2023-mt"
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
