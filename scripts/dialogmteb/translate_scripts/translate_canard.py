#!/usr/bin/env python3
"""Translate DeepPavlov/canard to Spanish using offline vLLM inference.

CANARD is a context-aware question-rewriting retrieval dataset built on QuAC:
`corpus` (24,915 unique Wikipedia-derived answer sentences), `queries` (each with
`text` -- the current, possibly context-dependent question -- and `history` -- the
FLAT STRING concatenation of every prior turn's question+answer in that conversation,
e.g. "Where is Malayali located? 30,803,747 speakers of Malayalam in Kerala..."),
`qrels` (relevance judgments, no text -- copied through unchanged).

`history` is NOT translated by reconstructing it from sibling rows (unlike similar
fields in other DeepPavlov datasets): it's a plain string, not a structured turn list,
so there's no reliable separator to split it back into individual Q/A units without
risking a fragile/incorrect parse -- and per-row it barely deduplicates anyway
(25,747 of 25,750 rows have a unique `history` string, since each is a distinct
growing concatenation). Given the dataset's modest overall scale, this script instead
translates `history` directly as its own pool, alongside separate pools for
`queries.text` (questions, 22,685 unique) and `corpus.text` (answer sentences, 24,915
unique) -- verified against the live data.

Usage (run one model per process -- see translate_common.py for why):

    python translate_canard.py translate --model gemma
    python translate_canard.py smoke-test --model gemma
    python translate_canard.py assemble    # no vLLM needed, reads whatever models were run

Checkpointed to translations/canard/*.jsonl -- safe to interrupt/resume.
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

OUT_DIR = SCRIPT_DIR / "translations" / "canard"
SAVE_DIR = SCRIPT_DIR / "translations" / "canard_final"

CORPUS_EXAMPLES = {
    "es": [
        (
            "30,803,747 speakers of Malayalam in Kerala, making up 93.2% of the total number of Malayalam speakers in India.",
            "30.803.747 hablantes de malayalam en Kerala, lo que representa el 93,2% del total de hablantes de malayalam en la India.",
        ),
    ],
    "fr": [
        (
            "30,803,747 speakers of Malayalam in Kerala, making up 93.2% of the total number of Malayalam speakers in India.",
            "30 803 747 locuteurs du malayalam au Kerala, soit 93,2 % du nombre total de locuteurs du malayalam en Inde.",
        ),
    ],
}

CORPUS_SYSTEM_PROMPT = (
    "You are a professional translator localizing Wikipedia-derived answer sentences "
    "for a conversational question-answering retrieval system. Translate the text "
    "from English into {lang_name}, producing accurate, natural {lang_name} prose. "
    "Keep proper nouns as their correct {lang_name} form where a standard one exists, "
    "otherwise leave them as written. Preserve facts, numbers, dates, and percentages "
    "exactly (adapting number formatting conventions, e.g. decimal separators, to "
    "{lang_name} norms). Do not add, remove, summarize, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes.\n\n"
    "Examples:\n{examples_block}"
)

QA_EXAMPLES = {
    "es": [
        ("What other languages are spoken there?", "Que otros idiomas se hablan alli?"),
        ("Where is Malayali located?", "Donde se encuentra Malayali?"),
        (
            "Where is Malayali located? 30,803,747 speakers of Malayalam in Kerala, making up 93.2% of the total number of Malayalam speakers in India.",
            "Donde se encuentra Malayali? 30.803.747 hablantes de malayalam en Kerala, lo que representa el 93,2% del total de hablantes de malayalam en la India.",
        ),
    ],
    "fr": [
        ("What other languages are spoken there?", "Quelles autres langues y sont parlees ?"),
        ("Where is Malayali located?", "Ou se trouve Malayali ?"),
        (
            "Where is Malayali located? 30,803,747 speakers of Malayalam in Kerala, making up 93.2% of the total number of Malayalam speakers in India.",
            "Ou se trouve Malayali ? 30 803 747 locuteurs du malayalam au Kerala, soit 93,2 % du nombre total de locuteurs du malayalam en Inde.",
        ),
    ],
}

QA_SYSTEM_PROMPT = (
    "You are a professional translator localizing turns of a Wikipedia-grounded "
    "question-answering conversation. The text may be a single question, or a "
    "concatenation of several prior questions and their factual answers (dialogue "
    "history, with no special separator between turns -- just translate straight "
    "through, keeping the same segment order and boundaries implied by the sentence "
    "punctuation). Translate the text from English into {lang_name}, keeping the same "
    "meaning and register, natural to a native {lang_name} speaker. Keep proper nouns "
    "as their correct {lang_name} form where one exists, and preserve facts/numbers/"
    "dates exactly. Do not add, remove, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(examples: dict, lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in examples.get(lang_code, [])]
    return "\n\n".join(lines)


def build_corpus_system_prompt(lang_code: str) -> str:
    return CORPUS_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(CORPUS_EXAMPLES, lang_code))


def build_qa_system_prompt(lang_code: str) -> str:
    return QA_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(QA_EXAMPLES, lang_code))


def load_canard():
    raw_corpus = load_dataset("DeepPavlov/canard", "corpus")
    raw_queries = load_dataset("DeepPavlov/canard", "queries")
    raw_qrels = load_dataset("DeepPavlov/canard", "qrels")
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
        texts.update(raw_queries[split]["text"])
    texts.discard("")
    return sorted(texts)


def collect_histories(raw_queries) -> list[str]:
    texts = set()
    for split in raw_queries:
        for row in raw_queries[split]:
            if row["history"].strip():
                texts.add(row["history"])
    return sorted(texts)


def corpus_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"corpus_{lang_code}_{model_key}.jsonl"


def queries_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"queries_{lang_code}_{model_key}.jsonl"


def history_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"history_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _ = load_canard()
    corpus_texts = collect_corpus_texts(raw_corpus)
    query_texts = collect_query_texts(raw_queries)
    histories = collect_histories(raw_queries)
    print(f"{len(corpus_texts)} unique corpus texts, {len(query_texts)} unique questions, {len(histories)} unique histories")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in corpus_texts], build_corpus_system_prompt(lang_code),
            corpus_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )
        translator.translate_units(
            [((s,), s) for s in query_texts], build_qa_system_prompt(lang_code),
            queries_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )
        translator.translate_units(
            [((s,), s) for s in histories], build_qa_system_prompt(lang_code),
            history_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, _ = load_canard()
    c_sample = raw_corpus["train"]["text"][:5]
    q_sample = raw_queries["train"].select(range(5))

    for lang_code in args.langs:
        c_out = OUT_DIR / f"smoketest_corpus_{lang_code}_{args.model}.jsonl"
        c_result = translator.translate_units([((s,), s) for s in c_sample], build_corpus_system_prompt(lang_code), c_out, KEY_FIELDS)
        for s in c_sample[:2]:
            print(f"[corpus/{lang_code}/{args.model}] {s[:80]!r} -> {c_result[(s,)][:80]!r}")

        q_texts = sorted(set(q_sample["text"]))
        q_out = OUT_DIR / f"smoketest_queries_{lang_code}_{args.model}.jsonl"
        q_result = translator.translate_units([((s,), s) for s in q_texts], build_qa_system_prompt(lang_code), q_out, KEY_FIELDS)
        for row in q_sample:
            print(f"[query/{lang_code}/{args.model}] {row['text']!r} -> {q_result[(row['text'],)]!r}")

        histories = sorted({row["history"] for row in q_sample if row["history"].strip()})
        if histories:
            h_out = OUT_DIR / f"smoketest_history_{lang_code}_{args.model}.jsonl"
            h_result = translator.translate_units([((s,), s) for s in histories], build_qa_system_prompt(lang_code), h_out, KEY_FIELDS)
            for s in histories[:2]:
                print(f"[history/{lang_code}/{args.model}] {s[:80]!r} -> {h_result[(s,)][:80]!r}")
        print()


def cmd_assemble(args):
    raw_corpus, raw_queries, raw_qrels = load_canard()
    corpus_texts = collect_corpus_texts(raw_corpus)
    query_texts = collect_query_texts(raw_queries)
    histories = collect_histories(raw_queries)

    final_corpus, final_queries = {}, {}
    for lang_code in args.langs:
        for model_key in MODELS:
            c_path = corpus_checkpoint_path(lang_code, model_key)
            q_path = queries_checkpoint_path(lang_code, model_key)
            h_path = history_checkpoint_path(lang_code, model_key)
            if not (c_path.exists() and q_path.exists() and h_path.exists()):
                print(f"skipping {lang_code}/{model_key}: missing checkpoint(s) among {c_path}, {q_path}, {h_path}")
                continue

            c_lookup_raw = load_checkpoint(c_path, KEY_FIELDS)
            q_lookup_raw = load_checkpoint(q_path, KEY_FIELDS)
            h_lookup_raw = load_checkpoint(h_path, KEY_FIELDS)
            missing_c = [s for s in corpus_texts if (s,) not in c_lookup_raw]
            missing_q = [s for s in query_texts if (s,) not in q_lookup_raw]
            missing_h = [s for s in histories if (s,) not in h_lookup_raw]
            if missing_c or missing_q or missing_h:
                print(
                    f"skipping {lang_code}/{model_key}: incomplete "
                    f"({len(corpus_texts) - len(missing_c)}/{len(corpus_texts)} corpus, "
                    f"{len(query_texts) - len(missing_q)}/{len(query_texts)} questions, "
                    f"{len(histories) - len(missing_h)}/{len(histories)} histories translated) "
                    f"-- run `translate` first"
                )
                continue

            c_lookup = {k[0]: v for k, v in c_lookup_raw.items()}
            q_lookup = {k[0]: v for k, v in q_lookup_raw.items()}
            h_lookup = {k[0]: v for k, v in h_lookup_raw.items()}

            dd_corpus = DatasetDict()
            for split_name in raw_corpus:
                rows = [{"id": row["id"], "text": c_lookup.get(row["text"], row["text"])} for row in raw_corpus[split_name]]
                dd_corpus[split_name] = Dataset.from_list(rows)
            final_corpus[(lang_code, model_key)] = dd_corpus

            dd_queries = DatasetDict()
            for split_name in raw_queries:
                rows = [
                    {
                        "id": row["id"],
                        "text": q_lookup.get(row["text"], row["text"]),
                        "history": h_lookup.get(row["history"], row["history"]) if row["history"].strip() else row["history"],
                    }
                    for row in raw_queries[split_name]
                ]
                dd_queries[split_name] = Dataset.from_list(rows)
            final_queries[(lang_code, model_key)] = dd_queries

            print(lang_code, model_key, "corpus", {s: len(dd_corpus[s]) for s in dd_corpus}, "queries", {s: len(dd_queries[s]) for s in dd_queries})

    if not (final_corpus or final_queries):
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check and final_corpus and final_queries:
        lang_code, model_key = next(iter(final_corpus))
        print("CORPUS:", final_corpus[(lang_code, model_key)]["train"][0]["text"][:250])
        qrow = final_queries[(lang_code, model_key)]["train"][1]
        print("QUESTION:", qrow["text"])
        print("HISTORY:", qrow["history"][:250])

    for (lang_code, model_key), dd in final_corpus.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-corpus"))
    for (lang_code, model_key), dd in final_queries.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-queries"))
    if final_corpus or final_queries:
        raw_qrels.save_to_disk(str(SAVE_DIR / "qrels"))
        print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/canard-mt"
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

    p_smoke = sub.add_parser("smoke-test", help="Translate a small corpus/query/history sample and print it")
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
