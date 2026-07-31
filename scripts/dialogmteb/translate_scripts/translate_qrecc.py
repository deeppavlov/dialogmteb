#!/usr/bin/env python3
"""Translate DeepPavlov/qrecc to Spanish (and French) using offline vLLM inference.

QReCC (Question Rewriting in Conversational Context) has `train` (63,501 rows) and
`test` (16,451 rows). Each row is one conversational turn: `question` (the raw,
possibly context-dependent user question), `rewrite` (the same question resolved into
a self-contained form), `answer` (a short factual answer span), `answer_url` (source
URL -- not translated), `conversation_no`/`turn_no` (ints -- not translated),
`conversation_source` (categorical tag, 'quac'/'nq' -- not translated), and `context`
(the accumulated prior turns of that conversation, as `{content, role}` pairs).

`context` at turn N is normally just the `rewrite`/`answer` pairs of turns 1..N-1 of the
same conversation ('user' role = rewrite, 'assistant' role = answer) -- but turn numbers
have gaps in this dataset release (e.g. conversation 7 jumps from turn 7 to turn 9), so
~17% of rows can't have their `context` reconstructed purely from sibling rows' fields
(the missing turn's text only survives inside later rows' `context`). So instead of
reconstructing `context` from other rows, this script pools ALL text globally --
`question` + `rewrite` + `context` user-turns into one "questions" pool, `answer` +
`context` assistant-turns into one "answers" pool -- deduplicates (702,699 naive
occurrences collapse to 208,464 unique strings, ~3.4x), translates each pool once, and
reconstructs every field (including `context`) by lookup. This is correct regardless of
the turn-number gaps, since it never assumes `context` is derivable from sibling rows.

Usage (run one model per process -- see translate_common.py for why):

    python translate_qrecc.py translate --model gemma
    python translate_qrecc.py translate --model qwen
    python translate_qrecc.py smoke-test --model gemma
    python translate_qrecc.py assemble    # no vLLM needed, reads both checkpoints

Checkpointed to translations/qrecc/*.jsonl -- safe to interrupt/resume.
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
OUT_DIR = SCRIPT_DIR / "translations" / "qrecc"
SAVE_DIR = SCRIPT_DIR / "translations" / "qrecc_final"

QUESTION_EXAMPLES = {
    "es": [
        ("What can you tell me about Gary Cherone?", "Que me puedes decir sobre Gary Cherone?"),
        ("Did Gary Cherone sing well?", "Canto bien Gary Cherone?"),
        ("What did Cherone do after Van Halen?", "Que hizo Cherone despues de Van Halen?"),
    ],
    "fr": [
        ("What can you tell me about Gary Cherone?", "Que pouvez-vous me dire sur Gary Cherone ?"),
        ("Did Gary Cherone sing well?", "Est-ce que Gary Cherone chantait bien ?"),
        ("What did Cherone do after Van Halen?", "Qu'a fait Cherone apres Van Halen ?"),
    ],
}

QUESTION_SYSTEM_PROMPT = (
    "You are a professional translator localizing questions from a conversational "
    "search dataset (a user asking follow-up questions about a topic; some questions "
    "use pronouns/references resolved from earlier turns, others are already "
    "self-contained rewrites of an earlier question). Translate the question from "
    "English into {lang_name}, keeping the same meaning and casual conversational "
    "register, natural to a native {lang_name} speaker. Keep proper nouns (people, "
    "places, titles of works) as their correct {lang_name} form where one exists. "
    "Do not add, remove, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)

ANSWER_EXAMPLES = {
    "es": [
        (
            "Gary Francis Caine Cherone is an American rock singer and songwriter, known for his work as the lead vocalist of Extreme and for his short stint for Van Halen.",
            "Gary Francis Caine Cherone es un cantante y compositor de rock estadounidense, conocido por su trabajo como vocalista principal de Extreme y por su breve etapa en Van Halen.",
        ),
    ],
    "fr": [
        (
            "Gary Francis Caine Cherone is an American rock singer and songwriter, known for his work as the lead vocalist of Extreme and for his short stint for Van Halen.",
            "Gary Francis Caine Cherone est un chanteur et auteur-compositeur de rock americain, connu pour son travail en tant que chanteur principal d'Extreme et pour son bref passage chez Van Halen.",
        ),
    ],
}

ANSWER_SYSTEM_PROMPT = (
    "You are a professional translator localizing short factual answer passages "
    "extracted from web sources (Wikipedia and similar) for a conversational search "
    "dataset. Translate the passage from English into {lang_name}, producing accurate, "
    "natural {lang_name} prose. Keep proper nouns (people, places, titles of works) as "
    "their correct {lang_name} form where a standard one exists, otherwise leave them "
    "as written. Preserve facts, numbers, and dates exactly. Do not add, remove, "
    "summarize, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(examples: dict, lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in examples.get(lang_code, [])]
    return "\n\n".join(lines)


def build_question_system_prompt(lang_code: str) -> str:
    return QUESTION_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(QUESTION_EXAMPLES, lang_code))


def build_answer_system_prompt(lang_code: str) -> str:
    return ANSWER_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(ANSWER_EXAMPLES, lang_code))


def load_qrecc():
    raw = load_dataset("DeepPavlov/qrecc")
    for split in raw:
        print(f"{split}: {len(raw[split])} rows")
    return raw


def collect_pools(raw) -> tuple[list[str], list[str]]:
    questions, answers = set(), set()
    for split in raw:
        for row in raw[split]:
            questions.add(row["question"])
            questions.add(row["rewrite"])
            if row["answer"].strip():
                answers.add(row["answer"])
            for turn in row["context"]:
                (questions if turn["role"] == "user" else answers).add(turn["content"])
    questions.discard("")
    answers.discard("")
    return sorted(questions), sorted(answers)


def questions_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"questions_{lang_code}_{model_key}.jsonl"


def answers_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"answers_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_qrecc()
    questions, answers = collect_pools(raw)
    print(f"{len(questions)} unique questions/rewrites, {len(answers)} unique answers to translate")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in questions], build_question_system_prompt(lang_code),
            questions_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )
        translator.translate_units(
            [((s,), s) for s in answers], build_answer_system_prompt(lang_code),
            answers_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_qrecc()
    sample = {"train": raw["train"].select(range(6))}
    questions, answers = collect_pools(sample)

    for lang_code in args.langs:
        q_out = OUT_DIR / f"smoketest_questions_{lang_code}_{args.model}.jsonl"
        q_result = translator.translate_units([((s,), s) for s in questions], build_question_system_prompt(lang_code), q_out, KEY_FIELDS)
        for s in questions:
            print(f"[question/{lang_code}/{args.model}] {s!r} -> {q_result[(s,)]!r}")

        a_out = OUT_DIR / f"smoketest_answers_{lang_code}_{args.model}.jsonl"
        a_result = translator.translate_units([((s,), s) for s in answers], build_answer_system_prompt(lang_code), a_out, KEY_FIELDS)
        for s in answers:
            print(f"[answer/{lang_code}/{args.model}] {s[:80]!r} -> {a_result[(s,)][:80]!r}")
        print()


def cmd_assemble(args):
    raw = load_qrecc()
    questions, answers = collect_pools(raw)

    final = {}
    for lang_code in args.langs:
        for model_key in MODELS:
            q_path = questions_checkpoint_path(lang_code, model_key)
            a_path = answers_checkpoint_path(lang_code, model_key)
            if not q_path.exists() or not a_path.exists():
                print(f"skipping {lang_code}/{model_key}: missing checkpoint(s) at {q_path} / {a_path}")
                continue

            q_lookup_raw = load_checkpoint(q_path, KEY_FIELDS)
            a_lookup_raw = load_checkpoint(a_path, KEY_FIELDS)
            missing_q = [s for s in questions if (s,) not in q_lookup_raw]
            missing_a = [s for s in answers if (s,) not in a_lookup_raw]
            if missing_q or missing_a:
                print(
                    f"skipping {lang_code}/{model_key}: incomplete "
                    f"({len(questions) - len(missing_q)}/{len(questions)} questions, "
                    f"{len(answers) - len(missing_a)}/{len(answers)} answers translated) "
                    f"-- run `translate` first"
                )
                continue

            q_lookup = {k[0]: v for k, v in q_lookup_raw.items()}
            a_lookup = {k[0]: v for k, v in a_lookup_raw.items()}

            dd = DatasetDict()
            for split_name in raw:
                rows = []
                for row in raw[split_name]:
                    translated_answer = a_lookup[row["answer"]] if row["answer"].strip() else row["answer"]
                    context = [
                        {"content": (q_lookup if t["role"] == "user" else a_lookup).get(t["content"], t["content"]), "role": t["role"]}
                        for t in row["context"]
                    ]
                    rows.append({
                        "context": context,
                        "question": q_lookup[row["question"]],
                        "rewrite": q_lookup[row["rewrite"]],
                        "answer": translated_answer,
                        "answer_url": row["answer_url"],
                        "conversation_no": row["conversation_no"],
                        "turn_no": row["turn_no"],
                        "conversation_source": row["conversation_source"],
                    })
                dd[split_name] = Dataset.from_list(rows)
            final[(lang_code, model_key)] = dd
            print(lang_code, model_key, {s: len(dd[s]) for s in dd})

    if not final:
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check:
        lang_code, model_key = next(iter(final))
        dd = final[(lang_code, model_key)]
        split_name = "train" if "train" in dd else next(iter(dd))
        idxs = random.sample(range(len(dd[split_name])), min(3, len(dd[split_name])))
        for i in idxs:
            row = dd[split_name][i]
            print("QUESTION:", row["question"])
            print("REWRITE:", row["rewrite"])
            print("ANSWER:", row["answer"][:200])
            print("CONTEXT TURNS:", len(row["context"]))
            print()

    for (lang_code, model_key), dd in final.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}"))
    print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/qrecc-mt"
    # for (lang_code, model_key), dd in final.items():
    #     dd.push_to_hub(repo_id, config_name=f"{lang_code}-{model_key}")


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

    p_smoke = sub.add_parser("smoke-test", help="Translate a small question/answer sample and print it")
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
