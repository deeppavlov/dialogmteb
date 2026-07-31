#!/usr/bin/env python3
"""Translate DeepPavlov/coqa_abg to Spanish (and French) using offline vLLM inference.

CoQA-Abg is conversational reading-comprehension QA with ambiguity/clarification
annotations, split `train`(6,468)/`test`(1,055)/`val`(1,092). Each row has: `id`,
`story` (a reading passage from gutenberg/wikipedia/mctest/race/cnn -- `source`, not
translated), `target_turn` ({question, answer, rationale, span_start, span_end,
turn_id}), `history_turns` (list of prior {question, answer, rationale, turn_id}),
`ambiguity` ('ambiguous'/'non_ambiguous', not translated), and `clarification_turn` /
`clarification_turn_2` (present only for ambiguous rows: a clarifying question plus up
to 4 candidate answers per interpretation).

Two structural notes baked into this script:

1. `story` repeats heavily -- the same passage is reused across every QA turn asked
   about it (verified: train's 6,468 rows share only 2,917 unique story texts, ~2.2x;
   similar for test/val). `history_turns`/`target_turn` text also repeats across a
   story's turns (a later row's `history_turns` duplicates an earlier row's
   `target_turn`, like DeepPavlov/qrecc's `context`) -- verified 50,208 naive turn-text
   occurrences collapse to 29,203 unique strings (~1.7x) in train alone. So this script
   pools `story` and turn-text (question/answer/rationale/clarification) separately,
   deduplicates each pool, translates once, and reconstructs every row by lookup.

2. `rationale` is a verbatim character-offset SPAN of `story`
   (`story[span_start:span_end] == rationale`, verified on a sample). Once `story` is
   independently translated, those character offsets no longer point at the
   corresponding text -- there's no way to recover a valid span without re-aligning
   translated text to a translated substring, which off-the-shelf translation doesn't
   guarantee. So `span_start`/`span_end` are set to `None` in the translated output
   rather than left as silently-wrong offsets into the translated `story`.

Usage (run one model per process -- see translate_common.py for why):

    python translate_coqa_abg.py translate --model gemma
    python translate_coqa_abg.py translate --model qwen
    python translate_coqa_abg.py smoke-test --model gemma
    python translate_coqa_abg.py assemble    # no vLLM needed, reads both checkpoints

Checkpointed to translations/coqa_abg/*.jsonl -- safe to interrupt/resume.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from datasets import Dataset, DatasetDict, Features, Value, load_dataset  # noqa: E402

from translate_common import MODELS, OfflineTranslator, load_checkpoint, parse_engine_kwargs  # noqa: E402

LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
}
DEFAULT_LANGS = ["es"]

SPLITS = ["train", "test", "val"]
OUT_DIR = SCRIPT_DIR / "translations" / "coqa_abg"
SAVE_DIR = SCRIPT_DIR / "translations" / "coqa_abg_final"

STORY_EXAMPLES = {
    "es": [
        (
            "The morning of that Wednesday of Corpus Christi, fateful to all concerned in this chronicle, dawned misty and grey, and the air was chilled by the wind that blew from the sea.",
            "La manana de aquel miercoles de Corpus Christi, fatidico para todos los involucrados en esta cronica, amanecio brumosa y gris, y el aire estaba helado por el viento que soplaba desde el mar.",
        ),
    ],
    "fr": [
        (
            "The morning of that Wednesday of Corpus Christi, fateful to all concerned in this chronicle, dawned misty and grey, and the air was chilled by the wind that blew from the sea.",
            "Le matin de ce mercredi de la Fete-Dieu, fatidique pour tous ceux concernes par cette chronique, se leva brumeux et gris, et l'air etait glace par le vent qui soufflait de la mer.",
        ),
    ],
}

STORY_SYSTEM_PROMPT = (
    "You are a professional translator localizing reading-comprehension passages "
    "(literary excerpts, encyclopedia articles, news articles, educational texts) for "
    "a conversational QA dataset. Translate the passage from English into {lang_name}, "
    "preserving its original register and style (literary, journalistic, encyclopedic, "
    "etc. as appropriate) and producing natural, accurate {lang_name} prose. Keep "
    "proper nouns (people, places, titles of works) as their correct {lang_name} form "
    "where a standard one exists, otherwise leave them as written. Preserve facts, "
    "numbers, and dates exactly. Do not add, remove, summarize, or explain anything -- "
    "translate the full passage. Reply with ONLY the translation: no quotes, no notes.\n\n"
    "Examples:\n{examples_block}"
)

TURN_EXAMPLES = {
    "es": [
        ("Who should be reinforced?", "A quien se deberia reforzar?"),
        ("the single man-at-arms patrolling the walls.", "el unico hombre de armas que patrullaba las murallas."),
        ("Do you mean Buda or Pest?", "Te refieres a Buda o a Pest?"),
    ],
    "fr": [
        ("Who should be reinforced?", "Qui faudrait-il renforcer ?"),
        ("the single man-at-arms patrolling the walls.", "l'unique homme d'armes qui patrouillait sur les murailles."),
        ("Do you mean Buda or Pest?", "Voulez-vous dire Buda ou Pest ?"),
    ],
}

TURN_SYSTEM_PROMPT = (
    "You are a professional translator localizing conversational reading-comprehension "
    "QA text: questions about a story, short extracted answers, supporting rationale "
    "snippets, and clarifying questions/answers for ambiguous turns. Translate the text "
    "from English into {lang_name}, keeping the same meaning and register -- these are "
    "often short phrases or sentence fragments (e.g. an extracted answer span), not "
    "always full grammatical sentences; translate them as such rather than padding them "
    "into complete sentences. Keep proper nouns as their correct {lang_name} form where "
    "one exists. Do not add, remove, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(examples: dict, lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in examples.get(lang_code, [])]
    return "\n\n".join(lines)


def build_story_system_prompt(lang_code: str) -> str:
    return STORY_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(STORY_EXAMPLES, lang_code))


def build_turn_system_prompt(lang_code: str) -> str:
    return TURN_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(TURN_EXAMPLES, lang_code))


def load_coqa_abg():
    raw = load_dataset("DeepPavlov/coqa_abg")
    for split in raw:
        print(f"{split}: {len(raw[split])} rows")
    return raw


def collect_turn_texts(row, out: set) -> None:
    tt = row["target_turn"]
    out.update(x for x in (tt["question"], tt["answer"], tt["rationale"]) if x)
    for h in row["history_turns"]:
        out.update(x for x in (h["question"], h["answer"], h["rationale"]) if x)
    for ct_key in ("clarification_turn", "clarification_turn_2"):
        ct = row[ct_key]
        if ct and ct["question"] is not None:
            out.add(ct["question"])
            for a in ct["answers"] or []:
                out.update(x for x in (a["clr_ans"], a["org_ans"], a["org_ans_2"], a["org_ans_3"]) if x)


def collect_pools(raw) -> tuple[list[str], list[str]]:
    stories, turns = set(), set()
    for split in raw:
        for row in raw[split]:
            stories.add(row["story"])
            collect_turn_texts(row, turns)
    stories.discard("")
    turns.discard("")
    return sorted(stories), sorted(turns)


def stories_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"stories_{lang_code}_{model_key}.jsonl"


def turns_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"turns_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)

# Explicit schema for the assembled rows. Needed because `clarification_turn_2` (and
# `org_ans_2`/`org_ans_3` within clarification answers) are ALWAYS None across the
# entire `train` split but populated in `test`/`val` (verified against the live data)
# -- letting Dataset.from_list infer types per-split independently makes `train`
# collapse those all-None columns to an untyped `null`, which then conflicts with
# `test`/`val`'s real struct type when the splits are combined into one DatasetDict.
_CLARIFICATION_ANSWER_FEATURES = {
    "clr_ans": Value("string"),
    "org_ans": Value("string"),
    "org_ans_2": Value("string"),
    "org_ans_3": Value("string"),
}
_CLARIFICATION_FEATURES = {
    # A list-of-structs feature must be a plain list literal `[{...}]` in this
    # `datasets` version -- `Sequence({...})` triggers an internal encoding bug.
    "answers": [_CLARIFICATION_ANSWER_FEATURES],
    "question": Value("string"),
}
ROW_FEATURES = Features({
    "id": Value("string"),
    "story": Value("string"),
    "target_turn": {
        "answer": Value("string"),
        "question": Value("string"),
        "rationale": Value("string"),
        "span_end": Value("int64"),
        "span_start": Value("int64"),
        "turn_id": Value("int64"),
    },
    "history_turns": [{
        "answer": Value("string"),
        "question": Value("string"),
        "rationale": Value("string"),
        "turn_id": Value("int64"),
    }],
    "ambiguity": Value("string"),
    "clarification_turn": _CLARIFICATION_FEATURES,
    "source": Value("string"),
    "clarification_turn_2": _CLARIFICATION_FEATURES,
})


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_coqa_abg()
    stories, turns = collect_pools(raw)
    print(f"{len(stories)} unique stories, {len(turns)} unique turn-text strings to translate")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in stories], build_story_system_prompt(lang_code),
            stories_checkpoint_path(lang_code, args.model), KEY_FIELDS, batch_size=1000,
        )
        translator.translate_units(
            [((s,), s) for s in turns], build_turn_system_prompt(lang_code),
            turns_checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_coqa_abg()
    sample = {"train": raw["train"].select(range(6))}
    stories, turns = collect_pools(sample)

    for lang_code in args.langs:
        s_out = OUT_DIR / f"smoketest_stories_{lang_code}_{args.model}.jsonl"
        s_result = translator.translate_units([((s,), s) for s in stories], build_story_system_prompt(lang_code), s_out, KEY_FIELDS)
        for s in stories:
            print(f"[story/{lang_code}/{args.model}] {s[:80]!r} -> {s_result[(s,)][:80]!r}")

        t_out = OUT_DIR / f"smoketest_turns_{lang_code}_{args.model}.jsonl"
        t_result = translator.translate_units([((s,), s) for s in turns], build_turn_system_prompt(lang_code), t_out, KEY_FIELDS)
        for s in turns[:10]:
            print(f"[turn/{lang_code}/{args.model}] {s!r} -> {t_result[(s,)]!r}")
        print()


def translate_clarification(ct, lookup: dict):
    if ct is None or ct["question"] is None:
        return ct
    return {
        "question": lookup.get(ct["question"], ct["question"]),
        "answers": [
            {field: (lookup.get(a[field], a[field]) if a[field] else a[field]) for field in ("clr_ans", "org_ans", "org_ans_2", "org_ans_3")}
            for a in (ct["answers"] or [])
        ],
    }


def cmd_assemble(args):
    raw = load_coqa_abg()
    stories, turns = collect_pools(raw)

    final = {}
    for lang_code in args.langs:
        for model_key in MODELS:
            s_path = stories_checkpoint_path(lang_code, model_key)
            t_path = turns_checkpoint_path(lang_code, model_key)
            if not s_path.exists() or not t_path.exists():
                print(f"skipping {lang_code}/{model_key}: missing checkpoint(s) at {s_path} / {t_path}")
                continue

            s_lookup_raw = load_checkpoint(s_path, KEY_FIELDS)
            t_lookup_raw = load_checkpoint(t_path, KEY_FIELDS)
            missing_s = [s for s in stories if (s,) not in s_lookup_raw]
            missing_t = [s for s in turns if (s,) not in t_lookup_raw]
            if missing_s or missing_t:
                print(
                    f"skipping {lang_code}/{model_key}: incomplete "
                    f"({len(stories) - len(missing_s)}/{len(stories)} stories, "
                    f"{len(turns) - len(missing_t)}/{len(turns)} turn-texts translated) "
                    f"-- run `translate` first"
                )
                continue

            s_lookup = {k[0]: v for k, v in s_lookup_raw.items()}
            t_lookup = {k[0]: v for k, v in t_lookup_raw.items()}

            def tr(x):
                return t_lookup.get(x, x) if x else x

            dd = DatasetDict()
            for split_name in raw:
                rows = []
                for row in raw[split_name]:
                    tt = row["target_turn"]
                    rows.append({
                        "id": row["id"],
                        "story": s_lookup.get(row["story"], row["story"]),
                        "target_turn": {
                            "answer": tr(tt["answer"]),
                            "question": tr(tt["question"]),
                            "rationale": tr(tt["rationale"]),
                            "span_end": None,
                            "span_start": None,
                            "turn_id": tt["turn_id"],
                        },
                        "history_turns": [
                            {"answer": tr(h["answer"]), "question": tr(h["question"]), "rationale": tr(h["rationale"]), "turn_id": h["turn_id"]}
                            for h in row["history_turns"]
                        ],
                        "ambiguity": row["ambiguity"],
                        "clarification_turn": translate_clarification(row["clarification_turn"], t_lookup),
                        "source": row["source"],
                        "clarification_turn_2": translate_clarification(row["clarification_turn_2"], t_lookup),
                    })
                dd[split_name] = Dataset.from_list(rows, features=ROW_FEATURES)
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
            print("STORY:", row["story"][:200])
            print("QUESTION:", row["target_turn"]["question"])
            print("ANSWER:", row["target_turn"]["answer"])
            print("AMBIGUITY:", row["ambiguity"])
            if row["clarification_turn"] and row["clarification_turn"]["question"]:
                print("CLARIFICATION:", row["clarification_turn"])
            print()

    for (lang_code, model_key), dd in final.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}"))
    print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/coqa-abg-mt"
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

    p_smoke = sub.add_parser("smoke-test", help="Translate a small story/turn sample and print it")
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
