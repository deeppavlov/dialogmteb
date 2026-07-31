#!/usr/bin/env python3
"""Translate DeepPavlov/daily_dialog to Spanish using offline vLLM inference.

DailyDialog rows are NOT one row per dialogue -- each row's `dialog` is a growing
PREFIX of the same underlying conversation (row 1 = utterances[0:1], row 2 =
utterances[0:2], row 3 = utterances[0:3], ...), with `act_label`/`act_label_text`
(dialogue-act class, e.g. 'directive') and `emotion_label`/`emotion_label_text`
(emotion class, e.g. 'no emotion') describing the last utterance in that prefix.
There's no explicit conversation-id field to group by (unlike similar patterns in
other DeepPavlov datasets), so instead of detecting groups this script just pools
every utterance string globally and deduplicates by exact content: 559,933 naive
occurrences (utterances summed across every row's `dialog` list) collapse to 83,959
unique strings (~6.7x) -- verified against the live data.

`act_label`/`act_label_text`/`emotion_label`/`emotion_label_text` are classification
targets, not translated.

Usage (run one model per process -- see translate_common.py for why):

    python translate_daily_dialog.py translate --model gemma
    python translate_daily_dialog.py smoke-test --model gemma
    python translate_daily_dialog.py assemble    # no vLLM needed, reads whatever models were run

Checkpointed to translations/daily_dialog/*.jsonl -- safe to interrupt/resume.
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

SPLITS = ["train", "validation", "test"]
OUT_DIR = SCRIPT_DIR / "translations" / "daily_dialog"
SAVE_DIR = SCRIPT_DIR / "translations" / "daily_dialog_final"

EXAMPLES = {
    "es": [
        ("Say , Jim , how about going for a few beers after dinner ? ", "Oye , Jim , que tal si vamos por unas cervezas despues de cenar ? "),
        (
            " You know that is tempting but is really not good for our fitness . ",
            " Sabes que es tentador pero no es nada bueno para nuestra condicion fisica . ",
        ),
    ],
    "fr": [
        ("Say , Jim , how about going for a few beers after dinner ? ", "Dis , Jim , si on allait boire quelques bieres apres le diner ? "),
        (
            " You know that is tempting but is really not good for our fitness . ",
            " Tu sais que c'est tentant mais ce n'est vraiment pas bon pour notre forme physique . ",
        ),
    ],
}

SYSTEM_PROMPT = (
    "You are a professional translator localizing everyday casual conversation "
    "utterances (DailyDialog dataset). Translate the utterance from English into "
    "{lang_name}, keeping the same meaning, casual register, and spacing/punctuation "
    "style of the source (which has spaces before punctuation marks, e.g. 'dinner ? ' "
    "-- preserve that pattern rather than 'fixing' it) natural to a native {lang_name} "
    "speaker. Do not add, remove, or explain anything. "
    "Reply with ONLY the translated utterance: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in EXAMPLES.get(lang_code, [])]
    return "\n\n".join(lines)


def build_system_prompt(lang_code: str) -> str:
    return SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(lang_code))


def load_daily_dialog():
    raw = load_dataset("DeepPavlov/daily_dialog")
    for split in raw:
        print(f"{split}: {len(raw[split])} rows")
    return raw


def collect_texts(raw) -> list[str]:
    texts = set()
    for split in raw:
        for row in raw[split]:
            texts.update(row["dialog"])
    texts.discard("")
    return sorted(texts)


def checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"texts_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_daily_dialog()
    texts = collect_texts(raw)
    print(f"{len(texts)} unique utterances to translate")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in texts], build_system_prompt(lang_code),
            checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_daily_dialog()
    sample = raw["train"].select(range(3))
    texts = collect_texts({"train": sample})

    for lang_code in args.langs:
        out_path = OUT_DIR / f"smoketest_{lang_code}_{args.model}.jsonl"
        result = translator.translate_units([((s,), s) for s in texts], build_system_prompt(lang_code), out_path, KEY_FIELDS)
        for row in sample:
            print(f"[{lang_code}/{args.model}] dialog[-1]: {row['dialog'][-1]!r} -> {result[(row['dialog'][-1],)]!r}")
        print()


def cmd_assemble(args):
    raw = load_daily_dialog()
    texts = collect_texts(raw)

    final = {}
    for lang_code in args.langs:
        for model_key in MODELS:
            path = checkpoint_path(lang_code, model_key)
            if not path.exists():
                print(f"skipping {lang_code}/{model_key}: no checkpoint at {path}")
                continue
            lookup_raw = load_checkpoint(path, KEY_FIELDS)
            missing = [s for s in texts if (s,) not in lookup_raw]
            if missing:
                print(
                    f"skipping {lang_code}/{model_key}: incomplete "
                    f"({len(texts) - len(missing)}/{len(texts)} translated) -- run `translate` first"
                )
                continue
            lookup = {k[0]: v for k, v in lookup_raw.items()}

            dd = DatasetDict()
            for split_name in raw:
                rows = [
                    {
                        "dialog": [lookup.get(u, u) for u in row["dialog"]],
                        "act_label": row["act_label"],
                        "act_label_text": row["act_label_text"],
                        "emotion_label": row["emotion_label"],
                        "emotion_label_text": row["emotion_label_text"],
                    }
                    for row in raw[split_name]
                ]
                dd[split_name] = Dataset.from_list(rows)
            final[(lang_code, model_key)] = dd
            print(lang_code, model_key, {s: len(dd[s]) for s in dd})

    if not final:
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check:
        lang_code, model_key = next(iter(final))
        dd = final[(lang_code, model_key)]
        idxs = random.sample(range(len(dd["test"])), min(3, len(dd["test"])))
        for i in idxs:
            row = dd["test"][i]
            print("DIALOG:", row["dialog"])
            print("ACT:", row["act_label_text"], "EMOTION:", row["emotion_label_text"])
            print()

    for (lang_code, model_key), dd in final.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}"))
    print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/daily-dialog-mt"
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

    p_smoke = sub.add_parser("smoke-test", help="Translate a small sample and print it")
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
