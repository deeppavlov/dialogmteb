#!/usr/bin/env python3
"""Translate DeepPavlov/MultiWOZ-2.1 (`default` config only) to Spanish using offline
vLLM inference.

MultiWOZ-2.1 has domain-specific configs (`attraction`, `hospital`, `hotel`,
`restaurant`, `taxi`, `train`) in addition to `default`, but only `default` is
translated here, as requested.

Same structure as DeepPavlov/XRISAWOZ: each row is one user turn (`text`) plus
`history` (accumulated prior turns, `{content, role}`, verified to embed synthesized
assistant-turn text with no corresponding row of its own -- not reconstructable from
sibling rows), 30 `{domain}-{slot}` dialogue-state columns (e.g. `hotel-pricerange`,
mostly the literal string `'none'`), `dialogue_id`, and `topic` (the domain, e.g.
'hotel'). So this pools every row's `text` plus every entry of every row's `history`
globally and translates that deduplicated pool directly: 561,158 naive occurrences
collapse to 120,342 unique strings (~4.7x) -- verified against the live data.

The 30 `{domain}-{slot}` columns and `topic` are dialogue-state-tracking / domain
labels for benchmark evaluation, not free text -- left untouched.

Usage (run one model per process -- see translate_common.py for why):

    python translate_multiwoz.py translate --model gemma
    python translate_multiwoz.py smoke-test --model gemma
    python translate_multiwoz.py assemble    # no vLLM needed, reads whatever models were run

Checkpointed to translations/multiwoz/*.jsonl -- safe to interrupt/resume.
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

CONFIG = "default"
SPLITS = ["train", "dev", "test"]
OUT_DIR = SCRIPT_DIR / "translations" / "multiwoz"
SAVE_DIR = SCRIPT_DIR / "translations" / "multiwoz_final"

EXAMPLES = {
    "es": [
        (
            "am looking for a place to to stay that has cheap price range it should be in a type of hotel",
            "estoy buscando un lugar para hospedarme que tenga un rango de precio economico y que sea del tipo hotel",
        ),
        (
            "okay , do you have a specific area you want to stay in ?",
            "de acuerdo , tiene alguna zona especifica en la que quiera hospedarse ?",
        ),
    ],
    "fr": [
        (
            "am looking for a place to to stay that has cheap price range it should be in a type of hotel",
            "je cherche un endroit ou loger avec des prix bon marche et qui soit de type hotel",
        ),
        (
            "okay , do you have a specific area you want to stay in ?",
            "d'accord , avez-vous une zone particuliere ou vous souhaitez loger ?",
        ),
    ],
}

SYSTEM_PROMPT = (
    "You are a professional translator localizing turns of a task-oriented dialogue "
    "(a user booking/asking about hotels, restaurants, taxis, trains, attractions, or "
    "a hospital department; an assistant replying with information or booking "
    "confirmations). Translate the text from English into {lang_name}, keeping the "
    "same meaning and register, and the lowercase, loosely-punctuated style of the "
    "source (space before punctuation, e.g. 'hotel ?' -- preserve that rather than "
    "'fixing' it) natural to a native {lang_name} speaker. Keep proper nouns, "
    "reference/confirmation codes, and numbers exactly as written. Do not add, "
    "remove, or explain anything. "
    "Reply with ONLY the translated text: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in EXAMPLES.get(lang_code, [])]
    return "\n\n".join(lines)


def build_system_prompt(lang_code: str) -> str:
    return SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(lang_code))


def load_multiwoz():
    raw = load_dataset("DeepPavlov/MultiWOZ-2.1", CONFIG)
    for split in raw:
        print(f"{split}: {len(raw[split])} rows")
    return raw


def collect_texts(raw) -> list[str]:
    texts = set()
    for split in raw:
        for row in raw[split]:
            texts.add(row["text"])
            for h in row["history"]:
                texts.add(h["content"])
    texts.discard("")
    return sorted(texts)


def checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"texts_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_multiwoz()
    texts = collect_texts(raw)
    print(f"{len(texts)} unique strings to translate")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in texts], build_system_prompt(lang_code),
            checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_multiwoz()
    sample = raw["train"].select(range(5))
    texts = collect_texts({"train": sample})

    for lang_code in args.langs:
        out_path = OUT_DIR / f"smoketest_{lang_code}_{args.model}.jsonl"
        result = translator.translate_units([((s,), s) for s in texts], build_system_prompt(lang_code), out_path, KEY_FIELDS)
        for row in sample:
            print(f"[{lang_code}/{args.model}] {row['text']!r} -> {result[(row['text'],)]!r}")
        print()


def cmd_assemble(args):
    raw = load_multiwoz()
    texts = collect_texts(raw)
    non_text_cols = [c for c in raw["train"].features if c not in ("text", "history")]

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
                rows = []
                for row in raw[split_name]:
                    new_row = {c: row[c] for c in non_text_cols}
                    new_row["text"] = lookup.get(row["text"], row["text"])
                    new_row["history"] = [{"content": lookup.get(h["content"], h["content"]), "role": h["role"]} for h in row["history"]]
                    rows.append(new_row)
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
            print("TEXT:", row["text"])
            for h in row["history"][-2:]:
                print(f"  [{h['role']}]", h["content"][:150])
            print("TOPIC:", row["topic"])
            print()

    for (lang_code, model_key), dd in final.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}"))
    print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/multiwoz-2.1-mt"
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
