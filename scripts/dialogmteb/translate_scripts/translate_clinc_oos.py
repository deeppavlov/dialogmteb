#!/usr/bin/env python3
"""Translate DeepPavlov/clinc_oos to Spanish (and French) using offline vLLM inference.

CLINC-OOS (CLINC150) is a flat intent-classification dataset (short user utterances,
e.g. "what are the steps for setting up direct deposit for my paycheck", labeled with
one of 150 intents plus an out-of-scope class) with three configs -- `imbalanced`
(10,625 train), `plus` (15,250 train), `small` (7,600 train) -- that differ only in
how `train` is sampled; `validation` (3,100) and `test` (5,500) are byte-identical
across all three configs (verified against the live data).

So this script deduplicates globally: `validation`/`test` are translated once
regardless of how many configs are processed, and `train` is pooled across all three
configs' overlapping-but-not-identical text (verified NOT simple subsets of each
other) before translating. 42,075 naive occurrences (train x3 + validation + test)
collapse to 23,846 unique strings (~43% savings).

`label` (int) and `label_text` (the intent name, e.g. 'direct_deposit') are copied
through unchanged -- these are classification target identifiers, not natural-language
content to translate.

Usage (run one model per process -- see translate_common.py for why):

    python translate_clinc_oos.py translate --model gemma
    python translate_clinc_oos.py smoke-test --model gemma
    python translate_clinc_oos.py assemble    # no vLLM needed, reads whatever models were run

Checkpointed to translations/clinc_oos/*.jsonl -- safe to interrupt/resume.
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

CONFIGS = ["imbalanced", "plus", "small"]
OUT_DIR = SCRIPT_DIR / "translations" / "clinc_oos"
SAVE_DIR = SCRIPT_DIR / "translations" / "clinc_oos_final"

EXAMPLES = {
    "es": [
        (
            "what are the steps for setting up direct deposit for my paycheck",
            "cuales son los pasos para configurar el deposito directo de mi cheque de pago",
        ),
        (
            "what expression would i use to say i love you if i were an italian",
            "que expresion usaria para decir te quiero si fuera italiano",
        ),
    ],
    "fr": [
        (
            "what are the steps for setting up direct deposit for my paycheck",
            "quelles sont les etapes pour configurer le depot direct de mon salaire",
        ),
        (
            "what expression would i use to say i love you if i were an italian",
            "quelle expression utiliserais-je pour dire je t'aime si j'etais italien",
        ),
    ],
}

SYSTEM_PROMPT = (
    "You are a professional translator localizing short user utterances for a virtual "
    "assistant's intent-classification benchmark (banking, travel, smalltalk, and "
    "many other domains). Translate the utterance from English into {lang_name}, "
    "keeping the same meaning, casual register, and lowercase style (the source is "
    "lowercase and lightly punctuated) natural to a native {lang_name} speaker. Do "
    "not add, remove, or explain anything. "
    "Reply with ONLY the translated utterance: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in EXAMPLES.get(lang_code, [])]
    return "\n\n".join(lines)


def build_system_prompt(lang_code: str) -> str:
    return SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(lang_code))


def load_clinc_oos():
    raw = {cfg: load_dataset("DeepPavlov/clinc_oos", cfg) for cfg in CONFIGS}
    for cfg in CONFIGS:
        for split in raw[cfg]:
            print(f"{cfg}/{split}: {len(raw[cfg][split])} rows")

    val_texts = {cfg: raw[cfg]["validation"]["text"] for cfg in CONFIGS}
    test_texts = {cfg: raw[cfg]["test"]["text"] for cfg in CONFIGS}
    assert all(v == val_texts[CONFIGS[0]] for v in val_texts.values()), "expected validation to be identical across configs"
    assert all(t == test_texts[CONFIGS[0]] for t in test_texts.values()), "expected test to be identical across configs"
    print("verified: validation/test are byte-identical across all three configs")

    return raw


def collect_texts(raw) -> list[str]:
    texts = set()
    for cfg in CONFIGS:
        for split in raw[cfg]:
            texts.update(raw[cfg][split]["text"])
    texts.discard("")
    return sorted(texts)


def checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"texts_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_clinc_oos()
    texts = collect_texts(raw)
    print(f"{len(texts)} unique utterances to translate")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in texts], build_system_prompt(lang_code),
            checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_clinc_oos()
    sample = raw["small"]["test"]["text"][:5]

    for lang_code in args.langs:
        out_path = OUT_DIR / f"smoketest_{lang_code}_{args.model}.jsonl"
        result = translator.translate_units([((s,), s) for s in sample], build_system_prompt(lang_code), out_path, KEY_FIELDS)
        for s in sample:
            print(f"[{lang_code}/{args.model}] {s!r} -> {result[(s,)]!r}")
        print()


def cmd_assemble(args):
    raw = load_clinc_oos()
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

            for cfg in CONFIGS:
                dd = DatasetDict()
                for split_name in raw[cfg]:
                    rows = [
                        {"text": lookup.get(row["text"], row["text"]), "label": row["label"], "label_text": row["label_text"]}
                        for row in raw[cfg][split_name]
                    ]
                    dd[split_name] = Dataset.from_list(rows)
                final[(lang_code, model_key, cfg)] = dd
                print(lang_code, model_key, cfg, {s: len(dd[s]) for s in dd})

    if not final:
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check:
        (lang_code, model_key, cfg), dd = next(iter(final.items()))
        idxs = random.sample(range(len(dd["test"])), min(5, len(dd["test"])))
        for i in idxs:
            row = dd["test"][i]
            print(f"[{cfg}] {row['label_text']}: {row['text']}")

    for (lang_code, model_key, cfg), dd in final.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-{cfg}"))
    print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/clinc-oos-mt"
    # for (lang_code, model_key, cfg), dd in final.items():
    #     dd.push_to_hub(repo_id, config_name=f"{lang_code}-{model_key}-{cfg}")


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
