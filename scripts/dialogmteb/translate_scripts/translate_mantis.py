#!/usr/bin/env python3
"""Translate DeepPavlov/Mantis to Spanish using offline vLLM inference.

MANtIS is a multi-domain conversational-search dataset built from StackExchange Q&A
threads across 14 categories (apple, askubuntu, dba, diy, electronics, english, gaming,
gis, physics, scifi, security, stats, travel, worldbuilding). Each row is a `dialog`
(list of `{message, role}` turns), plus a `title`, `category`, and `dialog_time` (an
ISO timestamp -- not translated). Messages are real StackExchange posts: often long,
technical, and frequently containing embedded HTML (`<a href=...>`, `<img>`) and code
(`<code>`/`<pre>` blocks) that must be preserved verbatim -- only the surrounding
natural-language prose is translated.

Usage (run one model per process -- see translate_common.py for why):

    python translate_mantis.py translate --model gemma
    python translate_mantis.py translate --model qwen
    python translate_mantis.py translate --model gemma --splits dev test  # defer train
    python translate_mantis.py smoke-test --model gemma
    python translate_mantis.py assemble    # no vLLM needed, reads both checkpoints

Checkpointed to translations/mantis/*.jsonl -- safe to interrupt/resume.
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

# Only Spanish was originally requested for Mantis -- keep that as the default scope,
# but French is fully supported (examples below) if you pass --langs fr or --langs fr es.
DEFAULT_LANGS = ["es"]

ALL_SPLITS = ["dev", "test", "train"]
OUT_DIR = SCRIPT_DIR / "translations" / "mantis"
SAVE_DIR = SCRIPT_DIR / "translations" / "mantis_final"

EXAMPLES = {
    "es": [
        (
            "It's hardware related. Logic board needs to be replaced",
            "Es un problema de hardware. Hay que reemplazar la placa logica",
        ),
        (
            "Verify your MacBook sn: https://selfsolve.apple.com/agreementWarrantyDynamic.do",
            "Verifica el numero de serie de tu MacBook: https://selfsolve.apple.com/agreementWarrantyDynamic.do",
        ),
        (
            "Run <code>sudo apt-get update</code> first, then reboot.",
            "Ejecuta <code>sudo apt-get update</code> primero, y despues reinicia.",
        ),
    ],
    "fr": [
        (
            "It's hardware related. Logic board needs to be replaced",
            "C'est un probleme materiel. La carte mere doit etre remplacee",
        ),
        (
            "Verify your MacBook sn: https://selfsolve.apple.com/agreementWarrantyDynamic.do",
            "Verifiez le numero de serie de votre MacBook : https://selfsolve.apple.com/agreementWarrantyDynamic.do",
        ),
        (
            "Run <code>sudo apt-get update</code> first, then reboot.",
            "Executez <code>sudo apt-get update</code> d'abord, puis redemarrez.",
        ),
    ],
}

SYSTEM_PROMPT = (
    "You are a professional translator localizing StackExchange technical Q&A posts and "
    "chat messages (topics include Apple hardware, Ubuntu/Linux, databases, electronics, "
    "physics, security, travel, and more). Translate the message from English into "
    "{lang_name}. Keep the same meaning, tone, and register -- technical but "
    "conversational -- and produce something that reads naturally to a native "
    "{lang_name} speaker. "
    "CRITICAL: preserve ALL HTML tags (e.g. <a href=...>, <img>, <pre>, <code>) and "
    "their attributes EXACTLY as written -- do not translate URLs, do not alter tag "
    "syntax. Preserve the contents of <code>/<pre> blocks, shell commands, file paths, "
    "error messages, and variable/product names (e.g. 'MacBook', 'Ubuntu', 'SSD') "
    "UNTRANSLATED -- only translate the surrounding natural-language prose. "
    "Do not add, remove, or explain anything. "
    "Reply with ONLY the translated message: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {es}" for en, es in EXAMPLES.get(lang_code, [])]
    return "\n\n".join(lines)


def build_system_prompt(lang_code: str) -> str:
    return SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(lang_code))


def build_units(dataset) -> list[tuple[tuple, str]]:
    units = []
    for dialog_idx, row in enumerate(dataset):
        units.append(((dialog_idx, "title", 0), row["title"]))
        for turn_idx, turn in enumerate(row["dialog"]):
            units.append(((dialog_idx, "message", turn_idx), turn["message"]))
    return units


KEY_FIELDS = ("dialog_idx", "kind", "turn_idx")


def checkpoint_path(split_name: str, lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"{split_name}_{lang_code}_{model_key}.jsonl"


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_dataset("DeepPavlov/Mantis")
    for split in raw:
        n_turns = sum(len(r["dialog"]) for r in raw[split])
        print(f"{split}: {len(raw[split])} dialogs, {n_turns} turns")

    for split_name in args.splits:
        units = build_units(raw[split_name])
        for lang_code in args.langs:
            translator.translate_units(
                units,
                build_system_prompt(lang_code),
                checkpoint_path(split_name, lang_code, args.model),
                KEY_FIELDS,
            )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_dataset("DeepPavlov/Mantis")
    sample = raw["dev"].select(range(3))
    units = build_units(sample)
    for lang_code in args.langs:
        out_path = OUT_DIR / f"smoketest_{lang_code}_{args.model}.jsonl"
        result = translator.translate_units(
            units, build_system_prompt(lang_code), out_path, KEY_FIELDS,
        )
        for dialog_idx, row in enumerate(sample):
            print(f"[{lang_code}/{args.model}] TITLE: {row['title']!r} -> {result[(dialog_idx, 'title', 0)]!r}")
            for turn_idx, turn in enumerate(row["dialog"][:2]):
                print(f"  [{turn['role']}] {turn['message'][:100]!r} -> {result[(dialog_idx, 'message', turn_idx)][:100]!r}")
        print()


def cmd_assemble(args):
    raw = load_dataset("DeepPavlov/Mantis")

    final = {}
    for lang_code in args.langs:
        for model_key in MODELS:
            dd = DatasetDict()
            incomplete = False
            for split_name in args.splits:
                out_path = checkpoint_path(split_name, lang_code, model_key)
                if not out_path.exists():
                    print(f"skipping {lang_code}/{model_key}: no checkpoint at {out_path}")
                    incomplete = True
                    break
                units = build_units(raw[split_name])
                lookup = load_checkpoint(out_path, KEY_FIELDS)
                missing = [key for key, _ in units if key not in lookup]
                if missing:
                    print(
                        f"skipping {lang_code}/{model_key}/{split_name}: checkpoint incomplete "
                        f"({len(lookup)}/{len(units)} translated, {len(missing)} missing e.g. {missing[:3]}) "
                        f"-- run `translate` for this model/split first"
                    )
                    incomplete = True
                    break
                rows = []
                for dialog_idx, row in enumerate(raw[split_name]):
                    dialog = [
                        {"role": turn["role"], "message": lookup[(dialog_idx, "message", turn_idx)]}
                        for turn_idx, turn in enumerate(row["dialog"])
                    ]
                    rows.append({
                        "dialog": dialog,
                        "title": lookup[(dialog_idx, "title", 0)],
                        "category": row["category"],
                        "dialog_time": row["dialog_time"],
                    })
                dd[split_name] = Dataset.from_list(rows)
            if not incomplete:
                final[(lang_code, model_key)] = dd
                print(lang_code, model_key, {s: len(dd[s]) for s in dd})

    if not final:
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check:
        lang_code, model_key = next(iter(final))
        split_name = args.splits[0]
        dd = final[(lang_code, model_key)]
        idxs = random.sample(range(len(dd[split_name])), min(3, len(dd[split_name])))
        for i in idxs:
            row = dd[split_name][i]
            print("TITLE:", row["title"])
            for turn in row["dialog"][:2]:
                print(f"  [{turn['role']}]", turn["message"][:200])
            print()

    for (lang_code, model_key), dd in final.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}"))
        print("saved", SAVE_DIR / f"{lang_code}-{model_key}")

    # repo_id = "<your-username>/mantis-mt"
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
    p_translate.add_argument("--splits", nargs="+", choices=ALL_SPLITS, default=ALL_SPLITS)
    p_translate.set_defaults(func=cmd_translate)

    p_smoke = sub.add_parser("smoke-test", help="Translate a handful of dialogs and print them")
    add_common(p_smoke)
    add_engine_kwargs(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke_test)

    p_assemble = sub.add_parser("assemble", help="Build final datasets from existing checkpoints (no vLLM)")
    add_common(p_assemble, model_required=False)
    p_assemble.add_argument("--splits", nargs="+", choices=ALL_SPLITS, default=ALL_SPLITS)
    p_assemble.add_argument("--spot-check", action="store_true", default=True)
    p_assemble.add_argument("--no-spot-check", dest="spot_check", action="store_false")
    p_assemble.set_defaults(func=cmd_assemble)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
