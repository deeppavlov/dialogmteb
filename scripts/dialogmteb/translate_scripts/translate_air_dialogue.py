#!/usr/bin/env python3
"""Translate DeepPavlov/air_dialogue to Spanish using offline vLLM inference.

AirDialogue is a synthetic airline customer-service dialogue dataset: `text` (a full
dialogue per row, `{content, role}` turns) and `label` (a call-outcome class: 'book',
'no_flight', 'no_reservation', 'cancel', 'change' -- not translated).

Unlike several other DeepPavlov conversational datasets, rows here are NOT growing
windows of a shared conversation -- each row is its own complete, essentially unique
dialogue (321,458/321,459 unique in train alone, verified). But at the individual
UTTERANCE level there's real repetition (templated/self-play-generated customer
service scripts -- greetings, hold messages, confirmations recur across many
different dialogues): 5,221,350 naive turn occurrences collapse to 1,282,696 unique
utterances (~4.1x) -- verified against the live data.

**Scale warning**: 1.28M unique strings is by far the largest pool translated by any
of these scripts so far (~5x bigger than the previous largest, CORAL's corpus). This
will take a long time even on a fast local server. Use `--max-items N` to test on a
subset before committing to a full run.

Usage (run one model per process -- see translate_common.py for why):

    python translate_air_dialogue.py translate --model gemma --max-items 2000   # test first
    python translate_air_dialogue.py translate --model gemma                    # full run
    python translate_air_dialogue.py smoke-test --model gemma
    python translate_air_dialogue.py assemble    # no vLLM needed, reads whatever models were run

Checkpointed to translations/air_dialogue/*.jsonl -- safe to interrupt/resume.
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

SPLITS = ["train", "dev", "test"]
OUT_DIR = SCRIPT_DIR / "translations" / "air_dialogue"
SAVE_DIR = SCRIPT_DIR / "translations" / "air_dialogue_final"

EXAMPLES = {
    "es": [
        ("Hello.", "Hola."),
        ("Hello. How may I help you?", "Hola. En que puedo ayudarle?"),
        (
            "Can you help me to change my recent reservation because my trip dates are got postponed?",
            "Me puede ayudar a cambiar mi reserva reciente porque mis fechas de viaje se han pospuesto?",
        ),
        ("Please wait for a while.", "Por favor espere un momento."),
    ],
    "fr": [
        ("Hello.", "Bonjour."),
        ("Hello. How may I help you?", "Bonjour. Comment puis-je vous aider ?"),
        (
            "Can you help me to change my recent reservation because my trip dates are got postponed?",
            "Pouvez-vous m'aider a modifier ma reservation recente car mes dates de voyage ont ete reportees ?",
        ),
        ("Please wait for a while.", "Veuillez patienter un instant."),
    ],
}

SYSTEM_PROMPT = (
    "You are a professional translator localizing turns of an airline "
    "customer-service chat (a customer booking/changing/cancelling a flight "
    "reservation, an agent replying with information or confirmations). Translate the "
    "text from English into {lang_name}, keeping the same meaning and register "
    "(sometimes slightly informal or imperfect English -- keep that tone rather than "
    "smoothing it into perfect grammar), natural to a native {lang_name} speaker. Keep "
    "proper nouns, airport/airline codes, confirmation numbers, and dates exactly as "
    "written. Do not add, remove, or explain anything. "
    "Reply with ONLY the translated text: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in EXAMPLES.get(lang_code, [])]
    return "\n\n".join(lines)


def build_system_prompt(lang_code: str) -> str:
    return SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(lang_code))


def load_air_dialogue():
    raw = load_dataset("DeepPavlov/air_dialogue")
    for split in raw:
        print(f"{split}: {len(raw[split])} rows")
    return raw


def collect_texts(raw) -> list[str]:
    texts = set()
    for split in raw:
        for row in raw[split]:
            texts.update(t["content"] for t in row["text"])
    texts.discard("")
    return sorted(texts)


def checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"texts_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_air_dialogue()
    texts = collect_texts(raw)
    if args.max_items is not None:
        texts = texts[: args.max_items]
    print(f"{len(texts)} unique utterances to translate")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in texts], build_system_prompt(lang_code),
            checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw = load_air_dialogue()
    sample = raw["train"].select(range(3))
    texts = collect_texts({"train": sample})

    for lang_code in args.langs:
        out_path = OUT_DIR / f"smoketest_{lang_code}_{args.model}.jsonl"
        result = translator.translate_units([((s,), s) for s in texts], build_system_prompt(lang_code), out_path, KEY_FIELDS)
        for row in sample:
            for t in row["text"][:3]:
                print(f"[{lang_code}/{args.model}] [{t['role']}] {t['content']!r} -> {result[(t['content'],)]!r}")
            print()


def cmd_assemble(args):
    raw = load_air_dialogue()
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
                    f"({len(texts) - len(missing)}/{len(texts)} translated) -- run `translate` first "
                    f"(or with --max-items for a subset)"
                )
                continue
            lookup = {k[0]: v for k, v in lookup_raw.items()}

            dd = DatasetDict()
            for split_name in raw:
                rows = [
                    {
                        "text": [{"content": lookup.get(t["content"], t["content"]), "role": t["role"]} for t in row["text"]],
                        "label": row["label"],
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
            print("LABEL:", row["label"])
            for t in row["text"][:4]:
                print(f"  [{t['role']}]", t["content"][:150])
            print()

    for (lang_code, model_key), dd in final.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}"))
    print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/air-dialogue-mt"
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
    p_translate.add_argument("--max-items", type=int, default=None,
                              help="translate only the first N unique utterances (for testing before the full ~1.28M)")
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
