#!/usr/bin/env python3
"""Translate DeepPavlov/XRISAWOZ to Spanish using offline vLLM inference.

XRISAWOZ is a cross-lingual task-oriented dialogue dataset (attractions, hotels,
restaurants, trains, hospitals, ...). It already ships NATIVE `fr`/`hi`/`ko`/`zh`
configs (professionally translated, not machine-translated) -- there is no Spanish
config, which is the actual gap this script fills. Default scope is Spanish only;
translating to French/Hindi/Korean/Chinese here would just be a redundant, lower-
quality duplicate of what already exists on the Hub.

Structure (all verified against the live data): the full `en` config (4,643 test +
4,058 valid + 659 train rows, ~145 `inform-*`/`request-*` dialogue-state columns) is
the source of truth -- the 8 domain-specific configs (`en_attraction`, `en_car`,
`en_class`, `en_hospital`, `en_movie`, `en_pc`, `en_train`, `en_transport`) are row+
column SUBSETS of it (confirmed: every domain config's dialogue ids are a subset of
`en`'s, and each keeps only the `inform-*` columns relevant to its own domain, no
`request-*` columns at all). So only `en` needs to be translated; every domain config
is derived afterward by looking up its rows' already-translated `text`/`history` from
`en` and keeping its own non-text columns as-is.

`history` is NOT simply derivable from sibling rows' `text` (unlike similar fields in
other DeepPavlov datasets) -- it embeds synthesized assistant-turn text that has no
corresponding row of its own in this dataset. So this script pools ALL text globally
(every row's own `text` plus every entry in every row's `history`) and translates that
deduplicated pool directly, rather than reconstructing `history` from other rows:
74,388 naive occurrences collapse to 13,828 unique strings (~5.4x).

The ~145 `inform-*` (string, e.g. slot values like a city name) and `request-*`
(bool) columns are dialogue-state-tracking annotations for benchmark evaluation, not
free text -- left untouched, like `label`/`label_text` in the flat classification
datasets. `domains`/`turn_domain`/`dialogue_id`/`turn_id` are also left untouched.

Usage (run one model per process -- see translate_common.py for why):

    python translate_xrisawoz.py translate --model gemma
    python translate_xrisawoz.py smoke-test --model gemma
    python translate_xrisawoz.py assemble    # no vLLM needed, reads whatever models were run

Checkpointed to translations/xrisawoz/*.jsonl -- safe to interrupt/resume.
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
    "fr": "French",  # already exists natively on the Hub -- see module docstring
}
DEFAULT_LANGS = ["es"]

MAIN_CONFIG = "en"
DOMAINS = ["attraction", "car", "class", "hospital", "movie", "pc", "train", "transport"]
SPLITS = ["train", "valid", "test"]

OUT_DIR = SCRIPT_DIR / "translations" / "xrisawoz"
SAVE_DIR = SCRIPT_DIR / "translations" / "xrisawoz_final"

EXAMPLES = {
    "es": [
        (
            "Hello! I'm from out-of-town and here to visit Suzhou. Can you recommend a fun place?",
            "Hola! Soy de fuera de la ciudad y estoy aqui para visitar Suzhou. Me puedes recomendar un lugar divertido?",
        ),
        (
            "The small bridges and winding waterways are a must-see in Suzhou. So I recommend Zhouzhuang Town.",
            "Los pequenos puentes y los canales sinuosos son de visita obligada en Suzhou. Asi que te recomiendo el pueblo de Zhouzhuang.",
        ),
    ],
    "fr": [
        (
            "Hello! I'm from out-of-town and here to visit Suzhou. Can you recommend a fun place?",
            "Bonjour ! Je viens d'une autre ville pour visiter Suzhou. Pouvez-vous me recommander un endroit sympa ?",
        ),
        (
            "The small bridges and winding waterways are a must-see in Suzhou. So I recommend Zhouzhuang Town.",
            "Les petits ponts et les canaux sinueux sont incontournables a Suzhou. Je vous recommande donc la ville de Zhouzhuang.",
        ),
    ],
}

SYSTEM_PROMPT = (
    "You are a professional translator localizing a task-oriented dialogue dataset "
    "(a user booking/asking about attractions, hotels, restaurants, trains, hospitals, "
    "computers, classes, and similar). Translate the text from English into "
    "{lang_name}, keeping the same meaning and register -- it may be a user request or "
    "an assistant reply providing information. Keep proper nouns (place names, "
    "product/brand names) as their correct {lang_name} form where one exists, "
    "otherwise leave them as written. Do not add, remove, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in EXAMPLES.get(lang_code, [])]
    return "\n\n".join(lines)


def build_system_prompt(lang_code: str) -> str:
    return SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(lang_code))


def load_main():
    raw = load_dataset("DeepPavlov/XRISAWOZ", MAIN_CONFIG)
    for split in raw:
        print(f"{MAIN_CONFIG}/{split}: {len(raw[split])} rows")
    return raw


def load_domain_configs():
    return {domain: load_dataset("DeepPavlov/XRISAWOZ", f"en_{domain}") for domain in DOMAINS}


def collect_texts(raw_main) -> list[str]:
    texts = set()
    for split in raw_main:
        for row in raw_main[split]:
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
    raw_main = load_main()
    texts = collect_texts(raw_main)
    print(f"{len(texts)} unique strings to translate")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in texts], build_system_prompt(lang_code),
            checkpoint_path(lang_code, args.model), KEY_FIELDS,
        )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_main = load_main()
    sample = raw_main["train"].select(range(5))
    texts = collect_texts({"train": sample})

    for lang_code in args.langs:
        out_path = OUT_DIR / f"smoketest_{lang_code}_{args.model}.jsonl"
        result = translator.translate_units([((s,), s) for s in texts], build_system_prompt(lang_code), out_path, KEY_FIELDS)
        for row in sample:
            print(f"[{lang_code}/{args.model}] {row['text']!r} -> {result[(row['text'],)]!r}")
        print()


def cmd_assemble(args):
    raw_main = load_main()
    domain_raw = load_domain_configs()
    texts = collect_texts(raw_main)

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

            def tr_text(t):
                return lookup.get(t, t) if t else t

            def tr_history(hist):
                return [{"content": lookup.get(h["content"], h["content"]), "role": h["role"]} for h in hist]

            # Build (dialogue_id, turn_id) -> (translated text, translated history) once.
            turn_lookup = {}
            for split_name in raw_main:
                for row in raw_main[split_name]:
                    turn_lookup[(row["dialogue_id"], row["turn_id"])] = (tr_text(row["text"]), tr_history(row["history"]))

            dd_main = DatasetDict()
            for split_name in raw_main:
                rows = []
                for row in raw_main[split_name]:
                    new_row = dict(row)
                    new_row["text"], new_row["history"] = turn_lookup[(row["dialogue_id"], row["turn_id"])]
                    rows.append(new_row)
                dd_main[split_name] = Dataset.from_list(rows)
            final[(lang_code, model_key, MAIN_CONFIG)] = dd_main
            print(lang_code, model_key, MAIN_CONFIG, {s: len(dd_main[s]) for s in dd_main})

            for domain in DOMAINS:
                dd_domain = DatasetDict()
                for split_name in domain_raw[domain]:
                    rows = []
                    for row in domain_raw[domain][split_name]:
                        new_row = dict(row)
                        new_row["text"], new_row["history"] = turn_lookup[(row["dialogue_id"], row["turn_id"])]
                        rows.append(new_row)
                    dd_domain[split_name] = Dataset.from_list(rows)
                final[(lang_code, model_key, f"{MAIN_CONFIG}_{domain}")] = dd_domain
                print(lang_code, model_key, f"{MAIN_CONFIG}_{domain}", {s: len(dd_domain[s]) for s in dd_domain})

    if not final:
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check:
        (lang_code, model_key, cfg), dd = next(iter(final.items()))
        idxs = random.sample(range(len(dd["test"])), min(3, len(dd["test"])))
        for i in idxs:
            row = dd["test"][i]
            print(f"[{cfg}] TEXT:", row["text"])
            for h in row["history"][-2:]:
                print(f"  [{h['role']}]", h["content"][:150])
            print()

    for (lang_code, model_key, cfg), dd in final.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-{cfg}"))
    print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/xrisawoz-mt"
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
