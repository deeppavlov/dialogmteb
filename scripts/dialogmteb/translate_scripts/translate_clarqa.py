#!/usr/bin/env python3
"""Translate DeepPavlov/clarqa to Spanish (and French) using offline vLLM inference.

ClarQA is a clarification-question entity-disambiguation dataset with two configs,
`single_turn` and `multi_turn`. Every row has `entity1`/`entity2` fields shaped exactly
like `"{name} <S> {freebase type tags} <S> {description sentence}"` (verified: 100% of
~24k single_turn and ~44k multi_turn entity mentions split into exactly this 3-part
shape) plus a `label` ('0'/'1') and a `context`:

- `single_turn.context` is one short clarification question/topic phrase, e.g.
  "Directors of Two Trains Running" or "What is the price list of ePrompter?".
- `multi_turn.context` is always exactly 3 turns `[question, entity_name, followup]` --
  the middle turn is verified (100% of ~22k rows) to be an exact copy of that row's
  entity1 or entity2 name, i.e. it's a structural placeholder, not free text.

Only the description sentence is translated in entity1/entity2 -- the `name` and the
freebase type tags (e.g. `media_common.cataloged_instance award.winning_work`) are left
untouched (translating an ontology code would corrupt it, and keeping `name` verbatim
keeps it consistent with `context`'s middle turn, which is also left untouched for the
same reason). `context` questions are translated as natural language. Descriptions
repeat across rows (entities are referenced many times), so this script deduplicates
per config across all its splits before translating (~13% savings for single_turn,
~58% for multi_turn).

Usage (run one model per process -- see translate_common.py for why):

    python translate_clarqa.py translate --model gemma
    python translate_clarqa.py translate --model qwen
    python translate_clarqa.py smoke-test --model gemma
    python translate_clarqa.py assemble    # no vLLM needed, reads both checkpoints

Checkpointed to translations/clarqa/*.jsonl -- safe to interrupt/resume.
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

CONFIGS = ["single_turn", "multi_turn"]
SPLITS = ["train", "dev", "test"]
OUT_DIR = SCRIPT_DIR / "translations" / "clarqa"
SAVE_DIR = SCRIPT_DIR / "translations" / "clarqa_final"

ENTITY_EXAMPLES = {
    "es": [
        (
            "Two Trains Running is a 2006-2007 theater production of the play by August Wilson.",
            "Two Trains Running es una produccion teatral de 2006-2007 de la obra de August Wilson.",
        ),
    ],
    "fr": [
        (
            "Two Trains Running is a 2006-2007 theater production of the play by August Wilson.",
            "Two Trains Running est une production theatrale 2006-2007 de la piece d'August Wilson.",
        ),
    ],
}

ENTITY_SYSTEM_PROMPT = (
    "You are a professional translator localizing knowledge-base entity descriptions "
    "(one factual sentence per entity, Freebase/Wikidata-style). Translate the sentence "
    "from English into {lang_name}, producing accurate, encyclopedic {lang_name} prose. "
    "Keep proper nouns (people, places, titles of works) as their correct {lang_name} "
    "form where a standard one exists, otherwise leave them as written. Preserve facts, "
    "numbers, and dates exactly. Do not add, remove, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)

CONTEXT_EXAMPLES = {
    "es": [
        ("Which company is game developer for Rescue", "Que compania es la desarrolladora del juego para Rescue"),
        (
            "state the contributions that has been done for Dil Chahta Hai",
            "indica las contribuciones que se han hecho para Dil Chahta Hai",
        ),
        ("What is its style?", "Cual es su estilo?"),
    ],
    "fr": [
        ("Which company is game developer for Rescue", "Quelle entreprise est le developpeur du jeu pour Rescue"),
        (
            "state the contributions that has been done for Dil Chahta Hai",
            "indiquez les contributions qui ont ete faites pour Dil Chahta Hai",
        ),
        ("What is its style?", "Quel est son style ?"),
    ],
}

CONTEXT_SYSTEM_PROMPT = (
    "You are a professional translator localizing short clarification questions/topic "
    "phrases from an entity-disambiguation dataset (often informal or slightly "
    "ungrammatical, template-generated text -- e.g. 'state the contributions that has "
    "been done for X'). Translate the text from English into {lang_name}, keeping the "
    "same meaning, register, and any awkward phrasing rather than smoothing it into "
    "perfect grammar -- but produce something a native {lang_name} speaker would "
    "recognize as the same kind of question/phrase. Keep entity/proper names embedded "
    "in the text as their correct {lang_name} form where one exists, otherwise leave "
    "them as written. Do not add, remove, or explain anything. "
    "Reply with ONLY the translation: no quotes, no notes, no alternatives.\n\n"
    "Examples:\n{examples_block}"
)


def build_examples_block(examples: dict, lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in examples.get(lang_code, [])]
    return "\n\n".join(lines)


def build_entity_system_prompt(lang_code: str) -> str:
    return ENTITY_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(ENTITY_EXAMPLES, lang_code))


def build_context_system_prompt(lang_code: str) -> str:
    return CONTEXT_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(CONTEXT_EXAMPLES, lang_code))


# --- entity1/entity2 parsing ---------------------------------------------------------

def parse_entity(s: str) -> tuple[str, str, str]:
    name, tags, desc = s.split(" <S> ", 2)
    return name, tags, desc


def rebuild_entity(name: str, tags: str, translated_desc: str) -> str:
    return f"{name} <S> {tags} <S> {translated_desc}"


def load_clarqa(config: str):
    raw = load_dataset("DeepPavlov/clarqa", config)
    for split in raw:
        print(f"{config}/{split}: {len(raw[split])} rows")
    return raw


def collect_entity_descriptions(raw) -> list[str]:
    descs = set()
    for split in raw:
        for row in raw[split]:
            for field in ("entity1", "entity2"):
                _, _, desc = parse_entity(row[field])
                descs.add(desc)
    return sorted(descs)


def collect_context_texts(raw, config: str) -> list[str]:
    texts = set()
    for split in raw:
        for row in raw[split]:
            if config == "single_turn":
                texts.add(row["context"])
            else:
                # turn[1] is always an exact copy of an entity name (verified over the
                # whole dataset) -- a structural placeholder, not free text to translate.
                texts.add(row["context"][0])
                texts.add(row["context"][2])
    return sorted(texts)


def entities_checkpoint_path(config: str, lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"entities_{config}_{lang_code}_{model_key}.jsonl"


def contexts_checkpoint_path(config: str, lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"contexts_{config}_{lang_code}_{model_key}.jsonl"


KEY_FIELDS = ("en",)


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))

    for config in args.configs:
        raw = load_clarqa(config)
        descs = collect_entity_descriptions(raw)
        contexts = collect_context_texts(raw, config)
        print(f"{config}: {len(descs)} unique entity descriptions, {len(contexts)} unique context texts")

        for lang_code in args.langs:
            translator.translate_units(
                [((s,), s) for s in descs], build_entity_system_prompt(lang_code),
                entities_checkpoint_path(config, lang_code, args.model), KEY_FIELDS,
            )
            translator.translate_units(
                [((s,), s) for s in contexts], build_context_system_prompt(lang_code),
                contexts_checkpoint_path(config, lang_code, args.model), KEY_FIELDS,
            )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))

    for config in args.configs:
        raw = load_clarqa(config)
        sample = raw["dev"].select(range(3))
        descs = sorted({parse_entity(row[f])[2] for row in sample for f in ("entity1", "entity2")})
        contexts = collect_context_texts({"dev": sample}, config)

        for lang_code in args.langs:
            e_out = OUT_DIR / f"smoketest_entities_{config}_{lang_code}_{args.model}.jsonl"
            e_result = translator.translate_units([((s,), s) for s in descs], build_entity_system_prompt(lang_code), e_out, KEY_FIELDS)
            for s in descs:
                print(f"[entity/{config}/{lang_code}/{args.model}] {s[:80]!r} -> {e_result[(s,)][:80]!r}")

            c_out = OUT_DIR / f"smoketest_contexts_{config}_{lang_code}_{args.model}.jsonl"
            c_result = translator.translate_units([((s,), s) for s in contexts], build_context_system_prompt(lang_code), c_out, KEY_FIELDS)
            for s in contexts:
                print(f"[context/{config}/{lang_code}/{args.model}] {s!r} -> {c_result[(s,)]!r}")
            print()


def cmd_assemble(args):
    final = {}
    for config in args.configs:
        raw = load_clarqa(config)
        descs = collect_entity_descriptions(raw)
        contexts = collect_context_texts(raw, config)

        for lang_code in args.langs:
            for model_key in MODELS:
                e_path = entities_checkpoint_path(config, lang_code, model_key)
                c_path = contexts_checkpoint_path(config, lang_code, model_key)
                if not e_path.exists() or not c_path.exists():
                    print(f"skipping {config}/{lang_code}/{model_key}: missing checkpoint(s) at {e_path} / {c_path}")
                    continue

                entity_lookup_raw = load_checkpoint(e_path, KEY_FIELDS)
                context_lookup_raw = load_checkpoint(c_path, KEY_FIELDS)
                missing_e = [s for s in descs if (s,) not in entity_lookup_raw]
                missing_c = [s for s in contexts if (s,) not in context_lookup_raw]
                if missing_e or missing_c:
                    print(
                        f"skipping {config}/{lang_code}/{model_key}: incomplete "
                        f"({len(descs) - len(missing_e)}/{len(descs)} entities, "
                        f"{len(contexts) - len(missing_c)}/{len(contexts)} contexts translated) "
                        f"-- run `translate` first"
                    )
                    continue

                entity_lookup = {k[0]: v for k, v in entity_lookup_raw.items()}
                context_lookup = {k[0]: v for k, v in context_lookup_raw.items()}

                def translate_entity_field(s):
                    name, tags, desc = parse_entity(s)
                    return rebuild_entity(name, tags, entity_lookup[desc])

                dd = DatasetDict()
                for split_name in raw:
                    rows = []
                    for row in raw[split_name]:
                        if config == "single_turn":
                            new_context = context_lookup[row["context"]]
                        else:
                            new_context = [context_lookup[row["context"][0]], row["context"][1], context_lookup[row["context"][2]]]
                        rows.append({
                            "entity1": translate_entity_field(row["entity1"]),
                            "entity2": translate_entity_field(row["entity2"]),
                            "label": row["label"],
                            "context": new_context,
                        })
                    dd[split_name] = Dataset.from_list(rows)
                final[(config, lang_code, model_key)] = dd
                print(config, lang_code, model_key, {s: len(dd[s]) for s in dd})

    if not final:
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check:
        (config, lang_code, model_key), dd = next(iter(final.items()))
        split_name = "dev" if "dev" in dd else next(iter(dd))
        idxs = random.sample(range(len(dd[split_name])), min(3, len(dd[split_name])))
        for i in idxs:
            row = dd[split_name][i]
            print("ENTITY1:", row["entity1"][:200])
            print("ENTITY2:", row["entity2"][:200])
            print("CONTEXT:", row["context"])
            print("LABEL:", row["label"])
            print()

    for (config, lang_code, model_key), dd in final.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-{config}"))
    print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/clarqa-mt"
    # for (config, lang_code, model_key), dd in final.items():
    #     dd.push_to_hub(repo_id, config_name=f"{lang_code}-{model_key}-{config}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p, model_required=True):
        p.add_argument("--model", choices=list(MODELS), required=model_required)
        p.add_argument("--langs", nargs="+", choices=list(LANGUAGES), default=DEFAULT_LANGS)
        p.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)

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

    p_smoke = sub.add_parser("smoke-test", help="Translate a small entity/context sample and print it")
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
