#!/usr/bin/env python3
"""Translate DeepPavlov/coral to Spanish (and French) using offline vLLM inference.

CORAL is a large conversational-retrieval dataset: `corpus` (201,196 web/wiki
passages, `train` split only), `queries` (multi-turn conversations, `train`+`test`),
`rewritten_queries` (a second query representation, same ids), `qrels` (relevance
judgments, no text -- copied through unchanged).

Three dedup strategies, one per config (all verified against the live data):

1. `corpus`: plain exact-text dedup -- 201,196 rows share only 162,964 unique texts
   (~19% savings, no cross-row structure needed).
2. `queries`: ids look like `Train_a_{conv}_{step}` -- every row sharing a
   `Train_a_{conv}_*` prefix has byte-identical turns (100% consistent, same pattern as
   DeepPavlov/wizard_of_wikipedia's queries), so translation happens once per unique
   conversation (8,000 total) and gets replayed across every row in its group -- 132,144
   unique turns instead of ~396k raw occurrences.
3. `rewritten_queries`: despite sharing the same ids, this does NOT have the queries'
   growing-duplicate structure -- each row is its own 2-turn
   `[rewritten self-contained question, some passage]` pair, and rows sharing an id
   prefix are NOT identical (100% of groups differ, verified). So this pools by exact
   turn-content string globally instead (68,222 unique strings) rather than by
   conversation id.

`corpus` text is mostly English but a small minority of documents are already in other
languages (verified: a few Spanish/Italian passages found by inspection) -- the corpus
system prompt explicitly tells the model to leave already-non-English passages
unchanged rather than mistranslating them.

Usage (run one model per process -- see translate_common.py for why):

    python translate_coral.py translate --model gemma
    python translate_coral.py translate --model qwen
    python translate_coral.py smoke-test --model gemma
    python translate_coral.py assemble    # no vLLM needed, reads both checkpoints

Checkpointed to translations/coral/*.jsonl -- safe to interrupt/resume.
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

QUERY_SPLITS = ["test", "train"]  # smallest/fastest-feedback split first
CORPUS_BATCH_SIZE = 300  # smaller than the default: individual docs can run long

OUT_DIR = SCRIPT_DIR / "translations" / "coral"
SAVE_DIR = SCRIPT_DIR / "translations" / "coral_final"

CORPUS_EXAMPLES = {
    "es": [
        (
            "the 2016 world series was the championship series of major league baseball 's 2016 season .",
            "la serie mundial de 2016 fue la serie de campeonato de la temporada 2016 de las grandes ligas de beisbol .",
        ),
    ],
    "fr": [
        (
            "the 2016 world series was the championship series of major league baseball 's 2016 season .",
            "les world series 2016 etaient la serie de championnat de la saison 2016 de la ligue majeure de baseball .",
        ),
    ],
}

CORPUS_SYSTEM_PROMPT = (
    "You are a professional translator localizing web/encyclopedia passages for a "
    "conversational search system. Translate the passage from English into "
    "{lang_name}. The source text is often lowercase and loosely punctuated (tokenized "
    "web/wiki text, e.g. a space before periods) -- preserve that register rather than "
    "'fixing' it into formal capitalization. Preserve citation markers (e.g. '[1270]'), "
    "URLs, source names, dates, and numbers exactly. Keep proper nouns as their correct "
    "{lang_name} form where a standard one exists, otherwise leave them as written. "
    "IMPORTANT: if the passage is already written in {lang_name} or in a language other "
    "than English, reproduce it EXACTLY unchanged instead of attempting to translate it "
    "-- do not guess or mistranslate non-English source text. "
    "Do not add, remove, summarize, or explain anything. "
    "Reply with ONLY the translation (or the unchanged passage, per the rule above): no "
    "quotes, no notes.\n\nExamples:\n{examples_block}"
)

QUERY_EXAMPLES = {
    "es": [
        (
            "Who won the 2016 World Series in Major League Baseball?",
            "Quien gano la Serie Mundial de 2016 en las Grandes Ligas de Beisbol?",
        ),
        ("Can you give me a detailed summary of Game 7?", "Me puedes dar un resumen detallado del Juego 7?"),
    ],
    "fr": [
        (
            "Who won the 2016 World Series in Major League Baseball?",
            "Qui a remporte les World Series 2016 de la Ligue majeure de baseball ?",
        ),
        ("Can you give me a detailed summary of Game 7?", "Pouvez-vous me donner un resume detaille du match 7 ?"),
    ],
}

QUERY_SYSTEM_PROMPT = (
    "You are a professional translator localizing turns of a conversational search "
    "dialogue (a user asking questions, an assistant replying with information -- "
    "replies may be extended factual passages). Translate the text from English into "
    "{lang_name}, keeping the same meaning and register, natural to a native "
    "{lang_name} speaker. Keep proper nouns as their correct {lang_name} form where one "
    "exists. IMPORTANT: if the text is already written in {lang_name} or in a language "
    "other than English, reproduce it EXACTLY unchanged instead of attempting to "
    "translate it. Do not add, remove, or explain anything. "
    "Reply with ONLY the translation (or the unchanged text, per the rule above): no "
    "quotes, no notes, no alternatives.\n\nExamples:\n{examples_block}"
)


def build_examples_block(examples: dict, lang_code: str) -> str:
    lines = [f"EN: {en}\n{lang_code.upper()}: {translated}" for en, translated in examples.get(lang_code, [])]
    return "\n\n".join(lines)


def build_corpus_system_prompt(lang_code: str) -> str:
    return CORPUS_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(CORPUS_EXAMPLES, lang_code))


def build_query_system_prompt(lang_code: str) -> str:
    return QUERY_SYSTEM_PROMPT.format(lang_name=LANGUAGES[lang_code], examples_block=build_examples_block(QUERY_EXAMPLES, lang_code))


def conv_id_of(row_id: str) -> str:
    return row_id.rsplit("_", 1)[0]


def group_queries(dataset) -> dict[str, list]:
    groups: dict[str, list] = {}
    for row in dataset:
        groups.setdefault(conv_id_of(row["id"]), []).append(row)
    return groups


def verify_query_groups(query_groups: dict[str, dict]) -> None:
    inconsistent = []
    for split, groups in query_groups.items():
        for conv_id, rows in groups.items():
            sigs = {tuple((t["content"], t["role"]) for t in r["text"]) for r in rows}
            if len(sigs) > 1:
                inconsistent.append((split, conv_id))
        print(f"queries/{split}: {sum(len(v) for v in groups.values())} rows -> {len(groups)} unique conversations")
    if inconsistent:
        print(
            f"WARNING: {len(inconsistent)} conversation groups are NOT identical across "
            f"their rows (e.g. {inconsistent[:5]}) -- those groups' translations will "
            f"only reflect their first row."
        )
    else:
        print("verified: every conversation group is identical across its rows -- safe to dedupe")


def load_coral():
    raw_corpus = load_dataset("DeepPavlov/coral", "corpus")
    raw_queries = load_dataset("DeepPavlov/coral", "queries")
    raw_rewritten = load_dataset("DeepPavlov/coral", "rewritten_queries")
    raw_qrels = load_dataset("DeepPavlov/coral", "qrels")

    print(f"corpus/train: {len(raw_corpus['train'])} docs")
    for split in raw_queries:
        print(f"queries/{split}: {len(raw_queries[split])} rows")
    for split in raw_rewritten:
        print(f"rewritten_queries/{split}: {len(raw_rewritten[split])} rows")

    return raw_corpus, raw_queries, raw_rewritten, raw_qrels


def collect_corpus_texts(raw_corpus) -> list[str]:
    texts = set(raw_corpus["train"]["text"])
    texts.discard("")
    return sorted(texts)


def collect_query_units(groups: dict[str, list]) -> list[tuple[tuple, str]]:
    units = []
    for conv_id, rows in groups.items():
        rep = rows[0]  # verified identical across the group by verify_query_groups()
        for turn_idx, turn in enumerate(rep["text"]):
            units.append(((conv_id, turn_idx), turn["content"]))
    return units


def collect_rewritten_texts(raw_rewritten) -> list[str]:
    texts = set()
    for split in raw_rewritten:
        for row in raw_rewritten[split]:
            for turn in row["text"]:
                texts.add(turn["content"])
    texts.discard("")
    return sorted(texts)


def corpus_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"corpus_{lang_code}_{model_key}.jsonl"


def queries_checkpoint_path(split_name: str, lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"queries_{split_name}_{lang_code}_{model_key}.jsonl"


def rewritten_checkpoint_path(lang_code: str, model_key: str) -> Path:
    return OUT_DIR / f"rewritten_{lang_code}_{model_key}.jsonl"


FLAT_KEY_FIELDS = ("en",)
QUERY_KEY_FIELDS = ("conv_id", "turn_idx")


def cmd_translate(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, raw_rewritten, _ = load_coral()
    query_groups = {split: group_queries(raw_queries[split]) for split in raw_queries}
    verify_query_groups(query_groups)

    corpus_texts = collect_corpus_texts(raw_corpus)
    rewritten_texts = collect_rewritten_texts(raw_rewritten)
    print(f"{len(corpus_texts)} unique corpus texts, {len(rewritten_texts)} unique rewritten-query texts")

    for lang_code in args.langs:
        translator.translate_units(
            [((s,), s) for s in corpus_texts], build_corpus_system_prompt(lang_code),
            corpus_checkpoint_path(lang_code, args.model), FLAT_KEY_FIELDS, batch_size=CORPUS_BATCH_SIZE,
        )
        translator.translate_units(
            [((s,), s) for s in rewritten_texts], build_query_system_prompt(lang_code),
            rewritten_checkpoint_path(lang_code, args.model), FLAT_KEY_FIELDS,
        )

    for split_name in args.splits:
        units = collect_query_units(query_groups[split_name])
        for lang_code in args.langs:
            translator.translate_units(
                units, build_query_system_prompt(lang_code),
                queries_checkpoint_path(split_name, lang_code, args.model), QUERY_KEY_FIELDS,
            )


def cmd_smoke_test(args):
    translator = OfflineTranslator(args.model, engine_kwargs=parse_engine_kwargs(args.engine_kwarg))
    raw_corpus, raw_queries, raw_rewritten, _ = load_coral()
    query_groups = {split: group_queries(raw_queries[split]) for split in raw_queries}

    smoke_conv_ids = list(query_groups["test"])[:3]
    smoke_groups = {cid: query_groups["test"][cid] for cid in smoke_conv_ids}

    for lang_code in args.langs:
        c_texts = raw_corpus["train"]["text"][:5]
        c_out = OUT_DIR / f"smoketest_corpus_{lang_code}_{args.model}.jsonl"
        c_result = translator.translate_units([((s,), s) for s in c_texts], build_corpus_system_prompt(lang_code), c_out, FLAT_KEY_FIELDS)
        for s in c_texts[:2]:
            print(f"[corpus/{lang_code}/{args.model}] {s[:80]!r} -> {c_result[(s,)][:80]!r}")

        q_units = collect_query_units(smoke_groups)
        q_out = OUT_DIR / f"smoketest_queries_{lang_code}_{args.model}.jsonl"
        q_result = translator.translate_units(q_units, build_query_system_prompt(lang_code), q_out, QUERY_KEY_FIELDS)
        for key, text in q_units[:4]:
            print(f"[query/{lang_code}/{args.model}] {text[:80]!r} -> {q_result[key][:80]!r}")

        rw_sample = collect_rewritten_texts({"test": raw_rewritten["test"].select(range(3))})
        rw_out = OUT_DIR / f"smoketest_rewritten_{lang_code}_{args.model}.jsonl"
        rw_result = translator.translate_units([((s,), s) for s in rw_sample], build_query_system_prompt(lang_code), rw_out, FLAT_KEY_FIELDS)
        for s in rw_sample:
            print(f"[rewritten/{lang_code}/{args.model}] {s[:80]!r} -> {rw_result[(s,)][:80]!r}")
        print()


def cmd_assemble(args):
    raw_corpus, raw_queries, raw_rewritten, raw_qrels = load_coral()
    query_groups = {split: group_queries(raw_queries[split]) for split in raw_queries}
    corpus_texts = collect_corpus_texts(raw_corpus)
    rewritten_texts = collect_rewritten_texts(raw_rewritten)

    final_corpus, final_queries, final_rewritten = {}, {}, {}
    for lang_code in args.langs:
        for model_key in MODELS:
            c_path = corpus_checkpoint_path(lang_code, model_key)
            rw_path = rewritten_checkpoint_path(lang_code, model_key)

            if c_path.exists():
                c_lookup_raw = load_checkpoint(c_path, FLAT_KEY_FIELDS)
                missing_c = [s for s in corpus_texts if (s,) not in c_lookup_raw]
                if not missing_c:
                    c_lookup = {k[0]: v for k, v in c_lookup_raw.items()}
                    rows = [{"id": row["id"], "text": c_lookup.get(row["text"], row["text"])} for row in raw_corpus["train"]]
                    final_corpus[(lang_code, model_key)] = DatasetDict({"train": Dataset.from_list(rows)})
                    print("corpus", lang_code, model_key, len(rows), "docs")
                else:
                    print(f"skipping corpus {lang_code}/{model_key}: {len(corpus_texts) - len(missing_c)}/{len(corpus_texts)} translated -- run `translate` first")
            else:
                print(f"skipping corpus {lang_code}/{model_key}: no checkpoint at {c_path}")

            if rw_path.exists():
                rw_lookup_raw = load_checkpoint(rw_path, FLAT_KEY_FIELDS)
                missing_rw = [s for s in rewritten_texts if (s,) not in rw_lookup_raw]
                if not missing_rw:
                    rw_lookup = {k[0]: v for k, v in rw_lookup_raw.items()}
                    dd = DatasetDict()
                    for split_name in raw_rewritten:
                        rows = [
                            {"id": row["id"], "text": [{"content": rw_lookup.get(t["content"], t["content"]), "role": t["role"]} for t in row["text"]]}
                            for row in raw_rewritten[split_name]
                        ]
                        dd[split_name] = Dataset.from_list(rows)
                    final_rewritten[(lang_code, model_key)] = dd
                    print("rewritten_queries", lang_code, model_key, {s: len(dd[s]) for s in dd})
                else:
                    print(f"skipping rewritten_queries {lang_code}/{model_key}: {len(rewritten_texts) - len(missing_rw)}/{len(rewritten_texts)} translated -- run `translate` first")
            else:
                print(f"skipping rewritten_queries {lang_code}/{model_key}: no checkpoint at {rw_path}")

            dd = DatasetDict()
            incomplete = False
            for split_name in args.splits:
                out_path = queries_checkpoint_path(split_name, lang_code, model_key)
                if not out_path.exists():
                    print(f"skipping queries {lang_code}/{model_key}/{split_name}: no checkpoint at {out_path}")
                    incomplete = True
                    break
                expected_units = collect_query_units(query_groups[split_name])
                lookup = load_checkpoint(out_path, QUERY_KEY_FIELDS)
                missing = [key for key, _ in expected_units if key not in lookup]
                if missing:
                    print(
                        f"skipping queries {lang_code}/{model_key}/{split_name}: checkpoint incomplete "
                        f"({len(lookup)}/{len(expected_units)} translated, {len(missing)} missing "
                        f"e.g. {missing[:3]}) -- run `translate` for this model/split first"
                    )
                    incomplete = True
                    break
                rows = []
                for row in raw_queries[split_name]:
                    conv_id = conv_id_of(row["id"])
                    text = [{"content": lookup[(conv_id, i)], "role": turn["role"]} for i, turn in enumerate(row["text"])]
                    rows.append({"id": row["id"], "text": text})
                dd[split_name] = Dataset.from_list(rows)
            if not incomplete:
                final_queries[(lang_code, model_key)] = dd
                print("queries", lang_code, model_key, {s: len(dd[s]) for s in dd})

    if not (final_corpus or final_queries or final_rewritten):
        print("nothing to assemble -- no complete checkpoints found")
        return

    if args.spot_check and final_corpus and final_queries:
        lang_code, model_key = next(iter(final_corpus))
        print("CORPUS:", final_corpus[(lang_code, model_key)]["train"][0]["text"][:300])
        if (lang_code, model_key) in final_queries:
            qrow = final_queries[(lang_code, model_key)][args.splits[0]][0]
            for turn in qrow["text"][:3]:
                print(f"  [{turn['role']}]", turn["content"][:150])

    for (lang_code, model_key), dd in final_corpus.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-corpus"))
    for (lang_code, model_key), dd in final_queries.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-queries"))
    for (lang_code, model_key), dd in final_rewritten.items():
        dd.save_to_disk(str(SAVE_DIR / f"{lang_code}-{model_key}-rewritten_queries"))
    if final_corpus or final_queries or final_rewritten:
        raw_qrels.save_to_disk(str(SAVE_DIR / "qrels"))
        print("saved to", SAVE_DIR)

    # repo_id = "<your-username>/coral-mt"
    # for (lang_code, model_key), dd in final_corpus.items():
    #     dd.push_to_hub(repo_id, config_name=f"{lang_code}-{model_key}-corpus")
    # for (lang_code, model_key), dd in final_queries.items():
    #     dd.push_to_hub(repo_id, config_name=f"{lang_code}-{model_key}-queries")
    # for (lang_code, model_key), dd in final_rewritten.items():
    #     dd.push_to_hub(repo_id, config_name=f"{lang_code}-{model_key}-rewritten_queries")
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
    p_translate.add_argument("--splits", nargs="+", choices=QUERY_SPLITS, default=QUERY_SPLITS,
                              help="query splits to translate (corpus/rewritten_queries are always translated fully)")
    p_translate.set_defaults(func=cmd_translate)

    p_smoke = sub.add_parser("smoke-test", help="Translate a small corpus/query/rewritten sample and print it")
    add_common(p_smoke)
    add_engine_kwargs(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke_test)

    p_assemble = sub.add_parser("assemble", help="Build final datasets from existing checkpoints (no vLLM)")
    add_common(p_assemble, model_required=False)
    p_assemble.add_argument("--splits", nargs="+", choices=QUERY_SPLITS, default=QUERY_SPLITS)
    p_assemble.add_argument("--spot-check", action="store_true", default=True)
    p_assemble.add_argument("--no-spot-check", dest="spot_check", action="store_false")
    p_assemble.set_defaults(func=cmd_assemble)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
