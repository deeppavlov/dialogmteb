"""Shared helpers for offline-vLLM dataset translation scripts (translate_mantis.py,
translate_wow.py).

Two vLLM engines don't coexist well in one process/GPU, so these scripts translate
with ONE model per process:

    python translate_mantis.py translate --model gemma
    python translate_mantis.py translate --model qwen
    python translate_mantis.py assemble        # no vLLM needed, reads both checkpoints

`vllm` is imported lazily inside OfflineTranslator so `--help` / `assemble` work in an
environment that doesn't have it installed (e.g. a laptop, vs. the GPU server).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
REASONING_MARKERS = re.compile(
    r"^\s*(here'?s a thinking process|let'?s (think|analyze)|step \d|\d+\.\s+\*\*)",
    re.IGNORECASE,
)


def clean_translation(raw_out: str) -> str:
    # Hybrid-thinking models sometimes emit reasoning before a closing </think> even
    # when no opening tag was generated (the chat template injects it) -- split on the
    # closing tag first, then also strip any well-formed <think>...</think> pair.
    out = raw_out.split("</think>")[-1] if "</think>" in raw_out else raw_out
    out = THINK_RE.sub("", out).strip()
    return out.strip('"').strip("'").strip()


def is_valid(out: str) -> bool:
    return bool(out) and not REASONING_MARKERS.search(out)


MODELS = {
    "gemma": {
        "model": "google/gemma-4-31B-it",
        "engine_kwargs": {
            "max_model_len": 32000,
        },
        "chat_template_kwargs": {},
    },
    "qwen": {
        "model": "Qwen/Qwen3.6-27B-FP8",
        "engine_kwargs": {
            "max_model_len": 32000,
            # Enables vLLM's structured parsing of Qwen3's reasoning output.
            "reasoning_parser": "qwen3",
            "language_model_only": True,
        },
        # Qwen3 is a hybrid-thinking model: without this it emits its chain-of-thought
        # as the actual response content instead of a final answer.
        "chat_template_kwargs": {"enable_thinking": False},
    },
}

RETRY_TEMPERATURE = 0.4  # temp=0 retries reproduce the exact same failure -- pointless
MAX_RETRY_ROUNDS = 3
DEFAULT_BATCH_SIZE = 3000  # prompts per llm.chat() call, for periodic checkpointing


def parse_engine_kwargs(pairs: list[str]) -> dict:
    """Parse CLI `--engine-kwarg KEY=VALUE` strings into a kwargs dict for vllm.LLM(...),
    e.g. ["tensor_parallel_size=2", "gpu_memory_utilization=0.85", "dtype=bfloat16"].
    Values are parsed with ast.literal_eval when possible (so ints/floats/bools/None
    come through as real Python values), falling back to the raw string otherwise.
    """
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--engine-kwarg expects KEY=VALUE, got {pair!r}")
        key, raw_value = pair.split("=", 1)
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        result[key] = value
    return result


class OfflineTranslator:
    """Loads ONE model's vLLM engine and runs checkpointed, batched translation."""

    def __init__(self, model_key: str, engine_kwargs: dict | None = None):
        from vllm import LLM

        cfg = MODELS[model_key]
        self.model_key = model_key
        self.chat_template_kwargs = cfg["chat_template_kwargs"]
        merged_engine_kwargs = {**cfg["engine_kwargs"], **(engine_kwargs or {})}
        print(f"Loading {model_key} ({cfg['model']}) into vLLM (engine_kwargs={merged_engine_kwargs})...")
        self.llm = LLM(model=cfg["model"], **merged_engine_kwargs)

    def _generate(self, conversations, temperature, max_tokens=None):
        from vllm import SamplingParams

        # max_tokens is optional in vLLM's SamplingParams -- omit it entirely rather
        # than pass an artificial cap, so generation isn't truncated.
        kwargs = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        sp = SamplingParams(**kwargs)
        outputs = self.llm.chat(
            conversations, sp, chat_template_kwargs=self.chat_template_kwargs, use_tqdm=True,
        )
        return [o.outputs[0].text for o in outputs]

    def _translate_batch_with_retry(self, texts, system_prompt, max_tokens=None):
        n = len(texts)
        current_idx = list(range(n))
        current_texts = list(texts)
        final = [None] * n
        last_cleaned = {}
        temperature = 0.0

        for round_num in range(MAX_RETRY_ROUNDS + 1):
            conversations = [
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": t}]
                for t in current_texts
            ]
            raw_outputs = self._generate(conversations, temperature, max_tokens)

            next_idx, next_texts = [], []
            for j, orig_i in enumerate(current_idx):
                cleaned = clean_translation(raw_outputs[j])
                last_cleaned[orig_i] = cleaned
                if is_valid(cleaned):
                    final[orig_i] = cleaned
                else:
                    next_idx.append(orig_i)
                    next_texts.append(current_texts[j])
            current_idx, current_texts = next_idx, next_texts
            temperature = RETRY_TEMPERATURE
            if not current_idx:
                break

        if current_idx:
            print(
                f"  WARNING: {len(current_idx)}/{n} items still look like leaked "
                f"reasoning after {MAX_RETRY_ROUNDS} retries -- using best-effort "
                f"cleaned output anyway"
            )
            for orig_i in current_idx:
                final[orig_i] = last_cleaned[orig_i]
        return final

    def translate_units(
        self,
        units: list[tuple[tuple, str]] | list[tuple[tuple, str, dict]],
        system_prompt: str,
        out_path: Path,
        key_fields: tuple[str, ...],
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_tokens: int | None = None,
    ) -> dict[tuple, str]:
        """units: list of (key_tuple, text) or (key_tuple, text, extra_fields). key_fields
        names key_tuple's positions, e.g. ("dialog_idx", "kind", "turn_idx") -- checkpointed
        as JSONL rows {**dict(zip(key_fields, key)), **extra_fields, "translated": ...}
        under out_path (extra_fields are metadata only, e.g. "role" -- not part of the
        lookup key). Returns dict[key_tuple] -> translated, merging any pre-existing
        checkpoint content.
        """
        units = [(u[0], u[1], u[2] if len(u) > 2 else {}) for u in units]

        done: dict[tuple, str] = {}
        if out_path.exists():
            with out_path.open() as f:
                for line in f:
                    row = json.loads(line)
                    key = tuple(row[k] for k in key_fields)
                    done[key] = row["translated"]

        todo = [(key, text, extra) for key, text, extra in units if key not in done]
        print(f"[{out_path.name}] {len(done)} cached, {len(todo)} to translate via {self.model_key}")
        if not todo:
            return done

        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_batches = (len(todo) + batch_size - 1) // batch_size
        with out_path.open("a") as f:
            for b in range(n_batches):
                batch = todo[b * batch_size : (b + 1) * batch_size]
                texts = [t for _, t, _ in batch]
                translations = self._translate_batch_with_retry(texts, system_prompt, max_tokens)
                for (key, _, extra), translated in zip(batch, translations):
                    row = dict(zip(key_fields, key))
                    row.update(extra)
                    row["translated"] = translated
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done[key] = translated
                f.flush()
                print(f"  [{out_path.name}] batch {b + 1}/{n_batches} done ({len(done)} total cached)")

        return done


def load_checkpoint(out_path: Path, key_fields: tuple[str, ...]) -> dict[tuple, str]:
    """Read-only variant of translate_units' checkpoint loading, for `assemble` mode
    (which must not require vLLM/OfflineTranslator to be constructible)."""
    done: dict[tuple, str] = {}
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                row = json.loads(line)
                key = tuple(row[k] for k in key_fields)
                done[key] = row["translated"]
    return done
