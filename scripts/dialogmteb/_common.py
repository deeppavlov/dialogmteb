"""Shared utilities for DialogMTEB paper scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import mteb
from mteb.cache import ResultCache

BENCHMARK_NAME_ENG = "DialogMTEB(v1, eng)"
BENCHMARK_NAME_MULTILINGUAL = "DialogMTEB(v1, multilingual)"
BENCHMARK_NAME = BENCHMARK_NAME_MULTILINGUAL  # default (full benchmark)
BENCHMARKS: dict[str, str] = {
    "eng": BENCHMARK_NAME_ENG,
    "multilingual": BENCHMARK_NAME_MULTILINGUAL,
}
RESULT_CACHE_PATH = Path("/Users/samoed/Desktop/dialogmteb/result_cache")
OUT_DIR = Path(__file__).parent

cache = ResultCache(RESULT_CACHE_PATH)

TYPE_ABBREV = {
    "Classification": "Clf.",
    "MultilabelClassification": "M.Clf.",
    "PairClassification": "PC",
    "Reranking": "Rrnk.",
    "Retrieval": "Rtrvl.",
    "STS": "STS",
}
TYPE_ORDER = [
    "Classification",
    "MultilabelClassification",
    "PairClassification",
    "Reranking",
    "Retrieval",
    "STS",
]
TYPE_COLORS = {
    "Classification": "#4e79a7",
    "MultilabelClassification": "#f28e2b",
    "PairClassification": "#e15759",
    "Reranking": "#76b7b2",
    "Retrieval": "#59a14f",
    "STS": "#edc948",
}


def load_unique_tasks(benchmark_name: str = BENCHMARK_NAME) -> list:
    benchmark = mteb.get_benchmark(benchmark_name)
    seen: set[str] = set()
    unique = []
    for task in benchmark.tasks:
        if task.metadata.name not in seen:
            seen.add(task.metadata.name)
            unique.append(task)
    return unique


def load_score_df(tasks: list, models: list[str] | None = None) -> pd.DataFrame:
    """Return DataFrame rows=tasks, cols=models (0–1 scale)."""
    results = cache.load_results(
        tasks=tasks, models=models if models is not None else RUN_MODELS
    )
    return results.to_dataframe().set_index("task_name")


def compute_borda(score_df: pd.DataFrame, task_names: list[str]) -> pd.Series:
    available = [t for t in task_names if t in score_df.index]
    borda = pd.Series(0.0, index=score_df.columns)
    for task in available:
        row = score_df.loc[task].dropna().astype(float)
        if row.empty:
            continue
        ranks = row.rank(ascending=False, method="min")
        borda[row.index] += len(row) - ranks + 1
    return borda


def model_mean_scores(score_df: pd.DataFrame, task_names: list[str]) -> pd.Series:
    available = [t for t in task_names if t in score_df.index]
    mt = score_df.loc[available].T.dropna(how="all")
    return mt[available].mean(axis=1, skipna=True)


RUN_MODELS: list[str] = [
    "google/embeddinggemma-300m",
    "intfloat/multilingual-e5-large-instruct",
    "microsoft/harrier-oss-v1-270m",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "microsoft/harrier-oss-v1-0.6b",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "perplexity-ai/pplx-embed-v1-0.6b",
    "perplexity-ai/pplx-embed-v1-4b",
    "perplexity-ai/pplx-embed-v1-8b",
    "BidirLM/BidirLM-270M-Embedding",
    "BidirLM/BidirLM-0.6B-Embedding",
    "BidirLM/BidirLM-1.7B-Embedding",
    "NovaSearch/stella_en_400M_v5",
    "NovaSearch/stella_en_1.5B_v5",
    "NovaSearch/jasper_en_vision_language_v1",
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "BAAI/bge-m3",
    "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2",
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    "codefuse-ai/F2LLM-v2-80M",
    "codefuse-ai/F2LLM-v2-160M",
    "codefuse-ai/F2LLM-v2-330M",
    "codefuse-ai/F2LLM-v2-0.6B",
    "codefuse-ai/F2LLM-v2-4B",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "nvidia/llama-embed-nemotron-8b",
    "princeton-nlp/sup-simcse-bert-base-uncased",
    "princeton-nlp/unsup-simcse-roberta-base",
    "princeton-nlp/unsup-simcse-bert-large-uncased",
    "princeton-nlp/unsup-simcse-roberta-large",
    "princeton-nlp/sup-simcse-bert-large-uncased",
    "princeton-nlp/sup-simcse-roberta-base",
    "princeton-nlp/sup-simcse-roberta-large",
    "TODBERT/TOD-BERT-MLM-V1",
    "AndrewZeng/futuretod-base-v1.0",
]


def get_complete_models(
    score_df: pd.DataFrame,
    task_names: list[str],
    min_coverage: int | None = None,
) -> list[str]:
    """Return run.py models that have results for all (or ≥ min_coverage) tasks.

    Default min_coverage = len(task_names) (all tasks required).
    """
    available_tasks = [t for t in task_names if t in score_df.index]
    threshold = min_coverage if min_coverage is not None else len(available_tasks)
    complete = []
    for model in RUN_MODELS:
        if model in score_df.columns:
            n = int(score_df.loc[available_tasks, model].notna().sum())
            if n >= threshold:
                complete.append(model)
    return complete


def fetch_model_meta(name: str) -> dict:
    try:
        m = mteb.get_model_meta(name)
        return {
            "n_parameters": m.n_parameters,
            "embed_dim": m.embed_dim,
            "max_tokens": m.max_tokens,
            "open_weights": m.open_weights,
        }
    except Exception:
        return {
            "n_parameters": None,
            "embed_dim": None,
            "max_tokens": None,
            "open_weights": None,
        }
