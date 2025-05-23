from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class NLPJournalAbsIntroRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NLPJournalAbsIntroRetrieval",
        description="This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given abstract.",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "sbintuitions/JMTEB",
            "revision": "e4af6c73182bebb41d94cb336846e5a452454ea7",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2023-12-31"),
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(
            name="nlp_journal_abs_intro-query",
            split=_EVAL_SPLIT,
            **self.metadata.dataset,
        )

        queries = {}
        qrels = {}
        for row_id, row in enumerate(query_list):
            queries[str(row_id)] = row["query"]
            qrels[str(row_id)] = {str(row["relevant_docs"]): 1}

        corpus_list = datasets.load_dataset(
            name="nlp_journal_abs_intro-corpus",
            split="corpus",
            **self.metadata.dataset,
        )

        corpus = {str(row["docid"]): {"text": row["text"]} for row in corpus_list}

        self.corpus = {_EVAL_SPLIT: corpus}
        self.queries = {_EVAL_SPLIT: queries}
        self.relevant_docs = {_EVAL_SPLIT: qrels}

        self.data_loaded = True
