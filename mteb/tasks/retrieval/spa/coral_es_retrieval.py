import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EsCoral(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EsCoral",
        description="coral",
        reference="https://huggingface.co/datasets/DeepPavlov/coral_es",
        dataset={
            "path": "DeepPavlov/coral_es",
            "revision": "66f1cc3a1d5e06c31ba80b21d0aad3f4c3304ac4",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test", "train"],
        eval_langs=["spa-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["Coral"],
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return

        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        corpus_table = datasets.load_dataset(
            path,
            "corpus",
            split="train",
            revision=revision,
            verification_mode="no_checks",
        )
        corpus = {str(row["id"]): {"text": row["text"]} for row in corpus_table}

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in self.metadata.eval_splits:
            queries_table = datasets.load_dataset(
                path,
                "queries",
                split=split,
                revision=revision,
                verification_mode="no_checks",
            )
            qrels_table = datasets.load_dataset(
                path,
                "qrels",
                split=split,
                revision=revision,
                verification_mode="no_checks",
            )

            self.queries[split] = {str(row["id"]): row["text"] for row in queries_table}
            relevant_docs: dict[str, dict[str, int]] = {}
            for row in qrels_table:
                relevant_docs.setdefault(str(row["query-id"]), {})[
                    str(row["corpus-id"])
                ] = row["score"]
            self.relevant_docs[split] = relevant_docs
            self.corpus[split] = corpus

        self.data_loaded = True
