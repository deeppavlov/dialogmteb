from typing import Any

from datasets import Value

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FrCoral(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FrCoral",
        description="coral",
        reference="https://huggingface.co/datasets/DeepPavlov/coral_fr",
        dataset={
            "path": "DeepPavlov/coral_fr",
            "revision": "ba6a23690ebf49b5509601d9ff6a0955aa09fc90",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test", "train"],
        eval_langs=["fra-Latn"],
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

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset:
            for split in self.dataset[subset]:
                self.dataset[subset][split]["corpus"] = self.dataset[subset][split][
                    "corpus"
                ].cast_column("id", Value("string"))
