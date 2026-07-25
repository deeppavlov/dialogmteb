from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EsIKAT2023(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EsIKAT2023",
        description="The task is to retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.",
        reference="https://huggingface.co/datasets/DeepPavlov/ikat_es",
        dataset={
            "path": "DeepPavlov/ikat_es",
            "revision": "2dfa938d842aa10414e89f5ba107826cc9f0a296",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="ndcg_at_10",
        date=("2023-11-14", "2023-11-17"),
        domains=["Spoken"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["iKAT2023"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def fix_null_text(row: dict[str, Any]) -> dict[str, Any]:
            if row["text"] is None:
                row["text"] = row["utterance"]
            return row

        for subset in self.dataset:
            for split in self.dataset[subset]:
                self.dataset[subset][split]["queries"] = self.dataset[subset][split][
                    "queries"
                ].map(fix_null_text)
