import ast
from typing import Any

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EsQRECC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="EsQRECC",
        description="QRECC.",
        reference="https://huggingface.co/datasets/DeepPavlov/qrecc_es",
        dataset={
            "path": "DeepPavlov/qrecc_es",
            "revision": "ad589f40427e2897ba72c59b0aa52474007cfc4a",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="max_ap",
        date=("2020-10-01", "2021-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["QRECC"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def transform(example: dict) -> dict:
            context = (
                ast.literal_eval(example["context"])
                if isinstance(example["context"], str)
                else example["context"]
            )
            context_str = ""
            for replic in context:
                if replic["role"] == "user":
                    context_str += "User: " + replic["content"] + " "
                else:
                    context_str += "Assistant: " + replic["content"] + " "
            context_str += example["question"]
            return {
                "sentence1": context_str,
                "sentence2": example["rewrite"],
                "labels": 1,
            }

        self.dataset = self.dataset.map(transform, num_proc=num_proc)
