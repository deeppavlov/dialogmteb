from __future__ import annotations

from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FrAbgCosQA(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrAbgCosQA",
        description="AbgCosQA",
        dataset={
            "path": "DeepPavlov/coqa_abg_fr",
            "revision": "c4e23fb92e0f4a6e6d85ca0fcb35c1e96abfa024",
        },
        reference="https://huggingface.co/datasets/DeepPavlov/coqa_abg_fr",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="f1",
        date=("2021-01-01", "2021-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["AbgCosQA"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            full_text = row["story"] + " "
            for turn in row["history_turns"]:
                full_text += (
                    "User: " + turn["question"] + " Assistant: " + turn["answer"] + " "
                )
            full_text += (
                "User: "
                + row["target_turn"]["question"]
                + " Assistant: "
                + row["target_turn"]["answer"]
            )
            row["text"] = full_text
            row["label"] = row["ambiguity"] == "ambiguous"
            return row

        for subset in self.dataset:
            self.dataset[subset] = (
                self.dataset[subset]
                .map(
                    process_history,
                    num_proc=num_proc,
                )
                .select_columns(["text", "label"])
            )
