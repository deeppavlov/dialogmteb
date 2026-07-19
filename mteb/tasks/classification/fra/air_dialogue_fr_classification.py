from __future__ import annotations

from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FrAirDialogueClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrAirDialogueClassification",
        description="AirDialogue is a dataset of goal-oriented customer-agent conversations focused on booking flights under various travel restrictions.",
        dataset={
            "path": "DeepPavlov/air_dialog_fr",
            "revision": "414ee3c46fd2abc4df245f43a7ac0a33439f97f7",
        },
        reference="https://huggingface.co/datasets/google/air_dialogue",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="f1",
        date=("2018-01-01", "2022-06-07"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["AirDialogueClassification"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["text"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            row["text"] = text
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
