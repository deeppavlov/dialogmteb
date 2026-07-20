from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EsAirDialogueClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EsAirDialogueClassification",
        description="AirDialogue is a dataset of goal-oriented customer-agent conversations focused on booking flights under various travel restrictions.",
        dataset={
            "path": "DeepPavlov/air_dialog_es",
            "revision": "d77904648805cb93361e08f20bd583918c531f59",
        },
        reference="https://huggingface.co/datasets/DeepPavlov/air_dialog_es",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
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
