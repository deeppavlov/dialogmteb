from __future__ import annotations

from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


def combine_dialogs(row: dict) -> dict:
    row["dialog"] = "\n".join(row["dialog"])
    return row


class FrDailyDialogClassificationAct(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrDailyDialogClassificationAct",
        description="",
        dataset={
            "path": "DeepPavlov/daily_dialog_fr",
            "revision": "4e1c4b0878f1bf5d6f41386e08aa9ad1ae787c4d",
        },
        reference="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2017-07-11", "2017-07-11"),
        domains=["Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["DailyDialogClassificationAct"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any):
        self.dataset = self.dataset.map(combine_dialogs)
        self.dataset = self.dataset.rename_columns(
            {"act_label": "label", "dialog": "text"}
        )


class FrDailyDialogClassificationEmotion(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrDailyDialogClassificationEmotion",
        description="",
        dataset={
            "path": "DeepPavlov/daily_dialog_fr",
            "revision": "4e1c4b0878f1bf5d6f41386e08aa9ad1ae787c4d",
        },
        reference="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2017-07-11", "2017-07-11"),
        domains=["Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["DailyDialogClassificationEmotion"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any):
        self.dataset = self.dataset.map(combine_dialogs)
        self.dataset = self.dataset.rename_columns(
            {"emotion_label": "label", "dialog": "text"}
        )
