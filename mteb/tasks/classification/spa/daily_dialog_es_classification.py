from __future__ import annotations

from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EsDailyDialogClassificationAct(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EsDailyDialogClassificationAct",
        description="",
        dataset={
            "path": "DeepPavlov/daily_dialog_es",
            "revision": "aab6643aca506d913826cc8a6a0a1e1cd9425c69",
        },
        reference="https://huggingface.co/datasets/DeepPavlov/daily_dialog_es",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["spa-Latn"],
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
        self.dataset = self.dataset.filter(
            lambda row: row["dialog_es"] is not None, num_proc=num_proc
        )
        self.dataset = self.dataset.rename_columns(
            {"act_label": "label", "dialog_es": "text"}
        )


class EsDailyDialogClassificationEmotion(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EsDailyDialogClassificationEmotion",
        description="",
        dataset={
            "path": "DeepPavlov/daily_dialog_es",
            "revision": "aab6643aca506d913826cc8a6a0a1e1cd9425c69",
        },
        reference="https://huggingface.co/datasets/DeepPavlov/daily_dialog_es",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["spa-Latn"],
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
        self.dataset = self.dataset.filter(
            lambda row: row["dialog_es"] is not None, num_proc=num_proc
        )
        self.dataset = self.dataset.rename_columns(
            {"emotion_label": "label", "dialog_es": "text"}
        )
