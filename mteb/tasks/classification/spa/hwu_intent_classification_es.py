from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EsHWUIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EsHWUIntentClassification",
        description="",
        dataset={
            "path": "DeepPavlov/hwu64_es",
            "revision": "c17547823dde9bceff708e99e0f39d3688ef0c2b",
        },
        reference="https://huggingface.co/datasets/DeepPavlov/hwu64_es",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="f1",
        date=("2019-03-26", "2019-03-26"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["HWUIntentClassification"],
    )
