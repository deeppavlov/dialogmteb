from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HWUIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HWUIntentClassificationRu",
        description="",
        dataset={
            "path": "DeepPavlov/hwu_intent_classification_ru",
            "revision": "4335d9ed6ee852568247a7927c197498b7a37ad1",
        },
        reference="https://arxiv.org/abs/1903.05566",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="f1",
        date=("26-03-2019", "26-03-2019"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""""",
    )