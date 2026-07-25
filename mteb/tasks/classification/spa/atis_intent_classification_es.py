from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EsAtisIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EsAtisIntentClassification",
        description="The ATIS Spoken Language Systems Pilot Corpus",
        dataset={
            "path": "DeepPavlov/atis_intent_classification_es",
            "revision": "9c8e7b38f923ce1b9ae3027695ca5fffc4716ac9",
        },
        reference="https://huggingface.co/datasets/DeepPavlov/atis_intent_classification_es",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        date=("1990-01-01", "1990-01-01"),
        domains=["Spoken"],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["AtisIntentClassification"],
    )
