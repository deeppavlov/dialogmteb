from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EsViraIntentClassification(AbsTaskClassification):
    input_column_name = "spanish_translated"

    metadata = TaskMetadata(
        name="EsViraIntentClassification",
        description="Chatbot-delivered COVID-19 vaccine communication message preferences of young adults and public health workers in urban American communities: qualitative study",
        dataset={
            "path": "DeepPavlov/vira_intents_live_es",
            "revision": "13d3cf131435d4c56d084ad8c1784dcdb2b69f44",
        },
        reference="https://huggingface.co/datasets/DeepPavlov/vira_intents_live_es",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["val", "test"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2022-07-06"),
        domains=["Medical"],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["ViraIntentClassification"],
    )
