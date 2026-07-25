from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EsBanking77Classification(AbsTaskClassification):
    input_column_name = "utterance_es"

    metadata = TaskMetadata(
        name="EsBanking77Classification",
        description="Dataset composed of online banking queries annotated with their corresponding intents.",
        reference="https://huggingface.co/datasets/DeepPavlov/banking77_es",
        dataset={
            "path": "DeepPavlov/banking77_es",
            "revision": "0315065befdd71dbc76f69c8576c20ddac870af2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-12-31"),
        domains=["Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["Banking77Classification"],
    )
