from __future__ import annotations

from datasets import Value

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EsClincIntentClassification(AbsTaskClassification):
    input_column_name = "utterance_es"

    metadata = TaskMetadata(
        name="EsClincIntentClassification",
        description="Task-oriented dialog systems need to know when a query falls outside their range of supported intents, but current text classification corpora only define label sets that cover every example. This is the single-config ('plus'-sized) packaging of CLINC150, DeepPavlov/clinc150_es, rather than the small/plus/imbalanced multi-config packaging used by ClincIntentClassification.",
        dataset={
            "path": "DeepPavlov/clinc150_es",
            "revision": "1a3ab3bb5110a8d31dbc4e2a61461936c756a5b7",
        },
        reference="https://huggingface.co/datasets/clinc/clinc_oos",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["val", "test"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-01-01"),
        domains=["Financial", "Web", "Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["ClincIntentClassification"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for subset in self.dataset:
            ds = self.dataset[subset].filter(
                lambda row: row["label"] is not None
                and row["utterance_es"] is not None,
                num_proc=num_proc,
            )
            self.dataset[subset] = ds.cast_column("label", Value("int64"))
