from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TeluguAndhraJyotiNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TeluguAndhraJyotiNewsClassification",
        description="A Telugu dataset for 5-class classification of Telugu news articles",
        reference="https://github.com/AnushaMotamarri/Telugu-Newspaper-Article-Dataset",
        dataset={
            "path": "mlexplorer008/telugu_news_classification",
            "revision": "3821aa93aa461c9263071e0897234e8d775ad616",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["tel-Telu"],
        main_score="f1",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"body": "text", "topic": "label"})
        self.dataset = self.stratified_subsampling(self.dataset, seed=self.seed)
