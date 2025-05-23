from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WongnaiReviewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WongnaiReviewsClassification",
        description="Wongnai features over 200,000 restaurants, beauty salons, and spas across Thailand on its platform, with detailed information about each merchant and user reviews. In this dataset there are 5 classes corressponding each star rating",
        reference="https://github.com/wongnai/wongnai-corpus",
        dataset={
            "path": "Wongnai/wongnai_reviews",
            "revision": "cd351eb26093aa4b232a2390a0da35e7fab21655",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tha-Thai"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"),
        dialect=[],
        domains=["Reviews", "Written"],
        task_subtypes=[],
        license="lgpl-3.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation="""
        @software{cstorm125_2020_3852912,
            author  = {cstorm125 and lukkiddd},
            title   = {PyThaiNLP/classification-benchmarks: v0.1-alpha},
            month   = may,
            year    = 2020,
            publisher = {Zenodo},
            version = {v0.1-alpha},
            doi     = {10.5281/zenodo.3852912},
            url     = {https://doi.org/10.5281/zenodo.3852912}
        }""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"review_body": "text", "star_rating": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
