from __future__ import annotations

from collections import Counter

import datasets
from datasets import DatasetDict

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TurkicClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TurkicClassification",
        description="A dataset of news classification in three Turkic languages.",
        dataset={
            "path": "Electrotubbie/classification_Turkic_languages",
            "revision": "db1a67c1bdd54fbb8536af026dc8596f00f9c41d",
        },
        reference="https://huggingface.co/datasets/Electrotubbie/classification_Turkic_languages/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={
            "ky": ["kir-Cyrl"],
            "kk": ["kaz-Cyrl"],
            "ba": ["bak-Cyrl"],
        },
        main_score="accuracy",
        date=("2023-02-16", "2023-09-03"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        """,
    )

    def transform_data(self, dataset, lang):
        dataset_lang = DatasetDict()
        label_count = Counter(dataset["train"]["label"])
        dataset_lang["train"] = dataset["train"].filter(
            lambda example: example["lang"] == lang
            and label_count[example["label"]] >= 20
        )
        dataset_lang = self.stratified_subsampling(
            dataset_lang, seed=self.seed, splits=["train"]
        )
        return dataset_lang["train"]

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        dataset = {}
        full_dataset = datasets.load_dataset(**self.metadata.dataset)
        full_dataset = full_dataset.rename_columns(
            {"processed_text": "text", "category": "label"}
        )
        for lang in self.hf_subsets:
            dataset[lang] = DatasetDict()
            filtered_dataset = self.transform_data(full_dataset, lang)

            dataset[lang]["train"] = filtered_dataset

        self.dataset = dataset
        self.dataset_transform()
        self.data_loaded = True
