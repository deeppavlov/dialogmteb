from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class MTOPIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MTOPIntentClassification",
        dataset={
            "path": "DeepPavlov/mtop_intent_ru",
            "revision": "07921d318b63abc8753408455c90eca559fac339",
        },
        description="MTOP: Multilingual Task-Oriented Semantic Parsing",
        reference="https://arxiv.org/pdf/2008.09335.pdf",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["Spoken", "Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""@inproceedings{li-etal-2021-mtop,
    title = "{MTOP}: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark",
    author = "Li, Haoran  and
      Arora, Abhinav  and
      Chen, Shuohui  and
      Gupta, Anchit  and
      Gupta, Sonal  and
      Mehdad, Yashar",
    editor = "Merlo, Paola  and
      Tiedemann, Jorg  and
      Tsarfaty, Reut",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-main.257",
    doi = "10.18653/v1/2021.eacl-main.257",
    pages = "2950--2962",
    abstract = "Scaling semantic parsing models for task-oriented dialog systems to new languages is often expensive and time-consuming due to the lack of available datasets. Available datasets suffer from several shortcomings: a) they contain few languages b) they contain small amounts of labeled examples per language c) they are based on the simple intent and slot detection paradigm for non-compositional queries. In this paper, we present a new multilingual dataset, called MTOP, comprising of 100k annotated utterances in 6 languages across 11 domains. We use this dataset and other publicly available datasets to conduct a comprehensive benchmarking study on using various state-of-the-art multilingual pre-trained models for task-oriented semantic parsing. We achieve an average improvement of +6.3 points on Slot F1 for the two existing multilingual datasets, over best results reported in their experiments. Furthermore, we demonstrate strong zero-shot performance using pre-trained models combined with automatic translation and alignment, and a proposed distant supervision method to reduce the noise in slot label projection.",
}
""",
        prompt="Classify the intent of the given utterance in task-oriented conversation",
    )
