from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ItalianLinguisticAcceptabilityClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Itacola",
        dataset={
            "path": "gsarti/itacola",
            "revision": "f8f98e5c4d3059cf1a00c8eb3d70aa271423f636",
            "trust_remote_code": True,
        },
        description="An Italian Corpus of Linguistic Acceptability taken from linguistic literature with a binary annotation made by the original authors themselves.",
        reference="https://aclanthology.org/2021.findings-emnlp.250/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),
        domains=["Non-fiction", "Spoken", "Written"],
        dialect=[],
        task_subtypes=["Linguistic acceptability"],
        license="not specified",
        annotations_creators="expert-annotated",
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{trotta-etal-2021-monolingual-cross,
    title = "Monolingual and Cross-Lingual Acceptability Judgments with the {I}talian {C}o{LA} corpus",
    author = "Trotta, Daniela  and
      Guarasci, Raffaele  and
      Leonardelli, Elisa  and
      Tonelli, Sara",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.250",
    doi = "10.18653/v1/2021.findings-emnlp.250",
    pages = "2929--2940"
}
        """,
    )

    def dataset_transform(self):
        self.dataset = (
            self.dataset.rename_columns({"acceptability": "label"})
            .rename_columns({"sentence": "text"})
            .remove_columns(["unique_id", "source"])
        )
