from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class FaroeseSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="FaroeseSTS",
        dataset={
            "path": "vesteinn/faroese-sts",
            "revision": "8cb36efa69428b3dc290e1125995a999963163c5",
        },
        description="Semantic Text Similarity (STS) corpus for Faroese.",
        reference="https://aclanthology.org/2023.nodalida-1.74.pdf",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["fao-Latn"],
        main_score="cosine_spearman",
        date=("2018-05-01", "2020-03-31"),
        domains=["News", "Web", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @inproceedings{snaebjarnarson-etal-2023-transfer,
            title = "{T}ransfer to a Low-Resource Language via Close Relatives: The Case Study on Faroese",
            author = "Snæbjarnarson, Vésteinn  and
            Simonsen, Annika  and
            Glavaš, Goran  and
            Vulić, Ivan",
            booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
            month = "may 22--24",
            year = "2023",
            address = "Tórshavn, Faroe Islands",
            publisher = {Link{\"o}ping University Electronic Press, Sweden},
        }
        """,
    )

    min_score = 0
    max_score = 5

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("label", "score")
