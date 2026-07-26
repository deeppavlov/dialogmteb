from typing import Any

from mteb.abstasks import AbsTaskReranking
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.reranking import OLD_FORMAT_RERANKING_TASKS

if "EsWebLINXCandidatesReranking" not in OLD_FORMAT_RERANKING_TASKS:
    OLD_FORMAT_RERANKING_TASKS.append("EsWebLINXCandidatesReranking")


class EsWebLINXCandidatesReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="EsWebLINXCandidatesReranking",
        description="WebLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.",
        reference="https://huggingface.co/datasets/DeepPavlov/WebLINX_es",
        dataset={
            "path": "DeepPavlov/WebLINX_es",
            "revision": "910ef9ed077be93b1021c1040e95dca63f2191bb",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[
            "validation",
            "test",
            "test_cat",
            "test_geo",
            "test_vis",
            "test_web",
        ],
        eval_langs=["spa-Latn"],
        main_score="mrr_at_10",
        date=("2023-03-01", "2023-10-30"),
        domains=["Academic", "Web", "Written"],
        task_subtypes=["Code retrieval", "Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["WebLINXCandidatesReranking"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any):
        for split in self.dataset:
            ds = self.dataset[split]
            self.dataset[split] = ds.select(range(min(len(ds), 300)))

        self.dataset = self.dataset.rename_columns(
            {"query": "query_en", "query_es": "query"}
        )
