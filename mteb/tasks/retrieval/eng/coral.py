from typing import Any

from datasets import Value

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Coral(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Coral",
        description="coral",
        reference="https://github.com/audio-captioning/clotho-dataset",
        dataset={
            "path": "DeepPavlov/coral",
            "revision": "f089b37a6975985a508abee879330e4a892be645",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test", "train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{cheng2024coralbenchmarkingmultiturnconversational,
  archiveprefix = {arXiv},
  author = {Yiruo Cheng and Kelong Mao and Ziliang Zhao and Guanting Dong and Hongjin Qian and Yongkang Wu and Tetsuya Sakai and Ji-Rong Wen and Zhicheng Dou},
  eprint = {2410.23090},
  primaryclass = {cs.IR},
  title = {CORAL: Benchmarking Multi-turn Conversational Retrieval-Augmentation Generation},
  url = {https://arxiv.org/abs/2410.23090},
  year = {2024},
}
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset:
            for split in self.dataset[subset]:
                self.dataset[subset][split]["corpus"] = self.dataset[subset][split][
                    "corpus"
                ].cast_column("id", Value("string"))
