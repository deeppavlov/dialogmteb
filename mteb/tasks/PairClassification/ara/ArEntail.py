from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ArEntail(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="ArEntail",
        dataset={
            "path": "arbml/ArEntail",
            "revision": "4da4316c6e3287746ab74ff67dd252ad128fceff",
        },
        description="A manually-curated Arabic natural language inference dataset from news headlines.",
        reference="https://link.springer.com/article/10.1007/s10579-024-09731-1",
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ara-Arab"],
        main_score="max_ap",
        date=(
            "2020-01-01",
            "2024-03-04",
        ),  # best guess based on google searching random samples
        domains=["News", "Written"],
        task_subtypes=["Textual Entailment"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{obeidat2024arentail,
        title={ArEntail: manually-curated Arabic natural language inference dataset from news headlines},
        author={Obeidat, Rasha and Al-Harahsheh, Yara and Al-Ayyoub, Mahmoud and Gharaibeh, Maram},
        journal={Language Resources and Evaluation},
        pages={1--27},
        year={2024},
        publisher={Springer}
        }""",
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["premise"],
                    "sentence2": self.dataset[split]["hypothesis"],
                    "labels": self.dataset[split]["label"],
                }
            ]
        self.dataset = _dataset
