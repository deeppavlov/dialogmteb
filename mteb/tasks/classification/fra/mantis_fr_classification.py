from __future__ import annotations

from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FrMantisClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrMantisClassification",
        description="Mantis",
        dataset={
            "path": "DeepPavlov/mantis_fr",
            "revision": "ca6ab868022ff5cee661d6d5f8466dd1f27d83fc",
        },
        reference=None,
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="f1",
        date=None,
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@misc{penha2019introducingmantisnovelmultidomain,
  archiveprefix = {arXiv},
  author = {Gustavo Penha and Alexandru Balan and Claudia Hauff},
  eprint = {1912.04639},
  primaryclass = {cs.CL},
  title = {Introducing MANtIS: a novel Multi-Domain Information Seeking Dialogues Dataset},
  url = {https://arxiv.org/abs/1912.04639},
  year = {2019},
}
""",
        adapted_from=["MantisClassification"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["dialog"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['message']}\n"
                    else:
                        text += f"Assistant: {entry['message']}\n"
            row["text"] = text
            return row

        for subset in self.dataset:
            self.dataset[subset] = (
                self.dataset[subset]
                .map(
                    process_history,
                )
                .rename_column("category", "label")
            )
