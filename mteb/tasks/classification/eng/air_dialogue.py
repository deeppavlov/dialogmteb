from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AirDialogueClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AirDialogueClassification",
        description="AirDialogue is a dataset of goal-oriented customer-agent conversations focused on booking flights under various travel restrictions.",
        dataset={
            "path": "DeepPavlov/air_dialogue",
            "revision": "main",
        },
        reference="https://huggingface.co/datasets/google/air_dialogue",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        date=("2018-01-01", "2022-06-07"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{wei-etal-2018-airdialogue,
  address = {Brussels, Belgium},
  author = {Wei, Wei  and
Le, Quoc  and
Dai, Andrew  and
Li, Jia},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D18-1419},
  editor = {Riloff, Ellen  and
Chiang, David  and
Hockenmaier, Julia  and
Tsujii, Jun{'}ichi},
  month = oct # {-} # nov,
  pages = {3844--3854},
  publisher = {Association for Computational Linguistics},
  title = {{A}ir{D}ialogue: An Environment for Goal-Oriented Dialogue Research},
  url = {https://aclanthology.org/D18-1419},
  year = {2018},
}
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["text"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            row["text"] = text
            return row

        for subset in self.dataset:
            self.dataset[subset] = (
                self.dataset[subset]
                .map(
                    process_history,
                    num_proc=num_proc,
                )
                .select_columns(["text", "label"])
            )
