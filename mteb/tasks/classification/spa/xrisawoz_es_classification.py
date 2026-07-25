import json
from typing import Any

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EsXRisaWoz(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EsXRisaWoz",
        description="XRisaWoz",
        reference="https://huggingface.co/datasets/DeepPavlov/XRISAWOZ_es",
        dataset={
            "path": "DeepPavlov/XRISAWOZ_es",
            "revision": "019126918fb32022c7ef4cc26778d72cd1f663f2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="accuracy",
        date=("2022-01-01", "2022-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["XRisaWoz"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = json.loads(row["history"]) if row["history"] else []
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            row["label"] = row["domains"][0]
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )
