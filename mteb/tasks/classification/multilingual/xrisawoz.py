from typing import Any

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class XRisaWoz(AbsTaskClassification):
    metadata = TaskMetadata(
        name="XRisaWoz",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/XRISAWOZ",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            "en": ["eng-Latn"],
            "fr": ["fra-Latn"],
            "enhi": ["hin-Deva", "eng-Latn"],
            "hi": ["hin-Deva"],
            "ko": ["kor-Hang"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
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
