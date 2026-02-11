from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class XRisaWozDomainClassification(AbsTaskClassification):
    n_experiments = 1

    input_column_name = "text"
    label_column_name = "domains"

    metadata = TaskMetadata(
        name="XRisaWozAttraction",
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
            f"en": ["eng-Latn"],
            f"fr": ["fra-Latn"],
            f"enhi": ["hin-Deva", "eng-Latn"],
            f"hi": ["hin-Deva"],
            f"ko": ["kor-Hang"],
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

    def dataset_transform(self) -> None:
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

            domains = row["domains"]
            if isinstance(domains, (list, tuple)):
                row["domains"] = domains[0] if domains else "none"
            elif domains is None:
                row["domains"] = "none"
            else:
                row["domains"] = str(domains)

            return row
        

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )