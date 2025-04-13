from __future__ import annotations

from typing import Any

import mteb
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.abs_dialog_state_tracking import AbsTaskDST


class MultiWoz21(AbsTaskDST):
    n_experiments = 1
    classification_columns = [
         'attraction-area',
         'attraction-name',
         'attraction-type',
         'hospital-department',
         'hotel-area',
         'hotel-book day',
         'hotel-book people',
         'hotel-book stay',
         'hotel-internet',
         'hotel-name',
         'hotel-parking',
         'hotel-pricerange',
         'hotel-stars',
         'hotel-type',
         'restaurant-area',
         'restaurant-book day',
         'restaurant-book people',
         'restaurant-book time',
         'restaurant-food',
         'restaurant-name',
         'restaurant-pricerange',
         'taxi-arriveby',
         'taxi-departure',
         'taxi-destination',
         'taxi-leaveat',
         'train-arriveby',
         'train-book people',
         'train-day',
         'train-departure',
         'train-destination',
         'train-leaveat',
        ]

    metadata = TaskMetadata(
        name="MultiWoz-v2.1",
        description="",
        reference="https://arxiv.org/abs/1810.00278",
        dataset={
            "path": "DeepPavlov/MultiWOZ-2.1",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
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
            return row

        self.dataset = self.dataset.map(
            process_history,
            remove_columns=["history"],
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    evaluator = mteb.MTEB([MultiWoz21()])
    model = mteb.get_model("minishlab/potion-base-2M")
    evaluator.run(
        model,
        encode_kwargs={"batch_size": 32},
    )
