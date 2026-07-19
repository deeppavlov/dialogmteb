from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FrClarQA(AbsTaskPairClassification):
    input1_column_name = "entity1"
    input2_column_name = "entity2"
    label_column_name = "label"

    metadata = TaskMetadata(
        name="FrClarQA",
        description="ClarQA.",
        reference="https://huggingface.co/datasets/DeepPavlov/clarqa_fr",
        dataset={
            "path": "DeepPavlov/clarqa_fr",
            "revision": "fe5d77ba4762df41c8004619c9af7a7a5d2926ad",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "single_turn": ["fra-Latn"],
            "multi_turn": ["fra-Latn"],
        },
        main_score="max_ap",
        date=("2019-01-01", "2019-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["ClarQA"],
    )
