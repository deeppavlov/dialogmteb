from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EsClarQA(AbsTaskPairClassification):
    input1_column_name = "entity1"
    input2_column_name = "entity2"
    label_column_name = "label"

    metadata = TaskMetadata(
        name="EsClarQA",
        description="ClarQA.",
        reference="https://huggingface.co/datasets/DeepPavlov/clarqa_es",
        dataset={
            "path": "DeepPavlov/clarqa_es",
            "revision": "7c15db2214f0817af90bb3baf934eaa142a55068",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "single_turn": ["spa-Latn"],
            "multi_turn": ["spa-Latn"],
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
