from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FrClarQA(AbsTaskPairClassification):
    input1_column_name = "entity1"
    input2_column_name = "entity2"
    label_column_name = "label"

    metadata = TaskMetadata(
        name="FrClarQA",
        description="ClarQA.",
        reference=None,
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
        date=None,
        domains=[],
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{xu-etal-2019-asking,
  address = {Hong Kong, China},
  author = {Xu, Jingjing  and
Wang, Yuechen  and
Tang, Duyu  and
Duan, Nan  and
Yang, Pengcheng  and
Zeng, Qi  and
Zhou, Ming  and
Sun, Xu},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  doi = {10.18653/v1/D19-1172},
  editor = {Inui, Kentaro  and
Jiang, Jing  and
Ng, Vincent  and
Wan, Xiaojun},
  month = nov,
  pages = {1618--1629},
  publisher = {Association for Computational Linguistics},
  title = {Asking Clarification Questions in Knowledge-Based Question Answering},
  url = {https://aclanthology.org/D19-1172/},
  year = {2019},
}
""",
        adapted_from=["ClarQA"],
    )
