from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class ClarQA(AbsTaskPairClassification):
    input1_column_name = "entity1"
    input2_column_name = "entity2"
    label_column_name = "label"

    metadata = TaskMetadata(
        name="ClarQA",
        description="ClarQA.",
        reference=None,
        dataset={
            "path": "DeepPavlov/clarqa",
            "revision": "1fedd5a0a9a75afdf121b46a8b4313ce9f960d3d",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "single_turn": ["eng-Latn"],
            "multi_turn": ["eng-Latn"],
        },
        main_score="max_ap",
        date=None,
        domains=[],
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
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
    )
