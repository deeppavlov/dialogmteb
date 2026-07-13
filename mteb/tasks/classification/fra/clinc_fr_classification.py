from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FrClincIntentClassification(AbsTaskClassification):
    input_column_name = "utterance"

    metadata = TaskMetadata(
        name="FrClincIntentClassification",
        description="Task-oriented dialog systems need to know when a query falls outside their range of supported intents, but current text classification corpora only define label sets that cover every example. This is the single-config ('plus'-sized) packaging of CLINC150, DeepPavlov/clinc150_fr, rather than the small/plus/imbalanced multi-config packaging used by ClincIntentClassification.",
        dataset={
            "path": "DeepPavlov/clinc150_fr",
            "revision": "b5c3e44dc6d605bafb9585ce125b187dcc14e6c9",
        },
        reference="https://huggingface.co/datasets/clinc/clinc_oos",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-01-01"),
        domains=["Financial", "Web", "Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{larson-etal-2019-evaluation,
  author = {Larson, Stefan  and
Mahendran, Anish  and
Peper, Joseph J.  and
Clarke, Christopher  and
Lee, Andrew  and
Hill, Parker  and
Kummerfeld, Jonathan K.  and
Leach, Kevin  and
Laurenzano, Michael A.  and
Tang, Lingjia  and
Mars, Jason},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  title = {An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction},
  url = {https://www.aclweb.org/anthology/D19-1131},
  year = {2019},
}
""",
        adapted_from=["ClincIntentClassification"],
    )
