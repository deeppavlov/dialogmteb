from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class STS13STS(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="STS13",
        dataset={
            "path": "mteb/sts13-sts",
            "revision": "7e90230a92c190f1bf69ae9002b8cea547a64cca",
        },
        description="SemEval STS 2013 dataset.",
        reference="https://www.aclweb.org/anthology/S13-1004/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2012-12-31"),
        domains=["Web", "News", "Non-fiction", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@inproceedings{Agirre2013SEM2S,
  title={*SEM 2013 shared task: Semantic Textual Similarity},
  author={Eneko Agirre and Daniel Matthew Cer and Mona T. Diab and Aitor Gonzalez-Agirre and Weiwei Guo},
  booktitle={International Workshop on Semantic Evaluation},
  year={2013},
  url={https://api.semanticscholar.org/CorpusID:10241043}
}""",
    )
