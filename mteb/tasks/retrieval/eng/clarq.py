from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Clarq(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Clarq",
        description="Clarq",
        reference=None,
        dataset={
            "path": "DeepPavlov/clarq",
            "revision": "2ef85307be595df1bcdc49f668eb93c650fe3ba6",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=[],
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{kumar2020clarqlargescalediversedataset,
  archiveprefix = {arXiv},
  author = {Vaibhav Kumar and Alan W. black},
  eprint = {2006.05986},
  primaryclass = {cs.CL},
  title = {ClarQ: A large-scale and diverse dataset for Clarification Question Generation},
  url = {https://arxiv.org/abs/2006.05986},
  year = {2020},
}
""",
    )
