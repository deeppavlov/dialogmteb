from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class IKAT2023(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="iKAT2023",
        description="The task is to retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.",
        reference="https://www.trecikat.com",
        dataset={
            "path": "DeepPavlov/iKAT_2023",
            "revision": "7cd389464922c22a68f42b81dd034a5583203db7",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-11-14", "2023-11-17"),
        domains=["Spoken"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Aliannejadi2024,
  series = {SIGIR 2024},
  title = {TREC iKAT 2023: A Test Collection for Evaluating Conversational and Interactive Knowledge Assistants},
  url = {http://dx.doi.org/10.1145/3626772.3657860},
  DOI = {10.1145/3626772.3657860},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  publisher = {ACM},
  author = {Aliannejadi,  Mohammad and Abbasiantaeb,  Zahra and Chatterjee,  Shubham and Dalton,  Jeffrey and Azzopardi,  Leif},
  year = {2024},
  month = jul,
  pages = {819–829},
  collection = {SIGIR 2024}
}
""",
    )
