from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FQuADRetrieval(AbsTaskRetrieval):
    _EVAL_SPLITS = ["test", "validation"]

    metadata = TaskMetadata(
        name="FQuADRetrieval",
        description="This dataset has been built from the French SQuad dataset.",
        reference="https://huggingface.co/datasets/manu/fquad2_test",
        dataset={
            "path": "manu/fquad2_test",
            "revision": "5384ce827bbc2156d46e6fcba83d75f8e6e1b4a6",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=_EVAL_SPLITS,
        eval_langs=["fra-Latn"],
        main_score="ndcg_at_10",
        date=("2019-11-01", "2020-05-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Article retrieval"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@inproceedings{dhoffschmidt-etal-2020-fquad,
    title = "{FQ}u{AD}: {F}rench Question Answering Dataset",
    author = "d{'}Hoffschmidt, Martin  and
      Belblidia, Wacim  and
      Heinrich, Quentin  and
      Brendl{\'e}, Tom  and
      Vidal, Maxime",
    editor = "Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.107",
    doi = "10.18653/v1/2020.findings-emnlp.107",
    pages = "1193--1208",
}""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        dataset_raw = datasets.load_dataset(
            **self.metadata.dataset,
        )

        # set valid_hasAns and test_hasAns as the validation and test splits (only queries with answers)
        dataset_raw["validation"] = dataset_raw["valid_hasAns"]
        del dataset_raw["valid_hasAns"]

        dataset_raw["test"] = dataset_raw["test_hasAns"]
        del dataset_raw["test_hasAns"]

        # rename  context column to text
        dataset_raw = dataset_raw.rename_column("context", "text")

        self.queries = {
            eval_split: {
                str(i): q["question"] for i, q in enumerate(dataset_raw[eval_split])
            }
            for eval_split in self.metadata.eval_splits
        }

        self.corpus = {
            eval_split: {str(row["title"]): row for row in dataset_raw[eval_split]}
            for eval_split in self.metadata.eval_splits
        }

        self.relevant_docs = {
            eval_split: {
                str(i): {str(q["title"]): 1}
                for i, q in enumerate(dataset_raw[eval_split])
            }
            for eval_split in self.metadata.eval_splits
        }

        self.data_loaded = True
