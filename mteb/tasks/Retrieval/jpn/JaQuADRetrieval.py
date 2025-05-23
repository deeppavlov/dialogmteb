from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class JaQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JaQuADRetrieval",
        dataset={
            "path": "SkelterLabsInc/JaQuAD",
            "revision": "05600ff310a0970823e70f82f428893b85c71ffe",
            "trust_remote_code": True,
        },
        description="Human-annotated question-answer pairs for Japanese wikipedia pages.",
        reference="https://arxiv.org/abs/2202.01764",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),  # approximate guess
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation="found",
        bibtex_citation="""@misc{so2022jaquad,
    title={{JaQuAD: Japanese Question Answering Dataset for Machine Reading Comprehension}},
    author={ByungHoon So and Kyuhong Byun and Kyungwon Kang and Seongjin Cho},
    year={2022},
    eprint={2202.01764},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        split = self.metadata.eval_splits[0]
        ds = load_dataset(**self.metadata.dataset, split=split)
        ds = ds.shuffle(seed=42)
        max_samples = min(2048, len(ds))
        ds = ds.select(
            range(max_samples)
        )  # limit the dataset size to make sure the task does not take too long to run
        title = ds["title"]
        question = ds["question"]
        context = ds["context"]
        answer = [a["text"][0] for a in ds["answers"]]

        self.corpus = {split: {}}
        self.relevant_docs = {split: {}}
        self.queries = {split: {}}

        text2id = {}
        n = 0
        for t, q, cont, ans in zip(title, question, context, answer):
            self.queries[split][str(n)] = q
            q_n = n
            n += 1
            if cont not in text2id:
                text2id[cont] = n
                self.corpus[split][str(n)] = {"title": t, "text": cont}
                n += 1
            if ans not in text2id:
                text2id[ans] = n
                self.corpus[split][str(n)] = {"title": t, "text": ans}
                n += 1

            self.relevant_docs[split][str(q_n)] = {
                str(text2id[ans]): 1,
                str(text2id[cont]): 1,
            }  # only two correct matches

        self.data_loaded = True
