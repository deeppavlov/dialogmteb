from datasets import load_dataset

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EsDialogSumRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EsDialogSumRetrieval",
        dataset={
            "path": "DeepPavlov/dialogsum_es",
            "revision": "99989a1cfdcaab54c1d9e001d7fcc9e0a59cf863",
        },
        reference="https://huggingface.co/datasets/DeepPavlov/dialogsum_es",
        description=(
            "DialogSum is a large-scale dialogue summarization dataset. "
            + "DeepPavlov/dialogsum_es ships raw dialogue/summary pairs rather than a "
            + "pre-built retrieval corpus/queries/qrels split, so this task builds the "
            + "retrieval triples itself: the query is the dialogue and the corpus "
            + "document is its gold summary, one document per dialogue."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken"],
        task_subtypes=["Conversational retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in kwargs.get("eval_splits", self.metadata.eval_splits):
            corpus, queries, qrels = self._load_data_for_split(split)
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True

    def _load_data_for_split(self, split):
        ds = load_dataset(split=split, **self.metadata.dataset)
        queries, corpus, qrels = {}, {}, {}
        for i, sample in enumerate(ds):
            doc_id = "doc:" + str(i)
            corpus[doc_id] = {
                "title": "",
                "text": sample["summary"],
            }
            query_id = "query:" + str(i)
            queries[query_id] = sample["dialogue"]
            qrels[query_id] = {doc_id: 1}

        return corpus, queries, qrels
