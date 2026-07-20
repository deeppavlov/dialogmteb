from datasets import load_dataset

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EsFaithDialRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EsFaithDialRetrieval",
        dataset={
            "path": "DeepPavlov/faithdial_es",
            "revision": "6d00b1c2496c9e54e2875b285536048338c494e6",
        },
        reference="https://huggingface.co/datasets/DeepPavlov/faithdial_es",
        description=(
            "FaithDial is a faithful knowledge-grounded dialogue benchmark."
            + "It was curated by asking annotators to amend hallucinated utterances in Wizard of Wikipedia (WoW). "
            + "It consists of conversation histories along with manually labelled relevant passage. "
            + "For the purpose of retrieval, we only consider the instances marked as 'Edification' in the VRM field, "
            + "as the gold passage associated with these instances is non-ambiguous."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-03-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["FaithDial"],
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
                "text": sample["knowledge_es"],
            }
            if "Edification" in sample["VRM"]:
                query_id = "query:" + str(i)
                query = sample["history_es"]
                queries[query_id] = query
                qrels[query_id] = {doc_id: 1}

        return corpus, queries, qrels
