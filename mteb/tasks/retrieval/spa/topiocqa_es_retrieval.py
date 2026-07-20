from datasets import load_dataset

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EsTopiOCQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EsTopiOCQARetrieval",
        dataset={
            "path": "DeepPavlov/topiocqa_es",
            "revision": "7611aeb9591bed4aea1e266e054507a7c01e815e",
        },
        reference="https://huggingface.co/datasets/DeepPavlov/topiocqa_es",
        description=(
            "TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) "
            + "is information-seeking conversational dataset with challenging topic switching phenomena. "
            + "It consists of conversation histories along with manually labelled relevant/gold passage. "
            + "Unlike the pre-built mteb/TopiOCQA corpus/queries/qrels split, DeepPavlov/topiocqa_es ships the "
            + "raw per-turn QA rows (Conversation_no, Turn_no, Question, Answer, Context, ...), so this task "
            + "builds the retrieval triples itself: for each turn, the query is the conversation history up to "
            + "and including that turn's Question, and the corpus document is that turn's gold Context "
            + "passage(s), one document per turn, following the same construction pattern used for "
            + "FrFaithDialRetrieval."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["spa-Latn"],
        main_score="ndcg_at_10",
        date=("2021-03-01", "2021-07-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["TopiOCQA"],
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
        ds = ds.sort(["Conversation_no", "Turn_no"])

        queries, corpus, qrels = {}, {}, {}
        history_by_conversation: dict[int, str] = {}
        for i, sample in enumerate(ds):
            conv_no = sample["Conversation_no"]
            history = history_by_conversation.get(conv_no, "")
            query_text = (
                f"{history}\nUser: {sample['Question']}"
                if history
                else f"User: {sample['Question']}"
            )

            doc_id = "doc:" + str(i)
            corpus[doc_id] = {
                "title": sample["Topic"] or "",
                "text": " ".join(sample["Context"]),
            }

            query_id = "query:" + str(i)
            queries[query_id] = query_text
            qrels[query_id] = {doc_id: 1}

            history_by_conversation[conv_no] = (
                f"{query_text}\nAssistant: {sample['Answer']}"
            )

        return corpus, queries, qrels
