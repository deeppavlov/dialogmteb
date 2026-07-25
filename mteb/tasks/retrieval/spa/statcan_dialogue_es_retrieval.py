import json

import datasets

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLITS = ["dev", "test"]


class EsStatcanDialogueDatasetRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EsStatcanDialogueDatasetRetrieval",
        description="A Dataset for Retrieving Data Tables through Conversations with Genuine Intents, available in English and French.",
        dataset={
            "path": "DeepPavlov/statcan_dialog_es",
            "revision": "4b4bfc7176982bf35568680ecdee544f71f486cb",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=_EVAL_SPLITS,
        eval_langs=["spa-Latn"],
        main_score="recall_at_10",
        reference="https://huggingface.co/datasets/DeepPavlov/statcan_dialog_es",
        date=("2020-01-01", "2020-04-15"),
        domains=["Government", "Web", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval/blob/main/LICENSE.md",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["StatcanDialogueDatasetRetrieval"],
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        corpus_table = datasets.load_dataset(
            path, "corpus", split="spanish", revision=revision
        )
        corpus = {
            row["doc_id"]: {"title": row["title"], "text": row["doc"]}
            for row in corpus_table
        }

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in self.metadata.eval_splits:
            query_table = datasets.load_dataset(
                path, "queries_es", split=split, revision=revision
            )
            queries = {}
            relevant_docs = {}
            for row in query_table:
                query_id = row["query_id"]
                queries[query_id] = json.loads(row["query"])
                relevant_docs[query_id] = {row["doc_id"]: 1}

            self.corpus[split] = corpus
            self.queries[split] = queries
            self.relevant_docs[split] = relevant_docs

        self.data_loaded = True
