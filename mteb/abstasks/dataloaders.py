from __future__ import annotations

import logging
from collections import defaultdict

from datasets import (
    Features,
    Sequence,
    Value,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class RetrievalDataLoader:
    """This dataloader handles the dataloading for retrieval-oriented tasks, including standard retrieval, reranking, and instruction-based variants of the above.

    If the `hf_repo` is provided, the dataloader will fetch the data from the HuggingFace hub. Otherwise, it will look for the data in the specified `data_folder`.

    Required files include the corpus, queries, and qrels files. Optionally, the dataloader can also load instructions and top-ranked (for reranking) files.
    """

    def __init__(
        self,
        hf_repo: str,
        revision: str,
        trust_remote_code: bool = False,
        split: str = "test",
        config: str | None = None,
    ):
        self.revision = revision
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.instructions = None
        self.top_ranked = None
        self.hf_repo = hf_repo
        self.trust_remote_code = trust_remote_code
        self.split = split
        self.config = config

    def load(
        self,
    ) -> tuple[
        dict[str, dict[str, str]],  # corpus
        dict[str, str | list[str]],  # queries
        dict[str, dict[str, int]],  # qrels/relevant_docs
        dict[str, str | list[str]] | None,  # instructions (optional)
        dict[str, list[str]]
        | dict[str, dict[str, float]]
        | None,  # top_ranked (optional)
    ]:
        configs = get_dataset_config_names(
            self.hf_repo, self.revision, trust_remote_code=self.trust_remote_code
        )

        logger.info("Loading Corpus...")
        self._load_corpus(self.config)
        logger.info("Loaded %d %s Documents.", len(self.corpus), self.split.upper())
        logger.info("Doc Example: %s", self.corpus[0])

        logger.info("Loading Queries...")
        self._load_queries(self.config)

        if any(c.endswith("top_ranked") for c in configs):
            logger.info("Loading Top Ranked")
            self._load_top_ranked(self.config)
            logger.info(
                f"Top ranked loaded: {len(self.top_ranked) if self.top_ranked else 0}"
            )

        if any(c.endswith("instruction") for c in configs):
            logger.info("Loading Instructions")
            self._load_instructions(self.config)
            logger.info(
                f"Instructions loaded: {len(self.instructions) if self.instructions else 0}"
            )

        self._load_qrels(self.config)
        # filter queries with no qrels
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.qrels.map(qrels_dict_init)
        self.qrels = qrels_dict
        self.queries = self.queries.filter(lambda x: x["id"] in self.qrels)
        logger.info("Loaded %d %s Queries.", len(self.queries), self.split.upper())
        logger.info("Query Example: %s", self.queries[0])

        self.queries = {query["id"]: query["text"] for query in self.queries}
        self.corpus = {
            doc["id"]: (
                doc["title"] + " " + doc["text"]
                if len(doc.get("title", "")) > 0
                else doc["text"]
            )
            for doc in self.corpus
        }

        return self.corpus, self.queries, self.qrels, self.instructions, self.top_ranked

    def load_corpus(self, config: str | None = None) -> dict[str, dict[str, str]]:
        logger.info("Loading Corpus...")
        self._load_corpus(config)
        logger.info("Loaded %d %s Documents.", len(self.corpus))
        logger.info("Doc Example: %s", self.corpus[0])

        return self.corpus

    def get_split(self, config: str) -> str:
        splits = get_dataset_split_names(
            self.hf_repo,
            revision=self.revision,
            config_name=config,
        )
        if self.split in splits:
            return self.split
        if len(splits) == 1:
            return splits[0]

    def _load_corpus(self, config: str | None = None):
        config = f"{config}-corpus" if config is not None else "corpus"
        corpus_ds = load_dataset(
            self.hf_repo,
            config,
            split=self.get_split(config),
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        corpus_ds = corpus_ds.cast_column("_id", Value("string"))
        corpus_ds = corpus_ds.rename_column("_id", "id")
        corpus_ds = corpus_ds.remove_columns(
            [
                col
                for col in corpus_ds.column_names
                if col not in ["id", "text", "title"]
            ]
        )
        self.corpus = corpus_ds

    def _load_queries(self, config: str | None = None):
        config = f"{config}-queries" if config is not None else "queries"
        queries_ds = load_dataset(
            self.hf_repo,
            config,
            split=self.get_split(config),
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        queries_ds = queries_ds.cast_column("_id", Value("string"))
        queries_ds = queries_ds.rename_column("_id", "id")
        queries_ds = queries_ds.remove_columns(
            [col for col in queries_ds.column_names if col not in ["id", "text"]]
        )
        self.queries = queries_ds

    def _load_qrels(self, config: str | None = None):
        config = f"{config}-qrels" if config is not None else "default"

        qrels_ds = load_dataset(
            self.hf_repo,
            name=config,
            split=self.get_split(config),
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )

        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds

    def _load_top_ranked(self, config: str | None = None):
        config = f"{config}-top_ranked" if config is not None else "top_ranked"
        top_ranked_ds = load_dataset(
            self.hf_repo,
            config,
            split=self.get_split(config),
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )

        if (
            "query-id" in top_ranked_ds.column_names
            and "corpus-ids" in top_ranked_ds.column_names
        ):
            # is a {query-id: str, corpus-ids: list[str]} format
            top_ranked_ds = top_ranked_ds.cast_column("query-id", Value("string"))
            top_ranked_ds = top_ranked_ds.cast_column(
                "corpus-ids", Sequence(Value("string"))
            )
        else:
            # is a {"query-id": {"corpus-id": score}} format, let's change it
            top_ranked_ds = top_ranked_ds.map(
                lambda x: {"query-id": x["query-id"], "corpus-ids": list(x.keys())},
                remove_columns=[
                    col for col in top_ranked_ds.column_names if col != "query-id"
                ],
            )

        top_ranked_ds = top_ranked_ds.remove_columns(
            [
                col
                for col in top_ranked_ds.column_names
                if col not in ["query-id", "corpus-ids"]
            ]
        )
        top_ranked_ds = {tr["query-id"]: tr["corpus-ids"] for tr in top_ranked_ds}
        self.top_ranked = top_ranked_ds

    def _load_instructions(self, config: str | None = None):
        config = f"{config}-instruction" if config is not None else "instruction"
        instructions_ds = load_dataset(
            self.hf_repo,
            config,
            split=self.get_split(config),
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        instructions_ds = instructions_ds.cast_column("query-id", Value("string"))
        instructions_ds = instructions_ds.cast_column("instruction", Value("string"))
        instructions_ds = instructions_ds.remove_columns(
            [
                col
                for col in instructions_ds.column_names
                if col not in ["query-id", "instruction"]
            ]
        )
        self.instructions = instructions_ds
        self.instructions = {
            row["query-id"]: row["instruction"] for row in self.instructions
        }
