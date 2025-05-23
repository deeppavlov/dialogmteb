from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import torch
import tqdm
from scipy.stats import pearsonr, spearmanr

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.encoder_interface import Encoder
from mteb.similarity_functions import cos_sim, dot_score

from ...create_dataloaders import create_dataloader_from_texts
from .Evaluator import Evaluator

# if later than python 3.13 use typing module
if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


class SummarizationEvaluator(Evaluator):
    def __init__(
        self,
        human_summaries: list[list[str]],
        machine_summaries: list[list[str]],
        texts: list[str],
        gold_scores: list[list[float]],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ):
        """Summarization Evaluator

        Args:
        human_summaries: shape: (-1, num_human_summaries)
        machine_summaries: shape: (-1, num_machine_summaries)
        texts: shape: (-1,)
        gold_scores: shape: (-1, num_machine_summaries)
        task_metadata: Name of the task
        hf_split: Split of task
        hf_subset: Subset of task
        **kwargs: Additional arguments to pass to the Evaluator
        """
        super().__init__(**kwargs)
        self.human_summaries = human_summaries
        self.machine_summaries = machine_summaries
        self.texts = texts
        self.gold_scores = gold_scores
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
    ):
        cosine_spearman_scores = []
        cosine_pearson_scores = []
        dot_spearman_scores = []
        dot_pearson_scores = []
        pearson_scores = []
        spearman_scores = []

        # Get the human & machine summaries for the text in one go for all
        human_lens = [len(human_summaries) for human_summaries in self.human_summaries]
        machine_lens = [
            len(machine_summaries) for machine_summaries in self.machine_summaries
        ]

        logger.info("Encoding human summaries...")
        embs_human_summaries_all = model.encode(
            create_dataloader_from_texts(
                [
                    summary
                    for human_summaries in self.human_summaries
                    for summary in human_summaries
                ]
            ),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        logger.info("Encoding machine summaries...")
        embs_machine_summaries_all = model.encode(
            create_dataloader_from_texts(
                [
                    summary
                    for machine_summaries in self.machine_summaries
                    for summary in machine_summaries
                ]
            ),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        # Split the embeddings into the original human & machine summaries
        embs_human_summaries_all = np.split(
            embs_human_summaries_all, np.cumsum(human_lens)[:-1]
        )
        embs_machine_summaries_all = np.split(
            embs_machine_summaries_all, np.cumsum(machine_lens)[:-1]
        )

        for i, (embs_human_summaries, embs_machine_summaries) in tqdm.tqdm(
            enumerate(zip(embs_human_summaries_all, embs_machine_summaries_all)),
            desc="Scoring",
            total=len(self.human_summaries),
        ):
            cosine_pred_scores = []  # Predicted quality score for a summary
            dot_pred_scores = []  # Predicted quality score for a summary
            sim_scores = []
            human_scores = []  # Human score for a summary

            for emb_machine_summary, human_eval_score in zip(
                embs_machine_summaries, self.gold_scores[i]
            ):  # Iterate through all machine summaries + scores for a single sample
                cosine_scores = cos_sim(emb_machine_summary, embs_human_summaries)
                dot_scores = dot_score(emb_machine_summary, embs_human_summaries)

                _sim_score = [
                    float(model.similarity(emb_machine_summary, emb_human_summary))  # type: ignore
                    for emb_human_summary in embs_human_summaries
                ]
                sim_score = torch.tensor(_sim_score)

                cosine_max_score = torch.max(cosine_scores).item()
                dot_max_score = torch.max(dot_scores).item()
                sim_max_score = torch.max(sim_score).item()

                cosine_pred_scores.append(cosine_max_score)
                dot_pred_scores.append(dot_max_score)
                sim_scores.append(sim_max_score)
                human_scores.append(human_eval_score)

            if (
                (len(set(human_scores)) == 1)
                or (len(set(dot_pred_scores)) == 1)
                or (len(set(cosine_pred_scores)) == 1)
            ):
                logger.info(f"Skipping sample {i} due to equal scores")
                continue

            cosine_spearman_scores.append(
                spearmanr(human_scores, cosine_pred_scores).statistic
            )
            cosine_pearson_scores.append(
                pearsonr(human_scores, cosine_pred_scores).statistic
            )
            dot_spearman_scores.append(
                spearmanr(human_scores, dot_pred_scores).statistic
            )
            dot_pearson_scores.append(pearsonr(human_scores, dot_pred_scores).statistic)
            spearman_scores.append(spearmanr(human_scores, sim_scores).statistic)
            pearson_scores.append(pearsonr(human_scores, sim_scores).statistic)

        return {
            "pearson": np.mean(pearson_scores),
            "spearman": np.mean(spearman_scores),
            "cosine_spearman": np.mean(cosine_spearman_scores),
            "cosine_pearson": np.mean(cosine_pearson_scores),
            "dot_spearman": np.mean(dot_spearman_scores),
            "dot_pearson": np.mean(dot_pearson_scores),
        }


@deprecated(
    "The used Evaluator is deprecated due to a bug (https://github.com/embeddings-benchmark/mteb/issues/1156). Use the latest version of the dataset to use the latest version of the Evaluator."
)
class DeprecatedSummarizationEvaluator(Evaluator):
    """A deprecated version of the SummarizationEvaluator that contains the bug outlines in https://github.com/embeddings-benchmark/mteb/issues/1156.
    It is kept here to maintain compatibility with older versions of the benchmark, but we do not recommend using it.
    """

    def __init__(
        self,
        human_summaries=None,
        machine_summaries=None,
        texts=None,
        gold_scores=None,
        limit: int | None = None,
        task_metadata: TaskMetadata | None = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        **kwargs,
    ):
        # human_summaries shape: (None, num_human_summaries)
        # machine_summaries shape: (None, num_machine_summaries)
        # gold scores shape: (None, num_machine_summaries)
        # texts: (None,)
        super().__init__(**kwargs)
        if limit is not None:
            human_summaries = human_summaries[:limit]
            machine_summaries = machine_summaries[:limit]
            gold_scores = gold_scores[:limit]
            texts = texts[:limit]
        self.human_summaries = human_summaries
        self.machine_summaries = machine_summaries
        self.texts = texts
        self.gold_scores = gold_scores
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
    ):
        cosine_spearman_scores = []
        cosine_pearson_scores = []
        dot_spearman_scores = []
        dot_pearson_scores = []
        pearson_scores = []
        spearman_scores = []

        # Get the human & machine summaries for the text in one go for all
        human_lens = [len(human_summaries) for human_summaries in self.human_summaries]
        machine_lens = [
            len(machine_summaries) for machine_summaries in self.machine_summaries
        ]

        logger.info("Encoding human summaries...")
        embs_human_summaries_all = model.encode(
            create_dataloader_from_texts(
                [
                    summary
                    for human_summaries in self.human_summaries
                    for summary in human_summaries
                ]
            ),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        logger.info("Encoding machine summaries...")
        embs_machine_summaries_all = model.encode(
            create_dataloader_from_texts(
                [
                    summary
                    for machine_summaries in self.machine_summaries
                    for summary in machine_summaries
                ]
            ),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        # Split the embeddings into the original human & machine summaries
        embs_human_summaries_all = np.split(
            embs_human_summaries_all, np.cumsum(human_lens)[:-1]
        )
        embs_machine_summaries_all = np.split(
            embs_machine_summaries_all, np.cumsum(machine_lens)[:-1]
        )

        for i, (embs_human_summaries, embs_machine_summaries) in tqdm.tqdm(
            enumerate(zip(embs_human_summaries_all, embs_machine_summaries_all)),
            desc="Scoring",
            total=len(self.human_summaries),
        ):
            cosine_pred_scores = []  # Predicted quality score for a summary
            dot_pred_scores = []  # Predicted quality score for a summary
            sim_scores = []
            human_scores = []  # Human score for a summary

            for emb_machine_summary, human_eval_score in zip(
                embs_machine_summaries, self.gold_scores[i]
            ):  # Iterate through all machine summaries + scores for a single sample
                cosine_scores = cos_sim(emb_machine_summary, embs_human_summaries)
                dot_scores = dot_score(emb_machine_summary, embs_human_summaries)

                # Pairwise similarity
                _sim_score = [
                    float(model.similarity(emb_machine_summary, emb_human_summary))  # type: ignore
                    for emb_human_summary in embs_human_summaries
                ]
                sim_score = torch.tensor(_sim_score)

                cosine_max_score = torch.max(cosine_scores).item()
                dot_max_score = torch.max(dot_scores).item()
                sim_max_score = torch.max(sim_score).item()

                cosine_pred_scores.append(cosine_max_score)
                dot_pred_scores.append(dot_max_score)
                sim_scores.append(sim_max_score)
                human_scores.append(human_eval_score)

            if (
                (len(set(human_scores)) == 1)
                or (len(set(dot_pred_scores)) == 1)
                or (len(set(cosine_pred_scores)) == 1)
            ):
                logger.info(f"Skipping sample {i} due to equal scores")
                continue

            cosine_spearman_scores.append(spearmanr(human_scores, cosine_pred_scores))
            cosine_pearson_scores.append(pearsonr(human_scores, cosine_pred_scores))
            dot_spearman_scores.append(spearmanr(human_scores, dot_pred_scores))
            dot_pearson_scores.append(pearsonr(human_scores, dot_pred_scores))
            spearman_scores.append(spearmanr(human_scores, sim_scores))
            pearson_scores.append(pearsonr(human_scores, sim_scores))

        return {
            "pearson": np.mean(pearson_scores),
            "spearman": np.mean(spearman_scores),
            "cosine_spearman": np.mean(cosine_spearman_scores),
            "cosine_pearson": np.mean(cosine_pearson_scores),
            "dot_spearman": np.mean(dot_spearman_scores),
            "dot_pearson": np.mean(dot_pearson_scores),
        }
