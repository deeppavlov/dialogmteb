from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from mteb.abstasks.TaskMetadata import DescriptiveStatistics
from mteb.encoder_interface import Encoder
from . import AbsTaskClassification

from ..evaluation.evaluators import (
    logRegClassificationEvaluator,
)
from ..load_results.task_results import HFSubset, ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class ClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Classification

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        number_texts_intersect_with_train: Number of texts in the train split

        min_text_length: Minimum length of text
        average_text_length: Average length of text
        max_text_length: Maximum length of text
        unique_texts: Number of unique texts

        min_labels_per_text: Minimum number of labels per text
        average_label_per_text: Average number of labels per text
        max_labels_per_text: Maximum number of labels per text
        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    num_samples: int
    number_of_characters: int
    number_texts_intersect_with_train: int | None

    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_texts: int

    min_labels_per_text: int
    average_label_per_text: float
    max_labels_per_text: int
    unique_labels: int
    labels: dict[str, dict[str, int]]


class AbsTaskDST(AbsTaskClassification):
    evaluator = logRegClassificationEvaluator
    abstask_prompt = "Classify user passages."
    samples_per_label: int = 8
    n_experiments: int = 10
    k: int = 3
    train_split = "train"
    classification_columns: list[str]

    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        scores = {}
        hf_subsets = self.hf_subsets
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            logger.info(
                f"Task: {self.metadata.name}, split: {split}, subset: {hf_subset}. Running..."
            )

            if hf_subset not in self.dataset and hf_subset == "default":
                ds = self.dataset
            else:
                ds = self.dataset[hf_subset]
            scores[hf_subset] = self._evaluate_subset(
                model,
                ds,
                hf_split=split,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
            self._add_main_score(scores[hf_subset])

        return scores

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: DatasetDict | Dataset,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> ScoresDict:
        train_split = dataset[self.train_split]
        eval_split = dataset[hf_split]
        params = {"k": self.k}
        params.update(kwargs)
        total_scores = {}

        for column in tqdm(self.classification_columns):
            current_train_split = train_split.rename_column(
                column, "label"
            )
            current_eval_split = eval_split.rename_column(
                column, "label"
            )

            scores = []
            test_cache, idxs = (
                None,
                None,
            )  # we store idxs to make the shuffling reproducible

            for i in range(self.n_experiments):
                logger.info(
                    "=" * 10 + f" Experiment {i + 1}/{self.n_experiments} " + "=" * 10
                )
                # Bootstrap `self.samples_per_label` samples per label for each split
                train_dataset, idxs = self._undersample_data(
                    current_train_split,
                    self.samples_per_label,
                    idxs,
                )

                evaluator = self.evaluator(
                    train_dataset,
                    current_eval_split,
                    task_metadata=self.metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    **params,
                )
                scores_exp, test_cache = evaluator(
                    model, encode_kwargs=encode_kwargs, test_cache=test_cache
                )
                scores.append(scores_exp)

            avg_scores: dict[str, Any] = {
                k: np.mean([s[k] for s in scores]) for k in scores[0].keys()
            }
            avg_scores["scores_per_experiment"] = scores
            total_scores[column] = avg_scores
        for metric in ["f1", "accuracy"]:
            total_scores[metric] = np.mean(
                [total_scores[column][metric] for column in self.classification_columns]
            )
        return total_scores

    def _undersample_data(
        self, dataset: Dataset, samples_per_label: int, idxs=None
    ) -> tuple[Dataset, list[int]]:
        """Undersample data to have `samples_per_label` samples of each label.

        Args:
            dataset: Hugging Face `datasets.Dataset` containing "text" and "label".
            samples_per_label: Number of samples per label to retain.
            idxs: Optional indices to shuffle and sample from.

        Returns:
            A new Dataset containing undersampled examples.
            The shuffled indices used for sampling.
        """
        if idxs is None:
            idxs = list(range(len(dataset)))

        rng_state = np.random.default_rng(self.seed)
        rng_state.shuffle(idxs)

        label_counter = defaultdict(int)
        sampled_idxs = []

        for i in idxs:
            label = dataset[i]["label"]
            if label_counter[label] < samples_per_label:
                sampled_idxs.append(i)
                label_counter[label] += 1

        return dataset.select(sampled_idxs), idxs

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> DescriptiveStatistics:
        pass
