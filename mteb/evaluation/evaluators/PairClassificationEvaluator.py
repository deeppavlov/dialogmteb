from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.encoder_interface import Encoder
from mteb.model_meta import ScoringFunction

from ...create_dataloaders import create_dataloader_from_texts
from ...similarity_functions import compute_pairwise_similarity
from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class PairClassificationEvaluator(Evaluator):
    """Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.
    The results are written in a CSV. If a CSV already exists, then values are appended.
    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    Args:
        sentences1: The first column of sentences
        sentences2: The second column of sentences
        labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
        name: Name for the output
        batch_size: Batch size used to compute embeddings
        write_csv: Write results to a CSV file
    """

    def __init__(
        self,
        sentences1,
        sentences2,
        labels,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert label == 0 or label == 1

    def __call__(
        self,
        model: Encoder,
        encode_kwargs: dict[str, Any],
    ):
        scores = self.compute_metrics(model, encode_kwargs=encode_kwargs)

        # Main score is the max of Average Precision (AP)
        main_score = max(scores[short_name]["ap"] for short_name in scores)
        scores["main_score"] = main_score
        return scores

    @staticmethod
    def _encode_unique_texts(
        all_texts: list[str],
        model: Encoder,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **encode_kwargs: Any,
    ):
        index_map, all_unique_texts, all_texts_indexes = {}, [], []
        for text in all_texts:
            text_hash = hash(text)
            if text_hash not in index_map:
                index_map[text_hash] = len(all_unique_texts)
                all_unique_texts.append(text)
            all_texts_indexes.append(index_map[text_hash])
        logger.warning(
            f"A total on {len(all_texts) - len(all_unique_texts)}/{len(all_texts)} duplicate texts were found during encoding. Only encoding unique text and duplicating embeddings across."
        )
        all_unique_texts_embs = np.asarray(
            model.encode(
                create_dataloader_from_texts(all_unique_texts),
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                **encode_kwargs,
            )
        )
        return all_unique_texts_embs[all_texts_indexes]

    def compute_metrics(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
    ):
        all_sentences = self.sentences1 + self.sentences2
        len_sentences1 = len(self.sentences1)
        embeddings = self._encode_unique_texts(
            all_sentences,
            model,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )
        embeddings1 = embeddings[:len_sentences1]
        embeddings2 = embeddings[len_sentences1:]

        logger.info("Computing similarity distances.")
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        similarity_scores = compute_pairwise_similarity(model, embeddings1, embeddings2)

        embeddings1_np = np.asarray(embeddings1)
        embeddings2_np = np.asarray(embeddings2)
        dot_scores = [
            np.dot(embeddings1_np[i], embeddings2_np[i])
            for i in range(len(embeddings1_np))
        ]

        logger.info("Computing metrics...")
        labels = np.asarray(self.labels)
        output_scores = {}
        max_scores = defaultdict(list)
        for short_name, name, scores, reverse in [
            [
                "similarity",
                "Model-Specified Similarity",
                similarity_scores,
                True,
            ],
            [ScoringFunction.COSINE.value, "Cosine-Similarity", cosine_scores, True],
            [
                ScoringFunction.MANHATTAN.value,
                "Manhattan-Distance",
                manhattan_distances,
                False,
            ],
            [
                ScoringFunction.EUCLIDEAN.value,
                "Euclidean-Distance",
                euclidean_distances,
                False,
            ],
            [ScoringFunction.DOT_PRODUCT.value, "Dot-Product", dot_scores, True],
        ]:
            metrics = self._compute_metrics(scores, labels, reverse)
            for metric_name, metric_value in metrics.items():
                output_scores[f"{short_name}_{metric_name}"] = metric_value
                max_scores[metric_name].append(metric_value)

        for metric in max_scores:
            if metric in ["f1", "ap", "f1", "precision", "recall", "accuracy"]:
                output_scores[f"max_{metric}"] = max(max_scores[metric])

        return output_scores

    @staticmethod
    def _compute_metrics(
        scores: np.ndarray, labels: np.ndarray, high_score_more_similar: bool
    ) -> dict[str, float]:
        """Compute the metrics for the given scores and labels.

        Args:
            scores: The similarity/dissimilarity scores for the pairs, specified as an array of shape (n_pairs, ).
            labels: The labels for the pairs, specified as an array of shape (n_pairs, ).
            high_score_more_similar: If true, then the higher the score, the more similar the pairs are.

        Returns:
            The metrics for the given scores and labels.
        """
        acc, acc_threshold = PairClassificationEvaluator.find_best_acc_and_threshold(
            scores, labels, high_score_more_similar
        )
        (
            f1,
            precision,
            recall,
            f1_threshold,
        ) = PairClassificationEvaluator.find_best_f1_and_threshold(
            scores, labels, high_score_more_similar
        )
        ap = PairClassificationEvaluator.ap_score(
            scores, labels, high_score_more_similar
        )

        return {
            "accuracy": float(acc),
            "accuracy_threshold": float(acc_threshold),
            "f1": float(f1),
            "f1_threshold": float(f1_threshold),
            "precision": float(precision),
            "recall": float(recall),
            "ap": float(ap),
        }

    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(np.array(labels) == 0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold

    @staticmethod
    def ap_score(scores, labels, high_score_more_similar: bool):
        return average_precision_score(
            labels, scores * (1 if high_score_more_similar else -1)
        )
