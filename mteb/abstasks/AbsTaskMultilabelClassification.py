from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder

from ..load_results.task_results import ScoresDict
from .AbsTaskClassification import AbsTaskClassification

logger = logging.getLogger(__name__)


def evaluate_classifier(
    embeddings_train: np.ndarray,
    y_train: np.ndarray,
    embeddings_test: np.ndarray,
    y_test: np.ndarray,
    classifier: ClassifierMixin,
) -> dict[str, float]:
    classifier = clone(classifier)
    classifier.fit(embeddings_train, y_train)
    y_pred = classifier.predict(embeddings_test)
    accuracy = classifier.score(embeddings_test, y_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    lrap = label_ranking_average_precision_score(y_test, y_pred)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "lrap": lrap,
    }


class AbsTaskMultilabelClassification(AbsTaskClassification):
    """Abstract class for multioutput classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        text: str
        label: list[list[int]]

    Attributes:
       samples_per_label: Number of samples to use pr. label. These samples are embedded and a classifier is fit using the labels and samples.

    """

    evaluator = KNeighborsClassifier(n_neighbors=5)

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: DatasetDict | Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> ScoresDict:
        train_split = dataset[self.train_split]
        eval_split = dataset[hf_split]

        scores = []
        # Bootstrap sample indices from training set for each experiment
        train_samples = []
        for _ in range(self.n_experiments):
            sample_indices, _ = self._undersample_data_indices(
                train_split["label"], self.samples_per_label, None
            )
            train_samples.append(sample_indices)
        # Encode all unique sentences at the indices
        unique_train_indices = list(set(itertools.chain.from_iterable(train_samples)))
        unique_train_dataset = train_split.select(unique_train_indices)

        _unique_train_embeddings = model.encode(
            DataLoader(unique_train_dataset),
            task_metadata=self.metadata,
            hf_split=self.train_split,
            hf_subset=hf_subset,
            **encode_kwargs,
        )
        unique_train_embeddings = dict(
            zip(unique_train_indices, _unique_train_embeddings)
        )
        # Stratified subsampling of test set to 2000 examples.
        test_dataset = eval_split
        try:
            if len(test_dataset) > 2000:
                split_dataset = eval_split.train_test_split(
                    test_size=2000, seed=42, stratify_by_column="label"
                )
                test_dataset = split_dataset["test"]
        except ValueError:
            logger.warning("Couldn't subsample, continuing with the entire test set.")

        X_test = model.encode(
            DataLoader(test_dataset),
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **encode_kwargs,
        )
        binarizer = MultiLabelBinarizer()
        y_test = binarizer.fit_transform(test_dataset["label"])

        for i_experiment, sample_indices in enumerate(train_samples):
            logger.info(
                "=" * 10
                + f" Experiment {i_experiment + 1}/{self.n_experiments} "
                + "=" * 10
            )
            X_train = np.stack([unique_train_embeddings[idx] for idx in sample_indices])
            y_train = train_split.select(sample_indices)["label"]
            y_train = binarizer.transform(y_train)
            scores_exp = evaluate_classifier(
                X_train, y_train, X_test, y_test, self.evaluator
            )
            scores.append(scores_exp)

        avg_scores: dict[str, Any] = {
            k: np.mean([s[k] for s in scores]) for k in scores[0].keys()
        }
        avg_scores["scores_per_experiment"] = scores

        return avg_scores

    def _undersample_data_indices(self, y, samples_per_label, idxs=None):
        """Undersample data to have samples_per_label samples of each label"""
        sample_indices = []
        if idxs is None:
            idxs = np.arange(len(y))
        self.np_rng.shuffle(idxs)
        label_counter = defaultdict(int)
        for i in idxs:
            if any((label_counter[label] < samples_per_label) for label in y[i]):
                sample_indices.append(i)
                for label in y[i]:
                    label_counter[label] += 1
        return sample_indices, idxs
