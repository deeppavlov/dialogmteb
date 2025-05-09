from __future__ import annotations

import itertools
import logging
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import sklearn
import sklearn.cluster
from datasets import Dataset, DatasetDict
from sklearn.metrics.cluster import v_measure_score
from torch.utils.data import DataLoader

from mteb.abstasks.TaskMetadata import DescriptiveStatistics
from mteb.encoder_interface import Encoder

from ..load_results.task_results import HFSubset
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


MultilingualDataset = dict[HFSubset, DatasetDict]


def evaluate_clustering_bootstrapped(
    embeddings: np.ndarray,
    labels: list[list[str]],
    n_clusters: int,
    cluster_size: int,
    kmean_batch_size: int,
    max_depth: int | None,
    rng_state: random.Random = random.Random(),
) -> dict[str, list[float]]:
    """Bootstrapped evaluation of clustering performance using V-measure.

    The bootstrapping is done by sampling N samples from the corpus and clustering them. It is done without replacement to get a diverse set of
    samples.
    """
    v_measures = defaultdict(list)
    if max_depth is not None:
        max_depth = min(max_depth, max(map(len, labels)))
    else:
        max_depth = max(map(len, labels))
    # Evaluate on each level til max depth
    for i_level in range(max_depth):
        level_labels = []
        # Assign -1 to gold label if the level is not there
        for label in labels:
            if len(label) > i_level:
                level_labels.append(label[i_level])
            else:
                level_labels.append(-1)
        level_labels = np.array(level_labels)
        valid_idx = np.array(
            [level_label != -1 for level_label in level_labels]
        )  # Could be level_labels != -1 but fails with FutureWarning: elementwise comparison failed
        level_labels = level_labels[valid_idx]
        level_embeddings = embeddings[valid_idx]
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=np.unique(level_labels).size,
            batch_size=kmean_batch_size,
            n_init="auto",
        )
        for _ in range(n_clusters):
            # sample N samples from the corpus with replacement
            n_embeddings = len(level_embeddings)
            cluster_indices = rng_state.choices(range(n_embeddings), k=cluster_size)

            _embeddings = level_embeddings[cluster_indices]
            _labels = level_labels[cluster_indices]
            cluster_assignment = clustering_model.fit_predict(_embeddings)
            v_measure = v_measure_score(_labels, cluster_assignment)
            v_measures[f"Level {i_level}"].append(v_measure)

    return v_measures


class ClusteringFastDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Clustering

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.

        min_text_length: Minimum length of text
        average_text_length: Average length of text
        max_text_length: Maximum length of text
        unique_texts: Number of unique texts

        min_labels_per_text: Minimum number of labels per text
        average_labels_per_text: Average number of labels per text
        max_labels_per_text: Maximum number of labels per text
        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    num_samples: int
    number_of_characters: int

    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_texts: int

    min_labels_per_text: int
    average_labels_per_text: float
    max_labels_per_text: int
    unique_labels: int
    labels: dict[str, dict[str, int]]


class AbsTaskClusteringFast(AbsTask):
    """Abstract class for Clustering tasks.

    This class embeds the corpus sentences then samples N samples from the corpus and clusters them.
    The similarity then is calculated using the V-measure metric, which is invariant to the permutation of the labels.
    This approach is then repeated K times.

    There are two ways to specify how a dataset is downsampled:
        - max_document_to_embe (int): default to None
        - max_fraction_of_documents_to_embed (float): default to 4%.
    If both parameters are set to None, no downsampling is done in self._evaluate_subset().
    Only one of these two parameters can be not None at the same time.

    If the clustering is hierarchical, and more than one label is specified in order for each observation,
    V-measures are calculated in the outlined way on each of the levels separately.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset.
    It must contain the following columns:
        sentences: list[str]
        labels: list[str] | list[list[str]]
    """

    max_fraction_of_documents_to_embed: float | None = 0.04
    max_document_to_embed: int | None = None
    max_documents_per_cluster: int = 16_384
    n_clusters: int = 10
    k_mean_batch_size: int = 512
    max_depth = None
    abstask_prompt = "Identify categories in user passages."

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, float | dict[str, list[float]]]:
        if (
            self.max_document_to_embed is not None
            and self.max_fraction_of_documents_to_embed is not None
        ):
            raise Exception(
                "Both max_document_to_embed and max_fraction_of_documents_to_embed are set. Please only set one."
            )

        if (
            self.max_document_to_embed is None
            and self.max_fraction_of_documents_to_embed is None
        ):
            downsampled_dataset = dataset
        else:
            if self.max_fraction_of_documents_to_embed is not None:
                max_documents_to_embed = int(
                    self.max_fraction_of_documents_to_embed * len(dataset)
                )
            else:
                max_documents_to_embed = self.max_document_to_embed

            max_documents_to_embed = min(len(dataset), max_documents_to_embed)  # type: ignore
            example_indices = self.rng_state.sample(
                range(len(dataset)), k=max_documents_to_embed
            )
            downsampled_dataset = dataset.select(example_indices)  # type: ignore

        downsampled_dataset = downsampled_dataset.rename_column(
            original_column_name="sentences", new_column_name="text"
        )
        embeddings = model.encode(
            DataLoader(downsampled_dataset),
            task_metadata=self.metadata,
            hf_subset=hf_subset,
            hf_split=hf_split,
            **encode_kwargs,
        )

        labels = []
        for label in downsampled_dataset["labels"]:
            if not isinstance(label, list):
                label = [label]
            labels.append(label)

        all_v_scores = evaluate_clustering_bootstrapped(
            embeddings,
            labels,
            n_clusters=self.n_clusters,
            cluster_size=self.max_documents_per_cluster,
            kmean_batch_size=self.k_mean_batch_size,
            max_depth=self.max_depth,
            rng_state=self.rng_state,
        )
        v_measures = list(itertools.chain.from_iterable(all_v_scores.values()))

        mean_v_measure = np.mean(v_measures)
        v_std = np.std(v_measures)
        scores = {
            "v_measures": all_v_scores,
            "v_measure": float(mean_v_measure),
            "v_measure_std": v_std,
        }
        self._add_main_score(scores)
        return scores

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ClusteringFastDescriptiveStatistics:
        if hf_subset:
            sentences = self.dataset[hf_subset][split]["sentences"]
            labels = self.dataset[hf_subset][split]["labels"]
        elif compute_overall:
            sentences = []
            labels = []
            for hf_subset in self.metadata.eval_langs:
                sentences.extend(self.dataset[hf_subset][split]["sentences"])
                labels.extend(self.dataset[hf_subset][split]["labels"])
        else:
            sentences = self.dataset[split]["sentences"]
            labels = self.dataset[split]["labels"]

        text_len = [len(t) for t in sentences]
        total_text_len = sum(text_len)
        total_labels = []
        for label in labels:
            if isinstance(label, list):
                total_labels.extend(label)
            else:
                total_labels.append(label)
        label_counter = Counter(total_labels)
        return ClusteringFastDescriptiveStatistics(
            num_samples=len(sentences),
            number_of_characters=total_text_len,
            min_text_length=min(text_len),
            average_text_length=total_text_len / len(sentences),
            max_text_length=max(text_len),
            unique_texts=len(set(text_len)),
            min_labels_per_text=min(label_counter.values()),
            average_labels_per_text=len(total_labels) / len(sentences),
            max_labels_per_text=max(label_counter.values()),
            unique_labels=len(label_counter),
            labels={
                str(label): {
                    "count": value,
                }
                for label, value in label_counter.items()
            },
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        self._upload_dataset_to_hub(repo_name, ["sentences", "labels"])


def convert_to_fast(
    dataset: DatasetDict, seed: int, max_size: int = 100_000
) -> DatasetDict:
    """Converts a clustering dataset to a fast version. This concats the cluster into two columns, sentences and labels.
    It additionally downsamples the dataset to max_size.
    """
    rng_state = random.Random(seed)

    ds = {}
    for split in dataset:
        sent_set = set()
        labels = []
        sentences = []
        n_clusters = len(dataset[split])
        all_labels_set = set(itertools.chain.from_iterable(dataset[split]["labels"]))
        for i in range(n_clusters):
            lab = dataset[split]["labels"][i]
            sents = dataset[split]["sentences"][i]

            # check that it is the same distribution
            row_label_set = set(lab)
            assert row_label_set.issubset(all_labels_set), (
                "The clusters are not sampled from the same distribution as they have different labels."
            )

            for l, s in zip(lab, sents):
                if s not in sent_set:
                    labels.append(l)
                    sentences.append(s)
                    sent_set.add(s)  # ensuring no duplicates

        ds[split] = Dataset.from_dict({"sentences": sentences, "labels": labels})

        if len(ds[split]) > max_size:
            idxs = rng_state.sample(range(len(ds[split])), max_size)
            ds[split] = ds[split].select(idxs)

    return DatasetDict(ds)


def check_label_distribution(ds: DatasetDict) -> None:
    """For older clustering dataset versions.
    ds is a DatasetDict at the split level
    """
    n_clusters = len(ds)
    if n_clusters > 50:
        return
    all_labels_set = set(itertools.chain.from_iterable(ds["labels"]))

    for i in range(n_clusters):
        lab = ds["labels"][i]

        # check that it is the same distribution
        row_label_set = set(lab)
        assert row_label_set.issubset(all_labels_set), (
            "The clusters are not sampled from the same distribution as they have different labels."
        )
