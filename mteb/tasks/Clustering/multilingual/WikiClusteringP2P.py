from __future__ import annotations

import itertools

import numpy as np
from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "bs": ["bos-Latn"],
    "ca": ["cat-Latn"],
    "cs": ["ces-Latn"],
    "da": ["dan-Latn"],
    "eu": ["eus-Latn"],
    "gv": ["glv-Latn"],
    "ilo": ["ilo-Latn"],
    "ku": ["kur-Latn"],
    "lv": ["lav-Latn"],
    "min": ["min-Latn"],
    "mt": ["mlt-Latn"],
    "sco": ["sco-Latn"],
    "sq": ["sqi-Latn"],
    "wa": ["wln-Latn"],
}


class WikiClusteringP2P(AbsTaskClustering):
    superseded_by = "WikiClusteringP2P.v2"
    metadata = TaskMetadata(
        name="WikiClusteringP2P",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).",
        reference="https://github.com/Rysias/wiki-clustering",
        dataset={
            "path": "ryzzlestrizzle/multi-wiki-clustering-p2p",
            "revision": "d4d92f8f28be71035be6a96bdfd4e200cf62faa8",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2001-01-15", "2024-04-15"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=None,  # None exists
    )


class WikiClusteringFastP2P(AbsTaskClusteringFast):
    max_document_to_embed = 2048
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="WikiClusteringP2P.v2",
        description="Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).",
        reference="https://github.com/Rysias/wiki-clustering",
        dataset={
            "path": "ryzzlestrizzle/multi-wiki-clustering-p2p",
            "revision": "d4d92f8f28be71035be6a96bdfd4e200cf62faa8",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2001-01-15", "2024-04-15"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",  # None exists
        adapted_from=["WikiClusteringP2P"],
    )

    def dataset_transform(self):
        ds = {}
        for lang in self.hf_subsets:
            labels = []
            sentences = []
            ds[lang] = {}
            lang_dict = {}
            for split in self.metadata.eval_splits:
                labels.extend(
                    itertools.chain.from_iterable(self.dataset[lang][split]["labels"])
                )
                sentences.extend(
                    itertools.chain.from_iterable(
                        self.dataset[lang][split]["sentences"]
                    )
                )

                # Remove sentences and labels with only 1 label example.
                unique_labels, counts = np.unique(labels, return_counts=True)
                solo_label_idx = np.where(counts == 1)
                solo_labels = unique_labels[solo_label_idx]
                is_solo = np.isin(labels, solo_labels)
                split_ds = Dataset.from_dict({"labels": labels, "sentences": sentences})
                if is_solo.any():
                    split_ds = split_ds.select(np.nonzero(is_solo == False)[0])  # noqa: E712
                lang_dict.update({split: split_ds})
            ds[lang] = DatasetDict(lang_dict)
        self.dataset = DatasetDict(ds)
        for lang in self.hf_subsets:
            self.dataset[lang] = self.stratified_subsampling(
                self.dataset[lang],
                self.seed,
                self.metadata.eval_splits,
                label="labels",
            )
