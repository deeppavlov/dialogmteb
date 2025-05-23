from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "ru": ["rus-Cyrl"],
    "sv": ["swe-Latn"],
}


class OpusparcusPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="OpusparcusPC",
        dataset={
            "path": "GEM/opusparcus",
            "revision": "9e9b1f8ef51616073f47f306f7f47dd91663f86a",
            "trust_remote_code": True,
        },
        description="Opusparcus is a paraphrase corpus for six European language: German, English, Finnish, French, Russian, and Swedish. The paraphrases consist of subtitles from movies and TV shows.",
        reference="https://gem-benchmark.com/data_cards/opusparcus",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test.full", "validation.full"],
        eval_langs=_LANGUAGES,
        main_score="max_ap",
        date=("2013-01-01", "2015-12-31"),
        domains=["Spoken", "Spoken"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@misc{creutz2018open,
      title={Open Subtitles Paraphrase Corpus for Six Languages},
      author={Mathias Creutz},
      year={2018},
      eprint={1809.06142},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(
                lang=lang,
                quality=100,
                **self.metadata.dataset,
            )
            self.dataset_transform(lang)
        self.data_loaded = True

    def dataset_transform(self, lang):
        for split in self.dataset[lang]:
            # Renaming features
            labels = self.dataset[lang][split]["annot_score"]
            sent1 = self.dataset[lang][split]["input"]
            sent2 = self.dataset[lang][split]["target"]
            new_dict = {}
            # Labels are a score between 1.0 and 4.0, and we need binary classification
            labels = [
                0 if label < 2.5 else 1 if label > 2.5 else 2.5 for label in labels
            ]
            # Get neutral label to delete them
            neutral = [i for i, val in enumerate(labels) if val == 2.5]
            for i in sorted(neutral, reverse=True):
                del labels[i]
                del sent1[i]
                del sent2[i]
            new_dict["labels"] = [labels]
            new_dict["sentence1"] = [sent1]
            new_dict["sentence2"] = [sent2]
            self.dataset[lang][split] = new_dict
