from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class AmazonCounterfactualClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonCounterfactualClassification",
        dataset={
            "path": "mteb/amazon_counterfactual",
            "revision": "1f7e6a9d6fa6e64c53d146e428565640410c0df1",
        },
        description=(
            "A collection of Amazon customer reviews annotated for counterfactual detection pair classification."
        ),
        reference="https://arxiv.org/abs/2104.06893",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs={
            "en-ext": ["eng-Latn"],
            "en": ["eng-Latn"],
            "de": ["deu-Latn"],
            "ja": ["jpn-Jpan"],
        },
        main_score="accuracy",
        date=(
            "2018-01-01",
            "2021-12-31",
        ),  # Estimated range for the collection of Amazon reviews
        domains=["Reviews", "Written"],
        task_subtypes=["Counterfactual Detection"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{oneill-etal-2021-wish,
    title = "{I} Wish {I} Would Have Loved This One, But {I} Didn{'}t {--} A Multilingual Dataset for Counterfactual Detection in Product Review",
    author = "O{'}Neill, James  and
      Rozenshtein, Polina  and
      Kiryo, Ryuichi  and
      Kubota, Motoko  and
      Bollegala, Danushka",
    editor = "Moens, Marie-Francine  and
      Huang, Xuanjing  and
      Specia, Lucia  and
      Yih, Scott Wen-tau",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.568",
    doi = "10.18653/v1/2021.emnlp-main.568",
    pages = "7092--7108",
    abstract = "Counterfactual statements describe events that did not or cannot take place. We consider the problem of counterfactual detection (CFD) in product reviews. For this purpose, we annotate a multilingual CFD dataset from Amazon product reviews covering counterfactual statements written in English, German, and Japanese languages. The dataset is unique as it contains counterfactuals in multiple languages, covers a new application area of e-commerce reviews, and provides high quality professional annotations. We train CFD models using different text representation methods and classifiers. We find that these models are robust against the selectional biases introduced due to cue phrase-based sentence selection. Moreover, our CFD dataset is compatible with prior datasets and can be merged to learn accurate CFD models. Applying machine translation on English counterfactual examples to create multilingual data performs poorly, demonstrating the language-specificity of this problem, which has been ignored so far.",
}""",
        prompt="Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
    )

    samples_per_label = 32
