from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SanskritShlokasClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SanskritShlokasClassification",
        description="This data set contains ~500 Shlokas  ",
        reference="https://github.com/goru001/nlp-for-sanskrit",
        dataset={
            "path": "bpHigh/iNLTK_Sanskrit_Shlokas_Dataset",
            "revision": "5a79d6472db143690c7ce6e974995d3610eee7f0",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2019-01-01", "2020-01-01"),
        eval_splits=["train", "validation"],
        eval_langs=["san-Deva"],
        main_score="accuracy",
        domains=["Religious", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
               @inproceedings{arora-2020-inltk,
           title = "i{NLTK}: Natural Language Toolkit for Indic Languages",
           author = "Arora, Gaurav",
           editor = "Park, Eunjeong L.  and
             Hagiwara, Masato  and
             Milajevs, Dmitrijs  and
             Liu, Nelson F.  and
             Chauhan, Geeticka  and
             Tan, Liling",
           booktitle = "Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)",
           month = nov,
           year = "2020",
           address = "Online",
           publisher = "Association for Computational Linguistics",
           url = "https://aclanthology.org/2020.nlposs-1.10",
           doi = "10.18653/v1/2020.nlposs-1.10",
           pages = "66--71",
           abstract = "We present iNLTK, an open-source NLP library consisting of pre-trained language models and out-of-the-box support for Data Augmentation, Textual Similarity, Sentence Embeddings, Word Embeddings, Tokenization and Text Generation in 13 Indic Languages. By using pre-trained models from iNLTK for text classification on publicly available datasets, we significantly outperform previously reported results. On these datasets, we also show that by using pre-trained models and data augmentation from iNLTK, we can achieve more than 95{\%} of the previous best performance by using less than 10{\%} of the training data. iNLTK is already being widely used by the community and has 40,000+ downloads, 600+ stars and 100+ forks on GitHub.",
           }
           """,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"Sloka": "text", "Class": "label"})
