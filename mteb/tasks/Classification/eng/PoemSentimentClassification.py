from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PoemSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PoemSentimentClassification",
        description="Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.",
        reference="https://arxiv.org/abs/2011.02686",
        dataset={
            "path": "google-research-datasets/poem_sentiment",
            "revision": "329d529d875a00c47ec71954a1a96ae167584770",
            "trust_remote_code": True,
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1700-01-01", "1900-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-Latn-US", "en-Latn-GB"],
        sample_creation="found",
        bibtex_citation="""
@misc{sheng2020investigating,
      title={Investigating Societal Biases in a Poetry Composition System},
      author={Emily Sheng and David Uthus},
      year={2020},
      eprint={2011.02686},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("verse_text", "text")
