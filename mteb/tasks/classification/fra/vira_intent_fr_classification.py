from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FrViraIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrViraIntentClassification",
        description="Chatbot-delivered COVID-19 vaccine communication message preferences of young adults and public health workers in urban American communities: qualitative study",
        dataset={
            "path": "DeepPavlov/vira-intents-live_fr",
            "revision": "430391e1240047b603e2e33aab4a446c25769140",
        },
        reference="https://huggingface.co/datasets/ibm-research/vira-intents-live",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["val", "test"],
        eval_langs=["fra-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2022-07-06"),
        domains=["Medical"],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@article{weeks2022chatbot,
  author = {Weeks, Rose and Cooper, Lyra and Sangha, Pooja and Sedoc, João and White, Sydney and Toledo, Assaf and Gretz, Shai and Lahav, Dan and Martin, Nina and Michel, Alexandra and others},
  journal = {Journal of medical Internet research},
  number = {7},
  pages = {e38418},
  publisher = {JMIR Publications Toronto, Canada},
  title = {Chatbot-delivered COVID-19 vaccine communication message preferences of young adults and public health workers in urban American communities: qualitative study},
  volume = {24},
  year = {2022},
}
""",
        adapted_from=["ViraIntentClassification"],
    )
