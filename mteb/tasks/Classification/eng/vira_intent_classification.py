from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ViraIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ViraIntentClassification",
        description="",
        dataset={
            "path": "ibm-research/vira-intents-live",
            "revision": "1f8d799c0974a1eec9499eb68a6a4c1092d4477d",
            # TODO in their repo spits are broken
        },
        reference="https://huggingface.co/datasets/ibm-research/vira-intents-live",
        type="Classification",
        category=None,
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license="not specified",
        annotations_creators=None,
        dialect=[],
        sample_creation=None,
        bibtex_citation="""@article{weeks2022chatbot,
          title={Chatbot-delivered COVID-19 vaccine communication message preferences of young adults and public health workers in urban American communities: qualitative study},
          author={Weeks, Rose and Cooper, Lyra and Sangha, Pooja and Sedoc, Jo{\~a}o and White, Sydney and Toledo, Assaf and Gretz, Shai and Lahav, Dan and Martin, Nina and Michel, Alexandra and others},
          journal={Journal of medical Internet research},
          volume={24},
          number={7},
          pages={e38418},
          year={2022},
          publisher={JMIR Publications Toronto, Canada}
        }
        """,
    )
