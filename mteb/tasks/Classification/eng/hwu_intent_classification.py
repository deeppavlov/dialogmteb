from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HWUIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HWUIntentClassification",
        description="",
        dataset={
            "path": "DeepPavlov/hwu_intent_classification",
            "revision": "050d2712be8b6f069a4350139c9c2d3ed7ce4aaf",
        },
        reference="https://huggingface.co/datasets/fathyshalab/atis_intents",
        type="Classification",
        category=None,
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        date=None,
        domains=None,
        task_subtypes=None,
        license="not specified",
        annotations_creators=None,
        dialect=[],
        sample_creation=None,
        bibtex_citation="""@misc{liu2019benchmarkingnaturallanguageunderstanding,
          title={Benchmarking Natural Language Understanding Services for building Conversational Agents}, 
          author={Xingkun Liu and Arash Eshghi and Pawel Swietojanski and Verena Rieser},
          year={2019},
          eprint={1903.05566},
          archivePrefix={arXiv},
          primaryClass={cs.CL},
          url={https://arxiv.org/abs/1903.05566}, 
    }""",
    )