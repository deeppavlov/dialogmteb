from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SkillOfMind(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SkillOfMind",
        description="The task is to understand which skills are the most relevant to use in this case",
        reference="https://huggingface.co/datasets/passing2961/multifaceted-skill-of-mind",
        dataset={
            "path": "DeepPavlov/multifaceted-skill-of-mind",
            "revision": "9b6081971002156832360e3fe82f84acc90beae2",
        },
        type="TextClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="lrap",
        date=("2024-11-01", "2024-11-30"),
        domains=["Spoken"],
        task_subtypes=["Object recognition"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{lee2024thanosenhancingconversationalagents,
      title={Thanos: Enhancing Conversational Agents with Skill-of-Mind-Infused Large Language Model}, 
      author={Young-Jun Lee and Dokyong Lee and Junyoung Youn and Kyeongjin Oh and Ho-Jin Choi},
      year={2024},
      eprint={2411.04496},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.04496}, 
}
""",
    )
