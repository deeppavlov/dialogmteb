from typing import Any

from mteb.abstasks import AbsTaskMultilabelClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SkillOfMind(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="SkillOfMind",
        description="The task is to understand which skills are the most relevant to use in this case",
        reference="https://huggingface.co/datasets/passing2961/multifaceted-skill-of-mind",
        dataset={
            "path": "DeepPavlov/multifaceted-skill-of-mind",
            "revision": "e19199040b03e41b02578e24d906e5af7394233a",
        },
        type="MultilabelClassification",
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

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["text"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            row["text"] = text
            return row

        for subset in self.dataset:
            self.dataset[subset] = (
                self.dataset[subset]
                .map(
                    process_history,
                )
                .select_columns(["text", "label"])
            )
