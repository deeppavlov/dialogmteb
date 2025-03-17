from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


def combine_dialogs(row: dict) -> dict:
    row["dialog"] = "\n".join(row["dialog"])
    return row


class DailyDialogClassificationAct(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DailyDialogClassificationAct",
        description="",
        dataset={
            "path": "DeepPavlov/daily_dialog",
            "revision": "12f30df52a776875c2e334d9d051fa67afd15756",
        },
        reference="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        type="Classification",
        category=None,
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=[],
        sample_creation=None,
        bibtex_citation="""@misc{li2017dailydialogmanuallylabelledmultiturn,
      title={DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset}, 
      author={Yanran Li and Hui Su and Xiaoyu Shen and Wenjie Li and Ziqiang Cao and Shuzi Niu},
      year={2017},
      eprint={1710.03957},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1710.03957}, 
}""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.map(combine_dialogs)
        self.dataset = self.dataset.rename_columns(
            {"act_label": "label", "dialog": "text"}
        )


class DailyDialogClassificationEmotion(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DailyDialogClassificationEmotion",
        description="",
        dataset={
            "path": "DeepPavlov/daily_dialog",
            "revision": "12f30df52a776875c2e334d9d051fa67afd15756",
        },
        reference="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        type="Classification",
        category=None,
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=[],
        sample_creation=None,
        bibtex_citation="""@misc{li2017dailydialogmanuallylabelledmultiturn,
      title={DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset}, 
      author={Yanran Li and Hui Su and Xiaoyu Shen and Wenjie Li and Ziqiang Cao and Shuzi Niu},
      year={2017},
      eprint={1710.03957},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1710.03957}, 
}""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.map(combine_dialogs)
        self.dataset = self.dataset.rename_columns(
            {"emotion_label": "label", "dialog": "text"}
        )


if __name__ == "__main__":
    tasks = [
        DailyDialogClassificationAct(),
        DailyDialogClassificationEmotion(),
    ]
    import mteb
    evaluator = mteb.MTEB(tasks)
    model = mteb.get_model("minishlab/potion-base-2M")
    evaluator.run(
        model,
    )