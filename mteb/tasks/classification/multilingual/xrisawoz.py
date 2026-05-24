from typing import Any

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class XRisaWoz(AbsTaskClassification):
    metadata = TaskMetadata(
        name="XRisaWoz",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/XRISAWOZ",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            "en": ["eng-Latn"],
            "fr": ["fra-Latn"],
            "enhi": ["hin-Deva", "eng-Latn"],
            "hi": ["hin-Deva"],
            "ko": ["kor-Hang"],
        },
        main_score="accuracy",
        date=("2020-01-01", "2020-01-01"),
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""
@misc{moradshahi2023xrisawozhighqualityendtoendmultilingual,
      title={X-RiSAWOZ: High-Quality End-to-End Multilingual Dialogue Datasets and Few-shot Agents}, 
      author={Mehrad Moradshahi and Tianhao Shen and Kalika Bali and Monojit Choudhury and Gaël de Chalendar and Anmol Goel and Sungkyun Kim and Prashant Kodali and Ponnurangam Kumaraguru and Nasredine Semmar and Sina J. Semnani and Jiwon Seo and Vivek Seshadri and Manish Shrivastava and Michael Sun and Aditya Yadavalli and Chaobin You and Deyi Xiong and Monica S. Lam},
      year={2023},
      eprint={2306.17674},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2306.17674}, 
}""",
        prompt=None,
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            row["label"] = row["domains"][0]
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )
