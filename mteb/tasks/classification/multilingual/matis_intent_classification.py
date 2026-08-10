from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MultilingualAtisIntentClassification(AbsTaskClassification):
    input_column_name = "text_gemma"

    metadata = TaskMetadata(
        name="MultilingualAtisIntentClassification",
        description="The ATIS Spoken Language Systems Pilot Corpus",
        dataset={
            "path": "DeepPavlov/atis_intent_classification_translated",
            "revision": "4e3114b24ebdfcb9d6fd44d6490ecd144b4b96e7",
        },
        reference="https://huggingface.co/datasets/fathyshalab/atis_intents",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "fr": ["fra-Latn"],
            "es": ["spa-Latn"],
        },
        main_score="accuracy",
        date=("1990-01-01", "1990-01-01"),
        domains=["Spoken"],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
""",
    )


class MultilingualBanking77Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualBanking77Classification",
        description="Dataset composed of online banking queries annotated with their corresponding intents.",
        reference="https://arxiv.org/abs/2003.04807",
        dataset={
            "path": "DeepPavlov/banking77-translated",
            "revision": "48246f6c99b06096d9dbd7fcbaf273c6d82df2f0",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "fr": ["fra-Latn"],
            "es": ["spa-Latn"],
        },
        main_score="accuracy",
        date=(
            "2019-01-01",
            "2019-12-31",
        ),  # Estimated range for the collection of queries
        domains=["Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
""",
        prompt="Given a online banking query, find the corresponding intents",
    )


class MultilingualViraIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualViraIntentClassification",
        description="Chatbot-delivered COVID-19 vaccine communication message preferences of young adults and public health workers in urban American communities: qualitative study",
        dataset={
            "path": "DeepPavlov/vira-intents-live",
            "revision": "a4141f65d270c89e13d269099aa9a6f188fa09f1",
        },
        reference="https://huggingface.co/datasets/ibm-research/vira-intents-live",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["val", "test"],
        eval_langs={
            "fr": ["fra-Latn"],
            "es": ["spa-Latn"],
        },
        main_score="accuracy",
        date=("2020-01-01", "2022-07-06"),
        domains=["Medical"],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.dataset = {
            "es": load_dataset("DeepPavlov/vira_intents_live_es").rename_column(
                "spanish_translated", "text"
            ),
            "fr": load_dataset("DeepPavlov/vira-intents-live_fr"),
        }
        self.data_loaded = True


class MultilingualHWUIntentClassification(AbsTaskClassification):
    input_column_name = "text_gemma"
    metadata = TaskMetadata(
        name="MultilingualHWUIntentClassification",
        description="",
        dataset={
            "path": "DeepPavlov/hwu64-translated",
            "revision": "8b2a047641347e51340269c2a14216902fe8702d",
        },
        reference="https://arxiv.org/abs/1903.05566",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "fr": ["fra-Latn"],
            "es": ["spa-Latn"],
        },
        main_score="f1",
        date=("2019-03-26", "2019-03-26"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
""",
    )


class MultilingualClincIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualClincIntentClassification",
        description="Task-oriented dialog systems need to know when a query falls outside their range of supported intents, but current text classification corpora only define label sets that cover every example",
        dataset={
            "path": "DeepPavlov/clinc_oos-translated",
            "revision": "132dc0a781b6ca8fcfd720d69641d937e24c2642",
        },
        reference="https://huggingface.co/datasets/clinc/clinc_oos",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs={
            "es-gemma-small": ["spa-Latn"],
            "es-gemma-plus": ["spa-Latn"],
            "es-gemma-imbalanced": ["spa-Latn"],
            "fr-gemma-small": ["fra-Latn"],
            "fr-gemma-plus": ["fra-Latn"],
            "fr-gemma-imbalanced": ["fra-Latn"],
        },
        main_score="accuracy",
        date=("2019-01-01", "2019-01-01"),
        domains=["Financial", "Web", "Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
""",
    )


class MultilingualXRisaWoz(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualXRisaWoz",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/xrisawoz-translated",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma-en": ["spa-Latn"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
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


def combine_dialogs(row: dict) -> dict:
    row["dialog"] = "\n".join(row["dialog"])
    return row


class MultilingualDailyDialogClassificationAct(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualDailyDialogClassificationAct",
        description="",
        dataset={
            "path": "DeepPavlov/daily_dialog-translated",
            "revision": "main",
        },
        reference="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="accuracy",
        date=("2017-07-11", "2017-07-11"),
        domains=["Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{li2017dailydialogmanuallylabelledmultiturn,
  archiveprefix = {arXiv},
  author = {Yanran Li and Hui Su and Xiaoyu Shen and Wenjie Li and Ziqiang Cao and Shuzi Niu},
  eprint = {1710.03957},
  primaryclass = {cs.CL},
  title = {DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
  url = {https://arxiv.org/abs/1710.03957},
  year = {2017},
}
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any):
        self.dataset = {
            key: dataset.map(combine_dialogs).rename_columns(
                {"act_label": "label", "dialog": "text"}
            )
            for key, dataset in self.dataset.items()
        }


class MultilingualDailyDialogClassificationEmotion(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualDailyDialogClassificationEmotion",
        description="",
        dataset={
            "path": "DeepPavlov/daily_dialog-translated",
            "revision": "main",
        },
        reference="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="accuracy",
        date=("2017-07-11", "2017-07-11"),
        domains=["Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{li2017dailydialogmanuallylabelledmultiturn,
  archiveprefix = {arXiv},
  author = {Yanran Li and Hui Su and Xiaoyu Shen and Wenjie Li and Ziqiang Cao and Shuzi Niu},
  eprint = {1710.03957},
  primaryclass = {cs.CL},
  title = {DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
  url = {https://arxiv.org/abs/1710.03957},
  year = {2017},
}
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any):
        self.dataset = {
            key: dataset.map(combine_dialogs).rename_columns(
                {"emotion_label": "label", "dialog": "text"}
            )
            for key, dataset in self.dataset.items()
        }


class MultilingualStatcanDialogueDatasetRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualStatcanDialogueDatasetRetrieval",
        description="A Dataset for Retrieving Data Tables through Conversations with Genuine Intents, available in English and French.",
        dataset={
            "path": "DeepPavlov/statcan-translated",
            "revision": "8ee9c43cb762d1337863913ba975e2bf9d39669d",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev", "test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
        },
        main_score="recall_at_10",
        reference="https://mcgill-nlp.github.io/statcan-dialogue-dataset/",
        date=("2020-01-01", "2020-04-15"),
        domains=["Government", "Web", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval/blob/main/LICENSE.md",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
""",
    )


class MultilingualFaithDialRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualFaithDialRetrieval",
        dataset={
            "path": "DeepPavlov/faithdial-translated",
            "revision": "e42d09175ef2b34219758575c9a09beadbc6caee",
        },
        reference="https://mcgill-nlp.github.io/FaithDial",
        description=(
            "FaithDial is a faithful knowledge-grounded dialogue benchmark."
            + "It was curated by asking annotators to amend hallucinated utterances in Wizard of Wikipedia (WoW). "
            + "It consists of conversation histories along with manually labelled relevant passage. "
            + "For the purpose of retrieval, we only consider the instances marked as 'Edification' in the VRM field, "
            + "as the gold passage associated with these instances is non-ambiguous."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-03-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
""",
    )


class MultilingualMultiWoz21(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualMultiWoz21",
        description="",
        reference="https://arxiv.org/abs/1810.00278",
        dataset={
            "path": "DeepPavlov/multiwoz-translated",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )
    label_column_name = "topic"

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
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class MultilingualAirDialogueClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualAirDialogueClassification",
        description="AirDialogue is a dataset of goal-oriented customer-agent conversations focused on booking flights under various travel restrictions.",
        dataset={
            "path": "DeepPavlov/air_dialogue-translated",
            "revision": "main",
        },
        reference="https://huggingface.co/datasets/google/air_dialogue",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="f1",
        date=("2018-01-01", "2022-06-07"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
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
                    num_proc=num_proc,
                )
                .select_columns(["text", "label"])
            )


class MultilingualCanard(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualCanard",
        description="canard",
        reference=None,
        dataset={
            "path": "DeepPavlov/canard-translated",
            "revision": "main",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="ndcg_at_10",
        date=None,
        domains=[],
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""

""",
    )


class MultilingualMantis(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualMantisClassification",
        description="Mantis",
        dataset={
            "path": "DeepPavlov/mantis-translated",
            "revision": "main",
        },
        reference=None,
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="f1",
        date=None,
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["dialog"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['message']}\n"
                    else:
                        text += f"Assistant: {entry['message']}\n"
            row["text"] = text
            return row

        for subset in self.dataset:
            self.dataset[subset] = (
                self.dataset[subset]
                .map(
                    process_history,
                )
                .rename_column("category", "label")
            )


class MultilingualWiardOfWikipedia(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualWiardOfWikipedia",
        description="WiardOfWikipedia",
        reference=None,
        dataset={
            "path": "DeepPavlov/wow-translated",
            "revision": "main",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="ndcg_at_10",
        date=None,
        domains=[],
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""

""",
    )


class MultilingualQRECC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="MultilingualQRECC",
        description="QRECC.",
        reference=None,
        dataset={
            "path": "DeepPavlov/qrecc-translated",
            "revision": "main",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="max_ap",
        date=None,
        domains=[],
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def transform(example: dict) -> dict:
            context_str = ""
            for replic in example["context"]:
                if replic["role"] == "user":
                    context_str += "User: " + replic["content"] + " "
                else:
                    context_str += "Assistant: " + replic["content"] + " "
            context_str += example["question"]
            return {
                "sentence1": context_str,
                "sentence2": example["rewrite"],
                "labels": 1,
            }

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                transform, num_proc=num_proc
            )


class MultilingualAbgCosQA(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualAbgCosQA",
        description="AbgCosQA",
        dataset={
            "path": "DeepPavlov/coqa_abg-translated",
            "revision": "main",
        },
        reference=None,
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="f1",
        date=None,
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            full_text = row["story"] + " "
            for turn in row["history_turns"]:
                full_text += (
                    "User: " + turn["question"] + " Assistant: " + turn["answer"] + " "
                )
            full_text += (
                "User: "
                + row["target_turn"]["question"]
                + " Assistant: "
                + row["target_turn"]["answer"]
            )
            row["text"] = full_text
            row["label"] = row["ambiguity"] == "ambiguous"
            return row

        for subset in self.dataset:
            self.dataset[subset] = (
                self.dataset[subset]
                .map(
                    process_history,
                    num_proc=num_proc,
                )
                .select_columns(["text", "label"])
            )


class MultilingualIKAT2023(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualIKAT2023",
        description="The task is to retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.",
        reference="https://www.trecikat.com",
        dataset={
            "path": "DeepPavlov/ikat-translated",
            "revision": "main",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2023-11-14", "2023-11-17"),
        domains=["Spoken"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
""",
    )


class MultilingualTopiOCQAHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualTopiOCQAHardNegatives",
        dataset={
            "path": "DeepPavlov/topiocqa-translated",
            "revision": "930de0eb6cd3d4b2d51650cd298dc12d7857d376",
        },
        reference="https://mcgill-nlp.github.io/topiocqa",
        description=(
            "TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) "
            + "is information-seeking conversational dataset with challenging topic switching phenomena. "
            + "It consists of conversation histories along with manually labelled relevant/gold passage. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2021-03-01", "2021-07-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
""",
        adapted_from=["TopiOCQA"],
    )


class DialogSum(AbsTaskClassification):
    label_column_name = "topic"
    metadata = TaskMetadata(
        name="DialogSum",
        description="",
        dataset={
            "path": "DeepPavlov/dialogsum",
            "revision": "main",
        },
        reference="https://huggingface.co/datasets/google/air_dialogue",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        date=("2018-01-01", "2022-06-07"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            dialog = row["dialog"]
            text = ""
            for entry in dialog:
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
                    num_proc=num_proc,
                )
                .select_columns(["text", "topic"])
            )


class MultilingualDialogSum(AbsTaskClassification):
    label_column_name = "topic"

    metadata = TaskMetadata(
        name="MultilingualDialogSum",
        description="",
        dataset={
            "path": "DeepPavlov/dialogsum-translated",
            "revision": "main",
        },
        reference="https://huggingface.co/datasets/google/air_dialogue",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "es-gemma": ["spa-Latn"],
            "fr-gemma": ["fra-Latn"],
        },
        main_score="f1",
        date=("2018-01-01", "2022-06-07"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        bibtex_citation=r"""
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            dialog = row["dialog"]
            text = ""
            for entry in dialog:
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
                    num_proc=num_proc,
                )
                .select_columns(["text", "topic"])
            )
