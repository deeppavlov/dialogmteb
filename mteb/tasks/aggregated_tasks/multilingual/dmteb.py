from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.tasks.classification import (
    AbgCosQA,
    AirDialogueClassification,
    AtisIntentClassification,
    Banking77ClassificationV2,
    ClincIntentClassification,
    DailyDialogClassificationAct,
    DailyDialogClassificationEmotion,
    DialogSum,
    HWUIntentClassification,
    Mantis,
    MultilingualAbgCosQA,
    MultilingualAirDialogueClassification,
    MultilingualAtisIntentClassification,
    MultilingualBanking77Classification,
    MultilingualCanard,
    MultilingualClincIntentClassification,
    MultilingualDailyDialogClassificationAct,
    MultilingualDailyDialogClassificationEmotion,
    MultilingualDialogSum,
    MultilingualFaithDialRetrieval,
    MultilingualHWUIntentClassification,
    MultilingualIKAT2023,
    MultilingualMantis,
    MultilingualMultiWoz21,
    MultilingualQRECC,
    MultilingualStatcanDialogueDatasetRetrieval,
    MultilingualTopiOCQAHardNegatives,
    MultilingualViraIntentClassification,
    MultilingualWiardOfWikipedia,
    MultilingualXRisaWoz,
    MultiWoz21,
    ViraIntentClassification,
    XRisaWoz,
)
from mteb.tasks.pair_classification import QRECC
from mteb.tasks.retrieval import (
    IKAT2023,
    Canard,
    FaithDialRetrieval,
    StatcanDialogueDatasetRetrieval,
    TopiOCQARetrievalHardNegatives,
WizardOfWikipedia,
)


class AggAtisIntent(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggAtisIntent",
        description="AbsTaskAggregate",
        reference="https://huggingface.co/datasets/fathyshalab/atis_intents",
        tasks=[
            AtisIntentClassification(),
            MultilingualAtisIntentClassification().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1990-01-01", "1990-01-01"),
        domains=["Spoken"],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=AtisIntentClassification.metadata.bibtex_citation,
    )


class AggBanking77Classification(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggBanking77Classification",
        description="AbsTaskAggregate",
        reference="https://arxiv.org/abs/2003.04807",
        tasks=[
            Banking77ClassificationV2(),
            MultilingualBanking77Classification().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        bibtex_citation=Banking77ClassificationV2.metadata.bibtex_citation,
    )


class AggViraIntentClassification(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggViraIntentClassification",
        description="AbsTaskAggregate",
        reference="https://huggingface.co/datasets/ibm-research/vira-intents-live",
        tasks=[
            ViraIntentClassification(),
            MultilingualViraIntentClassification().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["val", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        bibtex_citation=ViraIntentClassification.metadata.bibtex_citation,
    )


class AggHWUIntentClassification(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggHWUIntentClassification",
        description="AbsTaskAggregate",
        reference="https://arxiv.org/abs/1903.05566",
        tasks=[
            HWUIntentClassification(),
            MultilingualHWUIntentClassification().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        bibtex_citation=HWUIntentClassification.metadata.bibtex_citation,
    )


class AggClincIntentClassification(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggClincIntentClassification",
        description="AbsTaskAggregate",
        reference="https://huggingface.co/datasets/clinc/clinc_oos",
        tasks=[
            ClincIntentClassification(),
            MultilingualClincIntentClassification().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        bibtex_citation=ClincIntentClassification.metadata.bibtex_citation,
    )


class AggDailyDialogClassificationAct(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggDailyDialogClassificationAct",
        description="AbsTaskAggregate",
        reference="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        tasks=[
            DailyDialogClassificationAct(),
            MultilingualDailyDialogClassificationAct().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        bibtex_citation=DailyDialogClassificationAct.metadata.bibtex_citation,
    )


class AggDailyDialogClassificationEmotion(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggDailyDialogClassificationEmotion",
        description="AbsTaskAggregate",
        reference="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        tasks=[
            DailyDialogClassificationEmotion(),
            MultilingualDailyDialogClassificationEmotion().filter_languages(
                ["fra", "spa"]
            ),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        bibtex_citation=DailyDialogClassificationEmotion.metadata.bibtex_citation,
    )


class AggStatcanDialogueDatasetRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggStatcanDialogueDatasetRetrieval",
        description="AbsTaskAggregate",
        reference="https://mcgill-nlp.github.io/statcan-dialogue-dataset/",
        tasks=[
            StatcanDialogueDatasetRetrieval(),
            MultilingualStatcanDialogueDatasetRetrieval().filter_languages(["spa"]),
        ],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        bibtex_citation=StatcanDialogueDatasetRetrieval.metadata.bibtex_citation,
    )


class AggFaithDialRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggFaithDialRetrieval",
        description="AbsTaskAggregate",
        reference="https://mcgill-nlp.github.io/FaithDial",
        tasks=[
            FaithDialRetrieval(),
            MultilingualFaithDialRetrieval().filter_languages(["fra", "spa"]),
        ],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        bibtex_citation=FaithDialRetrieval.metadata.bibtex_citation,
    )


class AggMultiWoz21(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggMultiWoz21",
        description="AbsTaskAggregate",
        reference="https://arxiv.org/abs/1810.00278",
        tasks=[
            MultiWoz21(),
            MultilingualMultiWoz21().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        bibtex_citation=MultiWoz21.metadata.bibtex_citation,
    )


class AggAirDialogueClassification(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggAirDialogueClassification",
        description="AbsTaskAggregate",
        reference="https://huggingface.co/datasets/google/air_dialogue",
        tasks=[
            AirDialogueClassification(),
            MultilingualAirDialogueClassification().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        bibtex_citation=AirDialogueClassification.metadata.bibtex_citation,
    )


class AggCanard(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggCanard",
        description="AbsTaskAggregate",
        reference=None,
        tasks=[
            Canard(),
            MultilingualCanard().filter_languages(["fra", "spa"]),
        ],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        bibtex_citation=Canard.metadata.bibtex_citation,
    )


class AggMantisClassification(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggMantisClassification",
        description="AbsTaskAggregate",
        reference=None,
        tasks=[
            Mantis(),
            MultilingualMantis().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        bibtex_citation=Mantis.metadata.bibtex_citation,
    )


class AggWiardOfWikipedia(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggWiardOfWikipedia",
        description="AbsTaskAggregate",
        reference=None,
        tasks=[
            WizardOfWikipedia(),
            MultilingualWiardOfWikipedia().filter_languages(["fra", "spa"]),
        ],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        bibtex_citation=WizardOfWikipedia.metadata.bibtex_citation,
    )


class AggQRECC(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggQRECC",
        description="AbsTaskAggregate",
        reference=None,
        tasks=[
            QRECC(),
            MultilingualQRECC().filter_languages(["fra", "spa"]),
        ],
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        bibtex_citation=QRECC.metadata.bibtex_citation,
    )


class AggAbgCosQA(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggAbgCosQA",
        description="AbsTaskAggregate",
        reference=None,
        tasks=[
            AbgCosQA(),
            MultilingualAbgCosQA().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        bibtex_citation=AbgCosQA.metadata.bibtex_citation,
    )


class AggIKAT2023(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggIKAT2023",
        description="AbsTaskAggregate",
        reference="https://www.trecikat.com",
        tasks=[
            IKAT2023(),
            MultilingualIKAT2023().filter_languages(["fra", "spa"]),
        ],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        bibtex_citation=IKAT2023.metadata.bibtex_citation,
    )


class AggXRisaWoz(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggXRisaWoz",
        description="AbsTaskAggregate",
        reference=None,
        tasks=[
            # XRisaWoz natively also covers hi/ko; restrict it to eng/fra so this
            # aggregate stays within the eng/fra/spa languages covered by this file.
            XRisaWoz(), #.filter_languages(["eng", "fra"], hf_subsets=["en", "fr"]),
            MultilingualXRisaWoz().filter_languages(["spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        bibtex_citation=XRisaWoz.metadata.bibtex_citation,
    )


class AggTopiOCQAHardNegatives(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggTopiOCQAHardNegatives",
        description="AbsTaskAggregate",
        reference="https://mcgill-nlp.github.io/topiocqa",
        tasks=[
            TopiOCQARetrievalHardNegatives(),
            MultilingualTopiOCQAHardNegatives().filter_languages(["fra", "spa"]),
        ],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        bibtex_citation=TopiOCQARetrievalHardNegatives.metadata.bibtex_citation,
    )


class AggDialogSum(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="AggDialogSum",
        description="AbsTaskAggregate",
        reference="https://huggingface.co/datasets/google/air_dialogue",
        tasks=[
            DialogSum(),
            MultilingualDialogSum().filter_languages(["fra", "spa"]),
        ],
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        bibtex_citation=DialogSum.metadata.bibtex_citation,
    )
