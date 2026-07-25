from .abg_cosqa_es_classification import EsAbgCosQA
from .air_dialogue_es_classification import EsAirDialogueClassification
from .atis_intent_classification_es import EsAtisIntentClassification
from .banking77_es_classification import EsBanking77Classification
from .clinc_es_classification import EsClincIntentClassification
from .daily_dialog_es_classification import (
    EsDailyDialogClassificationAct,
    EsDailyDialogClassificationEmotion,
)
from .hwu_intent_classification_es import EsHWUIntentClassification
from .mantis_es_classification import EsMantisClassification
from .spanish_news_classification import (
    SpanishNewsClassification,
    SpanishNewsClassificationV2,
)
from .spanish_sentiment_classification import (
    SpanishSentimentClassification,
    SpanishSentimentClassificationV2,
)
from .vira_intent_es_classification import EsViraIntentClassification
from .xrisawoz_es_classification import EsXRisaWoz

__all__ = [
    "EsAbgCosQA",
    "EsAirDialogueClassification",
    "EsAtisIntentClassification",
    "EsBanking77Classification",
    "EsClincIntentClassification",
    "EsDailyDialogClassificationAct",
    "EsDailyDialogClassificationEmotion",
    "EsHWUIntentClassification",
    "EsMantisClassification",
    "EsViraIntentClassification",
    "EsXRisaWoz",
    "SpanishNewsClassification",
    "SpanishNewsClassificationV2",
    "SpanishSentimentClassification",
    "SpanishSentimentClassificationV2",
]
