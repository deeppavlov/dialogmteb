from .air_dialogue_es_classification import EsAirDialogueClassification
from .spanish_news_classification import (
    SpanishNewsClassification,
    SpanishNewsClassificationV2,
)
from .spanish_sentiment_classification import (
    SpanishSentimentClassification,
    SpanishSentimentClassificationV2,
)
from .xrisawoz_es_classification import EsXRisaWoz

__all__ = [
    "EsAirDialogueClassification",
    "EsXRisaWoz",
    "SpanishNewsClassification",
    "SpanishNewsClassificationV2",
    "SpanishSentimentClassification",
    "SpanishSentimentClassificationV2",
]
