from .abg_cosqa_fr_classification import FrAbgCosQA
from .air_dialogue_fr_classification import FrAirDialogueClassification
from .clinc_fr_classification import FrClincIntentClassification
from .daily_dialog_fr_classification import (
    FrDailyDialogClassificationAct,
    FrDailyDialogClassificationEmotion,
)
from .french_book_reviews import FrenchBookReviews, FrenchBookReviewsV2
from .mantis_fr_classification import FrMantisClassification
from .movie_review_sentiment_classification import (
    MovieReviewSentimentClassification,
    MovieReviewSentimentClassificationV2,
)
from .vira_intent_fr_classification import FrViraIntentClassification

__all__ = [
    "FrAbgCosQA",
    "FrAirDialogueClassification",
    "FrClincIntentClassification",
    "FrDailyDialogClassificationAct",
    "FrDailyDialogClassificationEmotion",
    "FrMantisClassification",
    "FrViraIntentClassification",
    "FrenchBookReviews",
    "FrenchBookReviewsV2",
    "MovieReviewSentimentClassification",
    "MovieReviewSentimentClassificationV2",
]
