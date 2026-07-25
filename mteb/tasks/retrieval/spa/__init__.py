from .canard_es_retrieval import EsCanard
from .coral_es_retrieval import EsCoral
from .dialogsum_es_retrieval import EsDialogSumRetrieval
from .faith_dial_es_retrieval import EsFaithDialRetrieval
from .ikat_es_retrieval import EsIKAT2023
from .spanish_passage_retrieval_s2p import SpanishPassageRetrievalS2P
from .spanish_passage_retrieval_s2s import SpanishPassageRetrievalS2S
from .statcan_dialogue_es_retrieval import EsStatcanDialogueDatasetRetrieval
from .topiocqa_es_retrieval import EsTopiOCQARetrieval
from .wizard_of_wikipedia_es_retrieval import EsWizardOfWikipedia

__all__ = [
    "EsCanard",
    "EsCoral",
    "EsDialogSumRetrieval",
    "EsFaithDialRetrieval",
    "EsIKAT2023",
    "EsStatcanDialogueDatasetRetrieval",
    "EsTopiOCQARetrieval",
    "EsWizardOfWikipedia",
    "SpanishPassageRetrievalS2P",
    "SpanishPassageRetrievalS2S",
]
