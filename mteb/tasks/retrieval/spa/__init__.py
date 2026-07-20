from .canard_es_retrieval import EsCanard
from .faith_dial_es_retrieval import EsFaithDialRetrieval
from .ikat_es_retrieval import EsIKAT2023
from .spanish_passage_retrieval_s2p import SpanishPassageRetrievalS2P
from .spanish_passage_retrieval_s2s import SpanishPassageRetrievalS2S
from .topiocqa_es_retrieval import EsTopiOCQARetrieval
from .wizard_of_wikipedia_es_retrieval import EsWizardOfWikipedia

__all__ = [
    "EsCanard",
    "EsFaithDialRetrieval",
    "EsIKAT2023",
    "EsTopiOCQARetrieval",
    "EsWizardOfWikipedia",
    "SpanishPassageRetrievalS2P",
    "SpanishPassageRetrievalS2S",
]
