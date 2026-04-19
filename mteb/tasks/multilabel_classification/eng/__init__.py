from .audio_set import (
    AudioSetMiniMultilingualClassification,
    AudioSetMultilingualClassification,
)
from .fsd50_hf import FSD50HFMultilingualClassification
from .fsd2019_kaggle import FSD2019KaggleMultilingualClassification
from .pascal_voc2007 import VOC2007Classification
from .skill_of_mind import SkillOfMind

__all__ = [
    "AudioSetMiniMultilingualClassification",
    "AudioSetMultilingualClassification",
    "FSD50HFMultilingualClassification",
    "FSD2019KaggleMultilingualClassification",
    "VOC2007Classification",
    "SkillOfMind",
]
