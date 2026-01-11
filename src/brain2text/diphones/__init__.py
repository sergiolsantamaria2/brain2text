"""
Diphone utilities for Brain-to-Text decoding.

Based on DCoND (Divide-Conquer Neural Decoder) approach.
"""

from .diphone_utils import (
    DiphoneConverter,
    N_PHONEMES,
    N_DIPHONES,
    N_DIPHONE_CLASSES,
    DIPHONE_BLANK_IDX,
    phoneme_to_diphone_index,
    diphone_index_to_phonemes,
    phoneme_sequence_to_diphones,
    marginalize_diphone_logits,
)

__all__ = [
    "DiphoneConverter",
    "N_PHONEMES",
    "N_DIPHONES", 
    "N_DIPHONE_CLASSES",
    "DIPHONE_BLANK_IDX",
    "phoneme_to_diphone_index",
    "diphone_index_to_phonemes",
    "phoneme_sequence_to_diphones",
    "marginalize_diphone_logits",
]
