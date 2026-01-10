"""
Diphone utilities for Brain-to-Text decoding (DCoND method).

Based on 1st place Brain-to-Text '24 competition entry.
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
    phoneme_labels_to_diphone_labels,
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
    "phoneme_labels_to_diphone_labels",
    "marginalize_diphone_logits",
]
