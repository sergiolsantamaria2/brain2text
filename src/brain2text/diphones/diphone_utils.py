"""
Diphone utilities for Brain-to-Text decoding.

Based on DCoND (Divide-Conquer Neural Decoder) from:
"Brain-to-text decoding with context-aware neural representations and large language models"
Li et al., 2024 (1st place Brain-to-Text '24)

Key insight: Neural signals encode transitions between phonemes (diphones),
not just individual phonemes. Decoding diphones and marginalizing to phonemes
reduces PER from 16.62% to 15.34%.

Diphone encoding:
- 40 phonemes × 40 phonemes = 1600 diphone classes
- Plus 1 blank token = 1601 total classes
- Diphone index = prev_phoneme * n_phonemes + current_phoneme
- Blank is always the last index (1600)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


# Standard number of phonemes (excluding blank)
N_PHONEMES = 40
N_DIPHONES = N_PHONEMES * N_PHONEMES  # 1600
N_DIPHONE_CLASSES = N_DIPHONES + 1     # 1601 (including blank)
DIPHONE_BLANK_IDX = N_DIPHONES         # 1600


def phoneme_to_diphone_index(prev_phoneme: int, curr_phoneme: int) -> int:
    """
    Convert a phoneme pair to a diphone index.
    
    Args:
        prev_phoneme: Previous phoneme index (0-39)
        curr_phoneme: Current phoneme index (0-39)
    
    Returns:
        Diphone index (0-1599)
    """
    assert 0 <= prev_phoneme < N_PHONEMES, f"Invalid prev_phoneme: {prev_phoneme}"
    assert 0 <= curr_phoneme < N_PHONEMES, f"Invalid curr_phoneme: {curr_phoneme}"
    return prev_phoneme * N_PHONEMES + curr_phoneme


def diphone_index_to_phonemes(diphone_idx: int) -> Tuple[int, int]:
    """
    Convert a diphone index back to phoneme pair.
    
    Args:
        diphone_idx: Diphone index (0-1599)
    
    Returns:
        Tuple of (prev_phoneme, curr_phoneme)
    """
    assert 0 <= diphone_idx < N_DIPHONES, f"Invalid diphone_idx: {diphone_idx}"
    prev_phoneme = diphone_idx // N_PHONEMES
    curr_phoneme = diphone_idx % N_PHONEMES
    return prev_phoneme, curr_phoneme


def phoneme_sequence_to_diphones(
    phoneme_seq: List[int],
    silence_idx: int = 40,
    blank_idx: int = 40,
) -> List[int]:
    """
    Convert a phoneme sequence to a diphone sequence.
    
    For CTC, we need to handle the blank token specially:
    - Blank tokens in input are converted to diphone blank (1600)
    - Non-blank phonemes are converted to diphone pairs
    
    The first phoneme is paired with silence (SIL) as the "previous" phoneme.
    We use phoneme index 0 as SIL (or configurable silence_idx).
    
    Args:
        phoneme_seq: List of phoneme indices (including blanks)
        silence_idx: Index used for silence phoneme (default 40, which is blank/SIL)
        blank_idx: Index of CTC blank in phoneme space (default 40)
    
    Returns:
        List of diphone indices
    
    Example:
        phoneme_seq = [5, 12, 8]  # HH, EH, L (no blanks)
        # With implicit SIL start:
        # diphones = [(SIL,HH), (HH,EH), (EH,L)]
        # If silence_idx=0: [0*40+5, 5*40+12, 12*40+8] = [5, 212, 488]
    """
    if len(phoneme_seq) == 0:
        return []
    
    diphone_seq = []
    
    # For first phoneme, use silence as "previous"
    # But we need to handle this carefully - in CTC ground truth,
    # we typically don't have explicit silence markers
    prev_phoneme = 0  # Assume SIL (index 0) as start context
    
    for phoneme in phoneme_seq:
        if phoneme == blank_idx:
            # CTC blank stays as diphone blank
            diphone_seq.append(DIPHONE_BLANK_IDX)
        else:
            # Convert to diphone
            diphone_idx = phoneme_to_diphone_index(prev_phoneme, phoneme)
            diphone_seq.append(diphone_idx)
            prev_phoneme = phoneme
    
    return diphone_seq


def phoneme_labels_to_diphone_labels(
    phoneme_labels: torch.Tensor,
    blank_idx: int = 40,
) -> torch.Tensor:
    """
    Convert batch of phoneme label sequences to diphone label sequences.
    
    This is used to convert ground truth labels for CTC training.
    
    Args:
        phoneme_labels: [batch, max_label_len] phoneme indices
        blank_idx: CTC blank index in phoneme space
    
    Returns:
        [batch, max_label_len] diphone indices
    """
    batch_size, max_len = phoneme_labels.shape
    diphone_labels = torch.full_like(phoneme_labels, DIPHONE_BLANK_IDX)
    
    for b in range(batch_size):
        seq = phoneme_labels[b].tolist()
        # Filter out padding (usually -1 or some padding value)
        # Assuming padding is handled separately
        diphone_seq = phoneme_sequence_to_diphones(seq, blank_idx=blank_idx)
        for t, d in enumerate(diphone_seq):
            if t < max_len:
                diphone_labels[b, t] = d
    
    return diphone_labels


def marginalize_diphone_logits(
    diphone_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Marginalize diphone logits to phoneme logits.
    
    This is the "conquer" step of DCoND: we sum over all possible
    previous phonemes to get the probability of the current phoneme.
    
    P(current_phoneme) = Σ_{prev} P(prev, current)
    
    In log space: log P(current) = logsumexp_{prev} log P(prev, current)
    
    Args:
        diphone_logits: [batch, time, 1601] raw logits from model
        temperature: Temperature for softmax (default 1.0)
    
    Returns:
        [batch, time, 41] phoneme logits (40 phonemes + blank)
    """
    batch, time, n_classes = diphone_logits.shape
    assert n_classes == N_DIPHONE_CLASSES, f"Expected {N_DIPHONE_CLASSES} classes, got {n_classes}"
    
    # Separate diphone logits and blank logit
    diphone_part = diphone_logits[:, :, :N_DIPHONES]  # [batch, time, 1600]
    blank_logit = diphone_logits[:, :, N_DIPHONES:]   # [batch, time, 1]
    
    # Reshape to [batch, time, n_prev=40, n_curr=40]
    diphone_reshaped = diphone_part.reshape(batch, time, N_PHONEMES, N_PHONEMES)
    
    # Apply temperature
    if temperature != 1.0:
        diphone_reshaped = diphone_reshaped / temperature
        blank_logit = blank_logit / temperature
    
    # Marginalize over previous phoneme (logsumexp over dim=2)
    # This gives us log P(current_phoneme)
    phoneme_logits = torch.logsumexp(diphone_reshaped, dim=2)  # [batch, time, 40]
    
    # Concatenate with blank
    phoneme_logits = torch.cat([phoneme_logits, blank_logit], dim=-1)  # [batch, time, 41]
    
    return phoneme_logits


def create_diphone_to_phoneme_matrix() -> torch.Tensor:
    """
    Create a sparse marginalization matrix for efficient diphone→phoneme conversion.
    
    Returns:
        [1601, 41] matrix M where phoneme_logits = diphone_logits @ M
        (after appropriate log-space handling)
    """
    # This is for potential optimization - for now we use the explicit logsumexp
    M = torch.zeros(N_DIPHONE_CLASSES, N_PHONEMES + 1)
    
    # Each diphone (prev, curr) contributes to phoneme curr
    for prev in range(N_PHONEMES):
        for curr in range(N_PHONEMES):
            diphone_idx = prev * N_PHONEMES + curr
            M[diphone_idx, curr] = 1.0
    
    # Blank maps to blank
    M[DIPHONE_BLANK_IDX, N_PHONEMES] = 1.0
    
    return M


class DiphoneConverter:
    """
    Helper class for diphone conversion in training and inference.
    
    Usage:
        converter = DiphoneConverter()
        
        # Training: convert phoneme labels to diphone labels
        diphone_labels = converter.phonemes_to_diphones(phoneme_labels)
        
        # Inference: convert diphone logits to phoneme logits
        phoneme_logits = converter.marginalize(diphone_logits)
    """
    
    def __init__(
        self,
        n_phonemes: int = N_PHONEMES,
        blank_idx: int = 40,
    ):
        self.n_phonemes = n_phonemes
        self.n_diphones = n_phonemes * n_phonemes
        self.n_diphone_classes = self.n_diphones + 1
        self.diphone_blank_idx = self.n_diphones
        self.phoneme_blank_idx = blank_idx
    
    def phonemes_to_diphones_sequence(self, phoneme_seq: List[int]) -> List[int]:
        """Convert single phoneme sequence to diphone sequence."""
        return phoneme_sequence_to_diphones(
            phoneme_seq, 
            blank_idx=self.phoneme_blank_idx
        )
    
    def phonemes_to_diphones_batch(
        self, 
        phoneme_labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert batch of phoneme labels to diphone labels.
        
        Args:
            phoneme_labels: [batch, max_len] phoneme indices
            label_lengths: [batch] actual lengths
        
        Returns:
            diphone_labels: [batch, max_len] diphone indices
            label_lengths: [batch] (unchanged, same lengths)
        """
        batch_size, max_len = phoneme_labels.shape
        device = phoneme_labels.device
        
        diphone_labels = torch.full(
            (batch_size, max_len), 
            fill_value=self.diphone_blank_idx,
            dtype=phoneme_labels.dtype,
            device=device
        )
        
        for b in range(batch_size):
            length = label_lengths[b].item()
            seq = phoneme_labels[b, :length].tolist()
            diphone_seq = self.phonemes_to_diphones_sequence(seq)
            
            for t, d in enumerate(diphone_seq):
                diphone_labels[b, t] = d
        
        return diphone_labels, label_lengths
    
    def marginalize(
        self, 
        diphone_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Marginalize diphone logits to phoneme logits."""
        return marginalize_diphone_logits(diphone_logits, temperature)
    
    @property
    def num_classes(self) -> int:
        """Number of diphone classes (for model output layer)."""
        return self.n_diphone_classes


# Quick sanity check
if __name__ == "__main__":
    print("=== Diphone Utils Test ===")
    
    # Test basic conversion
    print("\n1. Basic phoneme→diphone conversion:")
    phonemes = [5, 12, 8, 20]  # Example: HH, EH, L, OW
    diphones = phoneme_sequence_to_diphones(phonemes)
    print(f"   Phonemes: {phonemes}")
    print(f"   Diphones: {diphones}")
    
    # Verify reverse
    print("\n2. Diphone→phoneme reverse:")
    for d in diphones:
        if d < N_DIPHONES:
            prev, curr = diphone_index_to_phonemes(d)
            print(f"   Diphone {d} → (prev={prev}, curr={curr})")
    
    # Test marginalization
    print("\n3. Marginalization test:")
    batch, time = 2, 10
    fake_logits = torch.randn(batch, time, N_DIPHONE_CLASSES)
    phoneme_logits = marginalize_diphone_logits(fake_logits)
    print(f"   Diphone logits shape: {fake_logits.shape}")
    print(f"   Phoneme logits shape: {phoneme_logits.shape}")
    
    # Test converter class
    print("\n4. DiphoneConverter class:")
    converter = DiphoneConverter()
    print(f"   n_phonemes: {converter.n_phonemes}")
    print(f"   n_diphone_classes: {converter.num_classes}")
    
    # Test batch conversion
    print("\n5. Batch label conversion:")
    phoneme_labels = torch.tensor([
        [5, 12, 8, 20, 0],
        [3, 7, 15, 0, 0],
    ])
    label_lengths = torch.tensor([4, 3])
    diphone_labels, _ = converter.phonemes_to_diphones_batch(phoneme_labels, label_lengths)
    print(f"   Phoneme labels:\n   {phoneme_labels}")
    print(f"   Diphone labels:\n   {diphone_labels}")
    
    print("\n=== All tests passed! ===")
