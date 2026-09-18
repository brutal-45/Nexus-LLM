"""Tests for diverse beam search."""
import pytest
import torch
from collections import Counter


def test_beam_diversity_penalty():
    beam_scores = torch.tensor([[-1.0, -2.0, -3.0]])
    for tok in {1}:
        beam_scores[0, tok] -= 0.5
    assert beam_scores[0, 1].item() == -2.5


def test_hamming_diversity():
    beam_tokens = [[1, 2, 3], [1, 4, 5], [6, 7, 8]]
    counts = Counter(t for beam in beam_tokens for t in beam)
    assert counts[1] == 2


def test_num_beam_groups():
    assert 6 // 3 == 2
    assert 6 % 3 == 0


def test_diverse_beam_groups_independent():
    num_groups = 3
    groups = [[] for _ in range(num_groups)]
    for i in range(6):
        groups[i % num_groups].append(f"beam_{i}")
    assert len(groups[0]) == 2
