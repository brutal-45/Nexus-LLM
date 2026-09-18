"""Tests for beam search."""
import pytest
import torch
import torch.nn.functional as F


class BeamHypothesis:
    """A single beam hypothesis."""
    def __init__(self, token_ids, log_prob=0.0):
        self.token_ids = token_ids
        self.log_prob = log_prob

    def score(self, length_penalty=1.0, eos=False):
        length = len(self.token_ids)
        if eos and length > 0:
            length_penalty_factor = ((5 + length) / 6) ** length_penalty
        else:
            length_penalty_factor = ((5 + length) / 6) ** length_penalty
        return self.log_prob / length_penalty_factor


def test_beam_hypothesis_creation():
    """Test creating a beam hypothesis."""
    h = BeamHypothesis([1, 2, 3], log_prob=-1.5)
    assert h.token_ids == [1, 2, 3]
    assert h.log_prob == -1.5


def test_beam_hypothesis_score():
    """Test beam hypothesis scoring."""
    h = BeamHypothesis([1, 2, 3], log_prob=-3.0)
    s = h.score(length_penalty=1.0)
    assert s != 0


def test_beam_hypothesis_length_penalty():
    """Test that length penalty affects score."""
    h = BeamHypothesis([1, 2, 3], log_prob=-3.0)
    s_no_penalty = h.score(length_penalty=0.0)
    s_with_penalty = h.score(length_penalty=1.0)
    # Both scores should be negative, and they should differ
    assert s_no_penalty != s_with_penalty


def test_beam_search_top_k_selection():
    """Test selecting top-k from log probabilities."""
    log_probs = torch.tensor([[-0.1, -2.0, -0.5, -3.0, -0.2]])
    top_k = 3
    values, indices = log_probs.topk(top_k, dim=-1)
    assert len(values[0]) == top_k
    assert indices[0, 0].item() == 0  # Highest prob
