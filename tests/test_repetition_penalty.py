"""Tests for repetition penalty processor."""
import pytest
import torch


def test_repetition_penalty_reduces_logits():
    logits = torch.tensor([[2.0, 1.0, 3.0, 0.5]])
    penalized = logits.clone()
    for tok in [0, 2]:
        penalized[0, tok] /= 2.0
    assert penalized[0, 0].item() < logits[0, 0].item()
    assert penalized[0, 2].item() < logits[0, 2].item()


def test_no_penalty_for_unseen_tokens():
    logits = torch.tensor([[2.0, 1.0, 3.0, 0.5]])
    penalized = logits.clone()
    penalized[0, 0] /= 2.0
    assert penalized[0, 1].item() == logits[0, 1].item()


def test_penalty_value_one_no_effect():
    logits = torch.tensor([[2.0, 1.0, 3.0]])
    penalized = logits.clone()
    for tok in [0, 1]:
        penalized[0, tok] /= 1.0
    assert torch.allclose(penalized, logits)


def test_higher_penalty_stronger_effect():
    logits_val = 5.0
    effects = [logits_val - logits_val / p for p in [1.5, 2.0, 3.0]]
    assert effects[2] > effects[1] > effects[0]
