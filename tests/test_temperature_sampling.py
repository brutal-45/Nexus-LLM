"""Tests for temperature sampling."""
import pytest
import torch
import torch.nn.functional as F


def test_temperature_one_no_change():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(logits / 1.0, logits)


def test_low_temperature_sharpens():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert F.softmax(logits / 0.1, dim=-1)[0, 2] > F.softmax(logits / 2.0, dim=-1)[0, 2]


def test_high_temperature_flattens():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    probs = F.softmax(logits / 10.0, dim=-1)
    assert abs(probs[0, 0].item() - 1/3) < 0.1


def test_temperature_zero_approximates_greedy():
    logits = torch.tensor([[1.0, 2.0, 10.0]])
    assert F.softmax(logits / 0.001, dim=-1)[0, 2].item() > 0.99


def test_temperature_preserves_order():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    for temp in [0.5, 1.0, 2.0]:
        scaled = logits / temp
        assert scaled[0, 1] > scaled[0, 2] > scaled[0, 0]
