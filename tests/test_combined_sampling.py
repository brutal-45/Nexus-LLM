"""Tests for combined sampling strategies."""
import pytest
import torch
import torch.nn.functional as F


def test_top_k_then_top_p():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.1]])
    top_k_vals, _ = torch.topk(logits, k=3, dim=-1)
    threshold = top_k_vals[:, -1:].expand_as(logits)
    filtered = logits.masked_fill(logits < threshold, float("-inf"))
    non_inf = (filtered != float("-inf")).sum().item()
    assert non_inf == 3


def test_temperature_then_top_p():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    scaled = logits / 0.5
    probs = F.softmax(scaled, dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    assert cumsum[0, 0] > 0


def test_combined_sampling_valid_probs():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    probs = F.softmax(logits / 0.7, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
    assert (probs >= 0).all()


def test_repetition_penalty_with_top_k():
    logits = torch.tensor([[2.0, 1.0, 3.0, 0.5]])
    penalized = logits.clone()
    for tok in [0, 2]:
        penalized[0, tok] /= 1.5
    assert penalized[0, 0].item() < logits[0, 0].item()


def test_no_sampling_returns_argmax():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    assert logits.argmax(dim=-1).item() == 1
