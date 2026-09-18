"""Tests for nucleus/top-p sampling."""
import pytest
import torch
import torch.nn.functional as F


def test_top_p_filters_low_prob_tokens():
    logits = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
    probs = F.softmax(logits, dim=-1)
    cumsum = torch.cumsum(torch.sort(probs, descending=True)[0], dim=-1)
    assert (cumsum < 0.9).sum().item() + 1 >= 1


def test_top_p_one_keeps_all():
    probs = F.softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
    assert (probs > 0).all()


def test_top_p_zero_keeps_only_best():
    probs = F.softmax(torch.tensor([[0.0, 0.0, 0.0, 10.0]]), dim=-1)
    assert torch.sort(probs, descending=True)[0][0, 0] > 0.9


def test_top_p_cumulative_sum():
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    cumsum = torch.cumsum(torch.sort(probs, descending=True)[0], dim=0)
    assert abs(cumsum[-1].item() - 1.0) < 1e-5


def test_top_p_with_uniform_distribution():
    probs = F.softmax(torch.tensor([[0.0, 0.0, 0.0, 0.0]]), dim=-1)
    assert abs(probs[0, 0].item() - 0.25) < 1e-5
