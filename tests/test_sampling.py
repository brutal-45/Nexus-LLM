"""Tests for sampling strategies."""
import pytest
import torch
import torch.nn.functional as F
from nexus.inference.generator import TextGenerator


def test_greedy_sampling():
    """Test greedy (argmax) sampling."""
    logits = torch.tensor([[0.1, 0.2, 0.9, 0.3]])
    token = logits.argmax(dim=-1)
    assert token.item() == 2


def test_temperature_sampling():
    """Test temperature scaling."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    temp_low = logits / 0.5  # Sharper
    temp_high = logits / 2.0  # Softer
    # Low temperature should make distribution more peaked
    probs_low = F.softmax(temp_low, dim=-1)
    probs_high = F.softmax(temp_high, dim=-1)
    assert probs_low[0, 2] > probs_high[0, 2]


def test_top_k_sampling():
    """Test top-k filtering."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.05]])
    filtered = TextGenerator._apply_top_k(logits, k=2)
    non_inf_count = (filtered != float("-inf")).sum().item()
    assert non_inf_count == 2


def test_top_p_sampling():
    """Test nucleus (top-p) filtering."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 10.0, 0.01]])
    filtered = TextGenerator._apply_top_p(logits, p=0.9)
    # Highest probability token should always remain
    assert filtered[0, 3] != float("-inf")


def test_top_k_with_small_k():
    """Test top-k with k=1 (equivalent to greedy)."""
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    filtered = TextGenerator._apply_top_k(logits, k=1)
    non_inf = (filtered != float("-inf")).sum().item()
    assert non_inf == 1


def test_repetition_penalty():
    """Test repetition penalty reduces probability of seen tokens."""
    logits = torch.tensor([[2.0, 1.0, 3.0, 0.5]])
    generated = torch.tensor([[0, 2]])
    penalized = TextGenerator._apply_repetition_penalty(logits.clone(), generated, penalty=2.0)
    # Tokens 0 and 2 should be penalized
    assert penalized[0, 0].item() < logits[0, 0].item()
    assert penalized[0, 2].item() < logits[0, 2].item()
    # Token 1 should be unchanged
    assert penalized[0, 1].item() == logits[0, 1].item()
