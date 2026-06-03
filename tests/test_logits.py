"""Tests for logits processing."""
import pytest
import torch
import torch.nn.functional as F
from nexus.inference.generator import TextGenerator


def test_logits_repetition_penalty_positive():
    """Test repetition penalty with positive logits."""
    logits = torch.tensor([[4.0, 2.0, 1.0]])
    generated = torch.tensor([[0]])
    penalized = TextGenerator._apply_repetition_penalty(logits.clone(), generated, penalty=2.0)
    # Positive logit should be divided by penalty
    assert penalized[0, 0].item() == pytest.approx(2.0)


def test_logits_repetition_penalty_negative():
    """Test repetition penalty with negative logits."""
    logits = torch.tensor([[-4.0, 2.0, 1.0]])
    generated = torch.tensor([[0]])
    penalized = TextGenerator._apply_repetition_penalty(logits.clone(), generated, penalty=2.0)
    # Negative logit should be multiplied by penalty
    assert penalized[0, 0].item() == pytest.approx(-8.0)


def test_logits_top_k_preserves_highest():
    """Test that top-k preserves the highest logits."""
    logits = torch.tensor([[0.1, 5.0, 0.2, 4.0, 0.3]])
    filtered = TextGenerator._apply_top_k(logits, k=2)
    assert filtered[0, 1] != float("-inf")  # 5.0 preserved
    assert filtered[0, 3] != float("-inf")  # 4.0 preserved


def test_logits_top_p_preserves_high_probability():
    """Test that top-p preserves high probability tokens."""
    logits = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
    filtered = TextGenerator._apply_top_p(logits, p=0.9)
    assert filtered[0, 3] != float("-inf")


def test_logits_softmax_distribution():
    """Test that softmax produces valid probability distribution."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    probs = F.softmax(logits, dim=-1)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (probs >= 0).all()


def test_logits_temperature_scaling():
    """Test temperature scaling effect on distribution."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    probs_t05 = F.softmax(logits / 0.5, dim=-1)
    probs_t20 = F.softmax(logits / 2.0, dim=-1)
    # Lower temperature = more peaked
    assert probs_t05[0, 2] > probs_t20[0, 2]
