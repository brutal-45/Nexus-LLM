"""Tests for loss functions."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_cross_entropy_loss():
    """Test cross-entropy loss computation."""
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    labels = torch.tensor([0])
    loss = F.cross_entropy(logits, labels)
    assert loss.item() > 0


def test_cross_entropy_loss_perfect():
    """Test cross-entropy with perfect prediction."""
    logits = torch.tensor([[100.0, 0.0, 0.0]])
    labels = torch.tensor([0])
    loss = F.cross_entropy(logits, labels)
    assert loss.item() < 0.01


def test_kl_divergence_loss():
    """Test KL divergence loss."""
    p = torch.tensor([[0.5, 0.5]])
    q = torch.tensor([[0.5, 0.5]])
    kl = F.kl_div(q.log(), p, reduction="sum")
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


def test_mse_loss():
    """Test MSE loss."""
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    loss = F.mse_loss(pred, target)
    assert loss.item() == 0.0


def test_label_smoothing_loss():
    """Test label smoothing via cross-entropy."""
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    labels = torch.tensor([0])
    loss_no_smooth = F.cross_entropy(logits, labels, label_smoothing=0.0)
    loss_smooth = F.cross_entropy(logits, labels, label_smoothing=0.1)
    # Smoothed loss should be higher (distributes probability mass)
    assert loss_smooth.item() > loss_no_smooth.item()
