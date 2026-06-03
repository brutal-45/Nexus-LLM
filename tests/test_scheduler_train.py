"""Tests for LR scheduler."""
import pytest
import torch
from nexus.training.scheduler import (
    CosineAnnealingWithWarmup, LinearWarmupWithDecay,
    WarmupStableDecay, InverseSquareRootDecay,
    ConstantWithWarmup, get_scheduler,
)


@pytest.fixture
def optimizer():
    model = torch.nn.Linear(10, 10)
    return torch.optim.SGD(model.parameters(), lr=1e-3)


def test_cosine_scheduler(optimizer):
    """Test cosine annealing with warmup."""
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps=100, total_steps=1000)
    # During warmup, LR should increase
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    # LR should increase during warmup
    assert lrs[-1] > lrs[0]


def test_linear_warmup_decay(optimizer):
    """Test linear warmup with decay."""
    scheduler = LinearWarmupWithDecay(optimizer, warmup_steps=100, total_steps=1000)
    assert scheduler is not None


def test_warmup_stable_decay(optimizer):
    """Test WSD scheduler."""
    scheduler = WarmupStableDecay(
        optimizer, warmup_steps=100, stable_steps=500, total_steps=1000
    )
    assert scheduler is not None


def test_inverse_sqrt_decay(optimizer):
    """Test inverse square root decay."""
    scheduler = InverseSquareRootDecay(optimizer, warmup_steps=100, total_steps=1000)
    assert scheduler is not None


def test_constant_with_warmup(optimizer):
    """Test constant LR with warmup."""
    scheduler = ConstantWithWarmup(optimizer, warmup_steps=100)
    assert scheduler is not None


def test_get_scheduler_cosine(optimizer):
    """Test get_scheduler factory with cosine."""
    scheduler = get_scheduler("cosine", optimizer, warmup_steps=100, total_steps=1000)
    assert isinstance(scheduler, CosineAnnealingWithWarmup)


def test_get_scheduler_linear(optimizer):
    """Test get_scheduler factory with linear."""
    scheduler = get_scheduler("linear", optimizer, warmup_steps=100, total_steps=1000)
    assert isinstance(scheduler, LinearWarmupWithDecay)


def test_get_scheduler_invalid(optimizer):
    """Test get_scheduler with invalid name."""
    with pytest.raises(ValueError):
        get_scheduler("invalid_scheduler", optimizer)
