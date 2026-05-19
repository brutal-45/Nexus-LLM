"""Tests for training metrics."""
import pytest
import math


class TrainingMetrics:
    """Simple training metrics tracker."""
    def __init__(self):
        self.losses = []
        self.learning_rates = []
        self.steps = []

    def log(self, step, loss, lr):
        self.steps.append(step)
        self.losses.append(loss)
        self.learning_rates.append(lr)

    @property
    def avg_loss(self):
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)

    @property
    def min_loss(self):
        if not self.losses:
            return float("inf")
        return min(self.losses)

    @property
    def perplexity(self):
        if not self.losses:
            return float("inf")
        return math.exp(min(self.avg_loss, 20))  # Cap to avoid overflow


@pytest.fixture
def metrics():
    return TrainingMetrics()


def test_metrics_log(metrics):
    """Test logging metrics."""
    metrics.log(step=1, loss=3.5, lr=1e-4)
    assert len(metrics.losses) == 1


def test_metrics_avg_loss(metrics):
    """Test average loss calculation."""
    metrics.log(1, 3.0, 1e-4)
    metrics.log(2, 2.5, 1e-4)
    assert metrics.avg_loss == pytest.approx(2.75)


def test_metrics_min_loss(metrics):
    """Test minimum loss tracking."""
    metrics.log(1, 3.0, 1e-4)
    metrics.log(2, 2.0, 1e-4)
    metrics.log(3, 2.5, 1e-4)
    assert metrics.min_loss == 2.0


def test_metrics_perplexity(metrics):
    """Test perplexity calculation."""
    metrics.log(1, 2.0, 1e-4)
    assert metrics.perplexity == pytest.approx(math.exp(2.0), rel=0.01)


def test_metrics_empty(metrics):
    """Test metrics with no data."""
    assert metrics.avg_loss == 0.0
    assert metrics.min_loss == float("inf")
