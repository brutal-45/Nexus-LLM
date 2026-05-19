"""Tests for stopping criteria."""
import pytest
import torch


class StoppingCriteria:
    """Simple stopping criteria for testing."""
    def __init__(self, max_length=100, eos_token_id=2, stop_token_ids=None):
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.stop_token_ids = stop_token_ids or []

    def should_stop(self, generated_ids, step):
        if step >= self.max_length:
            return True, "max_length"
        last_token = generated_ids[-1] if len(generated_ids) > 0 else -1
        if last_token == self.eos_token_id:
            return True, "eos"
        if last_token in self.stop_token_ids:
            return True, "stop_token"
        return False, None


@pytest.fixture
def criteria():
    return StoppingCriteria(max_length=50, eos_token_id=2, stop_token_ids=[100, 200])


def test_stopping_max_length(criteria):
    """Test stopping at max length."""
    should, reason = criteria.should_stop([1, 2, 3], step=50)
    assert should is True
    assert reason == "max_length"


def test_stopping_eos(criteria):
    """Test stopping at EOS token."""
    should, reason = criteria.should_stop([1, 5, 2], step=5)
    assert should is True
    assert reason == "eos"


def test_stopping_custom_token(criteria):
    """Test stopping at custom stop token."""
    should, reason = criteria.should_stop([1, 5, 100], step=5)
    assert should is True
    assert reason == "stop_token"


def test_stopping_continue(criteria):
    """Test that generation continues normally."""
    should, reason = criteria.should_stop([1, 5, 10], step=5)
    assert should is False
    assert reason is None


def test_stopping_before_max_length(criteria):
    """Test that generation doesn't stop before max length."""
    should, reason = criteria.should_stop([1, 5], step=10)
    assert should is False
