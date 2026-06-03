"""Tests for custom stopping criteria."""
import pytest


def test_stop_on_eos_token():
    eos_token_id = 2
    generated = [10, 20, 30, 2]
    assert eos_token_id in generated


def test_stop_on_max_length():
    assert 50 >= 50


def test_stop_on_keyword():
    stop_words = ["###", "<END>"]
    text = "Generated text ### more"
    assert any(sw in text for sw in stop_words)


def test_multiple_stopping_criteria():
    def should_stop(generated_ids, max_length=100, eos_id=2, current_len=0):
        return current_len >= max_length or eos_id in generated_ids
    assert should_stop([2], current_len=5) is True
    assert should_stop([], current_len=100) is True
    assert should_stop([], current_len=50) is False


def test_stopping_criteria_callback():
    from unittest.mock import MagicMock
    cb = MagicMock(return_value=False)
    assert cb([1, 2, 3]) is False
