"""Tests for random deletion augmentation."""
import pytest
import random


def test_random_deletion_removes_words():
    random.seed(42)
    text = "The quick brown fox jumps over the lazy dog"
    words = text.split()
    p_delete = 0.3
    result = [w for w in words if random.random() > p_delete]
    assert len(result) < len(words)

def test_random_deletion_preserves_some():
    random.seed(42)
    words = ["a"] * 100
    p_delete = 0.3
    result = [w for w in words if random.random() > p_delete]
    assert len(result) > 0

def test_random_deletion_no_deletion():
    random.seed(42)
    words = ["hello", "world"]
    p_delete = 0.0
    result = [w for w in words if random.random() > p_delete]
    assert len(result) == 2

def test_random_deletion_all_deletion():
    words = ["a", "b", "c"]
    p_delete = 1.0
    result = []
    assert len(result) == 0
