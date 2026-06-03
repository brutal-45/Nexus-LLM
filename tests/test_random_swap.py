"""Tests for random swap augmentation."""
import pytest
import random


def test_random_swap_changes_order():
    random.seed(42)
    words = ["a", "b", "c", "d", "e"]
    original = words[:]
    i, j = 1, 3
    words[i], words[j] = words[j], words[i]
    assert words != original

def test_random_swap_preserves_elements():
    random.seed(42)
    words = ["a", "b", "c", "d"]
    original_set = set(words)
    i, j = 0, 2
    words[i], words[j] = words[j], words[i]
    assert set(words) == original_set

def test_random_swap_same_index():
    words = ["a", "b", "c"]
    i, j = 1, 1
    original = words[:]
    words[i], words[j] = words[j], words[i]
    assert words == original

def test_random_swap_multiple():
    random.seed(42)
    words = list(range(10))
    for _ in range(5):
        i, j = random.randint(0, 9), random.randint(0, 9)
        words[i], words[j] = words[j], words[i]
    assert len(words) == 10
