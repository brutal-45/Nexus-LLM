"""Tests for data augmentation."""
import pytest
import random


class TextAugmenter:
    """Simple text augmentation for testing."""
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def random_delete(self, text, p=0.1):
        """Randomly delete characters with probability p."""
        return "".join(c for c in text if self.rng.random() > p)

    def random_swap(self, text):
        """Randomly swap two characters."""
        if len(text) < 2:
            return text
        chars = list(text)
        i, j = self.rng.sample(range(len(chars)), 2)
        chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)

    def random_insert(self, text, chars="abcdefghijklmnopqrstuvwxyz"):
        """Randomly insert a character."""
        if not text:
            return text
        pos = self.rng.randint(0, len(text))
        char = self.rng.choice(chars)
        return text[:pos] + char + text[pos:]


@pytest.fixture
def augmenter():
    return TextAugmenter(seed=42)


def test_random_delete(augmenter):
    """Test random deletion."""
    text = "Hello World"
    result = augmenter.random_delete(text, p=0.5)
    assert len(result) <= len(text)


def test_random_swap(augmenter):
    """Test random swap."""
    text = "abcdef"
    result = augmenter.random_swap(text)
    assert len(result) == len(text)
    assert sorted(result) == sorted(text)


def test_random_insert(augmenter):
    """Test random insertion."""
    text = "abc"
    result = augmenter.random_insert(text)
    assert len(result) == len(text) + 1


def test_random_delete_preserves_most(augmenter):
    """Test that low-probability deletion preserves most text."""
    text = "Hello World"
    result = augmenter.random_delete(text, p=0.01)
    assert len(result) >= len(text) * 0.8
