"""Tests for data deduplication."""
import pytest


def test_exact_deduplication():
    docs = ["hello world", "hello world", "different doc"]
    unique = list(dict.fromkeys(docs))
    assert len(unique) == 2

def test_fuzzy_deduplication():
    docs = ["The quick brown fox", "The quick brown fox.", "Totally different"]
    # Simple similarity check
    def similar(a, b):
        return len(set(a.split()) & set(b.split())) / len(set(a.split()) | set(b.split()))
    assert similar(docs[0], docs[1]) > 0.8

def test_dedup_preserves_order():
    docs = ["first", "second", "first", "third"]
    seen = set()
    unique = []
    for d in docs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    assert unique == ["first", "second", "third"]

def test_dedup_count():
    docs = ["a", "b", "a", "c", "b", "a"]
    from collections import Counter
    counts = Counter(docs)
    assert counts["a"] == 3
