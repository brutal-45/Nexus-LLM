"""Tests for data splitting functions."""
import pytest
import random


def test_train_test_split_ratios():
    data = list(range(1000))
    split_idx = int(len(data) * 0.8)
    train, test = data[:split_idx], data[split_idx:]
    assert len(train) == 800
    assert len(test) == 200


def test_train_val_test_split():
    data = list(range(1000))
    train_end, val_end = int(len(data) * 0.7), int(len(data) * 0.85)
    train, val, test = data[:train_end], data[train_end:val_end], data[val_end:]
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150


def test_stratified_split_preserves_distribution():
    labels = [0] * 500 + [1] * 300 + [2] * 200
    total = len(labels)
    assert abs(labels.count(0) / total - 0.5) < 0.01


def test_split_no_overlap():
    data = list(range(100))
    train, test = set(data[:80]), set(data[80:])
    assert train.isdisjoint(test)


def test_split_with_seed_reproducible():
    data = list(range(100))
    random.seed(42)
    s1 = data[:]; random.shuffle(s1)
    random.seed(42)
    s2 = data[:]; random.shuffle(s2)
    assert s1 == s2
