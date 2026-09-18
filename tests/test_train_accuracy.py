"""Tests for training accuracy metric."""
import pytest


def test_accuracy_perfect():
    preds = [0, 1, 2, 3]
    labels = [0, 1, 2, 3]
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    assert acc == 1.0

def test_accuracy_zero():
    preds = [1, 0, 3, 2]
    labels = [0, 1, 2, 3]
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    assert acc == 0.0

def test_accuracy_partial():
    preds = [0, 1, 0, 3]
    labels = [0, 1, 2, 3]
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    assert acc == 0.75

def test_accuracy_top_k():
    probs = [[0.1, 0.7, 0.2], [0.3, 0.2, 0.5]]
    top2 = [sorted(range(len(p)), key=lambda i: -p[i])[:2] for p in probs]
    labels = [1, 0]
    correct = sum(l in t2 for l, t2 in zip(labels, top2))
    assert correct == 2
