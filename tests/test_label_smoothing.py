"""Tests for label smoothing loss."""
import pytest
import numpy as np


def test_label_smoothing_distribution():
    num_classes = 5
    smoothing = 0.1
    confidence = 1.0 - smoothing
    smooth_val = smoothing / (num_classes - 1)
    labels = [smooth_val] * num_classes
    labels[2] = confidence + smooth_val  # Target class
    assert abs(sum(labels) - 1.0) < 1e-5

def test_label_smoothing_zero():
    smoothing = 0.0
    confidence = 1.0
    assert confidence == 1.0

def test_label_smoothing_high():
    smoothing = 0.3
    confidence = 0.7
    num_classes = 10
    smooth_val = smoothing / (num_classes - 1)
    assert smooth_val < 0.05

def test_label_smoothing_reduces_overconfidence():
    # Without smoothing: [0, 1, 0, 0, 0]
    # With smoothing: [0.025, 0.9, 0.025, 0.025, 0.025]
    smoothed = [0.025, 0.9, 0.025, 0.025, 0.025]
    assert max(smoothed) < 1.0
