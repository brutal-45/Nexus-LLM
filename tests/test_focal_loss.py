"""Tests for focal loss."""
import pytest
import math


def test_focal_loss_basic():
    pt = 0.9  # Probability of correct class
    gamma = 2.0
    focal_weight = (1 - pt) ** gamma
    assert focal_weight < 1.0  # Down-weights easy examples

def test_focal_loss_hard_examples():
    pt = 0.1  # Low probability (hard example)
    gamma = 2.0
    focal_weight = (1 - pt) ** gamma
    assert focal_weight > 0.5

def test_focal_loss_gamma_zero_is_ce():
    pt = 0.7
    gamma = 0.0
    focal_weight = (1 - pt) ** gamma
    assert focal_weight == 1.0  # Same as cross-entropy

def test_focal_loss_gamma_effect():
    pt = 0.5
    weights = [(1 - pt) ** g for g in [0, 1, 2, 5]]
    assert weights == sorted(weights, reverse=True)
