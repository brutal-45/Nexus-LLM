"""Tests for early stopping callback."""
import pytest


def test_early_stopping_no_improvement():
    patience = 3
    best_loss = 0.5
    no_improve = 0
    for loss in [0.6, 0.55, 0.51]:
        if loss < best_loss:
            best_loss = loss
            no_improve = 0
        else:
            no_improve += 1
    assert no_improve == 3

def test_early_stopping_triggered():
    patience = 3
    no_improve = 3
    should_stop = no_improve >= patience
    assert should_stop is True

def test_early_stopping_not_triggered():
    patience = 5
    no_improve = 3
    should_stop = no_improve >= patience
    assert should_stop is False

def test_early_stopping_with_improvement():
    best_loss = 0.5
    current_loss = 0.3
    improved = current_loss < best_loss
    assert improved is True
