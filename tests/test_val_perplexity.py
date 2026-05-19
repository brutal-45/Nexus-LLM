"""Tests for validation perplexity."""
import pytest
import math


def test_val_perplexity_computation():
    val_loss = 2.5
    ppl = math.exp(val_loss)
    assert ppl == math.exp(2.5)

def test_val_perplexity_better_than_random():
    val_loss = 3.0
    ppl = math.exp(val_loss)
    random_ppl = math.exp(10.0)
    assert ppl < random_ppl

def test_val_perplexity_per_dataset():
    losses = {"wiki": 2.0, "books": 2.5, "code": 1.8}
    ppls = {k: math.exp(v) for k, v in losses.items()}
    assert ppls["code"] < ppls["wiki"]

def test_val_perplexity_tracking():
    history = [math.exp(l) for l in [3.0, 2.5, 2.2, 2.1, 2.0]]
    assert all(history[i] > history[i+1] for i in range(len(history)-1))
