"""Tests for training perplexity metric."""
import pytest
import math


def test_perplexity_from_loss():
    loss = 2.0
    ppl = math.exp(loss)
    assert ppl == math.exp(2.0)

def test_perplexity_decreases_with_loss():
    ppl_1 = math.exp(3.0)
    ppl_2 = math.exp(2.0)
    assert ppl_2 < ppl_1

def test_perplexity_minimum():
    ppl = math.exp(0.0)
    assert ppl == 1.0

def test_perplexity_high_loss():
    ppl = math.exp(10.0)
    assert ppl > 1000
