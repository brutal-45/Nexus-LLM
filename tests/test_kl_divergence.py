"""Tests for KL divergence loss."""
import pytest
import math


def test_kl_divergence_identical():
    p = [0.25, 0.25, 0.25, 0.25]
    q = [0.25, 0.25, 0.25, 0.25]
    kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    assert abs(kl) < 1e-10

def test_kl_divergence_different():
    p = [0.5, 0.5]
    q = [0.9, 0.1]
    kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    assert kl > 0

def test_kl_divergence_non_negative():
    p = [0.3, 0.7]
    q = [0.4, 0.6]
    kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    assert kl >= 0

def test_kl_divergence_asymmetry():
    p = [0.5, 0.5]
    q = [0.9, 0.1]
    kl_pq = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    kl_qp = sum(qi * math.log(qi / pi) for pi, qi in zip(p, q))
    assert abs(kl_pq - kl_qp) > 0.01
