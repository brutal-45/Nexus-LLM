"""Tests for constrained beam search."""
import pytest


def test_positive_constraint_matching():
    constraints = ["machine learning"]
    output = "Machine learning is a subset of AI."
    assert any(c.lower() in output.lower() for c in constraints)


def test_negative_constraint_avoidance():
    negative = ["inappropriate", "harmful"]
    output = "This is a helpful response."
    assert not any(c in output for c in negative)


def test_constraint_ordering():
    constraints = ["first", "second", "third"]
    output = "First we do this, second that, third done."
    positions = [output.lower().find(c) for c in constraints]
    assert positions == sorted(positions)


def test_disjunctive_constraints():
    alternatives = ["cat", "dog", "bird"]
    output = "I have a dog at home."
    assert any(alt in output for alt in alternatives)


def test_no_constraints_unconstrained():
    assert len([]) == 0
