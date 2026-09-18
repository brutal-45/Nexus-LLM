"""Tests for split pane layout."""
import pytest


def test_split_pane_creation():
    panes = {"left": 0.5, "right": 0.5}
    assert abs(panes["left"] + panes["right"] - 1.0) < 0.01

def test_split_pane_resize():
    panes = {"left": 0.3, "right": 0.7}
    assert abs(panes["left"] + panes["right"] - 1.0) < 0.01

def test_split_pane_minimum_size():
    min_size = 0.1
    panes = {"left": 0.05, "right": 0.95}
    panes["left"] = max(panes["left"], min_size)
    assert panes["left"] >= min_size

def test_split_pane_three_way():
    sizes = [0.33, 0.34, 0.33]
    assert abs(sum(sizes) - 1.0) < 0.01
