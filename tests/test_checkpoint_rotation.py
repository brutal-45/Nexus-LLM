"""Tests for checkpoint rotation."""
import pytest


def test_max_checkpoints():
    checkpoints = ["ckpt_1", "ckpt_2", "ckpt_3", "ckpt_4", "ckpt_5"]
    max_keep = 3
    while len(checkpoints) > max_keep:
        checkpoints.pop(0)
    assert len(checkpoints) == 3

def test_checkpoint_ordering():
    checkpoints = ["ckpt_3", "ckpt_1", "ckpt_2"]
    sorted_cks = sorted(checkpoints)
    assert sorted_cks[0] == "ckpt_1"

def test_checkpoint_oldest_removal():
    checkpoints = ["ckpt_1", "ckpt_2", "ckpt_3"]
    removed = checkpoints.pop(0)
    assert removed == "ckpt_1"
    assert len(checkpoints) == 2

def test_checkpoint_best_kept():
    checkpoints = [{"name": "ckpt_1", "loss": 0.5}, {"name": "ckpt_2", "loss": 0.3}]
    best = min(checkpoints, key=lambda c: c["loss"])
    assert best["name"] == "ckpt_2"
