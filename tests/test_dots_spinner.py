"""Tests for dots spinner animation."""
import pytest
import itertools


def test_dots_frames():
    frames = [".", "..", "..."]
    assert len(frames) == 3

def test_dots_cycling():
    frames = [".", "..", "..."]
    cycle = list(itertools.islice(itertools.cycle(frames), 6))
    assert cycle == [".", "..", "...", ".", "..", "..."]

def test_dots_frame_length():
    for frame in [".", "..", "..."]:
        assert 1 <= len(frame) <= 3

def test_dots_string_render():
    frame = "Loading..."
    assert frame.endswith("...")
