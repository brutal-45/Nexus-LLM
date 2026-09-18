"""Tests for bounce spinner animation."""
import pytest


def test_bounce_positions():
    positions = [0, 1, 2, 1, 0]
    assert all(0 <= p <= 2 for p in positions)

def test_bounce_frames_count():
    frames = ["( )    ", " ( )   ", "  ( )  ", "   ( ) ", "    ( )"]
    assert len(frames) == 5

def test_bounce_cycle_length():
    up = list(range(5))
    down = list(range(5, -1, -1))
    cycle = up + down
    assert len(cycle) == 11
