"""Tests for cursor control."""
import pytest


def test_cursor_move_left():
    cursor = 5
    cursor -= 1
    assert cursor == 4

def test_cursor_move_right():
    cursor = 5
    cursor += 1
    assert cursor == 6

def test_cursor_home():
    cursor = 10
    cursor = 0
    assert cursor == 0

def test_cursor_end():
    line = "Hello world"
    cursor = len(line)
    assert cursor == 11

def test_cursor_bounds():
    line = "Hello"
    cursor = 10
    cursor = min(cursor, len(line))
    assert cursor == 5
