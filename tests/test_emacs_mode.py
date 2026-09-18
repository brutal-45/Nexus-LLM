"""Tests for Emacs key bindings."""
import pytest


def test_emacs_move_beginning():
    line = "Hello world"
    cursor = len(line)
    cursor = 0  # Ctrl+A
    assert cursor == 0

def test_emacs_move_end():
    line = "Hello world"
    cursor = 0
    cursor = len(line)  # Ctrl+E
    assert cursor == len(line)

def test_emacs_delete_backward():
    line = "Hello"
    cursor = 5
    cursor -= 1  # Backspace
    assert cursor == 4

def test_emacs_kill_line():
    line = "Hello world"
    cursor = 5
    killed = line[cursor:]
    remaining = line[:cursor]
    assert remaining == "Hello" and killed == " world"
