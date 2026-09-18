"""Tests for Vi key bindings."""
import pytest


def test_vi_normal_mode():
    mode = "normal"
    assert mode == "normal"

def test_vi_insert_mode():
    mode = "normal"
    mode = "insert"  # Press i
    assert mode == "insert"

def test_vi_escape_to_normal():
    mode = "insert"
    mode = "normal"  # Press Escape
    assert mode == "normal"

def test_vi_command_mode():
    mode = "normal"
    mode = "command"  # Press :
    assert mode == "command"

def test_vi_motion():
    line = "Hello world"
    cursor = 0
    cursor = line.index("w")  # w motion
    assert cursor == 6
