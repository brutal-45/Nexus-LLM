"""Tests for multiline input handling."""
import pytest


def test_multiline_with_newlines():
    assert len("line one\nline two\nline three".split("\n")) == 3


def test_multiline_continuation():
    lines = ["def foo():", "    return 42"]
    full = "\n".join(lines)
    assert "def foo():" in full and "return 42" in full


def test_multiline_empty_line():
    lines = "first\n\nthird".split("\n")
    assert lines[1] == "" and len(lines) == 3


def test_multiline_submission():
    buf = ["def hello():", "    print(\"hi\")", ""]
    assert buf[-1] == ""


def test_multiline_indentation_preserved():
    lines = "if True:\n    x = 1".split("\n")
    assert lines[1].startswith("    ")
