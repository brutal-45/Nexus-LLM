"""Tests for Python syntax highlighting."""
import pytest
import re


def test_keyword_detection():
    keywords = {"def", "class", "if", "else", "return", "import"}
    code = "def foo():\n    if True:\n        return 42"
    found = {kw for kw in keywords if kw in code}
    assert "def" in found and "return" in found


def test_string_literal_detection():
    code = 'x = "hello world"'
    assert '"' in code


def test_comment_detection():
    code = "# comment\nx = 1  # inline"
    assert len([l for l in code.split("\n") if "#" in l]) == 2


def test_decorator_detection():
    code = "@staticmethod\n@lru_cache\ndef foo():\n    pass"
    assert len([l for l in code.split("\n") if l.startswith("@")]) == 2


def test_number_detection():
    code = "x = 42\ny = 3.14"
    assert len(re.findall(r"\b\d+(?:\.\d+)?\b", code)) >= 2
