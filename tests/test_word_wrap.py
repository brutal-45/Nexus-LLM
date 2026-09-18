"""Tests for word wrapping."""
import pytest
import textwrap


def test_word_wrap_basic():
    wrapped = textwrap.fill("The quick brown fox jumps over the lazy dog.", width=20)
    for line in wrapped.split("\n"):
        assert len(line) <= 20


def test_word_wrap_long_word():
    wrapped = textwrap.fill("supercalifragilisticexpialidocious is long", width=20)
    assert len(wrapped) > 0


def test_word_wrap_preserves_content():
    text = "Hello world this is a test"
    assert textwrap.fill(text, width=15).replace("\n", " ") == text


def test_word_wrap_empty_string():
    assert textwrap.fill("", width=20) == ""


def test_word_wrap_short_text():
    assert textwrap.fill("Short", width=80) == "Short"
