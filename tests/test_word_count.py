"""Tests for word counting."""
import pytest


def test_word_count_basic():
    text = "Hello world this is a test"
    assert len(text.split()) == 6

def test_word_count_empty():
    assert len("".split()) == 0

def test_word_count_with_punctuation():
    text = "Hello, world! This is a test."
    words = text.split()
    assert len(words) == 6

def test_word_count_multiline():
    text = "line one\nline two\nline three"
    assert len(text.split()) == 6

def test_word_count_with_extra_spaces():
    text = "hello   world   test"
    assert len(text.split()) == 3
