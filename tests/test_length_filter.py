"""Tests for length filtering."""
import pytest


def test_min_length_filter():
    min_len = 10
    texts = ["short", "this is a longer text that passes"]
    filtered = [t for t in texts if len(t.split()) >= min_len]
    assert len(filtered) == 1

def test_max_length_filter():
    max_len = 5
    texts = ["hi", "hello world foo bar baz"]
    filtered = [t for t in texts if len(t.split()) <= max_len]
    assert len(filtered) == 1

def test_length_filter_range():
    min_len, max_len = 3, 10
    texts = ["a b", "a b c d e", "a b c d e f g h i j k l m"]
    filtered = [t for t in texts if min_len <= len(t.split()) <= max_len]
    assert len(filtered) == 1

def test_character_length_filter():
    max_chars = 100
    text = "x" * 200
    assert len(text) > max_chars

def test_length_filter_empty():
    texts = []
    filtered = [t for t in texts if len(t.split()) >= 1]
    assert len(filtered) == 0
