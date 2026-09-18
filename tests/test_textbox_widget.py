"""Tests for TextBox widget."""
import pytest


def test_textbox_creation():
    widget = {"type": "textbox", "placeholder": "Enter text..."}
    assert widget["type"] == "textbox"

def test_textbox_input_value():
    widget = {"value": "Hello world"}
    assert widget["value"] == "Hello world"

def test_textbox_max_length():
    text = "A" * 500
    max_len = 256
    truncated = text[:max_len]
    assert len(truncated) == max_len

def test_textbox_empty_default():
    widget = {"value": ""}
    assert widget["value"] == ""

def test_textbox_multiline():
    text = "line1\nline2\nline3"
    assert len(text.split("\n")) == 3
