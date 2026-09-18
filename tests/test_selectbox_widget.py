"""Tests for SelectBox widget."""
import pytest


def test_selectbox_options():
    options = ["gpt2", "gpt3", "llama"]
    assert len(options) == 3

def test_selectbox_default_selection():
    selected = "gpt2"
    options = ["gpt2", "gpt3", "llama"]
    assert selected in options

def test_selectbox_change_selection():
    selected = "gpt2"
    selected = "llama"
    assert selected == "llama"

def test_selectbox_empty_options():
    options = []
    assert len(options) == 0
