"""Tests for sentence splitting."""
import pytest
import re


def test_sentence_split_basic():
    text = "Hello world. How are you? I am fine!"
    sentences = re.split(r"[.!?]+", text)
    assert len(sentences) == 4  # Includes empty trailing

def test_sentence_split_preserves_content():
    text = "First sentence. Second sentence."
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    assert len(sentences) == 2

def test_sentence_split_no_punctuation():
    text = "no punctuation here"
    sentences = re.split(r"[.!?]+", text)
    assert len(sentences) == 1

def test_sentence_split_abbreviation():
    text = "Dr. Smith went to Washington."
    # Simple split (doesn't handle abbreviations well)
    sentences = re.split(r"[.!?]+", text)
    assert len(sentences) >= 2

def test_sentence_split_empty():
    text = ""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    assert len(sentences) == 0
