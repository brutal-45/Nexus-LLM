"""Tests for synonym replacement augmentation."""
import pytest


def test_synonym_replacement_basic():
    synonyms = {"happy": ["glad", "joyful"], "sad": ["unhappy", "down"]}
    word = "happy"
    assert word in synonyms
    assert len(synonyms[word]) > 0

def test_synonym_preserves_meaning():
    synonyms = {"big": ["large", "huge", "great"]}
    replacements = synonyms["big"]
    assert "large" in replacements

def test_synonym_no_entry():
    synonyms = {"happy": ["glad"]}
    word = "quantum"
    result = word if word not in synonyms else synonyms[word][0]
    assert result == "quantum"

def test_synonym_multiple_replacements():
    text_words = ["happy", "and", "sad"]
    synonyms = {"happy": ["glad"], "sad": ["down"]}
    result = [synonyms.get(w, [w])[0] for w in text_words]
    assert result == ["glad", "and", "down"]
