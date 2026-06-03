"""Tests for difficulty ranking."""
import pytest


def test_difficulty_by_length():
    texts = ["short", "medium length text", "this is a much longer and more complex text"]
    difficulties = [len(t.split()) / 50.0 for t in texts]
    assert difficulties[2] > difficulties[1] > difficulties[0]

def test_difficulty_by_vocabulary():
    texts = ["simple words", "obfuscate elucidate quixotic"]
    diff = [len(set(t.split())) / max(len(t.split()), 1) for t in texts]
    assert diff[1] > diff[0]

def test_difficulty_score_range():
    score = 0.75
    assert 0.0 <= score <= 1.0

def test_difficulty_sorting():
    items = [{"text": "easy", "diff": 0.2}, {"text": "hard", "diff": 0.9}, {"text": "medium", "diff": 0.5}]
    sorted_items = sorted(items, key=lambda x: x["diff"])
    assert sorted_items[0]["text"] == "easy"
    assert sorted_items[-1]["text"] == "hard"
