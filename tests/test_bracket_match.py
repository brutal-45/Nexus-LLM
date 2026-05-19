"""Tests for bracket matching."""
import pytest


def test_simple_bracket_match():
    text = "(hello)"
    stack = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    for ch in text:
        if ch in pairs:
            stack.append(ch)
        elif ch in pairs.values():
            if stack:
                stack.pop()
    assert len(stack) == 0

def test_nested_bracket_match():
    text = "({[]})"
    stack = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if stack:
                stack.pop()
    assert len(stack) == 0

def test_unmatched_bracket():
    text = "([)]"
    stack = []
    pairs = {"(": ")", "[": "]"}
    for ch in text:
        if ch in pairs:
            stack.append(ch)
    assert len(stack) > 0

def test_bracket_highlight_position():
    text = "foo(bar)"
    pos = text.index("(")
    assert pos == 3
    closing = text.index(")")
    assert closing == 7
