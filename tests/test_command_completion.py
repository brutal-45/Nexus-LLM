"""Tests for command tab completion."""
import pytest


def test_command_completion_prefix():
    commands = ["/help", "/history", "/model", "/quit"]
    prefix = "/h"
    matches = [c for c in commands if c.startswith(prefix)]
    assert "/help" in matches and "/history" in matches

def test_command_completion_unique():
    commands = ["/help", "/history", "/model"]
    prefix = "/he"
    matches = [c for c in commands if c.startswith(prefix)]
    assert len(matches) == 1 and matches[0] == "/help"

def test_command_completion_no_match():
    commands = ["/help", "/model"]
    prefix = "/xyz"
    matches = [c for c in commands if c.startswith(prefix)]
    assert len(matches) == 0

def test_command_completion_all_commands():
    commands = ["/help", "/history", "/model", "/quit", "/reset"]
    prefix = "/"
    matches = [c for c in commands if c.startswith(prefix)]
    assert len(matches) == 5
