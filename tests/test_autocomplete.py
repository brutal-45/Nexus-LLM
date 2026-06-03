"""Tests for autocomplete."""
import pytest


class AutoCompleter:
    """Simple autocomplete for testing."""
    def __init__(self, commands):
        self.commands = sorted(commands)

    def complete(self, prefix):
        if not prefix:
            return []
        return [cmd for cmd in self.commands if cmd.startswith(prefix)]

    def closest_match(self, prefix):
        matches = self.complete(prefix)
        return matches[0] if matches else None


@pytest.fixture
def completer():
    return AutoCompleter(["/help", "/clear", "/save", "/load", "/model", "/quit", "/theme"])


def test_autocomplete_exact_prefix(completer):
    """Test autocomplete with exact prefix."""
    results = completer.complete("/he")
    assert "/help" in results


def test_autocomplete_no_match(completer):
    """Test autocomplete with no matches."""
    results = completer.complete("/xyz")
    assert len(results) == 0


def test_autocomplete_multiple_matches(completer):
    """Test autocomplete with multiple matches."""
    results = completer.complete("/c")
    assert "/clear" in results
    assert len(results) >= 1


def test_autocomplete_empty_prefix(completer):
    """Test autocomplete with empty prefix."""
    results = completer.complete("")
    assert len(results) == 0


def test_autocomplete_full_command(completer):
    """Test autocomplete with full command."""
    results = completer.complete("/help")
    assert "/help" in results


def test_closest_match(completer):
    """Test finding closest match."""
    match = completer.closest_match("/he")
    assert match == "/help"


def test_closest_match_none(completer):
    """Test closest match returns None for no matches."""
    assert completer.closest_match("/xyz") is None
