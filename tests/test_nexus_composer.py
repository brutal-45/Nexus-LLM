"""Tests for nexus_llm.nexus.composer module."""

import pytest
from nexus_llm.nexus.composer import NexusComposer


class TestNexusComposer:
    """Tests for the NexusComposer class."""

    def test_init_default(self):
        composer = NexusComposer()
        assert composer is not None

    def test_compose_simple(self):
        composer = NexusComposer()
        result = composer.compose("Hello, {name}!", context={"name": "World"})
        assert result == "Hello, World!"

    def test_compose_multiple_vars(self):
        composer = NexusComposer()
        result = composer.compose(
            "{greeting}, {name}! Today is {day}.",
            context={"greeting": "Hello", "name": "Alice", "day": "Monday"},
        )
        assert result == "Hello, Alice! Today is Monday."

    def test_compose_missing_var(self):
        composer = NexusComposer()
        with pytest.raises(KeyError):
            composer.compose("Hello, {name}!", context={})

    def test_compose_with_default(self):
        composer = NexusComposer()
        result = composer.compose(
            "Hello, {name}!",
            context={},
            defaults={"name": "User"},
        )
        assert result == "Hello, User!"

    def test_compose_json(self):
        composer = NexusComposer()
        result = composer.compose_json({"message": "Hello", "count": 42})
        assert isinstance(result, str)
        assert "Hello" in result

    def test_compose_markdown(self):
        composer = NexusComposer()
        result = composer.compose_markdown(
            title="Test", body="This is a test."
        )
        assert "# Test" in result
        assert "This is a test." in result
