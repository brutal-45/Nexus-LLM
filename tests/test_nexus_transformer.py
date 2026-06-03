"""Tests for nexus_llm.nexus.transformer module."""

import pytest
from nexus_llm.nexus.transformer import NexusTransformer


class TestNexusTransformer:
    """Tests for the NexusTransformer class."""

    def test_init_default(self):
        transformer = NexusTransformer()
        assert transformer is not None

    def test_transform_uppercase(self):
        transformer = NexusTransformer()
        result = transformer.transform("hello world", mode="uppercase")
        assert result == "HELLO WORLD"

    def test_transform_lowercase(self):
        transformer = NexusTransformer()
        result = transformer.transform("HELLO WORLD", mode="lowercase")
        assert result == "hello world"

    def test_transform_title(self):
        transformer = NexusTransformer()
        result = transformer.transform("hello world", mode="title")
        assert result == "Hello World"

    def test_transform_strip(self):
        transformer = NexusTransformer()
        result = transformer.transform("  hello  ", mode="strip")
        assert result == "hello"

    def test_transform_unknown_mode(self):
        transformer = NexusTransformer()
        with pytest.raises(ValueError):
            transformer.transform("hello", mode="unknown")

    def test_batch_transform(self):
        transformer = NexusTransformer()
        results = transformer.batch_transform(
            ["hello", "WORLD"], mode="uppercase"
        )
        assert results == ["HELLO", "WORLD"]

    def test_chain_transforms(self):
        transformer = NexusTransformer()
        result = transformer.chain_transform("  hello world  ", ["strip", "title"])
        assert result == "Hello World"
