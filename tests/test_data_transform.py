"""Tests for nexus_llm.pipeline.transform module."""

import pytest
from nexus_llm.pipeline.transform import DataTransform


class TestDataTransform:
    """Tests for the DataTransform class."""

    def test_init(self):
        transform = DataTransform()
        assert transform is not None

    def test_apply(self):
        transform = DataTransform()
        result = transform.apply("  hello  ", operation="strip")
        assert result == "hello"

    def test_apply_lower(self):
        transform = DataTransform()
        result = transform.apply("HELLO", operation="lowercase")
        assert result == "hello"

    def test_apply_upper(self):
        transform = DataTransform()
        result = transform.apply("hello", operation="uppercase")
        assert result == "HELLO"

    def test_apply_unknown(self):
        transform = DataTransform()
        with pytest.raises(ValueError):
            transform.apply("hello", operation="unknown")

    def test_batch_apply(self):
        transform = DataTransform()
        results = transform.batch_apply(["  a  ", "  b  "], operation="strip")
        assert results == ["a", "b"]
