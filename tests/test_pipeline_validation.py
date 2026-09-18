"""Tests for nexus_llm.pipeline.validation module."""

import pytest
from nexus_llm.pipeline.validation import PipelineValidator


class TestPipelineValidator:
    """Tests for the PipelineValidator class."""

    def test_init(self):
        validator = PipelineValidator()
        assert validator is not None

    def test_validate_text(self):
        validator = PipelineValidator()
        result = validator.validate("Hello world", schema="text")
        assert result.is_valid is True

    def test_validate_empty(self):
        validator = PipelineValidator()
        result = validator.validate("", schema="text")
        assert result.is_valid is False

    def test_validate_too_long(self):
        validator = PipelineValidator()
        result = validator.validate("x" * 10000, schema="text", max_length=1000)
        assert result.is_valid is False

    def test_validate_json(self):
        validator = PipelineValidator()
        result = validator.validate('{"key": "value"}', schema="json")
        assert result.is_valid is True

    def test_validate_invalid_json(self):
        validator = PipelineValidator()
        result = validator.validate("not json", schema="json")
        assert result.is_valid is False
