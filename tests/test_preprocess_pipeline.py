"""Tests for nexus_llm.pipeline.preprocess module."""

import pytest
from nexus_llm.pipeline.preprocess import PreprocessPipeline


class TestPreprocessPipeline:
    """Tests for the PreprocessPipeline class."""

    def test_init(self):
        pipeline = PreprocessPipeline()
        assert pipeline is not None

    def test_add_step(self):
        pipeline = PreprocessPipeline()
        pipeline.add_step(lambda x: x.strip())
        assert pipeline.step_count == 1

    def test_run(self):
        pipeline = PreprocessPipeline()
        pipeline.add_step(lambda x: x.strip())
        pipeline.add_step(lambda x: x.lower())
        result = pipeline.run("  Hello World  ")
        assert result == "hello world"

    def test_run_empty(self):
        pipeline = PreprocessPipeline()
        result = pipeline.run("hello")
        assert result == "hello"

    def test_clear_steps(self):
        pipeline = PreprocessPipeline()
        pipeline.add_step(lambda x: x)
        pipeline.clear()
        assert pipeline.step_count == 0
