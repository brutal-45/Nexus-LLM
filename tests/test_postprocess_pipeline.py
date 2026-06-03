"""Tests for nexus_llm.pipeline.postprocess module."""

import pytest
from nexus_llm.pipeline.postprocess import PostprocessPipeline


class TestPostprocessPipeline:
    """Tests for the PostprocessPipeline class."""

    def test_init(self):
        pipeline = PostprocessPipeline()
        assert pipeline is not None

    def test_add_step(self):
        pipeline = PostprocessPipeline()
        pipeline.add_step(lambda x: x.strip())
        assert pipeline.step_count == 1

    def test_run(self):
        pipeline = PostprocessPipeline()
        pipeline.add_step(lambda x: x.strip())
        pipeline.add_step(lambda x: x.capitalize())
        result = pipeline.run("  hello world  ")
        assert result == "Hello world"

    def test_clear(self):
        pipeline = PostprocessPipeline()
        pipeline.add_step(lambda x: x)
        pipeline.clear()
        assert pipeline.step_count == 0
