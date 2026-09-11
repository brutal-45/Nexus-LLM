"""Tests for nexus_llm.pipeline.postprocess module."""

import pytest
from nexus_llm.pipeline.postprocess import Postprocessor


class TestPostprocessor:
    """Tests for the Postprocessor class."""

    def test_init(self):
        pipeline = Postprocessor()
        assert pipeline is not None

    def test_add_step(self):
        pipeline = Postprocessor()
        pipeline.add_step(lambda x: x.strip())
        assert pipeline.step_count == 1

    def test_run(self):
        pipeline = Postprocessor()
        pipeline.add_step(lambda x: x.strip())
        pipeline.add_step(lambda x: x.capitalize())
        result = pipeline.run("  hello world  ")
        assert result == "Hello world"

    def test_clear(self):
        pipeline = Postprocessor()
        pipeline.add_step(lambda x: x)
        pipeline.clear()
        assert pipeline.step_count == 0
