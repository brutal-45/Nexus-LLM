"""Tests for the pipeline module.

Covers Pipeline, PipelineBuilder, and PipelineStep.
"""

from __future__ import annotations

import pytest

from nexus_llm.pipeline.pipeline import Pipeline
from nexus_llm.pipeline.builder import PipelineBuilder
from nexus_llm.pipeline.step import PipelineStep


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------

class TestPipelineStep:
    """Tests for PipelineStep."""

    def test_create_step(self):
        step = PipelineStep(name="preprocess", func=lambda x: x.strip())
        assert step.name == "preprocess"

    def test_run_step(self):
        step = PipelineStep(name="upper", func=lambda x: x.upper())
        result = step.run("hello")
        assert result == "HELLO"

    def test_step_with_kwargs(self):
        def multiply(x, factor=2):
            return x * factor

        step = PipelineStep(name="multiply", func=multiply, kwargs={"factor": 3})
        result = step.run(5)
        assert result == 15


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    """Tests for Pipeline."""

    def test_create_pipeline(self):
        pipeline = Pipeline(name="test-pipeline")
        assert pipeline.name == "test-pipeline"

    def test_add_step_and_run(self):
        pipeline = Pipeline(name="test")
        pipeline.add_step(PipelineStep(name="add1", func=lambda x: x + 1))
        pipeline.add_step(PipelineStep(name="double", func=lambda x: x * 2))
        result = pipeline.run(5)
        assert result == 12  # (5+1)*2

    def test_empty_pipeline(self):
        pipeline = Pipeline(name="empty")
        result = pipeline.run("input")
        assert result == "input"

    def test_pipeline_with_error(self):
        pipeline = Pipeline(name="error")
        pipeline.add_step(PipelineStep(name="fail", func=lambda x: 1 / 0))
        with pytest.raises(Exception):
            pipeline.run(None)

    def test_list_steps(self):
        pipeline = Pipeline(name="test")
        pipeline.add_step(PipelineStep(name="s1", func=lambda x: x))
        pipeline.add_step(PipelineStep(name="s2", func=lambda x: x))
        steps = pipeline.list_steps()
        assert len(steps) == 2


# ---------------------------------------------------------------------------
# PipelineBuilder
# ---------------------------------------------------------------------------

class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_build_pipeline(self):
        builder = PipelineBuilder()
        pipeline = builder.name("my-pipeline").build()
        assert pipeline.name == "my-pipeline"

    def test_build_with_steps(self):
        builder = PipelineBuilder()
        pipeline = (
            builder
            .name("test")
            .add_step("step1", lambda x: x + 1)
            .add_step("step2", lambda x: x * 2)
            .build()
        )
        result = pipeline.run(3)
        assert result == 8  # (3+1)*2

    def test_fluent_api(self):
        builder = PipelineBuilder()
        result = builder.name("test").add_step("s1", lambda x: x)
        assert result is builder  # returns self for fluent usage
