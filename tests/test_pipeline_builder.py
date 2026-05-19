"""Tests for nexus_llm.pipeline.builder module."""

import pytest
from nexus_llm.pipeline.builder import PipelineBuilder, BuiltPipeline, StepType


class TestPipelineBuilder:
    """Tests for the PipelineBuilder class."""

    def test_init(self):
        builder = PipelineBuilder()
        assert builder.name is not None
        assert builder.step_count == 0

    def test_init_with_name(self):
        builder = PipelineBuilder(name="test_pipeline")
        assert builder.name == "test_pipeline"

    def test_add_preprocess_step(self):
        builder = PipelineBuilder()
        builder.preprocess(lambda x: x.strip())
        assert builder.step_count == 1

    def test_add_process_step(self):
        builder = PipelineBuilder()
        builder.process(lambda x: x.upper())
        assert builder.step_count == 1

    def test_add_postprocess_step(self):
        builder = PipelineBuilder()
        builder.postprocess(lambda x: x.strip())
        assert builder.step_count == 1

    def test_add_validate_step(self):
        builder = PipelineBuilder()
        builder.validate(lambda x: len(x) > 0)
        assert builder.step_count == 1

    def test_chaining(self):
        builder = PipelineBuilder()
        result = (
            builder
            .preprocess(lambda x: x.strip())
            .process(lambda x: x.upper())
            .postprocess(lambda x: x + "!")
        )
        assert result is builder
        assert builder.step_count == 3

    def test_with_config(self):
        builder = PipelineBuilder()
        builder.with_config(timeout=30, retries=3)
        assert builder.step_count == 0  # config doesn't add steps

    def test_build(self):
        pipeline = (
            PipelineBuilder("test")
            .preprocess(lambda x: x.strip())
            .process(lambda x: x.upper())
            .build()
        )
        assert isinstance(pipeline, BuiltPipeline)
        assert pipeline.name == "test"
        assert len(pipeline.steps) == 2


class TestBuiltPipeline:
    """Tests for the BuiltPipeline class."""

    def test_run(self):
        pipeline = (
            PipelineBuilder()
            .preprocess(lambda x: x.strip())
            .process(lambda x: x.upper())
            .build()
        )
        result = pipeline.run("  hello  ")
        assert result == "HELLO"

    def test_run_empty_pipeline(self):
        pipeline = PipelineBuilder().build()
        result = pipeline.run("hello")
        assert result == "hello"

    def test_run_with_error_skip(self):
        def failing_step(x):
            raise ValueError("fail")

        pipeline = (
            PipelineBuilder()
            .process(lambda x: x.upper(), name="upper")
            .process(failing_step, name="fail")
            .on_error("skip")
            .build()
        )
        result = pipeline.run("hello")
        assert result == "HELLO"  # Failing step skipped

    def test_run_with_error_default(self):
        def failing_step(x):
            raise ValueError("fail")

        pipeline = (
            PipelineBuilder()
            .process(failing_step, name="fail")
            .on_error("default")
            .build()
        )
        result = pipeline.run("hello")
        assert result is None  # Default output

    def test_to_dict(self):
        pipeline = (
            PipelineBuilder("test")
            .process(lambda x: x)
            .build()
        )
        d = pipeline.to_dict()
        assert d["name"] == "test"
        assert len(d["steps"]) == 1
