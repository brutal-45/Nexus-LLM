"""Tests for the pipeline builder's fluent step construction."""

from __future__ import annotations

import pytest

from nexus_llm.pipeline.builder import PipelineBuilder
from nexus_llm.pipeline.pipeline import Pipeline


def test_requires_a_name():
    with pytest.raises(TypeError):
        PipelineBuilder()


def test_builder_builds_a_named_pipeline():
    pipeline = PipelineBuilder("demo").step("upper", lambda text: text.upper()).build()
    assert isinstance(pipeline, Pipeline)
    assert pipeline.name == "demo"


def test_steps_execute_in_order():
    pipeline = (
        PipelineBuilder("chain")
        .step("add", lambda value: value + 1)
        .step("double", lambda value: value * 2)
        .build()
    )
    result = pipeline.run(1)
    output = result.output if hasattr(result, "output") else result
    assert output == 4


def test_builder_returns_itself_for_chaining():
    builder = PipelineBuilder("fluent")
    assert builder.step("a", lambda x: x) is builder


def test_step_config_is_forwarded():
    pipeline = (
        PipelineBuilder("with-config")
        .step("add", lambda value, delta=0: value + delta, config={"delta": 10})
        .build()
    )
    result = pipeline.run(1)
    output = result.output if hasattr(result, "output") else result
    assert output == 11


def test_empty_pipeline_is_invalid():
    pipeline = PipelineBuilder("empty").build()
    with pytest.raises(Exception):
        pipeline.run("x")
