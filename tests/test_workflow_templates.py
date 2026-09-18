"""Tests for nexus_llm.workflow.templates module."""

import pytest
from nexus_llm.workflow.templates import (
    SequentialTemplate,
    FanOutFanInTemplate,
    ConditionalBranchTemplate,
    RetryTemplate,
    get_template,
    list_templates,
)
from nexus_llm.workflow.engine import WorkflowEngine


class TestSequentialTemplate:
    def test_build(self):
        template = SequentialTemplate()
        engine = template.build(steps=[lambda x: x.upper(), lambda x: x + "!"])
        assert engine.node_count == 4  # start + 2 steps + end

    def test_build_empty(self):
        template = SequentialTemplate()
        engine = template.build()
        assert engine.node_count == 2  # start + end


class TestFanOutFanInTemplate:
    def test_build(self):
        template = FanOutFanInTemplate()
        engine = template.build(branches=[lambda x: x, lambda x: x])
        assert engine.node_count == 5  # start + 2 branches + merge + end

    def test_build_empty(self):
        template = FanOutFanInTemplate()
        engine = template.build()
        assert engine.node_count == 3  # start + merge + end


class TestConditionalBranchTemplate:
    def test_build(self):
        template = ConditionalBranchTemplate()
        engine = template.build(
            condition_fn=lambda x: x > 0,
            true_fn=lambda x: x * 2,
            false_fn=lambda x: -x,
        )
        assert engine.node_count == 4


class TestRetryTemplate:
    def test_build(self):
        template = RetryTemplate()
        engine = template.build(process_fn=lambda x: x, max_retries=3)
        assert engine.node_count == 3  # start + process + end


class TestTemplateRegistry:
    def test_list_templates(self):
        templates = list_templates()
        assert "sequential" in templates
        assert "fan_out_fan_in" in templates

    def test_get_template(self):
        template = get_template("sequential")
        assert template is not None
        assert isinstance(template, SequentialTemplate)

    def test_get_missing_template(self):
        template = get_template("nonexistent")
        assert template is None
