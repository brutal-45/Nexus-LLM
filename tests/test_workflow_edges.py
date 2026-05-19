"""Tests for nexus_llm.workflow.edges module."""

import pytest
from nexus_llm.workflow.edges import WorkflowEdge, EdgeCondition


class TestEdgeCondition:
    def test_always(self):
        cond = EdgeCondition()
        assert cond.evaluate({}) is True

    def test_with_expression(self):
        cond = EdgeCondition(expression="status == 'success'")
        result = cond.evaluate({"status": "success"})
        assert isinstance(result, bool)

    def test_with_callable(self):
        cond = EdgeCondition(predicate=lambda ctx: ctx.get("ok", False))
        assert cond.evaluate({"ok": True}) is True
        assert cond.evaluate({"ok": False}) is False


class TestWorkflowEdge:
    def test_creation(self):
        edge = WorkflowEdge(source="a", target="b")
        assert edge.source == "a"
        assert edge.target == "b"

    def test_with_condition(self):
        cond = EdgeCondition(predicate=lambda ctx: True)
        edge = WorkflowEdge(source="a", target="b", condition=cond)
        assert edge.condition is not None

    def test_to_dict(self):
        edge = WorkflowEdge(source="a", target="b")
        d = edge.to_dict()
        assert d["source"] == "a"
        assert d["target"] == "b"
