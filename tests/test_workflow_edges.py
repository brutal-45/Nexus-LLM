"""Tests for nexus_llm.workflow.edges module."""

import pytest
from nexus_llm.workflow.edges import (
    WorkflowEdge,
    EdgeCondition,
    create_simple_edge,
    create_conditional_edge,
    create_default_edge,
)


class TestEdgeCondition:
    def test_always_no_condition(self):
        """No fn or expression means always traverse."""
        cond = EdgeCondition()
        assert cond.evaluate({}) is True
        assert cond.evaluate(None) is True

    def test_with_callable_fn(self):
        cond = EdgeCondition(fn=lambda data: data.get("ok", False))
        assert cond.evaluate({"ok": True}) is True
        assert cond.evaluate({"ok": False}) is False

    def test_with_expression(self):
        cond = EdgeCondition(expression="data == 'success'")
        assert cond.evaluate("success") is True
        assert cond.evaluate("fail") is False

    def test_with_expression_output_alias(self):
        cond = EdgeCondition(expression="output > 5")
        assert cond.evaluate(10) is True
        assert cond.evaluate(3) is False

    def test_negated(self):
        cond = EdgeCondition(fn=lambda data: True, negated=True)
        assert cond.evaluate({}) is False

    def test_negated_false(self):
        cond = EdgeCondition(fn=lambda data: False, negated=True)
        assert cond.evaluate({}) is True

    def test_callable_exception(self):
        """If the callable raises, evaluate should return False."""
        def bad_fn(data):
            raise RuntimeError("boom")

        cond = EdgeCondition(fn=bad_fn)
        assert cond.evaluate({}) is False

    def test_expression_exception(self):
        """If the expression raises, evaluate should return False."""
        cond = EdgeCondition(expression="1/0")
        assert cond.evaluate({}) is False

    def test_description(self):
        cond = EdgeCondition(description="Check status is ok")
        assert cond.description == "Check status is ok"


class TestWorkflowEdge:
    def test_creation(self):
        edge = WorkflowEdge(source="a", target="b")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.condition is None
        assert edge.label == ""
        assert edge.priority == 0

    def test_with_condition(self):
        cond = EdgeCondition(fn=lambda data: True, description="always true")
        edge = WorkflowEdge(source="a", target="b", condition=cond)
        assert edge.condition is not None
        assert edge.condition.description == "always true"

    def test_can_traverse_no_condition(self):
        edge = WorkflowEdge(source="a", target="b")
        assert edge.can_traverse("any data") is True

    def test_can_traverse_with_condition_true(self):
        cond = EdgeCondition(fn=lambda data: True)
        edge = WorkflowEdge(source="a", target="b", condition=cond)
        assert edge.can_traverse({}) is True

    def test_can_traverse_with_condition_false(self):
        cond = EdgeCondition(fn=lambda data: False)
        edge = WorkflowEdge(source="a", target="b", condition=cond)
        assert edge.can_traverse({}) is False

    def test_to_dict(self):
        edge = WorkflowEdge(source="a", target="b", label="my_edge", priority=1)
        d = edge.to_dict()
        assert d["source"] == "a"
        assert d["target"] == "b"
        assert d["label"] == "my_edge"
        assert d["priority"] == 1
        assert "condition" not in d

    def test_to_dict_with_condition(self):
        cond = EdgeCondition(expression="x > 0", description="positive check", negated=True)
        edge = WorkflowEdge(source="a", target="b", condition=cond)
        d = edge.to_dict()
        assert "condition" in d
        assert d["condition"]["expression"] == "x > 0"
        assert d["condition"]["description"] == "positive check"
        assert d["condition"]["negated"] is True

    def test_label_and_metadata(self):
        edge = WorkflowEdge(source="a", target="b", label="test", metadata={"weight": 0.5})
        assert edge.label == "test"
        assert edge.metadata["weight"] == 0.5


class TestCreateSimpleEdge:
    def test_create(self):
        edge = create_simple_edge("a", "b", label="simple")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.label == "simple"
        assert edge.condition is None


class TestCreateConditionalEdge:
    def test_create(self):
        edge = create_conditional_edge(
            "a", "b",
            condition_fn=lambda data: data > 0,
            label="positive",
            description="Check if positive",
        )
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.label == "positive"
        assert edge.condition is not None
        assert edge.condition.description == "Check if positive"


class TestCreateDefaultEdge:
    def test_create(self):
        edge = create_default_edge("a", "b")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.label == "default"
        assert edge.priority == 999
