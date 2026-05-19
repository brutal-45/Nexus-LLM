"""Tests for nexus_llm.workflow.engine module."""

import pytest
from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType
from nexus_llm.workflow.edges import WorkflowEdge


class TestWorkflowConfig:
    def test_default(self):
        config = WorkflowConfig()
        assert config.max_retries == 0
        assert config.timeout_seconds == 3600.0

    def test_custom(self):
        config = WorkflowConfig(name="test", max_retries=3)
        assert config.name == "test"
        assert config.max_retries == 3


class TestWorkflowEngine:
    def test_init(self):
        engine = WorkflowEngine()
        assert engine.node_count == 0
        assert engine.edge_count == 0

    def test_add_node(self):
        engine = WorkflowEngine()
        node = WorkflowNode(id="n1", type=NodeType.START, fn=lambda: "start")
        engine.add_node(node)
        assert engine.node_count == 1

    def test_add_duplicate_node(self):
        engine = WorkflowEngine()
        node = WorkflowNode(id="n1", type=NodeType.START, fn=lambda: "start")
        engine.add_node(node)
        with pytest.raises(ValueError):
            engine.add_node(node)

    def test_remove_node(self):
        engine = WorkflowEngine()
        node = WorkflowNode(id="n1", type=NodeType.START, fn=lambda: "start")
        engine.add_node(node)
        assert engine.remove_node("n1") is True
        assert engine.node_count == 0

    def test_remove_nonexistent(self):
        engine = WorkflowEngine()
        assert engine.remove_node("missing") is False

    def test_add_edge(self):
        engine = WorkflowEngine()
        engine.add_node(WorkflowNode(id="a", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="b", type=NodeType.END, fn=lambda: None))
        engine.add_edge("a", "b")
        assert engine.edge_count == 1

    def test_add_edge_missing_node(self):
        engine = WorkflowEngine()
        engine.add_node(WorkflowNode(id="a", type=NodeType.START, fn=lambda: None))
        with pytest.raises(ValueError):
            engine.add_edge("a", "missing")

    def test_validate_no_start(self):
        engine = WorkflowEngine()
        engine.add_node(WorkflowNode(id="b", type=NodeType.END, fn=lambda: None))
        errors = engine.validate()
        assert any("START" in e for e in errors)

    def test_validate_valid(self):
        engine = WorkflowEngine()
        engine.add_node(WorkflowNode(id="a", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="b", type=NodeType.END, fn=lambda: None))
        engine.add_edge("a", "b")
        errors = engine.validate()
        assert errors == []

    def test_to_dict(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        d = engine.to_dict()
        assert d["name"] == "test"
