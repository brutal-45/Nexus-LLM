"""Tests for nexus_llm.workflow.validators module."""

import pytest
from nexus_llm.workflow.validators import WorkflowValidator, ValidationError
from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType


class TestValidationError:
    def test_creation(self):
        err = ValidationError("nodes.a", "Node missing function")
        assert err.path == "nodes.a"
        assert err.message == "Node missing function"
        assert err.severity == "error"

    def test_str(self):
        err = ValidationError("nodes.a", "Missing function")
        assert "ERROR" in str(err)

    def test_to_dict(self):
        err = ValidationError("nodes.a", "Missing function", severity="warning")
        d = err.to_dict()
        assert d["severity"] == "warning"


class TestWorkflowValidator:
    def test_init(self):
        validator = WorkflowValidator()
        assert validator is not None

    def test_validate_empty(self):
        validator = WorkflowValidator()
        engine = WorkflowEngine()
        errors = validator.validate(engine)
        assert len(errors) > 0  # Empty workflow has errors

    def test_validate_valid(self):
        validator = WorkflowValidator()
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge("start", "end")
        errors = validator.validate(engine)
        error_messages = [e.message for e in errors if e.severity == "error"]
        assert len(error_messages) == 0

    def test_validate_no_start(self):
        validator = WorkflowValidator()
        engine = WorkflowEngine()
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda: None))
        errors = validator.validate(engine)
        assert any("START" in e.message for e in errors)

    def test_validate_no_end(self):
        validator = WorkflowValidator()
        engine = WorkflowEngine()
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        errors = validator.validate(engine)
        assert any("END" in e.message for e in errors)

    def test_is_valid(self):
        validator = WorkflowValidator()
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge("start", "end")
        assert validator.is_valid(engine) is True

    def test_custom_rule(self):
        def custom_rule(engine):
            if engine.node_count > 5:
                return [ValidationError("workflow", "Too many nodes", severity="warning")]
            return []

        validator = WorkflowValidator(custom_rules=[custom_rule])
        engine = WorkflowEngine()
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        errors = validator.validate(engine)
        assert len(errors) > 0  # At least structural errors
