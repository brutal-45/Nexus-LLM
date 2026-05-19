"""Tests for nexus_llm.workflow.executor module."""

import pytest
from nexus_llm.workflow.executor import WorkflowExecutor
from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType


class TestWorkflowExecutor:
    def test_init(self):
        executor = WorkflowExecutor()
        assert executor is not None

    def test_execute_simple(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: 1))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x * 2))
        engine.add_edge("start", "end")
        executor = WorkflowExecutor()
        result = executor.execute(engine)
        assert result is not None

    def test_execute_with_context(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: "hello"))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x.upper()))
        engine.add_edge("start", "end")
        executor = WorkflowExecutor()
        result = executor.execute(engine, context={"key": "value"})
        assert result is not None

    def test_get_execution_log(self):
        executor = WorkflowExecutor()
        log = executor.get_execution_log()
        assert isinstance(log, list)
