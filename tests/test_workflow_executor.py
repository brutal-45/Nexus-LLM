"""Tests for nexus_llm.workflow.executor module."""

import pytest
from nexus_llm.workflow.executor import WorkflowExecutor, ExecutionResult
from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType, NodeResult


class TestExecutionResult:
    def test_default(self):
        result = ExecutionResult()
        assert result.success is True
        assert result.node_results == {}
        assert result.nodes_completed == 0
        assert result.nodes_failed == 0
        assert result.error is None

    def test_custom(self):
        result = ExecutionResult(
            workflow_id="wf-1",
            success=False,
            error="something failed",
            total_duration_ms=150.0,
        )
        assert result.workflow_id == "wf-1"
        assert result.success is False
        assert result.error == "something failed"
        assert result.total_duration_ms == 150.0


class TestWorkflowExecutor:
    def test_init_with_engine(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        executor = WorkflowExecutor(engine)
        assert executor is not None

    def test_execute_simple(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: 1))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x * 2))
        engine.add_edge("start", "end")
        executor = WorkflowExecutor(engine)
        result = executor.execute()
        assert result is not None
        assert result.success is True
        assert result.nodes_completed == 2
        assert result.nodes_failed == 0

    def test_execute_with_initial_data(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: "hello"))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x.upper()))
        engine.add_edge("start", "end")
        executor = WorkflowExecutor(engine)
        result = executor.execute(initial_data="hello")
        assert result.success is True
        assert result.node_results["end"].output == "HELLO"

    def test_execute_three_step(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: 10))
        engine.add_node(WorkflowNode(id="double", type=NodeType.TASK, fn=lambda x: x * 2))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x + 5))
        engine.add_edge("start", "double")
        engine.add_edge("double", "end")
        executor = WorkflowExecutor(engine)
        result = executor.execute()
        assert result.success is True
        assert result.nodes_completed == 3
        assert result.node_results["end"].output == 25

    def test_execute_validation_fails(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        # No start/end nodes → validation should fail
        executor = WorkflowExecutor(engine)
        result = executor.execute()
        assert result.success is False
        assert result.error is not None
        assert "validation" in result.error.lower()

    def test_execute_node_failure(self):
        def bad_fn(x):
            raise RuntimeError("node failed")

        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: 1))
        engine.add_node(WorkflowNode(id="bad", type=NodeType.TASK, fn=bad_fn))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge("start", "bad")
        engine.add_edge("bad", "end")
        executor = WorkflowExecutor(engine)
        result = executor.execute()
        assert result.success is False
        assert result.nodes_failed > 0

    def test_execute_continue_on_error(self):
        def bad_fn(x):
            raise RuntimeError("node failed")

        engine = WorkflowEngine(WorkflowConfig(name="test", continue_on_error=True))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: 1))
        engine.add_node(WorkflowNode(id="bad", type=NodeType.TASK, fn=bad_fn))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge("start", "bad")
        engine.add_edge("bad", "end")
        executor = WorkflowExecutor(engine)
        result = executor.execute()
        assert result.nodes_failed > 0

    def test_execute_with_retries(self):
        attempt = {"count": 0}

        def flaky_fn(x):
            attempt["count"] += 1
            if attempt["count"] < 3:
                raise RuntimeError("not yet")
            return "success"

        engine = WorkflowEngine(WorkflowConfig(name="test", max_retries=3))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="flaky", type=NodeType.TASK, fn=flaky_fn, max_retries=3))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge("start", "flaky")
        engine.add_edge("flaky", "end")
        executor = WorkflowExecutor(engine)
        result = executor.execute()
        assert result.success is True
        assert result.node_results["flaky"].output == "success"

    def test_execution_result_has_duration(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge("start", "end")
        executor = WorkflowExecutor(engine)
        result = executor.execute()
        assert result.total_duration_ms > 0
