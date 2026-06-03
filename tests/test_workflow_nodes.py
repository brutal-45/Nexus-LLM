"""Tests for nexus_llm.workflow.nodes module."""

import pytest
from nexus_llm.workflow.nodes import WorkflowNode, NodeType, NodeStatus, NodeResult


class TestNodeType:
    def test_values(self):
        assert NodeType.START.value == "start"
        assert NodeType.END.value == "end"
        assert NodeType.TASK.value == "task"
        assert NodeType.DECISION.value == "decision"
        assert NodeType.PARALLEL.value == "parallel"
        assert NodeType.SUBPROCESS.value == "subprocess"
        assert NodeType.TOOL_CALL.value == "tool_call"
        assert NodeType.LLM_CALL.value == "llm_call"
        assert NodeType.TRANSFORM.value == "transform"
        assert NodeType.WAIT.value == "wait"


class TestNodeStatus:
    def test_values(self):
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.RUNNING.value == "running"
        assert NodeStatus.COMPLETED.value == "completed"
        assert NodeStatus.FAILED.value == "failed"
        assert NodeStatus.SKIPPED.value == "skipped"
        assert NodeStatus.CANCELLED.value == "cancelled"


class TestNodeResult:
    def test_success_result(self):
        result = NodeResult(node_id="n1", success=True, output=42, duration_ms=10.0)
        assert result.node_id == "n1"
        assert result.success is True
        assert result.output == 42

    def test_failure_result(self):
        result = NodeResult(node_id="n1", success=False, error="boom")
        assert result.success is False
        assert result.error == "boom"

    def test_default_values(self):
        result = NodeResult()
        assert result.node_id == ""
        assert result.success is True
        assert result.output is None
        assert result.error is None
        assert result.duration_ms == 0.0
        assert result.metadata == {}


class TestWorkflowNode:
    def test_creation(self):
        node = WorkflowNode(id="test", type=NodeType.TASK, fn=lambda: 42)
        assert node.id == "test"
        assert node.type == NodeType.TASK
        assert node.status == NodeStatus.PENDING

    def test_execute(self):
        node = WorkflowNode(id="test", type=NodeType.TASK, fn=lambda: 42)
        result = node.execute()
        assert isinstance(result, NodeResult)
        assert result.success is True
        assert result.output == 42
        assert node.status == NodeStatus.COMPLETED

    def test_execute_with_input(self):
        node = WorkflowNode(id="test", type=NodeType.TASK, fn=lambda x: x * 2)
        result = node.execute(input_data=5)
        assert result.success is True
        assert result.output == 10

    def test_execute_failure(self):
        def failing(x):
            raise RuntimeError("fail")

        node = WorkflowNode(id="test", type=NodeType.TASK, fn=failing)
        result = node.execute()
        assert result.success is False
        assert result.error == "fail"
        assert node.status == NodeStatus.FAILED

    def test_execute_no_function(self):
        node = WorkflowNode(id="test", type=NodeType.TASK, fn=None)
        result = node.execute(input_data="passthrough")
        assert result.success is True
        assert result.output == "passthrough"

    def test_reset(self):
        node = WorkflowNode(id="test", type=NodeType.TASK, fn=lambda: 42)
        node.execute()
        assert node.status == NodeStatus.COMPLETED
        node.reset()
        assert node.status == NodeStatus.PENDING
        assert node.result is None
        assert node.retries == 0

    def test_on_success_callback(self):
        results = []
        node = WorkflowNode(
            id="test",
            type=NodeType.TASK,
            fn=lambda: 42,
            on_success=lambda output: results.append(output),
        )
        node.execute()
        assert results == [42]

    def test_on_failure_callback(self):
        errors = []
        node = WorkflowNode(
            id="test",
            type=NodeType.TASK,
            fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            on_failure=lambda err: errors.append(err),
        )
        node.execute()
        assert len(errors) == 1
        assert "boom" in errors[0]

    def test_to_dict(self):
        node = WorkflowNode(id="test", type=NodeType.TASK, fn=lambda: 42, name="TestNode")
        d = node.to_dict()
        assert d["id"] == "test"
        assert d["type"] == "task"
        assert d["name"] == "TestNode"
        assert d["status"] == "pending"

    def test_max_retries(self):
        node = WorkflowNode(id="test", type=NodeType.TASK, max_retries=5)
        assert node.max_retries == 5

    def test_timeout_seconds(self):
        node = WorkflowNode(id="test", type=NodeType.TASK, timeout_seconds=30.0)
        assert node.timeout_seconds == 30.0
