"""Tests for nexus_llm.workflow.nodes module."""

import pytest
from nexus_llm.workflow.nodes import WorkflowNode, NodeType, NodeStatus


class TestNodeType:
    def test_values(self):
        assert NodeType.START.value == "start"
        assert NodeType.END.value == "end"
        assert NodeType.PROCESS.value == "process"
        assert NodeType.DECISION.value == "decision"


class TestNodeStatus:
    def test_values(self):
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.RUNNING.value == "running"
        assert NodeStatus.COMPLETED.value == "completed"
        assert NodeStatus.FAILED.value == "failed"


class TestWorkflowNode:
    def test_creation(self):
        node = WorkflowNode(id="test", type=NodeType.PROCESS, fn=lambda: 42)
        assert node.id == "test"
        assert node.type == NodeType.PROCESS
        assert node.status == NodeStatus.PENDING

    def test_execute(self):
        node = WorkflowNode(id="test", type=NodeType.PROCESS, fn=lambda: 42)
        result = node.execute()
        assert result == 42
        assert node.status == NodeStatus.COMPLETED

    def test_execute_failure(self):
        def failing():
            raise RuntimeError("fail")
        node = WorkflowNode(id="test", type=NodeType.PROCESS, fn=failing)
        with pytest.raises(RuntimeError):
            node.execute()
        assert node.status == NodeStatus.FAILED

    def test_reset(self):
        node = WorkflowNode(id="test", type=NodeType.PROCESS, fn=lambda: 42)
        node.execute()
        node.reset()
        assert node.status == NodeStatus.PENDING

    def test_to_dict(self):
        node = WorkflowNode(id="test", type=NodeType.PROCESS, fn=lambda: 42, name="TestNode")
        d = node.to_dict()
        assert d["id"] == "test"
        assert d["type"] == "process"
