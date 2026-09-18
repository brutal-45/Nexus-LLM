"""Tests for nexus_llm.workflow.serializer module."""

import json
import pytest
from nexus_llm.workflow.serializer import WorkflowSerializer
from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType


class TestWorkflowSerializer:
    def setup_method(self):
        self.serializer = WorkflowSerializer()
        self.engine = WorkflowEngine(WorkflowConfig(name="test_workflow"))
        self.engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        self.engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        self.engine.add_edge("start", "end")

    def test_to_dict(self):
        data = self.serializer.to_dict(self.engine)
        assert data["name"] == "test_workflow"
        assert "nodes" in data
        assert "edges" in data

    def test_to_json(self):
        json_str = self.serializer.to_json(self.engine)
        data = json.loads(json_str)
        assert data["name"] == "test_workflow"

    def test_from_dict(self):
        data = self.serializer.to_dict(self.engine)
        restored = self.serializer.from_dict(data)
        assert restored.name == "test_workflow"
        assert restored.node_count == 2
        assert restored.edge_count == 1

    def test_from_json(self):
        json_str = self.serializer.to_json(self.engine)
        restored = self.serializer.from_json(json_str)
        assert restored.node_count == 2

    def test_roundtrip(self):
        json_str = self.serializer.to_json(self.engine)
        restored = self.serializer.from_json(json_str)
        assert restored.name == self.engine.name
        assert restored.node_count == self.engine.node_count
        assert restored.edge_count == self.engine.edge_count

    def test_to_yaml(self):
        try:
            yaml_str = self.serializer.to_yaml(self.engine)
            assert isinstance(yaml_str, str)
            assert "test_workflow" in yaml_str
        except ImportError:
            pytest.skip("PyYAML not installed")
