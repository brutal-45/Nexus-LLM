"""Tests for workflow visualization utilities."""

import pytest
from unittest.mock import MagicMock
from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType


class TestWorkflowVisualizer:
    def test_init(self):
        from nexus_llm.workflow.visualizer import WorkflowVisualizer
        viz = WorkflowVisualizer()
        assert viz is not None

    def test_render_ascii(self):
        from nexus_llm.workflow.visualizer import WorkflowVisualizer
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda: None))
        engine.add_edge("start", "end")
        viz = WorkflowVisualizer()
        output = viz.render_ascii(engine)
        assert isinstance(output, str)

    def test_render_dict(self):
        from nexus_llm.workflow.visualizer import WorkflowVisualizer
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        viz = WorkflowVisualizer()
        output = viz.render_dict(engine)
        assert isinstance(output, dict)
