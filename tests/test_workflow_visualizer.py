"""Tests for workflow visualization utilities."""

import pytest
from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType
from nexus_llm.workflow.visualizer import WorkflowVisualizer


class TestWorkflowVisualizer:
    def test_init(self):
        viz = WorkflowVisualizer()
        assert viz is not None

    def test_render_ascii(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda: None))
        engine.add_edge("start", "end")
        viz = WorkflowVisualizer()
        output = viz.render_ascii(engine)
        assert isinstance(output, str)
        assert "start" in output
        assert "end" in output

    def test_render_ascii_empty(self):
        engine = WorkflowEngine()
        viz = WorkflowVisualizer()
        output = viz.render_ascii(engine)
        assert "empty" in output.lower()

    def test_render_dict(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda: None))
        engine.add_edge("start", "end")
        viz = WorkflowVisualizer()
        output = viz.render_dict(engine)
        assert isinstance(output, dict)
        assert "nodes" in output
        assert "edges" in output
        assert output["name"] == "test"

    def test_render_dot(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda: None))
        engine.add_edge("start", "end")
        viz = WorkflowVisualizer()
        output = viz.render_dot(engine)
        assert isinstance(output, str)
        assert "digraph" in output
        assert "start" in output
        assert "end" in output

    def test_render_mermaid(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda: None))
        engine.add_edge("start", "end")
        viz = WorkflowVisualizer()
        output = viz.render_mermaid(engine)
        assert isinstance(output, str)
        assert "graph" in output
        assert "start" in output

    def test_render_summary(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda: None))
        engine.add_edge("start", "end")
        viz = WorkflowVisualizer()
        output = viz.render_summary(engine)
        assert isinstance(output, str)
        assert "test" in output
        assert "Nodes" in output

    def test_render_dict_stats(self):
        engine = WorkflowEngine(WorkflowConfig(name="test"))
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="process", type=NodeType.TASK, fn=lambda x: x))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge("start", "process")
        engine.add_edge("process", "end")
        viz = WorkflowVisualizer()
        output = viz.render_dict(engine)
        assert output["stats"]["node_count"] == 3
        assert output["stats"]["edge_count"] == 2
        assert "start" in output["stats"]["start_nodes"]
        assert "end" in output["stats"]["end_nodes"]
