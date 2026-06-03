"""Nexus-LLM Workflow Visualizer.

Provides visualization capabilities for workflow graphs, including
ASCII rendering, dictionary representation, and DOT format output
for integration with Graphviz and other visualization tools.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from nexus_llm.workflow.engine import WorkflowEngine
from nexus_llm.workflow.nodes import WorkflowNode, NodeType, NodeStatus
from nexus_llm.workflow.edges import WorkflowEdge

logger = logging.getLogger(__name__)

# Unicode box-drawing characters for ASCII visualization
_CHARS = {
    "horizontal": "─",
    "vertical": "│",
    "corner_tl": "┌",
    "corner_tr": "┐",
    "corner_bl": "└",
    "corner_br": "┘",
    "tee_right": "├",
    "tee_left": "┤",
    "tee_down": "┬",
    "tee_up": "┴",
    "cross": "┼",
    "arrow_right": "→",
    "arrow_down": "↓",
}

# Node type icons for visual distinction
_NODE_ICONS = {
    NodeType.START: "▶",
    NodeType.END: "■",
    NodeType.TASK: "●",
    NodeType.DECISION: "◆",
    NodeType.PARALLEL: "≡",
    NodeType.SUBPROCESS: "⊞",
    NodeType.TOOL_CALL: "⚙",
    NodeType.LLM_CALL: "🤖",
    NodeType.TRANSFORM: "↔",
    NodeType.WAIT: "⏳",
}

# Status indicators
_STATUS_ICONS = {
    NodeStatus.PENDING: "⏳",
    NodeStatus.RUNNING: "🔄",
    NodeStatus.COMPLETED: "✓",
    NodeStatus.FAILED: "✗",
    NodeStatus.SKIPPED: "⊘",
    NodeStatus.CANCELLED: "⊘",
}


class WorkflowVisualizer:
    """Visualizer for workflow graphs.

    Provides multiple rendering formats for workflow visualization,
    including ASCII art, structured dictionaries, and DOT format.

    Example::

        viz = WorkflowVisualizer()
        ascii_art = viz.render_ascii(engine)
        dot_str = viz.render_dot(engine)
    """

    def render_ascii(self, engine: WorkflowEngine) -> str:
        """Render the workflow as an ASCII diagram.

        Args:
            engine: The workflow engine to visualize.

        Returns:
            A string containing the ASCII representation of the workflow.
        """
        if engine.node_count == 0:
            return "(empty workflow)"

        lines: List[str] = []
        lines.append(f"Workflow: {engine.name or engine.id}")
        lines.append(f"Nodes: {engine.node_count}  Edges: {engine.edge_count}")
        lines.append("")

        # Get topological order
        try:
            order = engine.topological_sort()
        except ValueError:
            lines.append("⚠ Cycle detected - showing nodes in insertion order")
            order = list(engine._nodes.keys())

        # Render each node with its connections
        for i, node_id in enumerate(order):
            node = engine.get_node(node_id)
            if node is None:
                continue

            icon = _NODE_ICONS.get(node.type, "?")
            status = _STATUS_ICONS.get(node.status, "?")
            name = node.name or node_id

            # Node line
            lines.append(f"  {icon} [{node_id}] {name} ({node.type.value}) {status}")

            # Show edges
            edges = engine.get_edges_from(node_id)
            for edge in edges:
                condition_label = ""
                if edge.condition is not None:
                    condition_label = f" ({edge.condition.description or 'conditional'})"
                label = f" {edge.label}" if edge.label else ""
                lines.append(f"    {_CHARS['arrow_right']} {edge.target}{label}{condition_label}")

            if i < len(order) - 1:
                lines.append(f"    {_CHARS['vertical']}")

        return "\n".join(lines)

    def render_dict(self, engine: WorkflowEngine) -> Dict[str, Any]:
        """Render the workflow as a structured dictionary.

        Args:
            engine: The workflow engine to visualize.

        Returns:
            A dictionary representation suitable for JSON serialization
            or further processing.
        """
        result: Dict[str, Any] = {
            "id": engine.id,
            "name": engine.name,
            "description": engine.config.description,
            "stats": {
                "node_count": engine.node_count,
                "edge_count": engine.edge_count,
                "start_nodes": engine.get_start_nodes(),
                "end_nodes": engine.get_end_nodes(),
            },
            "nodes": {},
            "edges": [],
        }

        # Nodes
        for nid, node in engine._nodes.items():
            node_dict: Dict[str, Any] = {
                "id": node.id,
                "name": node.name,
                "type": node.type.value,
                "status": node.status.value,
            }
            if node.config:
                node_dict["config"] = node.config
            result["nodes"][nid] = node_dict

        # Edges
        for edge in engine._edges:
            edge_dict: Dict[str, Any] = {
                "source": edge.source,
                "target": edge.target,
                "priority": edge.priority,
            }
            if edge.label:
                edge_dict["label"] = edge.label
            if edge.condition is not None:
                edge_dict["has_condition"] = True
                edge_dict["condition_description"] = edge.condition.description
            result["edges"].append(edge_dict)

        return result

    def render_dot(self, engine: WorkflowEngine) -> str:
        """Render the workflow in DOT format for Graphviz.

        Args:
            engine: The workflow engine to visualize.

        Returns:
            A DOT format string that can be rendered with Graphviz.
        """
        lines: List[str] = []
        lines.append("digraph workflow {")
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=box, style=rounded];')
        lines.append("")

        # Node styles by type
        node_styles = {
            NodeType.START: 'shape=ellipse, style=filled, fillcolor="#4CAF50", fontcolor=white',
            NodeType.END: 'shape=ellipse, style=filled, fillcolor="#F44336", fontcolor=white',
            NodeType.DECISION: 'shape=diamond, style=filled, fillcolor="#FF9800", fontcolor=white',
            NodeType.TASK: 'shape=box, style=filled, fillcolor="#2196F3", fontcolor=white',
            NodeType.LLM_CALL: 'shape=box, style=filled, fillcolor="#9C27B0", fontcolor=white',
            NodeType.TOOL_CALL: 'shape=box, style=filled, fillcolor="#607D8B", fontcolor=white',
            NodeType.PARALLEL: 'shape=box, style=filled, fillcolor="#00BCD4", fontcolor=white',
        }

        # Nodes
        for nid, node in engine._nodes.items():
            label = node.name or nid
            style = node_styles.get(node.type, "shape=box")
            lines.append(f'  "{nid}" [label="{label}\\n({node.type.value})", {style}];')

        lines.append("")

        # Edges
        for edge in engine._edges:
            label = edge.label or ""
            if edge.condition is not None:
                cond_label = edge.condition.description or "conditional"
                label = f"{label}: {cond_label}" if label else cond_label
            if label:
                lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{label}"];')
            else:
                lines.append(f'  "{edge.source}" -> "{edge.target}";')

        lines.append("}")
        return "\n".join(lines)

    def render_mermaid(self, engine: WorkflowEngine) -> str:
        """Render the workflow in Mermaid format.

        Args:
            engine: The workflow engine to visualize.

        Returns:
            A Mermaid diagram string suitable for markdown rendering.
        """
        lines: List[str] = []
        lines.append("graph TD")

        # Nodes
        for nid, node in engine._nodes.items():
            label = node.name or nid
            if node.type == NodeType.START:
                lines.append(f'    {nid}(["{label}"])')
            elif node.type == NodeType.END:
                lines.append(f'    {nid}(["{label}"])')
            elif node.type == NodeType.DECISION:
                lines.append(f'    {nid}{{"{label}"}}')
            else:
                lines.append(f'    {nid}["{label}"]')

        # Edges
        for edge in engine._edges:
            label = f"|{edge.label}|" if edge.label else ""
            if edge.condition is not None:
                cond_label = edge.condition.description or "conditional"
                label = f"|{cond_label}|" if not edge.label else f"|{edge.label}: {cond_label}|"
            lines.append(f"    {edge.source} -->{label} {edge.target}")

        return "\n".join(lines)

    def render_summary(self, engine: WorkflowEngine) -> str:
        """Render a text summary of the workflow.

        Args:
            engine: The workflow engine to summarize.

        Returns:
            A human-readable summary string.
        """
        lines: List[str] = []
        lines.append(f"Workflow: {engine.name or 'Unnamed'}")
        lines.append(f"  ID: {engine.id}")
        lines.append(f"  Nodes: {engine.node_count}")
        lines.append(f"  Edges: {engine.edge_count}")
        lines.append(f"  Start nodes: {engine.get_start_nodes()}")
        lines.append(f"  End nodes: {engine.get_end_nodes()}")

        # Validation status
        errors = engine.validate()
        if errors:
            lines.append(f"  Validation: {len(errors)} error(s)")
            for err in errors:
                lines.append(f"    - {err}")
        else:
            lines.append("  Validation: OK")

        return "\n".join(lines)
