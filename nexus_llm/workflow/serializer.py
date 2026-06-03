"""Nexus-LLM Workflow Serialization.

Provides serialization and deserialization of workflow definitions
to/from JSON and YAML formats, enabling workflow persistence and
sharing.
"""

import json
import logging
from typing import Any, Dict, Optional

from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType, NodeStatus
from nexus_llm.workflow.edges import WorkflowEdge, EdgeCondition

logger = logging.getLogger(__name__)

# Try to import yaml
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


class WorkflowSerializer:
    """Serializes and deserializes workflow definitions.

    Example::

        serializer = WorkflowSerializer()
        json_str = serializer.to_json(engine)
        restored = serializer.from_json(json_str)
    """

    def to_dict(self, engine: WorkflowEngine) -> Dict[str, Any]:
        """Serialize a workflow engine to a dictionary.

        Args:
            engine: The workflow engine to serialize.

        Returns:
            Dictionary representation of the workflow.
        """
        nodes = {}
        for nid, node in engine._nodes.items():
            node_dict = {
                "id": node.id,
                "type": node.type.value,
                "name": node.name,
                "status": node.status.value,
            }
            if node.config:
                node_dict["config"] = node.config
            nodes[nid] = node_dict

        edges = []
        for edge in engine._edges:
            edge_dict = {"source": edge.source, "target": edge.target}
            if edge.condition is not None:
                edge_dict["has_condition"] = True
            edges.append(edge_dict)

        return {
            "id": engine.id,
            "name": engine.name,
            "description": engine.config.description,
            "config": {
                "max_retries": engine.config.max_retries,
                "retry_delay_seconds": engine.config.retry_delay_seconds,
                "timeout_seconds": engine.config.timeout_seconds,
                "continue_on_error": engine.config.continue_on_error,
            },
            "nodes": nodes,
            "edges": edges,
        }

    def to_json(self, engine: WorkflowEngine, indent: int = 2) -> str:
        """Serialize a workflow engine to JSON.

        Args:
            engine: The workflow engine to serialize.
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        data = self.to_dict(engine)
        return json.dumps(data, indent=indent)

    def to_yaml(self, engine: WorkflowEngine) -> str:
        """Serialize a workflow engine to YAML.

        Args:
            engine: The workflow engine to serialize.

        Returns:
            YAML string representation.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not _HAS_YAML:
            raise ImportError("PyYAML is required for YAML serialization")
        data = self.to_dict(engine)
        return yaml.dump(data, default_flow_style=False)

    def from_dict(self, data: Dict[str, Any]) -> WorkflowEngine:
        """Deserialize a workflow engine from a dictionary.

        Args:
            data: Dictionary representation of the workflow.

        Returns:
            A WorkflowEngine instance.
        """
        config_data = data.get("config", {})
        config = WorkflowConfig(
            name=data.get("name", ""),
            description=data.get("description", ""),
            max_retries=config_data.get("max_retries", 0),
            retry_delay_seconds=config_data.get("retry_delay_seconds", 1.0),
            timeout_seconds=config_data.get("timeout_seconds", 3600.0),
            continue_on_error=config_data.get("continue_on_error", False),
        )
        engine = WorkflowEngine(config=config)

        for nid, node_data in data.get("nodes", {}).items():
            node_type = NodeType(node_data.get("type", "task"))
            node = WorkflowNode(
                id=nid,
                type=node_type,
                name=node_data.get("name", ""),
                config=node_data.get("config"),
            )
            engine.add_node(node)

        for edge_data in data.get("edges", []):
            condition = None
            if edge_data.get("has_condition"):
                condition = EdgeCondition()
            engine.add_edge(
                source=edge_data["source"],
                target=edge_data["target"],
                condition=condition,
            )

        return engine

    def from_json(self, json_str: str) -> WorkflowEngine:
        """Deserialize a workflow engine from JSON.

        Args:
            json_str: JSON string representation.

        Returns:
            A WorkflowEngine instance.
        """
        data = json.loads(json_str)
        return self.from_dict(data)

    def from_yaml(self, yaml_str: str) -> WorkflowEngine:
        """Deserialize a workflow engine from YAML.

        Args:
            yaml_str: YAML string representation.

        Returns:
            A WorkflowEngine instance.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not _HAS_YAML:
            raise ImportError("PyYAML is required for YAML deserialization")
        data = yaml.safe_load(yaml_str)
        return self.from_dict(data)

    def save(self, engine: WorkflowEngine, filepath: str) -> None:
        """Save a workflow engine to a file.

        Args:
            engine: The workflow engine to save.
            filepath: Path to save the file. Extension determines format.
        """
        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            content = self.to_yaml(engine)
        else:
            content = self.to_json(engine)

        with open(filepath, "w") as f:
            f.write(content)
        logger.info("Workflow saved to %s", filepath)

    def load(self, filepath: str) -> WorkflowEngine:
        """Load a workflow engine from a file.

        Args:
            filepath: Path to the workflow file.

        Returns:
            A WorkflowEngine instance.
        """
        with open(filepath, "r") as f:
            content = f.read()

        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            return self.from_yaml(content)
        return self.from_json(content)
