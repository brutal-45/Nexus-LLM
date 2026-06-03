"""Tool dataclass — the core unit describing an LLM-callable tool."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Describes a tool that an LLM can invoke.

    Attributes
    ----------
    name:
        Unique tool identifier (e.g. ``"calculator"``).
    description:
        Human-readable description used in prompts / function-calling schemas.
    func:
        The underlying Python callable.
    parameters_schema:
        A JSON-schema-style dict describing the expected parameters.
        Example::

            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
    """

    name: str
    description: str
    func: Callable
    parameters_schema: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> Any:
        """Run the tool with the supplied *kwargs*.

        Raises
        ------
        TypeError
            If parameter validation fails.
        Exception
            Any exception raised by the underlying callable is propagated.
        """
        self.validate_params(**kwargs)
        logger.debug("Tool %r executing with args %s", self.name, list(kwargs.keys()))
        try:
            result = self.func(**kwargs)
        except Exception as exc:
            logger.error("Tool %r execution failed: %s", self.name, exc)
            raise
        logger.debug("Tool %r completed successfully", self.name)
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_params(self, **kwargs: Any) -> bool:
        """Validate *kwargs* against :attr:`parameters_schema`.

        Checks that all ``required`` keys are present and that the type of
        each provided value matches the declared ``type`` where possible.

        Returns
        -------
        ``True`` if validation passes.

        Raises
        ------
        TypeError
            If a required parameter is missing or a type mismatch is detected.
        """
        schema = self.parameters_schema
        if not schema:
            return True

        properties: Dict[str, Any] = schema.get("properties", {})
        required: List[str] = schema.get("required", [])

        # Check required keys
        for key in required:
            if key not in kwargs:
                raise TypeError(
                    f"Tool {self.name!r} missing required parameter {key!r}"
                )

        # Basic type checking
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for key, value in kwargs.items():
            if key not in properties:
                continue
            expected_type_name = properties[key].get("type")
            if expected_type_name and expected_type_name in type_map:
                expected = type_map[expected_type_name]
                if not isinstance(value, expected):
                    raise TypeError(
                        f"Tool {self.name!r} parameter {key!r} expected "
                        f"{expected_type_name}, got {type(value).__name__}"
                    )
        return True

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation (excludes the callable)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters_schema": self.parameters_schema,
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, description={self.description!r})"
