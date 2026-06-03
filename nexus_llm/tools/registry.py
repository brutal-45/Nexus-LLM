"""Nexus-LLM Tool Registry.

Provides the ToolRegistry class that manages tool registration, discovery,
and execution routing.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type

from nexus_llm.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_global_registry: Optional["ToolRegistry"] = None


class ToolRegistry:
    """Central registry for managing and accessing tools.

    The ToolRegistry maintains a mapping of tool names to tool instances,
    supporting registration, unregistration, discovery, and batch execution.

    Example::

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = registry.execute("calculator", expression="2 + 2")
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._aliases: Dict[str, str] = {}
        logger.debug("ToolRegistry initialized")

    def register(self, tool: BaseTool, aliases: Optional[List[str]] = None) -> None:
        """Register a tool instance.

        Args:
            tool: The tool instance to register.
            aliases: Optional list of alternative names.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        if aliases:
            for alias in aliases:
                self._aliases[alias] = tool.name
        logger.info("Registered tool: %s", tool.name)

    def unregister(self, name: str) -> Optional[BaseTool]:
        """Unregister a tool by name.

        Args:
            name: Tool name or alias.

        Returns:
            The unregistered tool, or None if not found.
        """
        canonical = self._aliases.pop(name, name)
        # Also remove aliases pointing to this tool
        self._aliases = {a: t for a, t in self._aliases.items() if t != canonical}
        tool = self._tools.pop(canonical, None)
        if tool:
            logger.info("Unregistered tool: %s", canonical)
        return tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name or alias.

        Args:
            name: Tool name or alias.

        Returns:
            The tool instance, or None if not found.
        """
        canonical = self._aliases.get(name, name)
        return self._tools.get(canonical)

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a registered tool by name.

        Args:
            name: Tool name or alias.
            **kwargs: Parameters to pass to the tool.

        Returns:
            A ToolResult from the tool execution.

        Raises:
            KeyError: If the tool is not found.
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        return tool.run(**kwargs)

    def list_tools(self) -> List[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def list_enabled(self) -> List[str]:
        """Return sorted list of enabled tool names."""
        return sorted(name for name, tool in self._tools.items() if tool.enabled)

    def schemas(self) -> List[Dict[str, Any]]:
        """Return schemas for all registered tools."""
        return [tool.schema() for tool in self._tools.values()]

    def enable(self, name: str) -> None:
        """Enable a tool.

        Args:
            name: Tool name or alias.

        Raises:
            KeyError: If the tool is not found.
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        tool.enabled = True

    def disable(self, name: str) -> None:
        """Disable a tool.

        Args:
            name: Tool name or alias.

        Raises:
            KeyError: If the tool is not found.
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        tool.enabled = False

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()})"


def get_registry() -> ToolRegistry:
    """Return the global ToolRegistry, creating it if necessary.

    Returns:
        The global ToolRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry
