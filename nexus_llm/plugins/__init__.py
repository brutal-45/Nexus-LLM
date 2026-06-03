"""Plugins module for Nexus-LLM.

Provides a flexible plugin architecture with lifecycle management,
priority-based hooks, and a publish/subscribe event bus.
"""

from nexus_llm.plugins.manager import PluginManager
from nexus_llm.plugins.plugin import Plugin
from nexus_llm.plugins.loader import PluginLoader
from nexus_llm.plugins.hooks import HookSystem
from nexus_llm.plugins.config import PluginConfig
from nexus_llm.plugins.events import EventSystem

__all__ = [
    "PluginManager",
    "Plugin",
    "PluginLoader",
    "HookSystem",
    "PluginConfig",
    "EventSystem",
]
