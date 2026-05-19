"""Nexus-LLM Plugins Module.

Provides a plugin system with lifecycle management, hook system,
built-in plugins, and example plugins for extensibility.
"""

from nexus_llm.plugins.hook import Hook, HookManager, HookPriority
from nexus_llm.plugins.loader import PluginLoader, PluginValidationError
from nexus_llm.plugins.manager import PluginManager, PluginState

# Builtin plugins
from nexus_llm.plugins.builtin.calculator import CalculatorPlugin
from nexus_llm.plugins.builtin.code_runner import CodeRunnerPlugin
from nexus_llm.plugins.builtin.file_manager import FileManagerPlugin
from nexus_llm.plugins.builtin.note_taker import NoteTakerPlugin
from nexus_llm.plugins.builtin.system_monitor import SystemMonitorPlugin
from nexus_llm.plugins.builtin.weather import WeatherPlugin
from nexus_llm.plugins.builtin.web_search import WebSearchPlugin

# Example plugins
from nexus_llm.plugins.examples.custom_greet import CustomGreetPlugin
from nexus_llm.plugins.examples.echo import EchoPlugin

__all__ = [
    # Hook system
    "Hook",
    "HookManager",
    "HookPriority",
    # Loader
    "PluginLoader",
    "PluginValidationError",
    # Manager
    "PluginManager",
    "PluginState",
    # Builtin plugins
    "CalculatorPlugin",
    "CodeRunnerPlugin",
    "FileManagerPlugin",
    "NoteTakerPlugin",
    "SystemMonitorPlugin",
    "WeatherPlugin",
    "WebSearchPlugin",
    # Example plugins
    "CustomGreetPlugin",
    "EchoPlugin",
]
