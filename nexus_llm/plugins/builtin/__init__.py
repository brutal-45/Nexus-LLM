"""Builtin plugins package for Nexus-LLM."""

from nexus_llm.plugins.builtin.calculator import CalculatorPlugin
from nexus_llm.plugins.builtin.code_runner import CodeRunnerPlugin
from nexus_llm.plugins.builtin.file_manager import FileManagerPlugin
from nexus_llm.plugins.builtin.note_taker import NoteTakerPlugin
from nexus_llm.plugins.builtin.system_monitor import SystemMonitorPlugin
from nexus_llm.plugins.builtin.weather import WeatherPlugin
from nexus_llm.plugins.builtin.web_search import WebSearchPlugin

__all__ = [
    "CalculatorPlugin",
    "CodeRunnerPlugin",
    "FileManagerPlugin",
    "NoteTakerPlugin",
    "SystemMonitorPlugin",
    "WeatherPlugin",
    "WebSearchPlugin",
]
