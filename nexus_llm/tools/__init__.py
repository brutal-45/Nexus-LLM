"""Nexus-LLM Tools Module.

Provides a collection of built-in tools that can be used by agents
for computation, search, file operations, data processing, and more.
All tools inherit from BaseTool and are registered in the global ToolRegistry.
"""

from nexus_llm.tools.base_tool import BaseTool, ToolResult, ToolParameter
from nexus_llm.tools.registry import ToolRegistry, get_registry
from nexus_llm.tools.calculator import CalculatorTool
from nexus_llm.tools.search import SearchTool
from nexus_llm.tools.file_ops import FileOpsTool
from nexus_llm.tools.shell import ShellTool
from nexus_llm.tools.web_scraper import WebScraperTool
from nexus_llm.tools.json_tool import JSONTool
from nexus_llm.tools.text_tool import TextTool
from nexus_llm.tools.datetime_tool import DateTimeTool
from nexus_llm.tools.math_tool import MathTool
from nexus_llm.tools.sql_tool import SQLTool
from nexus_llm.tools.api_tool import APITool
from nexus_llm.tools.summarizer import SummarizerTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolParameter",
    "ToolRegistry",
    "get_registry",
    "CalculatorTool",
    "SearchTool",
    "FileOpsTool",
    "ShellTool",
    "WebScraperTool",
    "JSONTool",
    "TextTool",
    "DateTimeTool",
    "MathTool",
    "SQLTool",
    "APITool",
    "SummarizerTool",
]
