"""Nexus-LLM Agents Module.

Provides intelligent agents with think/act/observe loops, tool usage,
memory management, task planning, and specialized agent implementations.
"""

from nexus_llm.agents.base import Agent, AgentState, AgentConfig
from nexus_llm.agents.chat_agent import ChatAgent
from nexus_llm.agents.code_agent import CodeAgent
from nexus_llm.agents.executor import ActionExecutor
from nexus_llm.agents.memory import (
    AgentMemory,
    EpisodicMemory,
    LongTermMemory,
    ShortTermMemory,
)
from nexus_llm.agents.planner import TaskPlanner, Plan, Step
from nexus_llm.agents.research_agent import ResearchAgent
from nexus_llm.agents.tool_agent import ToolAgent
from nexus_llm.agents.tools import (
    CalculatorTool,
    CodeRunTool,
    FileReadTool,
    FileWriteTool,
    SearchTool,
    Tool,
    ToolResult,
    WeatherTool,
)

__all__ = [
    # Base
    "Agent",
    "AgentState",
    "AgentConfig",
    # Agents
    "ChatAgent",
    "CodeAgent",
    "ResearchAgent",
    "ToolAgent",
    # Executor
    "ActionExecutor",
    # Memory
    "AgentMemory",
    "EpisodicMemory",
    "LongTermMemory",
    "ShortTermMemory",
    # Planner
    "TaskPlanner",
    "Plan",
    "Step",
    # Tools
    "CalculatorTool",
    "CodeRunTool",
    "FileReadTool",
    "FileWriteTool",
    "SearchTool",
    "Tool",
    "ToolResult",
    "WeatherTool",
]
