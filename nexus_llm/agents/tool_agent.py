"""Tool-using agent with function calling capabilities.

Provides an agent specialized in tool selection, argument extraction,
result parsing, and multi-step tool usage for task completion.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.agents.base import Agent, AgentAction, AgentConfig, AgentObservation
from nexus_llm.agents.executor import ActionExecutor
from nexus_llm.agents.memory import AgentMemory, ShortTermMemory
from nexus_llm.agents.tools import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolAgent(Agent):
    """Agent specialized in tool usage and function calling.

    Excels at selecting appropriate tools, extracting arguments
    from natural language queries, executing tools via the
    ActionExecutor, and parsing results to form coherent responses.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Dict[str, Tool]] = None,
        memory: Optional[AgentMemory] = None,
        llm_fn: Optional[Callable] = None,
        executor: Optional[ActionExecutor] = None,
    ):
        config = config or AgentConfig(
            name="ToolAgent",
            description="An agent specialized in using tools to complete tasks",
            max_iterations=8,
        )
        super().__init__(config=config, tools=tools, memory=memory, llm_fn=llm_fn)

        self.executor = executor or ActionExecutor(tools=self.tools)

    def add_tool(self, tool: Tool) -> None:
        """Register a tool with the agent and executor."""
        super().add_tool(tool)
        self.executor.register_tool(tool)

    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Optional[AgentAction]:
        """Decide which tool to use based on the task."""
        # If we already have observations, check if we can respond
        if self.observation_history:
            last_obs = self.observation_history[-1]
            if last_obs.success and self._is_task_complete(task):
                final_response = self._compose_response(task)
                return AgentAction(
                    action_type="respond",
                    response=final_response,
                    thought="Task appears complete, composing final response.",
                )

        # Try LLM-based tool selection
        if self.llm_fn:
            return self._llm_select_tool(task)

        # Fall back to rule-based tool selection
        return self._rule_select_tool(task)

    def act(self, action: AgentAction) -> AgentObservation:
        """Execute an action, using the executor for tool calls."""
        if action.action_type == "tool_call" and action.tool_name:
            result = self.executor.execute(action.tool_name, **(action.tool_args or {}))
            return AgentObservation(
                action=action,
                result=result,
                observation_text=result.output if result.success else f"Error: {result.error}",
                success=result.success,
            )
        return super().act(action)

    def _llm_select_tool(self, task: str) -> Optional[AgentAction]:
        """Use LLM to select and configure a tool call."""
        tool_descriptions = []
        for name, tool in self.tools.items():
            params_desc = json.dumps(tool.parameters, indent=2) if tool.parameters else "{}"
            tool_descriptions.append(f"Tool: {name}\nDescription: {tool.description}\nParameters: {params_desc}")

        tools_text = "\n\n".join(tool_descriptions)

        # Include observation history for context
        obs_text = ""
        if self.observation_history:
            obs_parts = []
            for i, obs in enumerate(self.observation_history[-3:]):
                obs_parts.append(f"Step {i + 1}: {obs.observation_text[:200]}")
            obs_text = f"\nPrevious results:\n" + "\n".join(obs_parts)

        prompt = (
            f"You are a tool-using agent. Select the best tool to complete the task.\n\n"
            f"Task: {task}{obs_text}\n\n"
            f"Available tools:\n{tools_text}\n\n"
            f"Respond with JSON:\n"
            f'{{"tool": "tool_name", "args": {{"param": "value"}}, "thought": "reasoning"}}\n'
            f'Or if the task is complete:\n'
            f'{{"respond": "final answer", "thought": "reasoning"}}'
        )

        try:
            response = self.llm_fn(prompt)
            return self._parse_tool_response(response)
        except Exception as e:
            logger.error("LLM tool selection failed: %s", e)
            return self._rule_select_tool(task)

    def _parse_tool_response(self, response: str) -> Optional[AgentAction]:
        """Parse LLM response into a tool call or direct response."""
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if not json_match:
            return AgentAction(action_type="respond", response=response.strip())

        try:
            data = json.loads(json_match.group())

            if "respond" in data:
                return AgentAction(
                    action_type="respond",
                    response=data["respond"],
                    thought=data.get("thought", ""),
                )

            tool_name = data.get("tool", "")
            tool_args = data.get("args", {})
            thought = data.get("thought", "")

            if tool_name in self.tools:
                return AgentAction(
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    thought=thought,
                )
            else:
                return AgentAction(
                    action_type="respond",
                    response=f"I don't have the tool '{tool_name}' available. Let me try a different approach.",
                    thought=f"Requested tool '{tool_name}' not found.",
                )
        except json.JSONDecodeError:
            return AgentAction(action_type="respond", response=response.strip())

    def _rule_select_tool(self, task: str) -> Optional[AgentAction]:
        """Rule-based tool selection."""
        task_lower = task.lower()

        # Pattern matching for tool selection
        tool_patterns = {
            "calculator": [
                (r"(?:calculate|compute|what is|solve|evaluate)\s+([\d\+\-\*\/\.\(\)\s\^]+)", "expression"),
            ],
            "search": [
                (r"(?:search|look up|find)\s+(?:for\s+)?(.+?)(?:\?*$)", "query"),
                (r"(?:what|who|where|when)\s+(?:is|are|was|were)\s+(.+?)(?:\?*$)", "query"),
            ],
            "weather": [
                (r"weather\s+(?:in|for|at)\s+(\w+(?:\s+\w+)?)", "location"),
                (r"(?:temperature|forecast)\s+(?:in|for|at)\s+(\w+(?:\s+\w+)?)", "location"),
            ],
            "file_read": [
                (r"(?:read|open|show|display)\s+(?:file\s+)?(.+?)(?:\?*$)", "path"),
            ],
            "file_write": [
                (r"(?:write|save|create)\s+(?:to\s+)?(?:file\s+)?(.+?)(?:\?*$)", "path"),
            ],
            "code_run": [
                (r"(?:run|execute)\s+(?:code|script|program)\s*:?\s*(.+?)(?:$)", "code"),
                (r"(?:python|py)\s+(.+?)(?:$)", "code"),
            ],
        }

        for tool_name, patterns in tool_patterns.items():
            if tool_name not in self.tools:
                continue

            for pattern, arg_name in patterns:
                match = re.search(pattern, task_lower, re.IGNORECASE)
                if match:
                    arg_value = match.group(1).strip().rstrip("?")
                    return AgentAction(
                        action_type="tool_call",
                        tool_name=tool_name,
                        tool_args={arg_name: arg_value},
                        thought=f"Detected {tool_name} usage with {arg_name}='{arg_value}'",
                    )

        # If no tool matched, respond conversationally
        if self.observation_history:
            return AgentAction(
                action_type="respond",
                response=self._compose_response(task),
                thought="No matching tool found, composing response from observations.",
            )

        return AgentAction(
            action_type="respond",
            response=f"I'm not sure which tool to use for: '{task}'. Could you be more specific about what you'd like me to do?",
            thought="No tool matched the query pattern.",
        )

    def _is_task_complete(self, task: str) -> bool:
        """Heuristic check if the task is likely complete based on observations."""
        if not self.observation_history:
            return False

        # If we have at least one successful observation, consider the task
        # potentially complete after a few iterations
        successful_obs = [o for o in self.observation_history if o.success]
        if len(successful_obs) >= 1 and self.iteration >= 2:
            return True

        return self.iteration >= self.config.max_iterations - 1

    def _compose_response(self, task: str) -> str:
        """Compose a final response from observation history."""
        if not self.observation_history:
            return "I was unable to complete the task."

        parts = []
        for obs in self.observation_history:
            if obs.success and obs.observation_text:
                parts.append(obs.observation_text)

        if not parts:
            errors = [obs.observation_text for obs in self.observation_history if not obs.success]
            if errors:
                return f"I encountered errors while trying to complete the task:\n" + "\n".join(errors)
            return "I was unable to complete the task."

        # Combine results into a coherent response
        if len(parts) == 1:
            return parts[0]

        return "Here are the results:\n\n" + "\n\n".join(f"{i + 1}. {p}" for i, p in enumerate(parts))
