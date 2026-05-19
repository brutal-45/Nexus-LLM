"""Chat agent for conversational interactions.

Provides a conversational agent that maintains context across
messages, asks follow-up questions, seeks clarification, and
provides coherent multi-turn dialogue.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.agents.base import Agent, AgentAction, AgentAction, AgentConfig, AgentObservation
from nexus_llm.agents.memory import ShortTermMemory, AgentMemory
from nexus_llm.agents.tools import Tool

logger = logging.getLogger(__name__)


class ConversationContext:
    """Manages conversation context and history."""

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self._messages: List[Dict[str, str]] = []
        self._summary: str = ""

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self._messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self._messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get conversation messages."""
        return list(self._messages)

    def get_recent_messages(self, n: int = 5) -> List[Dict[str, str]]:
        """Get the n most recent messages."""
        return self._messages[-n:]

    def format_for_prompt(self, n: int = 10) -> str:
        """Format recent messages for inclusion in a prompt."""
        recent = self._messages[-n:]
        parts = []
        for msg in recent:
            role = msg["role"].capitalize()
            content = msg["content"][:500]  # Truncate long messages
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _trim_history(self) -> None:
        """Trim history to max size, keeping the most recent messages."""
        if len(self._messages) > self.max_history * 2:
            # Keep a summary of older messages
            old_messages = self._messages[:len(self._messages) - self.max_history]
            topics = self._extract_topics(old_messages)
            self._summary = f"Earlier conversation topics: {', '.join(topics[:5])}"
            self._messages = self._messages[-self.max_history:]

    def _extract_topics(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extract key topics from messages (simple keyword extraction)."""
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "i", "you", "me",
            "my", "your", "we", "our", "it", "its", "this", "that", "and",
            "or", "but", "in", "on", "at", "to", "for", "of", "with", "from",
            "can", "could", "would", "should", "do", "does", "did", "have",
            "has", "had", "will", "be", "been", "not", "no", "so", "if",
            "what", "how", "why", "when", "where", "who", "which",
        }

        word_counts: Dict[str, int] = {}
        for msg in messages:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", msg["content"].lower())
            for word in words:
                if word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""


class ChatAgent(Agent):
    """Conversational agent with context tracking and follow-up capabilities.

    Maintains a conversation context across messages, tracks topics,
    asks clarifying questions when needed, and provides coherent
    multi-turn dialogue.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Dict[str, Tool]] = None,
        memory: Optional[AgentMemory] = None,
        llm_fn: Optional[Callable] = None,
        max_context_messages: int = 20,
        system_prompt: Optional[str] = None,
    ):
        config = config or AgentConfig(name="ChatAgent", description="A conversational agent")
        super().__init__(config=config, tools=tools, memory=memory, llm_fn=llm_fn)

        self.conversation = ConversationContext(max_history=max_context_messages)
        self.system_prompt = system_prompt or (
            "You are a helpful, friendly conversational assistant. "
            "Answer questions clearly, ask for clarification when needed, "
            "and maintain a natural conversation flow."
        )

    def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a chat message and return a response.

        Args:
            message: The user's message.
            context: Optional additional context.

        Returns:
            The agent's response.
        """
        self.conversation.add_user_message(message)

        # Use the base agent's run method
        response = self.run(message, context=context)

        self.conversation.add_assistant_message(response)
        return response

    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Optional[AgentAction]:
        """Decide how to respond to the user message."""
        conversation_context = self.conversation.format_for_prompt(n=10)

        # Build the thinking prompt
        prompt_parts = [
            f"System: {self.system_prompt}",
            f"\nConversation so far:\n{conversation_context}",
            f"\nCurrent message: {task}",
        ]

        # Check if we have tools that might be relevant
        relevant_tools = self._find_relevant_tools(task)
        if relevant_tools:
            tool_names = [t.name for t in relevant_tools]
            prompt_parts.append(f"\nPotentially relevant tools: {', '.join(tool_names)}")

        if self.llm_fn:
            return self._llm_think(task, prompt_parts, relevant_tools)
        else:
            return self._rule_think(task, relevant_tools)

    def _llm_think(self, task: str, prompt_parts: List[str], relevant_tools: List[Tool]) -> Optional[AgentAction]:
        """Use LLM to decide how to respond."""
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in relevant_tools
        ) if relevant_tools else "No tools needed."

        prompt = "\n".join(prompt_parts) + (
            f"\n\nDecide how to respond. Either:\n"
            f"1. Call a tool (if it would help answer the question)\n"
            f"2. Respond directly to the user\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"Respond in JSON format:\n"
            f'{{"action": "tool_call", "tool": "tool_name", "args": {{...}}, "thought": "why"}}\n'
            f'or\n'
            f'{{"action": "respond", "response": "your response", "thought": "why"}}'
        )

        try:
            response = self.llm_fn(prompt)
            return self._parse_llm_response(response, relevant_tools)
        except Exception as e:
            logger.error("LLM thinking failed: %s", e)
            return AgentAction(action_type="respond", response=f"I'd be happy to help, but I encountered a processing error. Could you rephrase your question?")

    def _parse_llm_response(self, response: str, relevant_tools: List[Tool]) -> Optional[AgentAction]:
        """Parse the LLM's response into an AgentAction."""
        # Try to extract JSON from the response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                action_type = data.get("action", "respond")

                if action_type == "tool_call":
                    tool_name = data.get("tool", "")
                    tool_args = data.get("args", {})
                    thought = data.get("thought", "")

                    # Verify tool exists
                    if tool_name in self.tools:
                        return AgentAction(
                            action_type="tool_call",
                            tool_name=tool_name,
                            tool_args=tool_args,
                            thought=thought,
                        )

                elif action_type == "respond":
                    return AgentAction(
                        action_type="respond",
                        response=data.get("response", ""),
                        thought=data.get("thought", ""),
                    )
            except json.JSONDecodeError:
                pass

        # Fallback: treat the whole response as a direct reply
        return AgentAction(action_type="respond", response=response.strip())

    def _rule_think(self, task: str, relevant_tools: List[Tool]) -> AgentAction:
        """Rule-based thinking when LLM is not available."""
        task_lower = task.lower()

        # Check for simple queries that can be answered directly
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
        if any(g in task_lower for g in greetings):
            return AgentAction(
                action_type="respond",
                response="Hello! How can I help you today?",
                thought="User greeted, responding with greeting.",
            )

        # Check for calculator-related queries
        if any(t.name == "calculator" for t in relevant_tools):
            math_patterns = [
                r"(?:what is|calculate|compute|solve)\s+(.+)",
                r"(\d+\s*[\+\-\*\/\^]\s*\d+.*)",
            ]
            for pattern in math_patterns:
                match = re.search(pattern, task_lower)
                if match:
                    expr = match.group(1).strip().rstrip("?")
                    return AgentAction(
                        action_type="tool_call",
                        tool_name="calculator",
                        tool_args={"expression": expr},
                        thought=f"Detected math expression: {expr}",
                    )

        # Check for weather-related queries
        if any(t.name == "weather" for t in relevant_tools) and "weather" in task_lower:
            city_match = re.search(r"weather\s+(?:in|for|at)\s+(\w+(?:\s+\w+)?)", task_lower)
            if city_match:
                city = city_match.group(1).strip()
                return AgentAction(
                    action_type="tool_call",
                    tool_name="weather",
                    tool_args={"location": city},
                    thought=f"Detected weather query for: {city}",
                )

        # Check for search queries
        if any(t.name == "search" for t in relevant_tools):
            search_triggers = ["search", "look up", "find information", "what is", "who is", "tell me about"]
            if any(trigger in task_lower for trigger in search_triggers):
                query = re.sub(r"^(search|look up|find information about|what is|who is|tell me about)\s+", "", task_lower, flags=re.IGNORECASE)
                return AgentAction(
                    action_type="tool_call",
                    tool_name="search",
                    tool_args={"query": query.strip()},
                    thought=f"Detected search query: {query}",
                )

        # Default: respond conversationally
        return AgentAction(
            action_type="respond",
            response=self._generate_conversational_response(task),
            thought="No specific tool needed, responding conversationally.",
        )

    def _find_relevant_tools(self, message: str) -> List[Tool]:
        """Find tools that might be relevant to the message."""
        relevant = []
        message_lower = message.lower()

        keyword_map = {
            "calculator": ["calculate", "math", "compute", "solve", "equation", "+", "-", "*", "/", "sum", "average"],
            "search": ["search", "look up", "find", "information", "what is", "who is"],
            "weather": ["weather", "temperature", "forecast", "rain", "sunny"],
            "file_read": ["read file", "open file", "show file", "cat", "file content"],
            "file_write": ["write file", "save file", "create file", "store"],
            "code_run": ["run code", "execute", "python", "script", "program"],
        }

        for tool_name, keywords in keyword_map.items():
            if tool_name in self.tools:
                if any(kw in message_lower for kw in keywords):
                    relevant.append(self.tools[tool_name])

        return relevant

    def _generate_conversational_response(self, message: str) -> str:
        """Generate a conversational response without LLM."""
        if self.conversation.message_count <= 1:
            return f"I understand you're asking about: {message}. Could you provide more details so I can help you better?"

        recent = self.conversation.get_recent_messages(3)
        context_summary = "Based on our conversation, "

        if len(recent) > 1:
            context_summary += f"we've been discussing related topics. "

        context_summary += f"Regarding your question about '{message[:50]}', "
        context_summary += "I'd need more specific information to give you a detailed answer. Could you clarify what you're looking for?"

        return context_summary

    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if self.conversation._summary:
            return self.conversation._summary
        topics = self.conversation._extract_topics(self.conversation.get_messages())
        if topics:
            return f"Topics discussed: {', '.join(topics[:5])}"
        return "No conversation yet."

    def reset_conversation(self) -> None:
        """Reset the conversation context."""
        self.conversation.clear()
        self.reset()
