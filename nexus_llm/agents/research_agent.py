"""Research agent for multi-step research and synthesis.

Provides an agent that can conduct multi-step research, gather
sources, synthesize findings, and produce cited outputs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.agents.base import Agent, AgentAction, AgentConfig, AgentObservation
from nexus_llm.agents.executor import ActionExecutor
from nexus_llm.agents.memory import AgentMemory, ShortTermMemory
from nexus_llm.agents.planner import Plan, Step, StepStatus, TaskPlanner
from nexus_llm.agents.tools import SearchTool, Tool

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """A research source with citation information."""

    source_id: str
    title: str
    content: str
    relevance: float = 0.0
    url: str = ""
    retrieved_at: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.retrieved_at == 0.0:
            self.retrieved_at = time.time()

    def cite(self) -> str:
        """Generate a citation string."""
        if self.url:
            return f"[{self.source_id}] {self.title} - {self.url}"
        return f"[{self.source_id}] {self.title}"

    def __repr__(self) -> str:
        return f"Source(id={self.source_id}, title='{self.title[:40]}...', relevance={self.relevance:.2f})"


class ResearchAgent(Agent):
    """Agent specialized in multi-step research tasks.

    Conducts systematic research by:
    1. Planning research steps
    2. Searching for and gathering sources
    3. Evaluating source relevance and reliability
    4. Synthesizing findings into coherent output
    5. Providing proper citations
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Dict[str, Tool]] = None,
        memory: Optional[AgentMemory] = None,
        llm_fn: Optional[Callable] = None,
        executor: Optional[ActionExecutor] = None,
        planner: Optional[TaskPlanner] = None,
    ):
        config = config or AgentConfig(
            name="ResearchAgent",
            description="An agent that conducts multi-step research with citations",
            max_iterations=12,
        )
        super().__init__(config=config, tools=tools, memory=memory, llm_fn=llm_fn)

        # Ensure search tool is available
        if "search" not in self.tools:
            self.add_tool(SearchTool())

        self.executor = executor or ActionExecutor(tools=self.tools)
        self.planner = planner or TaskPlanner(llm_fn=llm_fn)

        self._sources: List[Source] = []
        self._research_plan: Optional[Plan] = None
        self._current_step_idx: int = 0
        self._findings: List[str] = []

    def research(self, topic: str, depth: int = 3) -> str:
        """Conduct research on a topic.

        Args:
            topic: The research topic or question.
            depth: Number of research iterations (1-5).

        Returns:
            Research report with citations.
        """
        depth = max(1, min(5, depth))
        self._sources.clear()
        self._findings.clear()
        self._current_step_idx = 0

        # Create a research plan
        self._research_plan = self.planner.create_plan(topic, strategy="research")

        # Execute the research
        return self.run(topic, context={"depth": depth})

    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Optional[AgentAction]:
        """Decide the next research action."""
        depth = (context or {}).get("depth", 3)

        # Phase 1: Initial search
        if self.iteration == 0:
            return AgentAction(
                action_type="tool_call",
                tool_name="search",
                tool_args={"query": task},
                thought=f"Starting research on: {task}",
            )

        # Phase 2: Follow-up searches based on findings
        if self.iteration < depth:
            # Generate follow-up queries
            follow_ups = self._generate_follow_up_queries(task)
            if follow_ups and self.iteration - 1 < len(follow_ups):
                query = follow_ups[self.iteration - 1]
                return AgentAction(
                    action_type="tool_call",
                    tool_name="search",
                    tool_args={"query": query},
                    thought=f"Follow-up search: {query}",
                )

        # Phase 3: Synthesize and respond
        return AgentAction(
            action_type="respond",
            response=self._synthesize_findings(task),
            thought="Research complete, synthesizing findings.",
        )

    def act(self, action: AgentAction) -> AgentObservation:
        """Execute research action."""
        if action.action_type == "tool_call" and action.tool_name == "search":
            result = self.executor.execute("search", **(action.tool_args or {}))
            obs = AgentObservation(
                action=action,
                result=result,
                observation_text=result.output if result.success else f"Search error: {result.error}",
                success=result.success,
            )

            # Extract and store sources from search results
            if result.success and result.data and "results" in result.data:
                for i, search_result in enumerate(result.data["results"]):
                    source_id = f"src_{len(self._sources) + 1}"
                    source = Source(
                        source_id=source_id,
                        title=search_result.get("title", "Untitled"),
                        content=search_result.get("snippet", ""),
                        relevance=search_result.get("relevance", 0.5),
                    )
                    self._sources.append(source)
                    self._findings.append(search_result.get("snippet", ""))

            return obs

        return super().act(action)

    def _generate_follow_up_queries(self, original_query: str) -> List[str]:
        """Generate follow-up research queries."""
        if self.llm_fn:
            prompt = (
                f"Original research question: {original_query}\n\n"
                f"Findings so far: {'; '.join(self._findings[:3])}\n\n"
                f"Generate 2-3 specific follow-up search queries to deepen the research. "
                f"One query per line."
            )
            try:
                response = self.llm_fn(prompt)
                queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
                return queries[:3]
            except Exception as e:
                logger.warning("LLM follow-up generation failed: %s", e)

        # Rule-based follow-up queries
        follow_ups = [
            f"{original_query} detailed explanation",
            f"{original_query} recent developments",
            f"{original_query} examples and applications",
        ]
        return follow_ups[:3]

    def _synthesize_findings(self, topic: str) -> str:
        """Synthesize research findings into a report with citations."""
        if not self._findings:
            return f"No research findings available for: {topic}"

        # Deduplicate sources
        unique_sources: List[Source] = []
        seen_content: set = set()
        for source in self._sources:
            content_key = source.content[:50]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_sources.append(source)

        if self.llm_fn:
            sources_text = "\n".join(
                f"[{s.source_id}] {s.content}" for s in unique_sources
            )
            prompt = (
                f"Research topic: {topic}\n\n"
                f"Sources:\n{sources_text}\n\n"
                f"Write a comprehensive research summary that:\n"
                f"1. Answers the research question\n"
                f"2. Synthesizes information from multiple sources\n"
                f"3. Uses citations like [src_1] to reference sources\n"
                f"4. Identifies any gaps or contradictions\n"
                f"5. Provides a conclusion"
            )
            try:
                return self.llm_fn(prompt)
            except Exception as e:
                logger.warning("LLM synthesis failed: %s", e)

        # Rule-based synthesis
        report_parts = [
            f"# Research Report: {topic}\n",
            "## Summary",
        ]

        # Combine findings
        for i, finding in enumerate(self._findings[:5]):
            source_ref = f"[src_{i + 1}]" if i < len(unique_sources) else ""
            report_parts.append(f"- {finding} {source_ref}")

        # Add citations
        if unique_sources:
            report_parts.append("\n## Sources")
            for source in unique_sources:
                report_parts.append(f"- {source.cite()}")

        report_parts.append("\n## Conclusion")
        report_parts.append(
            f"Based on {len(unique_sources)} source(s), "
            f"the research on '{topic}' yielded {len(self._findings)} finding(s). "
            f"Further research may be needed for a more comprehensive understanding."
        )

        return "\n".join(report_parts)

    def get_sources(self) -> List[Source]:
        """Get all gathered sources."""
        return list(self._sources)

    def reset(self) -> None:
        """Reset the research agent."""
        super().reset()
        self._sources.clear()
        self._findings.clear()
        self._research_plan = None
        self._current_step_idx = 0
