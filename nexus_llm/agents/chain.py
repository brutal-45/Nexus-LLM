"""Agent chain for Nexus-LLM.

Chains multiple agents sequentially, passing outputs between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus_llm.agents.agent import Agent, AgentResult
from nexus_llm.agents.config import AgentConfig
from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Chain result
# ---------------------------------------------------------------------------

@dataclass
class ChainResult:
    """Result of running an agent chain.

    Attributes:
        task: The original task.
        final_answer: The answer from the last agent in the chain.
        agent_results: Per-agent results in execution order.
        success: Whether all agents succeeded.
    """

    task: str
    final_answer: str
    agent_results: List[AgentResult] = field(default_factory=list)
    success: bool = True

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "PARTIAL/FAILED"
        return (
            f"[{status}] agents={len(self.agent_results)} "
            f"answer={self.final_answer[:100]}"
        )


# ---------------------------------------------------------------------------
# Agent chain
# ---------------------------------------------------------------------------

class AgentChain:
    """Sequential chain of agents that pass outputs between them.

    Each agent's output is fed as input context to the next agent in
    the chain, enabling multi-stage processing pipelines.

    Args:
        agents: Ordered list of agents to execute.
        name: Optional name for the chain.
    """

    def __init__(
        self,
        agents: Optional[List[Agent]] = None,
        name: str = "agent-chain",
    ) -> None:
        self.agents: List[Agent] = agents or []
        self.name = name
        logger.info(
            "AgentChain '%s' initialised with %d agent(s)",
            name, len(self.agents),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> ChainResult:
        """Run the chain: execute agents sequentially.

        The first agent receives *task*.  Each subsequent agent
        receives the previous agent's answer as additional context.

        Args:
            task: The initial task description.

        Returns:
            A :class:`ChainResult` with the final answer and per-agent
            results.
        """
        if not self.agents:
            logger.warning("AgentChain '%s' has no agents", self.name)
            return ChainResult(
                task=task,
                final_answer="No agents in chain",
                success=False,
            )

        agent_results: List[AgentResult] = []
        current_input = task
        all_success = True

        for i, agent in enumerate(self.agents):
            agent_name = agent.config.name
            # Augment the input with prior context
            if i > 0:
                prev_answer = agent_results[-1].answer
                augmented_input = (
                    f"Previous agent ({agent_results[-1].steps_taken} steps) "
                    f"produced:\n{prev_answer}\n\n"
                    f"Based on the above, please address:\n{task}"
                )
            else:
                augmented_input = current_input

            logger.info(
                "Chain '%s' – running agent %d/%d: %s",
                self.name, i + 1, len(self.agents), agent_name,
            )
            result = agent.run(augmented_input)
            agent_results.append(result)

            if not result.success:
                all_success = False
                logger.warning(
                    "Agent '%s' failed at chain position %d", agent_name, i + 1,
                )

        final_answer = agent_results[-1].answer if agent_results else ""

        chain_result = ChainResult(
            task=task,
            final_answer=final_answer,
            agent_results=agent_results,
            success=all_success,
        )
        logger.info("Chain '%s' completed: %s", self.name, chain_result.summary())
        return chain_result

    # ------------------------------------------------------------------
    # Chain management
    # ------------------------------------------------------------------

    def add_agent(self, agent: Agent) -> None:
        """Append an agent to the chain."""
        self.agents.append(agent)
        logger.debug("Added agent '%s' to chain '%s'", agent.config.name, self.name)

    def remove_agent(self, index: int) -> bool:
        """Remove an agent by index.  Returns ``True`` if removed."""
        if 0 <= index < len(self.agents):
            removed = self.agents.pop(index)
            logger.debug(
                "Removed agent '%s' from chain '%s'", removed.config.name, self.name,
            )
            return True
        return False

    def insert_agent(self, index: int, agent: Agent) -> None:
        """Insert an agent at the given position."""
        self.agents.insert(index, agent)
        logger.debug(
            "Inserted agent '%s' at position %d in chain '%s'",
            agent.config.name, index, self.name,
        )

    @property
    def length(self) -> int:
        """Number of agents in the chain."""
        return len(self.agents)
