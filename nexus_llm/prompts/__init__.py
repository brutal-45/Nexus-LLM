"""Prompts module for Nexus-LLM.

Provides prompt template management, a built-in prompt library,
prompt rendering with variable substitution, and prompt optimisation
utilities.
"""

from nexus_llm.prompts.library import PromptLibrary
from nexus_llm.prompts.manager import PromptManager
from nexus_llm.prompts.optimizer import PromptOptimizer
from nexus_llm.prompts.template import PromptTemplate

__all__ = [
    "PromptLibrary",
    "PromptManager",
    "PromptOptimizer",
    "PromptTemplate",
]
