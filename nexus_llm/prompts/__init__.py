"""Prompts module for Nexus-LLM.

Provides prompt template management, a built-in prompt library,
prompt rendering with variable substitution, and prompt optimisation
utilities.
"""

from nexus_llm.prompts.manager import PromptManager
from nexus_llm.prompts.template import PromptTemplate
from nexus_llm.prompts.library import PromptLibrary
from nexus_llm.prompts.optimizer import PromptOptimizer

__all__ = [
    "PromptManager",
    "PromptTemplate",
    "PromptLibrary",
    "PromptOptimizer",
]
