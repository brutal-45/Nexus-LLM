"""Prompt library for Nexus-LLM.

Provides a curated collection of built-in prompt templates
organised by category, covering common LLM use-cases.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.prompts.template import PromptTemplate, TemplateType

logger = logging.getLogger(__name__)


# ======================================================================
# Built-in template definitions
# ======================================================================

_BUILTIN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "chat": {
        "name": "chat",
        "template": "{user_message}",
        "template_type": TemplateType.CHAT,
        "system_prompt": (
            "You are a helpful, harmless, and honest AI assistant. "
            "Engage in natural conversation while being clear and concise."
        ),
        "expected_format": "Natural language response",
        "description": "General-purpose chat template",
    },
    "code": {
        "name": "code",
        "template": (
            "Write {language} code to: {task}\n\n"
            "Requirements:\n"
            "- Follow best practices and conventions\n"
            "- Include appropriate error handling\n"
            "- Add clear comments where necessary"
        ),
        "template_type": TemplateType.INSTRUCTION,
        "system_prompt": (
            "You are an expert software engineer. Write clean, efficient, "
            "and well-documented code."
        ),
        "expected_format": "Code block with explanation",
        "description": "Code generation template",
    },
    "creative": {
        "name": "creative",
        "template": (
            "Write a {content_type} about {topic}.\n\n"
            "Style: {style}\n"
            "Tone: {tone}\n"
            "Length: {length}"
        ),
        "template_type": TemplateType.INSTRUCTION,
        "system_prompt": (
            "You are a creative writer with a vivid imagination. "
            "Produce engaging, original content."
        ),
        "expected_format": "Creative text",
        "description": "Creative writing template",
    },
    "analysis": {
        "name": "analysis",
        "template": (
            "Analyse the following {subject_type}:\n\n"
            "{content}\n\n"
            "Provide:\n"
            "1. Key findings\n"
            "2. Strengths and weaknesses\n"
            "3. Recommendations"
        ),
        "template_type": TemplateType.INSTRUCTION,
        "system_prompt": (
            "You are an analytical expert. Provide thorough, objective, "
            "and well-structured analysis."
        ),
        "expected_format": "Structured analysis with numbered sections",
        "description": "Analysis template",
    },
    "summarize": {
        "name": "summarize",
        "template": (
            "Summarise the following text in {length}:\n\n"
            "{text}\n\n"
            "Focus on the main points and key takeaways."
        ),
        "template_type": TemplateType.INSTRUCTION,
        "system_prompt": (
            "You are a summarisation expert. Produce concise, accurate "
            "summaries that capture the essential information."
        ),
        "expected_format": "Concise summary",
        "description": "Summarisation template",
    },
    "translate": {
        "name": "translate",
        "template": (
            "Translate the following text from {source_lang} to {target_lang}:\n\n"
            "{text}\n\n"
            "Preserve the original meaning, tone, and style."
        ),
        "template_type": TemplateType.INSTRUCTION,
        "system_prompt": (
            "You are a professional translator. Provide accurate, "
            "natural-sounding translations."
        ),
        "expected_format": "Translated text",
        "description": "Translation template",
    },
    "explain": {
        "name": "explain",
        "template": (
            "Explain {concept} in simple terms.\n\n"
            "Target audience: {audience}\n"
            "Use analogies and examples where helpful."
        ),
        "template_type": TemplateType.INSTRUCTION,
        "system_prompt": (
            "You are a skilled educator. Explain complex concepts "
            "in simple, accessible language."
        ),
        "expected_format": "Clear explanation with examples",
        "description": "Explanation template",
    },
    "debug": {
        "name": "debug",
        "template": (
            "Debug the following {language} code:\n\n"
            "```{language}\n{code}\n```\n\n"
            "Error message:\n{error}\n\n"
            "Identify the bug, explain why it occurs, and provide a fix."
        ),
        "template_type": TemplateType.INSTRUCTION,
        "system_prompt": (
            "You are an expert debugger. Systematically identify and "
            "fix code issues with clear explanations."
        ),
        "expected_format": "Bug analysis and corrected code",
        "description": "Code debugging template",
    },
    "review": {
        "name": "review",
        "template": (
            "Review the following {content_type}:\n\n"
            "{content}\n\n"
            "Evaluate on: correctness, clarity, completeness, and style.\n"
            "Provide a rating (1-5) for each criterion and suggestions for improvement."
        ),
        "template_type": TemplateType.INSTRUCTION,
        "system_prompt": (
            "You are a thorough reviewer. Provide constructive, "
            "detailed feedback."
        ),
        "expected_format": "Structured review with ratings",
        "description": "Review and feedback template",
    },
    "teach": {
        "name": "teach",
        "template": (
            "Teach me about {topic}.\n\n"
            "Current knowledge level: {level}\n"
            "Learning goal: {goal}\n\n"
            "Structure the lesson with:\n"
            "1. Introduction\n"
            "2. Core concepts\n"
            "3. Practice exercises\n"
            "4. Further resources"
        ),
        "template_type": TemplateType.INSTRUCTION,
        "system_prompt": (
            "You are a patient, effective teacher. Adapt your "
            "explanations to the learner's level."
        ),
        "expected_format": "Structured lesson plan",
        "description": "Teaching / tutoring template",
    },
}


class PromptLibrary:
    """Curated collection of built-in prompt templates.

    Example::

        lib = PromptLibrary()
        chat_tpl = lib.get_by_name("chat")
        code_tpl = lib.get_by_name("code")
        creative = lib.get_by_category("instruction")
    """

    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_builtins()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_builtins(self) -> None:
        """Populate the library with built-in templates."""
        for name, data in _BUILTIN_TEMPLATES.items():
            self._templates[name] = PromptTemplate(**data)
        logger.info("Loaded %d built-in prompt templates", len(self._templates))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_by_name(self, name: str) -> PromptTemplate:
        """Retrieve a template by name.

        Args:
            name: Template identifier (e.g. ``"chat"``, ``"code"``).

        Returns:
            The PromptTemplate instance.

        Raises:
            KeyError: If the template does not exist.
        """
        if name not in self._templates:
            raise KeyError(
                f"Template {name!r} not found in library. "
                f"Available: {sorted(self._templates.keys())}"
            )
        return self._templates[name]

    def get_by_category(self, category: Optional[str] = None) -> Dict[str, PromptTemplate]:
        """Return templates filtered by type/category.

        Args:
            category: Optional filter — ``"simple"``, ``"chat"``,
                      ``"instruction"``, or ``"few_shot"``.
                      ``None`` returns all.

        Returns:
            Dict mapping template names to PromptTemplate instances.
        """
        if category is None:
            return dict(self._templates)

        try:
            target_type = TemplateType(category)
        except ValueError:
            raise ValueError(
                f"Invalid category {category!r}. "
                f"Valid: {[t.value for t in TemplateType]}"
            )

        return {
            name: tpl
            for name, tpl in self._templates.items()
            if tpl.template_type == target_type
        }

    def get_all(self) -> Dict[str, PromptTemplate]:
        """Return all templates in the library."""
        return dict(self._templates)

    def list_names(self) -> List[str]:
        """Return sorted list of all template names."""
        return sorted(self._templates.keys())

    def list_categories(self) -> List[str]:
        """Return sorted list of distinct template types."""
        return sorted({tpl.template_type.value for tpl in self._templates.values()})
