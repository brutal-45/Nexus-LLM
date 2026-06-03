"""Prompt template for Nexus-LLM.

Supports ``{variable}`` placeholder substitution, variable
extraction, validation, and multiple template types (simple, chat,
instruction, few-shot).
"""

import re
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Regex to find {variable} placeholders
_VARIABLE_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class TemplateType(str, Enum):
    """Supported prompt template types."""

    SIMPLE = "simple"
    CHAT = "chat"
    INSTRUCTION = "instruction"
    FEW_SHOT = "few_shot"


class PromptTemplate:
    """A reusable prompt template with variable placeholders.

    Templates use ``{variable}`` syntax for substitution.

    Example::

        tpl = PromptTemplate(
            name="greeting",
            template="Hello, {name}! Welcome to {place}.",
        )
        rendered = tpl.render(name="Alice", place="Wonderland")
        # "Hello, Alice! Welcome to Wonderland."

        valid, missing = tpl.validate(name="Alice")
        # (False, {"place"})
    """

    def __init__(
        self,
        name: str,
        template: str,
        template_type: TemplateType = TemplateType.SIMPLE,
        system_prompt: Optional[str] = None,
        expected_format: Optional[str] = None,
        description: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Initialise a PromptTemplate.

        Args:
            name: Unique template name.
            template: The template string with ``{variable}`` placeholders.
            template_type: Type of template (simple, chat, instruction, few_shot).
            system_prompt: Optional system prompt for chat-type templates.
            expected_format: Description of the expected output format.
            description: Human-readable description.
            examples: Optional list of example dicts for few-shot templates.
        """
        self.name = name
        self.template = template
        self.template_type = template_type
        self.system_prompt = system_prompt
        self.expected_format = expected_format
        self.description = description or ""
        self.examples = examples or []

    # ------------------------------------------------------------------
    # Variable handling
    # ------------------------------------------------------------------

    def extract_variables(self) -> List[str]:
        """Return a sorted list of variable names found in the template.

        Returns:
            List of variable names (without braces).
        """
        return sorted(set(_VARIABLE_RE.findall(self.template)))

    def validate(self, **kwargs: Any) -> Tuple[bool, Set[str]]:
        """Check whether the provided kwargs satisfy all template variables.

        Args:
            **kwargs: Variable values to validate.

        Returns:
            A tuple of ``(is_valid, missing_variables)``.
        """
        required = set(self.extract_variables())
        provided = set(kwargs.keys())
        missing = required - provided
        return (len(missing) == 0, missing)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, **kwargs: Any) -> str:
        """Render the template by substituting variables.

        Missing variables are replaced with their name wrapped in
        angle brackets (e.g. ``<name>``) rather than raising, so
        that partial renders are always possible.

        Args:
            **kwargs: Variable values.

        Returns:
            The rendered string.
        """
        valid, missing = self.validate(**kwargs)
        if not valid:
            logger.warning(
                "Template %r is missing variables: %s",
                self.name,
                missing,
            )

        # Build a safe format dict: replace missing with <varname>
        all_vars = self.extract_variables()
        format_dict: Dict[str, str] = {}
        for var in all_vars:
            format_dict[var] = kwargs.get(var, f"<{var}>")

        # Use str.format_map for safe substitution (ignores extra keys)
        try:
            result = self.template.format_map(format_dict)
        except (KeyError, IndexError, ValueError) as exc:
            logger.error("Template rendering failed for %r: %s", self.name, exc)
            result = self.template

        return result

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PromptTemplate(name={self.name!r}, "
            f"type={self.template_type.value!r}, "
            f"variables={self.extract_variables()})"
        )

    def __str__(self) -> str:
        return self.template

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the template to a dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "template_type": self.template_type.value,
            "system_prompt": self.system_prompt,
            "expected_format": self.expected_format,
            "description": self.description,
            "examples": self.examples,
            "variables": self.extract_variables(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Deserialise a template from a dictionary."""
        return cls(
            name=data["name"],
            template=data["template"],
            template_type=TemplateType(data.get("template_type", "simple")),
            system_prompt=data.get("system_prompt"),
            expected_format=data.get("expected_format"),
            description=data.get("description"),
            examples=data.get("examples"),
        )
