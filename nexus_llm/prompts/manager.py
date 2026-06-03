"""Prompt manager for Nexus-LLM.

Manages a registry of prompt templates, supports rendering by
name, and listing templates by category.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.prompts.template import PromptTemplate, TemplateType

logger = logging.getLogger(__name__)


class PromptManager:
    """Central registry for prompt templates.

    Example::

        pm = PromptManager()
        pm.add_template("greet", PromptTemplate(
            name="greet",
            template="Hello, {name}!",
        ))
        rendered = pm.render("greet", name="Alice")
    """

    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplate] = {}

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def add_template(self, name: str, template: PromptTemplate) -> None:
        """Register a prompt template.

        Args:
            name: Unique identifier for the template.
            template: A PromptTemplate instance.

        Raises:
            ValueError: If *name* is already registered.
        """
        if name in self._templates:
            raise ValueError(
                f"Template {name!r} already exists. Remove it first or use a different name."
            )
        self._templates[name] = template
        logger.info("Registered prompt template %r", name)

    def get_template(self, name: str) -> PromptTemplate:
        """Retrieve a template by name.

        Args:
            name: Template identifier.

        Returns:
            The PromptTemplate instance.

        Raises:
            KeyError: If the template does not exist.
        """
        if name not in self._templates:
            raise KeyError(
                f"Template {name!r} not found. "
                f"Available: {sorted(self._templates.keys())}"
            )
        return self._templates[name]

    def remove_template(self, name: str) -> None:
        """Remove a registered template.

        Args:
            name: Template identifier.

        Raises:
            KeyError: If the template does not exist.
        """
        if name not in self._templates:
            raise KeyError(f"Template {name!r} not found.")
        del self._templates[name]
        logger.info("Removed prompt template %r", name)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Render a named template with the given variables.

        Args:
            template_name: Template identifier.
            **kwargs: Variable values for substitution.

        Returns:
            The rendered prompt string.
        """
        template = self.get_template(template_name)
        return template.render(**kwargs)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_templates(
        self,
        category: Optional[str] = None,
    ) -> List[str]:
        """Return the names of registered templates.

        Args:
            category: Optional filter by template type
                      (``"simple"``, ``"chat"``, ``"instruction"``,
                       ``"few_shot"``).

        Returns:
            Sorted list of template names.
        """
        if category is None:
            return sorted(self._templates.keys())

        try:
            target_type = TemplateType(category)
        except ValueError:
            raise ValueError(
                f"Invalid category {category!r}. "
                f"Valid: {[t.value for t in TemplateType]}"
            )

        return sorted(
            name
            for name, tpl in self._templates.items()
            if tpl.template_type == target_type
        )

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def load_library(self, library: "PromptLibrary") -> None:
        """Load all templates from a PromptLibrary into this manager.

        Args:
            library: A PromptLibrary instance.
        """
        for name, template in library.get_all().items():
            try:
                self.add_template(name, template)
            except ValueError:
                logger.warning("Template %r already exists; skipping", name)
