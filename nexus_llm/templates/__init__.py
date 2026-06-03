"""
Nexus-LLM Templates Module

Provides prompt templates and system prompts for various tasks:
- Chat, code, summary, translate, creative, and analysis prompt templates
- System prompts for different assistant personas
- Template loading from YAML and text files
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


_TEMPLATES_DIR = Path(__file__).parent


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file."""
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML templates. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_text(path: Union[str, Path]) -> str:
    """Load a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


class TemplateLoader:
    """
    Load and manage prompt templates and system prompts.

    Templates are organized as:
    - prompts/*.yaml  — task-specific prompt templates
    - system/*.txt    — system prompt files
    """

    def __init__(self, templates_dir: Optional[Union[str, Path]] = None):
        self.templates_dir = Path(templates_dir) if templates_dir else _TEMPLATES_DIR
        self._prompt_cache: Dict[str, Dict[str, Any]] = {}
        self._system_cache: Dict[str, str] = {}

    def load_prompt_template(self, name: str) -> Dict[str, Any]:
        """
        Load a prompt template by name.

        Args:
            name: Template name (without .yaml extension), e.g. "chat", "code".

        Returns:
            Template dict with keys like 'system', 'user', 'assistant', 'variants'.
        """
        if name in self._prompt_cache:
            return self._prompt_cache[name]

        path = self.templates_dir / "prompts" / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        template = _load_yaml(path)
        self._prompt_cache[name] = template
        return template

    def load_system_prompt(self, name: str) -> str:
        """
        Load a system prompt by name.

        Args:
            name: Prompt name (without .txt extension), e.g. "default", "coding".

        Returns:
            System prompt string.
        """
        if name in self._system_cache:
            return self._system_cache[name]

        path = self.templates_dir / "system" / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"System prompt not found: {path}")

        prompt = _load_text(path)
        self._system_cache[name] = prompt
        return prompt

    def list_prompt_templates(self) -> List[str]:
        """List available prompt template names."""
        prompts_dir = self.templates_dir / "prompts"
        if not prompts_dir.exists():
            return []
        return [p.stem for p in prompts_dir.glob("*.yaml")]

    def list_system_prompts(self) -> List[str]:
        """List available system prompt names."""
        system_dir = self.templates_dir / "system"
        if not system_dir.exists():
            return []
        return [p.stem for p in system_dir.glob("*.txt")]

    def format_prompt(
        self,
        template_name: str,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Format a prompt template with variables.

        Args:
            template_name: Name of the prompt template.
            **kwargs: Variables to substitute in the template.

        Returns:
            Dict with 'system', 'user', 'assistant' formatted strings.
        """
        template = self.load_prompt_template(template_name)
        result: Dict[str, str] = {}

        for key in ("system", "user", "assistant"):
            if key in template:
                text = template[key]
                if isinstance(text, str):
                    try:
                        result[key] = text.format(**kwargs)
                    except KeyError:
                        result[key] = text
                else:
                    result[key] = str(text)

        return result

    def build_messages(
        self,
        template_name: str,
        user_input: str,
        system_override: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """
        Build a ChatML-style message list from a template.

        Args:
            template_name: Name of the prompt template.
            user_input: The user's input text.
            system_override: Override system prompt text.
            **kwargs: Additional template variables.

        Returns:
            List of {"role": ..., "content": ...} messages.
        """
        template = self.load_prompt_template(template_name)
        messages: List[Dict[str, str]] = []

        # System message
        system_text = system_override
        if system_text is None and "system" in template:
            system_text = template["system"]
            if isinstance(system_text, str):
                try:
                    system_text = system_text.format(**kwargs)
                except KeyError:
                    pass
        if system_text:
            messages.append({"role": "system", "content": system_text})

        # User message
        user_template = template.get("user", "{input}")
        if isinstance(user_template, str):
            try:
                user_text = user_template.format(input=user_input, **kwargs)
            except KeyError:
                user_text = user_template
        else:
            user_text = user_input
        messages.append({"role": "user", "content": user_text})

        # Assistant prefix (optional)
        if "assistant" in template:
            asst_text = template["assistant"]
            if isinstance(asst_text, str):
                try:
                    asst_text = asst_text.format(**kwargs)
                except KeyError:
                    pass
            if asst_text:
                messages.append({"role": "assistant", "content": asst_text})

        return messages

    def clear_cache(self) -> None:
        """Clear template cache."""
        self._prompt_cache.clear()
        self._system_cache.clear()


# Module-level convenience
_loader = TemplateLoader()


def load_prompt_template(name: str) -> Dict[str, Any]:
    """Load a prompt template (module-level convenience)."""
    return _loader.load_prompt_template(name)


def load_system_prompt(name: str) -> str:
    """Load a system prompt (module-level convenience)."""
    return _loader.load_system_prompt(name)


def list_prompt_templates() -> List[str]:
    """List available prompt templates."""
    return _loader.list_prompt_templates()


def list_system_prompts() -> List[str]:
    """List available system prompts."""
    return _loader.list_system_prompts()


__all__ = [
    "TemplateLoader",
    "load_prompt_template",
    "load_system_prompt",
    "list_prompt_templates",
    "list_system_prompts",
]
