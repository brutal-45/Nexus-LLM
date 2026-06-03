"""Built-in preset library for Nexus-LLM.

Contains pre-configured presets for common use cases: chat assistant,
code helper, creative writer, data analyst, and tutor.
"""

from typing import Any, Dict, List, Optional

from nexus_llm.presets.preset import Preset
from nexus_llm.presets.categories import PresetCategory


class PresetLibrary:
    """Collection of built-in presets organised by category.

    Example::

        lib = PresetLibrary()
        all_presets = lib.get_presets_by_category()
        code_presets = lib.list_presets(category="code")
    """

    def __init__(self) -> None:
        self._presets: Dict[str, Preset] = {}
        self._load_builtins()

    # ------------------------------------------------------------------
    # Built-in presets
    # ------------------------------------------------------------------

    def _load_builtins(self) -> None:
        """Populate the library with built-in presets."""
        builtins: List[Preset] = [
            Preset(
                name="chat_assistant",
                category=PresetCategory.CHAT,
                description=(
                    "General-purpose conversational assistant suitable "
                    "for everyday Q&A and dialogue."
                ),
                model_config={
                    "model": "claude-3-sonnet",
                    "context_length": 8192,
                    "provider": "anthropic",
                },
                generation_params={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1024,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                },
                system_prompt=(
                    "You are a helpful, harmless, and honest AI assistant. "
                    "Provide clear, accurate, and concise responses. "
                    "If you are unsure about something, say so."
                ),
            ),
            Preset(
                name="code_helper",
                category=PresetCategory.CODE,
                description=(
                    "Optimised for code generation, debugging, and "
                    "software engineering tasks."
                ),
                model_config={
                    "model": "claude-3-sonnet",
                    "context_length": 16384,
                    "provider": "anthropic",
                },
                generation_params={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_tokens": 2048,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                },
                system_prompt=(
                    "You are an expert software engineer. Write clean, "
                    "well-documented code. Follow best practices and "
                    "design patterns. Explain your reasoning when "
                    "suggesting changes or fixes."
                ),
            ),
            Preset(
                name="creative_writer",
                category=PresetCategory.CREATIVE,
                description=(
                    "Optimised for creative writing, storytelling, "
                    "and brainstorming sessions."
                ),
                model_config={
                    "model": "claude-3-opus",
                    "context_length": 8192,
                    "provider": "anthropic",
                },
                generation_params={
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_tokens": 2048,
                    "frequency_penalty": 0.3,
                    "presence_penalty": 0.3,
                },
                system_prompt=(
                    "You are a creative writer with a vivid imagination. "
                    "Craft engaging narratives, explore unusual perspectives, "
                    "and use rich, evocative language. Be original and "
                    "expressive while remaining coherent."
                ),
            ),
            Preset(
                name="data_analyst",
                category=PresetCategory.ANALYSIS,
                description=(
                    "Optimised for data analysis, summarisation, "
                    "and structured reasoning."
                ),
                model_config={
                    "model": "claude-3-sonnet",
                    "context_length": 16384,
                    "provider": "anthropic",
                },
                generation_params={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 2048,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                },
                system_prompt=(
                    "You are a data analyst expert. Provide structured, "
                    "evidence-based analysis. Present findings clearly "
                    "with appropriate formatting. Highlight key insights "
                    "and caveats. When working with data, be precise "
                    "and methodical."
                ),
            ),
            Preset(
                name="tutor",
                category=PresetCategory.EDUCATION,
                description=(
                    "Optimised for teaching, tutoring, and explaining "
                    "concepts at varying levels of complexity."
                ),
                model_config={
                    "model": "claude-3-sonnet",
                    "context_length": 8192,
                    "provider": "anthropic",
                },
                generation_params={
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "max_tokens": 1024,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.1,
                },
                system_prompt=(
                    "You are a patient and knowledgeable tutor. Explain "
                    "concepts step by step, using simple language before "
                    "introducing technical terms. Ask clarifying questions "
                    "to gauge understanding. Provide examples and "
                    "analogies to make complex topics accessible."
                ),
            ),
        ]

        for preset in builtins:
            self._presets[preset.name] = preset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> Preset:
        """Retrieve a built-in preset by name.

        Args:
            name: Preset name (e.g. ``"chat_assistant"``).

        Returns:
            The matching :class:`Preset`.

        Raises:
            KeyError: If no preset with the given name exists.
        """
        if name not in self._presets:
            raise KeyError(f"Built-in preset {name!r} not found")
        return self._presets[name]

    def list_presets(
        self, category: Optional[str] = None
    ) -> List[Preset]:
        """List built-in presets, optionally filtered by category.

        Args:
            category: If provided, only return presets in this category.

        Returns:
            List of :class:`Preset` objects.
        """
        presets = list(self._presets.values())
        if category is not None:
            presets = [p for p in presets if p.category == category]
        return presets

    def get_presets_by_category(self) -> Dict[str, List[Preset]]:
        """Group all built-in presets by category.

        Returns:
            Dict mapping category name to a list of presets.
        """
        result: Dict[str, List[Preset]] = {}
        for preset in self._presets.values():
            result.setdefault(preset.category, []).append(preset)
        return result

    @property
    def names(self) -> List[str]:
        """Return the names of all built-in presets."""
        return list(self._presets.keys())

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<PresetLibrary presets={len(self._presets)}>"


