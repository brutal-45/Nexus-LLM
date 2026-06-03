"""Preset data class for Nexus-LLM.

Encapsulates a named, categorised configuration of model settings,
generation parameters, and a system prompt.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class Preset:
    """A pre-configured set of model parameters and system prompt.

    Attributes:
        name: Unique preset name (e.g. ``"chat_assistant"``).
        category: Category key (see :mod:`nexus_llm.presets.categories`).
        description: Human-readable description of the preset.
        model_config: Model-level configuration (model name, provider,
                      context length, etc.).
        generation_params: Generation-time parameters (temperature,
                           top_p, max_tokens, etc.).
        system_prompt: Default system prompt template.

    Example::

        p = Preset(
            name="code_helper",
            category="code",
            description="Optimised for code generation",
            model_config={"model": "claude-3-sonnet", "context_length": 8192},
            generation_params={"temperature": 0.2, "max_tokens": 2048},
            system_prompt="You are an expert software engineer.",
        )
        d = p.to_dict()
        p2 = Preset.from_dict(d)
    """

    name: str = ""
    category: str = "custom"
    description: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the preset to a plain dict.

        Returns:
            A JSON-friendly dict representation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Preset":
        """Deserialise a preset from a plain dict.

        Missing keys fall back to their dataclass defaults.

        Args:
            data: Dict with optional keys matching the dataclass fields.

        Returns:
            A new :class:`Preset` instance.
        """
        return cls(
            name=data.get("name", ""),
            category=data.get("category", "custom"),
            description=data.get("description", ""),
            model_config=data.get("model_config", {}),
            generation_params=data.get("generation_params", {}),
            system_prompt=data.get("system_prompt", ""),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate the preset fields and raise on problems.

        Raises:
            ValueError: If the preset name is empty or config dicts
                        contain non-string keys.
        """
        if not self.name:
            raise ValueError("Preset name must not be empty")
        for key in self.model_config:
            if not isinstance(key, str):
                raise ValueError(
                    f"model_config keys must be strings, got {type(key).__name__}"
                )
        for key in self.generation_params:
            if not isinstance(key, str):
                raise ValueError(
                    f"generation_params keys must be strings, got {type(key).__name__}"
                )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Preset name={self.name!r} category={self.category!r}>"
        )
