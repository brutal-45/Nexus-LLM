"""Configuration for model quantization.

Provides the QuantConfig dataclass that controls quantization behaviour
including method selection, group sizing, and symmetry options.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

# Valid quantization methods
QuantMethod = Literal["int8", "int4", "fp16", "bf16", "gguf"]
VALID_METHODS: tuple = ("int8", "int4", "fp16", "bf16", "gguf")


@dataclass
class QuantConfig:
    """Configuration for model quantization.

    Attributes:
        method: Quantization method to apply.
        group_size: Group size for group-wise quantization (primarily int4).
            A group size of 128 means quantization parameters are shared
            across every 128 consecutive weights.
        sym: Whether to use symmetric quantization. When True, the
            zero-point is fixed at 0 and only a single scale factor
            is stored per group. When False, an additional zero-point
            offset is stored.

    Example::

        config = QuantConfig(method="int4", group_size=128, sym=True)
        quantized = quantizer.quantize(model, config)
    """

    method: QuantMethod = "int8"
    group_size: int = 128
    sym: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"Invalid quantization method '{self.method}'. "
                f"Choose from: {', '.join(VALID_METHODS)}"
            )
        if self.group_size <= 0:
            raise ValueError(
                f"group_size must be a positive integer, got {self.group_size}"
            )
        if self.method == "int4" and self.group_size not in (32, 64, 128, 256):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Non-standard group_size=%d for int4 quantization. "
                "Common values are 32, 64, 128, or 256.",
                self.group_size,
            )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain dictionary."""
        return {
            "method": self.method,
            "group_size": self.group_size,
            "sym": self.sym,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantConfig":
        """Deserialise a configuration from a plain dictionary.

        Args:
            data: Dictionary with keys matching QuantConfig fields.
                Unknown keys are silently ignored.

        Returns:
            A new QuantConfig instance.

        Raises:
            ValueError: If the resulting configuration is invalid.
        """
        return cls(
            method=data.get("method", "int8"),
            group_size=data.get("group_size", 128),
            sym=data.get("sym", True),
        )
