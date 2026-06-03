"""Configuration for knowledge distillation.

Provides the DistillationConfig dataclass that governs distillation
hyper-parameters including temperature, loss balancing, and training
schedule.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation.

    Attributes:
        temperature: Softmax temperature applied to both teacher and
            student logits.  Higher values produce softer probability
            distributions, transferring more "dark knowledge" from the
            teacher.  Typical range: 1.0 – 20.0.
        alpha: Interpolation weight between the distillation (soft) loss
            and the hard-label cross-entropy loss.

            * ``alpha * soft_loss + (1 - alpha) * hard_loss``
            * ``alpha = 1.0`` → pure distillation loss.
            * ``alpha = 0.0`` → pure hard-label loss.
        learning_rate: Peak learning rate for the student optimiser.
        batch_size: Mini-batch size for training.
        epochs: Total number of training epochs.

    Example::

        config = DistillationConfig(
            temperature=4.0,
            alpha=0.7,
            learning_rate=5e-5,
            batch_size=8,
            epochs=3,
        )
    """

    temperature: float = 2.0
    alpha: float = 0.5
    learning_rate: float = 5e-5
    batch_size: int = 8
    epochs: int = 3

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {self.temperature}"
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"alpha must be in [0, 1], got {self.alpha}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {self.batch_size}"
            )
        if self.epochs <= 0:
            raise ValueError(
                f"epochs must be a positive integer, got {self.epochs}"
            )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain dictionary."""
        return {
            "temperature": self.temperature,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistillationConfig":
        """Deserialise a configuration from a plain dictionary.

        Args:
            data: Dictionary with keys matching DistillationConfig fields.
                Unknown keys are silently ignored.

        Returns:
            A new DistillationConfig instance.

        Raises:
            ValueError: If the resulting configuration is invalid.
        """
        return cls(
            temperature=data.get("temperature", 2.0),
            alpha=data.get("alpha", 0.5),
            learning_rate=data.get("learning_rate", 5e-5),
            batch_size=data.get("batch_size", 8),
            epochs=data.get("epochs", 3),
        )
