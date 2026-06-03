"""Configuration for RLHF / DPO alignment training.

Provides the RLHFConfig dataclass that controls alignment method
selection, loss hyper-parameters, and the training schedule.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal

# Valid alignment methods
AlignMethod = Literal["dpo", "rlhf"]
VALID_METHODS: tuple = ("dpo", "rlhf")


@dataclass
class RLHFConfig:
    """Configuration for preference alignment training.

    Supports two methods:

    * **dpo** – Direct Preference Optimization.  Directly optimises the
      policy using preference pairs without a separate reward model.
      The ``beta`` parameter controls the KL-penalty strength.
    * **rlhf** – Reinforcement Learning from Human Feedback.  Uses a
      trained reward model to provide scalar signals for PPO-style
      policy optimisation.

    Attributes:
        method: Alignment method (``"dpo"`` or ``"rlhf"``).
        beta: Inverse temperature / KL penalty for DPO.  Higher values
            penalise deviation from the reference policy more strongly.
            Typical range: 0.05 – 0.5.
        learning_rate: Peak learning rate for the policy optimiser.
        batch_size: Mini-batch size for training.
        epochs: Total number of training epochs.

    Example::

        config = RLHFConfig(method="dpo", beta=0.1, learning_rate=1e-5)
        aligned = trainer.train(model, preference_dataset, config)
    """

    method: AlignMethod = "dpo"
    beta: float = 0.1
    learning_rate: float = 1e-5
    batch_size: int = 4
    epochs: int = 1

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"Invalid alignment method '{self.method}'. "
                f"Choose from: {', '.join(VALID_METHODS)}"
            )
        if self.beta <= 0:
            raise ValueError(
                f"beta must be positive, got {self.beta}"
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
            "method": self.method,
            "beta": self.beta,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLHFConfig":
        """Deserialise a configuration from a plain dictionary.

        Args:
            data: Dictionary with keys matching RLHFConfig fields.
                Unknown keys are silently ignored.

        Returns:
            A new RLHFConfig instance.

        Raises:
            ValueError: If the resulting configuration is invalid.
        """
        return cls(
            method=data.get("method", "dpo"),
            beta=data.get("beta", 0.1),
            learning_rate=data.get("learning_rate", 1e-5),
            batch_size=data.get("batch_size", 4),
            epochs=data.get("epochs", 1),
        )
