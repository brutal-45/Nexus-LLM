"""Model serving configuration for Nexus-LLM."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ServingConfig:
    """Configuration for the model serving stack.

    Controls host, port, workers, queue size, timeouts, and health
    check behaviour.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    max_queue_size: int = 1000
    timeout: float = 30.0
    max_concurrent_requests: int = 10
    health_check_interval: float = 30.0
    api_key: Optional[str] = None
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60
    log_requests: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServingConfig":
        """Create a ``ServingConfig`` from a dictionary.

        Unknown keys are silently ignored.

        Args:
            data: Dictionary with configuration values.

        Returns:
            ``ServingConfig`` instance.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the config to a dictionary.

        Returns:
            Dict representation of the configuration.
        """
        import dataclasses
        return dataclasses.asdict(self)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port} (must be 1-65535)")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if self.max_queue_size < 1:
            raise ValueError(f"max_queue_size must be >= 1, got {self.max_queue_size}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if self.max_concurrent_requests < 1:
            raise ValueError(
                f"max_concurrent_requests must be >= 1, got {self.max_concurrent_requests}"
            )
        if self.health_check_interval <= 0:
            raise ValueError(
                f"health_check_interval must be > 0, got {self.health_check_interval}"
            )

    def __post_init__(self) -> None:
        self.validate()
