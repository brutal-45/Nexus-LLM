"""Worker pool configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RetryPolicy:
    """Configuration for task retry behaviour.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries).
        backoff: Base backoff interval in seconds.  Actual delay is
            ``backoff * 2 ** attempt``.
    """

    max_retries: int = 3
    backoff: float = 1.0

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the delay in seconds before the *attempt*-th retry.

        Uses exponential backoff: ``backoff * 2 ** attempt``.

        Args:
            attempt: Zero-based retry attempt number.

        Returns:
            Delay in seconds.
        """
        return self.backoff * (2 ** attempt)


@dataclass
class WorkerConfig:
    """Configuration for :class:`~nexus_llm.workers.pool.WorkerPool`.

    Attributes:
        num_workers: Number of worker threads.
        max_queue_size: Maximum pending tasks (0 = unbounded).
        timeout: Default task timeout in seconds (0 = no timeout).
        heartbeat_interval: Seconds between heartbeat pulses.
        retry_policy: Retry configuration for failed tasks.
    """

    num_workers: int = 4
    max_queue_size: int = 0
    timeout: float = 0.0
    heartbeat_interval: float = 5.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain dict."""
        return {
            "num_workers": self.num_workers,
            "max_queue_size": self.max_queue_size,
            "timeout": self.timeout,
            "heartbeat_interval": self.heartbeat_interval,
            "retry_policy": {
                "max_retries": self.retry_policy.max_retries,
                "backoff": self.retry_policy.backoff,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerConfig:
        """Create a :class:`WorkerConfig` from a dict.

        Unknown keys are silently ignored so that configs loaded from
        external files (e.g. YAML) are forward-compatible.

        Args:
            data: Configuration dictionary.

        Returns:
            A new :class:`WorkerConfig` instance.
        """
        retry_data = data.get("retry_policy", {})
        retry_policy = RetryPolicy(
            max_retries=retry_data.get("max_retries", 3),
            backoff=retry_data.get("backoff", 1.0),
        )

        return cls(
            num_workers=data.get("num_workers", 4),
            max_queue_size=data.get("max_queue_size", 0),
            timeout=data.get("timeout", 0.0),
            heartbeat_interval=data.get("heartbeat_interval", 5.0),
            retry_policy=retry_policy,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.num_workers < 1:
            raise ValueError(
                f"num_workers must be >= 1, got {self.num_workers}"
            )
        if self.max_queue_size < 0:
            raise ValueError(
                f"max_queue_size must be >= 0, got {self.max_queue_size}"
            )
        if self.timeout < 0:
            raise ValueError(
                f"timeout must be >= 0, got {self.timeout}"
            )
        if self.heartbeat_interval <= 0:
            raise ValueError(
                f"heartbeat_interval must be > 0, got {self.heartbeat_interval}"
            )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<WorkerConfig workers={self.num_workers} "
            f"queue={self.max_queue_size} "
            f"timeout={self.timeout}s "
            f"retries={self.retry_policy.max_retries}>"
        )
