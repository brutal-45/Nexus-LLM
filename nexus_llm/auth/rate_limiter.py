"""Token-bucket rate limiter for API key and IP-based throttling."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_minute: Max requests allowed per minute.
        requests_per_hour: Max requests allowed per hour.
        requests_per_day: Max requests allowed per day.
    """

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000


@dataclass
class _TokenBucket:
    """Internal token bucket state.

    Attributes:
        tokens: Current number of available tokens.
        max_tokens: Bucket capacity.
        refill_rate: Tokens added per second.
        last_refill: Timestamp of last refill.
    """

    tokens: float
    max_tokens: float
    refill_rate: float
    last_refill: float = field(default_factory=time.monotonic)

    def refill(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        added = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + added)
        self.last_refill = now

    def consume(self, count: float = 1.0) -> bool:
        """Try to consume *count* tokens.

        Returns:
            ``True`` if tokens were consumed, ``False`` if insufficient.
        """
        self.refill()
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False

    def time_until_available(self, count: float = 1.0) -> float:
        """Return seconds until *count* tokens are available."""
        self.refill()
        if self.tokens >= count:
            return 0.0
        deficit = count - self.tokens
        return deficit / self.refill_rate


class RateLimiter:
    """Per-key rate limiter using the token-bucket algorithm.

    Maintains separate token buckets for minute, hour, and day windows,
    each keyed by an arbitrary string (e.g. API key, IP address).

    Example::

        limiter = RateLimiter(RateLimitConfig(requests_per_minute=30))
        allowed, remaining, reset = limiter.check_rate("user-1")
        if not allowed:
            raise TooManyRequestsError(reset)
    """

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        self._config = config or RateLimitConfig()
        self._buckets: Dict[str, Dict[str, _TokenBucket]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_rate(self, key: str) -> Tuple[bool, int, float]:
        """Check whether a request from *key* is allowed.

        Consumes one token from each bucket (minute, hour, day).

        Args:
            key: Identifier for the client (API key, IP, etc.).

        Returns:
            A ``(allowed, remaining, reset_time)`` tuple:

            - *allowed*: ``True`` if the request is permitted.
            - *remaining*: Approximate requests remaining in the
              most restrictive window.
            - *reset_time*: Seconds until the rate limit resets
              (0.0 if allowed).
        """
        with self._lock:
            buckets = self._get_or_create_buckets(key)

            # Check all windows first without consuming
            for window_name, bucket in buckets.items():
                if not bucket.consume(0):  # type: ignore[arg-type]
                    # Refill-only check — consume(0) is a no-op but
                    # we need to test real consumption
                    pass

            # Actually consume tokens
            allowed = True
            min_remaining = float("inf")
            reset_time = 0.0

            for window_name, bucket in buckets.items():
                if not bucket.consume():
                    allowed = False
                    wait = bucket.time_until_available()
                    if wait > reset_time:
                        reset_time = wait
                else:
                    remaining = int(bucket.tokens)
                    if remaining < min_remaining:
                        min_remaining = remaining

            if allowed:
                return (True, int(min_remaining), 0.0)
            else:
                return (False, 0, math.ceil(reset_time))

    def get_remaining(self, key: str) -> Dict[str, int]:
        """Return remaining requests per window for *key*.

        Returns:
            A dict with ``minute``, ``hour``, and ``day`` keys.
        """
        with self._lock:
            buckets = self._get_or_create_buckets(key)
            result: Dict[str, int] = {}
            for window_name, bucket in buckets.items():
                bucket.refill()
                result[window_name] = int(bucket.tokens)
            return result

    def reset(self, key: str) -> None:
        """Reset all rate-limit buckets for *key*."""
        with self._lock:
            self._buckets.pop(key, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_buckets(self, key: str) -> Dict[str, _TokenBucket]:
        """Return (or create) the token buckets for *key*."""
        if key not in self._buckets:
            cfg = self._config
            self._buckets[key] = {
                "minute": _TokenBucket(
                    tokens=float(cfg.requests_per_minute),
                    max_tokens=float(cfg.requests_per_minute),
                    refill_rate=cfg.requests_per_minute / 60.0,
                ),
                "hour": _TokenBucket(
                    tokens=float(cfg.requests_per_hour),
                    max_tokens=float(cfg.requests_per_hour),
                    refill_rate=cfg.requests_per_hour / 3600.0,
                ),
                "day": _TokenBucket(
                    tokens=float(cfg.requests_per_day),
                    max_tokens=float(cfg.requests_per_day),
                    refill_rate=cfg.requests_per_day / 86400.0,
                ),
            }
        return self._buckets[key]
