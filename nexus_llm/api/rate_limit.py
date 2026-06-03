"""Rate limiting: token bucket, sliding window, per-user limits."""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nexus_llm.api.errors import RateLimitExceededError

logger = logging.getLogger("nexus_llm.api.rate_limit")


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    requests_per_day: int = 86400
    tokens_per_minute: int = 100000
    tokens_per_day: int = 1000000
    burst_size: int = 10
    enabled: bool = True


class TokenBucket:
    """Token bucket rate limiter implementation.

    Allows burst traffic up to the bucket capacity while maintaining
    an average rate governed by the refill rate.
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        initial_tokens: Optional[int] = None,
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(initial_tokens if initial_tokens is not None else capacity)
        self.last_refill_time = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if insufficient.
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_and_consume(self, tokens: int = 1, timeout: float = 30.0) -> bool:
        """Wait until tokens are available, then consume.

        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if consumed within timeout, False otherwise.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.consume(tokens):
                return True
            wait_time = tokens / self.refill_rate
            time.sleep(min(wait_time, 0.1))
        return False

    def get_available(self) -> float:
        """Return the current number of available tokens."""
        with self._lock:
            self._refill()
            return self.tokens

    def get_wait_time(self, tokens: int = 1) -> float:
        """Calculate how long to wait for the requested tokens.

        Args:
            tokens: Number of tokens needed.

        Returns:
            Wait time in seconds (0 if tokens are available).
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            deficit = tokens - self.tokens
            return deficit / self.refill_rate

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_refill_time = now

    def reset(self) -> None:
        """Reset the bucket to full capacity."""
        with self._lock:
            self.tokens = float(self.capacity)
            self.last_refill_time = time.time()


class SlidingWindowCounter:
    """Sliding window rate limiter using time-bucketed counters.

    Provides accurate rate limiting with configurable window sizes
    for per-minute, per-hour, and per-day limits.
    """

    def __init__(
        self,
        window_sizes: Optional[List[int]] = None,
        max_requests: Optional[Dict[int, int]] = None,
    ):
        self.window_sizes = window_sizes or [60, 3600, 86400]
        self.max_requests = max_requests or {
            60: 60,
            3600: 3600,
            86400: 86400,
        }
        self._counters: Dict[str, Dict[int, List[Tuple[float, int]]]] = defaultdict(
            lambda: {ws: [] for ws in self.window_sizes}
        )
        self._lock = threading.Lock()

    def check(self, key: str, increment: int = 1) -> Tuple[bool, Dict[str, Any]]:
        """Check if a request is within rate limits.

        Args:
            key: Identifier (user ID, API key, IP).
            increment: Number to add to the counter.

        Returns:
            Tuple of (is_allowed, rate_limit_info).
        """
        with self._lock:
            now = time.time()
            user_counters = self._counters[key]
            is_allowed = True
            info: Dict[str, Any] = {}

            for window_size in self.window_sizes:
                window_start = now - window_size
                user_counters[window_size] = [
                    (ts, count)
                    for ts, count in user_counters[window_size]
                    if ts > window_start
                ]

                current_count = sum(c for _, c in user_counters[window_size])
                max_for_window = self.max_requests.get(window_size, float("inf"))

                if current_count + increment > max_for_window:
                    is_allowed = False
                    oldest = min(
                        (ts for ts, _ in user_counters[window_size]),
                        default=now,
                    )
                    reset_at = oldest + window_size

                    info[f"window_{window_size}s"] = {
                        "current": current_count,
                        "limit": max_for_window,
                        "remaining": max(0, max_for_window - current_count),
                        "reset_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%S", time.localtime(reset_at)
                        ),
                    }
                else:
                    info[f"window_{window_size}s"] = {
                        "current": current_count + increment,
                        "limit": max_for_window,
                        "remaining": max(0, max_for_window - current_count - increment),
                    }

            if is_allowed:
                for window_size in self.window_sizes:
                    user_counters[window_size].append((now, increment))

            return is_allowed, info

    def get_usage(self, key: str) -> Dict[str, Any]:
        """Get current usage stats for a key.

        Args:
            key: User identifier.

        Returns:
            Dictionary with usage information per window.
        """
        with self._lock:
            now = time.time()
            user_counters = self._counters.get(key, {})
            result = {}

            for window_size in self.window_sizes:
                entries = user_counters.get(window_size, [])
                window_start = now - window_size
                active = [(ts, c) for ts, c in entries if ts > window_start]
                current = sum(c for _, c in active)
                limit = self.max_requests.get(window_size, 0)
                result[f"window_{window_size}s"] = {
                    "current": current,
                    "limit": limit,
                    "remaining": max(0, limit - current),
                }

            return result

    def reset(self, key: Optional[str] = None) -> None:
        """Reset counters for a specific key or all keys.

        Args:
            key: Optional key to reset. If None, resets all.
        """
        with self._lock:
            if key:
                self._counters.pop(key, None)
            else:
                self._counters.clear()

    def cleanup(self, max_age: int = 86400) -> int:
        """Remove expired entries to free memory.

        Args:
            max_age: Maximum age of entries to keep in seconds.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            now = time.time()
            removed = 0
            for key in list(self._counters.keys()):
                for window_size in self.window_sizes:
                    before = len(self._counters[key][window_size])
                    self._counters[key][window_size] = [
                        (ts, c) for ts, c in self._counters[key][window_size]
                        if now - ts < max_age
                    ]
                    removed += before - len(self._counters[key][window_size])
            return removed


class RateLimiter:
    """Composite rate limiter combining token bucket and sliding window.

    Provides both burst control (token bucket) and sustained rate
    limiting (sliding window) with per-user and per-model granularity.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._token_buckets: Dict[str, TokenBucket] = {}
        self._sliding_window = SlidingWindowCounter(
            max_requests={
                60: self.config.requests_per_minute,
                3600: self.config.requests_per_hour,
                86400: self.config.requests_per_day,
            }
        )
        self._token_usage: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def check_rate(
        self,
        key: str,
        tokens: int = 1,
        model: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if a request is within rate limits.

        Args:
            key: User identifier (API key hash, user ID, IP).
            tokens: Number of tokens in the request.
            model: Optional model name for model-specific limits.

        Returns:
            Tuple of (is_allowed, rate_limit_info).
        """
        if not self.config.enabled:
            return True, {"rate_limit": "disabled"}

        is_allowed, window_info = self._sliding_window.check(key)

        bucket = self._get_or_create_bucket(key)
        bucket_allowed = bucket.consume()
        if not bucket_allowed:
            is_allowed = False
            window_info["burst_limit"] = {
                "capacity": self.config.burst_size,
                "refill_rate": self.config.requests_per_minute / 60.0,
                "wait_time_seconds": bucket.get_wait_time(),
            }

        token_bucket = self._get_or_create_token_bucket(key)
        token_allowed = token_bucket.consume(tokens)
        if not token_allowed:
            is_allowed = False
            window_info["token_limit"] = {
                "per_minute": self.config.tokens_per_minute,
                "wait_time_seconds": token_bucket.get_wait_time(tokens),
            }

        window_info["allowed"] = is_allowed
        return is_allowed, window_info

    def enforce_rate(
        self,
        key: str,
        tokens: int = 1,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enforce rate limits, raising an error if exceeded.

        Args:
            key: User identifier.
            tokens: Number of tokens in the request.
            model: Optional model name.

        Returns:
            Rate limit info if allowed.

        Raises:
            RateLimitExceededError: If rate limit is exceeded.
        """
        is_allowed, info = self.check_rate(key, tokens, model)
        if not is_allowed:
            window_info = {k: v for k, v in info.items() if k != "allowed"}
            raise RateLimitExceededError(
                limit=str(window_info),
                reset_at=info.get("window_60s", {}).get("reset_at"),
            )
        return info

    def get_status(self, key: str) -> Dict[str, Any]:
        """Get rate limit status for a user.

        Args:
            key: User identifier.

        Returns:
            Dictionary with current rate limit status.
        """
        window_status = self._sliding_window.get_usage(key)
        bucket = self._get_or_create_bucket(key)
        token_bucket = self._get_or_create_token_bucket(key)

        return {
            "sliding_window": window_status,
            "burst_bucket": {
                "available": bucket.get_available(),
                "capacity": self.config.burst_size,
            },
            "token_bucket": {
                "available": token_bucket.get_available(),
                "per_minute_limit": self.config.tokens_per_minute,
            },
        }

    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limit counters.

        Args:
            key: Optional key to reset. If None, resets all.
        """
        with self._lock:
            if key:
                self._token_buckets.pop(key, None)
                self._token_usage.pop(key, None)
            else:
                self._token_buckets.clear()
                self._token_usage.clear()
        self._sliding_window.reset(key)

    def _get_or_create_bucket(self, key: str) -> TokenBucket:
        """Get or create a token bucket for a user."""
        with self._lock:
            if key not in self._token_buckets:
                self._token_buckets[key] = TokenBucket(
                    capacity=self.config.burst_size,
                    refill_rate=self.config.requests_per_minute / 60.0,
                )
            return self._token_buckets[key]

    def _get_or_create_token_bucket(self, key: str) -> TokenBucket:
        """Get or create a token-usage bucket for a user."""
        with self._lock:
            if key not in self._token_usage:
                self._token_usage[key] = TokenBucket(
                    capacity=self.config.tokens_per_minute,
                    refill_rate=self.config.tokens_per_minute / 60.0,
                )
            return self._token_usage[key]


# Global rate limiter
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter singleton."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def init_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Initialize the global rate limiter with configuration."""
    global _rate_limiter
    _rate_limiter = RateLimiter(config)
    return _rate_limiter
