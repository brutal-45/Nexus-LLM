"""Test rate limiting for Nexus-LLM."""
import time
import pytest
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional


class RateLimitError(Exception):
    pass


@dataclass
class RateLimitConfig:
    max_requests: int = 60
    window_seconds: int = 60
    burst_limit: int = 10


class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def _refill(self):
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    @property
    def available_tokens(self):
        self._refill()
        return self._tokens


class SlidingWindowCounter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        now = time.time()
        cutoff = now - self.window_seconds
        current = [t for t in self._requests[key] if t > cutoff]
        return max(0, self.max_requests - len(current))

    def reset(self, key: str = None):
        if key:
            self._requests[key] = []
        else:
            self._requests.clear()


class RateLimiter:
    def __init__(self, config: RateLimitConfig = None):
        self._config = config or RateLimitConfig()
        self._counter = SlidingWindowCounter(self._config.max_requests, self._config.window_seconds)

    def check(self, key: str) -> bool:
        return self._counter.is_allowed(key)

    def get_remaining(self, key: str) -> int:
        return self._counter.get_remaining(key)

    def reset(self, key: str = None):
        self._counter.reset(key)


class TestTokenBucket:
    def test_initial_capacity(self):
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.available_tokens == 10.0

    def test_consume(self):
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume(1) is True
        assert bucket.available_tokens < 10

    def test_consume_more_than_available(self):
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        assert bucket.consume(6) is False

    def test_refill(self):
        bucket = TokenBucket(capacity=10, refill_rate=1000.0)
        bucket.consume(10)
        time.sleep(0.01)
        assert bucket.available_tokens > 0

    def test_max_capacity(self):
        bucket = TokenBucket(capacity=10, refill_rate=1000.0)
        time.sleep(0.01)
        assert bucket.available_tokens <= 10.0


class TestSlidingWindowCounter:
    def test_allows_within_limit(self):
        counter = SlidingWindowCounter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert counter.is_allowed("user1") is True

    def test_blocks_over_limit(self):
        counter = SlidingWindowCounter(max_requests=3, window_seconds=60)
        for _ in range(3):
            counter.is_allowed("user1")
        assert counter.is_allowed("user1") is False

    def test_different_keys_independent(self):
        counter = SlidingWindowCounter(max_requests=2, window_seconds=60)
        counter.is_allowed("user1")
        counter.is_allowed("user1")
        assert counter.is_allowed("user1") is False
        assert counter.is_allowed("user2") is True

    def test_get_remaining(self):
        counter = SlidingWindowCounter(max_requests=5, window_seconds=60)
        counter.is_allowed("user1")
        counter.is_allowed("user1")
        assert counter.get_remaining("user1") == 3

    def test_reset_specific_key(self):
        counter = SlidingWindowCounter(max_requests=2, window_seconds=60)
        counter.is_allowed("user1")
        counter.is_allowed("user1")
        counter.reset("user1")
        assert counter.is_allowed("user1") is True

    def test_reset_all(self):
        counter = SlidingWindowCounter(max_requests=1, window_seconds=60)
        counter.is_allowed("user1")
        counter.is_allowed("user2")
        counter.reset()
        assert counter.is_allowed("user1") is True
        assert counter.is_allowed("user2") is True


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(RateLimitConfig(max_requests=5, window_seconds=60))
        for _ in range(5):
            assert limiter.check("user1") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(RateLimitConfig(max_requests=3, window_seconds=60))
        for _ in range(3):
            limiter.check("user1")
        assert limiter.check("user1") is False

    def test_get_remaining(self):
        limiter = RateLimiter(RateLimitConfig(max_requests=10, window_seconds=60))
        limiter.check("user1")
        assert limiter.get_remaining("user1") == 9

    def test_reset(self):
        limiter = RateLimiter(RateLimitConfig(max_requests=1, window_seconds=60))
        limiter.check("user1")
        limiter.reset("user1")
        assert limiter.check("user1") is True
