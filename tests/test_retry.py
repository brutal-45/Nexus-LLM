"""Test retry logic utilities for Nexus-LLM."""
import time
import pytest
from unittest.mock import MagicMock, patch, call


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts, last_exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_exception}")


class RetryConfig:
    def __init__(self, max_attempts=3, base_delay=0.01, max_delay=1.0,
                 exponential_base=2, jitter=False, exceptions=(Exception,)):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions


def calculate_backoff(attempt, config):
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)
    if config.jitter:
        import random
        delay *= (0.5 + random.random() * 0.5)
    return delay


def retry_with_config(func, config, *args, **kwargs):
    last_exception = None
    for attempt in range(config.max_attempts):
        try:
            return func(*args, **kwargs)
        except config.exceptions as e:
            last_exception = e
            if attempt < config.max_attempts - 1:
                delay = calculate_backoff(attempt, config)
                time.sleep(delay)
    raise RetryExhausted(config.max_attempts, last_exception)


def retry_with_callback(func, config, on_retry=None, *args, **kwargs):
    last_exception = None
    for attempt in range(config.max_attempts):
        try:
            return func(*args, **kwargs)
        except config.exceptions as e:
            last_exception = e
            if on_retry:
                on_retry(attempt, e)
            if attempt < config.max_attempts - 1:
                delay = calculate_backoff(attempt, config)
                time.sleep(delay)
    raise RetryExhausted(config.max_attempts, last_exception)


class TestRetryConfig:
    def test_default_config(self):
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 0.01
        assert config.max_delay == 1.0
        assert config.exponential_base == 2
        assert config.jitter is False

    def test_custom_config(self):
        config = RetryConfig(max_attempts=5, base_delay=0.1)
        assert config.max_attempts == 5
        assert config.base_delay == 0.1


class TestCalculateBackoff:
    def test_first_attempt_delay(self):
        config = RetryConfig(base_delay=0.01)
        delay = calculate_backoff(0, config)
        assert delay == 0.01

    def test_exponential_growth(self):
        config = RetryConfig(base_delay=0.01, exponential_base=2)
        d0 = calculate_backoff(0, config)
        d1 = calculate_backoff(1, config)
        d2 = calculate_backoff(2, config)
        assert d1 > d0
        assert d2 > d1

    def test_max_delay_cap(self):
        config = RetryConfig(base_delay=0.01, max_delay=0.05, exponential_base=10)
        delay = calculate_backoff(5, config)
        assert delay <= 0.05

    def test_jitter_modifies_delay(self):
        config = RetryConfig(base_delay=1.0, jitter=True)
        delays = {calculate_backoff(0, config) for _ in range(100)}
        assert len(delays) > 1


class TestRetryWithConfig:
    def test_succeeds_immediately(self):
        config = RetryConfig(max_attempts=3)
        result = retry_with_config(lambda: 42, config)
        assert result == 42

    def test_retries_and_succeeds(self):
        attempts = [0]
        def flaky():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("fail")
            return "success"
        config = RetryConfig(max_attempts=5, base_delay=0.001)
        result = retry_with_config(flaky, config)
        assert result == "success"
        assert attempts[0] == 3

    def test_exhausts_retries(self):
        def always_fail():
            raise RuntimeError("always fails")
        config = RetryConfig(max_attempts=2, base_delay=0.001)
        with pytest.raises(RetryExhausted, match="exhausted after 2"):
            retry_with_config(always_fail, config)

    def test_specific_exceptions_only(self):
        call_count = [0]
        def raise_unexpected():
            call_count[0] += 1
            raise TypeError("unexpected")
        config = RetryConfig(max_attempts=3, base_delay=0.001, exceptions=(ValueError,))
        with pytest.raises(TypeError):
            retry_with_config(raise_unexpected, config)
        assert call_count[0] == 1

    def test_preserves_return_value(self):
        config = RetryConfig()
        result = retry_with_config(lambda: {"key": "value"}, config)
        assert result == {"key": "value"}


class TestRetryWithCallback:
    def test_callback_called_on_retry(self):
        attempts = [0]
        callback = MagicMock()
        def flaky():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("fail")
            return "ok"
        config = RetryConfig(max_attempts=5, base_delay=0.001)
        retry_with_callback(flaky, config, on_retry=callback)
        assert callback.call_count == 2

    def test_callback_receives_attempt_and_error(self):
        callback = MagicMock()
        def flaky():
            raise ValueError("fail")
        config = RetryConfig(max_attempts=2, base_delay=0.001)
        with pytest.raises(RetryExhausted):
            retry_with_callback(flaky, config, on_retry=callback)
        callback.assert_called()
        args = callback.call_args[0]
        assert isinstance(args[0], int)
        assert isinstance(args[1], ValueError)

    def test_no_callback_when_succeeds(self):
        callback = MagicMock()
        config = RetryConfig(max_attempts=3)
        retry_with_callback(lambda: 1, config, on_retry=callback)
        callback.assert_not_called()


class TestRetryExhausted:
    def test_contains_attempt_count(self):
        err = RetryExhausted(5, RuntimeError("test"))
        assert err.attempts == 5

    def test_contains_last_exception(self):
        original = RuntimeError("original")
        err = RetryExhausted(3, original)
        assert err.last_exception is original

    def test_message_format(self):
        err = RetryExhausted(3, ValueError("bad"))
        assert "3" in str(err)
        assert "bad" in str(err)
