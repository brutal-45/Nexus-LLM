"""Retry logic: exponential backoff, max retries, retryable exceptions."""

import time
import random
import logging
from typing import Optional, Callable, Any, Tuple, Type, List

logger = logging.getLogger(__name__)


class RetryPolicy:
    """Configurable retry policy with exponential backoff and jitter."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        jitter_range: float = 0.1,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
        on_failure: Optional[Callable[[int, Exception], None]] = None,
    ):
        """Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts.
            initial_delay: Initial delay in seconds before the first retry.
            max_delay: Maximum delay in seconds between retries.
            backoff_factor: Multiplier for exponential backoff.
            jitter: Whether to add random jitter to delays.
            jitter_range: Range of jitter as a fraction of the delay.
            retryable_exceptions: Tuple of exception types that should trigger a retry.
            on_retry: Callback(attempt, exception, delay) called before each retry.
            on_failure: Callback(total_attempts, exception) called after all retries fail.
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.jitter_range = jitter_range
        self.retryable_exceptions = retryable_exceptions or (Exception,)
        self.on_retry = on_retry
        self.on_failure = on_failure

    def compute_delay(self, attempt: int) -> float:
        """Compute the delay before the next retry.

        Args:
            attempt: The retry attempt number (0-based).

        Returns:
            Delay in seconds.
        """
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.0, delay)

        return delay

    def should_retry(self, exception: Exception) -> bool:
        """Check if the exception should trigger a retry.

        Args:
            exception: The caught exception.

        Returns:
            True if the exception is retryable.
        """
        return isinstance(exception, self.retryable_exceptions)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with retry logic.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The function's return value.

        Raises:
            The last exception if all retries fail.
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt >= self.max_retries or not self.should_retry(e):
                    if self.on_failure:
                        self.on_failure(attempt + 1, e)
                    raise

                delay = self.compute_delay(attempt)

                if self.on_retry:
                    self.on_retry(attempt + 1, e, delay)

                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed "
                    f"for {func.__name__}: {e}. Retrying in {delay:.1f}s..."
                )

                time.sleep(delay)

        raise last_exception


def with_retry(
    func: Optional[Callable] = None,
    *,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable] = None,
) -> Any:
    """Execute a function with retry logic. Can be used as a decorator or directly.

    As decorator:
        @with_retry(max_retries=3)
        def unstable_function():
            ...

    Direct call:
        result = with_retry(unstable_function, max_retries=3)

    Args:
        func: Function to execute (when used directly).
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        backoff_factor: Backoff multiplier.
        jitter: Whether to add jitter.
        retryable_exceptions: Exceptions that trigger retry.
        on_retry: Callback before each retry.

    Returns:
        Decorated function or function result.
    """
    policy = RetryPolicy(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        on_retry=on_retry,
    )

    if func is not None:
        # Used as a direct call: with_retry(func, max_retries=3)
        if callable(func) and not kwargs_were_passed(func):
            return policy.execute(func)

    # Used as a decorator: @with_retry(max_retries=3)
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return policy.execute(fn, *args, **kwargs)
        wrapper.__wrapped__ = fn
        wrapper.__name__ = fn.__name__
        return wrapper

    if func is not None and callable(func):
        return decorator(func)

    return decorator


def kwargs_were_passed(func):
    """Helper to detect if function was called with kwargs."""
    return False


class CircuitBreaker:
    """Circuit breaker pattern to prevent repeated calls to failing services."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_attempts: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half_open

    @property
    def state(self) -> str:
        if self._state == "open":
            if self._last_failure_time and (
                time.time() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = "half_open"
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function through the circuit breaker.

        Args:
            func: Function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            Exception: If circuit is open or function fails.
        """
        state = self.state

        if state == "open":
            raise RuntimeError("Circuit breaker is OPEN - calls are blocked")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self._failure_count = 0
        self._state = "closed"

    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = "open"

    def reset(self):
        """Reset the circuit breaker to closed state."""
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "closed"
