"""Decorators: timing, retry, cache, singleton, deprecation, validate_args."""

import time
import functools
import logging
import warnings
from typing import Optional, Callable, Any, Dict, Tuple

logger = logging.getLogger(__name__)


def timing(func: Optional[Callable] = None, *, log_level: str = "info"):
    """Decorator to time function execution.

    Can be used as @timing or @timing(log_level="debug").

    Args:
        func: Function to decorate (when used without parentheses).
        log_level: Logging level for timing messages.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log_fn = getattr(logger, log_level, logger.info)
            log_fn(f"{fn.__name__} took {elapsed:.4f}s")
            return result
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """Decorator to retry a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay between retries.
        exceptions: Tuple of exception types to catch.
        on_retry: Optional callback(attempt, exception) called before retry.

    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def unstable_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        if on_retry:
                            on_retry(attempt, e)
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


def cache_result(ttl: Optional[float] = None, max_size: int = 128):
    """Decorator to cache function results with optional TTL.

    Args:
        ttl: Time-to-live in seconds for cached results. None for no expiry.
        max_size: Maximum number of cached results.

    Example:
        @cache_result(ttl=60.0)
        def expensive_computation(key):
            ...
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[Any, Tuple[Any, float]] = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or (now - timestamp) < ttl:
                    return result
                del cache[key]

            result = func(*args, **kwargs)

            if len(cache) >= max_size:
                oldest_key = min(cache, key=lambda k: cache[k][1])
                del cache[oldest_key]

            cache[key] = (result, now)
            return result

        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_size = lambda: len(cache)
        return wrapper
    return decorator


def singleton(cls: type) -> type:
    """Decorator to make a class a singleton.

    Ensures only one instance of the class is created.

    Example:
        @singleton
        class DatabaseConnection:
            def __init__(self):
                self.connected = False
    """
    _instances: Dict[type, Any] = {}

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    get_instance._instances = _instances
    get_instance._original_class = cls
    return get_instance


def deprecated(
    reason: str = "",
    version: Optional[str] = None,
    alternative: Optional[str] = None,
):
    """Decorator to mark a function as deprecated.

    Args:
        reason: Reason for deprecation.
        version: Version in which the function was deprecated.
        alternative: Alternative function to use.

    Example:
        @deprecated(version="2.0", alternative="new_function")
        def old_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        msg = f"{func.__name__} is deprecated"
        if version:
            msg += f" since version {version}"
        if alternative:
            msg += f". Use {alternative} instead"
        if reason:
            msg += f". {reason}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__deprecated__ = True
        wrapper.__deprecation_message__ = msg
        return wrapper
    return decorator


def validate_args(**validators):
    """Decorator to validate function arguments.

    Args:
        **validators: Mapping of argument names to validation functions.
                     Each validator takes a value and returns True if valid.

    Example:
        @validate_args(x=lambda v: v > 0, name=lambda v: isinstance(v, str))
        def process(x, name):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for arg_name, validator in validators.items():
                if arg_name in bound.arguments:
                    value = bound.arguments[arg_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for argument '{arg_name}': "
                            f"invalid value {value!r}"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def memoize(func: Callable) -> Callable:
    """Simple memoization decorator (unbounded cache, no TTL)."""
    cache = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    wrapper.cache = cache
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def suppress_exceptions(*exceptions, default=None):
    """Decorator to suppress specified exceptions and return a default value.

    Args:
        *exceptions: Exception types to suppress.
        default: Default value to return when an exception is caught.

    Example:
        @suppress_exceptions(ValueError, TypeError, default=0)
        def safe_parse(text):
            return int(text)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.debug(f"Suppressed {type(e).__name__} in {func.__name__}: {e}")
                return default
        return wrapper
    return decorator
