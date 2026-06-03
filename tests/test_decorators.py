"""Test decorator utilities for Nexus-LLM."""
import time
import functools
import pytest
from unittest.mock import MagicMock, patch


# --- Decorator implementations to test ---

def deprecated(message: str = ""):
    """Mark a function as deprecated."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import warnings
            msg = f"{func.__name__} is deprecated."
            if message:
                msg += f" {message}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 0.01, exceptions: tuple = (Exception,)):
    """Retry a function on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def validate_args(**validators):
    """Validate function arguments using provided validators."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Validation failed for parameter '{param_name}': {value}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def memoize(func):
    """Simple memoization decorator."""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache
    return wrapper


def timing(func):
    """Measure and print execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        wrapper.last_elapsed = elapsed
        return result
    wrapper.last_elapsed = None
    return wrapper


def singleton(cls):
    """Make a class a singleton."""
    instances = {}
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    get_instance._instances = instances
    return get_instance


def typecheck(**expected_types):
    """Enforce type checking on function arguments."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for param_name, expected_type in expected_types.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' expected {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class TestDeprecatedDecorator:
    def test_function_still_works(self):
        @deprecated()
        def old_func():
            return 42
        assert old_func() == 42

    def test_emits_deprecation_warning(self):
        @deprecated()
        def old_func():
            return 1
        with pytest.warns(DeprecationWarning, match="deprecated"):
            old_func()

    def test_custom_message(self):
        @deprecated("Use new_func instead")
        def old_func():
            return 1
        with pytest.warns(DeprecationWarning, match="Use new_func"):
            old_func()

    def test_preserves_name(self):
        @deprecated()
        def my_func():
            pass
        assert my_func.__name__ == "my_func"


class TestRetryDecorator:
    def test_succeeds_first_try(self):
        call_count = 0
        @retry(max_attempts=3)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"
        result = succeed()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_failure(self):
        call_count = 0
        @retry(max_attempts=3, delay=0.001)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"
        result = fail_then_succeed()
        assert result == "ok"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        @retry(max_attempts=2, delay=0.001)
        def always_fail():
            raise RuntimeError("fail")
        with pytest.raises(RuntimeError, match="fail"):
            always_fail()

    def test_specific_exceptions_only(self):
        call_count = 0
        @retry(max_attempts=3, delay=0.001, exceptions=(ValueError,))
        def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong type")
        with pytest.raises(TypeError):
            raise_type_error()
        assert call_count == 1


class TestValidateArgsDecorator:
    def test_valid_args_pass(self):
        @validate_args(x=lambda v: v > 0)
        def func(x):
            return x * 2
        assert func(5) == 10

    def test_invalid_args_fail(self):
        @validate_args(x=lambda v: v > 0)
        def func(x):
            return x
        with pytest.raises(ValueError, match="Validation failed"):
            func(-1)

    def test_multiple_validators(self):
        @validate_args(
            name=lambda v: len(v) > 0,
            age=lambda v: 0 < v < 150,
        )
        def func(name, age):
            return f"{name} is {age}"
        assert func("Alice", 30) == "Alice is 30"


class TestMemoizeDecorator:
    def test_caches_result(self):
        call_count = 0
        @memoize
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        assert expensive(5) == 10
        assert expensive(5) == 10
        assert call_count == 1

    def test_different_args_different_cache(self):
        @memoize
        def func(x):
            return x * 2
        assert func(1) == 2
        assert func(2) == 4

    def test_cache_attribute(self):
        @memoize
        def func(x):
            return x
        func(1)
        assert hasattr(func, "cache")
        assert (1,) in func.cache


class TestTimingDecorator:
    def test_measures_time(self):
        @timing
        def slow_func():
            time.sleep(0.01)
            return 42
        result = slow_func()
        assert result == 42
        assert slow_func.last_elapsed is not None
        assert slow_func.last_elapsed >= 0.01

    def test_fast_function(self):
        @timing
        def fast_func():
            return 1
        fast_func()
        assert fast_func.last_elapsed >= 0


class TestSingletonDecorator:
    def test_same_instance(self):
        @singleton
        class MyClass:
            def __init__(self, value=0):
                self.value = value
        a = MyClass(1)
        b = MyClass(2)
        assert a is b
        assert a.value == 1

    def test_different_classes_different_instances(self):
        @singleton
        class A:
            pass
        @singleton
        class B:
            pass
        assert A() is not B()


class TestTypecheckDecorator:
    def test_correct_types_pass(self):
        @typecheck(x=int, y=str)
        def func(x, y):
            return f"{y}: {x}"
        assert func(42, "answer") == "answer: 42"

    def test_wrong_type_raises(self):
        @typecheck(x=int)
        def func(x):
            return x
        with pytest.raises(TypeError, match="expected int"):
            func("not an int")

    def test_multiple_type_checks(self):
        @typecheck(name=str, count=int)
        def func(name, count):
            return name * count
        assert func("ha", 3) == "hahaha"
