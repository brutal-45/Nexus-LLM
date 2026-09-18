"""Test action executor for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional


class ExecutorError(Exception):
    pass


@dataclass
class Action:
    name: str
    handler: Callable
    args: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    retries: int = 0

    def execute(self) -> Any:
        return self.handler(**self.args)


@dataclass
class ExecutionResult:
    action_name: str
    success: bool
    result: Any = None
    error: str = ""
    attempts: int = 1
    elapsed: float = 0.0


class ActionExecutor:
    def __init__(self, max_retries: int = 0, default_timeout: float = 30.0):
        self._max_retries = max_retries
        self._default_timeout = default_timeout
        self._results: List[ExecutionResult] = []

    def execute(self, action: Action) -> ExecutionResult:
        attempts = 0
        last_error = ""
        import time
        start = time.perf_counter()
        while attempts <= action.retries:
            attempts += 1
            try:
                result = action.execute()
                elapsed = time.perf_counter() - start
                exec_result = ExecutionResult(
                    action_name=action.name, success=True,
                    result=result, attempts=attempts, elapsed=elapsed,
                )
                self._results.append(exec_result)
                return exec_result
            except Exception as e:
                last_error = str(e)
        elapsed = time.perf_counter() - start
        exec_result = ExecutionResult(
            action_name=action.name, success=False,
            error=last_error, attempts=attempts, elapsed=elapsed,
        )
        self._results.append(exec_result)
        return exec_result

    def execute_sequential(self, actions: List[Action]) -> List[ExecutionResult]:
        results = []
        for action in actions:
            result = self.execute(action)
            results.append(result)
            if not result.success:
                break
        return results

    def execute_parallel(self, actions: List[Action]) -> List[ExecutionResult]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(actions)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.execute, action): i for i, action in enumerate(actions)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def get_results(self) -> List[ExecutionResult]:
        return list(self._results)

    def clear_results(self):
        self._results.clear()


class TestAction:
    def test_creation(self):
        action = Action(name="add", handler=lambda a, b: a + b, args={"a": 1, "b": 2})
        assert action.name == "add"

    def test_execute(self):
        action = Action(name="add", handler=lambda a, b: a + b, args={"a": 3, "b": 4})
        assert action.execute() == 7

    def test_execute_with_error(self):
        action = Action(name="fail", handler=lambda: 1 / 0, args={})
        with pytest.raises(ZeroDivisionError):
            action.execute()


class TestExecutionResult:
    def test_success(self):
        result = ExecutionResult(action_name="test", success=True, result=42)
        assert result.success is True
        assert result.result == 42

    def test_failure(self):
        result = ExecutionResult(action_name="test", success=False, error="fail")
        assert result.success is False
        assert result.error == "fail"


class TestActionExecutor:
    def test_execute_success(self):
        executor = ActionExecutor()
        action = Action(name="add", handler=lambda a, b: a + b, args={"a": 1, "b": 2})
        result = executor.execute(action)
        assert result.success is True
        assert result.result == 3

    def test_execute_failure(self):
        executor = ActionExecutor()
        action = Action(name="fail", handler=lambda: 1 / 0, args={})
        result = executor.execute(action)
        assert result.success is False
        assert result.error != ""

    def test_execute_with_retry(self):
        call_count = [0]
        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("not yet")
            return "ok"
        action = Action(name="flaky", handler=flaky, args={}, retries=3)
        executor = ActionExecutor()
        result = executor.execute(action)
        assert result.success is True
        assert result.attempts == 3

    def test_execute_sequential(self):
        executor = ActionExecutor()
        actions = [
            Action(name="a1", handler=lambda: 1, args={}),
            Action(name="a2", handler=lambda: 2, args={}),
        ]
        results = executor.execute_sequential(actions)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_execute_sequential_stops_on_failure(self):
        executor = ActionExecutor()
        actions = [
            Action(name="fail", handler=lambda: 1 / 0, args={}),
            Action(name="ok", handler=lambda: 1, args={}),
        ]
        results = executor.execute_sequential(actions)
        assert len(results) == 1
        assert results[0].success is False

    def test_execute_parallel(self):
        executor = ActionExecutor()
        actions = [
            Action(name="a1", handler=lambda: 1, args={}),
            Action(name="a2", handler=lambda: 2, args={}),
            Action(name="a3", handler=lambda: 3, args={}),
        ]
        results = executor.execute_parallel(actions)
        assert len(results) == 3

    def test_results_stored(self):
        executor = ActionExecutor()
        executor.execute(Action(name="test", handler=lambda: 1, args={}))
        assert len(executor.get_results()) == 1

    def test_clear_results(self):
        executor = ActionExecutor()
        executor.execute(Action(name="test", handler=lambda: 1, args={}))
        executor.clear_results()
        assert len(executor.get_results()) == 0
