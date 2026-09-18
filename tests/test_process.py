"""Test process management utilities for Nexus-LLM."""
import os
import sys
import signal
import subprocess
import pytest
from unittest.mock import patch, MagicMock


# --- Process management implementations to test ---

def get_process_id() -> int:
    return os.getpid()


def get_parent_process_id() -> int:
    return os.getppid()


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def terminate_process(pid: int, timeout: int = 5) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


class ProcessInfo:
    def __init__(self, pid, name="", status="unknown", cpu_percent=0.0, memory_mb=0.0):
        self.pid = pid
        self.name = name
        self.status = status
        self.cpu_percent = cpu_percent
        self.memory_mb = memory_mb


def get_current_process_info() -> ProcessInfo:
    return ProcessInfo(
        pid=os.getpid(),
        name="python",
        status="running",
        cpu_percent=0.0,
        memory_mb=0.0,
    )


def run_command(cmd: list, timeout: int = 30, capture: bool = True) -> dict:
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout if capture else "",
            "stderr": result.stderr if capture else "",
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": "Timeout", "success": False}
    except FileNotFoundError:
        return {"returncode": -1, "stdout": "", "stderr": "Command not found", "success": False}


class ProcessPool:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._processes = []

    def submit(self, func, *args):
        if len(self._processes) >= self.max_workers:
            raise RuntimeError("Process pool is full")
        proc_info = {"func": func, "args": args, "result": None, "completed": False}
        self._processes.append(proc_info)
        return len(self._processes) - 1

    def execute_all(self):
        for proc in self._processes:
            proc["result"] = proc["func"](*proc["args"])
            proc["completed"] = True

    def get_result(self, index):
        proc = self._processes[index]
        if not proc["completed"]:
            raise RuntimeError("Process not yet completed")
        return proc["result"]

    def active_count(self):
        return len([p for p in self._processes if not p["completed"]])


def set_environment(name: str, value: str) -> None:
    os.environ[name] = value


def get_environment(name: str, default: str = None) -> str:
    return os.environ.get(name, default)


def remove_environment(name: str) -> None:
    os.environ.pop(name, None)


class TestProcessIds:
    def test_get_pid(self):
        pid = get_process_id()
        assert isinstance(pid, int)
        assert pid > 0

    def test_get_ppid(self):
        ppid = get_parent_process_id()
        assert isinstance(ppid, int)
        assert ppid > 0

    def test_current_pid_is_self(self):
        assert get_process_id() == os.getpid()


class TestProcessAlive:
    def test_own_process_alive(self):
        assert is_process_alive(os.getpid()) is True

    def test_nonexistent_process(self):
        assert is_process_alive(999999999) is False

    def test_invalid_pid(self):
        assert is_process_alive(-1) is False


class TestProcessInfo:
    def test_creation(self):
        info = ProcessInfo(pid=1, name="test", status="running")
        assert info.pid == 1
        assert info.name == "test"
        assert info.status == "running"

    def test_defaults(self):
        info = ProcessInfo(pid=1)
        assert info.cpu_percent == 0.0
        assert info.memory_mb == 0.0

    def test_current_process_info(self):
        info = get_current_process_info()
        assert info.pid == os.getpid()
        assert info.name == "python"
        assert info.status == "running"


class TestRunCommand:
    def test_successful_command(self):
        result = run_command(["echo", "hello"])
        assert result["success"] is True
        assert "hello" in result["stdout"]

    def test_failed_command(self):
        result = run_command(["false"])
        assert result["success"] is False
        assert result["returncode"] != 0

    def test_nonexistent_command(self):
        result = run_command(["nonexistent_command_xyz"])
        assert result["success"] is False
        assert "not found" in result["stderr"].lower()

    def test_command_with_output(self):
        result = run_command(["python3", "-c", "print(42)"])
        assert result["success"] is True
        assert "42" in result["stdout"]


class TestProcessPool:
    def test_submit_and_execute(self):
        pool = ProcessPool(max_workers=2)
        idx = pool.submit(lambda x: x * 2, 5)
        pool.execute_all()
        assert pool.get_result(idx) == 10

    def test_multiple_submissions(self):
        pool = ProcessPool(max_workers=4)
        idx1 = pool.submit(lambda x: x + 1, 1)
        idx2 = pool.submit(lambda x: x + 1, 2)
        idx3 = pool.submit(lambda x: x + 1, 3)
        pool.execute_all()
        assert pool.get_result(idx1) == 2
        assert pool.get_result(idx2) == 3
        assert pool.get_result(idx3) == 4

    def test_pool_full(self):
        pool = ProcessPool(max_workers=2)
        pool.submit(lambda: None)
        pool.submit(lambda: None)
        with pytest.raises(RuntimeError, match="full"):
            pool.submit(lambda: None)

    def test_get_result_before_complete(self):
        pool = ProcessPool()
        idx = pool.submit(lambda: 42)
        with pytest.raises(RuntimeError, match="not yet completed"):
            pool.get_result(idx)

    def test_active_count(self):
        pool = ProcessPool()
        pool.submit(lambda: 1)
        pool.submit(lambda: 2)
        assert pool.active_count() == 2
        pool.execute_all()
        assert pool.active_count() == 0


class TestEnvironment:
    def test_set_and_get(self):
        set_environment("NEXUS_TEST_KEY", "test_value")
        assert get_environment("NEXUS_TEST_KEY") == "test_value"
        remove_environment("NEXUS_TEST_KEY")

    def test_get_default(self):
        assert get_environment("NEXUS_MISSING_KEY", "default") == "default"

    def test_remove(self):
        set_environment("NEXUS_TEMP_KEY", "temp")
        remove_environment("NEXUS_TEMP_KEY")
        assert get_environment("NEXUS_TEMP_KEY") is None
