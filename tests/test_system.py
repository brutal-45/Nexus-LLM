"""Test system information utilities for Nexus-LLM."""
import os
import sys
import platform
import pytest
from unittest.mock import patch, MagicMock


# --- System info implementations to test ---

def get_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_platform() -> str:
    return platform.system()


def get_cpu_count() -> int:
    return os.cpu_count() or 1


def get_memory_info() -> dict:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent_used": mem.percent,
        }
    except ImportError:
        return {"total_gb": 0, "available_gb": 0, "percent_used": 0}


def is_gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_info() -> list:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_mem / (1024**3), 2),
            })
        return info
    except ImportError:
        return []


def get_system_info() -> dict:
    return {
        "python_version": get_python_version(),
        "platform": get_platform(),
        "cpu_count": get_cpu_count(),
        "memory": get_memory_info(),
        "gpu_available": is_gpu_available(),
    }


def check_min_python(major: int, minor: int) -> bool:
    return sys.version_info >= (major, minor)


def get_env_var(name: str, default: str = None) -> str:
    return os.environ.get(name, default)


class TestPythonVersion:
    def test_returns_string(self):
        version = get_python_version()
        assert isinstance(version, str)

    def test_version_format(self):
        version = get_python_version()
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_version_is_current(self):
        version = get_python_version()
        major = sys.version_info.major
        assert version.startswith(f"{major}.")


class TestPlatform:
    def test_returns_known_platform(self):
        p = get_platform()
        assert p in ("Linux", "Windows", "Darwin", "FreeBSD")

    def test_platform_matches_sys(self):
        assert get_platform() == platform.system()


class TestCPUCount:
    def test_returns_int(self):
        count = get_cpu_count()
        assert isinstance(count, int)

    def test_count_positive(self):
        count = get_cpu_count()
        assert count >= 1

    def test_count_matches_os(self):
        assert get_cpu_count() == (os.cpu_count() or 1)


class TestMemoryInfo:
    def test_returns_dict(self):
        info = get_memory_info()
        assert isinstance(info, dict)

    def test_has_required_keys(self):
        info = get_memory_info()
        assert "total_gb" in info
        assert "available_gb" in info
        assert "percent_used" in info

    def test_values_are_numeric(self):
        info = get_memory_info()
        assert isinstance(info["total_gb"], (int, float))
        assert isinstance(info["percent_used"], (int, float))


class TestGPUInfo:
    def test_gpu_available_returns_bool(self):
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_gpu_info_returns_list(self):
        info = get_gpu_info()
        assert isinstance(info, list)


class TestSystemInfo:
    def test_composite_info(self):
        info = get_system_info()
        assert "python_version" in info
        assert "platform" in info
        assert "cpu_count" in info
        assert "memory" in info
        assert "gpu_available" in info

    def test_info_values_valid(self):
        info = get_system_info()
        assert isinstance(info["cpu_count"], int)
        assert info["cpu_count"] >= 1
        assert isinstance(info["gpu_available"], bool)


class TestMinPythonCheck:
    def test_python_3_7_check(self):
        assert check_min_python(3, 7) is True

    def test_python_3_0_check(self):
        assert check_min_python(3, 0) is True

    def test_python_99_check(self):
        assert check_min_python(99, 0) is False


class TestEnvVar:
    def test_existing_var(self):
        os.environ["NEXUS_TEST_VAR"] = "test_value"
        assert get_env_var("NEXUS_TEST_VAR") == "test_value"
        del os.environ["NEXUS_TEST_VAR"]

    def test_missing_var_returns_none(self):
        assert get_env_var("NEXUS_NONEXISTENT_VAR") is None

    def test_missing_var_with_default(self):
        assert get_env_var("NEXUS_NONEXISTENT_VAR", "fallback") == "fallback"
