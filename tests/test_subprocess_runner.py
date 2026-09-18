"""Tests for subprocess runner."""
import pytest
import subprocess


def test_subprocess_run_simple():
    result = subprocess.run(["echo", "hello"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "hello" in result.stdout

def test_subprocess_run_with_args():
    result = subprocess.run(["python3", "-c", "print(42)"], capture_output=True, text=True)
    assert "42" in result.stdout

def test_subprocess_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(["sleep", "10"], timeout=0.1)

def test_subprocess_return_code():
    result = subprocess.run(["false"])
    assert result.returncode != 0
