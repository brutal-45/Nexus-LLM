"""Tests for nexus_llm.security.sandbox module."""

import pytest
from nexus_llm.security.sandbox import CodeSandbox


class TestCodeSandbox:
    """Tests for the CodeSandbox class."""

    def test_init(self):
        sandbox = CodeSandbox()
        assert sandbox is not None

    def test_execute_safe_code(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("result = 2 + 2")
        assert result["success"] is True
        assert result["output"] == 4

    def test_execute_unsafe_code(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("import os; os.system('echo hack')")
        assert result["success"] is False

    def test_execute_syntax_error(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("invalid python syntax !!")
        assert result["success"] is False

    def test_execute_timeout(self):
        sandbox = CodeSandbox(timeout=1)
        result = sandbox.execute("while True: pass")
        assert result["success"] is False

    def test_execute_with_globals(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("result = x * 2", globals={"x": 21})
        assert result["success"] is True
        assert result["output"] == 42
