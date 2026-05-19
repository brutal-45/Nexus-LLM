"""Test code agent for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class CodeAgentConfig:
    name: str = "code-agent"
    language: str = "python"
    max_iterations: int = 5
    auto_execute: bool = False
    sandbox: bool = True


class CodeAgent:
    def __init__(self, config: CodeAgentConfig = None):
        self._config = config or CodeAgentConfig()
        self._artifacts: List[Dict[str, Any]] = []

    @property
    def config(self):
        return self._config

    @property
    def language(self):
        return self._config.language

    def generate_code(self, description: str, language: str = None) -> Dict[str, Any]:
        if not description:
            raise ValueError("Description cannot be empty")
        lang = language or self._config.language
        code = f"# Generated {lang} code for: {description}\ndef solution():\n    pass"
        artifact = {"type": "code", "language": lang, "code": code, "description": description}
        self._artifacts.append(artifact)
        return artifact

    def explain_code(self, code: str) -> str:
        if not code:
            raise ValueError("Code cannot be empty")
        return f"This code appears to be written in {self._detect_language(code)}. It implements some functionality."

    def debug_code(self, code: str, error_message: str = "") -> Dict[str, Any]:
        if not code:
            raise ValueError("Code cannot be empty")
        suggestions = []
        if "IndentationError" in error_message:
            suggestions.append("Check indentation consistency")
        if "NameError" in error_message:
            suggestions.append("Check for undefined variables")
        if "TypeError" in error_message:
            suggestions.append("Check type compatibility")
        if not suggestions:
            suggestions.append("Review the code for common errors")
        return {
            "error": error_message,
            "suggestions": suggestions,
            "fixed_code": code + "\n# Bug fix applied",
        }

    def test_code(self, code: str) -> Dict[str, Any]:
        if not code:
            raise ValueError("Code cannot be empty")
        return {
            "passed": True,
            "test_count": 3,
            "failures": [],
        }

    def review_code(self, code: str) -> Dict[str, Any]:
        if not code:
            raise ValueError("Code cannot be empty")
        issues = []
        if len(code.split("\n")) > 50:
            issues.append({"type": "style", "message": "Function is too long"})
        if "except:" in code:
            issues.append({"type": "best_practice", "message": "Bare except clause"})
        return {
            "issues": issues,
            "score": max(0, 10 - len(issues)),
        }

    def _detect_language(self, code: str) -> str:
        if "def " in code or "import " in code:
            return "python"
        if "function " in code or "const " in code:
            return "javascript"
        return "unknown"

    def get_artifacts(self) -> List[Dict[str, Any]]:
        return list(self._artifacts)

    def clear_artifacts(self):
        self._artifacts.clear()


class TestCodeAgentConfig:
    def test_defaults(self):
        config = CodeAgentConfig()
        assert config.language == "python"
        assert config.auto_execute is False
        assert config.sandbox is True


class TestCodeAgent:
    def test_generate_code(self):
        agent = CodeAgent()
        result = agent.generate_code("sort a list")
        assert result["type"] == "code"
        assert "sort" in result["description"]

    def test_generate_code_empty(self):
        agent = CodeAgent()
        with pytest.raises(ValueError, match="empty"):
            agent.generate_code("")

    def test_generate_code_custom_language(self):
        agent = CodeAgent()
        result = agent.generate_code("hello world", language="rust")
        assert result["language"] == "rust"

    def test_explain_code(self):
        agent = CodeAgent()
        explanation = agent.explain_code("def hello(): pass")
        assert "code" in explanation.lower()

    def test_explain_code_empty(self):
        agent = CodeAgent()
        with pytest.raises(ValueError):
            agent.explain_code("")

    def test_debug_code(self):
        agent = CodeAgent()
        result = agent.debug_code("x = 1 / 0", "ZeroDivisionError")
        assert "suggestions" in result

    def test_debug_with_specific_error(self):
        agent = CodeAgent()
        result = agent.debug_code("print(x)", "NameError: name 'x' is not defined")
        assert any("undefined" in s.lower() for s in result["suggestions"])

    def test_test_code(self):
        agent = CodeAgent()
        result = agent.test_code("def add(a, b): return a + b")
        assert "passed" in result
        assert "test_count" in result

    def test_review_code(self):
        agent = CodeAgent()
        result = agent.review_code("def hello(): print('hi')")
        assert "issues" in result
        assert "score" in result

    def test_review_code_with_issues(self):
        agent = CodeAgent()
        code = "try:\n    pass\nexcept:\n    pass"
        result = agent.review_code(code)
        assert len(result["issues"]) > 0

    def test_artifacts_stored(self):
        agent = CodeAgent()
        agent.generate_code("task1")
        agent.generate_code("task2")
        assert len(agent.get_artifacts()) == 2

    def test_clear_artifacts(self):
        agent = CodeAgent()
        agent.generate_code("task1")
        agent.clear_artifacts()
        assert len(agent.get_artifacts()) == 0

    def test_language_property(self):
        agent = CodeAgent(CodeAgentConfig(language="javascript"))
        assert agent.language == "javascript"
