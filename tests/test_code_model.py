"""Test code model wrapper for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


class ModelError(Exception):
    pass


@dataclass
class CodeConfig:
    name: str = "code-model"
    max_length: int = 4096
    temperature: float = 0.2
    top_p: float = 0.8
    language: str = "python"
    stop_tokens: List[str] = None

    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = ["```", "<|endoftext|>"]


class CodeModel:
    def __init__(self, config: CodeConfig = None):
        self._config = config or CodeConfig()
        self._loaded = False

    @property
    def config(self):
        return self._config

    @property
    def is_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def generate_code(self, prompt: str, language: str = None, **kwargs) -> str:
        if not self._loaded:
            raise ModelError("Model not loaded")
        if not prompt:
            raise ModelError("Prompt cannot be empty")
        lang = language or self._config.language
        return f"[CODE:{lang}] {prompt}"

    def complete_code(self, code: str, cursor_position: int = None, **kwargs) -> str:
        if not self._loaded:
            raise ModelError("Model not loaded")
        return f"[COMPLETE] {code}"

    def explain_code(self, code: str, **kwargs) -> str:
        if not self._loaded:
            raise ModelError("Model not loaded")
        if not code:
            raise ModelError("Code cannot be empty")
        return f"[EXPLAIN] This code does: {code[:50]}"

    def refactor_code(self, code: str, instructions: str = "", **kwargs) -> str:
        if not self._loaded:
            raise ModelError("Model not loaded")
        return f"[REFACTOR] {code}"

    def generate_tests(self, code: str, framework: str = "pytest", **kwargs) -> str:
        if not self._loaded:
            raise ModelError("Model not loaded")
        return f"[TEST:{framework}] {code}"

    def detect_language(self, code: str) -> str:
        if "def " in code or "import " in code:
            return "python"
        if "function " in code or "const " in code or "=>" in code:
            return "javascript"
        if "fn " in code and "let " in code:
            return "rust"
        if "#include" in code:
            return "c"
        return "unknown"

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self._config.name,
            "type": "code",
            "language": self._config.language,
            "max_length": self._config.max_length,
            "is_loaded": self._loaded,
        }


class TestCodeConfig:
    def test_defaults(self):
        config = CodeConfig()
        assert config.temperature == 0.2
        assert config.language == "python"
        assert len(config.stop_tokens) >= 1

    def test_custom(self):
        config = CodeConfig(name="codellama", language="javascript")
        assert config.language == "javascript"


class TestCodeModel:
    def test_init(self):
        model = CodeModel()
        assert model.is_loaded is False

    def test_load_unload(self):
        model = CodeModel()
        model.load()
        assert model.is_loaded is True
        model.unload()
        assert model.is_loaded is False

    def test_generate_code(self):
        model = CodeModel()
        model.load()
        result = model.generate_code("Write a hello world function")
        assert "[CODE:python]" in result

    def test_generate_code_custom_language(self):
        model = CodeModel()
        model.load()
        result = model.generate_code("hello world", language="rust")
        assert "[CODE:rust]" in result

    def test_generate_code_not_loaded(self):
        model = CodeModel()
        with pytest.raises(ModelError):
            model.generate_code("test")

    def test_generate_code_empty_prompt(self):
        model = CodeModel()
        model.load()
        with pytest.raises(ModelError):
            model.generate_code("")

    def test_complete_code(self):
        model = CodeModel()
        model.load()
        result = model.complete_code("def hello():")
        assert "[COMPLETE]" in result

    def test_explain_code(self):
        model = CodeModel()
        model.load()
        result = model.explain_code("def add(a, b): return a + b")
        assert "[EXPLAIN]" in result

    def test_explain_code_empty(self):
        model = CodeModel()
        model.load()
        with pytest.raises(ModelError):
            model.explain_code("")

    def test_refactor_code(self):
        model = CodeModel()
        model.load()
        result = model.refactor_code("x=1+2", instructions="Use better naming")
        assert "[REFACTOR]" in result

    def test_generate_tests(self):
        model = CodeModel()
        model.load()
        result = model.generate_tests("def add(a, b): return a + b")
        assert "[TEST:pytest]" in result

    def test_generate_tests_custom_framework(self):
        model = CodeModel()
        model.load()
        result = model.generate_tests("def add(a, b): return a + b", framework="unittest")
        assert "[TEST:unittest]" in result

    def test_detect_language_python(self):
        model = CodeModel()
        assert model.detect_language("def hello(): pass") == "python"
        assert model.detect_language("import os") == "python"

    def test_detect_language_javascript(self):
        model = CodeModel()
        assert model.detect_language("function hello() {}") == "javascript"

    def test_detect_language_unknown(self):
        model = CodeModel()
        assert model.detect_language("blah blah") == "unknown"

    def test_get_info(self):
        model = CodeModel()
        info = model.get_info()
        assert info["type"] == "code"
        assert info["language"] == "python"
