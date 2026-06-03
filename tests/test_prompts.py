"""Tests for the prompts module.

Covers PromptManager, PromptTemplate, PromptLibrary, and PromptOptimizer.
"""

from __future__ import annotations

import pytest

from nexus_llm.prompts.manager import PromptManager
from nexus_llm.prompts.template import PromptTemplate
from nexus_llm.prompts.library import PromptLibrary
from nexus_llm.prompts.optimizer import PromptOptimizer


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------

class TestPromptTemplate:
    """Tests for PromptTemplate."""

    def test_create_template(self):
        tmpl = PromptTemplate(name="greeting", template="Hello, {name}!")
        assert tmpl.name == "greeting"
        assert tmpl.template == "Hello, {name}!"

    def test_render(self):
        tmpl = PromptTemplate(name="greeting", template="Hello, {name}!")
        result = tmpl.render(name="World")
        assert result == "Hello, World!"

    def test_render_missing_variable(self):
        tmpl = PromptTemplate(name="test", template="Hello, {name}!")
        # Missing variable should either keep placeholder or raise
        result = tmpl.render()
        assert isinstance(result, str)

    def test_render_with_defaults(self):
        tmpl = PromptTemplate(
            name="test",
            template="Hello, {name}! Welcome to {place}.",
            defaults={"place": "Nexus"},
        )
        result = tmpl.render(name="World")
        assert "Nexus" in result

    def test_variables(self):
        tmpl = PromptTemplate(name="test", template="{a} and {b} and {c}")
        vars_ = tmpl.variables()
        assert "a" in vars_
        assert "b" in vars_
        assert "c" in vars_

    def test_to_dict(self):
        tmpl = PromptTemplate(name="test", template="Hello {name}")
        d = tmpl.to_dict()
        assert d["name"] == "test"

    def test_from_dict(self):
        data = {"name": "test", "template": "Hello {name}", "defaults": {}}
        tmpl = PromptTemplate.from_dict(data)
        assert tmpl.name == "test"


# ---------------------------------------------------------------------------
# PromptLibrary
# ---------------------------------------------------------------------------

class TestPromptLibrary:
    """Tests for PromptLibrary."""

    def test_add_and_get(self):
        lib = PromptLibrary()
        tmpl = PromptTemplate(name="greeting", template="Hello, {name}!")
        lib.add(tmpl)
        retrieved = lib.get("greeting")
        assert retrieved is not None
        assert retrieved.name == "greeting"

    def test_get_nonexistent(self):
        lib = PromptLibrary()
        assert lib.get("nonexistent") is None

    def test_list_templates(self):
        lib = PromptLibrary()
        lib.add(PromptTemplate(name="t1", template="A"))
        lib.add(PromptTemplate(name="t2", template="B"))
        templates = lib.list_templates()
        assert len(templates) == 2

    def test_remove(self):
        lib = PromptLibrary()
        lib.add(PromptTemplate(name="t1", template="A"))
        lib.remove("t1")
        assert lib.get("t1") is None

    def test_search(self):
        lib = PromptLibrary()
        lib.add(PromptTemplate(name="greeting", template="Hello {name}"))
        lib.add(PromptTemplate(name="farewell", template="Goodbye {name}"))
        results = lib.search("greet")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------

class TestPromptManager:
    """Tests for PromptManager."""

    def test_init(self):
        pm = PromptManager()
        assert pm is not None

    def test_get_template(self):
        pm = PromptManager()
        tmpl = pm.get_template("default")
        # May return None if not loaded, but should not crash
        assert tmpl is None or isinstance(tmpl, PromptTemplate)

    def test_render(self):
        pm = PromptManager()
        result = pm.render("Hello {name}", name="World")
        assert result == "Hello World"

    def test_register_template(self):
        pm = PromptManager()
        tmpl = PromptTemplate(name="custom", template="Hi {name}")
        pm.register_template(tmpl)
        retrieved = pm.get_template("custom")
        assert retrieved is not None

    def test_list_templates(self):
        pm = PromptManager()
        templates = pm.list_templates()
        assert isinstance(templates, list)


# ---------------------------------------------------------------------------
# PromptOptimizer
# ---------------------------------------------------------------------------

class TestPromptOptimizer:
    """Tests for PromptOptimizer."""

    def test_create_optimizer(self):
        opt = PromptOptimizer()
        assert opt is not None

    def test_optimize(self):
        opt = PromptOptimizer()
        result = opt.optimize("Write a story about a cat")
        assert isinstance(result, str)

    def test_suggest_improvements(self):
        opt = PromptOptimizer()
        suggestions = opt.suggest_improvements("Tell me about AI")
        assert isinstance(suggestions, list)
