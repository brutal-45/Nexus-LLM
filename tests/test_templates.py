"""Tests for template loading and rendering."""
import pytest

from nexus_llm.templates import TemplateLoader, list_prompt_templates, list_system_prompts


class TestTemplateLoader:
    """Test TemplateLoader."""

    @pytest.fixture
    def loader(self):
        return TemplateLoader()

    def test_list_prompt_templates(self, loader):
        templates = loader.list_prompt_templates()
        assert isinstance(templates, list)
        # Should find at least the default templates
        assert len(templates) > 0

    def test_list_system_prompts(self, loader):
        prompts = loader.list_system_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_load_prompt_template(self, loader):
        templates = loader.list_prompt_templates()
        if templates:
            template = loader.load_prompt_template(templates[0])
            assert isinstance(template, dict)

    def test_load_system_prompt(self, loader):
        prompts = loader.list_system_prompts()
        if prompts:
            prompt = loader.load_system_prompt(prompts[0])
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_load_nonexistent_prompt_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_prompt_template("nonexistent_template_xyz")

    def test_load_nonexistent_system_prompt_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_system_prompt("nonexistent_prompt_xyz")

    def test_prompt_template_caching(self, loader):
        templates = loader.list_prompt_templates()
        if templates:
            t1 = loader.load_prompt_template(templates[0])
            t2 = loader.load_prompt_template(templates[0])
            assert t1 is t2  # Same object due to caching

    def test_system_prompt_caching(self, loader):
        prompts = loader.list_system_prompts()
        if prompts:
            p1 = loader.load_system_prompt(prompts[0])
            p2 = loader.load_system_prompt(prompts[0])
            assert p1 is p2  # Same object due to caching

    def test_clear_cache(self, loader):
        loader.clear_cache()
        assert len(loader._prompt_cache) == 0
        assert len(loader._system_cache) == 0

    def test_format_prompt(self, loader):
        templates = loader.list_prompt_templates()
        if templates:
            result = loader.format_prompt(templates[0])
            assert isinstance(result, dict)

    def test_build_messages(self, loader):
        templates = loader.list_prompt_templates()
        if templates:
            messages = loader.build_messages(templates[0], user_input="Hello")
            assert isinstance(messages, list)
            assert len(messages) >= 1
            # At least a user message should be present
            roles = [m["role"] for m in messages]
            assert "user" in roles

    def test_build_messages_with_system_override(self, loader):
        templates = loader.list_prompt_templates()
        if templates:
            messages = loader.build_messages(
                templates[0],
                user_input="Test",
                system_override="Custom system prompt",
            )
            roles = [m["role"] for m in messages]
            assert "system" in roles


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_list_prompt_templates_function(self):
        result = list_prompt_templates()
        assert isinstance(result, list)

    def test_list_system_prompts_function(self):
        result = list_system_prompts()
        assert isinstance(result, list)


class TestTemplateLoaderCustomDir:
    """Test TemplateLoader with custom directory."""

    def test_nonexistent_dir(self, tmp_dir):
        loader = TemplateLoader(templates_dir=str(tmp_dir / "nonexistent"))
        assert loader.list_prompt_templates() == []
        assert loader.list_system_prompts() == []
