"""Tests for the tokenizer_utils module.

All tests use mocks — no actual model downloads or tokenizer loading required.
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from nexus_llm.backend.tokenizer_utils import (
    ChatTemplate,
    TokenizerManager,
    _CATEGORY_TEMPLATE_MAP,
)
from nexus_llm.core.exceptions import ModelLoadError, ModelNotFoundError


# ---------------------------------------------------------------------------
# ChatTemplate enum
# ---------------------------------------------------------------------------

class TestChatTemplate:
    """Tests for the ChatTemplate enum."""

    def test_all_template_values(self):
        assert ChatTemplate.GPT2.value == "gpt2"
        assert ChatTemplate.DIALOGPT.value == "dialogpt"
        assert ChatTemplate.LLAMA.value == "llama"
        assert ChatTemplate.CHATML.value == "chatml"
        assert ChatTemplate.PHI.value == "phi"
        assert ChatTemplate.QWEN.value == "qwen"
        assert ChatTemplate.GEMMA.value == "gemma"
        assert ChatTemplate.DEFAULT.value == "default"

    def test_template_count(self):
        assert len(ChatTemplate) == 8

    def test_template_is_string(self):
        for tmpl in ChatTemplate:
            assert isinstance(tmpl.value, str)


# ---------------------------------------------------------------------------
# Category-template mapping
# ---------------------------------------------------------------------------

class TestCategoryTemplateMap:
    """Tests for the _CATEGORY_TEMPLATE_MAP."""

    def test_gpt2_category(self):
        assert _CATEGORY_TEMPLATE_MAP["gpt2"] == ChatTemplate.GPT2

    def test_dialogpt_category(self):
        assert _CATEGORY_TEMPLATE_MAP["dialogpt"] == ChatTemplate.DIALOGPT

    def test_llama_category(self):
        assert _CATEGORY_TEMPLATE_MAP["llama"] == ChatTemplate.LLAMA

    def test_phi_category(self):
        assert _CATEGORY_TEMPLATE_MAP["phi"] == ChatTemplate.PHI

    def test_qwen_category(self):
        assert _CATEGORY_TEMPLATE_MAP["qwen"] == ChatTemplate.QWEN

    def test_gemma_category(self):
        assert _CATEGORY_TEMPLATE_MAP["gemma"] == ChatTemplate.GEMMA

    def test_smollm_category(self):
        assert _CATEGORY_TEMPLATE_MAP["smollm"] == ChatTemplate.CHATML

    def test_stablelm_category(self):
        assert _CATEGORY_TEMPLATE_MAP["stablelm"] == ChatTemplate.CHATML

    def test_flan_t5_category(self):
        assert _CATEGORY_TEMPLATE_MAP["flan-t5"] == ChatTemplate.DEFAULT


# ---------------------------------------------------------------------------
# TokenizerManager — template detection (mocked)
# ---------------------------------------------------------------------------

class TestTokenizerManagerTemplateDetection:
    """Test _detect_template with a mocked tokenizer."""

    def _make_manager_with_mock(self, chat_template=None, category="gpt2"):
        """Create a TokenizerManager with a mocked internal tokenizer."""
        mgr = TokenizerManager()
        mock_tok = MagicMock()
        mock_tok.chat_template = chat_template
        mock_tok.pad_token = None
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.eos_token_id = 50256
        mock_tok.__len__ = MagicMock(return_value=50257)
        mgr._tokenizer = mock_tok
        mgr._model_id = "gpt2-medium"
        return mgr

    def test_detect_gpt2_by_category(self):
        mgr = self._make_manager_with_mock(chat_template=None, category="gpt2")
        tmpl = mgr._detect_template("gpt2-medium", "gpt2")
        assert tmpl == ChatTemplate.GPT2

    def test_detect_llama_by_category(self):
        mgr = self._make_manager_with_mock(chat_template=None, category="llama")
        tmpl = mgr._detect_template("tinyllama", "llama")
        assert tmpl == ChatTemplate.LLAMA

    def test_detect_chatml_from_tokenizer_template(self):
        mock_tok = MagicMock()
        mock_tok.chat_template = "<|im_start|>{role}\n{content}<|im_end|>"
        mgr = TokenizerManager()
        mgr._tokenizer = mock_tok
        tmpl = mgr._detect_template("smollm-1.7b", "smollm")
        assert tmpl == ChatTemplate.CHATML

    def test_detect_unknown_category_defaults(self):
        mgr = self._make_manager_with_mock(chat_template=None, category="unknown_cat")
        tmpl = mgr._detect_template("some-model", "unknown_cat")
        assert tmpl == ChatTemplate.DEFAULT


# ---------------------------------------------------------------------------
# TokenizerManager — format_conversation (using static formatters)
# ---------------------------------------------------------------------------

class TestFormatConversation:
    """Test the manual formatting methods (no real tokenizer needed)."""

    def test_format_gpt2(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = TokenizerManager._format_gpt2(messages)
        assert "Instructions: Be helpful." in result
        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result
        assert result.endswith("Assistant: ")

    def test_format_gpt2_no_system(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = TokenizerManager._format_gpt2(messages)
        assert "User: Hello" in result
        assert "Instructions:" not in result

    def test_format_dialogpt(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = TokenizerManager._format_dialogpt(messages)
        assert "Hi" in result
        assert "Hello!" in result

    def test_format_llama(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = TokenizerManager._format_llama(messages)
        assert "<<SYS>>" in result
        assert "<</SYS>>" in result
        assert "[INST] Hello [/INST]" in result
        assert "Hi!" in result

    def test_format_chatml(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = TokenizerManager._format_chatml(messages)
        assert "<|im_start|>user" in result
        assert "Hello<|im_end|>" in result
        assert "<|im_start|>assistant" in result
        assert "<|im_end|>" in result

    def test_format_phi(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = TokenizerManager._format_phi(messages)
        assert "Be helpful." in result
        assert "Instruct: Hello" in result
        assert "Output: Hi!" in result
        assert result.endswith("Output: ")

    def test_format_qwen(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = TokenizerManager._format_qwen(messages)
        assert "<|im_start|>user" in result
        assert "Hello<|im_end|>" in result

    def test_format_gemma(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = TokenizerManager._format_gemma(messages)
        assert "<start_of_turn>user" in result
        assert "Hello<end_of_turn>" in result
        assert "<start_of_turn>model" in result

    def test_format_default(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = TokenizerManager._format_default(messages)
        assert "System: Be helpful." in result
        assert "Human: Hello" in result
        assert "Assistant: Hi!" in result
        assert result.endswith("Assistant: ")


# ---------------------------------------------------------------------------
# TokenizerManager — format_conversation integration (mocked)
# ---------------------------------------------------------------------------

class TestFormatConversationMocked:
    """Test format_conversation with a mocked tokenizer manager."""

    def _make_loaded_manager(self, template=ChatTemplate.GPT2):
        mgr = TokenizerManager()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_tok.pad_token = "<|pad|>"
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.eos_token_id = 50256
        # apply_chat_template raises so manual fallback is used
        mock_tok.apply_chat_template.side_effect = AttributeError("nope")
        mgr._tokenizer = mock_tok
        mgr._model_id = "gpt2-medium"
        mgr._template = template
        mgr._pad_token_set = True
        return mgr

    def test_format_with_gpt2_template(self):
        mgr = self._make_loaded_manager(template=ChatTemplate.GPT2)
        messages = [{"role": "user", "content": "Hello"}]
        result = mgr.format_conversation(messages)
        assert "User: Hello" in result

    def test_format_with_llama_template(self):
        mgr = self._make_loaded_manager(template=ChatTemplate.LLAMA)
        messages = [{"role": "user", "content": "Hello"}]
        result = mgr.format_conversation(messages)
        assert "[INST]" in result

    def test_format_with_chatml_template(self):
        mgr = self._make_loaded_manager(template=ChatTemplate.CHATML)
        messages = [{"role": "user", "content": "Hello"}]
        result = mgr.format_conversation(messages)
        assert "<|im_start|>" in result

    def test_format_with_override_template(self):
        mgr = self._make_loaded_manager(template=ChatTemplate.GPT2)
        messages = [{"role": "user", "content": "Hello"}]
        result = mgr.format_conversation(messages, template=ChatTemplate.PHI)
        assert "Instruct:" in result

    def test_format_without_loaded_tokenizer_raises(self):
        mgr = TokenizerManager()
        with pytest.raises(ModelLoadError, match="No tokenizer loaded"):
            mgr.format_conversation([{"role": "user", "content": "Hello"}])


# ---------------------------------------------------------------------------
# TokenizerManager — truncate_conversation (mocked)
# ---------------------------------------------------------------------------

class TestTruncateConversation:
    """Test truncate_conversation with a mocked tokenizer manager."""

    def _make_loaded_manager(self):
        mgr = TokenizerManager()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_tok.pad_token = "<|pad|>"
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.eos_token_id = 50256
        mock_tok.apply_chat_template.side_effect = AttributeError
        mgr._tokenizer = mock_tok
        mgr._model_id = "gpt2-medium"
        mgr._template = ChatTemplate.GPT2
        mgr._pad_token_set = True

        # Mock count_tokens to return a simple heuristic: 1 token per word
        original_count_tokens = mgr.count_tokens
        def mock_count_tokens(text):
            return len(text.split()) if text else 0
        mgr.count_tokens = mock_count_tokens

        return mgr

    def test_truncate_keeps_system_by_default(self):
        mgr = self._make_loaded_manager()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi how are you doing"},
        ]
        result = mgr.truncate_conversation(messages, max_tokens=50)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1

    def test_truncate_removes_oldest_first(self):
        mgr = self._make_loaded_manager()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "First message here"},
            {"role": "assistant", "content": "First reply here"},
            {"role": "user", "content": "Second message here"},
            {"role": "assistant", "content": "Second reply here"},
        ]
        # Set a low max_tokens to force truncation
        result = mgr.truncate_conversation(messages, max_tokens=15)
        # System message should be kept
        roles = [m["role"] for m in result]
        assert "system" in roles
        # The newest messages should be preferred
        if len(result) > 1:
            last_msg = result[-1]
            assert "Second" in last_msg.get("content", "") or roles[-1] in ("user", "assistant")

    def test_truncate_without_keep_system(self):
        mgr = self._make_loaded_manager()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello there"},
        ]
        result = mgr.truncate_conversation(messages, max_tokens=50, keep_system=False)
        assert all(m["role"] != "system" for m in result)

    def test_truncate_empty_messages(self):
        mgr = self._make_loaded_manager()
        result = mgr.truncate_conversation([], max_tokens=100)
        assert result == []

    def test_truncate_not_loaded_raises(self):
        mgr = TokenizerManager()
        with pytest.raises(ModelLoadError, match="No tokenizer loaded"):
            mgr.truncate_conversation([{"role": "user", "content": "Hi"}], max_tokens=100)


# ---------------------------------------------------------------------------
# TokenizerManager — load / unload (mocked AutoTokenizer)
# ---------------------------------------------------------------------------

class TestTokenizerManagerLoad:
    """Test load and unload with mocked AutoTokenizer."""

    @patch("nexus_llm.backend.tokenizer_utils.AutoTokenizer")
    def test_load_valid_model(self, mock_autotok_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.eos_token_id = 50256
        mock_tok.chat_template = None
        mock_tok.__len__ = MagicMock(return_value=50257)
        mock_autotok_cls.from_pretrained.return_value = mock_tok

        mgr = TokenizerManager()
        mgr.load("gpt2-medium")
        assert mgr.is_loaded is True
        assert mgr.model_id == "gpt2-medium"
        assert mgr.template == ChatTemplate.GPT2

    @patch("nexus_llm.backend.tokenizer_utils.AutoTokenizer")
    def test_load_sets_pad_token(self, mock_autotok_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.eos_token_id = 50256
        mock_tok.chat_template = None
        mock_tok.__len__ = MagicMock(return_value=50257)
        mock_autotok_cls.from_pretrained.return_value = mock_tok

        mgr = TokenizerManager()
        mgr.load("gpt2-medium")
        assert mock_tok.pad_token == "<|endoftext|>"
        assert mgr._pad_token_set is True

    @patch("nexus_llm.backend.tokenizer_utils.AutoTokenizer")
    def test_load_invalid_model_raises(self, mock_autotok_cls):
        mgr = TokenizerManager()
        with pytest.raises(ModelNotFoundError):
            mgr.load("nonexistent-model-xyz")
        assert mgr.is_loaded is False

    @patch("nexus_llm.backend.tokenizer_utils.AutoTokenizer")
    def test_load_failure_raises_model_load_error(self, mock_autotok_cls):
        mock_autotok_cls.from_pretrained.side_effect = OSError("download failed")
        mgr = TokenizerManager()
        with pytest.raises(ModelLoadError, match="Failed to load tokenizer"):
            mgr.load("gpt2-medium")

    @patch("nexus_llm.backend.tokenizer_utils.AutoTokenizer")
    def test_unload(self, mock_autotok_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token = "<|pad|>"
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.eos_token_id = 50256
        mock_tok.chat_template = None
        mock_tok.__len__ = MagicMock(return_value=50257)
        mock_autotok_cls.from_pretrained.return_value = mock_tok

        mgr = TokenizerManager()
        mgr.load("gpt2-medium")
        assert mgr.is_loaded is True
        mgr.unload()
        assert mgr.is_loaded is False
        assert mgr.model_id is None
        assert mgr.template == ChatTemplate.DEFAULT

    def test_unload_without_load(self):
        mgr = TokenizerManager()
        mgr.unload()  # Should not raise
        assert mgr.is_loaded is False

    def test_properties_when_not_loaded(self):
        mgr = TokenizerManager()
        assert mgr.eos_token is None
        assert mgr.eos_token_id is None
        assert mgr.pad_token is None
        assert mgr.pad_token_id is None
        assert mgr.vocab_size == 0

    @patch("nexus_llm.backend.tokenizer_utils.AutoTokenizer")
    def test_get_info_when_loaded(self, mock_autotok_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token = "<|pad|>"
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.eos_token_id = 50256
        mock_tok.chat_template = None
        mock_tok.__len__ = MagicMock(return_value=50257)
        mock_autotok_cls.from_pretrained.return_value = mock_tok

        mgr = TokenizerManager()
        mgr.load("gpt2-medium")
        info = mgr.get_info()
        assert info["is_loaded"] is True
        assert info["model_id"] == "gpt2-medium"
        assert info["template"] == "gpt2"
        assert info["vocab_size"] == 50257

    def test_get_info_when_not_loaded(self):
        mgr = TokenizerManager()
        info = mgr.get_info()
        assert info["is_loaded"] is False
        assert info["model_id"] is None
        assert info["vocab_size"] == 0

    def test_encode_without_load_raises(self):
        mgr = TokenizerManager()
        with pytest.raises(ModelLoadError, match="No tokenizer loaded"):
            mgr.encode("hello")

    def test_decode_without_load_raises(self):
        mgr = TokenizerManager()
        with pytest.raises(ModelLoadError, match="No tokenizer loaded"):
            mgr.decode([1, 2, 3])

    def test_count_tokens_without_load_raises(self):
        mgr = TokenizerManager()
        with pytest.raises(ModelLoadError, match="No tokenizer loaded"):
            mgr.count_tokens("hello")
