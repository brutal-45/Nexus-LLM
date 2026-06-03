"""Tests for the helpers module."""

import pytest

from nexus_llm.core.exceptions import ModelNotFoundError
from nexus_llm.utils.helpers import (
    format_bytes,
    format_time,
    truncate_text,
    count_words,
    validate_model_name,
)


# ---------------------------------------------------------------------------
# format_bytes
# ---------------------------------------------------------------------------

class TestFormatBytes:
    """Tests for the format_bytes function."""

    def test_zero_bytes(self):
        assert format_bytes(0) == "0 B"

    def test_bytes(self):
        assert format_bytes(512) == "512 B"

    def test_kibibytes(self):
        result = format_bytes(1024)
        assert "KiB" in result

    def test_mebibytes(self):
        result = format_bytes(1024 * 1024)
        assert "MiB" in result

    def test_gibibytes(self):
        result = format_bytes(1024 ** 3)
        assert "GiB" in result

    def test_tebibytes(self):
        result = format_bytes(1024 ** 4)
        assert "TiB" in result

    def test_pebibytes(self):
        result = format_bytes(1024 ** 5)
        assert "PiB" in result

    def test_exbibytes(self):
        result = format_bytes(1024 ** 6)
        assert "EiB" in result

    def test_exact_kibibyte(self):
        result = format_bytes(1024)
        assert result == "1.00 KiB"

    def test_exact_mebibyte(self):
        result = format_bytes(1024 ** 2)
        assert result == "1.00 MiB"

    def test_exact_gibibyte(self):
        result = format_bytes(1024 ** 3)
        assert result == "1.00 GiB"

    def test_fractional_value(self):
        result = format_bytes(1536)  # 1.5 KiB
        assert "KiB" in result
        assert "1.50" in result

    def test_large_value(self):
        result = format_bytes(5 * 1024 ** 3)  # 5 GiB
        assert "GiB" in result
        assert "5.00" in result


# ---------------------------------------------------------------------------
# format_time
# ---------------------------------------------------------------------------

class TestFormatTime:
    """Tests for the format_time function."""

    def test_negative_seconds(self):
        assert format_time(-1) == "—"

    def test_zero_seconds(self):
        result = format_time(0)
        assert "0.0s" in result

    def test_fractional_seconds(self):
        result = format_time(1.5)
        assert "1.5s" in result

    def test_under_sixty_seconds(self):
        result = format_time(45.2)
        assert "45.2s" in result

    def test_exactly_sixty_seconds(self):
        result = format_time(60)
        assert "1m" in result

    def test_minutes_and_seconds(self):
        result = format_time(125)  # 2m 5s
        assert "2m" in result
        assert "5s" in result

    def test_hours(self):
        result = format_time(3600)  # 1h
        assert "1h" in result

    def test_hours_minutes_seconds(self):
        result = format_time(3750)  # 1h 2m 30s
        assert "1h" in result
        assert "2m" in result
        assert "30s" in result

    def test_days(self):
        result = format_time(86400)  # 1d
        assert "1d" in result

    def test_days_skips_seconds(self):
        """When days are shown, seconds are omitted."""
        result = format_time(86400 + 3600 + 60 + 5)  # 1d 1h 1m 5s
        assert "1d" in result
        assert "1h" in result
        assert "1m" in result
        # Seconds should be skipped when showing days
        assert "5s" not in result

    def test_complex_duration(self):
        result = format_time(90061)  # 1d 1h 1m 1s
        assert "1d" in result
        assert "1h" in result
        assert "1m" in result


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------

class TestTruncateText:
    """Tests for the truncate_text function."""

    def test_short_text_not_truncated(self):
        text = "Hello"
        assert truncate_text(text, max_len=100) == text

    def test_exact_length_not_truncated(self):
        text = "a" * 100
        assert truncate_text(text, max_len=100) == text

    def test_long_text_truncated(self):
        text = "a" * 200
        result = truncate_text(text, max_len=100)
        assert len(result) == 100
        assert result.endswith("…")

    def test_custom_suffix(self):
        text = "a" * 200
        result = truncate_text(text, max_len=100, suffix="...")
        assert result.endswith("...")
        assert len(result) == 100

    def test_default_max_len(self):
        text = "a" * 150
        result = truncate_text(text)
        assert len(result) == 100
        assert result.endswith("…")

    def test_empty_string(self):
        assert truncate_text("", max_len=10) == ""

    def test_max_len_equals_suffix_len(self):
        text = "Hello World"
        result = truncate_text(text, max_len=1, suffix="…")
        assert result == "…"


# ---------------------------------------------------------------------------
# count_words
# ---------------------------------------------------------------------------

class TestCountWords:
    """Tests for the count_words function."""

    def test_simple_text(self):
        assert count_words("hello world") == 2

    def test_single_word(self):
        assert count_words("hello") == 1

    def test_empty_string(self):
        assert count_words("") == 0

    def test_extra_whitespace(self):
        assert count_words("hello   world   foo") == 3

    def test_with_punctuation(self):
        assert count_words("Hello, world! How are you?") == 5

    def test_with_markdown(self):
        # Markdown symbols are stripped before counting
        result = count_words("**bold** and _italic_")
        assert result == 3  # "bold", "and", "italic"

    def test_with_code_fences(self):
        result = count_words("```python\nprint('hi')\n```")
        # Backticks are stripped, then words are counted
        assert result >= 2  # at least "python" and "print"

    def test_with_headers(self):
        result = count_words("# Header ## Subheader")
        # # symbols are stripped
        assert result == 2


# ---------------------------------------------------------------------------
# validate_model_name
# ---------------------------------------------------------------------------

class TestValidateModelName:
    """Tests for the validate_model_name function."""

    def test_valid_short_id(self):
        info = validate_model_name("gpt2-medium")
        assert info.id == "gpt2-medium"

    def test_valid_hf_id(self):
        info = validate_model_name("openai-community/gpt2-medium")
        assert info.id == "gpt2-medium"

    def test_valid_phi2(self):
        info = validate_model_name("phi-2")
        assert info.id == "phi-2"
        assert info.category == "phi"

    def test_invalid_name_raises(self):
        with pytest.raises(ModelNotFoundError):
            validate_model_name("nonexistent-model-xyz")

    def test_fuzzy_match_unique(self):
        """A partial match that resolves to exactly one model."""
        info = validate_model_name("dialogpt-small")
        assert info.id == "dialogpt-small"

    def test_case_insensitive_match(self):
        """The fuzzy matcher lowercases both sides."""
        # "GPT2" should partially match models containing "gpt2"
        # But since multiple models contain "gpt2", it should raise
        with pytest.raises(ModelNotFoundError):
            validate_model_name("GPT2")

    def test_error_message_includes_available(self):
        with pytest.raises(ModelNotFoundError, match="Available models"):
            validate_model_name("zzz-nonexistent")
