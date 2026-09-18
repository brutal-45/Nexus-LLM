"""Tests for preprocessing."""
import pytest
import re


class TextPreprocessor:
    """Simple text preprocessor for testing."""
    def normalize_whitespace(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def remove_urls(self, text):
        return re.sub(r"https?://\S+", "", text)

    def remove_html_tags(self, text):
        return re.sub(r"<[^>]+>", "", text)

    def lowercase(self, text):
        return text.lower()


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_normalize_whitespace(preprocessor):
    """Test whitespace normalization."""
    text = "Hello   World\n\tTest"
    result = preprocessor.normalize_whitespace(text)
    assert result == "Hello World Test"


def test_remove_urls(preprocessor):
    """Test URL removal."""
    text = "Visit https://example.com for more info"
    result = preprocessor.remove_urls(text)
    assert "https://" not in result
    assert "more info" in result


def test_remove_html_tags(preprocessor):
    """Test HTML tag removal."""
    text = "<p>Hello <b>World</b></p>"
    result = preprocessor.remove_html_tags(text)
    assert result == "Hello World"


def test_lowercase(preprocessor):
    """Test lowercase conversion."""
    assert preprocessor.lowercase("Hello WORLD") == "hello world"
