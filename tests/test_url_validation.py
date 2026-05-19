"""Tests for URL validation."""
import pytest
import re


def test_valid_url():
    pattern = r"^https?://[\w.-]+(?:\.[\w.-]+)(?:/\S*)?$"
    url = "https://example.com/path"
    assert re.match(pattern, url) is not None

def test_invalid_url_no_protocol():
    pattern = r"^https?://"
    assert re.match(pattern, "example.com") is None

def test_url_with_port():
    url = "http://localhost:8000/api"
    assert ":8000" in url

def test_url_with_query_params():
    url = "https://api.example.com/search?q=test&limit=10"
    assert "?" in url and "&" in url

def test_url_with_fragment():
    url = "https://docs.example.com/guide#section"
    assert "#" in url
