"""Tests for email validation."""
import pytest
import re


def test_valid_email():
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    assert re.match(pattern, "user@example.com") is not None

def test_invalid_email_no_at():
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    assert re.match(pattern, "userexample.com") is None

def test_invalid_email_no_domain():
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    assert re.match(pattern, "user@") is None

def test_email_with_subdomain():
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    assert re.match(pattern, "user@mail.example.com") is not None

def test_email_with_plus():
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    assert re.match(pattern, "user+tag@example.com") is not None
