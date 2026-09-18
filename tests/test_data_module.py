"""Tests for data module __init__ imports."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_data_module_importable():
    """Test that the data module can be imported."""
    import nexus_llm.data as data_module
    assert data_module is not None


def test_data_module_has_loader():
    """Test that data module exposes loader."""
    from nexus_llm.data import loader
    assert loader is not None


def test_data_module_has_processor():
    """Test that data module exposes processor."""
    from nexus_llm.data import processor
    assert processor is not None


def test_data_module_has_validator():
    """Test that data module exposes validator."""
    from nexus_llm.data import validator
    assert validator is not None


def test_data_module_has_converter():
    """Test that data module exposes converter."""
    from nexus_llm.data import converter
    assert converter is not None
