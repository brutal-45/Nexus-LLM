"""Shared pytest fixtures and configuration for Nexus-LLM tests."""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    logger = MagicMock()
    return logger


@pytest.fixture
def sample_config_dict():
    """Provide a sample configuration dictionary."""
    return {
        "model": {
            "name": "test-model",
            "type": "causal_lm",
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "cors_origins": ["*"],
        },
        "safety": {
            "enabled": True,
            "content_filter": True,
            "max_toxicity": 0.5,
        },
        "rag": {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "top_k": 5,
        },
    }


@pytest.fixture
def sample_text():
    """Provide sample text for processing tests."""
    return (
        "The quick brown fox jumps over the lazy dog. "
        "This is a sample text for testing various NLP utilities. "
        "It contains multiple sentences with different punctuation! "
        "Some numbers like 42 and 3.14 are also present. "
        "Special characters: @#$%^&*() are included too."
    )


@pytest.fixture
def sample_documents():
    """Provide sample documents for RAG tests."""
    return [
        {"id": "doc1", "text": "Python is a programming language.", "metadata": {"source": "wiki"}},
        {"id": "doc2", "text": "Machine learning uses algorithms.", "metadata": {"source": "book"}},
        {"id": "doc3", "text": "Deep learning is a subset of ML.", "metadata": {"source": "paper"}},
        {"id": "doc4", "text": "Neural networks are inspired by the brain.", "metadata": {"source": "wiki"}},
        {"id": "doc5", "text": "Transformers revolutionized NLP.", "metadata": {"source": "paper"}},
    ]


@pytest.fixture
def sample_embeddings():
    """Provide sample embedding vectors for tests."""
    import random
    random.seed(42)
    dim = 128
    return {
        f"vec_{i}": [random.gauss(0, 1) for _ in range(dim)]
        for i in range(10)
    }


@pytest.fixture
def sample_messages():
    """Provide sample chat messages for agent/model tests."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a field of AI."},
        {"role": "user", "content": "Tell me more about deep learning."},
    ]


@pytest.fixture
def sample_tool_definitions():
    """Provide sample tool definitions for agent tests."""
    return [
        {
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
        },
    ]
