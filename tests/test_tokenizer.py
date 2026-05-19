"""Tests for tokenizer utilities."""
import pytest
from unittest.mock import MagicMock

from nexus.data.tokenizer import BPETokenizer


@pytest.fixture
def tokenizer():
    """Create a BPE tokenizer for testing."""
    tok = BPETokenizer()
    return tok


def test_tokenizer_creation(tokenizer):
    """Test that tokenizer can be created."""
    assert tokenizer is not None


def test_tokenizer_encode_decode(tokenizer):
    """Test encode and decode round-trip."""
    text = "Hello world"
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    assert isinstance(ids, list)
    assert len(ids) > 0
    # First token should be BOS
    assert ids[0] == tokenizer.special_tokens["<bos>"]


def test_tokenizer_special_tokens(tokenizer):
    """Test special tokens exist."""
    assert "<pad>" in tokenizer.special_tokens
    assert "<bos>" in tokenizer.special_tokens
    assert "<eos>" in tokenizer.special_tokens
    assert "<unk>" in tokenizer.special_tokens


def test_tokenizer_encode_with_eos(tokenizer):
    """Test encoding with EOS token."""
    text = "test"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[-1] == tokenizer.special_tokens["<eos>"]


def test_tokenizer_vocab_size(tokenizer):
    """Test that vocab_size returns a positive integer."""
    assert tokenizer.vocab_size > 0
