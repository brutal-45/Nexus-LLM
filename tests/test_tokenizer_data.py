"""Test tokenizer data for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter


@dataclass
class TokenizerConfig:
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: List[str] = field(default_factory=lambda: ["<pad>", "<unk>", "<bos>", "<eos>"])
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"


class SimpleTokenizer:
    def __init__(self, config: TokenizerConfig = None):
        self._config = config or TokenizerConfig()
        self._vocab: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._initialized = False

    @property
    def config(self):
        return self._config

    @property
    def vocab_size(self):
        return len(self._vocab)

    def build_vocab(self, texts: List[str]):
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        self._vocab = {}
        idx = 0
        for token in self._config.special_tokens:
            self._vocab[token] = idx
            idx += 1

        for word, count in word_counts.most_common():
            if count >= self._config.min_frequency and idx < self._config.vocab_size:
                self._vocab[word] = idx
                idx += 1

        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._initialized = True

    def encode(self, text: str) -> List[int]:
        if not self._initialized:
            raise RuntimeError("Tokenizer not initialized. Call build_vocab first.")
        words = text.lower().split()
        ids = []
        for word in words:
            ids.append(self._vocab.get(word, self._vocab.get(self._config.unk_token, 1)))
        return ids

    def decode(self, ids: List[int]) -> str:
        if not self._initialized:
            raise RuntimeError("Tokenizer not initialized.")
        tokens = []
        for id_ in ids:
            tokens.append(self._id_to_token.get(id_, self._config.unk_token))
        return " ".join(tokens)

    def token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get(self._config.unk_token, 1))

    def id_to_token(self, id_: int) -> str:
        return self._id_to_token.get(id_, self._config.unk_token)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def is_initialized(self) -> bool:
        return self._initialized


class TestTokenizerConfig:
    def test_defaults(self):
        config = TokenizerConfig()
        assert config.vocab_size == 32000
        assert config.min_frequency == 2
        assert "<pad>" in config.special_tokens

    def test_custom(self):
        config = TokenizerConfig(vocab_size=5000, min_frequency=1)
        assert config.vocab_size == 5000


class TestSimpleTokenizer:
    def test_build_vocab(self):
        tokenizer = SimpleTokenizer()
        texts = ["hello world", "hello there", "world peace"]
        tokenizer.build_vocab(texts)
        assert tokenizer.vocab_size > 0
        assert tokenizer.is_initialized()

    def test_encode(self):
        tokenizer = SimpleTokenizer(TokenizerConfig(min_frequency=1))
        tokenizer.build_vocab(["hello world", "hello there"])
        ids = tokenizer.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode(self):
        tokenizer = SimpleTokenizer(TokenizerConfig(min_frequency=1))
        tokenizer.build_vocab(["hello world"])
        ids = tokenizer.encode("hello world")
        decoded = tokenizer.decode(ids)
        assert decoded == "hello world"

    def test_encode_unknown(self):
        tokenizer = SimpleTokenizer(TokenizerConfig(min_frequency=10))
        tokenizer.build_vocab(["hello world"])
        ids = tokenizer.encode("unknown word")
        unk_id = tokenizer.token_to_id("<unk>")
        assert all(i == unk_id for i in ids)

    def test_encode_not_initialized(self):
        tokenizer = SimpleTokenizer()
        with pytest.raises(RuntimeError, match="not initialized"):
            tokenizer.encode("test")

    def test_decode_not_initialized(self):
        tokenizer = SimpleTokenizer()
        with pytest.raises(RuntimeError, match="not initialized"):
            tokenizer.decode([0])

    def test_roundtrip(self):
        tokenizer = SimpleTokenizer(TokenizerConfig(min_frequency=1))
        tokenizer.build_vocab(["the cat sat on the mat"])
        ids = tokenizer.encode("the cat sat")
        decoded = tokenizer.decode(ids)
        assert decoded == "the cat sat"

    def test_special_tokens_in_vocab(self):
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["test"])
        vocab = tokenizer.get_vocab()
        assert "<pad>" in vocab
        assert "<unk>" in vocab

    def test_vocab_size_limit(self):
        tokenizer = SimpleTokenizer(TokenizerConfig(vocab_size=10, min_frequency=1))
        tokenizer.build_vocab(["a b c d e f g h i j k l m n o p"])
        assert tokenizer.vocab_size <= 10

    def test_token_to_id(self):
        tokenizer = SimpleTokenizer(TokenizerConfig(min_frequency=1))
        tokenizer.build_vocab(["hello world"])
        id_ = tokenizer.token_to_id("hello")
        assert isinstance(id_, int)

    def test_id_to_token(self):
        tokenizer = SimpleTokenizer(TokenizerConfig(min_frequency=1))
        tokenizer.build_vocab(["hello world"])
        id_ = tokenizer.token_to_id("hello")
        assert tokenizer.id_to_token(id_) == "hello"

    def test_min_frequency(self):
        tokenizer = SimpleTokenizer(TokenizerConfig(min_frequency=3))
        tokenizer.build_vocab(["rare common common common other other other"])
        # "rare" appears once, should not be in vocab
        assert "rare" not in tokenizer.get_vocab()
