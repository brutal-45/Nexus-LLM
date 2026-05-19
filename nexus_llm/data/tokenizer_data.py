"""Nexus-LLM Tokenizer Data Builder Module.

Provides tools for building tokenizer training data, including:

- **Vocabulary construction** from raw text corpora.
- **Byte-Pair Encoding (BPE) merge rule** computation.
- **Special token** management (pad, unk, bos, eos, etc.).
- **TokenizerDataBuilder** class for end-to-end tokenizer data prep.

The implementation is pure-Python and dependency-free, making it
suitable for experimentation and small-to-medium corpora.  For
production tokenizers, the output can be fed into libraries like
``tokenizers`` or ``sentencepiece``.

Example::

    from nexus_llm.data.tokenizer_data import TokenizerDataBuilder, TokenizerConfig

    builder = TokenizerDataBuilder(TokenizerConfig(vocab_size=8000))
    builder.build_from_texts(corpus_texts)
    builder.save("my_tokenizer.json")
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TokenizerConfig:
    """Configuration for tokenizer data building.

    Attributes:
        vocab_size: Maximum vocabulary size (including special tokens).
        min_frequency: Minimum corpus frequency for a token to be included.
        special_tokens: List of special tokens added at the start of the vocab.
        unk_token: The unknown-token string.
        pad_token: The padding-token string.
        bos_token: The beginning-of-sequence token string.
        eos_token: The end-of-sequence token string.
    """

    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: List[str] = field(
        default_factory=lambda: ["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"


# ---------------------------------------------------------------------------
# SimpleTokenizer – lightweight word-level tokenizer (test-friendly)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """Minimal word-level tokenizer for vocabulary building and encoding.

    Not intended as a production tokenizer; this class supports the
    test suite and quick prototyping.  It splits on whitespace after
    lowercasing and maps words to integer IDs.

    Args:
        config: Tokenizer configuration.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self._config = config or TokenizerConfig()
        self._vocab: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._initialized: bool = False

    @property
    def config(self) -> TokenizerConfig:
        """Return the active configuration."""
        return self._config

    @property
    def vocab_size(self) -> int:
        """Current number of tokens in the vocabulary."""
        return len(self._vocab)

    # -- Vocabulary building ------------------------------------------------

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from *texts*.

        Special tokens are assigned the first IDs, then the most frequent
        words (meeting ``min_frequency``) are added up to ``vocab_size``.
        """
        word_counts: Counter = Counter()
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
        logger.info("Built vocabulary: %d tokens", len(self._vocab))

    # -- Encoding / Decoding ------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode *text* into a list of token IDs."""
        if not self._initialized:
            raise RuntimeError("Tokenizer not initialized. Call build_vocab first.")
        words = text.lower().split()
        unk_id = self._vocab.get(self._config.unk_token, 1)
        return [self._vocab.get(word, unk_id) for word in words]

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back into a string."""
        if not self._initialized:
            raise RuntimeError("Tokenizer not initialized.")
        return " ".join(
            self._id_to_token.get(id_, self._config.unk_token) for id_ in ids
        )

    # -- Lookup helpers -----------------------------------------------------

    def token_to_id(self, token: str) -> int:
        """Return the integer ID for *token*, or the UNK ID."""
        return self._vocab.get(token, self._vocab.get(self._config.unk_token, 1))

    def id_to_token(self, id_: int) -> str:
        """Return the string for *id_*, or the UNK token."""
        return self._id_to_token.get(id_, self._config.unk_token)

    def get_vocab(self) -> Dict[str, int]:
        """Return a copy of the vocabulary mapping."""
        return dict(self._vocab)

    def is_initialized(self) -> bool:
        """Whether :meth:`build_vocab` has been called."""
        return self._initialized


# ---------------------------------------------------------------------------
# BPE merge rule computation
# ---------------------------------------------------------------------------

def _get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """Return set of adjacent symbol pairs in *word*."""
    pairs: Set[Tuple[str, str]] = set()
    prev = word[0]
    for symbol in word[1:]:
        pairs.add((prev, symbol))
        prev = symbol
    return pairs


def compute_bpe_merges(
    texts: List[str],
    num_merges: int = 1000,
    min_frequency: int = 2,
) -> List[Tuple[str, str]]:
    """Compute BPE merge rules from *texts*.

    Args:
        texts: Corpus of raw strings.
        num_merges: Number of merge operations to learn.
        min_frequency: Minimum pair frequency to consider.

    Returns:
        Ordered list of merge pairs ``(left, right)``.
    """
    # Build initial word frequencies (character-level split)
    word_freqs: Dict[Tuple[str, ...], int] = Counter()
    for text in texts:
        for word in text.lower().split():
            # Each character becomes a symbol; add end-of-word marker
            symbols = tuple(word) + ("</w>",)
            word_freqs[symbols] += 1

    merges: List[Tuple[str, str]] = []

    for _ in range(num_merges):
        # Count all pairs
        pair_counts: Counter = Counter()
        for word, freq in word_freqs.items():
            pairs = _get_pairs(word)
            for pair in pairs:
                pair_counts[pair] += freq

        if not pair_counts:
            break

        best_pair = pair_counts.most_common(1)[0][0]
        if pair_counts[best_pair] < min_frequency:
            break

        merges.append(best_pair)

        # Merge the best pair in all words
        new_word_freqs: Dict[Tuple[str, ...], int] = {}
        for word, freq in word_freqs.items():
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        word_freqs = new_word_freqs

    logger.info("Computed %d BPE merge rules", len(merges))
    return merges


# ---------------------------------------------------------------------------
# TokenizerDataBuilder
# ---------------------------------------------------------------------------

class TokenizerDataBuilder:
    """End-to-end tokenizer data preparation.

    Builds a vocabulary, computes BPE merge rules, and manages special
    tokens.  The result can be serialized to JSON for later use.

    Args:
        config: Tokenizer configuration.

    Example::

        builder = TokenizerDataBuilder()
        builder.build_from_texts(["hello world", "hello there"])
        builder.save("tokenizer.json")
    """

    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self._config = config or TokenizerConfig()
        self._tokenizer = SimpleTokenizer(self._config)
        self._merges: List[Tuple[str, str]] = []
        self._built = False

    @property
    def config(self) -> TokenizerConfig:
        """Return the active configuration."""
        return self._config

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return self._tokenizer.vocab_size

    @property
    def merges(self) -> List[Tuple[str, str]]:
        """BPE merge rules (empty until :meth:`build_from_texts` is called)."""
        return list(self._merges)

    @property
    def vocab(self) -> Dict[str, int]:
        """Current vocabulary mapping."""
        return self._tokenizer.get_vocab()

    # -- Building -----------------------------------------------------------

    def build_from_texts(
        self,
        texts: List[str],
        num_merges: int = 1000,
    ) -> "TokenizerDataBuilder":
        """Build vocabulary and BPE merges from a corpus.

        Args:
            texts: List of raw text strings.
            num_merges: Number of BPE merge operations to learn.

        Returns:
            ``self`` for method chaining.
        """
        self._tokenizer.build_vocab(texts)
        self._merges = compute_bpe_merges(
            texts, num_merges=num_merges, min_frequency=self._config.min_frequency
        )
        self._built = True
        logger.info(
            "TokenizerDataBuilder: vocab_size=%d, merges=%d",
            self.vocab_size, len(self._merges),
        )
        return self

    def add_special_token(self, token: str) -> int:
        """Add a special token to the vocabulary.

        Args:
            token: Token string to add.

        Returns:
            The assigned integer ID.

        Raises:
            RuntimeError: If the tokenizer has not been built yet.
        """
        if not self._tokenizer.is_initialized():
            raise RuntimeError("Call build_from_texts() before adding special tokens.")
        if token in self._tokenizer.get_vocab():
            return self._tokenizer.token_to_id(token)
        new_id = self._tokenizer.vocab_size
        self._tokenizer._vocab[token] = new_id
        self._tokenizer._id_to_token[new_id] = token
        return new_id

    def add_special_tokens(self, tokens: List[str]) -> List[int]:
        """Add multiple special tokens.

        Returns:
            List of assigned IDs.
        """
        return [self.add_special_token(t) for t in tokens]

    # -- Encoding / Decoding ------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode *text* using the built vocabulary."""
        return self._tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back into text."""
        return self._tokenizer.decode(ids)

    def token_to_id(self, token: str) -> int:
        """Map a token string to its ID."""
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id_: int) -> str:
        """Map an ID to its token string."""
        return self._tokenizer.id_to_token(id_)

    # -- Application of BPE merges ------------------------------------------

    def apply_bpe(self, word: str) -> List[str]:
        """Apply learned BPE merges to a single word.

        Args:
            word: Input word string.

        Returns:
            List of sub-word tokens after all applicable merges.
        """
        symbols = list(word) + ["</w>"]
        for left, right in self._merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == left and symbols[i + 1] == right:
                    symbols[i : i + 2] = [left + right]
                else:
                    i += 1
        return symbols

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the tokenizer data to a dictionary."""
        return {
            "config": {
                "vocab_size": self._config.vocab_size,
                "min_frequency": self._config.min_frequency,
                "special_tokens": self._config.special_tokens,
                "unk_token": self._config.unk_token,
                "pad_token": self._config.pad_token,
                "bos_token": self._config.bos_token,
                "eos_token": self._config.eos_token,
            },
            "vocab": self._tokenizer.get_vocab(),
            "merges": [[left, right] for left, right in self._merges],
        }

    def save(self, path: str) -> None:
        """Save tokenizer data to a JSON file.

        Args:
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Saved tokenizer data to %s", path)

    @classmethod
    def load(cls, path: str) -> "TokenizerDataBuilder":
        """Load tokenizer data from a JSON file.

        Args:
            path: Path to a previously saved tokenizer JSON.

        Returns:
            A ``TokenizerDataBuilder`` with restored state.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg_data = data.get("config", {})
        config = TokenizerConfig(
            vocab_size=cfg_data.get("vocab_size", 32000),
            min_frequency=cfg_data.get("min_frequency", 2),
            special_tokens=cfg_data.get("special_tokens", ["<pad>", "<unk>", "<bos>", "<eos>"]),
            unk_token=cfg_data.get("unk_token", "<unk>"),
            pad_token=cfg_data.get("pad_token", "<pad>"),
            bos_token=cfg_data.get("bos_token", "<bos>"),
            eos_token=cfg_data.get("eos_token", "<eos>"),
        )
        builder = cls(config)
        builder._tokenizer._vocab = {k: int(v) for k, v in data.get("vocab", {}).items()}
        builder._tokenizer._id_to_token = {int(v): k for k, v in data.get("vocab", {}).items()}
        builder._merges = [tuple(m) for m in data.get("merges", [])]  # type: ignore[misc]
        builder._tokenizer._initialized = True
        builder._built = True
        return builder

    # -- Introspection ------------------------------------------------------

    def is_built(self) -> bool:
        """Whether vocabulary and merges have been computed."""
        return self._built

    def get_special_token_ids(self) -> Dict[str, int]:
        """Return a mapping of special token names to their IDs."""
        vocab = self._tokenizer.get_vocab()
        return {t: vocab[t] for t in self._config.special_tokens if t in vocab}

    def summary(self) -> str:
        """Return a human-readable summary of the tokenizer state."""
        lines = [
            f"TokenizerDataBuilder(built={self._built})",
            f"  vocab_size: {self.vocab_size}",
            f"  merges:     {len(self._merges)}",
            f"  special:    {self._config.special_tokens}",
        ]
        return "\n".join(lines)
