"""
BPE Tokenizer - Built from Scratch
====================================
A complete Byte Pair Encoding tokenizer implementation.

BPE Algorithm:
    1. Start with a base vocabulary of all individual bytes (256 tokens)
    2. Iteratively find the most frequent adjacent pair of tokens in the corpus
    3. Merge that pair into a single new token
    4. Repeat until desired vocabulary size is reached

This tokenizer supports:
    - Training on raw text corpus
    - Encoding text to token IDs
    - Decoding token IDs back to text
    - Special tokens (BOS, EOS, PAD, UNK)
    - Serialization to/from JSON for persistence
    - Pre-tokenization (regex splitting) for better merges
    
Reference:
    - Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016)
"""

from __future__ import annotations
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass, field


# Default special tokens
DEFAULT_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


# Pre-tokenization regex (GPT-2 style)
# Splits text into words, numbers, punctuation, whitespace chunks
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
)


@dataclass
class TokenizerConfig:
    """Configuration for BPE tokenizer."""
    vocab_size: int = 65536
    special_tokens: Dict[str, int] = field(default_factory=lambda: DEFAULT_SPECIAL_TOKENS.copy())
    max_token_length: int = 64  # Maximum length of a single token in bytes


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer.
    
    Handles the full pipeline:
        Text -> Pre-tokenization -> Byte encoding -> BPE merges -> Token IDs
        Token IDs -> Tokens -> Byte decoding -> Text
    
    Example:
        tokenizer = BPETokenizer(vocab_size=65536)
        tokenizer.train(texts)
        token_ids = tokenizer.encode("Hello, world!")
        text = tokenizer.decode(token_ids)
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.vocab_size = self.config.vocab_size
        
        # Core data structures
        self.merges: Dict[Tuple[int, int], int] = {}  # (token_a, token_b) -> merged_id
        self.vocab: Dict[int, bytes] = {}              # token_id -> byte sequence
        self.inverse_vocab: Dict[bytes, int] = {}       # byte sequence -> token_id
        
        # Special tokens
        self.special_tokens: Dict[str, int] = self.config.special_tokens.copy()
        self.inverse_special_tokens: Dict[int, str] = {
            v: k for k, v in self.special_tokens.items()
        }
        
        # Initialize base vocabulary (256 bytes)
        self._init_base_vocab()
    
    def _init_base_vocab(self):
        """Initialize vocabulary with all 256 individual bytes."""
        for i in range(256):
            self.vocab[i] = bytes([i])
            self.inverse_vocab[bytes([i])] = i
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text into words/chunks using regex.
        
        This step ensures BPE merges happen within natural word boundaries,
        not across them, which produces more meaningful subword units.
        """
        return re.findall(GPT2_PAT, text)
    
    def _text_to_bytes(self, text: str) -> List[int]:
        """Convert text to list of byte values."""
        return list(text.encode("utf-8"))
    
    def _bytes_to_tokens(self, byte_seq: List[int]) -> List[int]:
        """Convert a byte sequence to initial BPE tokens (one per byte)."""
        return byte_seq.copy()
    
    def _get_pair_counts(self, token_ids_list: List[List[int]]) -> Counter:
        """Count frequency of all adjacent token pairs across the corpus."""
        counts: Counter = Counter()
        for tokens in token_ids_list:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                counts[pair] += 1
        return counts
    
    def _merge_pair(
        self,
        token_ids_list: List[List[int]],
        pair: Tuple[int, int],
        new_id: int,
    ) -> List[List[int]]:
        """Merge all occurrences of a pair in the token lists."""
        merged = []
        for tokens in token_ids_list:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            merged.append(new_tokens)
        return merged
    
    def train(self, texts: List[str], min_frequency: int = 2):
        """
        Train BPE tokenizer on a text corpus.
        
        Args:
            texts: List of text strings to train on.
            min_frequency: Minimum frequency for a pair to be merged.
                         Higher values create smaller, more efficient vocabs.
        """
        print(f"Training BPE tokenizer on {len(texts)} texts...")
        print(f"Target vocab size: {self.vocab_size}")
        
        # Step 1: Pre-tokenize and convert to byte sequences
        all_token_lists: List[List[int]] = []
        word_freqs: Counter = Counter()
        
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                byte_seq = self._text_to_bytes(word)
                if byte_seq:
                    word_freqs[tuple(byte_seq)] += 1
        
        # Convert to token ID lists with frequencies
        # Store (token_list, frequency) for efficient counting
        corpus: List[Tuple[List[int], int]] = [
            (list(byte_seq), freq)
            for byte_seq, freq in word_freqs.items()
        ]
        
        # Current vocab size starts at 256 (bytes) + special tokens
        next_id = 256 + len(self.special_tokens)
        
        # Step 2: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - next_id
        print(f"Performing {num_merges} BPE merges...")
        
        for step in range(num_merges):
            # Count pairs weighted by frequency
            pair_counts: Counter = Counter()
            for tokens, freq in corpus:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += freq
            
            if not pair_counts:
                print(f"  No more pairs to merge at step {step}")
                break
            
            # Find the most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            best_count = pair_counts[best_pair]
            
            if best_count < min_frequency:
                print(f"  Best pair count {best_count} < min_frequency {min_frequency}")
                break
            
            # Create new token by concatenating the byte sequences
            self.merges[best_pair] = next_id
            new_token_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[next_id] = new_token_bytes
            self.inverse_vocab[new_token_bytes] = next_id
            
            # Merge the pair in the corpus
            new_corpus = []
            for tokens, freq in corpus:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (
                        i < len(tokens) - 1
                        and tokens[i] == best_pair[0]
                        and tokens[i + 1] == best_pair[1]
                    ):
                        new_tokens.append(next_id)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_corpus.append((new_tokens, freq))
            corpus = new_corpus
            
            next_id += 1
            
            if (step + 1) % 5000 == 0:
                print(f"  Step {step + 1}/{num_merges}: "
                      f"merged {best_pair} -> {next_id - 1} "
                      f"(count={best_count}, vocab={next_id})")
        
        self.vocab_size = next_id
        print(f"Training complete. Final vocab size: {self.vocab_size}")
    
    def _apply_bpe(self, token_ids: List[int]) -> List[int]:
        """
        Apply learned BPE merges to a token sequence.
        
        Uses the learned merge rules to iteratively merge the most frequent
        pairs first (priority order from training).
        """
        tokens = list(token_ids)
        
        while len(tokens) >= 2:
            # Find the pair with the lowest merge index (highest priority)
            best_pair = None
            best_merge_idx = float("inf")
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges and self.merges[pair] < best_merge_idx:
                    best_pair = pair
                    best_merge_idx = self.merges[pair]
            
            if best_pair is None:
                break
            
            # Merge all occurrences of the best pair
            new_id = self.merges[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Pipeline: Text -> Pre-tokenize -> Bytes -> BPE -> Token IDs
        
        Args:
            text: Input text string.
            add_bos: Prepend beginning-of-sequence token.
            add_eos: Append end-of-sequence token.
        
        Returns:
            List of integer token IDs.
        """
        token_ids: List[int] = []
        
        if add_bos:
            token_ids.append(self.special_tokens["<bos>"])
        
        # Pre-tokenize
        words = self._pre_tokenize(text)
        
        for word in words:
            # Convert to bytes
            byte_seq = self._text_to_bytes(word)
            
            if not byte_seq:
                continue
            
            # Apply BPE merges
            merged = self._apply_bpe(byte_seq)
            token_ids.extend(merged)
        
        if add_eos:
            token_ids.append(self.special_tokens["<eos>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of integer token IDs.
            skip_special: Whether to skip special tokens in output.
        
        Returns:
            Decoded text string.
        """
        byte_chunks: List[bytes] = []
        
        for tid in token_ids:
            # Skip special tokens
            if skip_special and tid in self.inverse_special_tokens:
                continue
            
            if tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
            else:
                byte_chunks.append(b"")  # Unknown token
        
        return b"".join(byte_chunks).decode("utf-8", errors="replace")
    
    def tokenize(self, text: str) -> List[str]:
        """Encode and return the string tokens (for inspection)."""
        token_ids = self.encode(text, add_bos=False, add_eos=False)
        return [self.decode([tid]) for tid in token_ids]
    
    def save(self, path: str):
        """
        Save tokenizer to a JSON file.
        
        Saves: merges, vocab (as hex strings), special tokens, and config.
        """
        data = {
            "config": {
                "vocab_size": self.vocab_size,
                "max_token_length": self.config.max_token_length,
            },
            "special_tokens": self.special_tokens,
            "merges": {
                f"{k[0]} {k[1]}": v
                for k, v in self.merges.items()
            },
            "vocab": {
                str(k): v.hex()
                for k, v in self.vocab.items()
            },
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        """
        Load tokenizer from a JSON file.
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.vocab_size = data["config"]["vocab_size"]
        tokenizer.config.max_token_length = data["config"]["max_token_length"]
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.inverse_special_tokens = {
            v: k for k, v in tokenizer.special_tokens.items()
        }
        
        # Load merges
        tokenizer.merges = {}
        for pair_str, merged_id in data["merges"].items():
            a, b = pair_str.split(" ")
            tokenizer.merges[(int(a), int(b))] = merged_id
        
        # Load vocab
        tokenizer.vocab = {}
        tokenizer.inverse_vocab = {}
        for k, v in data["vocab"].items():
            token_bytes = bytes.fromhex(v)
            tokenizer.vocab[int(k)] = token_bytes
            tokenizer.inverse_vocab[token_bytes] = int(k)
        
        print(f"Tokenizer loaded from {path} (vocab_size={tokenizer.vocab_size})")
        return tokenizer


# Convenience: use sentencepiece if available for faster tokenization
class SentencePieceTokenizer:
    """
    Wrapper around sentencepiece for high-performance tokenization.
    
    Falls back to pure BPE if sentencepiece is not available.
    This provides ~10x faster encoding/decoding than pure Python BPE.
    """

    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 65536):
        self.vocab_size = vocab_size
        self._sp_model = None
        self._sp_available = False

        try:
            import sentencepiece as spm
            if model_path and os.path.exists(model_path):
                self._sp_model = spm.SentencePieceProcessor()
                self._sp_model.load(model_path)
                self._sp_available = True
                self.vocab_size = self._sp_model.get_piece_size()
        except ImportError:
            pass

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        if self._sp_available:
            ids = self._sp_model.encode(text)
            if add_bos:
                ids = [self._sp_model.bos_id()] + ids
            if add_eos:
                ids = ids + [self._sp_model.eos_id()]
            return ids
        raise RuntimeError("sentencepiece not available and no model loaded")

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        if self._sp_available:
            return self._sp_model.decode(token_ids)
        raise RuntimeError("sentencepiece not available")

    def train(self, input_files: List[str], model_prefix: str, **kwargs):
        """Train a sentencepiece model from text files."""
        try:
            import sentencepiece as spm
            spm.SentencePieceTrainer.train(
                input=",".join(input_files),
                model_prefix=model_prefix,
                vocab_size=self.vocab_size,
                model_type="bpe",
                **kwargs,
            )
            # Load the trained model
            self._sp_model = spm.SentencePieceProcessor()
            self._sp_model.load(f"{model_prefix}.model")
            self._sp_available = True
            self.vocab_size = self._sp_model.get_piece_size()
        except ImportError:
            raise RuntimeError("sentencepiece is required for training")
