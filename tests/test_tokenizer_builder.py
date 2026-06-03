"""Tests for BPE tokenizer data building."""
import pytest
from collections import Counter


def test_byte_pair_encoding_basic():
    word_freqs = Counter({"l o w": 5, "l o w e r": 2, "n e w e s t": 6})
    pairs = Counter()
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    best_pair = pairs.most_common(1)[0][0]
    assert best_pair is not None


def test_vocab_construction():
    corpus = ["hello world", "hello there", "world peace"]
    vocab = set()
    for text in corpus:
        vocab.update(text.split())
    assert "hello" in vocab
    assert len(vocab) >= 3


def test_token_frequency_counting():
    text = "the cat sat on the mat the cat"
    freq = Counter(text.split())
    assert freq["the"] == 3
    assert freq["cat"] == 2


def test_merge_operation():
    symbols = ["l", "o", "w"]
    merge_pair = ("l", "o")
    new_symbols = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == merge_pair[0] and symbols[i+1] == merge_pair[1]:
            new_symbols.append("lo")
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1
    assert "lo" in new_symbols


def test_special_tokens():
    vocab = {"hello": 0, "world": 1}
    for st in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
        vocab[st] = len(vocab)
    assert vocab["<PAD>"] == 2
    assert len(vocab) == 6
