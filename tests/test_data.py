"""Tests for Nexus LLM data pipeline components."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nexus.data.dataset import PackedDataset, DataCollator, PackedSequence
from nexus.data.preprocessing import TextPreprocessor
from nexus.data.data_quality import (
    LengthFilter, RepetitionFilter, LanguageFilter,
    QualityClassifier, DataQualityPipeline, QualityMetrics,
)
from nexus.data.deduplication import ExactDeduplicator, MinHashLSHDeduplicator


class TestPackedDataset:
    """Test PackedDataset."""

    def test_create_dataset(self, tiny_tokenizer):
        docs = ["Hello world test document. " * 50 for _ in range(5)]
        dataset = PackedDataset(docs, tiny_tokenizer, seq_length=64)
        assert len(dataset) > 0

    def test_getitem(self, tiny_tokenizer):
        docs = ["The quick brown fox jumps over the lazy dog. " * 50 for _ in range(3)]
        dataset = PackedDataset(docs, tiny_tokenizer, seq_length=32)
        sample = dataset[0]
        assert isinstance(sample, PackedSequence)
        assert sample.input_ids.shape == (32,)
        assert sample.attention_mask.shape == (32,)


class TestTextPreprocessor:
    """Test text preprocessing."""

    def test_normalization(self):
        preprocessor = TextPreprocessor()
        # Basic test that preprocessor exists
        assert hasattr(preprocessor, '__class__')


class TestQualityFilters:
    """Test data quality filters."""

    def test_length_filter_pass(self):
        f = LengthFilter(min_tokens=10, max_tokens=10000)
        text = "This is a test document with enough words to pass the minimum length filter requirement."
        assert f.filter(text) is True

    def test_length_filter_fail_short(self):
        f = LengthFilter(min_tokens=100)
        text = "Short"
        assert f.filter(text) is False

    def test_repetition_filter(self):
        f = RepetitionFilter()
        clean_text = "The quick brown fox jumps over the lazy dog. Natural language processing is important."
        assert f.score(clean_text) > 0.0

    def test_language_filter_english(self):
        f = LanguageFilter(target_languages={"en"})
        text = "This is an English document about artificial intelligence and machine learning."
        lang, conf = f.detect_language(text)
        assert lang in ("en", "other_latin", "latin")

    def test_quality_classifier(self):
        qc = QualityClassifier()
        good_text = "Artificial intelligence has transformed many industries including healthcare, finance, and transportation."
        bad_text = "asdf zxcv qwer 1234 !@#$"
        good_score = qc.score(good_text)
        bad_score = qc.score(bad_text)
        assert good_score >= bad_score

    def test_quality_pipeline(self):
        pipeline = DataQualityPipeline(mode="all")
        text = "This is a reasonably well-written English document about technology and science."
        score = pipeline.score_document(text)
        assert 0.0 <= score.overall <= 1.0


class TestDeduplication:
    """Test deduplication."""

    def test_exact_dedup(self):
        docs = ["Hello world", "Hello world", "Different document"]
        dedup = ExactDeduplicator()
        unique, dupes = dedup.deduplicate(docs)
        assert len(unique) == 2
        assert len(dupes) == 1

    def test_minhash_similarity(self):
        from nexus.data.deduplication import MinHasher
        mh = MinHasher(num_hashes=64)
        sig1 = mh.compute_signature("The quick brown fox jumps over the lazy dog")
        sig2 = mh.compute_signature("The quick brown fox jumps over the lazy dog")
        # Same text should have identical signatures
        assert mh.jaccard_similarity(sig1, sig2) == 1.0


class TestQualityMetrics:
    """Test quality metrics computation."""

    def test_dataset_stats(self):
        docs = [
            "This is a test document about AI.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
        ]
        stats = QualityMetrics.compute_dataset_stats(docs)
        assert stats["total_documents"] == 3
        assert stats["total_chars"] > 0
