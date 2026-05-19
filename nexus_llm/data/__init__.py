"""Nexus-LLM Data Module.

Provides data loading, processing, splitting, tokenization, format
conversion, and validation utilities for LLM training pipelines.

Public API Exports:
    - DataLoader: Load datasets from JSONL, CSV, JSON, Parquet, and HuggingFace.
    - DataProcessor: Chainable data transformations with lazy/eager execution.
    - DataSplitter: Random, stratified, k-fold, and time-based data splitting.
    - TokenizerDataBuilder: Build tokenizer vocabularies, BPE merge rules, special tokens.
    - FormatConverter: Convert between Alpaca, ChatML, ShareGPT, and HF formats.
    - DataValidator: Schema validation, completeness, quality scoring, duplication checks.
"""

from __future__ import annotations

from nexus_llm.data.loader import DataLoader, DatasetInfo
from nexus_llm.data.processor import DataProcessor
from nexus_llm.data.splitter import DataSplitter, SplitConfig
from nexus_llm.data.tokenizer_data import TokenizerDataBuilder, TokenizerConfig
from nexus_llm.data.converter import FormatConverter
from nexus_llm.data.validator import DataValidator, ValidationResult

__all__ = [
    # Loading
    "DataLoader",
    "DatasetInfo",
    # Processing
    "DataProcessor",
    # Splitting
    "DataSplitter",
    "SplitConfig",
    # Tokenizer data
    "TokenizerDataBuilder",
    "TokenizerConfig",
    # Format conversion
    "FormatConverter",
    # Validation
    "DataValidator",
    "ValidationResult",
]
