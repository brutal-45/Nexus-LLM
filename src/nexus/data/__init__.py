"""
Data Package Init
=================
Complete data pipeline for Nexus:
    - tokenizer: BPE tokenizer built from scratch (128K vocab, special tokens, byte fallback)
    - dataset: Streaming dataset with packing and memory-mapped sharding
    - preprocessing: Text cleaning, deduplication (exact + fuzzy MinHash LSH)
    - synthetic: Synthetic data generation (Self-Instruct, Evol-Instruct, rejection sampling, persona-based, math, code)
"""

from .tokenizer import BPETokenizer, SentencePieceTokenizer, TokenizerConfig
from .dataset import StreamingDataset, PackedDataset, DataCollator
from .preprocessing import TextCleaner, DataDeduplicator
from .synthetic import (
    Instruction,
    SyntheticDataConfig,
    SyntheticDataset,
    SelfInstructGenerator,
    EvolInstructGenerator,
    RejectionSamplingGenerator,
    PersonaGenerator,
    MathDataGenerator,
    CodeDataGenerator,
    SyntheticDataPipeline,
)

__all__ = [
    "BPETokenizer",
    "SentencePieceTokenizer",
    "TokenizerConfig",
    "StreamingDataset",
    "PackedDataset",
    "DataCollator",
    "TextCleaner",
    "DataDeduplicator",
    "Instruction",
    "SyntheticDataConfig",
    "SyntheticDataset",
    "SelfInstructGenerator",
    "EvolInstructGenerator",
    "RejectionSamplingGenerator",
    "PersonaGenerator",
    "MathDataGenerator",
    "CodeDataGenerator",
    "SyntheticDataPipeline",
]
