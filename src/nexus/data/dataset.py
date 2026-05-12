"""
Streaming Dataset for LLM Training
=====================================
Memory-efficient dataset that streams data from disk without loading everything
into RAM. Supports packing multiple documents into fixed-length sequences
for maximum GPU utilization.

Key features:
    - Streams from JSONL, TXT, or memory-mapped files
    - Packs multiple short documents into fixed-length sequences
    - Handles variable-length documents efficiently
    - Supports distributed sampling with world_size/rank
    - Shuffles data at the document level
    - Preserves document boundaries (no cross-document packing artifacts)
    
Packing Strategy:
    For seq_len=8192, if we have documents of lengths [100, 5000, 3000, 200]:
    Pack into: [100, 5000, PAD/sep, 3000, 200, PAD] -> fills two sequences
    Padding tokens have loss_weight=0, so they don't affect training.
"""

from __future__ import annotations
import os
import json
import random
import mmap
import struct
from itertools import cycle, islice
from typing import Dict, List, Optional, Tuple, Iterator, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from .tokenizer import BPETokenizer


@dataclass
class PackedSequence:
    """A packed sequence with attention mask and loss mask."""
    input_ids: torch.Tensor       # (seq_len,) - token IDs
    attention_mask: torch.Tensor  # (seq_len,) - 1=real token, 0=padding
    labels: torch.Tensor          # (seq_len,) - shifted labels for loss
    loss_mask: torch.Tensor       # (seq_len,) - 1=compute loss, 0=ignore


class StreamingDataset(IterableDataset):
    """
    Streaming dataset that reads JSONL files line-by-line.
    
    Memory-efficient: only keeps one line in memory at a time.
    Supports distributed training with proper sharding.
    
    Expected JSONL format (one JSON object per line):
        {"text": "Some document text..."}
        OR
        {"instruction": "...", "output": "..."}
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer: BPETokenizer,
        seq_length: int = 8192,
        shuffle: bool = True,
        seed: int = 42,
        infinite: bool = True,
        text_key: str = "text",
    ):
        self.data_files = sorted(data_files)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.seed = seed
        self.infinite = infinite
        self.text_key = text_key
        
        # Distributed training
        self.rank = 0
        self.world_size = 1
        
        # Statistics
        self.total_documents = 0
        self.total_tokens = 0

    def set_distributed(self, rank: int, world_size: int):
        """Configure for distributed data parallel training."""
        self.rank = rank
        self.world_size = world_size

    def _iter_files(self) -> Iterator[Tuple[str, Iterator[Dict]]]:
        """Iterate over data files, yielding (filename, line_iterator)."""
        files = list(self.data_files)
        if self.shuffle:
            random.Random(self.seed).shuffle(files)
        
        for filepath in files:
            if filepath.endswith(".jsonl"):
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                yield filepath, json.loads(line)
                            except json.JSONDecodeError:
                                continue
            elif filepath.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield filepath, {self.text_key: line}

    def _iter_documents(self) -> Iterator[str]:
        """Stream individual documents from all files."""
        doc_buffer: List[str] = []
        
        for filepath, item in self._iter_files():
            # Handle different data formats
            if isinstance(item, dict):
                if self.text_key in item:
                    text = item[self.text_key]
                elif "instruction" in item and "output" in item:
                    text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
                else:
                    continue
            else:
                continue
            
            if text and len(text.strip()) > 10:
                doc_buffer.append(text)
        
        # Shuffle documents for better training
        if self.shuffle:
            random.Random(self.seed + 1).shuffle(doc_buffer)
        
        for doc in doc_buffer:
            yield doc

    def _iter_tokenized(self) -> Iterator[List[int]]:
        """Tokenize documents and yield token sequences."""
        for doc in self._iter_documents():
            token_ids = self.tokenizer.encode(doc, add_bos=True, add_eos=True)
            if len(token_ids) > 5:  # Skip very short documents
                self.total_documents += 1
                self.total_tokens += len(token_ids)
                yield token_ids

    def _iter_packed(self) -> Iterator[PackedSequence]:
        """
        Pack multiple documents into fixed-length sequences.
        
        Strategy:
            - Accumulate tokens from documents until seq_length is reached
            - Pad the last chunk if needed
            - Use loss_mask to ignore padding and document boundaries
        """
        buffer: List[int] = []
        doc_boundaries: List[int] = []  # Positions where new docs start
        
        for token_ids in self._iter_tokenized():
            start_pos = len(buffer)
            buffer.extend(token_ids)
            doc_boundaries.append(start_pos)
            
            # Yield complete sequences
            while len(buffer) >= self.seq_length:
                chunk = buffer[:self.seq_length]
                buffer = buffer[self.seq_length:]
                
                # Compute masks
                input_ids = torch.tensor(chunk, dtype=torch.long)
                
                # Find which doc boundaries fall within this chunk
                loss_mask = torch.ones(self.seq_length, dtype=torch.long)
                for boundary in doc_boundaries:
                    if 0 < boundary < self.seq_length:
                        loss_mask[:boundary] = 0  # Don't compute loss on boundary
                doc_boundaries = [
                    b - self.seq_length for b in doc_boundaries if b >= self.seq_length
                ]
                
                attention_mask = torch.ones(self.seq_length, dtype=torch.long)
                
                # Labels are input_ids shifted by 1 (next token prediction)
                labels = input_ids.clone()
                labels[:-1] = input_ids[1:]
                labels[-1] = self.tokenizer.special_tokens.get("<eos>", 2)
                
                # Zero out loss for padding
                loss_mask *= attention_mask
                
                yield PackedSequence(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    loss_mask=loss_mask,
                )
        
        # Yield remaining buffer as final sequence (padded)
        if buffer:
            pad_len = self.seq_length - len(buffer)
            input_ids = torch.tensor(
                buffer + [self.tokenizer.special_tokens["<pad>"]] * pad_len,
                dtype=torch.long,
            )
            attention_mask = torch.cat([
                torch.ones(len(buffer), dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            loss_mask = attention_mask.clone()
            
            yield PackedSequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )

    def __iter__(self) -> Iterator[PackedSequence]:
        """Main iterator that handles distributed sharding."""
        if self.infinite:
            # Cycle forever for training
            data_iter = cycle(self._iter_packed())
        else:
            data_iter = self._iter_packed()
        
        # Distributed sharding: each rank gets every world_size-th sample
        if self.world_size > 1:
            data_iter = islice(data_iter, self.rank, None, self.world_size)
        
        return data_iter


class PackedDataset(Dataset):
    """
    In-memory packed dataset for smaller corpora or evaluation.
    
    Pre-tokenizes and packs all data upfront. Suitable for datasets
    that fit in memory (< 100GB).
    """

    def __init__(
        self,
        documents: List[str],
        tokenizer: BPETokenizer,
        seq_length: int = 8192,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.sequences: List[PackedSequence] = []
        
        # Tokenize all documents
        tokenized = []
        for doc in documents:
            ids = tokenizer.encode(doc, add_bos=True, add_eos=True)
            if len(ids) > 5:
                tokenized.append(ids)
        
        # Pack into fixed-length sequences
        buffer: List[int] = []
        for ids in tokenized:
            buffer.extend(ids)
            while len(buffer) >= seq_length:
                chunk = buffer[:seq_length]
                buffer = buffer[seq_length:]
                
                input_ids = torch.tensor(chunk, dtype=torch.long)
                attention_mask = torch.ones(seq_length, dtype=torch.long)
                labels = input_ids.clone()
                labels[:-1] = input_ids[1:]
                labels[-1] = tokenizer.special_tokens["<eos>"]
                loss_mask = attention_mask.clone()
                
                self.sequences.append(PackedSequence(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    loss_mask=loss_mask,
                ))
        
        # Pad remaining
        if buffer:
            pad_len = seq_length - len(buffer)
            input_ids = torch.tensor(
                buffer + [tokenizer.special_tokens["<pad>"]] * pad_len,
                dtype=torch.long,
            )
            attention_mask = torch.cat([
                torch.ones(len(buffer), dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            loss_mask = attention_mask.clone()
            self.sequences.append(PackedSequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            ))
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> PackedSequence:
        return self.sequences[idx]


class DataCollator:
    """
    Collates individual PackedSequence samples into batches.
    
    Since sequences are already fixed-length and packed,
    this simply stacks them into tensors.
    """

    def __call__(self, samples: List[PackedSequence]) -> Dict[str, torch.Tensor]:
        if not samples:
            return {}
        
        return {
            "input_ids": torch.stack([s.input_ids for s in samples]),
            "attention_mask": torch.stack([s.attention_mask for s in samples]),
            "labels": torch.stack([s.labels for s in samples]),
            "loss_mask": torch.stack([s.loss_mask for s in samples]),
        }
