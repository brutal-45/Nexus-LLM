"""Tests for dataset handling."""
import pytest
import json
import os
import torch
from nexus.data.dataset import PackedSequence, DataCollator


def test_packed_sequence_creation():
    """Test creating a PackedSequence."""
    seq_len = 32
    ps = PackedSequence(
        input_ids=torch.randint(0, 100, (seq_len,)),
        attention_mask=torch.ones(seq_len, dtype=torch.long),
        labels=torch.randint(0, 100, (seq_len,)),
        loss_mask=torch.ones(seq_len, dtype=torch.long),
    )
    assert ps.input_ids.shape == (seq_len,)
    assert ps.attention_mask.shape == (seq_len,)
    assert ps.labels.shape == (seq_len,)
    assert ps.loss_mask.shape == (seq_len,)


def test_packed_sequence_shapes():
    """Test that PackedSequence tensors have matching shapes."""
    seq_len = 16
    ps = PackedSequence(
        input_ids=torch.randint(0, 50, (seq_len,)),
        attention_mask=torch.ones(seq_len, dtype=torch.long),
        labels=torch.randint(0, 50, (seq_len,)),
        loss_mask=torch.ones(seq_len, dtype=torch.long),
    )
    assert ps.input_ids.shape[0] == ps.attention_mask.shape[0]
    assert ps.input_ids.shape[0] == ps.labels.shape[0]
    assert ps.input_ids.shape[0] == ps.loss_mask.shape[0]


def test_data_collator_single():
    """Test DataCollator with a single sample."""
    seq_len = 8
    sample = PackedSequence(
        input_ids=torch.randint(0, 100, (seq_len,)),
        attention_mask=torch.ones(seq_len, dtype=torch.long),
        labels=torch.randint(0, 100, (seq_len,)),
        loss_mask=torch.ones(seq_len, dtype=torch.long),
    )
    collator = DataCollator()
    batch = collator([sample])
    assert "input_ids" in batch
    assert batch["input_ids"].shape == (1, seq_len)


def test_data_collator_batch():
    """Test DataCollator with a batch of samples."""
    seq_len = 8
    batch_size = 4
    samples = [
        PackedSequence(
            input_ids=torch.randint(0, 100, (seq_len,)),
            attention_mask=torch.ones(seq_len, dtype=torch.long),
            labels=torch.randint(0, 100, (seq_len,)),
            loss_mask=torch.ones(seq_len, dtype=torch.long),
        )
        for _ in range(batch_size)
    ]
    collator = DataCollator()
    batch = collator(samples)
    assert batch["input_ids"].shape == (batch_size, seq_len)
    assert batch["attention_mask"].shape == (batch_size, seq_len)


def test_data_collator_empty():
    """Test DataCollator with empty list."""
    collator = DataCollator()
    batch = collator([])
    assert batch == {}
