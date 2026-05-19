"""Tests for data collation."""
import pytest
import torch
from nexus.data.dataset import DataCollator, PackedSequence


def test_collator_single_sample():
    """Test collating a single sample."""
    seq_len = 16
    sample = PackedSequence(
        input_ids=torch.randint(0, 100, (seq_len,)),
        attention_mask=torch.ones(seq_len, dtype=torch.long),
        labels=torch.randint(0, 100, (seq_len,)),
        loss_mask=torch.ones(seq_len, dtype=torch.long),
    )
    collator = DataCollator()
    batch = collator([sample])
    assert batch["input_ids"].shape == (1, seq_len)
    assert batch["labels"].shape == (1, seq_len)


def test_collator_batch():
    """Test collating a batch of samples."""
    seq_len = 16
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
    assert batch["input_ids"].shape[0] == batch_size
    assert batch["attention_mask"].shape[0] == batch_size


def test_collator_preserves_dtypes():
    """Test that collator preserves tensor dtypes."""
    sample = PackedSequence(
        input_ids=torch.randint(0, 100, (8,)),
        attention_mask=torch.ones(8, dtype=torch.long),
        labels=torch.randint(0, 100, (8,)),
        loss_mask=torch.ones(8, dtype=torch.long),
    )
    collator = DataCollator()
    batch = collator([sample])
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long


def test_collator_empty():
    """Test collator with empty input."""
    collator = DataCollator()
    batch = collator([])
    assert batch == {}
