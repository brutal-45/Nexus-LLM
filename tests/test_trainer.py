"""Tests for training loop."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from nexus.training.trainer import Trainer, TrainingArguments


def test_training_arguments_defaults():
    """Test TrainingArguments default values."""
    args = TrainingArguments()
    assert args.learning_rate == 1e-4
    assert args.weight_decay == 0.1
    assert args.max_grad_norm == 1.0
    assert args.precision == "bf16_mixed"
    assert args.gradient_checkpointing is True


def test_training_arguments_custom():
    """Test custom TrainingArguments."""
    args = TrainingArguments(
        learning_rate=5e-5,
        max_steps=10000,
        warmup_steps=500,
    )
    assert args.learning_rate == 5e-5
    assert args.max_steps == 10000
    assert args.warmup_steps == 500


def test_training_arguments_seq_length():
    """Test seq_length configuration."""
    args = TrainingArguments(seq_length=4096)
    assert args.seq_length == 4096


def test_training_arguments_batch_sizes():
    """Test batch size configuration."""
    args = TrainingArguments(
        micro_batch_size=2,
        global_batch_size=512,
    )
    assert args.micro_batch_size == 2
    assert args.global_batch_size == 512


def test_training_arguments_precision_options():
    """Test precision configuration."""
    for precision in ["bf16_mixed", "fp16_mixed", "fp32"]:
        args = TrainingArguments(precision=precision)
        assert args.precision == precision


def test_training_arguments_save_interval():
    """Test checkpoint save interval."""
    args = TrainingArguments(save_interval=1000)
    assert args.save_interval == 1000
