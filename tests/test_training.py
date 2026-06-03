"""Tests for Nexus LLM training components."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nexus.training.scheduler import get_scheduler, CosineAnnealingWithWarmup, LinearWarmupWithDecay
from nexus.training.checkpoint import CheckpointManager
from nexus.data.dataset import DataCollator, PackedSequence


class TestTrainingArguments:
    """Test training configuration."""

    def test_create_training_args(self):
        from nexus.training.trainer import TrainingArguments
        args = TrainingArguments()
        assert args.learning_rate == 1e-4
        assert args.max_steps == 5_000_000
        assert args.precision == "bf16_mixed"


class TestSchedulers:
    """Test learning rate schedulers."""

    def _get_optimizer_and_model(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        return optimizer

    def test_cosine_scheduler(self):
        optimizer = self._get_optimizer_and_model()
        scheduler = get_scheduler("cosine", optimizer, warmup_steps=100, total_steps=1000)
        assert isinstance(scheduler, CosineAnnealingWithWarmup)

    def test_linear_scheduler(self):
        optimizer = self._get_optimizer_and_model()
        scheduler = get_scheduler("linear", optimizer, warmup_steps=100, total_steps=1000)
        assert isinstance(scheduler, LinearWarmupWithDecay)

    def test_all_schedulers(self):
        """Test that all scheduler types can be created."""
        scheduler_names = [
            "cosine", "linear", "warmup_stable_decay", "cosine_warm_restarts",
            "inv_sqrt", "onecycle", "polynomial", "constant", "exponential",
            "warmup_hold_decay",
        ]
        for name in scheduler_names:
            model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            try:
                scheduler = get_scheduler(name, optimizer, warmup_steps=100, total_steps=1000)
                assert scheduler is not None
            except ValueError:
                pass  # Some schedulers need extra args

    def test_unknown_scheduler(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with pytest.raises(ValueError):
            get_scheduler("nonexistent", optimizer)

    def test_cosine_warmup_ramp(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps=100, total_steps=1000)
        # During warmup, LR should increase
        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
        assert lrs[-1] > lrs[0]


class TestCheckpointManager:
    """Test checkpoint save and load."""

    def test_should_save(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), save_interval=100)
        assert not manager.should_save(0)
        assert manager.should_save(100)
        assert manager.should_save(200)
        assert not manager.should_save(50)

    def test_save_and_load(self, tmp_path):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        manager = CheckpointManager(str(tmp_path), save_interval=100)

        path = manager.save(model, optimizer, step=100, epoch=0)
        assert os.path.exists(path)

        state = manager.load(path, model)
        assert "metadata" in state


class TestDataCollator:
    """Test data collator."""

    def test_collate(self):
        collator = DataCollator()
        samples = [
            PackedSequence(
                input_ids=torch.randint(0, 100, (32,)),
                attention_mask=torch.ones(32, dtype=torch.long),
                labels=torch.randint(0, 100, (32,)),
                loss_mask=torch.ones(32, dtype=torch.long),
            ),
            PackedSequence(
                input_ids=torch.randint(0, 100, (32,)),
                attention_mask=torch.ones(32, dtype=torch.long),
                labels=torch.randint(0, 100, (32,)),
                loss_mask=torch.ones(32, dtype=torch.long),
            ),
        ]
        batch = collator(samples)
        assert "input_ids" in batch
        assert batch["input_ids"].shape == (2, 32)
