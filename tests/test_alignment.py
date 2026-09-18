"""Tests for the alignment module.

Covers AlignmentTrainer, RLHFConfig, PreferenceDataset, and RewardModel.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexus_llm.alignment.trainer import AlignmentTrainer
from nexus_llm.alignment.config import RLHFConfig
from nexus_llm.alignment.preference import PreferenceDataset
from nexus_llm.alignment.reward import RewardModel


# ---------------------------------------------------------------------------
# RLHFConfig
# ---------------------------------------------------------------------------

class TestRLHFConfig:
    """Tests for RLHFConfig."""

    def test_defaults(self):
        config = RLHFConfig()
        assert config is not None

    def test_custom_config(self):
        config = RLHFConfig(learning_rate=1e-5, kl_coefficient=0.1)
        assert config.learning_rate == 1e-5
        assert config.kl_coefficient == 0.1

    def test_to_dict(self):
        config = RLHFConfig()
        d = config.to_dict()
        assert isinstance(d, dict)

    def test_from_dict(self):
        data = {"learning_rate": 5e-6, "kl_coefficient": 0.2}
        config = RLHFConfig.from_dict(data)
        assert config.learning_rate == 5e-6


# ---------------------------------------------------------------------------
# PreferenceDataset
# ---------------------------------------------------------------------------

class TestPreferenceDataset:
    """Tests for PreferenceDataset."""

    def test_create_dataset(self):
        ds = PreferenceDataset()
        assert ds is not None

    def test_add_sample(self):
        ds = PreferenceDataset()
        ds.add(prompt="Hello", chosen="Hi there!", rejected="Go away")
        assert len(ds) == 1

    def test_get_sample(self):
        ds = PreferenceDataset()
        ds.add(prompt="Hello", chosen="Hi!", rejected="Nope")
        sample = ds.get(0)
        assert sample["prompt"] == "Hello"

    def test_iterate(self):
        ds = PreferenceDataset()
        ds.add(prompt="Q1", chosen="A1", rejected="B1")
        ds.add(prompt="Q2", chosen="A2", rejected="B2")
        samples = list(ds)
        assert len(samples) == 2


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

class TestRewardModel:
    """Tests for RewardModel."""

    def test_create_reward_model(self):
        rm = RewardModel()
        assert rm is not None

    def test_compute_reward(self):
        rm = RewardModel()
        reward = rm.compute_reward("Hello, how are you?")
        assert isinstance(reward, float)

    def test_batch_rewards(self):
        rm = RewardModel()
        texts = ["Hello", "Goodbye", "Thanks"]
        rewards = rm.batch_rewards(texts)
        assert len(rewards) == 3


# ---------------------------------------------------------------------------
# AlignmentTrainer
# ---------------------------------------------------------------------------

class TestAlignmentTrainer:
    """Tests for AlignmentTrainer."""

    def test_create_trainer(self):
        trainer = AlignmentTrainer()
        assert trainer is not None

    def test_trainer_with_config(self):
        config = RLHFConfig()
        trainer = AlignmentTrainer(config=config)
        assert trainer is not None

    def test_train_step(self):
        trainer = AlignmentTrainer()
        # Should have train-related methods
        assert hasattr(trainer, "train") or hasattr(trainer, "train_step")
