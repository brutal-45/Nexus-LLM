"""Tests for the distillation module.

Covers Distiller, DistillationConfig, and DistillationTrainer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexus_llm.distillation.distiller import Distiller
from nexus_llm.distillation.config import DistillationConfig
from nexus_llm.distillation.trainer import DistillationTrainer


# ---------------------------------------------------------------------------
# DistillationConfig
# ---------------------------------------------------------------------------

class TestDistillationConfig:
    """Tests for DistillationConfig."""

    def test_defaults(self):
        config = DistillationConfig()
        assert config is not None

    def test_custom_config(self):
        config = DistillationConfig(temperature=2.0, alpha=0.5)
        assert config.temperature == 2.0
        assert config.alpha == 0.5

    def test_to_dict(self):
        config = DistillationConfig()
        d = config.to_dict()
        assert isinstance(d, dict)

    def test_from_dict(self):
        data = {"temperature": 3.0, "alpha": 0.7}
        config = DistillationConfig.from_dict(data)
        assert config.temperature == 3.0


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------

class TestDistiller:
    """Tests for Distiller."""

    def test_create_distiller(self):
        d = Distiller()
        assert d is not None

    def test_distiller_with_config(self):
        config = DistillationConfig(temperature=2.0)
        d = Distiller(config=config)
        assert d.config is not None

    def test_get_config(self):
        d = Distiller()
        config = d.get_config()
        assert isinstance(config, DistillationConfig)


# ---------------------------------------------------------------------------
# DistillationTrainer
# ---------------------------------------------------------------------------

class TestDistillationTrainer:
    """Tests for DistillationTrainer."""

    def test_create_trainer(self):
        trainer = DistillationTrainer()
        assert trainer is not None

    def test_trainer_with_config(self):
        config = DistillationConfig()
        trainer = DistillationTrainer(config=config)
        assert trainer is not None

    def test_train_step(self):
        trainer = DistillationTrainer()
        # Should have train-related methods
        assert hasattr(trainer, "train") or hasattr(trainer, "train_step")
