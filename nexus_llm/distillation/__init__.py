"""Distillation module for Nexus-LLM - knowledge distillation pipeline.

Enables training a compact student model to replicate the behaviour of
a larger teacher model using soft-label distillation with configurable
temperature and loss balancing.
"""

from nexus_llm.distillation.distiller import Distiller
from nexus_llm.distillation.config import DistillationConfig
from nexus_llm.distillation.trainer import DistillationTrainer

__all__ = ["Distiller", "DistillationConfig", "DistillationTrainer"]
