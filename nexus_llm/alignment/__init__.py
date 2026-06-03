"""Alignment module for Nexus-LLM - RLHF and DPO preference alignment.

Provides tools for aligning language models with human preferences
using Direct Preference Optimization (DPO) and RLHF-style training,
including preference data management and reward modelling.
"""

from nexus_llm.alignment.trainer import AlignmentTrainer
from nexus_llm.alignment.config import RLHFConfig
from nexus_llm.alignment.preference import PreferenceDataset
from nexus_llm.alignment.reward import RewardModel

__all__ = ["AlignmentTrainer", "RLHFConfig", "PreferenceDataset", "RewardModel"]
