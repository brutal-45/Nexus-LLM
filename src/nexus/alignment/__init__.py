"""
Alignment Package Init
=====================
Complete post-training alignment infrastructure for Nexus:
    - sft: Supervised Fine-Tuning
    - dpo: Direct Preference Optimization (DPO, IPO, KTO, SimPO, ORPO)
    - reward: Reward model for RLHF
    - ppo: Proximal Policy Optimization with Actor-Critic, GAE, KL adaptive controller
"""

from .sft import SFTTrainer
from .dpo import DPOTrainer
from .reward import RewardModel
from .ppo import (
    PPOConfig,
    PPOTrainingStats,
    ValueHead,
    CriticModel,
    RolloutBuffer,
    PPOTrainer,
)

__all__ = [
    "SFTTrainer",
    "DPOTrainer",
    "RewardModel",
    "PPOConfig",
    "PPOTrainingStats",
    "ValueHead",
    "CriticModel",
    "RolloutBuffer",
    "PPOTrainer",
]
