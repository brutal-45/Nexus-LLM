"""Nexus-LLM Training Module.

Provides complete training infrastructure including training loops, dataset handling,
fine-tuning, data collation, learning rate scheduling, callbacks, metrics,
checkpointing, distributed training, augmentation, preprocessing, evaluation,
optimizers, loss functions, curriculum learning, and model export.
"""

from nexus_llm.training.trainer import Trainer, TrainingConfig
from nexus_llm.training.dataset import TextDataset, DataConfig, load_dataset
from nexus_llm.training.fine_tune import LoRAConfig, FineTuner
from nexus_llm.training.collator import DataCollator
from nexus_llm.training.scheduler import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from nexus_llm.training.callbacks import (
    EarlyStoppingCallback,
    LoggingCallback,
    CheckpointCallback,
    LearningRateTrackingCallback,
    GradientMonitoringCallback,
    CallbackManager,
)
from nexus_llm.training.metrics import MetricsTracker, Metrics
from nexus_llm.training.checkpoint import CheckpointManager
from nexus_llm.training.distributed import DistributedManager
from nexus_llm.training.augmentation import TextAugmenter
from nexus_llm.training.preprocessing import DataPreprocessor
from nexus_llm.training.evaluation import Evaluator
from nexus_llm.training.optimizer import build_optimizer, OptimizerConfig
from nexus_llm.training.loss import (
    LabelSmoothingCrossEntropy,
    FocalLoss,
    KLDivergenceLoss,
    CombinedLoss,
)
from nexus_llm.training.curriculum import CurriculumLearner, DifficultyRanker, PacingFunction
from nexus_llm.training.export import ModelExporter, ExportConfig

__all__ = [
    "Trainer",
    "TrainingConfig",
    "TextDataset",
    "DataConfig",
    "load_dataset",
    "LoRAConfig",
    "FineTuner",
    "DataCollator",
    "get_linear_schedule_with_warmup",
    "get_cosine_schedule_with_warmup",
    "get_cosine_with_restarts_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "get_constant_schedule_with_warmup",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "CheckpointCallback",
    "LearningRateTrackingCallback",
    "GradientMonitoringCallback",
    "CallbackManager",
    "MetricsTracker",
    "Metrics",
    "CheckpointManager",
    "DistributedManager",
    "TextAugmenter",
    "DataPreprocessor",
    "Evaluator",
    "build_optimizer",
    "OptimizerConfig",
    "LabelSmoothingCrossEntropy",
    "FocalLoss",
    "KLDivergenceLoss",
    "CombinedLoss",
    "CurriculumLearner",
    "DifficultyRanker",
    "PacingFunction",
    "ModelExporter",
    "ExportConfig",
]
