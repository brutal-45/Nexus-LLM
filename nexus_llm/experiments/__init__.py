"""Nexus-LLM Experiments Module.

Provides comprehensive experiment tracking, comparison, hyperparameter
optimization, and ablation study capabilities for ML experiment
management.
"""

from nexus_llm.experiments.experiment import (
    Artifact,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    MetricRecord,
    MetricType,
)
from nexus_llm.experiments.tracker import (
    ExperimentSummary,
    ExperimentTracker,
    TrackerConfig,
)
from nexus_llm.experiments.comparison import (
    ComparisonResult,
    ExperimentComparator,
    MetricComparison,
    ParamDifference,
)
from nexus_llm.experiments.hyperparams import (
    HyperparameterSearch,
    ParamSpace,
    ParamSpec,
    ParamType,
    SearchConfig,
    SearchStrategy,
    Trial,
)
from nexus_llm.experiments.ablation import (
    AblationComponent,
    AblationResult,
    AblationStatus,
    AblationStudy,
    AblationStudyConfig,
    AblationType,
    AblationVariant,
)

__all__ = [
    # Experiment
    "Artifact",
    "Experiment",
    "ExperimentConfig",
    "ExperimentStatus",
    "MetricRecord",
    "MetricType",
    # Tracker
    "ExperimentSummary",
    "ExperimentTracker",
    "TrackerConfig",
    # Comparison
    "ComparisonResult",
    "ExperimentComparator",
    "MetricComparison",
    "ParamDifference",
    # Hyperparameters
    "HyperparameterSearch",
    "ParamSpace",
    "ParamSpec",
    "ParamType",
    "SearchConfig",
    "SearchStrategy",
    "Trial",
    # Ablation
    "AblationComponent",
    "AblationResult",
    "AblationStatus",
    "AblationStudy",
    "AblationStudyConfig",
    "AblationType",
    "AblationVariant",
]
