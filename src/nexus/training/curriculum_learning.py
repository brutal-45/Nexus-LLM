"""
Curriculum Learning Strategies for Nexus LLM.

Provides multiple curriculum learning strategies that organize training data
from easy to hard, enabling models to learn more efficiently by gradually
increasing difficulty throughout training.

Classes:
    CurriculumScheduler: Base class for all curriculum schedulers.
    LinearCurriculum: Linearly increase difficulty over training.
    StepCurriculum: Step-wise difficulty at predefined milestones.
    CompetenceBasedCurriculum: Adapt difficulty based on model performance.
    SelfPacedCurriculum: Model selects samples by loss (easy first).
    BabyStepCurriculum: Gradually increase sequence length.
    MultiTaskCurriculum: Schedule multiple training objectives.
    CurriculumDataset: Wrapper dataset sampling by curriculum schedule.
    CurriculumEvaluator: Evaluate readiness for next difficulty level.
    DifficultyScorer: Score data difficulty across multiple dimensions.
"""

from __future__ import annotations

import abc
import copy
import heapq
import json
import logging
import math
import os
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class DifficultyLevel(Enum):
    """Enumeration of difficulty levels for curriculum learning."""
    TRIVIAL = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4

    @classmethod
    def from_float(cls, value: float) -> "DifficultyLevel":
        """Convert a float [0, 1] to a DifficultyLevel."""
        if value < 0.2:
            return cls.TRIVIAL
        elif value < 0.4:
            return cls.EASY
        elif value < 0.6:
            return cls.MEDIUM
        elif value < 0.8:
            return cls.HARD
        else:
            return cls.EXPERT


class CurriculumStage(Enum):
    """Stage of the curriculum training."""
    PRETRAIN = auto()
    WARMUP = auto()
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()
    FINE_TUNE = auto()


class TransitionStrategy(Enum):
    """How to transition between difficulty levels."""
    HARD = auto()
    SOFT = auto()
    LINEAR_BLEND = auto()
    RANDOM_MIX = auto()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DifficultyBounds:
    """Bounds for difficulty scoring normalization."""
    min_perplexity: float = 1.0
    max_perplexity: float = 1000.0
    min_length: int = 1
    max_length: int = 4096
    min_vocab_rarity: float = 0.0
    max_vocab_rarity: float = 1.0

    def normalize_perplexity(self, perplexity: float) -> float:
        """Normalize perplexity to [0, 1] range."""
        if self.max_perplexity <= self.min_perplexity:
            return 0.0
        normalized = (perplexity - self.min_perplexity) / (
            self.max_perplexity - self.min_perplexity
        )
        return max(0.0, min(1.0, normalized))

    def normalize_length(self, length: int) -> float:
        """Normalize sequence length to [0, 1] range."""
        if self.max_length <= self.min_length:
            return 0.0
        normalized = (length - self.min_length) / (
            self.max_length - self.min_length
        )
        return max(0.0, min(1.0, normalized))

    def normalize_vocab_rarity(self, rarity: float) -> float:
        """Normalize vocabulary rarity to [0, 1] range."""
        if self.max_vocab_rarity <= self.min_vocab_rarity:
            return 0.0
        normalized = (rarity - self.min_vocab_rarity) / (
            self.max_vocab_rarity - self.min_vocab_rarity
        )
        return max(0.0, min(1.0, normalized))


@dataclass
class CurriculumSample:
    """A single sample annotated with difficulty metadata."""
    data: Dict[str, Any]
    difficulty_score: float = 0.0
    difficulty_level: DifficultyLevel = DifficultyLevel.EASY
    domain: str = ""
    tags: List[str] = field(default_factory=list)
    perplexity: float = 0.0
    sequence_length: int = 0
    vocab_rarity: float = 0.0
    sample_id: int = -1
    epoch_seen: int = 0
    times_selected: int = 0
    loss_history: List[float] = field(default_factory=list)


@dataclass
class CurriculumState:
    """Current state of the curriculum scheduler."""
    current_step: int = 0
    current_epoch: int = 0
    total_steps: int = 0
    current_difficulty: float = 0.0
    current_level: DifficultyLevel = DifficultyLevel.TRIVIAL
    stage: CurriculumStage = CurriculumStage.PRETRAIN
    samples_processed: int = 0
    avg_loss_at_level: float = 0.0
    accuracy_at_level: float = 0.0
    ready_for_next_level: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "total_steps": self.total_steps,
            "current_difficulty": self.current_difficulty,
            "current_level": self.current_level.name,
            "stage": self.stage.name,
            "samples_processed": self.samples_processed,
            "avg_loss_at_level": self.avg_loss_at_level,
            "accuracy_at_level": self.accuracy_at_level,
            "ready_for_next_level": self.ready_for_next_level,
            "history_len": len(self.history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumState":
        """Deserialize state from dictionary."""
        state = cls(
            current_step=data.get("current_step", 0),
            current_epoch=data.get("current_epoch", 0),
            total_steps=data.get("total_steps", 0),
            current_difficulty=data.get("current_difficulty", 0.0),
            samples_processed=data.get("samples_processed", 0),
            avg_loss_at_level=data.get("avg_loss_at_level", 0.0),
            accuracy_at_level=data.get("accuracy_at_level", 0.0),
            ready_for_next_level=data.get("ready_for_next_level", False),
        )
        if "current_level" in data:
            state.current_level = DifficultyLevel[data["current_level"]]
        if "stage" in data:
            state.stage = CurriculumStage[data["stage"]]
        return state


@dataclass
class LevelMetrics:
    """Metrics collected for a single difficulty level."""
    level: DifficultyLevel
    num_samples: int = 0
    total_loss: float = 0.0
    correct_predictions: int = 0
    total_predictions: int = 0
    step_losses: List[float] = field(default_factory=list)
    start_step: int = 0
    end_step: int = 0
    start_epoch: int = 0
    end_epoch: int = 0

    @property
    def avg_loss(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.total_loss / self.num_samples

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def update(self, loss: float, correct: bool):
        """Update metrics with a new sample."""
        self.num_samples += 1
        self.total_loss += loss
        self.step_losses.append(loss)
        if correct:
            self.correct_predictions += 1
        self.total_predictions += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.name,
            "num_samples": self.num_samples,
            "avg_loss": self.avg_loss,
            "accuracy": self.accuracy,
            "start_step": self.start_step,
            "end_step": self.end_step,
        }


@dataclass
class StepMilestone:
    """A milestone defining a step in StepCurriculum."""
    step_threshold: int
    difficulty: float
    level: DifficultyLevel
    description: str = ""


# ---------------------------------------------------------------------------
# Base Curriculum Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler(abc.ABC):
    """Abstract base class for curriculum schedulers.

    All curriculum schedulers must implement step() and get_difficulty()
    to define how training difficulty changes over time.
    """

    def __init__(
        self,
        total_steps: int,
        total_epochs: int = -1,
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
        transition_strategy: TransitionStrategy = TransitionStrategy.SOFT,
        transition_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.transition_strategy = transition_strategy
        self.transition_fraction = transition_fraction
        self.seed = seed
        self.rng = random.Random(seed)
        self.state = CurriculumState(total_steps=total_steps)
        self.level_metrics: Dict[DifficultyLevel, LevelMetrics] = {}
        for level in DifficultyLevel:
            self.level_metrics[level] = LevelMetrics(level=level)
        self._callbacks: List[Callable[[CurriculumState], None]] = []

    @abc.abstractmethod
    def step(self, loss: float = 0.0, accuracy: float = 0.0, **kwargs) -> float:
        """Advance the scheduler and return the new difficulty value.

        Args:
            loss: Current training loss.
            accuracy: Current accuracy on validation set.
            **kwargs: Additional metrics.

        Returns:
            The new difficulty value in [min_difficulty, max_difficulty].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_difficulty(self) -> float:
        """Return the current difficulty level."""
        raise NotImplementedError

    def get_level(self) -> DifficultyLevel:
        """Return the current difficulty as a DifficultyLevel enum."""
        return DifficultyLevel.from_float(self.get_difficulty())

    def get_stage(self) -> CurriculumStage:
        """Return the current training stage."""
        difficulty = self.get_difficulty()
        if difficulty < 0.05:
            return CurriculumStage.PRETRAIN
        elif difficulty < 0.15:
            return CurriculumStage.WARMUP
        elif difficulty < 0.45:
            return CurriculumStage.EASY
        elif difficulty < 0.75:
            return CurriculumStage.MEDIUM
        elif difficulty < 0.95:
            return CurriculumStage.HARD
        else:
            return CurriculumStage.FINE_TUNE

    def register_callback(self, callback: Callable[[CurriculumState], None]):
        """Register a callback to be called after each step."""
        self._callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(copy.deepcopy(self.state))
            except Exception as e:
                logger.warning(f"Curriculum callback failed: {e}")

    def update_metrics(self, loss: float, correct: bool = False):
        """Update metrics for the current difficulty level."""
        level = self.get_level()
        metrics = self.level_metrics[level]
        if metrics.num_samples == 0:
            metrics.start_step = self.state.current_step
            metrics.start_epoch = self.state.current_epoch
        metrics.update(loss, correct)
        self.state.avg_loss_at_level = metrics.avg_loss
        self.state.accuracy_at_level = metrics.accuracy

    def record_history(self):
        """Record current state to history."""
        entry = {
            "step": self.state.current_step,
            "epoch": self.state.current_epoch,
            "difficulty": self.state.current_difficulty,
            "level": self.state.current_level.name,
            "stage": self.state.stage.name,
            "avg_loss": self.state.avg_loss_at_level,
            "accuracy": self.state.accuracy_at_level,
        }
        self.state.history.append(entry)

    def save_state(self, path: str):
        """Save curriculum state to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = self.state.to_dict()
        data["level_metrics"] = {
            level.name: metrics.to_dict()
            for level, metrics in self.level_metrics.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Curriculum state saved to {path}")

    def load_state(self, path: str):
        """Load curriculum state from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self.state = CurriculumState.from_dict(data)
        if "level_metrics" in data:
            for level_name, metrics_data in data["level_metrics"].items():
                level = DifficultyLevel[level_name]
                m = self.level_metrics[level]
                m.num_samples = metrics_data.get("num_samples", 0)
                m.total_loss = metrics_data.get("avg_loss", 0.0) * m.num_samples
                m.correct_predictions = int(
                    metrics_data.get("accuracy", 0.0) * metrics_data.get("total_predictions", 1)
                )
                m.total_predictions = metrics_data.get("total_predictions", 0)
                m.start_step = metrics_data.get("start_step", 0)
                m.end_step = metrics_data.get("end_step", 0)
        logger.info(f"Curriculum state loaded from {path}")

    def reset(self):
        """Reset the curriculum to its initial state."""
        self.state = CurriculumState(total_steps=self.total_steps)
        for level in DifficultyLevel:
            self.level_metrics[level] = LevelMetrics(level=level)
        logger.info("Curriculum scheduler reset")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of curriculum progress."""
        return {
            "current_step": self.state.current_step,
            "progress": self.state.current_step / max(1, self.total_steps),
            "current_difficulty": self.state.current_difficulty,
            "current_level": self.state.current_level.name,
            "current_stage": self.state.current_stage.name,
            "avg_loss": self.state.avg_loss_at_level,
            "accuracy": self.state.accuracy_at_level,
            "samples_processed": self.state.samples_processed,
            "levels_completed": sum(
                1 for m in self.level_metrics.values() if m.num_samples > 0
            ),
        }


# ---------------------------------------------------------------------------
# Linear Curriculum
# ---------------------------------------------------------------------------

class LinearCurriculum(CurriculumScheduler):
    """Linearly increase data difficulty from easy to hard over training.

    The difficulty increases linearly from min_difficulty to max_difficulty
    based on the fraction of total steps completed. Optionally includes
    a warmup phase at the beginning where difficulty stays at minimum.

    Args:
        total_steps: Total number of training steps.
        warmup_steps: Number of steps to keep difficulty at minimum.
        warmup_fraction: Fraction of total_steps for warmup (overrides warmup_steps if set).
        min_difficulty: Starting difficulty.
        max_difficulty: Target ending difficulty.
        linear_fn: Optional custom linear function (step_fraction) -> difficulty.
    """

    def __init__(
        self,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_fraction: float = 0.0,
        linear_fn: Optional[Callable[[float], float]] = None,
        **kwargs,
    ):
        super().__init__(total_steps=total_steps, **kwargs)
        self.warmup_steps = warmup_steps
        if warmup_fraction > 0:
            self.warmup_steps = max(warmup_steps, int(total_steps * warmup_fraction))
        self.linear_fn = linear_fn
        self._difficulty = self.min_difficulty

    def _compute_difficulty(self, step: int) -> float:
        """Compute difficulty at a given step."""
        if step < self.warmup_steps:
            return self.min_difficulty

        effective_step = step - self.warmup_steps
        effective_total = max(1, self.total_steps - self.warmup_steps)
        fraction = min(1.0, effective_step / effective_total)

        if self.linear_fn is not None:
            difficulty = self.linear_fn(fraction)
        else:
            difficulty = self.min_difficulty + fraction * (
                self.max_difficulty - self.min_difficulty
            )

        return max(self.min_difficulty, min(self.max_difficulty, difficulty))

    def step(self, loss: float = 0.0, accuracy: float = 0.0, **kwargs) -> float:
        """Advance one step and return current difficulty."""
        self.state.current_step += 1
        if "epoch" in kwargs:
            self.state.current_epoch = kwargs["epoch"]

        self._difficulty = self._compute_difficulty(self.state.current_step)
        self.state.current_difficulty = self._difficulty
        self.state.current_level = self.get_level()
        self.state.stage = self.get_stage()
        self.state.samples_processed += kwargs.get("batch_size", 1)

        if loss > 0:
            self.update_metrics(loss, kwargs.get("correct", False))

        if self.state.current_step % 100 == 0:
            self.record_history()

        self._notify_callbacks()
        return self._difficulty

    def get_difficulty(self) -> float:
        return self._difficulty

    def get_difficulty_at_step(self, step: int) -> float:
        """Preview difficulty at a future step without modifying state."""
        return self._compute_difficulty(step)


# ---------------------------------------------------------------------------
# Step Curriculum
# ---------------------------------------------------------------------------

class StepCurriculum(CurriculumScheduler):
    """Step-wise difficulty increases at predefined milestones.

    Instead of a smooth linear increase, difficulty jumps to new levels
    at specific step thresholds. This allows the model to fully train
    on one difficulty level before moving to the next.

    Args:
        total_steps: Total training steps.
        milestones: List of StepMilestone defining when to increase difficulty.
                    If None, uses default milestones.
        repeat_last_level: If True, after all milestones, keep last difficulty.
    """

    def __init__(
        self,
        total_steps: int,
        milestones: Optional[List[StepMilestone]] = None,
        repeat_last_level: bool = True,
        **kwargs,
    ):
        super().__init__(total_steps=total_steps, **kwargs)
        self.repeat_last_level = repeat_last_level
        if milestones is not None:
            self.milestones = sorted(milestones, key=lambda m: m.step_threshold)
        else:
            self.milestones = self._default_milestones()
        self._current_milestone_idx = 0
        self._difficulty = self.milestones[0].difficulty if self.milestones else self.min_difficulty

    def _default_milestones(self) -> List[StepMilestone]:
        """Create default step milestones evenly spaced across training."""
        num_steps = len(DifficultyLevel)
        step_interval = self.total_steps // (num_steps + 1)
        milestones = []
        for i, level in enumerate(DifficultyLevel):
            step_threshold = step_interval * (i + 1)
            difficulty = (i + 1) / (num_steps + 1)
            milestones.append(
                StepMilestone(
                    step_threshold=step_threshold,
                    difficulty=difficulty,
                    level=level,
                    description=f"Transition to {level.name} at step {step_threshold}",
                )
            )
        return milestones

    def _find_current_milestone(self, step: int) -> int:
        """Find the index of the milestone that should be active at given step."""
        idx = 0
        for i, milestone in enumerate(self.milestones):
            if step >= milestone.step_threshold:
                idx = i
            else:
                break
        return idx

    def step(self, loss: float = 0.0, accuracy: float = 0.0, **kwargs) -> float:
        """Advance one step and return current difficulty."""
        self.state.current_step += 1
        if "epoch" in kwargs:
            self.state.current_epoch = kwargs["epoch"]

        prev_idx = self._current_milestone_idx
        self._current_milestone_idx = self._find_current_milestone(self.state.current_step)

        if self._current_milestone_idx < len(self.milestones):
            milestone = self.milestones[self._current_milestone_idx]
            self._difficulty = milestone.difficulty
            if self._current_milestone_idx != prev_idx:
                logger.info(
                    f"Step curriculum: reaching {milestone.description} "
                    f"(difficulty={self._difficulty:.3f})"
                )
        elif self.repeat_last_level and self.milestones:
            self._difficulty = self.milestones[-1].difficulty
        else:
            self._difficulty = self.max_difficulty

        self._difficulty = max(self.min_difficulty, min(self.max_difficulty, self._difficulty))
        self.state.current_difficulty = self._difficulty
        self.state.current_level = self.get_level()
        self.state.stage = self.get_stage()
        self.state.samples_processed += kwargs.get("batch_size", 1)

        if loss > 0:
            self.update_metrics(loss, kwargs.get("correct", False))

        if self.state.current_step % 100 == 0:
            self.record_history()

        self._notify_callbacks()
        return self._difficulty

    def get_difficulty(self) -> float:
        return self._difficulty

    def add_milestone(self, milestone: StepMilestone):
        """Add a new milestone dynamically."""
        self.milestones.append(milestone)
        self.milestones.sort(key=lambda m: m.step_threshold)

    def remove_milestone(self, step_threshold: int):
        """Remove a milestone by its step threshold."""
        self.milestones = [
            m for m in self.milestones if m.step_threshold != step_threshold
        ]


# ---------------------------------------------------------------------------
# Competence-Based Curriculum
# ---------------------------------------------------------------------------

class CompetenceBasedCurriculum(CurriculumScheduler):
    """Adapt difficulty based on model accuracy/performance on current level.

    The model's competence is estimated from its accuracy (or inverse loss)
    on the current difficulty level. When competence exceeds a threshold,
    difficulty increases. This ensures the model is ready before advancing.

    Args:
        total_steps: Total training steps.
        competence_threshold: Accuracy threshold to advance to next level.
        patience: Steps to wait at threshold before advancing.
        smoothing: EMA smoothing factor for competence estimation.
        regression_tolerance: Allowed accuracy drop before reverting difficulty.
        revert_steps: Steps of poor performance before reverting.
    """

    def __init__(
        self,
        total_steps: int,
        competence_threshold: float = 0.75,
        patience: int = 100,
        smoothing: float = 0.9,
        regression_tolerance: float = 0.1,
        revert_steps: int = 500,
        **kwargs,
    ):
        super().__init__(total_steps=total_steps, **kwargs)
        self.competence_threshold = competence_threshold
        self.patience = patience
        self.smoothing = smoothing
        self.regression_tolerance = regression_tolerance
        self.revert_steps = revert_steps
        self._competence = 0.0
        self._difficulty = self.min_difficulty
        self._patience_counter = 0
        self._regression_counter = 0
        self._peak_competence = 0.0
        self._competence_history: deque = deque(maxlen=1000)
        self._difficulty_adjustments: List[Dict[str, Any]] = []

    def _estimate_competence(self, loss: float, accuracy: float) -> float:
        """Estimate model competence from loss and accuracy."""
        if accuracy > 0:
            competence = accuracy
        elif loss > 0:
            competence = 1.0 / (1.0 + loss)
        else:
            competence = self._competence

        smoothed = (
            self.smoothing * self._competence + (1 - self.smoothing) * competence
        )
        return smoothed

    def _should_advance(self) -> bool:
        """Check if the model is ready to advance to next difficulty."""
        if self._patience_counter >= self.patience:
            return True
        if self._competence >= self.competence_threshold:
            self._patience_counter += 1
            if self._patience_counter >= self.patience:
                return True
        else:
            self._patience_counter = 0
        return False

    def _should_revert(self) -> bool:
        """Check if the model should revert to a lower difficulty."""
        if len(self._competence_history) < 50:
            return False
        recent = list(self._competence_history)[-50:]
        recent_avg = sum(recent) / len(recent)
        if self._peak_competence > 0 and (self._peak_competence - recent_avg) > self.regression_tolerance:
            self._regression_counter += 1
            if self._regression_counter >= self.revert_steps:
                return True
        else:
            self._regression_counter = max(0, self._regression_counter - 1)
        return False

    def _adjust_difficulty(self, delta: float):
        """Adjust difficulty by a delta, clamped to valid range."""
        old_difficulty = self._difficulty
        self._difficulty = max(
            self.min_difficulty, min(self.max_difficulty, self._difficulty + delta)
        )
        if abs(self._difficulty - old_difficulty) > 1e-6:
            self._difficulty_adjustments.append({
                "step": self.state.current_step,
                "old_difficulty": old_difficulty,
                "new_difficulty": self._difficulty,
                "competence": self._competence,
                "direction": "advance" if delta > 0 else "revert",
            })
            logger.info(
                f"Competence curriculum: difficulty {old_difficulty:.3f} -> "
                f"{self._difficulty:.3f} (competence={self._competence:.3f})"
            )

    def step(self, loss: float = 0.0, accuracy: float = 0.0, **kwargs) -> float:
        """Advance one step and adjust difficulty based on competence."""
        self.state.current_step += 1
        if "epoch" in kwargs:
            self.state.current_epoch = kwargs["epoch"]

        self._competence = self._estimate_competence(loss, accuracy)
        self._competence_history.append(self._competence)
        self._peak_competence = max(self._peak_competence, self._competence)

        difficulty_step = (self.max_difficulty - self.min_difficulty) * 0.02

        if self._should_advance():
            self._adjust_difficulty(difficulty_step)
            self._patience_counter = 0

        if self._should_revert():
            self._adjust_difficulty(-difficulty_step * 1.5)
            self._regression_counter = 0
            self._peak_competence = self._competence

        progress = self.state.current_step / max(1, self.total_steps)
        linear_floor = self.min_difficulty + progress * (
            self.max_difficulty - self.min_difficulty
        ) * 0.5
        self._difficulty = max(self._difficulty, linear_floor)

        self.state.current_difficulty = self._difficulty
        self.state.current_level = self.get_level()
        self.state.stage = self.get_stage()
        self.state.avg_loss_at_level = loss if loss > 0 else self.state.avg_loss_at_level
        self.state.accuracy_at_level = accuracy if accuracy > 0 else self.state.accuracy_at_level
        self.state.samples_processed += kwargs.get("batch_size", 1)

        if self.state.current_step % 100 == 0:
            self.record_history()

        self._notify_callbacks()
        return self._difficulty

    def get_difficulty(self) -> float:
        return self._difficulty

    def get_competence(self) -> float:
        """Return the current estimated competence."""
        return self._competence

    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """Return history of difficulty adjustments."""
        return list(self._difficulty_adjustments)


# ---------------------------------------------------------------------------
# Self-Paced Curriculum
# ---------------------------------------------------------------------------

class SelfPacedCurriculum(CurriculumScheduler):
    """Model selects training samples based on loss (easy samples first).

    In self-paced learning, the model trains on the easiest samples first,
    then gradually incorporates harder samples as training progresses.
    The difficulty of each sample is determined by the model's own loss
    on that sample.

    Args:
        total_steps: Total training steps.
        initial_threshold: Initial loss threshold (include all samples).
        threshold_decay: Rate at which threshold decreases (harder samples added).
        min_threshold: Minimum threshold (include all samples).
        lambda_reg: Regularization weight for self-paced formulation.
        use_ema: Use exponential moving average for loss estimation.
    """

    def __init__(
        self,
        total_steps: int,
        initial_threshold: float = 100.0,
        threshold_decay: float = 0.995,
        min_threshold: float = 0.1,
        lambda_reg: float = 0.5,
        use_ema: bool = True,
        **kwargs,
    ):
        super().__init__(total_steps=total_steps, **kwargs)
        self.initial_threshold = initial_threshold
        self.threshold_decay = threshold_decay
        self.min_threshold = min_threshold
        self.lambda_reg = lambda_reg
        self.use_ema = use_ema
        self._threshold = initial_threshold
        self._difficulty = self.min_difficulty
        self._sample_losses: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self._ema_losses: Dict[int, float] = {}
        self._inclusion_rate = 1.0

    def update_sample_loss(self, sample_id: int, loss: float):
        """Update the stored loss for a training sample."""
        self._sample_losses[sample_id].append(loss)
        avg_loss = sum(self._sample_losses[sample_id]) / len(self._sample_losses[sample_id])
        if sample_id in self._ema_losses:
            self._ema_losses[sample_id] = 0.9 * self._ema_losses[sample_id] + 0.1 * avg_loss
        else:
            self._ema_losses[sample_id] = avg_loss

    def get_sample_weight(self, sample_id: int) -> float:
        """Get the self-paced weight for a sample.

        Weight = 1 if loss < threshold, 0 otherwise.
        With soft weighting: weight = max(0, 1 - loss / threshold).
        """
        if sample_id not in self._ema_losses:
            return 1.0
        loss = self._ema_losses[sample_id]
        if self.use_ema:
            weight = max(0.0, 1.0 - loss / max(self._threshold, 1e-8))
        else:
            weight = 1.0 if loss < self._threshold else 0.0
        return weight

    def get_included_sample_ids(self, all_ids: List[int]) -> List[int]:
        """Return IDs of samples that should be included based on current threshold."""
        included = []
        for sid in all_ids:
            if self.get_sample_weight(sid) > 0.01:
                included.append(sid)
        return included

    def _compute_difficulty_from_inclusion(self):
        """Compute difficulty based on fraction of samples included."""
        if not self._sample_losses:
            return self.min_difficulty
        total = len(self._sample_losses)
        included = sum(
            1 for sid in self._sample_losses if self.get_sample_weight(sid) > 0.01
        )
        self._inclusion_rate = included / max(1, total)
        difficulty = 1.0 - self._inclusion_rate
        return max(self.min_difficulty, min(self.max_difficulty, difficulty))

    def step(self, loss: float = 0.0, accuracy: float = 0.0, **kwargs) -> float:
        """Advance one step and update the self-paced threshold."""
        self.state.current_step += 1
        if "epoch" in kwargs:
            self.state.current_epoch = kwargs["epoch"]

        self._threshold = max(
            self.min_threshold, self._threshold * self.threshold_decay
        )
        self._difficulty = self._compute_difficulty_from_inclusion()

        self.state.current_difficulty = self._difficulty
        self.state.current_level = self.get_level()
        self.state.stage = self.get_stage()
        self.state.samples_processed += kwargs.get("batch_size", 1)

        if self.state.current_step % 100 == 0:
            self.record_history()

        self._notify_callbacks()
        return self._difficulty

    def get_difficulty(self) -> float:
        return self._difficulty

    def get_threshold(self) -> float:
        """Return the current loss threshold."""
        return self._threshold

    def get_inclusion_rate(self) -> float:
        """Return the fraction of samples currently included."""
        return self._inclusion_rate

    def get_loss_statistics(self) -> Dict[str, float]:
        """Return statistics about sample losses."""
        if not self._ema_losses:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}
        losses = list(self._ema_losses.values())
        return {
            "mean": sum(losses) / len(losses),
            "median": sorted(losses)[len(losses) // 2],
            "max": max(losses),
            "min": min(losses),
            "count": len(losses),
        }


# ---------------------------------------------------------------------------
# Baby Step Curriculum
# ---------------------------------------------------------------------------

class BabyStepCurriculum(CurriculumScheduler):
    """Gradually increase sequence length during training.

    Inspired by the "Baby Steps" paper, this scheduler starts with very
    short sequences and gradually increases the maximum length. This
    allows the model to learn basic patterns before dealing with long-range
    dependencies.

    Args:
        total_steps: Total training steps.
        initial_length: Starting sequence length.
        target_length: Target maximum sequence length.
        growth_fn: How to grow sequence length ('linear', 'exponential', 'sqrt').
        length_bucket_size: Round to nearest bucket for caching efficiency.
        pad_to_max: Pad sequences to current max length.
    """

    def __init__(
        self,
        total_steps: int,
        initial_length: int = 16,
        target_length: int = 4096,
        growth_fn: str = "linear",
        length_bucket_size: int = 16,
        pad_to_max: bool = True,
        **kwargs,
    ):
        super().__init__(total_steps=total_steps, **kwargs)
        self.initial_length = initial_length
        self.target_length = target_length
        self.growth_fn = growth_fn
        self.length_bucket_size = length_bucket_size
        self.pad_to_max = pad_to_max
        self._current_length = initial_length
        self._difficulty = self.min_difficulty
        self._length_history: List[Tuple[int, int]] = []

    def _compute_length(self, step: int) -> int:
        """Compute the sequence length at a given step."""
        progress = min(1.0, step / max(1, self.total_steps))
        if self.growth_fn == "linear":
            length = self.initial_length + progress * (
                self.target_length - self.initial_length
            )
        elif self.growth_fn == "exponential":
            log_initial = math.log2(max(1, self.initial_length))
            log_target = math.log2(max(1, self.target_length))
            log_length = log_initial + progress * (log_target - log_initial)
            length = 2 ** log_length
        elif self.growth_fn == "sqrt":
            length = self.initial_length + math.sqrt(progress) * (
                self.target_length - self.initial_length
            )
        elif self.growth_fn == "log":
            if progress == 0:
                length = self.initial_length
            else:
                length = self.initial_length + (
                    math.log(1 + progress * 9) / math.log(10)
                ) * (self.target_length - self.initial_length)
        else:
            length = self.initial_length + progress * (
                self.target_length - self.initial_length
            )

        if self.length_bucket_size > 1:
            length = max(
                self.initial_length,
                int(math.ceil(length / self.length_bucket_size)) * self.length_bucket_size,
            )
        return min(int(length), self.target_length)

    def step(self, loss: float = 0.0, accuracy: float = 0.0, **kwargs) -> float:
        """Advance one step and update sequence length."""
        self.state.current_step += 1
        if "epoch" in kwargs:
            self.state.current_epoch = kwargs["epoch"]

        self._current_length = self._compute_length(self.state.current_step)
        length_progress = (self._current_length - self.initial_length) / max(
            1, self.target_length - self.initial_length
        )
        self._difficulty = self.min_difficulty + length_progress * (
            self.max_difficulty - self.min_difficulty
        )

        if self._length_history and self._length_history[-1][1] != self._current_length:
            self._length_history.append(
                (self.state.current_step, self._current_length)
            )
            logger.info(
                f"Babystep: sequence length -> {self._current_length} "
                f"at step {self.state.current_step}"
            )

        self.state.current_difficulty = self._difficulty
        self.state.current_level = self.get_level()
        self.state.stage = self.get_stage()
        self.state.samples_processed += kwargs.get("batch_size", 1)

        if self.state.current_step % 100 == 0:
            self.record_history()

        self._notify_callbacks()
        return self._difficulty

    def get_difficulty(self) -> float:
        return self._difficulty

    def get_current_length(self) -> int:
        """Return the current maximum sequence length."""
        return self._current_length

    def truncate_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Truncate a sequence to the current maximum length."""
        if sequence.size(-1) > self._current_length:
            return sequence[..., : self._current_length]
        return sequence

    def pad_sequence(self, sequence: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
        """Pad a sequence to the current maximum length if needed."""
        if self.pad_to_max and sequence.size(-1) < self._current_length:
            pad_size = self._current_length - sequence.size(-1)
            pad_shape = list(sequence.shape)
            pad_shape[-1] = pad_size
            padding = torch.full(pad_shape, pad_value, dtype=sequence.dtype, device=sequence.device)
            return torch.cat([sequence, padding], dim=-1)
        return sequence


# ---------------------------------------------------------------------------
# Multi-Task Curriculum
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Configuration for a single task in multi-task curriculum."""
    name: str
    weight: float = 1.0
    difficulty: float = 0.5
    start_step: int = 0
    end_step: int = -1
    warmup_steps: int = 0
    cooldown_steps: int = 0
    min_weight: float = 0.0
    max_weight: float = 1.0
    schedule: str = "constant"
    data_fraction: float = 1.0
    is_active: bool = True


class MultiTaskCurriculum(CurriculumScheduler):
    """Schedule multiple training objectives over the course of training.

    Manages when different tasks are active, their relative weights, and
    how to schedule transitions between tasks for optimal learning.

    Args:
        total_steps: Total training steps.
        tasks: List of TaskConfig for each training objective.
        default_weight: Default weight for tasks without explicit scheduling.
        cycle_length: If > 0, cycle through tasks with this period.
        annealing: Whether to anneal task weights over time.
    """

    def __init__(
        self,
        total_steps: int,
        tasks: Optional[List[TaskConfig]] = None,
        default_weight: float = 1.0,
        cycle_length: int = 0,
        annealing: bool = False,
        **kwargs,
    ):
        super().__init__(total_steps=total_steps, **kwargs)
        self.default_weight = default_weight
        self.cycle_length = cycle_length
        self.annealing = annealing
        self._tasks: List[TaskConfig] = tasks or []
        self._task_weights: Dict[str, float] = {}
        self._difficulty = self.min_difficulty
        self._task_histories: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    def add_task(self, task: TaskConfig):
        """Add a new task to the curriculum."""
        self._tasks.append(task)
        self._task_weights[task.name] = task.weight
        logger.info(f"MultiTask: added task '{task.name}' (weight={task.weight})")

    def remove_task(self, name: str):
        """Remove a task from the curriculum."""
        self._tasks = [t for t in self._tasks if t.name != name]
        self._task_weights.pop(name, None)
        logger.info(f"MultiTask: removed task '{name}'")

    def _get_task_weight(self, task: TaskConfig, step: int) -> float:
        """Compute the current weight for a task."""
        if task.end_step > 0 and step > task.end_step:
            return task.min_weight
        if step < task.start_step:
            return task.min_weight

        effective_step = step - task.start_step
        effective_end = (
            task.end_step - task.start_step if task.end_step > 0
            else self.total_steps - task.start_step
        )

        if task.schedule == "constant":
            weight = task.weight
        elif task.schedule == "linear_warmup":
            if task.warmup_steps > 0 and effective_step < task.warmup_steps:
                weight = task.min_weight + (
                    (task.weight - task.min_weight) * effective_step / task.warmup_steps
                )
            else:
                weight = task.weight
        elif task.schedule == "cosine":
            progress = effective_step / max(1, effective_end)
            weight = task.min_weight + 0.5 * (task.weight - task.min_weight) * (
                1 + math.cos(math.pi * progress)
            )
        elif task.schedule == "inverse_sqrt":
            if effective_step == 0:
                weight = task.weight
            else:
                weight = task.weight / math.sqrt(1 + effective_step / 1000)
        elif task.schedule == "triangular":
            progress = effective_step / max(1, effective_end)
            weight = task.min_weight + (task.weight - task.min_weight) * (
                1 - abs(2 * progress - 1)
            )
        else:
            weight = task.weight

        if task.cooldown_steps > 0 and effective_step > (effective_end - task.cooldown_steps):
            remaining = effective_end - effective_step
            cooldown_frac = remaining / task.cooldown_steps
            weight = task.min_weight + (weight - task.min_weight) * cooldown_frac

        if self.annealing:
            global_progress = step / max(1, self.total_steps)
            weight *= (1.0 - 0.5 * global_progress)

        weight = max(task.min_weight, min(task.max_weight, weight))
        return weight if task.is_active else task.min_weight

    def _compute_difficulty(self) -> float:
        """Compute overall difficulty from active task difficulties."""
        active_tasks = [
            t for t in self._tasks
            if t.is_active and self._task_weights.get(t.name, 0) > 0.01
        ]
        if not active_tasks:
            return self.min_difficulty
        avg_difficulty = sum(t.difficulty for t in active_tasks) / len(active_tasks)
        return avg_difficulty

    def step(self, loss: float = 0.0, accuracy: float = 0.0, **kwargs) -> float:
        """Advance one step and update task weights."""
        self.state.current_step += 1
        if "epoch" in kwargs:
            self.state.current_epoch = kwargs["epoch"]

        for task in self._tasks:
            weight = self._get_task_weight(task, self.state.current_step)
            self._task_weights[task.name] = weight
            self._task_histories[task.name].append(
                (self.state.current_step, weight)
            )

        if self.cycle_length > 0:
            cycle_pos = self.state.current_step % self.cycle_length
            cycle_frac = cycle_pos / self.cycle_length
            self._difficulty = self.min_difficulty + cycle_frac * (
                self.max_difficulty - self.min_difficulty
            )
        else:
            self._difficulty = self._compute_difficulty()

        self.state.current_difficulty = self._difficulty
        self.state.current_level = self.get_level()
        self.state.stage = self.get_stage()
        self.state.samples_processed += kwargs.get("batch_size", 1)

        if self.state.current_step % 100 == 0:
            self.record_history()

        self._notify_callbacks()
        return self._difficulty

    def get_difficulty(self) -> float:
        return self._difficulty

    def get_task_weight(self, task_name: str) -> float:
        """Get the current weight for a specific task."""
        return self._task_weights.get(task_name, self.default_weight)

    def get_all_weights(self) -> Dict[str, float]:
        """Get current weights for all tasks."""
        return dict(self._task_weights)

    def get_active_tasks(self) -> List[str]:
        """Return names of currently active tasks."""
        return [
            t.name for t in self._tasks
            if t.is_active and self._task_weights.get(t.name, 0) > 0.01
        ]

    def sample_task(self) -> str:
        """Sample a task based on current weights for this training step."""
        active = self.get_active_tasks()
        if not active:
            return self._tasks[0].name if self._tasks else "default"
        weights = [self._task_weights.get(t, 0) for t in active]
        total = sum(weights)
        if total == 0:
            return self.rng.choice(active)
        probs = [w / total for w in weights]
        return self.rng.choices(active, weights=probs, k=1)[0]

    def get_task_schedule_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of task schedules."""
        summary = []
        for task in self._tasks:
            current_weight = self._task_weights.get(task.name, 0)
            summary.append({
                "name": task.name,
                "current_weight": current_weight,
                "difficulty": task.difficulty,
                "schedule": task.schedule,
                "is_active": task.is_active,
                "start_step": task.start_step,
                "end_step": task.end_step,
            })
        return summary


# ---------------------------------------------------------------------------
# Difficulty Scorer
# ---------------------------------------------------------------------------

class DifficultyScorer:
    """Score data difficulty across multiple dimensions.

    Combines perplexity, sequence length, vocabulary rarity, syntactic
    complexity, and other signals into a unified difficulty score.

    Args:
        weights: Dict mapping feature name to its weight in the composite score.
        bounds: DifficultyBounds for normalization.
        perplexity_model: Optional model for computing perplexity.
        vocab_frequencies: Optional Counter of vocabulary frequencies.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        bounds: Optional[DifficultyBounds] = None,
        vocab_frequencies: Optional[Counter] = None,
    ):
        self.weights = weights or {
            "perplexity": 0.3,
            "length": 0.2,
            "vocab_rarity": 0.2,
            "syntactic_complexity": 0.15,
            "semantic_complexity": 0.15,
        }
        self.bounds = bounds or DifficultyBounds()
        self.vocab_frequencies = vocab_frequencies or Counter()
        self._total_vocab = sum(self.vocab_frequencies.values()) if self.vocab_frequencies else 1
        self._cache: Dict[int, float] = {}

    def score_perplexity(self, text: str) -> float:
        """Score difficulty based on estimated perplexity.

        Uses simple heuristic perplexity estimation based on word length
        variance, uncommon character sequences, and structural complexity.
        """
        if not text:
            return self.bounds.min_perplexity

        words = text.split()
        if not words:
            return self.bounds.min_perplexity

        avg_word_len = sum(len(w) for w in words) / len(words)
        word_len_var = sum((len(w) - avg_word_len) ** 2 for w in words) / len(words)
        unique_ratio = len(set(words)) / len(words) if words else 0

        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(1, len(text))

        sentence_lengths = [len(s.split()) for s in re.split(r'[.!?]', text) if s.strip()]
        if sentence_lengths:
            avg_sent_len = sum(sentence_lengths) / len(sentence_lengths)
            sent_len_var = sum((l - avg_sent_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        else:
            avg_sent_len = len(words)
            sent_len_var = 0

        perplexity_estimate = (
            1.0
            + word_len_var * 0.1
            + (1 - unique_ratio) * 5.0
            + special_ratio * 10.0
            + sent_len_var * 0.05
            + avg_sent_len * 0.01
        )
        perplexity_estimate = max(self.bounds.min_perplexity, perplexity_estimate)
        return self.bounds.normalize_perplexity(perplexity_estimate)

    def score_length(self, text: str) -> float:
        """Score difficulty based on sequence length."""
        length = len(text.split())
        return self.bounds.normalize_length(length)

    def score_vocab_rarity(self, text: str) -> float:
        """Score difficulty based on vocabulary rarity."""
        if not self.vocab_frequencies or not text:
            return 0.0

        words = text.lower().split()
        if not words:
            return 0.0

        rarities = []
        for word in words:
            freq = self.vocab_frequencies.get(word, 0)
            if freq > 0:
                rarity = -math.log(freq / self._total_vocab)
            else:
                rarity = 10.0
            rarities.append(rarity)

        avg_rarity = sum(rarities) / len(rarities)
        max_rarity = 10.0
        normalized = min(1.0, avg_rarity / max_rarity)
        return self.bounds.normalize_vocab_rarity(normalized)

    def score_syntactic_complexity(self, text: str) -> float:
        """Score difficulty based on syntactic complexity."""
        if not text:
            return 0.0

        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        if not sentences:
            return 0.0

        parentheticals = text.count('(') + text.count('[') + text.count('{')
        quotes = text.count('"') + text.count("'")
        conjunctions = sum(
            1 for w in text.lower().split()
            if w in {"and", "but", "or", "however", "therefore", "moreover", "furthermore", "although", "because", "since", "while", "whereas"}
        )
        subordinate_markers = sum(
            1 for w in text.lower().split()
            if w in {"which", "that", "who", "whom", "whose", "where", "when", "if", "unless", "until", "before", "after", "during"}
        )

        avg_words_per_sentence = len(text.split()) / max(1, len(sentences))
        depth_estimate = (
            parentheticals * 0.3
            + quotes * 0.1
            + conjunctions * 0.15
            + subordinate_markers * 0.2
            + avg_words_per_sentence * 0.01
        )
        return min(1.0, depth_estimate / 10.0)

    def score_semantic_complexity(self, text: str) -> float:
        """Score difficulty based on semantic complexity."""
        if not text:
            return 0.0

        words = text.lower().split()
        if not words:
            return 0.0

        abstract_words = sum(
            1 for w in words
            if w in {
                "theory", "concept", "abstract", "principle", "fundamental",
                "meta", "paradigm", "framework", "architecture", "mechanism",
                "process", "system", "structure", "function", "property",
                "therefore", "consequently", "nevertheless", "notwithstanding",
                "hence", "thus", "accordingly", "moreover", "furthermore",
                "nevertheless", "nonetheless", "notwithstanding", "albeit",
            }
        )
        technical_terms = sum(
            1 for w in words
            if any(c.isdigit() for c in w) or w.endswith("tion") or w.endswith("ment") or w.endswith("ness") or w.endswith("ity")
        )
        negation_count = sum(1 for w in words if w in {"not", "no", "never", "none", "neither", "nor", "cannot", "without"})

        unique_ratio = len(set(words)) / max(1, len(words))
        avg_word_length = sum(len(w) for w in words) / max(1, len(words))

        complexity = (
            abstract_words * 0.15
            + technical_terms * 0.1
            + negation_count * 0.2
            + (1 - unique_ratio) * 2.0
            + avg_word_length * 0.03
        )
        return min(1.0, complexity / 5.0)

    def compute_score(self, text: str, sample_id: Optional[int] = None) -> float:
        """Compute the composite difficulty score for a text.

        Args:
            text: Input text to score.
            sample_id: Optional ID for caching.

        Returns:
            Composite difficulty score in [0, 1].
        """
        if sample_id is not None and sample_id in self._cache:
            return self._cache[sample_id]

        perplexity_score = self.score_perplexity(text)
        length_score = self.score_length(text)
        vocab_score = self.score_vocab_rarity(text)
        syntactic_score = self.score_syntactic_complexity(text)
        semantic_score = self.score_semantic_complexity(text)

        scores = {
            "perplexity": perplexity_score,
            "length": length_score,
            "vocab_rarity": vocab_score,
            "syntactic_complexity": syntactic_score,
            "semantic_complexity": semantic_score,
        }

        composite = 0.0
        total_weight = 0.0
        for feature, score in scores.items():
            w = self.weights.get(feature, 0.0)
            composite += w * score
            total_weight += w

        if total_weight > 0:
            composite /= total_weight

        composite = max(0.0, min(1.0, composite))

        if sample_id is not None:
            self._cache[sample_id] = composite

        return composite

    def compute_detailed_scores(self, text: str) -> Dict[str, float]:
        """Compute and return all individual difficulty scores."""
        return {
            "perplexity": self.score_perplexity(text),
            "length": self.score_length(text),
            "vocab_rarity": self.score_vocab_rarity(text),
            "syntactic_complexity": self.score_syntactic_complexity(text),
            "semantic_complexity": self.score_semantic_complexity(text),
            "composite": self.compute_score(text),
        }

    def clear_cache(self):
        """Clear the score cache."""
        self._cache.clear()

    def set_vocab_frequencies(self, frequencies: Counter):
        """Update vocabulary frequency distribution."""
        self.vocab_frequencies = frequencies
        self._total_vocab = sum(self.vocab_frequencies.values())

    @classmethod
    def from_corpus(cls, texts: List[str], **kwargs) -> "DifficultyScorer":
        """Create a DifficultyScorer pre-fitted on a corpus."""
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            word_freq.update(words)
        return cls(vocab_frequencies=word_freq, **kwargs)


# ---------------------------------------------------------------------------
# Curriculum Evaluator
# ---------------------------------------------------------------------------

class CurriculumEvaluator:
    """Evaluate if model is ready for the next difficulty level.

    Monitors training metrics and determines when the model has
    sufficiently mastered the current difficulty to advance.

    Args:
        accuracy_threshold: Accuracy needed to advance.
        loss_stability_window: Window to check loss stability.
        loss_stability_threshold: Max coefficient of variation for stability.
        min_samples: Minimum samples before evaluating.
        eval_frequency: How often to evaluate (in steps).
        regression_detection: Whether to detect performance regression.
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.70,
        loss_stability_window: int = 200,
        loss_stability_threshold: float = 0.15,
        min_samples: int = 500,
        eval_frequency: int = 100,
        regression_detection: bool = True,
    ):
        self.accuracy_threshold = accuracy_threshold
        self.loss_stability_window = loss_stability_window
        self.loss_stability_threshold = loss_stability_threshold
        self.min_samples = min_samples
        self.eval_frequency = eval_frequency
        self.regression_detection = regression_detection
        self._losses: deque = deque(maxlen=loss_stability_window * 2)
        self._accuracies: deque = deque(maxlen=loss_stability_window * 2)
        self._evaluations: List[Dict[str, Any]] = []
        self._level_start_losses: List[float] = []
        self._current_level_samples = 0
        self._best_accuracy = 0.0
        self._best_loss = float("inf")

    def record(self, loss: float, accuracy: float, step: int):
        """Record a training observation."""
        self._losses.append(loss)
        if accuracy >= 0:
            self._accuracies.append(accuracy)
        self._current_level_samples += 1
        self._best_accuracy = max(self._best_accuracy, accuracy)
        self._best_loss = min(self._best_loss, loss)

    def reset_level(self):
        """Reset tracking for a new difficulty level."""
        self._level_start_losses = list(self._losses)[-20:] if self._losses else []
        self._current_level_samples = 0
        self._best_accuracy = 0.0
        self._best_loss = float("inf")

    def check_loss_stability(self) -> bool:
        """Check if loss has stabilized (low coefficient of variation)."""
        if len(self._losses) < self.loss_stability_window:
            return False
        recent = list(self._losses)[-self.loss_stability_window:]
        mean = sum(recent) / len(recent)
        if mean == 0:
            return True
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        std = math.sqrt(variance)
        cv = std / abs(mean)
        return cv < self.loss_stability_threshold

    def check_accuracy(self) -> bool:
        """Check if accuracy exceeds the threshold."""
        if len(self._accuracies) < 20:
            return False
        recent = list(self._accuracies)[-50:]
        avg = sum(recent) / len(recent)
        return avg >= self.accuracy_threshold

    def check_regression(self) -> bool:
        """Check for performance regression."""
        if not self.regression_detection:
            return False
        if len(self._losses) < 100:
            return False
        window = min(100, len(self._losses))
        recent = list(self._losses)[-window:]
        older = list(self._losses)[-2 * window:-window] if len(self._losses) >= 2 * window else recent

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if older_avg == 0:
            return False
        regression_ratio = (recent_avg - older_avg) / older_avg
        return regression_ratio > 0.2

    def check_improvement(self) -> bool:
        """Check if the model has improved since the start of the level."""
        if not self._level_start_losses:
            return True
        start_avg = sum(self._level_start_losses) / len(self._level_start_losses)
        recent = list(self._losses)[-20:] if self._losses else self._level_start_losses
        recent_avg = sum(recent) / len(recent)
        if start_avg == 0:
            return recent_avg == 0
        improvement = (start_avg - recent_avg) / start_avg
        return improvement > 0.1

    def should_advance(self, step: int) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate whether the model should advance to the next level.

        Returns:
            Tuple of (should_advance, evaluation_details).
        """
        if self._current_level_samples < self.min_samples:
            return False, {"reason": "insufficient_samples"}

        if step % self.eval_frequency != 0:
            return False, {"reason": "not_eval_step"}

        loss_stable = self.check_loss_stability()
        accuracy_ok = self.check_accuracy()
        regression = self.check_regression()
        improved = self.check_improvement()

        details = {
            "loss_stable": loss_stable,
            "accuracy_ok": accuracy_ok,
            "regression": regression,
            "improved": improved,
            "current_samples": self._current_level_samples,
            "best_accuracy": self._best_accuracy,
            "best_loss": self._best_loss,
            "avg_loss": sum(list(self._losses)[-50:]) / max(1, min(50, len(self._losses))),
        }

        if regression:
            should = False
            details["reason"] = "regression_detected"
        elif accuracy_ok and loss_stable:
            should = True
            details["reason"] = "ready"
        elif accuracy_ok and improved:
            should = True
            details["reason"] = "improving_and_accurate"
        else:
            should = False
            details["reason"] = "not_ready"

        self._evaluations.append({"step": step, **details})
        return should, details

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a summary of all evaluations."""
        if not self._evaluations:
            return {"total_evaluations": 0}
        return {
            "total_evaluations": len(self._evaluations),
            "advancements": sum(1 for e in self._evaluations if e.get("reason") == "ready"),
            "regressions": sum(1 for e in self._evaluations if e.get("reason") == "regression_detected"),
            "last_evaluation": self._evaluations[-1] if self._evaluations else None,
        }


# ---------------------------------------------------------------------------
# Curriculum Dataset
# ---------------------------------------------------------------------------

class CurriculumDataset(Dataset):
    """Wrapper dataset that samples based on curriculum schedule.

    Annotates each sample with difficulty and uses the curriculum scheduler
    to determine which samples to include at each training step.

    Args:
        base_dataset: The underlying dataset.
        scheduler: Curriculum scheduler controlling difficulty.
        difficulty_scorer: Optional scorer for automatic difficulty annotation.
        text_field: Name of the field containing text for scoring.
        cache_scores: Whether to cache difficulty scores.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        scheduler: CurriculumScheduler,
        difficulty_scorer: Optional[DifficultyScorer] = None,
        text_field: str = "text",
        cache_scores: bool = True,
    ):
        self.base_dataset = base_dataset
        self.scheduler = scheduler
        self.difficulty_scorer = difficulty_scorer
        self.text_field = text_field
        self.cache_scores = cache_scores
        self._samples: List[CurriculumSample] = []
        self._score_cache: Dict[int, float] = {}
        self._annotate_all()

    def _annotate_all(self):
        """Annotate all samples with difficulty scores."""
        self._samples = []
        for idx in range(len(self.base_dataset)):
            sample = self._get_base_sample(idx)
            text = sample.get(self.text_field, "")
            if isinstance(text, torch.Tensor):
                text = str(text)

            if self.cache_scores and idx in self._score_cache:
                score = self._score_cache[idx]
            elif self.difficulty_scorer:
                score = self.difficulty_scorer.compute_score(text, sample_id=idx)
                self._score_cache[idx] = score
            else:
                score = 0.5

            cs = CurriculumSample(
                data=sample,
                difficulty_score=score,
                difficulty_level=DifficultyLevel.from_float(score),
                sample_id=idx,
            )
            cs.sequence_length = len(text.split())
            self._samples.append(cs)

        self._sort_by_difficulty()

    def _get_base_sample(self, idx: int) -> Dict[str, Any]:
        """Extract a sample from the base dataset as a dictionary."""
        item = self.base_dataset[idx]
        if isinstance(item, dict):
            return item
        elif isinstance(item, (list, tuple)):
            return {"data": item}
        else:
            return {"data": item}

    def _sort_by_difficulty(self):
        """Sort samples by difficulty score."""
        self._samples.sort(key=lambda s: s.difficulty_score)

    def update_scheduler(self, loss: float = 0.0, **kwargs):
        """Update the curriculum scheduler with current training metrics."""
        self.scheduler.step(loss=loss, **kwargs)

    def get_eligible_indices(self) -> List[int]:
        """Get indices of samples eligible for current difficulty level."""
        current_difficulty = self.scheduler.get_difficulty()
        tolerance = 0.05
        eligible = [
            s.sample_id for s in self._samples
            if s.difficulty_score <= current_difficulty + tolerance
        ]
        return eligible

    def sample_batch_indices(self, batch_size: int) -> List[int]:
        """Sample a batch of indices based on current curriculum difficulty."""
        eligible = self.get_eligible_indices()
        if not eligible:
            eligible = [s.sample_id for s in self._samples]

        current_difficulty = self.scheduler.get_difficulty()
        weights = []
        for sid in eligible:
            sample = self._samples[sid]
            distance = abs(sample.difficulty_score - current_difficulty)
            weight = max(0.01, 1.0 - distance)
            weights.append(weight)

        total = sum(weights)
        probs = [w / total for w in weights]

        sampled = self.scheduler.rng.choices(eligible, weights=probs, k=batch_size)
        return sampled

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self._samples):
            raise IndexError(f"Index {idx} out of range for {len(self._samples)} samples")
        return self._samples[idx].data

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get the count of samples at each difficulty level."""
        distribution = Counter()
        for sample in self._samples:
            distribution[sample.difficulty_level.name] += 1
        return dict(distribution)

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get statistics about the current curriculum state."""
        eligible = self.get_eligible_indices()
        scores = [self._samples[sid].difficulty_score for sid in eligible]
        return {
            "total_samples": len(self._samples),
            "eligible_samples": len(eligible),
            "current_difficulty": self.scheduler.get_difficulty(),
            "current_level": self.scheduler.get_level().name,
            "eligible_avg_score": sum(scores) / max(1, len(scores)),
            "eligible_min_score": min(scores) if scores else 0,
            "eligible_max_score": max(scores) if scores else 0,
            "difficulty_distribution": self.get_difficulty_distribution(),
        }


# ---------------------------------------------------------------------------
# Curriculum Sampler
# ---------------------------------------------------------------------------

class CurriculumSampler(Sampler):
    """A PyTorch Sampler that respects curriculum difficulty levels.

    Samples from the dataset based on the current curriculum state,
    preferring samples near the current difficulty level.

    Args:
        dataset: CurriculumDataset or Dataset to sample from.
        scheduler: Curriculum scheduler.
        batch_size: Batch size for grouping.
        difficulty_scorer: Optional scorer for online difficulty estimation.
        shuffle_within_level: Whether to shuffle within the difficulty window.
        oversample_current_level: Whether to oversample samples near current difficulty.
    """

    def __init__(
        self,
        dataset: Dataset,
        scheduler: CurriculumScheduler,
        batch_size: int = 32,
        difficulty_scorer: Optional[DifficultyScorer] = None,
        shuffle_within_level: bool = True,
        oversample_current_level: bool = True,
    ):
        self.dataset = dataset
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.difficulty_scorer = difficulty_scorer
        self.shuffle_within_level = shuffle_within_level
        self.oversample_current_level = oversample_current_level
        self._epoch = 0
        self._sample_scores: Dict[int, float] = {}
        self._precompute_scores()

    def _precompute_scores(self):
        """Precompute difficulty scores for all samples."""
        if self.difficulty_scorer is None:
            return
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            if isinstance(sample, dict):
                text = sample.get("text", "")
                if isinstance(text, torch.Tensor):
                    text = str(text)
                self._sample_scores[idx] = self.difficulty_scorer.compute_score(text, idx)
            else:
                self._sample_scores[idx] = 0.5

    def _get_sample_weights(self) -> List[float]:
        """Get sampling weights based on current difficulty."""
        current_difficulty = self.scheduler.get_difficulty()
        n = len(self.dataset)
        weights = []

        for idx in range(n):
            if idx in self._sample_scores:
                score = self._sample_scores[idx]
            else:
                score = 0.5

            if self.oversample_current_level:
                distance = abs(score - current_difficulty)
                weight = math.exp(-distance * 5.0)
            else:
                if score <= current_difficulty:
                    weight = 1.0
                else:
                    weight = 0.1

            weights.append(max(0.01, weight))

        total = sum(weights)
        return [w / total for w in weights]

    def __iter__(self) -> Iterator[int]:
        """Iterate over dataset indices based on curriculum schedule."""
        n = len(self.dataset)
        indices = list(range(n))

        weights = self._get_sample_weights()

        num_batches = (n + self.batch_size - 1) // self.batch_size
        result_indices = []

        for _ in range(num_batches):
            batch_indices = random.choices(
                indices, weights=weights, k=min(self.batch_size, n)
            )
            if self.shuffle_within_level:
                random.shuffle(batch_indices)
            result_indices.extend(batch_indices)

        if self.shuffle_within_level:
            random.shuffle(result_indices)

        self._epoch += 1
        return iter(result_indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        """Set the current epoch for reproducibility."""
        self._epoch = epoch


# ---------------------------------------------------------------------------
# Curriculum Manager (High-level orchestrator)
# ---------------------------------------------------------------------------

class CurriculumManager:
    """High-level manager that orchestrates curriculum learning.

    Combines a scheduler, evaluator, dataset, and scorer into a
    unified interface for curriculum-based training.

    Args:
        scheduler: The curriculum scheduler to use.
        evaluator: Optional evaluator for readiness checks.
        difficulty_scorer: Optional difficulty scorer.
        auto_advance: Whether to auto-advance difficulty based on evaluation.
    """

    def __init__(
        self,
        scheduler: CurriculumScheduler,
        evaluator: Optional[CurriculumEvaluator] = None,
        difficulty_scorer: Optional[DifficultyScorer] = None,
        auto_advance: bool = True,
    ):
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.difficulty_scorer = difficulty_scorer
        self.auto_advance = auto_advance
        self._history: List[Dict[str, Any]] = []
        self._level_transitions: List[Dict[str, Any]] = []
        self._prev_level = scheduler.get_level()

    def step(self, loss: float = 0.0, accuracy: float = 0.0, **kwargs) -> float:
        """Advance the curriculum and optionally auto-advance levels."""
        difficulty = self.scheduler.step(loss=loss, accuracy=accuracy, **kwargs)

        if self.evaluator:
            step = kwargs.get("step", self.scheduler.state.current_step)
            self.evaluator.record(loss, accuracy, step)

            if self.auto_advance:
                should, details = self.evaluator.should_advance(step)
                if should:
                    current = self.scheduler.get_difficulty()
                    step_size = (self.scheduler.max_difficulty - self.scheduler.min_difficulty) * 0.05
                    self.scheduler.state.current_difficulty = min(
                        self.scheduler.max_difficulty, current + step_size
                    )

        current_level = self.scheduler.get_level()
        if current_level != self._prev_level:
            self._level_transitions.append({
                "step": self.scheduler.state.current_step,
                "from_level": self._prev_level.name,
                "to_level": current_level.name,
                "difficulty": difficulty,
            })
            if self.evaluator:
                self.evaluator.reset_level()
            self._prev_level = current_level

        self._history.append({
            "step": self.scheduler.state.current_step,
            "difficulty": difficulty,
            "level": current_level.name,
            "loss": loss,
            "accuracy": accuracy,
        })

        return difficulty

    def get_curriculum_dataset(
        self, base_dataset: Dataset, text_field: str = "text"
    ) -> CurriculumDataset:
        """Create a CurriculumDataset wrapping the base dataset."""
        return CurriculumDataset(
            base_dataset=base_dataset,
            scheduler=self.scheduler,
            difficulty_scorer=self.difficulty_scorer,
            text_field=text_field,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of curriculum state."""
        summary = self.scheduler.get_summary()
        summary["level_transitions"] = len(self._level_transitions)
        summary["history_length"] = len(self._history)
        if self.evaluator:
            summary["evaluation"] = self.evaluator.get_evaluation_summary()
        return summary

    def save(self, path: str):
        """Save curriculum state and history."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "scheduler_state": self.scheduler.state.to_dict(),
            "history": self._history[-1000:],
            "level_transitions": self._level_transitions,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load curriculum state and history."""
        with open(path, "r") as f:
            data = json.load(f)
        if "scheduler_state" in data:
            self.scheduler.state = CurriculumState.from_dict(data["scheduler_state"])
        self._history = data.get("history", [])
        self._level_transitions = data.get("level_transitions", [])
        self._prev_level = self.scheduler.get_level()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def create_default_curriculum(
    total_steps: int,
    strategy: str = "linear",
    **kwargs,
) -> CurriculumScheduler:
    """Factory function to create a curriculum scheduler.

    Args:
        total_steps: Total training steps.
        strategy: Curriculum strategy type.
        **kwargs: Additional arguments passed to the scheduler constructor.

    Returns:
        A CurriculumScheduler instance.
    """
    if strategy == "linear":
        return LinearCurriculum(total_steps=total_steps, **kwargs)
    elif strategy == "step":
        return StepCurriculum(total_steps=total_steps, **kwargs)
    elif strategy == "competence":
        return CompetenceBasedCurriculum(total_steps=total_steps, **kwargs)
    elif strategy == "self_paced":
        return SelfPacedCurriculum(total_steps=total_steps, **kwargs)
    elif strategy == "baby_step":
        return BabyStepCurriculum(total_steps=total_steps, **kwargs)
    else:
        raise ValueError(f"Unknown curriculum strategy: {strategy}")


def compute_sample_difficulties(
    texts: List[str],
    scorer: Optional[DifficultyScorer] = None,
) -> List[float]:
    """Compute difficulty scores for a list of texts.

    Args:
        texts: List of text strings.
        scorer: Optional DifficultyScorer. Created from corpus if None.

    Returns:
        List of difficulty scores in [0, 1].
    """
    if scorer is None:
        scorer = DifficultyScorer.from_corpus(texts)
    return [scorer.compute_score(text) for text in texts]


def bucket_samples_by_difficulty(
    samples: List[Any],
    difficulties: List[float],
    num_buckets: int = 5,
) -> List[List[Any]]:
    """Bucket samples into difficulty groups.

    Args:
        samples: List of samples.
        difficulties: Corresponding difficulty scores.
        num_buckets: Number of difficulty buckets.

    Returns:
        List of buckets, each containing samples.
    """
    paired = list(zip(difficulties, samples))
    paired.sort(key=lambda x: x[0])

    bucket_size = max(1, len(paired) // num_buckets)
    buckets = []
    for i in range(0, len(paired), bucket_size):
        bucket = [s for _, s in paired[i:i + bucket_size]]
        buckets.append(bucket)

    return buckets


def difficulty_aware_collate_fn(
    batch: List[Dict[str, Any]],
    max_length: Optional[int] = None,
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Collate function that respects difficulty-based padding.

    Args:
        batch: List of sample dictionaries.
        max_length: Maximum sequence length (None for no limit).
        pad_value: Padding value for tensors.

    Returns:
        Dictionary of collated tensors.
    """
    keys = set()
    for item in batch:
        keys.update(item.keys())

    result = {}
    for key in keys:
        values = [item.get(key) for item in batch]
        if isinstance(values[0], torch.Tensor):
            tensors = [v for v in values if isinstance(v, torch.Tensor)]
            if not tensors:
                continue

            max_len = min(v.size(-1) for v in tensors)
            if max_length is not None:
                max_len = min(max_len, max_length)

            padded = []
            for v in tensors:
                if v.size(-1) > max_len:
                    padded.append(v[..., :max_len])
                elif v.size(-1) < max_len:
                    pad_size = max_len - v.size(-1)
                    padding = torch.full(
                        (v.shape[0], pad_size), pad_value,
                        dtype=v.dtype, device=v.device,
                    )
                    padded.append(torch.cat([v, padding], dim=-1))
                else:
                    padded.append(v)

            result[key] = torch.stack(padded, dim=0)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values, dtype=torch.float32)
        elif isinstance(values[0], str):
            result[key] = values
        else:
            result[key] = values

    return result


def plot_curriculum_schedule(
    scheduler: CurriculumScheduler,
    steps: Optional[List[int]] = None,
    num_points: int = 1000,
) -> Dict[str, List[float]]:
    """Generate data for plotting a curriculum schedule.

    Args:
        scheduler: The curriculum scheduler.
        steps: Specific steps to evaluate (None for automatic).
        num_points: Number of points if steps is None.

    Returns:
        Dictionary with 'steps' and 'difficulties' lists.
    """
    if steps is None:
        steps = list(range(0, scheduler.total_steps, max(1, scheduler.total_steps // num_points)))

    difficulties = []
    for step in steps:
        if hasattr(scheduler, "_compute_difficulty"):
            d = scheduler._compute_difficulty(step)
        elif hasattr(scheduler, "get_difficulty_at_step"):
            d = scheduler.get_difficulty_at_step(step)
        else:
            d = scheduler.min_difficulty + (step / max(1, scheduler.total_steps)) * (
                scheduler.max_difficulty - scheduler.min_difficulty
            )
        difficulties.append(d)

    return {"steps": steps, "difficulties": difficulties}


def compute_curriculum_metrics(
    scheduler: CurriculumScheduler,
) -> Dict[str, float]:
    """Compute summary metrics about the curriculum.

    Args:
        scheduler: The curriculum scheduler.

    Returns:
        Dictionary of curriculum metrics.
    """
    state = scheduler.state
    progress = state.current_step / max(1, state.total_steps)

    level_counts = Counter()
    for h in state.history:
        if "level" in h:
            level_counts[h["level"]] += 1

    metrics = {
        "progress": progress,
        "current_difficulty": state.current_difficulty,
        "current_level": state.current_level.name,
        "samples_processed": state.samples_processed,
        "avg_loss": state.avg_loss_at_level,
        "accuracy": state.accuracy_at_level,
        "difficulty_per_step": state.current_difficulty / max(1, state.current_step),
    }
    metrics["level_distribution"] = dict(level_counts)
    return metrics


def export_curriculum_report(
    scheduler: CurriculumScheduler,
    path: str,
    include_history: bool = True,
):
    """Export a comprehensive curriculum report to JSON.

    Args:
        scheduler: The curriculum scheduler.
        path: Output file path.
        include_history: Whether to include full history.
    """
    report = {
        "summary": scheduler.get_summary(),
        "level_metrics": {
            level.name: metrics.to_dict()
            for level, metrics in scheduler.level_metrics.items()
        },
    }

    if include_history:
        report["history"] = scheduler.state.history

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Curriculum report exported to {path}")
