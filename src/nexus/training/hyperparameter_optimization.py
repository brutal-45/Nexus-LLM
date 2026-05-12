"""
Hyperparameter Optimization for Nexus LLM.

Provides a comprehensive suite of hyperparameter optimization algorithms
implemented from scratch: Grid Search, Random Search, Bayesian Optimization
with Gaussian Processes, Hyperband, ASHA, Population Based Training, CMA-ES,
and TPE (Tree-structured Parzen Estimator).

Classes:
    HPSearchSpace: Define hyperparameter search space.
    HPTrial: Dataclass for a single optimization trial.
    ExperimentTracker: Track all HP optimization trials.
    Study: Manage optimization study with best-params tracking.
    GridSearch: Exhaustive grid search.
    RandomSearch: Random sampling from search space.
    BayesianOptimizer: GP-based Bayesian optimization (from scratch).
    Hyperband: Early-stopping based HP optimization.
    ASHA: Asynchronous Successive Halving Algorithm.
    PopulationBasedTraining: PBT with mutation and selection.
    CMAES: Covariance Matrix Adaptation Evolution Strategy (from scratch).
    OptunaStyleSampler: TPE sampler.
"""

from __future__ import annotations

import abc
import copy
import hashlib
import json
import logging
import math
import os
import random
import time
import uuid
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class TrialStatus(Enum):
    """Status of a hyperparameter trial."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PRUNED = auto()
    TIMEOUT = auto()


class ParamType(Enum):
    """Type of hyperparameter."""
    CONTINUOUS = auto()
    DISCRETE = auto()
    INTEGER = auto()
    CATEGORICAL = auto()


class SamplingStrategy(Enum):
    """Strategy for sampling from distributions."""
    UNIFORM = auto()
    LOG_UNIFORM = auto()
    NORMAL = auto()
    LOG_NORMAL = auto()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ParamDistribution:
    """Distribution specification for a single hyperparameter.

    Attributes:
        name: Name of the hyperparameter.
        param_type: Type of the parameter (continuous, discrete, etc.).
        low: Lower bound (for numeric types).
        high: Upper bound (for numeric types).
        choices: Possible values (for categorical type).
        sampling: Sampling strategy.
        log_scale: Whether to sample in log space.
        step: Step size (for discrete/integer types).
        default: Default value.
    """
    name: str
    param_type: ParamType = ParamType.CONTINUOUS
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    sampling: SamplingStrategy = SamplingStrategy.UNIFORM
    log_scale: bool = False
    step: Optional[float] = None
    default: Optional[Any] = None

    def sample(self, rng: Optional[random.Random] = None) -> Any:
        """Sample a value from this distribution."""
        r = rng or random.random
        if self.param_type == ParamType.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"No choices for categorical param '{self.name}'")
            return r.choice(self.choices)
        elif self.param_type == ParamType.INTEGER:
            low = int(self.low) if self.low is not None else 0
            high = int(self.high) if self.high is not None else 10
            if self.log_scale:
                log_low = math.log(max(1, low))
                log_high = math.log(high + 1)
                val = math.exp(r.uniform(log_low, log_high))
                return max(low, min(high, round(val)))
            val = r.uniform(low, high)
            if self.step is not None:
                val = round(val / self.step) * self.step
            return max(low, min(high, int(round(val))))
        elif self.param_type == ParamType.DISCRETE:
            low = self.low if self.low is not None else 0.0
            high = self.high if self.high is not None else 1.0
            val = r.uniform(low, high)
            if self.step is not None:
                val = round(val / self.step) * self.step
            return max(low, min(high, val))
        else:
            low = self.low if self.low is not None else 0.0
            high = self.high if self.high is not None else 1.0
            if self.log_scale:
                log_low = math.log(max(1e-10, low))
                log_high = math.log(high)
                return math.exp(r.uniform(log_low, log_high))
            if self.sampling == SamplingStrategy.NORMAL:
                mean = (low + high) / 2
                std = (high - low) / 6
                val = r.gauss(mean, std)
                return max(low, min(high, val))
            elif self.sampling == SamplingStrategy.LOG_UNIFORM:
                log_low = math.log(max(1e-10, low))
                log_high = math.log(high)
                return math.exp(r.uniform(log_low, log_high))
            elif self.sampling == SamplingStrategy.LOG_NORMAL:
                log_low = math.log(max(1e-10, low))
                log_high = math.log(high)
                mean = (log_low + log_high) / 2
                std = (log_high - log_low) / 6
                val = math.exp(r.gauss(mean, std))
                return max(low, min(high, val))
            return r.uniform(low, high)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "param_type": self.param_type.name,
            "low": self.low,
            "high": self.high,
            "choices": self.choices,
            "sampling": self.sampling.name,
            "log_scale": self.log_scale,
            "step": self.step,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParamDistribution":
        return cls(
            name=data["name"],
            param_type=ParamType[data.get("param_type", "CONTINUOUS")],
            low=data.get("low"),
            high=data.get("high"),
            choices=data.get("choices"),
            sampling=SamplingStrategy[data.get("sampling", "UNIFORM")],
            log_scale=data.get("log_scale", False),
            step=data.get("step"),
            default=data.get("default"),
        )

    @classmethod
    def categorical(cls, name: str, choices: List[Any], **kwargs) -> "ParamDistribution":
        return cls(name=name, param_type=ParamType.CATEGORICAL, choices=choices, **kwargs)

    @classmethod
    def uniform(cls, name: str, low: float, high: float, **kwargs) -> "ParamDistribution":
        return cls(name=name, param_type=ParamType.CONTINUOUS, low=low, high=high, **kwargs)

    @classmethod
    def log_uniform(cls, name: str, low: float, high: float, **kwargs) -> "ParamDistribution":
        return cls(name=name, param_type=ParamType.CONTINUOUS, low=low, high=high, log_scale=True, **kwargs)

    @classmethod
    def integer(cls, name: str, low: int, high: int, **kwargs) -> "ParamDistribution":
        return cls(name=name, param_type=ParamType.INTEGER, low=low, high=high, **kwargs)

    @classmethod
    def log_integer(cls, name: str, low: int, high: int, **kwargs) -> "ParamDistribution":
        return cls(name=name, param_type=ParamType.INTEGER, low=low, high=high, log_scale=True, **kwargs)


@dataclass
class HPTrial:
    """Dataclass for a single hyperparameter optimization trial.

    Attributes:
        trial_id: Unique identifier.
        params: Dictionary of hyperparameter name -> value.
        metrics: Dictionary of metric name -> value.
        status: Current status of the trial.
        start_time: When the trial started.
        end_time: When the trial ended.
        duration: Duration in seconds.
        error: Error message if trial failed.
        metadata: Additional metadata.
    """
    trial_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    status: TrialStatus = TrialStatus.PENDING
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    _intermediate_metrics: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.trial_id:
            self.trial_id = str(uuid.uuid4())[:8]

    def start(self):
        """Mark trial as running."""
        self.status = TrialStatus.RUNNING
        self.start_time = time.time()

    def complete(self, metrics: Dict[str, float]):
        """Mark trial as completed with metrics."""
        self.status = TrialStatus.COMPLETED
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.metrics = metrics

    def fail(self, error: str):
        """Mark trial as failed."""
        self.status = TrialStatus.FAILED
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.error = error

    def prune(self):
        """Mark trial as pruned."""
        self.status = TrialStatus.PRUNED
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def record_intermediate(self, step: int, metrics: Dict[str, float]):
        """Record intermediate metrics during training."""
        self._intermediate_metrics.append({"step": step, **metrics})

    def get_objective_value(self, objective_key: str = "loss") -> Optional[float]:
        """Get the objective metric value."""
        return self.metrics.get(objective_key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "metrics": self.metrics,
            "status": self.status.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "error": self.error,
            "metadata": self.metadata,
            "intermediate_count": len(self._intermediate_metrics),
        }


# ---------------------------------------------------------------------------
# HP Search Space
# ---------------------------------------------------------------------------

class HPSearchSpace:
    """Define and manage a hyperparameter search space.

    Provides methods to define distributions for each hyperparameter,
    sample random configurations, and enumerate grid points.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self._distributions: Dict[str, ParamDistribution] = {}
        self._param_order: List[str] = []

    def add(self, distribution: ParamDistribution):
        """Add a parameter distribution to the search space."""
        self._distributions[distribution.name] = distribution
        if distribution.name not in self._param_order:
            self._param_order.append(distribution.name)

    def add_categorical(self, name: str, choices: List[Any], default: Any = None):
        """Add a categorical parameter."""
        dist = ParamDistribution.categorical(name, choices, default=default)
        self.add(dist)

    def add_uniform(self, name: str, low: float, high: float, default: float = None):
        """Add a uniformly distributed continuous parameter."""
        dist = ParamDistribution.uniform(name, low, high, default=default)
        self.add(dist)

    def add_log_uniform(self, name: str, low: float, high: float, default: float = None):
        """Add a log-uniformly distributed continuous parameter."""
        dist = ParamDistribution.log_uniform(name, low, high, default=default)
        self.add(dist)

    def add_integer(self, name: str, low: int, high: int, default: int = None):
        """Add an integer parameter."""
        dist = ParamDistribution.integer(name, low, high, default=default)
        self.add(dist)

    def add_log_integer(self, name: str, low: int, high: int, default: int = None):
        """Add a log-scaled integer parameter."""
        dist = ParamDistribution.log_integer(name, low, high, default=default)
        self.add(dist)

    def sample(self) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        return {name: dist.sample(self.rng) for name, dist in self._distributions.items()}

    def sample_uniform(self) -> Dict[str, Any]:
        """Sample using uniform distribution regardless of configured sampling."""
        config = {}
        for name, dist in self._distributions.items():
            if dist.param_type == ParamType.CATEGORICAL:
                config[name] = self.rng.choice(dist.choices)
            else:
                config[name] = dist.sample(self.rng)
        return config

    def get_grid_points(self, num_points_per_param: int = 5) -> List[Dict[str, Any]]:
        """Generate grid points for grid search.

        Args:
            num_points_per_param: Number of points for continuous/discrete params.

        Returns:
            List of parameter configurations.
        """
        param_values = {}
        for name, dist in self._distributions.items():
            if dist.param_type == ParamType.CATEGORICAL:
                param_values[name] = dist.choices
            elif dist.param_type == ParamType.INTEGER:
                low = int(dist.low) if dist.low is not None else 0
                high = int(dist.high) if dist.high is not None else 10
                if dist.log_scale and high > low:
                    points = []
                    log_low = math.log(max(1, low))
                    log_high = math.log(high + 1)
                    for i in range(num_points_per_param):
                        val = math.exp(log_low + i * (log_high - log_low) / (num_points_per_param - 1))
                        points.append(max(low, min(high, int(round(val)))))
                    param_values[name] = sorted(set(points))
                else:
                    step = max(1, (high - low) // max(1, num_points_per_param - 1))
                    param_values[name] = list(range(low, high + 1, step))[:num_points_per_param]
            elif dist.param_type in (ParamType.CONTINUOUS, ParamType.DISCRETE):
                low = dist.low if dist.low is not None else 0.0
                high = dist.high if dist.high is not None else 1.0
                if dist.log_scale:
                    log_low = math.log(max(1e-10, low))
                    log_high = math.log(high)
                    points = [
                        math.exp(log_low + i * (log_high - log_low) / (num_points_per_param - 1))
                        for i in range(num_points_per_param)
                    ]
                else:
                    step_size = (high - low) / max(1, num_points_per_param - 1)
                    points = [low + i * step_size for i in range(num_points_per_param)]
                if dist.step is not None:
                    points = [round(p / dist.step) * dist.step for p in points]
                param_values[name] = points
            else:
                param_values[name] = [dist.default] if dist.default else [dist.sample(self.rng)]

        configs = [{}]
        for name in self._param_order:
            if name not in param_values:
                continue
            new_configs = []
            for config in configs:
                for value in param_values[name]:
                    new_config = dict(config)
                    new_config[name] = value
                    new_configs.append(new_config)
            configs = new_configs

        return configs

    def get_param_names(self) -> List[str]:
        """Get all parameter names."""
        return list(self._param_order)

    def get_distribution(self, name: str) -> Optional[ParamDistribution]:
        """Get a parameter distribution by name."""
        return self._distributions.get(name)

    def get_bounds(self, name: str) -> Tuple[float, float]:
        """Get the bounds for a parameter."""
        dist = self._distributions.get(name)
        if dist is None:
            raise KeyError(f"Parameter '{name}' not found")
        low = dist.low if dist.low is not None else 0.0
        high = dist.high if dist.high is not None else 1.0
        return (low, high)

    def get_default_config(self) -> Dict[str, Any]:
        """Get a configuration with default values."""
        config = {}
        for name, dist in self._distributions.items():
            if dist.default is not None:
                config[name] = dist.default
            else:
                config[name] = dist.sample(self.rng)
        return config

    def size(self) -> int:
        """Return the number of parameters."""
        return len(self._distributions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "distributions": {name: d.to_dict() for name, d in self._distributions.items()},
            "param_order": self._param_order,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HPSearchSpace":
        space = cls(seed=data.get("seed", 42))
        for name, d in data.get("distributions", {}).items():
            space.add(ParamDistribution.from_dict(d))
        space._param_order = data.get("param_order", list(space._distributions.keys()))
        return space

    @classmethod
    def create_default_training_space(cls, **overrides) -> "HPSearchSpace":
        """Create a search space for common training hyperparameters."""
        space = cls()
        space.add_log_uniform("learning_rate", 1e-6, 1e-2, default=3e-4)
        space.add_integer("batch_size", 8, 256, default=32)
        space.add_log_uniform("weight_decay", 1e-6, 1e-1, default=1e-4)
        space.add_uniform("dropout", 0.0, 0.5, default=0.1)
        space.add_categorical("optimizer", ["adam", "adamw", "sgd"], default="adamw")
        space.add_uniform("warmup_ratio", 0.0, 0.2, default=0.06)
        space.add_categorical("scheduler", ["cosine", "linear", "constant"], default="cosine")
        space.add_log_uniform("grad_clip", 0.1, 10.0, default=1.0)
        space.add_categorical("activation", ["gelu", "relu", "silu"], default="gelu")
        space.add_integer("num_warmup_steps", 0, 5000, default=1000)
        for key, value in overrides.items():
            if isinstance(value, ParamDistribution):
                space.add(value)
        return space


# ---------------------------------------------------------------------------
# Experiment Tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Track all hyperparameter optimization trials.

    Stores trial history, computes statistics, and manages persistence.
    """

    def __init__(self, study_name: str = "default", storage_dir: str = "./hp_results"):
        self.study_name = study_name
        self.storage_dir = storage_dir
        self._trials: List[HPTrial] = []
        self._best_trial: Optional[HPTrial] = None
        self._best_params: Dict[str, Any] = {}
        self._best_value: float = float("inf")
        self._objective_direction: str = "minimize"

    def add_trial(self, trial: HPTrial):
        """Add a completed trial to the tracker."""
        self._trials.append(trial)
        if trial.status == TrialStatus.COMPLETED:
            self._update_best(trial)

    def _update_best(self, trial: HPTrial):
        """Update the best trial if this trial is better."""
        for key, value in trial.metrics.items():
            if key not in self._best_params or (
                self._objective_direction == "minimize" and value < self._best_value
            ) or (
                self._objective_direction == "maximize" and value > self._best_value
            ):
                self._best_value = value
                self._best_trial = trial
                self._best_params = dict(trial.params)

    def get_best_trial(self) -> Optional[HPTrial]:
        """Return the trial with the best objective value."""
        return self._best_trial

    def get_best_params(self) -> Dict[str, Any]:
        """Return the best parameters found so far."""
        return dict(self._best_params)

    def get_best_value(self) -> float:
        """Return the best objective value found so far."""
        return self._best_value

    def get_completed_trials(self) -> List[HPTrial]:
        """Return all completed trials."""
        return [t for t in self._trials if t.status == TrialStatus.COMPLETED]

    def get_trials_by_status(self, status: TrialStatus) -> List[HPTrial]:
        """Return trials filtered by status."""
        return [t for t in self._trials if t.status == status]

    def get_all_trials(self) -> List[HPTrial]:
        """Return all trials."""
        return list(self._trials)

    def get_number_of_trials(self) -> int:
        """Return total number of trials."""
        return len(self._trials)

    def set_direction(self, direction: str):
        """Set optimization direction ('minimize' or 'maximize')."""
        self._objective_direction = direction

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of all trials."""
        completed = self.get_completed_trials()
        if not completed:
            return {
                "total_trials": len(self._trials),
                "completed_trials": 0,
                "failed_trials": len(self.get_trials_by_status(TrialStatus.FAILED)),
            }

        values = [t.metrics.get("loss", t.metrics.get("score", 0)) for t in completed]
        durations = [t.duration for t in completed]

        return {
            "total_trials": len(self._trials),
            "completed_trials": len(completed),
            "failed_trials": len(self.get_trials_by_status(TrialStatus.FAILED)),
            "pruned_trials": len(self.get_trials_by_status(TrialStatus.PRUNED)),
            "best_value": min(values) if values else float("inf"),
            "worst_value": max(values) if values else float("-inf"),
            "mean_value": sum(values) / len(values) if values else 0,
            "std_value": (
                math.sqrt(sum((v - sum(values)/len(values))**2 for v in values) / len(values))
                if len(values) > 1 else 0
            ),
            "median_value": sorted(values)[len(values) // 2] if values else 0,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "total_duration": sum(durations),
            "objective_direction": self._objective_direction,
            "best_params": self._best_params,
        }

    def get_param_importance(self) -> Dict[str, float]:
        """Estimate parameter importance from completed trials.

        Uses variance-based analysis to rank parameter importance.
        """
        completed = self.get_completed_trials()
        if len(completed) < 5:
            return {}

        param_values: Dict[str, List[Tuple[Any, float]]] = defaultdict(list)
        for trial in completed:
            obj_value = trial.metrics.get("loss", trial.metrics.get("score", 0))
            for name, value in trial.params.items():
                param_values[name].append((value, obj_value))

        importance = {}
        for name, value_obj_pairs in param_values.items():
            values = [v for v, _ in value_obj_pairs]
            objectives = [o for _, o in value_obj_pairs]

            if all(isinstance(v, (int, float)) for v in values):
                values_num = list(values)
                mean_v = sum(values_num) / len(values_num)
                std_v = math.sqrt(
                    sum((v - mean_v) ** 2 for v in values_num) / len(values_num)
                ) if len(values_num) > 1 else 0

                mean_o = sum(objectives) / len(objectives)
                cov = sum(
                    (values_num[i] - mean_v) * (objectives[i] - mean_o)
                    for i in range(len(values_num))
                ) / len(values_num)

                var_o = sum((o - mean_o) ** 2 for o in objectives) / len(objectives)
                if std_v > 1e-10 and var_o > 1e-10:
                    r_squared = (cov ** 2) / (std_v ** 2 * var_o)
                else:
                    r_squared = 0.0
                importance[name] = abs(r_squared)
            else:
                categorical_values = list(set(values))
                if len(categorical_values) > 1:
                    means = {}
                    for cat_val in categorical_values:
                        cat_objs = [
                            o for v, o in value_obj_pairs if v == cat_val
                        ]
                        if cat_objs:
                            means[cat_val] = sum(cat_objs) / len(cat_objs)
                    if means:
                        overall_mean = sum(objectives) / len(objectives)
                        ss_between = sum(
                            len([v for v, _ in value_obj_pairs if v == cat]) * (m - overall_mean) ** 2
                            for cat, m in means.items()
                        )
                        ss_total = sum((o - overall_mean) ** 2 for o in objectives)
                        importance[name] = ss_between / max(1e-10, ss_total)
                    else:
                        importance[name] = 0.0
                else:
                    importance[name] = 0.0

        total = sum(importance.values()) if importance else 1
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        return importance

    def save(self, path: Optional[str] = None):
        """Save tracker state to a JSON file."""
        if path is None:
            path = os.path.join(self.storage_dir, f"{self.study_name}_results.json")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "study_name": self.study_name,
            "objective_direction": self._objective_direction,
            "best_params": self._best_params,
            "best_value": self._best_value,
            "trials": [t.to_dict() for t in self._trials],
            "statistics": self.get_statistics(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None):
        """Load tracker state from a JSON file."""
        if path is None:
            path = os.path.join(self.storage_dir, f"{self.study_name}_results.json")
        with open(path, "r") as f:
            data = json.load(f)
        self._objective_direction = data.get("objective_direction", "minimize")
        self._best_params = data.get("best_params", {})
        self._best_value = data.get("best_value", float("inf"))
        for trial_data in data.get("trials", []):
            trial = HPTrial(
                trial_id=trial_data.get("trial_id", ""),
                params=trial_data.get("params", {}),
                metrics=trial_data.get("metrics", {}),
                status=TrialStatus[trial_data.get("status", "PENDING")],
                duration=trial_data.get("duration", 0),
            )
            self._trials.append(trial)


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

class Study:
    """Manage a hyperparameter optimization study.

    Coordinates the search space, optimizer, and tracker.
    """

    def __init__(
        self,
        study_name: str = "default",
        search_space: Optional[HPSearchSpace] = None,
        direction: str = "minimize",
        objective_key: str = "loss",
        storage_dir: str = "./hp_results",
    ):
        self.study_name = study_name
        self.search_space = search_space or HPSearchSpace()
        self.direction = direction
        self.objective_key = objective_key
        self.storage_dir = storage_dir
        self.tracker = ExperimentTracker(study_name, storage_dir)
        self.tracker.set_direction(direction)
        self._callbacks: List[Callable[[HPTrial, Dict], None]] = []

    def on_trial_complete(self, callback: Callable[[HPTrial, Dict], None]):
        """Register a callback for trial completion."""
        self._callbacks.append(callback)

    def create_trial(self, params: Optional[Dict[str, Any]] = None) -> HPTrial:
        """Create a new trial with given or sampled parameters."""
        trial = HPTrial(params=params or self.search_space.sample())
        return trial

    def report(self, trial: HPTrial, metrics: Dict[str, float]):
        """Report metrics for a trial."""
        trial.complete(metrics)
        self.tracker.add_trial(trial)
        summary = self.get_summary()
        for cb in self._callbacks:
            try:
                cb(trial, summary)
            except Exception as e:
                logger.warning(f"Study callback failed: {e}")

    def get_best_params(self) -> Dict[str, Any]:
        """Return the best parameters found."""
        return self.tracker.get_best_params()

    def get_best_value(self) -> float:
        """Return the best objective value."""
        return self.tracker.get_best_value()

    def get_summary(self) -> Dict[str, Any]:
        """Get study summary."""
        stats = self.tracker.get_statistics()
        stats["study_name"] = self.study_name
        stats["direction"] = self.direction
        stats["objective_key"] = self.objective_key
        stats["search_space_size"] = self.search_space.size()
        stats["best_params"] = self.tracker.get_best_params()
        return stats

    def save(self):
        """Save study state."""
        self.tracker.save()

    def load(self):
        """Load study state."""
        self.tracker.load()


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

class GridSearch:
    """Exhaustive grid search over the hyperparameter space.

    Evaluates all combinations of hyperparameters on a grid.
    Simple but can be very expensive for high-dimensional spaces.

    Args:
        search_space: The hyperparameter search space.
        num_points: Number of grid points per continuous parameter.
    """

    def __init__(
        self,
        search_space: HPSearchSpace,
        num_points: int = 5,
    ):
        self.search_space = search_space
        self.num_points = num_points
        self._grid_points = search_space.get_grid_points(num_points)
        self._current_idx = 0

    def total_configurations(self) -> int:
        """Return the total number of configurations to evaluate."""
        return len(self._grid_points)

    def has_next(self) -> bool:
        """Check if there are more configurations to evaluate."""
        return self._current_idx < len(self._grid_points)

    def next_config(self) -> Dict[str, Any]:
        """Get the next configuration."""
        if not self.has_next():
            raise StopIteration("No more grid configurations")
        config = self._grid_points[self._current_idx]
        self._current_idx += 1
        return config

    def reset(self):
        """Reset the search to the beginning."""
        self._current_idx = 0

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], Dict[str, float]],
        max_trials: Optional[int] = None,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the full grid search optimization.

        Args:
            objective: Function that takes params and returns metrics.
            max_trials: Maximum number of trials (None for all).
            callback: Called after each trial.

        Returns:
            Dictionary with best params, value, and all trial results.
        """
        logger.info(
            f"Grid search: {self.total_configurations()} configurations, "
            f"{self.num_points} points per param"
        )

        trials = []
        limit = max_trials or self.total_configurations()

        for i in range(min(limit, self.total_configurations())):
            config = self._grid_points[i]
            try:
                start = time.time()
                metrics = objective(config)
                duration = time.time() - start

                trial_data = {
                    "params": config,
                    "metrics": metrics,
                    "duration": duration,
                    "trial_index": i,
                }
                trials.append(trial_data)

                if callback:
                    callback(config, metrics)

            except Exception as e:
                logger.warning(f"Grid search trial {i} failed: {e}")
                trials.append({
                    "params": config,
                    "metrics": {},
                    "error": str(e),
                    "trial_index": i,
                })

        completed = [t for t in trials if t.get("metrics")]
        if not completed:
            return {"best_params": {}, "best_value": float("inf"), "trials": trials}

        objective_key = "loss" if "loss" in completed[0]["metrics"] else list(completed[0]["metrics"].keys())[0]
        direction = "minimize" if objective_key == "loss" else "maximize"

        if direction == "minimize":
            best = min(completed, key=lambda t: t["metrics"][objective_key])
        else:
            best = max(completed, key=lambda t: t["metrics"][objective_key])

        return {
            "best_params": best["params"],
            "best_value": best["metrics"][objective_key],
            "objective_key": objective_key,
            "total_trials": len(trials),
            "completed_trials": len(completed),
            "trials": trials,
        }


# ---------------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------------

class RandomSearch:
    """Random sampling from the hyperparameter search space.

    More efficient than grid search for high-dimensional spaces as shown
    by Bergstra & Bengio (2012).

    Args:
        search_space: The hyperparameter search space.
        n_trials: Number of random trials to run.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        search_space: HPSearchSpace,
        n_trials: int = 100,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.seed = seed
        self._rng = random.Random(seed)
        self._trials_completed = 0

    def sample_config(self) -> Dict[str, Any]:
        """Sample a random configuration."""
        return self.search_space.sample()

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], Dict[str, float]],
        n_trials: Optional[int] = None,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, float], int], None]] = None,
    ) -> Dict[str, Any]:
        """Run random search optimization.

        Args:
            objective: Function that takes params and returns metrics.
            n_trials: Number of trials (overrides constructor).
            callback: Called after each trial.

        Returns:
            Dictionary with best params, value, and results.
        """
        num_trials = n_trials or self.n_trials
        logger.info(f"Random search: {num_trials} trials")

        trials = []

        for i in range(num_trials):
            config = self.search_space.sample()
            try:
                start = time.time()
                metrics = objective(config)
                duration = time.time() - start

                trial_data = {
                    "params": config,
                    "metrics": metrics,
                    "duration": duration,
                    "trial_index": i,
                }
                trials.append(trial_data)

                if callback:
                    callback(config, metrics, i)

                self._trials_completed += 1

            except Exception as e:
                logger.warning(f"Random search trial {i} failed: {e}")
                trials.append({
                    "params": config,
                    "metrics": {},
                    "error": str(e),
                    "trial_index": i,
                })

        completed = [t for t in trials if t.get("metrics")]
        if not completed:
            return {"best_params": {}, "best_value": float("inf"), "trials": trials}

        objective_key = "loss" if "loss" in completed[0]["metrics"] else list(completed[0]["metrics"].keys())[0]
        direction = "minimize" if objective_key == "loss" else "maximize"

        if direction == "minimize":
            best = min(completed, key=lambda t: t["metrics"][objective_key])
        else:
            best = max(completed, key=lambda t: t["metrics"][objective_key])

        return {
            "best_params": best["params"],
            "best_value": best["metrics"][objective_key],
            "objective_key": objective_key,
            "total_trials": len(trials),
            "completed_trials": len(completed),
            "trials": trials,
        }


# ---------------------------------------------------------------------------
# Gaussian Process Kernel
# ---------------------------------------------------------------------------

class RBFKernel:
    """Radial Basis Function (Squared Exponential) kernel for GP."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance

    def __call__(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """Compute the kernel matrix between x1 and x2."""
        sq_dist = self._squared_distance(x1, x2)
        return self.variance * torch.exp(-0.5 * sq_dist / (self.length_scale ** 2))

    def _squared_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute squared Euclidean distances."""
        x1_sq = (x1 ** 2).sum(dim=-1, keepdim=True)
        x2_sq = (x2 ** 2).sum(dim=-1, keepdim=True)
        dist = x1_sq + x2_sq.T - 2 * x1 @ x2.T
        return torch.clamp(dist, min=1e-10)

    def compute_kernel_matrix(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """Compute full kernel matrix."""
        return self(x1, x2)

    def compute_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """Compute diagonal of kernel matrix (self-kernel)."""
        return torch.full((x.size(0),), self.variance)


class MaternKernel:
    """Matern kernel for GP (more flexible than RBF)."""

    def __init__(self, nu: float = 2.5, length_scale: float = 1.0, variance: float = 1.0):
        self.nu = nu
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist = torch.sqrt(self._squared_distance(x1, x2) + 1e-10)
        scaled_dist = dist / self.length_scale

        if self.nu == 0.5:
            return self.variance * torch.exp(-scaled_dist)
        elif self.nu == 1.5:
            sqrt3 = math.sqrt(3)
            return self.variance * (1 + sqrt3 * scaled_dist) * torch.exp(-sqrt3 * scaled_dist)
        elif self.nu == 2.5:
            sqrt5 = math.sqrt(5)
            return self.variance * (
                1 + sqrt5 * scaled_dist + (5.0 / 3.0) * scaled_dist ** 2
            ) * torch.exp(-sqrt5 * scaled_dist)
        else:
            return self.variance * torch.exp(-scaled_dist)

    def _squared_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1_sq = (x1 ** 2).sum(dim=-1, keepdim=True)
        x2_sq = (x2 ** 2).sum(dim=-1, keepdim=True)
        dist = x1_sq + x2_sq.T - 2 * x1 @ x2.T
        return torch.clamp(dist, min=1e-10)


# ---------------------------------------------------------------------------
# Gaussian Process
# ---------------------------------------------------------------------------

class GaussianProcess:
    """Gaussian Process regressor implemented from scratch.

    Used by BayesianOptimizer for surrogate modeling.
    """

    def __init__(
        self,
        kernel: Optional[Any] = None,
        noise_variance: float = 1e-4,
        normalize_y: bool = True,
    ):
        self.kernel = kernel or RBFKernel()
        self.noise_variance = noise_variance
        self.normalize_y = normalize_y
        self.X_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self.L: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None
        self._is_fitted = False

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit the GP to training data.

        Args:
            X: Training inputs, shape (n, d).
            y: Training targets, shape (n,).
        """
        self.X_train = X
        if self.normalize_y:
            self.y_mean = y.mean().item()
            self.y_std = max(y.std().item(), 1e-8)
            self.y_train = (y - self.y_mean) / self.y_std
        else:
            self.y_train = y

        K = self.kernel.compute_kernel_matrix(X, X)
        K += torch.eye(X.size(0)) * self.noise_variance

        try:
            self.L = torch.linalg.cholesky(K)
        except torch.linalg.LinAlgError:
            jitter = torch.eye(X.size(0)) * 1e-6
            self.L = torch.linalg.cholesky(K + jitter)

        self.alpha = torch.linalg.solve_triangular(
            self.L, self.y_train, upper=False
        )
        self.alpha = torch.linalg.solve_triangular(
            self.L.T, self.alpha, upper=True
        )
        self._is_fitted = True

    def predict(
        self, X: torch.Tensor, return_std: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict mean and optionally standard deviation.

        Args:
            X: Test inputs, shape (n, d).
            return_std: Whether to return standard deviation.

        Returns:
            Tuple of (mean, std) tensors.
        """
        if not self._is_fitted:
            raise RuntimeError("GP not fitted")

        K_star = self.kernel.compute_kernel_matrix(X, self.X_train)
        mu = K_star @ self.alpha

        if self.normalize_y:
            mu = mu * self.y_std + self.y_mean

        if return_std:
            v = torch.linalg.solve_triangular(
                self.L, K_star.T, upper=False
            )
            K_ss = self.kernel.compute_diagonal(X)
            var = K_ss - (v ** 2).sum(dim=0)
            var = torch.clamp(var, min=1e-10)
            std = torch.sqrt(var)
            if self.normalize_y:
                std = std * self.y_std
            return mu, std
        return mu, None

    def update(self, X_new: torch.Tensor, y_new: torch.Tensor):
        """Incrementally update the GP with new observations."""
        if self.X_train is None:
            self.fit(X_new, y_new)
            return

        self.X_train = torch.cat([self.X_train, X_new], dim=0)
        if self.normalize_y:
            y_new_normalized = (y_new - self.y_mean) / self.y_std
            self.y_train = torch.cat([self.y_train, y_new_normalized], dim=0)
        else:
            self.y_train = torch.cat([self.y_train, y_new], dim=0)

        K = self.kernel.compute_kernel_matrix(self.X_train, self.X_train)
        K += torch.eye(self.X_train.size(0)) * self.noise_variance
        try:
            self.L = torch.linalg.cholesky(K)
        except torch.linalg.LinAlgError:
            jitter = torch.eye(self.X_train.size(0)) * 1e-6
            self.L = torch.linalg.cholesky(K + jitter)

        self.alpha = torch.linalg.solve_triangular(
            self.L, self.y_train, upper=False
        )
        self.alpha = torch.linalg.solve_triangular(
            self.L.T, self.alpha, upper=True
        )


# ---------------------------------------------------------------------------
# Acquisition Functions
# ---------------------------------------------------------------------------

class AcquisitionFunction(abc.ABC):
    """Base class for Bayesian optimization acquisition functions."""

    @abc.abstractmethod
    def __call__(
        self, mean: torch.Tensor, std: torch.Tensor, best_value: float
    ) -> torch.Tensor:
        raise NotImplementedError


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""

    def __init__(self, xi: float = 0.01, maximize: bool = False):
        self.xi = xi
        self.maximize = maximize

    def __call__(
        self, mean: torch.Tensor, std: torch.Tensor, best_value: float
    ) -> torch.Tensor:
        if self.maximize:
            improvement = mean - best_value - self.xi
        else:
            improvement = best_value - mean - self.xi

        Z = improvement / torch.clamp(std, min=1e-10)
        ei = improvement * torch.distributions.Normal(0, 1).cdf(Z)
        ei += std * torch.distributions.Normal(0, 1).log_prob(Z).exp()
        return torch.clamp(ei, min=0)


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound (GP-UCB) acquisition function."""

    def __init__(self, kappa: float = 2.0, maximize: bool = False):
        self.kappa = kappa
        self.maximize = maximize

    def __call__(
        self, mean: torch.Tensor, std: torch.Tensor, best_value: float
    ) -> torch.Tensor:
        if self.maximize:
            return mean + self.kappa * std
        else:
            return -mean + self.kappa * std


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement acquisition function."""

    def __init__(self, xi: float = 0.01, maximize: bool = False):
        self.xi = xi
        self.maximize = maximize

    def __call__(
        self, mean: torch.Tensor, std: torch.Tensor, best_value: float
    ) -> torch.Tensor:
        if self.maximize:
            Z = (mean - best_value - self.xi) / torch.clamp(std, min=1e-10)
        else:
            Z = (best_value - mean - self.xi) / torch.clamp(std, min=1e-10)
        return torch.distributions.Normal(0, 1).cdf(Z)


# ---------------------------------------------------------------------------
# Bayesian Optimizer
# ---------------------------------------------------------------------------

class BayesianOptimizer:
    """Gaussian Process based Bayesian optimization implemented from scratch.

    Efficiently finds optimal hyperparameters by building a surrogate model
    and using acquisition functions to guide exploration.

    Args:
        search_space: Hyperparameter search space.
        n_initial: Number of initial random evaluations.
        kernel: GP kernel to use.
        acquisition: Acquisition function.
        noise_variance: Observation noise for GP.
        random_seed: Random seed.
    """

    def __init__(
        self,
        search_space: HPSearchSpace,
        n_initial: int = 10,
        kernel: Optional[Any] = None,
        acquisition: Optional[AcquisitionFunction] = None,
        noise_variance: float = 1e-4,
        random_seed: int = 42,
    ):
        self.search_space = search_space
        self.n_initial = n_initial
        self.kernel = kernel or MaternKernel(nu=2.5)
        self.acquisition = acquisition or ExpectedImprovement()
        self.noise_variance = noise_variance
        self.seed = random_seed
        self._rng = random.Random(random_seed)
        self._torch_rng = torch.Generator()
        self._torch_rng.manual_seed(random_seed)

        self.gp = GaussianProcess(
            kernel=self.kernel,
            noise_variance=noise_variance,
        )
        self._observations: List[Tuple[Dict[str, Any], float]] = []
        self._X_obs: Optional[torch.Tensor] = None
        self._y_obs: Optional[torch.Tensor] = None
        self._best_value: float = float("inf")
        self._best_params: Dict[str, Any] = {}
        self._candidate_cache: List[Dict[str, Any]] = []

    def _config_to_tensor(self, config: Dict[str, Any]) -> torch.Tensor:
        """Convert a config dict to a normalized tensor."""
        values = []
        for name in self.search_space.get_param_names():
            dist = self.search_space.get_distribution(name)
            val = config.get(name, dist.default if dist.default else 0)

            if dist.param_type == ParamType.CATEGORICAL:
                if dist.choices:
                    idx = dist.choices.index(val) if val in dist.choices else 0
                    values.append(idx / max(1, len(dist.choices) - 1))
                else:
                    values.append(0.0)
            elif dist.param_type == ParamType.INTEGER:
                low = dist.low if dist.low is not None else 0
                high = dist.high if dist.high is not None else 10
                if high > low:
                    values.append((val - low) / (high - low))
                else:
                    values.append(0.5)
            else:
                low = dist.low if dist.low is not None else 0.0
                high = dist.high if dist.high is not None else 1.0
                if high > low:
                    if dist.log_scale and low > 0:
                        log_low = math.log(low)
                        log_high = math.log(high)
                        val_log = math.log(max(low, val))
                        values.append((val_log - log_low) / (log_high - log_low))
                    else:
                        values.append((val - low) / (high - low))
                else:
                    values.append(0.5)

        return torch.tensor(values, dtype=torch.float32)

    def _generate_candidates(self, n_candidates: int = 1000) -> torch.Tensor:
        """Generate candidate configurations."""
        candidates = []
        for _ in range(n_candidates):
            config = self.search_space.sample()
            candidates.append(self._config_to_tensor(config))
        if candidates:
            return torch.stack(candidates)
        return torch.zeros(n_candidates, self.search_space.size())

    def suggest(self) -> Dict[str, Any]:
        """Suggest the next configuration to evaluate.

        Returns:
            Dictionary of hyperparameters.
        """
        n_obs = len(self._observations)

        if n_obs < self.n_initial:
            config = self.search_space.sample()
            self._candidate_cache.append(config)
            return config

        X_obs_tensor = torch.stack([self._config_to_tensor(c) for c, _ in self._observations])
        y_obs_tensor = torch.tensor([v for _, v in self._observations], dtype=torch.float32)

        self.gp.fit(X_obs_tensor, y_obs_tensor)

        candidates = self._generate_candidates(n_candidates=2000)
        with torch.no_grad():
            mean, std = self.gp.predict(candidates, return_std=True)
            acquisition_values = self.acquisition(mean, std, self._best_value)

        best_idx = torch.argmax(acquisition_values).item()
        best_candidate = candidates[best_idx]

        config = self._tensor_to_config(best_candidate)
        self._candidate_cache.append(config)
        return config

    def _tensor_to_config(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert a normalized tensor back to a config dict."""
        config = {}
        names = self.search_space.get_param_names()
        for i, name in enumerate(names):
            dist = self.search_space.get_distribution(name)
            normalized = tensor[i].item()

            if dist.param_type == ParamType.CATEGORICAL:
                if dist.choices:
                    idx = int(round(normalized * (len(dist.choices) - 1)))
                    idx = max(0, min(len(dist.choices) - 1, idx))
                    config[name] = dist.choices[idx]
                else:
                    config[name] = dist.default
            elif dist.param_type == ParamType.INTEGER:
                low = int(dist.low) if dist.low is not None else 0
                high = int(dist.high) if dist.high is not None else 10
                if dist.log_scale:
                    log_low = math.log(max(1, low))
                    log_high = math.log(high + 1)
                    val_log = log_low + normalized * (log_high - log_low)
                    val = math.exp(val_log)
                    config[name] = max(low, min(high, int(round(val))))
                else:
                    val = low + normalized * (high - low)
                    config[name] = max(low, min(high, int(round(val))))
            elif dist.param_type == ParamType.DISCRETE:
                low = dist.low if dist.low is not None else 0.0
                high = dist.high if dist.high is not None else 1.0
                if dist.log_scale:
                    log_low = math.log(max(1e-10, low))
                    log_high = math.log(high)
                    val_log = log_low + normalized * (log_high - log_low)
                    val = math.exp(val_log)
                else:
                    val = low + normalized * (high - low)
                if dist.step is not None:
                    val = round(val / dist.step) * dist.step
                config[name] = max(low, min(high, val))
            else:
                low = dist.low if dist.low is not None else 0.0
                high = dist.high if dist.high is not None else 1.0
                if dist.log_scale:
                    log_low = math.log(max(1e-10, low))
                    log_high = math.log(high)
                    val_log = log_low + normalized * (log_high - log_low)
                    val = math.exp(val_log)
                else:
                    val = low + normalized * (high - low)
                config[name] = max(low, min(high, val))

        return config

    def observe(self, config: Dict[str, Any], value: float):
        """Record an observation.

        Args:
            config: Hyperparameter configuration.
            value: Observed objective value.
        """
        self._observations.append((config, value))
        if value < self._best_value:
            self._best_value = value
            self._best_params = dict(config)

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        n_trials: int = 50,
        callback: Optional[Callable[[Dict[str, Any], float, int], None]] = None,
    ) -> Dict[str, Any]:
        """Run the full Bayesian optimization.

        Args:
            objective: Function that takes params and returns scalar value.
            n_trials: Total number of trials.
            callback: Called after each trial.

        Returns:
            Dictionary with best params, value, and all results.
        """
        logger.info(f"Bayesian optimization: {n_trials} trials, {self.n_initial} initial")

        trials = []
        for i in range(n_trials):
            config = self.suggest()
            try:
                start = time.time()
                value = objective(config)
                duration = time.time() - start
                self.observe(config, value)

                trial_data = {
                    "params": config,
                    "value": value,
                    "duration": duration,
                    "trial_index": i,
                    "is_initial": i < self.n_initial,
                }
                trials.append(trial_data)

                if callback:
                    callback(config, value, i)

            except Exception as e:
                logger.warning(f"BO trial {i} failed: {e}")
                trials.append({
                    "params": config,
                    "value": float("inf"),
                    "error": str(e),
                    "trial_index": i,
                })

        return {
            "best_params": self._best_params,
            "best_value": self._best_value,
            "total_trials": len(trials),
            "trials": trials,
        }

    def get_observations(self) -> List[Tuple[Dict[str, Any], float]]:
        """Return all observations."""
        return list(self._observations)


# ---------------------------------------------------------------------------
# Hyperband
# ---------------------------------------------------------------------------

class Hyperband:
    """Hyperband: early stopping based hyperparameter optimization.

    Uses successive halving to allocate resources efficiently, identifying
    promising configurations early and pruning poor ones.

    Args:
        search_space: Hyperparameter search space.
        min_resource: Minimum resource (e.g., epochs) per trial.
        max_resource: Maximum resource per trial.
        reduction_factor: Reduction factor for successive halving.
        seed: Random seed.
    """

    def __init__(
        self,
        search_space: HPSearchSpace,
        min_resource: int = 1,
        max_resource: int = 81,
        reduction_factor: int = 3,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.seed = seed
        self._rng = random.Random(seed)
        self._brackets = self._compute_brackets()

    def _compute_brackets(self) -> List[List[Dict[str, Any]]]:
        """Compute Hyperband brackets (rungs)."""
        brackets = []
        s_max = int(math.log(self.max_resource / self.min_resource, self.reduction_factor))

        for s in range(s_max + 1):
            bracket = []
            n_i = int(
                math.ceil(
                    (s_max + 1) * (self.reduction_factor ** s) / (s + 1)
                )
            )
            r_i = self.max_resource * (self.reduction_factor ** (-s))

            for i in range(s + 1):
                n_i_val = int(n_i * (self.reduction_factor ** (-i)))
                r_i_val = int(r_i * (self.reduction_factor ** i))
                r_i_val = max(self.min_resource, min(self.max_resource, r_i_val))
                n_i_val = max(1, n_i_val)
                bracket.append({
                    "n_configs": n_i_val,
                    "resource": r_i_val,
                    "bracket_id": s,
                    "rung": i,
                })
            brackets.append(bracket)
        return brackets

    def optimize(
        self,
        objective: Callable[[Dict[str, Any], int], float],
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run Hyperband optimization.

        Args:
            objective: Function(config, resource) -> metric.
            callback: Optional callback.

        Returns:
            Dictionary with best params and results.
        """
        logger.info(f"Hyperband: {len(self._brackets)} brackets")

        all_trials = []
        best_value = float("inf")
        best_params = {}

        for bracket in self._brackets:
            bracket_id = bracket[0]["bracket_id"]
            logger.info(f"  Bracket {bracket_id}: {len(bracket)} rungs")

            configs = [self.search_space.sample() for _ in range(bracket[0]["n_configs"])]
            config_metrics: Dict[str, float] = {}

            for rung in bracket:
                n_configs = min(rung["n_configs"], len(configs))
                resource = rung["resource"]

                evaluated = []
                for config in configs[:n_configs]:
                    try:
                        metric = objective(config, resource)
                        config_metrics[str(config)] = metric
                        evaluated.append((config, metric))
                        all_trials.append({
                            "params": config,
                            "metric": metric,
                            "resource": resource,
                            "bracket": bracket_id,
                            "rung": rung["rung"],
                        })
                    except Exception as e:
                        logger.debug(f"Hyperband eval failed: {e}")

                if not evaluated:
                    break

                evaluated.sort(key=lambda x: x[1])
                n_keep = max(1, len(evaluated) // self.reduction_factor)
                configs = [c for c, _ in evaluated[:n_keep]]

                best_in_rung = evaluated[0]
                if best_in_rung[1] < best_value:
                    best_value = best_in_rung[1]
                    best_params = best_in_rung[0]

        return {
            "best_params": best_params,
            "best_value": best_value,
            "total_trials": len(all_trials),
            "trials": all_trials,
        }


# ---------------------------------------------------------------------------
# ASHA (Asynchronous Successive Halving)
# ---------------------------------------------------------------------------

class ASHA:
    """Asynchronous Successive Halving Algorithm.

    Like Hyperband but runs asynchronously, allowing better resource
    utilization and faster convergence.

    Args:
        search_space: Hyperparameter search space.
        min_resource: Minimum resource per trial.
        max_resource: Maximum resource per trial.
        reduction_factor: Reduction factor.
        max_trials: Maximum total trials.
        seed: Random seed.
    """

    def __init__(
        self,
        search_space: HPSearchSpace,
        min_resource: int = 1,
        max_resource: int = 81,
        reduction_factor: int = 3,
        max_trials: int = 100,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.max_trials = max_trials
        self.seed = seed
        self._rng = random.Random(seed)
        self._rungs = self._compute_rungs()
        self._rung_configs: Dict[int, List[Tuple[Dict[str, Any], float]]] = defaultdict(list)
        self._active_configs: Dict[str, int] = {}

    def _compute_rungs(self) -> List[int]:
        """Compute resource thresholds for each rung."""
        rungs = [self.min_resource]
        current = self.min_resource
        while current < self.max_resource:
            current = int(current * self.reduction_factor)
            rungs.append(min(current, self.max_resource))
        return rungs

    def _get_next_rung(self, current_rung: int) -> Optional[int]:
        """Get the next rung after current."""
        try:
            idx = self._rungs.index(current_rung)
            if idx + 1 < len(self._rungs):
                return self._rungs[idx + 1]
        except ValueError:
            pass
        return None

    def optimize(
        self,
        objective: Callable[[Dict[str, Any], int], float],
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run ASHA optimization.

        Args:
            objective: Function(config, resource) -> metric.
            callback: Optional callback.

        Returns:
            Dictionary with best params and results.
        """
        logger.info(f"ASHA: {len(self._rungs)} rungs, max {self.max_trials} trials")

        all_trials = []
        best_value = float("inf")
        best_params = {}
        total_trials = 0

        rung_queue: deque = deque()
        rung_queue.append((self.search_space.sample(), self._rungs[0]))

        while total_trials < self.max_trials and rung_queue:
            config, resource = rung_queue.popleft()

            try:
                metric = objective(config, resource)
                total_trials += 1

                all_trials.append({
                    "params": config,
                    "metric": metric,
                    "resource": resource,
                    "trial": total_trials,
                })

                if metric < best_value:
                    best_value = metric
                    best_params = dict(config)

                rung_idx = self._rungs.index(resource)
                self._rung_configs[rung_idx].append((config, metric))

                next_rung = self._get_next_rung(resource)
                if next_rung is not None and total_trials < self.max_trials:
                    rung_configs = self._rung_configs[rung_idx]
                    rung_configs.sort(key=lambda x: x[1])
                    n_keep = max(1, len(rung_configs) // self.reduction_factor)

                    for c, m in rung_configs[:n_keep]:
                        config_key = json.dumps(c, sort_keys=True, default=str)
                        if self._active_configs.get(config_key, -1) < next_rung:
                            rung_queue.append((c, next_rung))
                            self._active_configs[config_key] = next_rung

                if total_trials < self.max_trials:
                    new_config = self.search_space.sample()
                    rung_queue.append((new_config, self._rungs[0]))

                if callback:
                    callback(config, metric, resource)

            except Exception as e:
                logger.debug(f"ASHA trial failed: {e}")

        return {
            "best_params": best_params,
            "best_value": best_value,
            "total_trials": total_trials,
            "trials": all_trials,
        }


# ---------------------------------------------------------------------------
# Population Based Training (PBT)
# ---------------------------------------------------------------------------

class PopulationBasedTraining:
    """Population Based Training with mutation and selection.

    Maintains a population of models trained in parallel, periodically
    exploiting (copying) good performers and exploring (mutating)
    hyperparameters.

    Args:
        search_space: Hyperparameter search space.
        population_size: Number of models in the population.
        num_generations: Number of PBT generations.
        steps_per_generation: Training steps between PBT operations.
        exploit_fraction: Fraction of population to exploit from.
        mutation_rate: Probability of mutating each hyperparameter.
        seed: Random seed.
    """

    def __init__(
        self,
        search_space: HPSearchSpace,
        population_size: int = 16,
        num_generations: int = 10,
        steps_per_generation: int = 100,
        exploit_fraction: float = 0.25,
        mutation_rate: float = 0.2,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.num_generations = num_generations
        self.steps_per_generation = steps_per_generation
        self.exploit_fraction = exploit_fraction
        self.mutation_rate = mutation_rate
        self.seed = seed
        self._rng = random.Random(seed)

    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize the population with random configurations."""
        return [self.search_space.sample() for _ in range(self.population_size)]

    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate hyperparameters."""
        mutated = dict(params)
        for name, dist in self.search_space._distributions.items():
            if self._rng.random() < self.mutation_rate:
                if dist.param_type == ParamType.CATEGORICAL:
                    if dist.choices:
                        new_val = self._rng.choice(dist.choices)
                        mutated[name] = new_val
                elif dist.param_type == ParamType.INTEGER:
                    low = int(dist.low) if dist.low is not None else 0
                    high = int(dist.high) if dist.high is not None else 10
                    current = params.get(name, (low + high) // 2)
                    delta = self._rng.randint(1, max(1, (high - low) // 5))
                    if self._rng.random() < 0.5:
                        new_val = current + delta
                    else:
                        new_val = current - delta
                    mutated[name] = max(low, min(high, new_val))
                elif dist.param_type in (ParamType.CONTINUOUS, ParamType.DISCRETE):
                    low = dist.low if dist.low is not None else 0.0
                    high = dist.high if dist.high is not None else 1.0
                    current = params.get(name, (low + high) / 2)
                    scale = (high - low) * 0.2
                    if dist.log_scale and low > 0:
                        log_current = math.log(current)
                        log_scale = scale / current
                        log_new = log_current + self._rng.gauss(0, log_scale)
                        new_val = math.exp(log_new)
                    else:
                        new_val = current + self._rng.gauss(0, scale)
                    mutated[name] = max(low, min(high, new_val))
        return mutated

    def _exploit(
        self,
        population: List[Dict[str, Any]],
        metrics: List[float],
    ) -> List[Dict[str, Any]]:
        """Exploit phase: replace bottom performers with top performers' params."""
        new_population = []
        n_exploit = max(1, int(self.population_size * self.exploit_fraction))

        paired = list(zip(population, metrics))
        paired.sort(key=lambda x: x[1])

        for i, (params, metric) in enumerate(paired):
            if i < n_exploit:
                top_idx = self._rng.randint(
                    self.population_size - n_exploit, self.population_size
                )
                new_params = dict(paired[top_idx][0])
                new_params = self._mutate(new_params)
                new_population.append(new_params)
            else:
                new_population.append(dict(params))

        return new_population

    def optimize(
        self,
        train_fn: Callable[[Dict[str, Any], int], float],
        eval_fn: Callable[[Dict[str, Any]], float],
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run PBT optimization.

        Args:
            train_fn: Function(config, steps) -> intermediate metric.
            eval_fn: Function(config) -> evaluation metric.
            callback: Optional callback.

        Returns:
            Dictionary with best params and results.
        """
        logger.info(
            f"PBT: population={self.population_size}, "
            f"generations={self.num_generations}"
        )

        population = self._initialize_population()
        history = []
        best_value = float("inf")
        best_params = {}

        for gen in range(self.num_generations):
            metrics = []
            for i, params in enumerate(population):
                try:
                    train_metric = train_fn(params, self.steps_per_generation)
                    eval_metric = eval_fn(params)
                    metrics.append(eval_metric)
                    history.append({
                        "generation": gen,
                        "member": i,
                        "params": params,
                        "train_metric": train_metric,
                        "eval_metric": eval_metric,
                    })
                    if eval_metric < best_value:
                        best_value = eval_metric
                        best_params = dict(params)
                except Exception as e:
                    logger.debug(f"PBT member {i} gen {gen} failed: {e}")
                    metrics.append(float("inf"))

            population = self._exploit(population, metrics)

            avg_metric = sum(m for m in metrics if m < float("inf")) / max(1, sum(1 for m in metrics if m < float("inf")))
            logger.info(f"  Gen {gen}: avg_metric={avg_metric:.4f}, best={best_value:.4f}")

            if callback:
                callback(population, metrics, gen)

        return {
            "best_params": best_params,
            "best_value": best_value,
            "total_generations": self.num_generations,
            "history": history,
        }


# ---------------------------------------------------------------------------
# CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
# ---------------------------------------------------------------------------

class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy, implemented from scratch.

    A powerful black-box optimization algorithm that adapts the full
    covariance matrix of the search distribution.

    Args:
        search_space: Hyperparameter search space.
        population_size: Population size (lambda). Auto-computed if None.
        max_generations: Maximum number of generations.
        sigma: Initial step size.
        seed: Random seed.
    """

    def __init__(
        self,
        search_space: HPSearchSpace,
        population_size: Optional[int] = None,
        max_generations: int = 100,
        sigma: Optional[float] = None,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.max_generations = max_generations
        self.seed = seed
        self._rng = random.Random(seed)
        self._torch_rng = torch.Generator()
        self._torch_rng.manual_seed(seed)

        self.dim = search_space.size()
        self.population_size = population_size or (4 + int(3 * math.log(self.dim)))
        self.lambda_ = self.population_size
        self.mu = self.lambda_ // 2

        self.weights = torch.log(
            torch.arange(1, self.mu + 1, dtype=torch.float64)
        )
        self.weights = (self.weights + 1e-10 - self.weights.mean()) / self.weights.std()
        self.weights = self.weights / self.weights.sum()

        self.mu_eff = 1.0 / (self.weights ** 2).sum().item()

        self.c_sigma = (self.mu_eff + 2.0) / (self.dim + self.mu_eff + 5.0)
        self.c_c = (4.0 + self.mu_eff / self.dim) / (
            self.dim + 4.0 + 2.0 * self.mu_eff / self.dim
        )
        self.c_1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c_1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((self.dim + 2.0) ** 2 + self.mu_eff),
        )
        self.d_sigma = (
            1.0 + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0)) / (self.dim + 1.0) - 1.0)
            + self.c_sigma
        ) if self.mu_eff > 1 else 1.0 + self.c_sigma

        if sigma is None:
            self.sigma = 0.3
        else:
            self.sigma = sigma

        self._mean: Optional[torch.Tensor] = None
        self._C: Optional[torch.Tensor] = None
        self._pc: Optional[torch.Tensor] = None
        self._ps: Optional[torch.Tensor] = None
        self._eig_decomp_age = 0

    def _initialize(self, x0: Optional[Dict[str, Any]] = None):
        """Initialize CMA-ES state."""
        if x0 is not None:
            self._mean = self._config_to_tensor(x0).double()
        else:
            default_config = self.search_space.get_default_config()
            self._mean = self._config_to_tensor(default_config).double()

        self._C = torch.eye(self.dim, dtype=torch.float64)
        self._pc = torch.zeros(self.dim, dtype=torch.float64)
        self._ps = torch.zeros(self.dim, dtype=torch.float64)

    def _config_to_tensor(self, config: Dict[str, Any]) -> torch.Tensor:
        """Convert config dict to normalized tensor."""
        values = []
        for name in self.search_space.get_param_names():
            dist = self.search_space.get_distribution(name)
            val = config.get(name, dist.default if dist.default else 0)
            if dist.param_type == ParamType.CATEGORICAL:
                if dist.choices:
                    idx = dist.choices.index(val) if val in dist.choices else 0
                    values.append(idx / max(1, len(dist.choices) - 1))
                else:
                    values.append(0.0)
            else:
                low = dist.low if dist.low is not None else 0.0
                high = dist.high if dist.high is not None else 1.0
                if high > low:
                    values.append((val - low) / (high - low))
                else:
                    values.append(0.5)
        return torch.tensor(values, dtype=torch.float32)

    def _tensor_to_config(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert tensor back to config dict."""
        config = {}
        names = self.search_space.get_param_names()
        for i, name in enumerate(names):
            dist = self.search_space.get_distribution(name)
            val = float(tensor[i].item())
            val = max(0.0, min(1.0, val))

            if dist.param_type == ParamType.CATEGORICAL:
                if dist.choices:
                    idx = int(round(val * (len(dist.choices) - 1)))
                    idx = max(0, min(len(dist.choices) - 1, idx))
                    config[name] = dist.choices[idx]
                else:
                    config[name] = dist.default
            elif dist.param_type == ParamType.INTEGER:
                low = int(dist.low) if dist.low is not None else 0
                high = int(dist.high) if dist.high is not None else 10
                int_val = low + val * (high - low)
                config[name] = max(low, min(high, int(round(int_val))))
            elif dist.param_type == ParamType.DISCRETE:
                low = dist.low if dist.low is not None else 0.0
                high = dist.high if dist.high is not None else 1.0
                real_val = low + val * (high - low)
                if dist.step is not None:
                    real_val = round(real_val / dist.step) * dist.step
                config[name] = max(low, min(high, real_val))
            else:
                low = dist.low if dist.low is not None else 0.0
                high = dist.high if dist.high is not None else 1.0
                config[name] = max(low, min(high, low + val * (high - low)))

        return config

    def _sample_population(self) -> torch.Tensor:
        """Sample population from current distribution."""
        try:
            L = torch.linalg.cholesky(self._C)
        except torch.linalg.LinAlgError:
            jitter = torch.eye(self.dim, dtype=torch.float64) * 1e-8
            L = torch.linalg.cholesky(self._C + jitter)

        z = torch.randn(self.lambda_, self.dim, dtype=torch.float64, generator=self._torch_rng)
        population = self._mean.unsqueeze(0) + self.sigma * (z @ L.T)
        return population

    def _update(self, population: torch.Tensor, fitness: List[float]):
        """Update CMA-ES distribution from population and fitness."""
        fitness_tensor = torch.tensor(fitness, dtype=torch.float64)
        sorted_indices = torch.argsort(fitness_tensor)

        x_selected = population[sorted_indices[:self.mu]]

        old_mean = self._mean.clone()
        self._mean = (self.weights.unsqueeze(0) @ x_selected).squeeze(0)

        z_selected = (x_selected - old_mean.unsqueeze(0)) / self.sigma
        z_mean = (self.weights.unsqueeze(0) @ z_selected).squeeze(0)

        self._ps = (
            (1 - self.c_sigma) * self._ps
            + math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * z_mean
        )

        h_sigma = (
            torch.norm(self._ps).item()
            / math.sqrt(1 - (1 - self.c_sigma) ** (2 * self.max_generations))
            < (2.0 + 4.0 / (self.dim + 1.0)) * math.sqrt(self.dim)
            if self.dim > 0 else False
        )

        delta_h = (1 - h_sigma) * self.c_c * (2 - self.c_c)

        self._pc = (1 - self.c_c) * self._pc + h_sigma * math.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * z_mean

        artmp = (x_selected - old_mean.unsqueeze(0)) / self.sigma
        self._C = (
            (1 - self.c_1 - self.c_mu) * self._C
            + self.c_1 * (
                self._pc.unsqueeze(1) @ self._pc.unsqueeze(0)
                + delta_h * self._C
            )
            + self.c_mu * (
                self.weights.view(-1, 1, 1) * artmp.unsqueeze(2) @ artmp.unsqueeze(1)
            ).sum(dim=0)
        )

        self._C = 0.5 * (self._C + self._C.T)

        sigma_norm = torch.norm(self._ps).item() / math.sqrt(self.dim)
        self.sigma *= math.exp(
            (self.c_sigma / self.d_sigma) * (sigma_norm - 1)
        )
        self.sigma *= math.exp(
            0.05 * self.mu_eff / (self.dim + 1) * (1 - torch.norm(self._C @ torch.eye(self.dim, dtype=torch.float64)) / self.dim)
        )
        self.sigma = max(1e-12, min(1e2, self.sigma))

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run CMA-ES optimization.

        Args:
            objective: Function(config) -> scalar (lower is better).
            callback: Optional callback.

        Returns:
            Dictionary with best params and results.
        """
        logger.info(f"CMA-ES: dim={self.dim}, pop={self.lambda_}, gens={self.max_generations}")

        self._initialize()
        all_trials = []
        best_value = float("inf")
        best_params = {}

        for gen in range(self.max_generations):
            population = self._sample_population()
            fitness = []

            for i in range(self.lambda_):
                config = self._tensor_to_config(population[i])
                try:
                    value = objective(config)
                    fitness.append(value)

                    if value < best_value:
                        best_value = value
                        best_params = dict(config)

                    all_trials.append({
                        "params": config,
                        "value": value,
                        "generation": gen,
                        "member": i,
                    })
                except Exception as e:
                    fitness.append(float("inf"))
                    logger.debug(f"CMA-ES trial failed: {e}")

            self._update(population, fitness)

            if gen % 10 == 0:
                logger.info(f"  Gen {gen}: sigma={self.sigma:.6f}, best={best_value:.4f}")

            if callback:
                callback(best_params, best_value, gen)

            if self.sigma < 1e-10:
                break

        return {
            "best_params": best_params,
            "best_value": best_value,
            "total_trials": len(all_trials),
            "final_sigma": self.sigma,
            "trials": all_trials,
        }


# ---------------------------------------------------------------------------
# TPE (Tree-structured Parzen Estimator) Sampler
# ---------------------------------------------------------------------------

class OptunaStyleSampler:
    """Tree-structured Parzen Estimator (TPE) sampler.

    Models the conditional distributions of good and bad parameter
    configurations using kernel density estimation, sampling new
    configurations that are more likely to be in the good distribution.

    Args:
        search_space: Hyperparameter search space.
        n_startup_trials: Random trials before using TPE.
        n_ei_candidates: Number of candidates for EI computation.
        gamma: Fraction of trials considered 'good'.
        seed: Random seed.
    """

    def __init__(
        self,
        search_space: HPSearchSpace,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 100,
        gamma: float = 0.25,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.seed = seed
        self._rng = random.Random(seed)

        self._observations: List[Tuple[Dict[str, Any], float]] = []
        self._param_distributions: Dict[str, List[float]] = defaultdict(list)

    def _split_observations(self) -> Tuple[List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]:
        """Split observations into good and bad based on gamma quantile."""
        if not self._observations:
            return [], []

        sorted_obs = sorted(self._observations, key=lambda x: x[1])
        n_good = max(1, int(len(sorted_obs) * self.gamma))

        good = sorted_obs[:n_good]
        bad = sorted_obs[n_good:]
        return good, bad

    def _log_likelihood(self, value: float, observations: List[float], param_name: str) -> float:
        """Compute log-likelihood of value under kernel density estimate."""
        if not observations:
            return 0.0

        dist = self.search_space.get_distribution(param_name)
        if dist is None:
            return 0.0

        n = len(observations)
        kernel_bandwidth = max(1e-10, (max(observations) - min(observations)) / max(1, n))

        l_values = sorted(observations)
        sigma = math.sqrt(max(1e-10, sum((v - sum(l_values)/n)**2 for v in l_values) / n))
        bandwidth = sigma / max(1, n ** 0.5)
        bandwidth = max(bandwidth, 1e-10)

        log_likelihood = float("-inf")
        for obs in l_values:
            log_p = -0.5 * ((value - obs) / bandwidth) ** 2 - math.log(bandwidth) - 0.5 * math.log(2 * math.pi)
            log_likelihood = math.logaddexp(log_likelihood, log_p)

        return log_likelihood

    def suggest(self) -> Dict[str, Any]:
        """Suggest the next configuration using TPE.

        Returns:
            Dictionary of hyperparameters.
        """
        if len(self._observations) < self.n_startup_trials:
            return self.search_space.sample()

        good, bad = self._split_observations()

        if not good:
            return self.search_space.sample()

        config = {}
        for name in self.search_space.get_param_names():
            dist = self.search_space.get_distribution(name)

            if dist.param_type == ParamType.CATEGORICAL:
                if not dist.choices:
                    config[name] = dist.default
                    continue

                good_values = [obs[0].get(name) for obs in good if name in obs[0]]
                bad_values = [obs[0].get(name) for obs in bad if name in obs[0]]

                if not good_values:
                    config[name] = self._rng.choice(dist.choices)
                    continue

                good_counts = Counter(good_values)
                bad_counts = Counter(bad_values)

                choice_scores = []
                for choice in dist.choices:
                    g_count = good_counts.get(choice, 0) + 1
                    b_count = bad_counts.get(choice, 0) + 1
                    score = (g_count / len(good_values)) / (b_count / len(bad_values))
                    choice_scores.append(score)

                total = sum(choice_scores)
                if total > 0:
                    probs = [s / total for s in choice_scores]
                else:
                    probs = [1.0 / len(dist.choices)] * len(dist.choices)

                config[name] = self._rng.choices(dist.choices, weights=probs, k=1)[0]

            elif dist.param_type == ParamType.INTEGER:
                good_vals = [float(obs[0].get(name, dist.low or 0)) for obs in good if name in obs[0]]
                bad_vals = [float(obs[0].get(name, dist.low or 0)) for obs in bad if name in obs[0]]

                if good_vals:
                    val = self._sample_tpe_numeric(good_vals, bad_vals, name)
                    low = int(dist.low) if dist.low is not None else 0
                    high = int(dist.high) if dist.high is not None else 10
                    config[name] = max(low, min(high, int(round(val))))
                else:
                    config[name] = dist.sample(self._rng)

            else:
                good_vals = [float(obs[0].get(name, dist.low or 0)) for obs in good if name in obs[0]]
                bad_vals = [float(obs[0].get(name, dist.low or 0)) for obs in bad if name in obs[0]]

                if good_vals:
                    val = self._sample_tpe_numeric(good_vals, bad_vals, name)
                    low = dist.low if dist.low is not None else 0.0
                    high = dist.high if dist.high is not None else 1.0
                    config[name] = max(low, min(high, val))
                else:
                    config[name] = dist.sample(self._rng)

        return config

    def _sample_tpe_numeric(
        self,
        good_vals: List[float],
        bad_vals: List[float],
        param_name: str,
    ) -> float:
        """Sample a numeric value using TPE's EI criterion."""
        candidates = []
        for _ in range(self.n_ei_candidates):
            dist = self.search_space.get_distribution(param_name)
            candidates.append(dist.sample(self._rng))

        if not candidates:
            return self.search_space.get_distribution(param_name).sample(self._rng)

        best_ei = float("-inf")
        best_val = candidates[0]

        for val in candidates:
            log_l_good = self._log_likelihood(val, good_vals, param_name)
            log_l_bad = self._log_likelihood(val, bad_vals, param_name)
            ei = log_l_good - log_l_bad

            if ei > best_ei:
                best_ei = ei
                best_val = val

        return best_val

    def observe(self, config: Dict[str, Any], value: float):
        """Record an observation."""
        self._observations.append((config, value))

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        callback: Optional[Callable[[Dict[str, Any], float, int], None]] = None,
    ) -> Dict[str, Any]:
        """Run TPE optimization.

        Args:
            objective: Function(config) -> scalar.
            n_trials: Number of trials.
            callback: Optional callback.

        Returns:
            Dictionary with best params and results.
        """
        logger.info(f"TPE: {n_trials} trials, {self.n_startup_trials} startup")

        trials = []
        best_value = float("inf")
        best_params = {}

        for i in range(n_trials):
            config = self.suggest()
            try:
                start = time.time()
                value = objective(config)
                duration = time.time() - start

                self.observe(config, value)

                if value < best_value:
                    best_value = value
                    best_params = dict(config)

                trials.append({
                    "params": config,
                    "value": value,
                    "duration": duration,
                    "trial_index": i,
                })

                if callback:
                    callback(config, value, i)

            except Exception as e:
                logger.warning(f"TPE trial {i} failed: {e}")

        return {
            "best_params": best_params,
            "best_value": best_value,
            "total_trials": len(trials),
            "trials": trials,
        }

    def get_observations(self) -> List[Tuple[Dict[str, Any], float]]:
        """Return all observations."""
        return list(self._observations)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def create_search_space_from_config(config: Dict[str, Any]) -> HPSearchSpace:
    """Create an HPSearchSpace from a configuration dictionary.

    Args:
        config: Dictionary mapping param names to distribution specs.

    Returns:
        Configured HPSearchSpace.
    """
    space = HPSearchSpace()
    for name, spec in config.items():
        if isinstance(spec, ParamDistribution):
            space.add(spec)
        elif isinstance(spec, dict):
            space.add(ParamDistribution.from_dict({"name": name, **spec}))
    return space


def print_hp_results(results: Dict[str, Any], top_k: int = 5):
    """Pretty-print hyperparameter optimization results.

    Args:
        results: Results dictionary from an optimizer.
        top_k: Number of top trials to display.
    """
    best = results.get("best_params", {})
    best_val = results.get("best_value", float("inf"))

    print(f"\n{'=' * 60}")
    print(f"Best Value: {best_val:.6f}")
    print(f"Best Params:")
    for name, value in best.items():
        print(f"  {name}: {value}")
    print(f"{'=' * 60}")

    trials = results.get("trials", [])
    if trials:
        obj_key = "value" if "value" in trials[0] else "metric"
        completed = [t for t in trials if t.get(obj_key) is not None]
        completed.sort(key=lambda t: t[obj_key])

        print(f"\nTop {min(top_k, len(completed))} trials:")
        for i, trial in enumerate(completed[:top_k]):
            params = trial.get("params", {})
            val = trial.get(obj_key, float("inf"))
            duration = trial.get("duration", 0)
            print(f"  {i+1}. {obj_key}={val:.4f} ({duration:.1f}s)")
            for name, value in params.items():
                print(f"     {name}: {value}")


def save_hp_results(results: Dict[str, Any], path: str):
    """Save hyperparameter optimization results to JSON.

    Args:
        results: Results dictionary.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    serializable = copy.deepcopy(results)
    trials = serializable.get("trials", [])
    for trial in trials:
        params = trial.get("params", {})
        for k, v in params.items():
            if not isinstance(v, (int, float, str, bool, list, type(None))):
                params[k] = str(v)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"HP results saved to {path}")


def load_hp_results(path: str) -> Dict[str, Any]:
    """Load hyperparameter optimization results from JSON.

    Args:
        path: Input file path.

    Returns:
        Results dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def get_recommended_hp_config(
    model_size: str = "medium",
    dataset_size: str = "medium",
) -> Dict[str, Any]:
    """Get recommended hyperparameters based on model and dataset size.

    Args:
        model_size: 'small', 'medium', or 'large'.
        dataset_size: 'small', 'medium', or 'large'.

    Returns:
        Dictionary of recommended hyperparameters.
    """
    configs = {
        ("small", "small"): {
            "learning_rate": 3e-4,
            "batch_size": 16,
            "weight_decay": 1e-4,
            "warmup_ratio": 0.06,
            "dropout": 0.1,
            "grad_clip": 1.0,
        },
        ("small", "medium"): {
            "learning_rate": 2e-4,
            "batch_size": 32,
            "weight_decay": 1e-4,
            "warmup_ratio": 0.06,
            "dropout": 0.1,
            "grad_clip": 1.0,
        },
        ("small", "large"): {
            "learning_rate": 1e-4,
            "batch_size": 64,
            "weight_decay": 1e-5,
            "warmup_ratio": 0.1,
            "dropout": 0.1,
            "grad_clip": 1.0,
        },
        ("medium", "small"): {
            "learning_rate": 1e-4,
            "batch_size": 16,
            "weight_decay": 1e-4,
            "warmup_ratio": 0.06,
            "dropout": 0.1,
            "grad_clip": 1.0,
        },
        ("medium", "medium"): {
            "learning_rate": 5e-5,
            "batch_size": 32,
            "weight_decay": 1e-4,
            "warmup_ratio": 0.1,
            "dropout": 0.1,
            "grad_clip": 1.0,
        },
        ("medium", "large"): {
            "learning_rate": 3e-5,
            "batch_size": 64,
            "weight_decay": 1e-5,
            "warmup_ratio": 0.1,
            "dropout": 0.05,
            "grad_clip": 1.0,
        },
        ("large", "small"): {
            "learning_rate": 5e-5,
            "batch_size": 8,
            "weight_decay": 1e-4,
            "warmup_ratio": 0.06,
            "dropout": 0.1,
            "grad_clip": 1.0,
        },
        ("large", "medium"): {
            "learning_rate": 3e-5,
            "batch_size": 16,
            "weight_decay": 1e-4,
            "warmup_ratio": 0.1,
            "dropout": 0.1,
            "grad_clip": 1.0,
        },
        ("large", "large"): {
            "learning_rate": 1e-5,
            "batch_size": 32,
            "weight_decay": 1e-5,
            "warmup_ratio": 0.15,
            "dropout": 0.05,
            "grad_clip": 1.0,
        },
    }

    return configs.get(
        (model_size, dataset_size),
        configs.get(("medium", "medium"), {}),
    )
