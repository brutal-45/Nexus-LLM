"""Hyperparameter Management and Search for Nexus-LLM.

Provides hyperparameter space definition, search strategies
(grid search, random search, Bayesian optimization), and trial
management for systematic hyperparameter optimization.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SearchStrategy(str, Enum):
    """Hyperparameter search strategy."""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


class ParamType(str, Enum):
    """Type of a hyperparameter."""
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOL = "bool"


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """Specification for a single hyperparameter."""
    name: str
    param_type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False

    def sample(self) -> Any:
        """Sample a random value for this parameter.

        Returns:
            A randomly sampled value within the parameter's range.
        """
        if self.param_type == ParamType.INT:
            lo = int(self.low or 0)
            hi = int(self.high or 1)
            return random.randint(lo, hi)
        elif self.param_type == ParamType.FLOAT:
            lo = self.low or 0.0
            hi = self.high or 1.0
            if self.log_scale and lo > 0:
                return math.exp(random.uniform(math.log(lo), math.log(hi)))
            return random.uniform(lo, hi)
        elif self.param_type == ParamType.CATEGORICAL:
            if self.choices:
                return random.choice(self.choices)
            raise ValueError(f"Categorical param '{self.name}' has no choices")
        elif self.param_type == ParamType.BOOL:
            return random.choice([True, False])
        raise ValueError(f"Unknown param type: {self.param_type}")

    def grid_values(self, num_points: int = 10) -> List[Any]:
        """Generate evenly-spaced values for grid search.

        Args:
            num_points: Number of points for continuous params.

        Returns:
            List of parameter values.
        """
        if self.param_type == ParamType.INT:
            lo = int(self.low or 0)
            hi = int(self.high or 1)
            step = max(1, (hi - lo) // max(num_points, 1))
            return list(range(lo, hi + 1, step))
        elif self.param_type == ParamType.FLOAT:
            lo = self.low or 0.0
            hi = self.high or 1.0
            if self.log_scale and lo > 0:
                logs = [math.exp(x) for x in itertools.islice(
                    (math.log(lo) + i * (math.log(hi) - math.log(lo)) / (num_points - 1)
                     for i in range(num_points)), num_points)]
                return logs
            return [lo + i * (hi - lo) / (num_points - 1) for i in range(num_points)]
        elif self.param_type == ParamType.CATEGORICAL:
            return self.choices or []
        elif self.param_type == ParamType.BOOL:
            return [True, False]
        return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "param_type": self.param_type.value,
            "low": self.low,
            "high": self.high,
            "choices": self.choices,
            "log_scale": self.log_scale,
        }


@dataclass
class ParamSpace:
    """A collection of hyperparameter specifications."""
    params: Dict[str, ParamSpec] = field(default_factory=dict)

    def add_int(self, name: str, low: int, high: int, log_scale: bool = False) -> "ParamSpace":
        """Add an integer parameter.

        Args:
            name: Parameter name.
            low: Minimum value.
            high: Maximum value.
            log_scale: Whether to use log-scale sampling.

        Returns:
            Self for chaining.
        """
        self.params[name] = ParamSpec(
            name=name, param_type=ParamType.INT,
            low=float(low), high=float(high), log_scale=log_scale,
        )
        return self

    def add_float(self, name: str, low: float, high: float, log_scale: bool = False) -> "ParamSpace":
        """Add a float parameter.

        Args:
            name: Parameter name.
            low: Minimum value.
            high: Maximum value.
            log_scale: Whether to use log-scale sampling.

        Returns:
            Self for chaining.
        """
        self.params[name] = ParamSpec(
            name=name, param_type=ParamType.FLOAT,
            low=low, high=high, log_scale=log_scale,
        )
        return self

    def add_categorical(self, name: str, choices: List[Any]) -> "ParamSpace":
        """Add a categorical parameter.

        Args:
            name: Parameter name.
            choices: List of possible values.

        Returns:
            Self for chaining.
        """
        self.params[name] = ParamSpec(
            name=name, param_type=ParamType.CATEGORICAL, choices=choices,
        )
        return self

    def add_bool(self, name: str) -> "ParamSpace":
        """Add a boolean parameter.

        Args:
            name: Parameter name.

        Returns:
            Self for chaining.
        """
        self.params[name] = ParamSpec(name=name, param_type=ParamType.BOOL)
        return self

    def sample(self) -> Dict[str, Any]:
        """Sample a random configuration from the space.

        Returns:
            Dictionary of parameter name -> sampled value.
        """
        return {name: spec.sample() for name, spec in self.params.items()}

    def grid(self, num_points: int = 10) -> Generator[Dict[str, Any], None, None]:
        """Generate grid search configurations.

        Args:
            num_points: Number of points per continuous dimension.

        Yields:
            Parameter configuration dictionaries.
        """
        param_names = list(self.params.keys())
        value_lists = [self.params[name].grid_values(num_points) for name in param_names]

        for combo in itertools.product(*value_lists):
            yield dict(zip(param_names, combo))

    def grid_size(self, num_points: int = 10) -> int:
        """Calculate the total number of configurations in the grid.

        Args:
            num_points: Points per continuous dimension.

        Returns:
            Total grid size.
        """
        total = 1
        for spec in self.params.values():
            total *= len(spec.grid_values(num_points))
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {name: spec.to_dict() for name, spec in self.params.items()}


# ---------------------------------------------------------------------------
# Trial tracking
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """A single hyperparameter trial."""
    trial_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed, pruned
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.trial_id:
            self.trial_id = str(uuid.uuid4())[:12]

    @property
    def objective_value(self) -> Optional[float]:
        """Get the primary objective metric value."""
        if self.metrics:
            return next(iter(self.metrics.values()))
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "metrics": self.metrics,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Search config
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""
    strategy: SearchStrategy = SearchStrategy.RANDOM
    num_trials: int = 50
    grid_points: int = 10  # Points per dimension for grid search
    objective_metric: str = "loss"
    objective_mode: str = "min"  # 'min' or 'max'
    early_stopping: bool = False
    early_stopping_patience: int = 10
    seed: Optional[int] = None
    max_concurrent: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "num_trials": self.num_trials,
            "grid_points": self.grid_points,
            "objective_metric": self.objective_metric,
            "objective_mode": self.objective_mode,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "seed": self.seed,
            "max_concurrent": self.max_concurrent,
        }


# ---------------------------------------------------------------------------
# Hyperparameter Search
# ---------------------------------------------------------------------------

class HyperparameterSearch:
    """Systematic hyperparameter search and optimization.

    Supports grid search, random search, and a simplified Bayesian
    optimization approach for finding optimal hyperparameters.

    Example::

        space = ParamSpace()
        space.add_float("learning_rate", 1e-5, 1e-3, log_scale=True)
        space.add_int("batch_size", 8, 64)
        space.add_categorical("optimizer", ["adam", "sgd", "adamw"])

        search = HyperparameterSearch(
            space=space,
            config=SearchConfig(strategy=SearchStrategy.RANDOM, num_trials=20),
        )

        # Define objective function
        def objective(params):
            model = train(params)
            return {"loss": evaluate(model)}

        results = search.run(objective)
        best = search.get_best_trial()
        print(f"Best params: {best.params}")
    """

    def __init__(
        self,
        space: ParamSpace,
        config: Optional[SearchConfig] = None,
    ) -> None:
        self._space = space
        self._config = config or SearchConfig()
        self._trials: List[Trial] = []
        self._best_trial: Optional[Trial] = None

        if self._config.seed is not None:
            random.seed(self._config.seed)

    @property
    def trials(self) -> List[Trial]:
        """All completed and pending trials."""
        return list(self._trials)

    @property
    def best_trial(self) -> Optional[Trial]:
        """Trial with the best objective value."""
        return self._best_trial

    @property
    def space(self) -> ParamSpace:
        """The hyperparameter search space."""
        return self._space

    # ------------------------------------------------------------------
    # Run search
    # ------------------------------------------------------------------

    def run(
        self,
        objective: Callable[[Dict[str, Any]], Dict[str, float]],
        callback: Optional[Callable[[Trial], None]] = None,
    ) -> List[Trial]:
        """Execute the hyperparameter search.

        Args:
            objective: Function that takes params dict and returns metrics dict.
            callback: Optional callback after each trial completes.

        Returns:
            List of all completed trials.
        """
        if self._config.strategy == SearchStrategy.GRID:
            return self._run_grid(objective, callback)
        elif self._config.strategy == SearchStrategy.RANDOM:
            return self._run_random(objective, callback)
        elif self._config.strategy == SearchStrategy.BAYESIAN:
            return self._run_bayesian(objective, callback)
        else:
            raise ValueError(f"Unknown strategy: {self._config.strategy}")

    def _run_grid(
        self,
        objective: Callable,
        callback: Optional[Callable] = None,
    ) -> List[Trial]:
        """Execute grid search."""
        count = 0
        no_improve = 0

        for params in self._space.grid(num_points=self._config.grid_points):
            if count >= self._config.num_trials:
                break

            trial = self._evaluate_trial(params, objective, count)
            self._trials.append(trial)
            self._update_best(trial)

            if callback:
                callback(trial)

            count += 1

            # Early stopping
            if self._config.early_stopping:
                if trial.status == "completed" and not self._is_better(trial):
                    no_improve += 1
                else:
                    no_improve = 0
                if no_improve >= self._config.early_stopping_patience:
                    break

        return self._trials

    def _run_random(
        self,
        objective: Callable,
        callback: Optional[Callable] = None,
    ) -> List[Trial]:
        """Execute random search."""
        no_improve = 0

        for i in range(self._config.num_trials):
            params = self._space.sample()
            trial = self._evaluate_trial(params, objective, i)
            self._trials.append(trial)
            self._update_best(trial)

            if callback:
                callback(trial)

            # Early stopping
            if self._config.early_stopping:
                if trial.status == "completed" and not self._is_better(trial):
                    no_improve += 1
                else:
                    no_improve = 0
                if no_improve >= self._config.early_stopping_patience:
                    break

        return self._trials

    def _run_bayesian(
        self,
        objective: Callable,
        callback: Optional[Callable] = None,
    ) -> List[Trial]:
        """Execute simplified Bayesian optimization.

        Uses a TPE-like approach: builds a model of good vs bad regions
        and samples from promising areas.
        """
        # Start with random exploration
        n_initial = min(5, self._config.num_trials // 3)
        for i in range(n_initial):
            params = self._space.sample()
            trial = self._evaluate_trial(params, objective, i)
            self._trials.append(trial)
            self._update_best(trial)
            if callback:
                callback(trial)

        # Bayesian iterations
        for i in range(n_initial, self._config.num_trials):
            params = self._bayesian_sample()
            trial = self._evaluate_trial(params, objective, i)
            self._trials.append(trial)
            self._update_best(trial)
            if callback:
                callback(trial)

        return self._trials

    def _bayesian_sample(self) -> Dict[str, Any]:
        """Sample using a simplified Bayesian strategy.

        Fits Gaussian distributions to the best-performing region
        and samples from them with some exploration noise.
        """
        completed = [t for t in self._trials if t.status == "completed" and t.objective_value is not None]
        if not completed:
            return self._space.sample()

        # Sort trials by objective
        reverse = self._config.objective_mode == "max"
        completed.sort(key=lambda t: t.objective_value or 0, reverse=reverse)

        # Use top 25% as "good" region
        n_good = max(1, len(completed) // 4)
        good_trials = completed[:n_good]

        # Compute mean and std for numeric params from good trials
        params: Dict[str, Any] = {}
        for name, spec in self._space.params.items():
            good_values = [t.params.get(name) for t in good_trials if name in t.params]
            good_values = [v for v in good_values if v is not None]

            if not good_values:
                params[name] = spec.sample()
                continue

            if spec.param_type in (ParamType.FLOAT, ParamType.INT):
                mean = sum(float(v) for v in good_values) / len(good_values)
                std = (sum((float(v) - mean) ** 2 for v in good_values) / len(good_values)) ** 0.5
                std = max(std, 1e-8)  # Minimum std for exploration

                # Sample with exploration
                lo = spec.low or float("-inf")
                hi = spec.high or float("inf")
                if spec.log_scale and lo > 0:
                    val = math.exp(random.gauss(math.log(max(mean, lo)), max(math.log(std), 0.1)))
                else:
                    val = random.gauss(mean, std)

                val = max(lo, min(hi, val))
                params[name] = int(round(val)) if spec.param_type == ParamType.INT else val

            elif spec.param_type == ParamType.CATEGORICAL:
                # Weighted choice favoring good values
                from collections import Counter
                counts = Counter(good_values)
                total = sum(counts.values())
                weights = [counts.get(c, 1) / (total + len(spec.choices or []))
                          for c in (spec.choices or good_values)]
                choices = spec.choices or good_values
                params[name] = random.choices(choices, weights=weights, k=1)[0]

            elif spec.param_type == ParamType.BOOL:
                true_frac = sum(1 for v in good_values if v is True) / len(good_values)
                params[name] = random.random() < true_frac

        return params

    # ------------------------------------------------------------------
    # Trial evaluation
    # ------------------------------------------------------------------

    def _evaluate_trial(
        self,
        params: Dict[str, Any],
        objective: Callable,
        step: int,
    ) -> Trial:
        """Evaluate a single trial."""
        trial = Trial(params=params)
        trial.status = "running"

        try:
            metrics = objective(params)
            trial.metrics = metrics
            trial.status = "completed"
        except Exception as e:
            trial.status = "failed"
            trial.error = str(e)

        trial.completed_at = time.time()
        return trial

    def _update_best(self, trial: Trial) -> None:
        """Update the best trial if the new one is better."""
        if trial.status != "completed" or trial.objective_value is None:
            return

        if self._best_trial is None:
            self._best_trial = trial
            return

        if self._is_better_value(trial.objective_value, self._best_trial.objective_value):
            self._best_trial = trial

    def _is_better(self, trial: Trial) -> bool:
        """Check if a trial is better than the current best."""
        if self._best_trial is None or trial.objective_value is None:
            return True
        return self._is_better_value(trial.objective_value, self._best_trial.objective_value)

    def _is_better_value(self, value: float, reference: Optional[float]) -> bool:
        """Check if a value is better than the reference."""
        if reference is None:
            return True
        if self._config.objective_mode == "min":
            return value < reference
        return value > reference

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_best_trial(self) -> Optional[Trial]:
        """Get the best trial by objective metric."""
        return self._best_trial

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the parameters from the best trial."""
        if self._best_trial:
            return dict(self._best_trial.params)
        return None

    def get_n_best(self, n: int = 5) -> List[Trial]:
        """Get the top N trials by objective metric.

        Args:
            n: Number of top trials.

        Returns:
            List of Trial objects sorted by objective.
        """
        completed = [t for t in self._trials if t.status == "completed" and t.objective_value is not None]
        reverse = self._config.objective_mode == "max"
        completed.sort(key=lambda t: t.objective_value or 0, reverse=reverse)
        return completed[:n]

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics.

        Returns:
            Dictionary with search progress and results.
        """
        completed = [t for t in self._trials if t.status == "completed"]
        failed = [t for t in self._trials if t.status == "failed"]

        return {
            "strategy": self._config.strategy.value,
            "total_trials": len(self._trials),
            "completed_trials": len(completed),
            "failed_trials": len(failed),
            "best_trial": self._best_trial.to_dict() if self._best_trial else None,
            "objective_metric": self._config.objective_metric,
            "objective_mode": self._config.objective_mode,
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> str:
        """Save search results to a JSON file.

        Args:
            path: Output file path.

        Returns:
            The path written.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "config": self._config.to_dict(),
            "space": self._space.to_dict(),
            "trials": [t.to_dict() for t in self._trials],
            "best_trial_id": self._best_trial.trial_id if self._best_trial else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "HyperparameterSearch":
        """Load search results from a JSON file.

        Args:
            path: Path to the saved search file.

        Returns:
            Reconstructed HyperparameterSearch object.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct space
        space = ParamSpace()
        for name, spec_data in data.get("space", {}).items():
            space.params[name] = ParamSpec(
                name=name,
                param_type=ParamType(spec_data["param_type"]),
                low=spec_data.get("low"),
                high=spec_data.get("high"),
                choices=spec_data.get("choices"),
                log_scale=spec_data.get("log_scale", False),
            )

        # Reconstruct config
        config_data = data.get("config", {})
        config = SearchConfig(
            strategy=SearchStrategy(config_data.get("strategy", "random")),
            num_trials=config_data.get("num_trials", 50),
            grid_points=config_data.get("grid_points", 10),
            objective_metric=config_data.get("objective_metric", "loss"),
            objective_mode=config_data.get("objective_mode", "min"),
            seed=config_data.get("seed"),
        )

        search = cls(space=space, config=config)

        # Reconstruct trials
        for t_data in data.get("trials", []):
            trial = Trial(
                trial_id=t_data["trial_id"],
                params=t_data.get("params", {}),
                metrics=t_data.get("metrics", {}),
                status=t_data.get("status", "pending"),
                created_at=t_data.get("created_at", 0),
                completed_at=t_data.get("completed_at"),
                error=t_data.get("error"),
            )
            search._trials.append(trial)
            if trial.trial_id == data.get("best_trial_id"):
                search._best_trial = trial

        return search
