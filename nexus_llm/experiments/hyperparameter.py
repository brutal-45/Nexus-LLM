"""Hyperparameter search for Nexus-LLM.

Supports grid search and random search over a defined parameter space,
evaluating an objective function to find the best configuration.
"""

import itertools
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ObjectiveFn = Callable[[Dict[str, Any]], float]


class HyperparameterSearch:
    """Search a hyperparameter space for the best configuration.

    The search space is a dict mapping parameter names to lists of
    candidate values.  The objective is a callable that receives a
    parameter dict and returns a numeric score (higher is better by
    default, or lower if ``direction="minimize"``).

    Example::

        search = HyperparameterSearch(direction="minimize")
        space = {"lr": [0.001, 0.01, 0.1], "batch_size": [16, 32]}
        best = search.search(
            space,
            objective=lambda p: train_and_evaluate(p),
            n_trials=10,
            method="random",
        )
    """

    def __init__(self, direction: str = "maximize") -> None:
        """Initialise the search.

        Args:
            direction: ``"maximize"`` to seek the highest objective score,
                       ``"minimize"`` to seek the lowest.
        """
        if direction not in ("maximize", "minimize"):
            raise ValueError(
                f"direction must be 'maximize' or 'minimize', got {direction!r}"
            )
        self._direction = direction
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        search_space: Dict[str, List[Any]],
        objective: ObjectiveFn,
        n_trials: int = 10,
        method: str = "grid",
    ) -> Dict[str, Any]:
        """Execute a hyperparameter search.

        Args:
            search_space: Mapping of parameter name to list of candidate values.
            objective: Callable that takes a params dict and returns a score.
            n_trials: Number of trials for random search (ignored for grid).
            method: ``"grid"`` for exhaustive grid search or ``"random"``
                    for random sampling.

        Returns:
            The best parameter dict found.

        Raises:
            ValueError: If *method* is not ``"grid"`` or ``"random"``, or
                        if the search space is empty.
        """
        if not search_space:
            raise ValueError("search_space must not be empty")

        self._history = []

        if method == "grid":
            return self._grid_search(search_space, objective)
        elif method == "random":
            return self._random_search(search_space, objective, n_trials)
        else:
            raise ValueError(
                f"Unsupported method {method!r}; expected 'grid' or 'random'."
            )

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return a copy of the trial history.

        Each entry has ``"params"`` and ``"score"`` keys.
        """
        return list(self._history)

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def _grid_search(
        self,
        search_space: Dict[str, List[Any]],
        objective: ObjectiveFn,
    ) -> Dict[str, Any]:
        """Exhaustively evaluate every combination in *search_space*."""
        keys = list(search_space.keys())
        value_lists = [search_space[k] for k in keys]

        best_params: Dict[str, Any] = {}
        best_score: Optional[float] = None

        for combo in itertools.product(*value_lists):
            params = dict(zip(keys, combo))
            score = self._evaluate(params, objective)

            if best_score is None or self._is_better(score, best_score):
                best_score = score
                best_params = dict(params)

        logger.info(
            "Grid search complete: %d trials, best score=%.6f, params=%s",
            len(self._history), best_score, best_params,
        )
        return best_params

    # ------------------------------------------------------------------
    # Random search
    # ------------------------------------------------------------------

    def _random_search(
        self,
        search_space: Dict[str, List[Any]],
        objective: ObjectiveFn,
        n_trials: int,
    ) -> Dict[str, Any]:
        """Sample *n_trials* random combinations from *search_space*."""
        keys = list(search_space.keys())
        value_lists = [search_space[k] for k in keys]

        best_params: Dict[str, Any] = {}
        best_score: Optional[float] = None

        for _ in range(n_trials):
            params = {k: random.choice(v) for k, v in zip(keys, value_lists)}
            score = self._evaluate(params, objective)

            if best_score is None or self._is_better(score, best_score):
                best_score = score
                best_params = dict(params)

        logger.info(
            "Random search complete: %d trials, best score=%.6f, params=%s",
            len(self._history), best_score, best_params,
        )
        return best_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self, params: Dict[str, Any], objective: ObjectiveFn
    ) -> float:
        """Evaluate the objective and record the result."""
        try:
            score = objective(params)
        except Exception as exc:
            logger.warning("Objective evaluation failed for %s: %s", params, exc)
            score = float("-inf") if self._direction == "maximize" else float("inf")

        self._history.append({"params": dict(params), "score": score})
        return score

    def _is_better(self, candidate: float, current: float) -> bool:
        """Return True if *candidate* improves on *current*."""
        if self._direction == "maximize":
            return candidate > current
        return candidate < current

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<HyperparameterSearch direction={self._direction!r}>"
