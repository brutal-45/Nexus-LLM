"""Experiment manager for Nexus-LLM.

Central registry that creates, retrieves, lists, deletes, and
compares experiments.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.experiments.experiment import Experiment, ExperimentState

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Create and manage :class:`Experiment` instances.

    The manager acts as the top-level entry point for the experiments
    module, providing CRUD operations and comparison utilities.

    Example::

        mgr = ExperimentManager()
        exp = mgr.create_experiment("bert-fine-tune", config={"lr": 2e-5})
        exp.start()
        exp.log_metric("loss", 0.42, step=1)
        exp.stop()
        print(mgr.get_experiment(exp.id).get_status())
    """

    def __init__(self) -> None:
        self._experiments: Dict[str, Experiment] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create a new experiment and register it.

        Args:
            name: Human-readable experiment name.
            config: Optional configuration dict.

        Returns:
            The newly created :class:`Experiment`.
        """
        experiment = Experiment(name=name, config=config)
        self._experiments[experiment.id] = experiment
        logger.info(
            "Created experiment %s (%s) with config keys: %s",
            experiment.name, experiment.id,
            list(config.keys()) if config else [],
        )
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Retrieve an experiment by ID.

        Args:
            experiment_id: The experiment identifier.

        Returns:
            The matching :class:`Experiment`.

        Raises:
            KeyError: If no experiment with the given ID exists.
        """
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment {experiment_id!r} not found")
        return self._experiments[experiment_id]

    def list_experiments(
        self,
        state: Optional[ExperimentState] = None,
    ) -> List[Experiment]:
        """List all registered experiments, optionally filtered by state.

        Args:
            state: If provided, only return experiments in this state.

        Returns:
            List of :class:`Experiment` objects.
        """
        experiments = list(self._experiments.values())
        if state is not None:
            experiments = [e for e in experiments if e.state == state]
        return experiments

    def delete_experiment(self, experiment_id: str) -> None:
        """Remove an experiment from the manager.

        Args:
            experiment_id: The experiment identifier.

        Raises:
            KeyError: If no experiment with the given ID exists.
        """
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment {experiment_id!r} not found")
        del self._experiments[experiment_id]
        logger.info("Deleted experiment %s", experiment_id)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_experiments(self, ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments side-by-side.

        The comparison includes status, parameters, and the final value
        of each metric across the specified experiments.

        Args:
            ids: List of experiment IDs to compare.

        Returns:
            A dict with keys:

            - ``"experiments"``: list of status dicts for each experiment
            - ``"metric_comparison"``: dict mapping metric names to
              ``{experiment_id: final_value}``
            - ``"parameter_comparison"``: dict mapping parameter names to
              ``{experiment_id: value}``

        Raises:
            KeyError: If any experiment ID is not found.
        """
        experiments = [self.get_experiment(eid) for eid in ids]

        statuses = [exp.get_status() for exp in experiments]

        # Metric comparison: collect the last recorded value per metric
        metric_comparison: Dict[str, Dict[str, Any]] = {}
        for exp in experiments:
            last_metrics: Dict[str, float] = {}
            for record in exp.metrics:
                last_metrics[record["name"]] = record["value"]
            for mname, mval in last_metrics.items():
                metric_comparison.setdefault(mname, {})[exp.id] = mval

        # Parameter comparison
        parameter_comparison: Dict[str, Dict[str, Any]] = {}
        for exp in experiments:
            for pname, pval in exp.parameters.items():
                parameter_comparison.setdefault(pname, {})[exp.id] = pval

        return {
            "experiments": statuses,
            "metric_comparison": metric_comparison,
            "parameter_comparison": parameter_comparison,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<ExperimentManager experiments={len(self._experiments)}>"
