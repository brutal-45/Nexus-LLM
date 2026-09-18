"""Tests for the experiments module.

Covers ExperimentManager, Experiment, ExperimentTracker, and HyperparameterSearch.
"""

from __future__ import annotations

import json

import pytest

from nexus_llm.experiments.experiment import Experiment, ExperimentState
from nexus_llm.experiments.manager import ExperimentManager
from nexus_llm.experiments.tracker import ExperimentTracker
from nexus_llm.experiments.hyperparameter import HyperparameterSearch


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class TestExperiment:
    """Tests for Experiment."""

    def test_create(self):
        exp = Experiment(name="test-exp")
        assert exp.name == "test-exp"
        assert exp.state == ExperimentState.CREATED
        assert exp.id  # auto-generated

    def test_start(self):
        exp = Experiment(name="test")
        exp.start()
        assert exp.state == ExperimentState.RUNNING

    def test_stop(self):
        exp = Experiment(name="test")
        exp.start()
        exp.stop()
        assert exp.state == ExperimentState.COMPLETED

    def test_pause_resume(self):
        exp = Experiment(name="test")
        exp.start()
        exp.pause()
        assert exp.state == ExperimentState.PAUSED
        exp.resume()
        assert exp.state == ExperimentState.RUNNING

    def test_fail(self):
        exp = Experiment(name="test")
        exp.start()
        exp.fail("Something went wrong")
        assert exp.state == ExperimentState.FAILED

    def test_start_from_wrong_state_raises(self):
        exp = Experiment(name="test")
        exp.start()
        with pytest.raises(RuntimeError):
            exp.start()  # already running

    def test_stop_from_wrong_state_raises(self):
        exp = Experiment(name="test")
        with pytest.raises(RuntimeError):
            exp.stop()  # not running

    def test_pause_from_wrong_state_raises(self):
        exp = Experiment(name="test")
        with pytest.raises(RuntimeError):
            exp.pause()  # not running

    def test_resume_from_wrong_state_raises(self):
        exp = Experiment(name="test")
        with pytest.raises(RuntimeError):
            exp.resume()  # not paused

    def test_log_metric(self):
        exp = Experiment(name="test")
        exp.start()
        exp.log_metric("loss", 0.5, step=1)
        exp.log_metric("loss", 0.3, step=2)
        assert len(exp.metrics) == 2

    def test_log_metric_not_running_raises(self):
        exp = Experiment(name="test")
        with pytest.raises(RuntimeError):
            exp.log_metric("loss", 0.5, step=1)

    def test_log_parameter(self):
        exp = Experiment(name="test")
        exp.log_parameter("lr", 0.01)
        exp.log_parameter("optimizer", "adam")
        assert exp.parameters["lr"] == 0.01
        assert exp.parameters["optimizer"] == "adam"

    def test_log_artifact(self):
        exp = Experiment(name="test")
        exp.log_artifact("/path/to/model.pt")
        assert len(exp.artifacts) == 1

    def test_get_status(self):
        exp = Experiment(name="test", config={"lr": 0.01})
        exp.start()
        exp.log_metric("loss", 0.5, step=1)
        status = exp.get_status()
        assert status["name"] == "test"
        assert status["state"] == "running"
        assert status["num_metrics"] == 1

    def test_repr(self):
        exp = Experiment(name="test")
        r = repr(exp)
        assert "test" in r
        assert "created" in r


# ---------------------------------------------------------------------------
# ExperimentManager
# ---------------------------------------------------------------------------

class TestExperimentManager:
    """Tests for ExperimentManager."""

    def test_create_experiment(self):
        mgr = ExperimentManager()
        exp = mgr.create_experiment("my-exp", config={"lr": 0.01})
        assert exp.name == "my-exp"
        assert exp.config == {"lr": 0.01}

    def test_get_experiment(self):
        mgr = ExperimentManager()
        exp = mgr.create_experiment("test")
        retrieved = mgr.get_experiment(exp.id)
        assert retrieved is exp

    def test_get_nonexistent_raises(self):
        mgr = ExperimentManager()
        with pytest.raises(KeyError):
            mgr.get_experiment("nonexistent")

    def test_list_experiments(self):
        mgr = ExperimentManager()
        mgr.create_experiment("exp1")
        mgr.create_experiment("exp2")
        experiments = mgr.list_experiments()
        assert len(experiments) == 2

    def test_list_experiments_by_state(self):
        mgr = ExperimentManager()
        exp1 = mgr.create_experiment("running_exp")
        exp1.start()
        exp2 = mgr.create_experiment("created_exp")
        running = mgr.list_experiments(state=ExperimentState.RUNNING)
        assert len(running) == 1
        assert running[0].name == "running_exp"

    def test_delete_experiment(self):
        mgr = ExperimentManager()
        exp = mgr.create_experiment("to-delete")
        mgr.delete_experiment(exp.id)
        with pytest.raises(KeyError):
            mgr.get_experiment(exp.id)

    def test_delete_nonexistent_raises(self):
        mgr = ExperimentManager()
        with pytest.raises(KeyError):
            mgr.delete_experiment("nonexistent")

    def test_compare_experiments(self):
        mgr = ExperimentManager()
        exp1 = mgr.create_experiment("exp1")
        exp1.start()
        exp1.log_metric("loss", 0.5, step=1)
        exp1.log_parameter("lr", 0.01)

        exp2 = mgr.create_experiment("exp2")
        exp2.start()
        exp2.log_metric("loss", 0.3, step=1)
        exp2.log_parameter("lr", 0.001)

        comparison = mgr.compare_experiments([exp1.id, exp2.id])
        assert "metric_comparison" in comparison
        assert "parameter_comparison" in comparison
        assert "loss" in comparison["metric_comparison"]

    def test_repr(self):
        mgr = ExperimentManager()
        r = repr(mgr)
        assert "ExperimentManager" in r


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    def test_track(self):
        tracker = ExperimentTracker()
        tracker.track("exp-001", {"loss": 0.5, "accuracy": 0.88})
        history = tracker.get_history("exp-001")
        assert len(history) == 1
        assert history[0]["metrics"]["loss"] == 0.5

    def test_track_multiple_steps(self):
        tracker = ExperimentTracker()
        tracker.track("exp-001", {"loss": 0.5})
        tracker.track("exp-001", {"loss": 0.3})
        history = tracker.get_history("exp-001")
        assert len(history) == 2

    def test_get_history_nonexistent_raises(self):
        tracker = ExperimentTracker()
        with pytest.raises(KeyError):
            tracker.get_history("nonexistent")

    def test_get_best_min(self):
        tracker = ExperimentTracker()
        tracker.track("exp-001", {"loss": 0.5})
        tracker.track("exp-001", {"loss": 0.3})
        tracker.track("exp-001", {"loss": 0.4})
        step, value = tracker.get_best("exp-001", "loss", mode="min")
        assert value == 0.3

    def test_get_best_max(self):
        tracker = ExperimentTracker()
        tracker.track("exp-001", {"accuracy": 0.8})
        tracker.track("exp-001", {"accuracy": 0.95})
        step, value = tracker.get_best("exp-001", "accuracy", mode="max")
        assert value == 0.95

    def test_get_best_nonexistent_metric_raises(self):
        tracker = ExperimentTracker()
        tracker.track("exp-001", {"loss": 0.5})
        with pytest.raises(KeyError):
            tracker.get_best("exp-001", "nonexistent_metric")

    def test_export_json(self):
        tracker = ExperimentTracker()
        tracker.track("exp-001", {"loss": 0.5})
        output = tracker.export("exp-001", format="json")
        data = json.loads(output)
        assert data["experiment_id"] == "exp-001"

    def test_export_csv(self):
        tracker = ExperimentTracker()
        tracker.track("exp-001", {"loss": 0.5, "accuracy": 0.9})
        output = tracker.export("exp-001", format="csv")
        assert "step" in output
        assert "loss" in output

    def test_export_markdown(self):
        tracker = ExperimentTracker()
        tracker.track("exp-001", {"loss": 0.5})
        output = tracker.export("exp-001", format="markdown")
        assert "exp-001" in output
        assert "|" in output

    def test_export_unsupported_format(self):
        tracker = ExperimentTracker()
        tracker.track("exp-001", {"loss": 0.5})
        with pytest.raises(ValueError, match="Unsupported"):
            tracker.export("exp-001", format="xml")

    def test_repr(self):
        tracker = ExperimentTracker()
        r = repr(tracker)
        assert "ExperimentTracker" in r


# ---------------------------------------------------------------------------
# HyperparameterSearch
# ---------------------------------------------------------------------------

class TestHyperparameterSearch:
    """Tests for HyperparameterSearch."""

    def test_grid_search(self):
        hs = HyperparameterSearch(direction="maximize")
        space = {"x": [1, 2, 3], "y": [10, 20]}
        result = hs.search(space, objective=lambda p: p["x"] + p["y"])
        assert result["x"] == 3
        assert result["y"] == 20

    def test_random_search(self):
        hs = HyperparameterSearch(direction="minimize")
        space = {"lr": [0.001, 0.01, 0.1], "batch_size": [16, 32]}
        result = hs.search(
            space,
            objective=lambda p: p["lr"] * 100 + p["batch_size"],
            n_trials=5,
            method="random",
        )
        assert "lr" in result
        assert "batch_size" in result

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            HyperparameterSearch(direction="invalid")

    def test_empty_search_space(self):
        hs = HyperparameterSearch()
        with pytest.raises(ValueError, match="empty"):
            hs.search({}, objective=lambda p: 0)

    def test_invalid_method(self):
        hs = HyperparameterSearch()
        with pytest.raises(ValueError, match="Unsupported method"):
            hs.search({"x": [1]}, objective=lambda p: 0, method="bayesian")

    def test_history(self):
        hs = HyperparameterSearch()
        space = {"x": [1, 2, 3]}
        hs.search(space, objective=lambda p: p["x"] ** 2)
        history = hs.history
        assert len(history) == 3
        assert all("params" in h and "score" in h for h in history)

    def test_minimize_direction(self):
        hs = HyperparameterSearch(direction="minimize")
        space = {"x": [1, 2, 10]}
        result = hs.search(space, objective=lambda p: p["x"])
        assert result["x"] == 1

    def test_objective_failure_handling(self):
        hs = HyperparameterSearch(direction="maximize")
        space = {"x": [1, 2]}

        def bad_objective(p):
            if p["x"] == 2:
                raise RuntimeError("Failed")
            return p["x"]

        result = hs.search(space, objective=bad_objective)
        assert result["x"] == 1

    def test_repr(self):
        hs = HyperparameterSearch()
        r = repr(hs)
        assert "HyperparameterSearch" in r
