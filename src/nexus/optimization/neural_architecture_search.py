"""
Neural Architecture Search Module
==================================

Production-grade NAS implementations including evolutionary search, DARTS-style
differentiable search, random search, Bayesian optimization, one-shot NAS, and
hardware-aware NAS.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nexus.optimization.optimization_config import NASConfig

logger = logging.getLogger(__name__)


# =============================================================================
# SearchSpace
# =============================================================================

class SearchSpace:
    """Defines the space of possible architectures.

    Supports discrete choices, continuous parameters, integer ranges,
    and boolean flags for comprehensive architecture search.
    """

    def __init__(self, space_dict: Optional[Dict[str, Any]] = None):
        """Initialize the search space.

        Args:
            space_dict: Dictionary defining the search space parameters.
        """
        self._parameters: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []
        if space_dict:
            for name, spec in space_dict.items():
                self.add_parameter(name, spec)

    def add_choice(self, name: str, options: List[Any]) -> SearchSpace:
        """Add a discrete choice parameter.

        Args:
            name: Parameter name.
            options: List of possible values.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If options has fewer than 2 elements.
        """
        if len(options) < 2:
            raise ValueError(f"Choice '{name}' must have at least 2 options, got {len(options)}")
        self._parameters[name] = {"type": "choice", "options": options}
        if name not in self._order:
            self._order.append(name)
        return self

    def add_continuous(self, name: str, low: float, high: float) -> SearchSpace:
        """Add a continuous parameter.

        Args:
            name: Parameter name.
            low: Minimum value (inclusive).
            high: Maximum value (exclusive).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If low >= high.
        """
        if low >= high:
            raise ValueError(f"Continuous '{name}': low ({low}) must be < high ({high})")
        self._parameters[name] = {"type": "continuous", "low": low, "high": high}
        if name not in self._order:
            self._order.append(name)
        return self

    def add_integer(self, name: str, low: int, high: int) -> SearchSpace:
        """Add an integer parameter.

        Args:
            name: Parameter name.
            low: Minimum value (inclusive).
            high: Maximum value (exclusive).

        Returns:
            Self for method chaining.
        """
        if low >= high:
            raise ValueError(f"Integer '{name}': low ({low}) must be < high ({high})")
        self._parameters[name] = {"type": "integer", "low": low, "high": high}
        if name not in self._order:
            self._order.append(name)
        return self

    def add_boolean(self, name: str) -> SearchSpace:
        """Add a boolean parameter.

        Args:
            name: Parameter name.

        Returns:
            Self for method chaining.
        """
        self._parameters[name] = {"type": "boolean", "options": [True, False]}
        if name not in self._order:
            self._order.append(name)
        return self

    def add_parameter(self, name: str, spec: Dict[str, Any]) -> SearchSpace:
        """Add a parameter from a specification dict.

        Args:
            name: Parameter name.
            spec: Specification dictionary with 'type' key.

        Returns:
            Self for method chaining.
        """
        ptype = spec.get("type", "").lower()
        if ptype == "choice":
            return self.add_choice(name, spec["options"])
        elif ptype == "continuous":
            return self.add_continuous(name, spec["low"], spec["high"])
        elif ptype == "integer":
            return self.add_integer(name, int(spec["low"]), int(spec["high"]))
        elif ptype == "boolean":
            return self.add_boolean(name)
        else:
            raise ValueError(f"Unknown parameter type: {ptype}")

    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Sample a random architecture from the space.

        Args:
            rng: Random number generator.

        Returns:
            Dictionary of parameter name to sampled value.
        """
        if rng is None:
            rng = random.Random()
        result = {}
        for name in self._order:
            spec = self._parameters[name]
            ptype = spec["type"]
            if ptype == "choice":
                result[name] = rng.choice(spec["options"])
            elif ptype == "continuous":
                result[name] = rng.uniform(spec["low"], spec["high"])
            elif ptype == "integer":
                result[name] = rng.randint(spec["low"], spec["high"] - 1)
            elif ptype == "boolean":
                result[name] = rng.choice([True, False])
        return result

    def size(self) -> int:
        """Estimate the total search space size.

        Returns:
            Approximate number of unique architectures (may be very large).
        """
        total = 1
        for name in self._order:
            spec = self._parameters[name]
            ptype = spec["type"]
            if ptype == "choice":
                total *= len(spec["options"])
            elif ptype == "continuous":
                total *= 1000
            elif ptype == "integer":
                total *= max(1, spec["high"] - spec["low"])
            elif ptype == "boolean":
                total *= 2
        return total

    def get_parameter(self, name: str) -> Dict[str, Any]:
        """Get a parameter specification.

        Args:
            name: Parameter name.

        Returns:
            Parameter specification dictionary.

        Raises:
            KeyError: If parameter does not exist.
        """
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found in search space")
        return self._parameters[name]

    @property
    def parameter_names(self) -> List[str]:
        """Get ordered list of parameter names.

        Returns:
            List of parameter names.
        """
        return list(self._order)

    def __len__(self) -> int:
        """Get number of parameters in the search space."""
        return len(self._parameters)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialize the search space to a dictionary.

        Returns:
            Dictionary representation.
        """
        return {name: self._parameters[name] for name in self._order}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> SearchSpace:
        """Create a search space from a dictionary.

        Args:
            data: Dictionary of parameter specifications.

        Returns:
            SearchSpace instance.
        """
        return cls(data)


# =============================================================================
# Architecture
# =============================================================================

class Architecture:
    """Represents a specific architecture configuration."""

    def __init__(self, config: Dict[str, Any], search_space: Optional[SearchSpace] = None):
        """Initialize an architecture.

        Args:
            config: Architecture configuration parameters.
            search_space: Optional reference search space for validation.
        """
        self.config = dict(config)
        self.search_space = search_space
        self._model: Optional[nn.Module] = None
        self._fitness: float = 0.0
        self._metrics: Dict[str, float] = {}
        self._flops: float = 0.0
        self._params: int = 0

        if search_space:
            self._validate_against_space(search_space)

    def _validate_against_space(self, space: SearchSpace):
        """Validate config against search space constraints.

        Args:
            space: Search space to validate against.
        """
        for name, spec in space._parameters.items():
            if name in self.config:
                ptype = spec["type"]
                if ptype == "choice" and self.config[name] not in spec["options"]:
                    raise ValueError(
                        f"Architecture: '{name}'={self.config[name]} not in {spec['options']}"
                    )
                elif ptype == "continuous":
                    val = self.config[name]
                    if not (spec["low"] <= val < spec["high"]):
                        raise ValueError(
                            f"Architecture: '{name}'={val} not in [{spec['low']}, {spec['high']})"
                        )
                elif ptype == "integer":
                    val = self.config[name]
                    if not (spec["low"] <= val < spec["high"]):
                        raise ValueError(
                            f"Architecture: '{name}'={val} not in [{spec['low']}, {spec['high']})"
                        )

    def to_model(self, base_model_fn: Callable[[Dict[str, Any]], nn.Module]) -> nn.Module:
        """Build actual model from architecture config.

        Args:
            base_model_fn: Function that takes config dict and returns a model.

        Returns:
            PyTorch model.
        """
        self._model = base_model_fn(self.config)
        return self._model

    def compute_flops(self, input_size: Tuple[int, ...] = (1, 512)) -> float:
        """Estimate FLOPs for this architecture.

        Args:
            input_size: Input tensor size.

        Returns:
            Estimated FLOPs.
        """
        if self._model is None:
            hidden_dim = self.config.get("hidden_dim", 768)
            num_layers = self.config.get("num_layers", 12)
            seq_len = input_size[-1] if len(input_size) > 1 else 512

            ffn_dim = self.config.get("ffn_dim", hidden_dim * 4)
            self._flops = num_layers * (
                2 * hidden_dim * ffn_dim * seq_len
                + 4 * hidden_dim * hidden_dim * seq_len
            ) * input_size[0]
        else:
            self._flops = self._estimate_model_flops(self._model, input_size)

        return self._flops

    def _estimate_model_flops(self, model: nn.Module, input_size: Tuple[int, ...]) -> float:
        """Estimate FLOPs by counting operations in linear layers.

        Args:
            model: PyTorch model.
            input_size: Input size.

        Returns:
            Estimated FLOPs.
        """
        total_flops = 0
        dummy_input = torch.randn(*input_size)

        def flops_hook(module, input, output):
            if isinstance(module, nn.Linear):
                if isinstance(input, tuple) and len(input) > 0:
                    inp = input[0]
                    batch = inp.shape[0]
                    seq = inp.numel() // inp.shape[-1]
                    total_flops_val = 2 * module.in_features * module.out_features * seq
                    return total_flops_val
            return 0

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(flops_hook)
                hooks.append(h)

        try:
            with torch.no_grad():
                model(dummy_input)
        except Exception:
            pass

        for h in hooks:
            h.remove()

        return max(total_flops, 1.0)

    def compute_params(self) -> int:
        """Count parameters in the architecture.

        Returns:
            Total parameter count.
        """
        if self._model is not None:
            self._params = sum(p.numel() for p in self._model.parameters())
        else:
            hidden_dim = self.config.get("hidden_dim", 768)
            num_layers = self.config.get("num_layers", 12)
            ffn_dim = self.config.get("ffn_dim", hidden_dim * 4)
            vocab_size = self.config.get("vocab_size", 30522)

            embed_params = vocab_size * hidden_dim
            layer_params = num_layers * (
                4 * hidden_dim * hidden_dim
                + 2 * hidden_dim * ffn_dim
                + 4 * hidden_dim
            )
            lm_head_params = vocab_size * hidden_dim
            self._params = embed_params + layer_params + lm_head_params

        return self._params

    def to_dict(self) -> Dict[str, Any]:
        """Serialize architecture to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "config": self.config,
            "fitness": self._fitness,
            "flops": self._flops,
            "params": self._params,
            "metrics": self._metrics,
        }

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize to JSON.

        Args:
            path: Optional file path.

        Returns:
            JSON string if path is None.
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        if path:
            path = os.path.abspath(os.path.expanduser(path))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(json_str)
            return None
        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Architecture:
        """Create from dictionary.

        Args:
            data: Dictionary with 'config' key.

        Returns:
            Architecture instance.
        """
        arch = cls(data["config"])
        arch._fitness = data.get("fitness", 0.0)
        arch._flops = data.get("flops", 0.0)
        arch._params = data.get("params", 0)
        arch._metrics = data.get("metrics", {})
        return arch

    @classmethod
    def from_json(cls, path_or_str: str) -> Architecture:
        """Create from JSON file or string.

        Args:
            path_or_str: File path or JSON string.

        Returns:
            Architecture instance.
        """
        if os.path.isfile(path_or_str):
            with open(path_or_str) as f:
                data = json.load(f)
        else:
            data = json.loads(path_or_str)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return f"Architecture(config={self.config}, fitness={self._fitness:.4f})"


# =============================================================================
# NASResult
# =============================================================================

class NASResult:
    """Result of a NAS run containing the best architecture and metrics."""

    def __init__(
        self,
        architecture: Architecture,
        accuracy: float,
        flops: float,
        params: int,
        latency_ms: float = 0.0,
        search_time_seconds: float = 0.0,
        total_trials: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize NAS result.

        Args:
            architecture: Best found architecture.
            accuracy: Best accuracy achieved.
            flops: FLOPs of best architecture.
            params: Parameter count of best architecture.
            latency_ms: Inference latency.
            search_time_seconds: Total search time.
            total_trials: Number of architectures evaluated.
            history: Optional search history.
        """
        self.architecture = architecture
        self.accuracy = accuracy
        self.flops = flops
        self.params = params
        self.latency_ms = latency_ms
        self.search_time_seconds = search_time_seconds
        self.total_trials = total_trials
        self.history = history or []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "architecture": self.architecture.to_dict(),
            "accuracy": self.accuracy,
            "flops": self.flops,
            "params": self.params,
            "latency_ms": self.latency_ms,
            "search_time_seconds": self.search_time_seconds,
            "total_trials": self.total_trials,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"NAS Result:\n"
            f"  Accuracy: {self.accuracy:.4f}\n"
            f"  FLOPs: {self.flops:.2e}\n"
            f"  Params: {self.params:,}\n"
            f"  Latency: {self.latency_ms:.2f}ms\n"
            f"  Search time: {self.search_time_seconds:.1f}s\n"
            f"  Trials: {self.total_trials}\n"
            f"  Config: {self.architecture.config}"
        )


# =============================================================================
# Base NAS Engine
# =============================================================================

class BaseNASEngine(ABC):
    """Abstract base class for NAS engines."""

    def __init__(self, config: NASConfig):
        """Initialize the NAS engine.

        Args:
            config: NAS configuration.
        """
        self.config = config
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.search_space = SearchSpace(config.search_space)
        self.device = _get_device()
        self._history: List[Dict[str, Any]] = []
        self._best_architecture: Optional[Architecture] = None
        self._best_fitness: float = float("-inf") if config.metric_mode == "maximize" else float("inf")

    @abstractmethod
    def search(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        model_fn: Optional[Callable] = None,
        max_trials: Optional[int] = None,
    ) -> NASResult:
        """Run the architecture search.

        Args:
            train_dataloader: Training data.
            eval_dataloader: Evaluation data.
            model_fn: Function to build models from configs.
            max_trials: Override max trials.

        Returns:
            NAS result with best architecture.
        """
        ...

    def _evaluate_architecture(
        self,
        architecture: Architecture,
        dataloader: DataLoader,
        model_fn: Callable,
        train_epochs: int = 1,
    ) -> float:
        """Evaluate a single architecture.

        Args:
            architecture: Architecture to evaluate.
            dataloader: Evaluation data.
            model_fn: Model builder function.
            train_epochs: Number of training epochs for quick evaluation.

        Returns:
            Fitness score.
        """
        try:
            model = architecture.to_model(model_fn)
            model = model.to(self.device)

            num_params = sum(p.numel() for p in model.parameters())

            if (self.config.budget_params > 0 and num_params > self.config.budget_params):
                return float("-inf") if self.config.metric_mode == "maximize" else float("inf")

            flops = architecture.compute_flops()

            if (self.config.budget_flops > 0 and flops > self.config.budget_flops):
                return float("-inf") if self.config.metric_mode == "maximize" else float("inf")

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            model.train()

            for epoch in range(train_epochs):
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(self.device) if isinstance(batch[0], torch.Tensor) else batch[0]
                        targets = batch[1].to(self.device) if len(batch) > 1 and isinstance(batch[1], torch.Tensor) else None
                    elif isinstance(batch, dict):
                        inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "labels"}
                        targets = batch.get("labels")
                        if targets is not None and isinstance(targets, torch.Tensor):
                            targets = targets.to(self.device)
                    else:
                        continue

                    optimizer.zero_grad()

                    if isinstance(inputs, dict):
                        outputs = model(**inputs)
                    else:
                        outputs = model(inputs)

                    if isinstance(outputs, dict):
                        logits = outputs.get("logits", outputs.get("loss"))
                    else:
                        logits = outputs

                    if targets is not None and isinstance(logits, torch.Tensor):
                        if logits.shape != targets.shape:
                            if logits.dim() > 1 and targets.dim() == 1:
                                loss = F.cross_entropy(logits.float(), targets.long())
                            else:
                                loss = F.mse_loss(logits.float(), targets.float())
                        else:
                            loss = F.cross_entropy(logits.float(), targets.long())
                    else:
                        loss = logits if isinstance(logits, torch.Tensor) else torch.tensor(0.0)

                    if isinstance(loss, torch.Tensor) and loss.requires_grad:
                        loss.backward()
                        optimizer.step()

            model.eval()
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(self.device) if isinstance(batch[0], torch.Tensor) else batch[0]
                        targets = batch[1].to(self.device) if len(batch) > 1 and isinstance(batch[1], torch.Tensor) else None
                    elif isinstance(batch, dict):
                        inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "labels"}
                        targets = batch.get("labels")
                        if targets is not None and isinstance(targets, torch.Tensor):
                            targets = targets.to(self.device)
                    else:
                        continue

                    if isinstance(inputs, dict):
                        outputs = model(**inputs)
                    else:
                        outputs = model(inputs)

                    if isinstance(outputs, dict):
                        logits = outputs.get("logits", outputs.get("loss"))
                    else:
                        logits = outputs

                    if targets is not None and isinstance(logits, torch.Tensor):
                        if logits.dim() > 1:
                            preds = logits.argmax(dim=-1)
                        else:
                            preds = (logits > 0).long()
                        total_correct += (preds == targets).sum().item()
                        total_samples += targets.numel()

            accuracy = total_correct / max(1, total_samples)
            architecture._fitness = accuracy
            architecture._metrics = {"accuracy": accuracy}
            architecture.compute_params()
            architecture.compute_flops()

            if self.config.metric_mode == "maximize":
                if accuracy > self._best_fitness:
                    self._best_fitness = accuracy
                    self._best_architecture = architecture
            else:
                if accuracy < self._best_fitness:
                    self._best_fitness = accuracy
                    self._best_architecture = architecture

            return accuracy

        except Exception as e:
            logger.debug("Architecture evaluation failed: %s", e)
            return float("-inf") if self.config.metric_mode == "maximize" else float("inf")


# =============================================================================
# EvolutionaryNAS
# =============================================================================

class EvolutionaryNAS(BaseNASEngine):
    """Evolution-based neural architecture search.

    Uses genetic algorithm with tournament selection, crossover, and
    mutation to evolve architectures toward better performance.
    """

    def __init__(self, config: NASConfig):
        """Initialize evolutionary NAS.

        Args:
            config: NAS configuration.
        """
        super().__init__(config)
        self.population_size = config.population_size
        self.mutation_rate = config.mutation_rate
        self.crossover_rate = config.crossover_rate
        self.tournament_size = config.tournament_size
        self.num_generations = config.num_generations

    def initialize_population(self, size: int) -> List[Architecture]:
        """Create initial random architectures.

        Args:
            size: Population size.

        Returns:
            List of random architectures.
        """
        population = []
        for _ in range(size):
            config = self.search_space.sample(self.rng)
            arch = Architecture(config, self.search_space)
            population.append(arch)
        return population

    def evaluate_population(
        self,
        population: List[Architecture],
        dataloader: DataLoader,
        model_fn: Callable,
    ) -> List[float]:
        """Evaluate all architectures in the population.

        Args:
            population: List of architectures.
            dataloader: Evaluation dataloader.
            model_fn: Model builder function.

        Returns:
            List of fitness scores.
        """
        fitnesses = []
        for i, arch in enumerate(population):
            fitness = self._evaluate_architecture(arch, dataloader, model_fn, train_epochs=1)
            fitnesses.append(fitness)
            self._history.append({
                "generation": len(self._history) // len(population),
                "individual": i,
                "config": arch.config,
                "fitness": fitness,
            })
            if (i + 1) % 10 == 0:
                logger.info("Evaluated %d/%d architectures, best fitness: %.4f", i + 1, len(population), self._best_fitness)
        return fitnesses

    def select_parents(
        self,
        population: List[Architecture],
        fitnesses: List[float],
    ) -> List[Architecture]:
        """Select parents using tournament selection.

        Args:
            population: Current population.
            fitnesses: Fitness scores.

        Returns:
            Selected parent architectures.
        """
        parents = []
        for _ in range(self.population_size):
            candidates = self.rng.sample(
                list(zip(population, fitnesses)),
                min(self.tournament_size, len(population))
            )
            if self.config.metric_mode == "maximize":
                best = max(candidates, key=lambda x: x[1])
            else:
                best = min(candidates, key=lambda x: x[1])
            parents.append(best[0])
        return parents

    def crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Combine two architectures via uniform crossover.

        Args:
            parent1: First parent architecture.
            parent2: Second parent architecture.

        Returns:
            Child architecture.
        """
        if self.rng.random() > self.crossover_rate:
            return Architecture(copy.deepcopy(parent1.config), self.search_space)

        child_config = {}
        for name in self.search_space.parameter_names:
            if self.rng.random() < 0.5:
                child_config[name] = copy.deepcopy(parent1.config.get(name))
            else:
                child_config[name] = copy.deepcopy(parent2.config.get(name))

        return Architecture(child_config, self.search_space)

    def mutate(self, architecture: Architecture, rate: Optional[float] = None) -> Architecture:
        """Randomly mutate an architecture.

        Args:
            architecture: Architecture to mutate.
            rate: Mutation rate (uses config default if None).

        Returns:
            Mutated architecture.
        """
        if rate is None:
            rate = self.mutation_rate

        config = copy.deepcopy(architecture.config)

        for name in self.search_space.parameter_names:
            if self.rng.random() < rate:
                spec = self.search_space.get_parameter(name)
                ptype = spec["type"]
                if ptype == "choice":
                    config[name] = self.rng.choice(spec["options"])
                elif ptype == "continuous":
                    config[name] = self.rng.uniform(spec["low"], spec["high"])
                elif ptype == "integer":
                    config[name] = self.rng.randint(spec["low"], spec["high"])
                elif ptype == "boolean":
                    config[name] = self.rng.choice([True, False])

        return Architecture(config, self.search_space)

    def search(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        model_fn: Optional[Callable] = None,
        max_trials: Optional[int] = None,
    ) -> NASResult:
        """Run evolutionary architecture search.

        Args:
            train_dataloader: Training data.
            eval_dataloader: Evaluation data.
            model_fn: Model builder function.
            max_trials: Maximum trials.

        Returns:
            NAS result.
        """
        if model_fn is None:
            model_fn = self._default_model_fn

        logger.info("Starting Evolutionary NAS: pop=%d, gen=%d, mut=%.2f",
                    self.population_size, self.num_generations, self.mutation_rate)

        start_time = time.time()
        population = self.initialize_population(self.population_size)
        dataloader = eval_dataloader or train_dataloader

        fitnesses = self.evaluate_population(population, dataloader, model_fn)

        for gen in range(self.num_generations):
            gen_start = time.time()

            parents = self.select_parents(population, fitnesses)

            offspring = []
            for i in range(0, self.population_size, 2):
                p1 = parents[i]
                p2 = parents[min(i + 1, len(parents) - 1)]
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                offspring.append(child)

            offspring_fitnesses = self.evaluate_population(offspring, dataloader, model_fn)

            combined = population + offspring
            combined_fitness = fitnesses + offspring_fitnesses

            sorted_pairs = sorted(
                zip(combined, combined_fitness),
                key=lambda x: x[1],
                reverse=(self.config.metric_mode == "maximize")
            )

            population = [p for p, _ in sorted_pairs[:self.population_size]]
            fitnesses = [f for _, f in sorted_pairs[:self.population_size]]

            gen_time = time.time() - gen_start
            logger.info(
                "Gen %d/%d: best=%.4f, avg=%.4f, time=%.1fs",
                gen + 1, self.num_generations,
                max(fitnesses) if self.config.metric_mode == "maximize" else min(fitnesses),
                sum(fitnesses) / len(fitnesses),
                gen_time,
            )

        total_time = time.time() - start_time

        if self._best_architecture is not None:
            self._best_architecture.compute_params()
            self._best_architecture.compute_flops()

        return NASResult(
            architecture=self._best_architecture or Architecture({}),
            accuracy=self._best_fitness,
            flops=self._best_architecture._flops if self._best_architecture else 0,
            params=self._best_architecture._params if self._best_architecture else 0,
            search_time_seconds=total_time,
            total_trials=len(self._history),
            history=self._history,
        )

    def _default_model_fn(self, config: Dict[str, Any]) -> nn.Module:
        """Default model builder from config."""
        hidden_dim = config.get("hidden_dim", 768)
        num_layers = config.get("num_layers", 6)
        num_classes = config.get("num_classes", 10)

        layers = []
        in_dim = config.get("input_dim", 784)
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.get("dropout", 0.1)),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_classes))
        return nn.Sequential(*layers)


# =============================================================================
# DifferentiableNAS
# =============================================================================

class DifferentiableNAS(BaseNASEngine):
    """DARTS-style differentiable architecture search.

    Builds a supernet with learnable architecture parameters (alphas) and
    performs bilevel optimization to find the best discrete architecture.
    """

    def __init__(self, config: NASConfig):
        """Initialize differentiable NAS.

        Args:
            config: NAS configuration.
        """
        super().__init__(config)
        self.arch_learning_rate = config.arch_learning_rate
        self.arch_weight_decay = config.arch_weight_decay
        self.num_arch_steps = config.num_arch_steps
        self.num_weight_steps = config.num_weight_steps
        self._arch_parameters: Dict[str, nn.Parameter] = {}
        self._supernet: Optional[nn.Module] = None

    def build_supernet(self, search_space: SearchSpace) -> nn.Module:
        """Build a supernet with all possible architectural choices.

        Args:
            search_space: Search space defining choices.

        Returns:
            Supernet model.
        """
        self._arch_parameters = {}

        choice_layers = nn.ModuleList()

        for name in search_space.parameter_names:
            spec = search_space.get_parameter(name)
            if spec["type"] == "choice":
                options = spec["options"]
                num_choices = len(options)

                alpha = nn.Parameter(torch.zeros(num_choices))
                self._arch_parameters[name] = alpha

                if all(isinstance(o, int) for o in options):
                    layer = nn.LazyLinear(max(options) * 2)
                else:
                    layer = nn.Identity()
                choice_layers.append(layer)

        self._supernet = nn.Sequential(*choice_layers)
        return self._supernet

    def compute_arch_parameters(self) -> Dict[str, torch.Tensor]:
        """Get learnable architecture parameters.

        Returns:
            Dictionary of architecture parameter tensors.
        """
        return {name: p.data for name, p in self._arch_parameters.items()}

    def forward_supernet(
        self,
        x: torch.Tensor,
        arch_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass through the supernet.

        Uses architecture parameters to compute weighted combination of all paths.

        Args:
            x: Input tensor.
            arch_params: Optional architecture parameters.

        Returns:
            Output tensor.
        """
        current_params = arch_params or self.compute_arch_parameters()

        for name, param in self._arch_parameters.items():
            alpha = current_params.get(name, param.data)
            weights = F.softmax(alpha.float(), dim=0)
            if weights.dim() == 1 and x.dim() > 1:
                batch_size = x.shape[0]
                weighted = torch.zeros_like(x)
                for i, w in enumerate(weights):
                    weighted = weighted + w * x
                x = weighted

        return x

    def compute_arch_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        arch_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute architecture-level loss.

        Args:
            logits: Model output logits.
            labels: Ground truth labels.
            arch_params: Architecture parameters.

        Returns:
            Architecture loss (cross-entropy + regularization).
        """
        ce_loss = F.cross_entropy(logits.float(), labels.long())

        reg_loss = torch.tensor(0.0, device=logits.device)
        for name, alpha in arch_params.items():
            reg_loss = reg_loss + self.arch_weight_decay * (alpha ** 2).sum()

        return ce_loss + reg_loss

    def search(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        model_fn: Optional[Callable] = None,
        max_trials: Optional[int] = None,
    ) -> NASResult:
        """Run DARTS-style differentiable search.

        Args:
            train_dataloader: Training data.
            eval_dataloader: Validation data.
            model_fn: Model builder (unused for differentiable search).
            max_trials: Maximum epochs.

        Returns:
            NAS result.
        """
        logger.info("Starting Differentiable NAS (DARTS-style)")

        start_time = time.time()
        self.build_supernet(self.search_space)

        arch_optimizer = torch.optim.Adam(
            list(self._arch_parameters.values()),
            lr=self.arch_learning_rate,
            weight_decay=self.arch_weight_decay,
        )

        dataloader = eval_dataloader or train_dataloader
        total_steps = len(dataloader) * self.config.num_epochs

        step_count = 0
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device) if isinstance(batch[0], torch.Tensor) else torch.zeros(2, 10)
                    y = batch[1].to(self.device) if len(batch) > 1 and isinstance(batch[1], torch.Tensor) else torch.zeros(2, dtype=torch.long)
                elif isinstance(batch, dict):
                    x = batch.get("input_ids", batch.get("inputs", torch.zeros(2, 10)))
                    if isinstance(x, torch.Tensor):
                        x = x.to(self.device)
                    y = batch.get("labels", torch.zeros(2, dtype=torch.long))
                    if isinstance(y, torch.Tensor):
                        y = y.to(self.device)
                else:
                    continue

                if x.dim() > 2:
                    x = x.reshape(x.shape[0], -1)

                if x.shape[1] > 1024:
                    x = x[:, :1024]

                if x.shape[-1] != self._supernet[-1].out_features if hasattr(self._supernet[-1], 'out_features') else 10:
                    adaptive = nn.Linear(x.shape[-1], 10).to(self.device)
                    x = adaptive(x)

                arch_optimizer.zero_grad()
                output = self.forward_supernet(x)

                if output.shape[-1] != y.shape[-1]:
                    output = output[:, :y.shape[-1]] if output.dim() > 1 else output

                if output.shape == y.shape:
                    loss = self.compute_arch_loss(output, y, self.compute_arch_parameters())
                else:
                    target = torch.zeros(output.shape[0], output.shape[-1], dtype=torch.long, device=self.device)
                    loss = self.compute_arch_loss(output, target, self.compute_arch_parameters())

                loss.backward()
                arch_optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                step_count += 1

            avg_loss = epoch_loss / max(1, num_batches)
            logger.info("DARTS Epoch %d/%d: arch_loss=%.4f", epoch + 1, self.config.num_epochs, avg_loss)

        best_arch_config = self.derive_architecture()
        best_arch = Architecture(best_arch_config, self.search_space)
        total_time = time.time() - start_time

        return NASResult(
            architecture=best_arch,
            accuracy=1.0 - avg_loss,
            flops=best_arch.compute_flops(),
            params=best_arch.compute_params(),
            search_time_seconds=total_time,
            total_trials=self.config.num_epochs * len(dataloader),
            history=self._history,
        )

    def derive_architecture(self) -> Dict[str, Any]:
        """Extract the best discrete architecture from learned alphas.

        Returns:
            Dictionary of best architecture choices.
        """
        best_config = {}

        for name in self.search_space.parameter_names:
            spec = self.search_space.get_parameter(name)

            if name in self._arch_parameters:
                alpha = self._arch_parameters[name].data
                best_idx = alpha.argmax().item()

                if spec["type"] == "choice":
                    best_config[name] = spec["options"][best_idx]
                elif spec["type"] == "boolean":
                    best_config[name] = best_idx == 0
                else:
                    best_config[name] = best_idx
            else:
                if spec["type"] == "choice":
                    best_config[name] = spec["options"][0]
                elif spec["type"] == "continuous":
                    best_config[name] = (spec["low"] + spec["high"]) / 2
                elif spec["type"] == "integer":
                    best_config[name] = (spec["low"] + spec["high"]) // 2
                elif spec["type"] == "boolean":
                    best_config[name] = True

        logger.info("Derived architecture: %s", best_config)
        return best_config


# =============================================================================
# RandomSearch
# =============================================================================

class RandomSearch(BaseNASEngine):
    """Random architecture search baseline.

    Samples random architectures and evaluates them, keeping the best.
    """

    def __init__(self, config: NASConfig):
        """Initialize random search.

        Args:
            config: NAS configuration.
        """
        super().__init__(config)

    def search(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        model_fn: Optional[Callable] = None,
        max_trials: Optional[int] = None,
    ) -> NASResult:
        """Run random architecture search.

        Args:
            train_dataloader: Training data.
            eval_dataloader: Evaluation data.
            model_fn: Model builder function.
            max_trials: Maximum number of trials.

        Returns:
            NAS result.
        """
        if model_fn is None:
            model_fn = self._default_model_fn

        num_trials = max_trials or self.config.max_trials
        logger.info("Starting Random Search with %d trials", num_trials)

        start_time = time.time()
        dataloader = eval_dataloader or train_dataloader

        for trial in range(num_trials):
            config = self.search_space.sample(self.rng)
            arch = Architecture(config, self.search_space)

            fitness = self._evaluate_architecture(arch, dataloader, model_fn, train_epochs=1)

            self._history.append({
                "trial": trial,
                "config": config,
                "fitness": fitness,
            })

            if (trial + 1) % 10 == 0 or trial == num_trials - 1:
                logger.info("Trial %d/%d: best=%.4f", trial + 1, num_trials, self._best_fitness)

        total_time = time.time() - start_time

        if self._best_architecture is not None:
            self._best_architecture.compute_params()
            self._best_architecture.compute_flops()

        return NASResult(
            architecture=self._best_architecture or Architecture({}),
            accuracy=self._best_fitness,
            flops=self._best_architecture._flops if self._best_architecture else 0,
            params=self._best_architecture._params if self._best_architecture else 0,
            search_time_seconds=total_time,
            total_trials=num_trials,
            history=self._history,
        )

    def _default_model_fn(self, config: Dict[str, Any]) -> nn.Module:
        """Default model builder."""
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 3)
        num_classes = config.get("num_classes", 10)
        layers = [nn.Flatten()]
        in_dim = 784
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        return nn.Sequential(*layers)


# =============================================================================
# BayesianOptimizerNAS
# =============================================================================

class BayesianOptimizerNAS(BaseNASEngine):
    """Bayesian optimization for architecture search.

    Uses Gaussian Process-based surrogate model to efficiently
    explore the architecture space.
    """

    def __init__(self, config: NASConfig):
        """Initialize Bayesian NAS.

        Args:
            config: NAS configuration.
        """
        super().__init__(config)
        self.initial_points = config.bayesian_initial_points
        self.acquisition = config.bayesian_acquisition
        self._observed_configs: List[Dict[str, Any]] = []
        self._observed_fitnesses: List[float] = []
        self._kernel_bandwidth = 1.0
        self._noise = 1e-4

    def _encode_config(self, config: Dict[str, Any]) -> torch.Tensor:
        """Encode architecture config to a feature vector.

        Args:
            config: Architecture configuration.

        Returns:
            Feature tensor.
        """
        features = []
        for name in self.search_space.parameter_names:
            spec = self.search_space.get_parameter(name)
            ptype = spec["type"]
            value = config.get(name)

            if ptype == "choice":
                vec = [0.0] * len(spec["options"])
                try:
                    idx = spec["options"].index(value)
                    vec[idx] = 1.0
                except ValueError:
                    pass
                features.extend(vec)
            elif ptype == "continuous":
                normalized = (value - spec["low"]) / max(1e-10, spec["high"] - spec["low"])
                features.append(normalized)
            elif ptype == "integer":
                normalized = (value - spec["low"]) / max(1, spec["high"] - spec["low"])
                features.append(normalized)
            elif ptype == "boolean":
                features.append(1.0 if value else 0.0)

        return torch.tensor(features, dtype=torch.float32)

    def _gaussian_kernel(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        bandwidth: float,
    ) -> torch.Tensor:
        """Compute RBF (Gaussian) kernel between two vectors.

        Args:
            x1: First vector.
            x2: Second vector.
            bandwidth: Kernel bandwidth.

        Returns:
            Kernel value.
        """
        diff = x1 - x2
        return torch.exp(-0.5 * torch.dot(diff, diff) / (bandwidth ** 2))

    def _kernel_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute full kernel matrix.

        Args:
            X: Matrix of encoded configs (n, d).

        Returns:
            Kernel matrix (n, n).
        """
        n = X.shape[0]
        K = torch.zeros(n, n)
        for i in range(n):
            for j in range(i, n):
                k = self._gaussian_kernel(X[i], X[j], self._kernel_bandwidth)
                K[i, j] = k
                K[j, i] = k
        return K

    def _predict(
        self,
        x: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        K: torch.Tensor,
    ) -> Tuple[float, float]:
        """GP prediction for a new configuration.

        Args:
            x: Encoded test config.
            X_train: Training encoded configs.
            y_train: Training fitness values.
            K: Training kernel matrix.

        Returns:
            Tuple of (mean, variance) predictions.
        """
        n = X_train.shape[0]
        k_star = torch.tensor([self._gaussian_kernel(x, X_train[i], self._kernel_bandwidth) for i in range(n)])

        K_noise = K + self._noise * torch.eye(n)
        try:
            L = torch.linalg.cholesky(K_noise)
            alpha = torch.linalg.solve_triangular(L.T, torch.linalg.solve_triangular(L, y_train), upper=True)
            mean = k_star @ alpha
            v = torch.linalg.solve_triangular(L, k_star, upper=False)
            variance = self._gaussian_kernel(x, x, self._kernel_bandwidth) - torch.dot(v, v)
            variance = max(variance.item(), 1e-10)
        except torch RuntimeError:
            mean = y_train.mean().item()
            variance = y_train.var().item() if len(y_train) > 1 else 1.0

        return mean.item(), variance

    def _expected_improvement(self, mean: float, std: float, best: float) -> float:
        """Compute Expected Improvement acquisition function.

        Args:
            mean: Predicted mean.
            std: Predicted standard deviation.
            best: Best observed value.

        Returns:
            Expected improvement value.
        """
        from math import erf, sqrt, exp
        z = (mean - best) / max(std, 1e-10)
        ei = (mean - best) * (0.5 * (1 + erf(z / sqrt(2)))) + std * (1 / sqrt(2 * math.pi)) * exp(-0.5 * z ** 2)
        return max(ei, 0.0)

    def search(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        model_fn: Optional[Callable] = None,
        max_trials: Optional[int] = None,
    ) -> NASResult:
        """Run Bayesian optimization search.

        Args:
            train_dataloader: Training data.
            eval_dataloader: Evaluation data.
            model_fn: Model builder.
            max_trials: Maximum trials.

        Returns:
            NAS result.
        """
        if model_fn is None:
            model_fn = self._default_model_fn

        num_trials = max_trials or self.config.max_trials
        logger.info("Starting Bayesian NAS with %d trials", num_trials)

        start_time = time.time()
        dataloader = eval_dataloader or train_dataloader

        for trial in range(min(self.initial_points, num_trials)):
            config = self.search_space.sample(self.rng)
            arch = Architecture(config, self.search_space)
            fitness = self._evaluate_architecture(arch, dataloader, model_fn, train_epochs=1)
            self._observed_configs.append(config)
            self._observed_fitnesses.append(fitness)

        X_train = torch.stack([self._encode_config(c) for c in self._observed_configs])
        y_train = torch.tensor(self._observed_fitnesses, dtype=torch.float32)
        K = self._kernel_matrix(X_train)

        for trial in range(self.initial_points, num_trials):
            best_fitness = max(self._observed_fitnesses) if self.config.metric_mode == "maximize" else min(self._observed_fitnesses)

            candidates = [self.search_space.sample(self.rng) for _ in range(50)]
            best_ei = float("-inf")
            best_candidate = candidates[0]

            for cand_config in candidates:
                x = self._encode_config(cand_config)
                mean, var = self._predict(x, X_train, y_train, K)
                std = math.sqrt(var)

                if self.acquisition in ("expected_improvement", "ei"):
                    ei = self._expected_improvement(mean, std, best_fitness)
                elif self.acquisition in ("upper_confidence_bound", "ucb"):
                    ei = mean + 2.0 * std
                elif self.acquisition in ("probability_of_improvement", "pi"):
                    from math import erf, sqrt
                    z = (mean - best_fitness) / max(std, 1e-10)
                    ei = 0.5 * (1 + erf(z / sqrt(2)))
                else:
                    ei = self._expected_improvement(mean, std, best_fitness)

                if ei > best_ei:
                    best_ei = ei
                    best_candidate = cand_config

            arch = Architecture(best_candidate, self.search_space)
            fitness = self._evaluate_architecture(arch, dataloader, model_fn, train_epochs=1)

            self._observed_configs.append(best_candidate)
            self._observed_fitnesses.append(fitness)

            x_new = self._encode_config(best_candidate)
            X_train = torch.cat([X_train, x_new.unsqueeze(0)])
            y_train = torch.cat([y_train, torch.tensor([fitness])])

            k_new = torch.tensor([self._gaussian_kernel(x_new, X_train[i], self._kernel_bandwidth) for i in range(X_train.shape[0])])
            K = torch.zeros(X_train.shape[0], X_train.shape[0])
            old_n = K.shape[0] - 1
            K[:old_n, :old_n] = self._kernel_matrix(X_train[:old_n])
            for i in range(X_train.shape[0]):
                K[i, -1] = k_new[i]
                K[-1, i] = k_new[i]

            self._history.append({"trial": trial, "fitness": fitness, "ei": best_ei})

            if (trial + 1) % 10 == 0:
                logger.info("Bayesian trial %d/%d: best=%.4f", trial + 1, num_trials, self._best_fitness)

        total_time = time.time() - start_time

        if self._best_architecture is not None:
            self._best_architecture.compute_params()
            self._best_architecture.compute_flops()

        return NASResult(
            architecture=self._best_architecture or Architecture({}),
            accuracy=self._best_fitness,
            flops=self._best_architecture._flops if self._best_architecture else 0,
            params=self._best_architecture._params if self._best_architecture else 0,
            search_time_seconds=total_time,
            total_trials=num_trials,
            history=self._history,
        )

    def _default_model_fn(self, config: Dict[str, Any]) -> nn.Module:
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 3)
        layers = [nn.Flatten(), nn.Linear(784, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 10))
        return nn.Sequential(*layers)


# =============================================================================
# OneShotNAS
# =============================================================================

class OneShotNAS(BaseNASEngine):
    """One-shot NAS: train a supernet once, then extract sub-networks.

    Trains a single supernet that contains all possible architectures,
    then evaluates individual sub-networks without additional training.
    """

    def __init__(self, config: NASConfig):
        """Initialize one-shot NAS.

        Args:
            config: NAS configuration.
        """
        super().__init__(config)
        self.supernet_epochs = config.supernet_epochs
        self._supernet_weights: Optional[Dict[str, torch.Tensor]] = None

    def search(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        model_fn: Optional[Callable] = None,
        max_trials: Optional[int] = None,
    ) -> NASResult:
        """Run one-shot NAS.

        Args:
            train_dataloader: Training data.
            eval_dataloader: Evaluation data.
            model_fn: Model builder.
            max_trials: Maximum sub-network evaluations.

        Returns:
            NAS result.
        """
        if model_fn is None:
            model_fn = self._default_model_fn

        num_trials = max_trials or self.config.max_trials
        logger.info("Starting One-Shot NAS: supernet_epochs=%d, eval_trials=%d",
                    self.supernet_epochs, num_trials)

        start_time = time.time()

        max_hidden = max(
            opt for name, spec in self.search_space._parameters.items()
            for opt in spec.get("options", [])
            if isinstance(opt, int)
        ) or 1024

        supernet = self._build_supernet(max_hidden)
        supernet = supernet.to(self.device)
        optimizer = torch.optim.AdamW(supernet.parameters(), lr=1e-3)

        logger.info("Training supernet for %d epochs...", self.supernet_epochs)
        supernet.train()
        for epoch in range(self.supernet_epochs):
            total_loss = 0.0
            num_batches = 0
            for batch in train_dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device) if isinstance(batch[0], torch.Tensor) else None
                    y = batch[1].to(self.device) if len(batch) > 1 and isinstance(batch[1], torch.Tensor) else None
                elif isinstance(batch, dict):
                    x = batch.get("input_ids", batch.get("inputs"))
                    y = batch.get("labels")
                    if isinstance(x, torch.Tensor):
                        x = x.to(self.device)
                    if isinstance(y, torch.Tensor):
                        y = y.to(self.device)
                else:
                    continue

                if x is None or y is None:
                    continue

                if x.dim() > 2:
                    x = x.reshape(x.shape[0], -1)

                try:
                    optimizer.zero_grad()
                    out = supernet(x)
                    if out.shape[-1] != y.shape[-1]:
                        adapt = nn.Linear(out.shape[-1], y.shape[-1]).to(self.device)
                        out = adapt(out)
                    loss = F.cross_entropy(out.float(), y.long())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
                except Exception:
                    continue

            if num_batches > 0:
                logger.info("Supernet Epoch %d/%d: loss=%.4f", epoch + 1, self.supernet_epochs, total_loss / num_batches)

        self._supernet_weights = {k: v.clone() for k, v in supernet.state_dict().items()}

        logger.info("Evaluating %d sub-networks...", num_trials)
        dataloader = eval_dataloader or train_dataloader

        for trial in range(num_trials):
            config = self.search_space.sample(self.rng)
            arch = Architecture(config, self.search_space)

            try:
                sub_model = self._extract_subnetwork(config, max_hidden)
                sub_model = sub_model.to(self.device)
                fitness = self._evaluate_subnetwork(sub_model, dataloader)

                if self.config.metric_mode == "maximize":
                    if fitness > self._best_fitness:
                        self._best_fitness = fitness
                        self._best_architecture = arch
                else:
                    if fitness < self._best_fitness:
                        self._best_fitness = fitness
                        self._best_architecture = arch

            except Exception as e:
                logger.debug("Sub-network evaluation failed: %s", e)

        total_time = time.time() - start_time

        if self._best_architecture is not None:
            self._best_architecture.compute_params()
            self._best_architecture.compute_flops()

        return NASResult(
            architecture=self._best_architecture or Architecture({}),
            accuracy=self._best_fitness,
            flops=self._best_architecture._flops if self._best_architecture else 0,
            params=self._best_architecture._params if self._best_architecture else 0,
            search_time_seconds=total_time,
            total_trials=num_trials,
            history=self._history,
        )

    def _build_supernet(self, max_hidden: int) -> nn.Module:
        """Build the supernet model.

        Args:
            max_hidden: Maximum hidden dimension.

        Returns:
            Supernet model.
        """
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, max_hidden),
            nn.ReLU(),
            nn.Linear(max_hidden, max_hidden),
            nn.ReLU(),
            nn.Linear(max_hidden, 10),
        )

    def _extract_subnetwork(self, config: Dict[str, Any], max_hidden: int) -> nn.Module:
        """Extract a sub-network matching the given config.

        Args:
            config: Architecture config.
            max_hidden: Supernet hidden dimension.

        Returns:
            Sub-network model.
        """
        hidden_dim = config.get("hidden_dim", 256)
        hidden_dim = min(hidden_dim, max_hidden)
        num_layers = config.get("num_layers", 3)

        layers = [nn.Flatten(), nn.Linear(784, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 10))

        if self._supernet_weights:
            supernet_state = self._supernet_weights
            sub_state = {}
            for i, layer in enumerate(layers):
                if isinstance(layer, nn.Linear):
                    src_name = f"{i}"
                    if src_name in supernet_state:
                        src_weight = supernet_state[src_name]
                        if src_weight.shape[0] >= layer.out_features and src_weight.shape[1] >= layer.in_features:
                            sub_state[f"{i}.weight"] = src_weight[:layer.out_features, :layer.in_features]
                            if f"{i}.bias" in supernet_state and layer.bias is not None:
                                sub_state[f"{i}.bias"] = supernet_state[f"{i}.bias"][:layer.out_features]

            model = nn.Sequential(*layers)
            model.load_state_dict(sub_state, strict=False)
            return model

        return nn.Sequential(*layers)

    def _evaluate_subnetwork(self, model: nn.Module, dataloader: DataLoader) -> float:
        """Evaluate a sub-network.

        Args:
            model: Sub-network model.
            dataloader: Evaluation data.

        Returns:
            Accuracy.
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device) if isinstance(batch[0], torch.Tensor) else None
                    y = batch[1].to(self.device) if len(batch) > 1 and isinstance(batch[1], torch.Tensor) else None
                elif isinstance(batch, dict):
                    x = batch.get("input_ids", batch.get("inputs"))
                    y = batch.get("labels")
                    if isinstance(x, torch.Tensor): x = x.to(self.device)
                    if isinstance(y, torch.Tensor): y = y.to(self.device)
                else:
                    continue

                if x is None or y is None:
                    continue

                if x.dim() > 2:
                    x = x.reshape(x.shape[0], -1)

                try:
                    out = model(x)
                    if out.shape[-1] != y.shape[-1]:
                        adapt = nn.Linear(out.shape[-1], y.shape[-1]).to(self.device)
                        out = adapt(out)
                    preds = out.argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total += y.numel()
                except Exception:
                    continue

        return correct / max(1, total)

    def _default_model_fn(self, config: Dict[str, Any]) -> nn.Module:
        h = config.get("hidden_dim", 256)
        n = config.get("num_layers", 3)
        layers = [nn.Flatten(), nn.Linear(784, h), nn.ReLU()]
        for _ in range(n - 1):
            layers.extend([nn.Linear(h, h), nn.ReLU()])
        layers.append(nn.Linear(h, 10))
        return nn.Sequential(*layers)


# =============================================================================
# HardwareAwareNAS
# =============================================================================

class HardwareAwareNAS(BaseNASEngine):
    """Hardware-aware NAS considering latency, memory, and energy constraints.

    Evaluates architectures on actual hardware metrics in addition to accuracy.
    """

    def __init__(self, config: NASConfig):
        """Initialize hardware-aware NAS.

        Args:
            config: NAS configuration.
        """
        super().__init__(config)
        self.latency_budget_ms = config.budget_latency_ms
        self.memory_budget_mb = config.budget_memory_mb
        self._latency_cache: Dict[str, float] = {}
        self._memory_cache: Dict[str, float] = {}

    def _measure_latency(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...] = (1, 784),
        num_warmup: int = 5,
        num_runs: int = 20,
    ) -> float:
        """Measure model inference latency.

        Args:
            model: Model to measure.
            input_size: Input tensor size.
            num_warmup: Warmup iterations.
            num_runs: Measurement iterations.

        Returns:
            Average latency in milliseconds.
        """
        model.eval()
        dummy = torch.randn(*input_size, device=self.device)

        for _ in range(num_warmup):
            with torch.no_grad():
                model(dummy)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.time() - start) / num_runs * 1000
        return elapsed

    def _estimate_memory_mb(self, model: nn.Module) -> float:
        """Estimate model memory usage.

        Args:
            model: PyTorch model.

        Returns:
            Estimated memory in MB.
        """
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        return total_bytes / (1024 * 1024)

    def _hardware_fitness(
        self,
        accuracy: float,
        latency_ms: float,
        memory_mb: float,
    ) -> float:
        """Compute hardware-aware fitness score.

        Args:
            accuracy: Model accuracy.
            latency_ms: Inference latency.
            memory_mb: Memory usage.

        Returns:
            Combined fitness score.
        """
        fitness = accuracy

        if self.latency_budget_ms > 0:
            latency_penalty = max(0, 1.0 - latency_ms / self.latency_budget_ms)
            fitness = fitness * (0.7 + 0.3 * latency_penalty)

        if self.memory_budget_mb > 0:
            memory_penalty = max(0, 1.0 - memory_mb / self.memory_budget_mb)
            fitness = fitness * (0.8 + 0.2 * memory_penalty)

        return fitness

    def search(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        model_fn: Optional[Callable] = None,
        max_trials: Optional[int] = None,
    ) -> NASResult:
        """Run hardware-aware NAS.

        Args:
            train_dataloader: Training data.
            eval_dataloader: Evaluation data.
            model_fn: Model builder.
            max_trials: Maximum trials.

        Returns:
            NAS result.
        """
        if model_fn is None:
            model_fn = self._default_model_fn

        num_trials = max_trials or self.config.max_trials
        logger.info("Starting Hardware-Aware NAS: %d trials, latency_budget=%.1fms, memory_budget=%.1fMB",
                    num_trials, self.latency_budget_ms, self.memory_budget_mb)

        start_time = time.time()
        dataloader = eval_dataloader or train_dataloader

        for trial in range(num_trials):
            config = self.search_space.sample(self.rng)
            arch = Architecture(config, self.search_space)

            accuracy = self._evaluate_architecture(arch, dataloader, model_fn, train_epochs=1)

            if accuracy <= 0:
                self._history.append({"trial": trial, "fitness": -1, "latency": -1, "memory": -1})
                continue

            model = arch.to_model(model_fn)
            model = model.to(self.device)

            latency = self._measure_latency(model)
            memory = self._estimate_memory_mb(model)

            hw_fitness = self._hardware_fitness(accuracy, latency, memory)

            if self.config.metric_mode == "maximize" and hw_fitness > self._best_fitness:
                self._best_fitness = hw_fitness
                self._best_architecture = arch
                arch._fitness = hw_fitness
                arch._metrics = {"accuracy": accuracy, "latency_ms": latency, "memory_mb": memory}
            elif self.config.metric_mode == "minimize" and hw_fitness < self._best_fitness:
                self._best_fitness = hw_fitness
                self._best_architecture = arch
                arch._fitness = hw_fitness

            self._history.append({
                "trial": trial,
                "accuracy": accuracy,
                "latency_ms": latency,
                "memory_mb": memory,
                "hw_fitness": hw_fitness,
            })

            if (trial + 1) % 10 == 0:
                logger.info("Trial %d/%d: best_hw_fitness=%.4f", trial + 1, num_trials, self._best_fitness)

        total_time = time.time() - start_time

        if self._best_architecture is not None:
            self._best_architecture.compute_params()
            self._best_architecture.compute_flops()

        return NASResult(
            architecture=self._best_architecture or Architecture({}),
            accuracy=self._best_fitness,
            flops=self._best_architecture._flops if self._best_architecture else 0,
            params=self._best_architecture._params if self._best_architecture else 0,
            search_time_seconds=total_time,
            total_trials=num_trials,
            history=self._history,
        )

    def _default_model_fn(self, config: Dict[str, Any]) -> nn.Module:
        h = config.get("hidden_dim", 256)
        n = config.get("num_layers", 3)
        layers = [nn.Flatten(), nn.Linear(784, h), nn.ReLU()]
        for _ in range(n - 1):
            layers.extend([nn.Linear(h, h), nn.ReLU()])
        layers.append(nn.Linear(h, 10))
        return nn.Sequential(*layers)
