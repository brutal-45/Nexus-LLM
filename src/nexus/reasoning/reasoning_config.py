"""
Reasoning Configuration Module
===============================

Comprehensive configuration system for all reasoning strategies in the Nexus LLM
framework. Each configuration class provides fine-grained control over reasoning
behavior, resource allocation, and quality trade-offs.

All configurations use frozen dataclasses for immutability and include validation
logic to ensure consistent, well-formed settings. Configuration values cascade
from the top-level ReasoningConfig to individual strategy configs, with explicit
overrides supported at each level.

Design Principles:
    1. Immutability: All configs are frozen dataclasses
    2. Validation: Built-in validation with clear error messages
    3. Defaults: Sensible defaults for production use
    4. Extensibility: Easy to add new strategies and parameters
    5. Serialization: Full JSON/YAML serialization support
"""

from __future__ import annotations

import copy
import json
import hashlib
import dataclasses
import re
from dataclasses import dataclass, field, fields, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path


# =============================================================================
# Enums
# =============================================================================

class ReasoningStrategy(Enum):
    """Available reasoning strategies for the Nexus LLM framework.

    Each strategy represents a distinct approach to multi-step reasoning,
    with different trade-offs between accuracy, speed, and resource usage.

    Attributes:
        CHAIN_OF_THOUGHT: Step-by-step linear reasoning with explicit thoughts.
        TREE_OF_THOUGHT: Exploration of multiple reasoning paths in a tree structure.
        PLANNING: Goal-oriented planning with subgoal decomposition.
        SELF_CONSISTENCY: Multiple independent solutions with majority voting.
        RETRIEVAL_AUGMENTED: Reasoning grounded in retrieved evidence.
        HYBRID: Adaptive combination of multiple strategies.
    """
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    PLANNING = "planning"
    SELF_CONSISTENCY = "self_consistency"
    RETRIEVAL_AUGMENTED = "retrieval_augmented"
    HYBRID = "hybrid"

    @classmethod
    def from_string(cls, value: str) -> ReasoningStrategy:
        """Parse a string into a ReasoningStrategy enum value.

        Args:
            value: String representation of the strategy.

        Returns:
            Corresponding ReasoningStrategy enum value.

        Raises:
            ValueError: If the string does not match any known strategy.
        """
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown reasoning strategy '{value}'. Valid strategies: {valid}"
        )

    def to_string(self) -> str:
        """Convert enum value to its string representation.

        Returns:
            String representation of the strategy.
        """
        return self.value

    def __str__(self) -> str:
        return self.value


class EvaluationMethod(Enum):
    """Methods for evaluating thought quality in tree-of-thought reasoning.

    Attributes:
        STATEFUL: Uses a value function to score states/thoughts.
        STATELESS: Direct model-based evaluation without state tracking.
        SAMPLING: Uses sampling-based estimation of thought quality.
        CRITIC: Uses a separate critic model for evaluation.
    """
    STATEFUL = "stateful"
    STATELESS = "stateless"
    SAMPLING = "sampling"
    CRITIC = "critic"

    @classmethod
    def from_string(cls, value: str) -> EvaluationMethod:
        """Parse a string into an EvaluationMethod enum value.

        Args:
            value: String representation of the evaluation method.

        Returns:
            Corresponding EvaluationMethod enum value.

        Raises:
            ValueError: If the string does not match any known method.
        """
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown evaluation method '{value}'. Valid methods: {valid}"
        )


class PruningStrategy(Enum):
    """Strategies for pruning branches in tree-of-thought reasoning.

    Attributes:
        THRESHOLD: Remove branches below a score threshold.
        TOP_K: Keep only the top K branches at each level.
        PROPORTIONAL: Keep a proportion of branches based on score distribution.
        CONFIDENCE_BASED: Prune based on confidence intervals.
        NONE: No pruning, explore all branches exhaustively.
    """
    THRESHOLD = "threshold"
    TOP_K = "top_k"
    PROPORTIONAL = "proportional"
    CONFIDENCE_BASED = "confidence_based"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str) -> PruningStrategy:
        """Parse a string into a PruningStrategy enum value.

        Args:
            value: String representation of the pruning strategy.

        Returns:
            Corresponding PruningStrategy enum value.
        """
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown pruning strategy '{value}'. Valid strategies: {valid}"
        )


class AggregationMethod(Enum):
    """Methods for aggregating multiple solutions in self-consistency.

    Attributes:
        MAJORITY_VOTE: Simple majority vote over solution strings.
        WEIGHTED_VOTE: Confidence-weighted voting.
        CLUSTER_BASED: Cluster solutions and select best cluster.
        RANK_BASED: Rank-based aggregation using solution rankings.
        BORDA_COUNT: Borda count voting method.
    """
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    CLUSTER_BASED = "cluster_based"
    RANK_BASED = "rank_based"
    BORDA_COUNT = "borda_count"

    @classmethod
    def from_string(cls, value: str) -> AggregationMethod:
        """Parse a string into an AggregationMethod enum value.

        Args:
            value: String representation of the aggregation method.

        Returns:
            Corresponding AggregationMethod enum value.
        """
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown aggregation method '{value}'. Valid methods: {valid}"
        )


class SelectionStrategy(Enum):
    """Strategies for selecting thoughts during tree exploration.

    Attributes:
        GREEDY: Always select the highest-scoring thought.
        UCB: Upper Confidence Bound for exploration-exploitation balance.
        SOFTMAX: Probabilistic selection with softmax over scores.
        EPSILON_GREEDY: Greedy with epsilon probability of random selection.
        THOMPSON: Thompson sampling for selection.
    """
    GREEDY = "greedy"
    UCB = "ucb"
    SOFTMAX = "softmax"
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON = "thompson"

    @classmethod
    def from_string(cls, value: str) -> SelectionStrategy:
        """Parse a string into a SelectionStrategy enum value.

        Args:
            value: String representation of the selection strategy.

        Returns:
            Corresponding SelectionStrategy enum value.
        """
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown selection strategy '{value}'. Valid strategies: {valid}"
        )


class BacktrackingMode(Enum):
    """Backtracking modes for chain-of-thought reasoning.

    Attributes:
        DISABLED: No backtracking, follow the reasoning path strictly.
        IMMEDIATE: Backtrack to the immediately preceding step.
        SMART: Backtrack to the most relevant previous step using heuristics.
        FULL: Consider all possible backtrack points.
    """
    DISABLED = "disabled"
    IMMEDIATE = "immediate"
    SMART = "smart"
    FULL = "full"

    @classmethod
    def from_string(cls, value: str) -> BacktrackingMode:
        """Parse a string into a BacktrackingMode enum value.

        Args:
            value: String representation of the backtracking mode.

        Returns:
            Corresponding BacktrackingMode enum value.
        """
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown backtracking mode '{value}'. Valid modes: {valid}"
        )


class ReplanningTrigger(Enum):
    """Triggers that cause a plan to be re-evaluated and potentially revised.

    Attributes:
        STEP_FAILURE: Replan when a subgoal execution fails.
        LOW_CONFIDENCE: Replan when confidence drops below threshold.
        NEW_INFORMATION: Replan when new relevant information is discovered.
        BUDGET_EXCEEDED: Replan when resource budget is exceeded.
        TIMEOUT: Replan when a step takes too long.
        MANUAL: Manual trigger for replanning.
    """
    STEP_FAILURE = "step_failure"
    LOW_CONFIDENCE = "low_confidence"
    NEW_INFORMATION = "new_information"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    MANUAL = "manual"

    @classmethod
    def from_string(cls, value: str) -> ReplanningTrigger:
        """Parse a string into a ReplanningTrigger enum value.

        Args:
            value: String representation of the replanning trigger.

        Returns:
            Corresponding ReplanningTrigger enum value.
        """
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown replanning trigger '{value}'. Valid triggers: {valid}"
        )


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_range(
    name: str,
    value: float,
    min_val: float,
    max_val: float,
    inclusive: bool = True,
) -> None:
    """Validate that a numeric value falls within a specified range.

    Args:
        name: Name of the parameter being validated (for error messages).
        value: The value to validate.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        inclusive: Whether the bounds are inclusive.

    Raises:
        ValueError: If the value is outside the specified range.
    """
    if inclusive:
        if value < min_val or value > max_val:
            raise ValueError(
                f"{name} must be between {min_val} and {max_val} (inclusive), "
                f"got {value}"
            )
    else:
        if value <= min_val or value >= max_val:
            raise ValueError(
                f"{name} must be strictly between {min_val} and {max_val}, "
                f"got {value}"
            )


def validate_positive_integer(name: str, value: int, max_val: Optional[int] = None) -> None:
    """Validate that a value is a positive integer, optionally with an upper bound.

    Args:
        name: Name of the parameter being validated.
        value: The value to validate.
        max_val: Optional upper bound for the value.

    Raises:
        ValueError: If the value is not positive or exceeds the maximum.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be at most {max_val}, got {value}")


def validate_non_empty_string(name: str, value: str) -> None:
    """Validate that a string is non-empty after stripping whitespace.

    Args:
        name: Name of the parameter being validated.
        value: The string value to validate.

    Raises:
        ValueError: If the string is empty or contains only whitespace.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got '{value}'")


def validate_enum_value(name: str, value: Any, expected_enum: type) -> None:
    """Validate that a value belongs to a specific enum type.

    Args:
        name: Name of the parameter being validated.
        value: The value to validate.
        expected_enum: The expected enum class.

    Raises:
        ValueError: If the value is not a member of the expected enum.
    """
    if not isinstance(value, expected_enum):
        valid = ", ".join(str(m.value) for m in expected_enum)
        raise ValueError(
            f"{name} must be one of {valid}, got {type(value).__name__}"
        )


# =============================================================================
# ChainOfThoughtConfig
# =============================================================================

@dataclass(frozen=True)
class ChainOfThoughtConfig:
    """Configuration for chain-of-thought reasoning strategy.

    Controls how the model generates step-by-step reasoning traces, including
    temperature settings for generation, maximum reasoning steps, and
    backtracking behavior.

    Attributes:
        strategy: Specific CoT variant to use (zero_shot, few_shot, auto, structured).
        max_steps: Maximum number of reasoning steps before termination.
        temperature: Temperature for CoT generation (lower for more deterministic).
        stop_sequences: Sequences that signal the end of a reasoning step.
        forced_reasoning: Whether to force the model to show its reasoning.
        backtracking_mode: How to handle backtracking when reasoning goes wrong.
        max_backtracks: Maximum number of backtrack attempts per reasoning chain.
        confidence_threshold: Minimum confidence for accepting a reasoning step.
        enable_verification: Whether to verify each reasoning step.
        enable_pruning: Whether to prune irrelevant reasoning steps.
        pattern_detection: Whether to detect reasoning patterns (deductive, etc.).
        trace_compression: Whether to compress long reasoning traces.
        min_confidence_for_backtrack: Confidence threshold below which to trigger backtrack.
        reasoning_prefix: Prefix to prompt reasoning (e.g., "Let's think step by step").
        step_separator: Separator between reasoning steps in the trace.
        max_tokens_per_step: Maximum tokens per individual reasoning step.
        include_examples: Number of few-shot examples to include (0 for zero-shot).
        example_pool: Pre-defined pool of reasoning examples for few-shot learning.
    """

    strategy: str = "zero_shot"
    max_steps: int = 10
    temperature: float = 0.3
    stop_sequences: Tuple[str, ...] = ("\n\n", "Therefore,", "So the answer is", "---")
    forced_reasoning: bool = True
    backtracking_mode: BacktrackingMode = BacktrackingMode.SMART
    max_backtracks: int = 3
    confidence_threshold: float = 0.5
    enable_verification: bool = True
    enable_pruning: bool = True
    pattern_detection: bool = True
    trace_compression: bool = True
    min_confidence_for_backtrack: float = 0.2
    reasoning_prefix: str = "Let's think step by step."
    step_separator: str = "\n"
    max_tokens_per_step: int = 512
    include_examples: int = 0
    example_pool: Tuple[Dict[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        valid_strategies = {"zero_shot", "few_shot", "auto", "structured"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"ChainOfThoughtConfig.strategy must be one of {valid_strategies}, "
                f"got '{self.strategy}'"
            )
        validate_positive_integer("max_steps", self.max_steps, max_val=100)
        validate_range("temperature", self.temperature, 0.0, 2.0)
        validate_positive_integer("max_backtracks", self.max_backtracks, max_val=20)
        validate_range("confidence_threshold", self.confidence_threshold, 0.0, 1.0)
        validate_range(
            "min_confidence_for_backtrack",
            self.min_confidence_for_backtrack,
            0.0, 1.0,
        )
        validate_positive_integer("max_tokens_per_step", self.max_tokens_per_step, max_val=4096)
        if self.include_examples < 0:
            raise ValueError(
                f"include_examples must be non-negative, got {self.include_examples}"
            )

    def get_effective_temperature(self, step_index: int) -> float:
        """Get the effective temperature for a given reasoning step.

        Implements a temperature schedule that increases slightly for later
        steps to encourage more creative exploration when the model gets stuck.

        Args:
            step_index: Zero-indexed step number in the reasoning chain.

        Returns:
            Effective temperature value for this step.
        """
        base_temp = self.temperature
        schedule_factor = 0.0
        if step_index > self.max_steps * 0.6:
            remaining = (step_index - self.max_steps * 0.6) / (self.max_steps * 0.4)
            schedule_factor = remaining * 0.2
        effective_temp = base_temp + schedule_factor
        return min(effective_temp, 2.0)

    def should_backtrack(self, confidence: float, backtrack_count: int) -> bool:
        """Determine whether to trigger backtracking based on confidence and history.

        Args:
            confidence: Confidence score of the current reasoning step.
            backtrack_count: Number of times backtracking has already occurred.

        Returns:
            True if backtracking should be attempted.
        """
        if self.backtracking_mode == BacktrackingMode.DISABLED:
            return False
        if backtrack_count >= self.max_backtracks:
            return False
        return confidence < self.min_confidence_for_backtrack

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                result[f.name] = value.value
            elif isinstance(value, tuple):
                result[f.name] = list(value)
            else:
                result[f.name] = value
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation of the configuration.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChainOfThoughtConfig:
        """Deserialize configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            A ChainOfThoughtConfig instance.

        Raises:
            TypeError: If data contains unexpected keys or invalid types.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {}
        for key, value in data.items():
            if key not in valid_fields:
                raise TypeError(f"Unexpected configuration key: {key}")
            if key == "backtracking_mode" and isinstance(value, str):
                value = BacktrackingMode.from_string(value)
            if key == "stop_sequences" and isinstance(value, list):
                value = tuple(value)
            if key == "example_pool" and isinstance(value, list):
                value = tuple(value)
            filtered_data[key] = value
        return cls(**filtered_data)

    def config_hash(self) -> str:
        """Compute a stable hash of this configuration for caching purposes.

        Returns:
            SHA-256 hash hex string of the configuration.
        """
        serialized = self.to_json()
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# TreeOfThoughtConfig
# =============================================================================

@dataclass(frozen=True)
class TreeOfThoughtConfig:
    """Configuration for tree-of-thought reasoning strategy.

    Controls the exploration of multiple reasoning paths through tree search
    algorithms including MCTS, beam search, BFS, and DFS.

    Attributes:
        num_thoughts: Number of candidate thoughts to generate at each node.
        breadth_limit: Maximum number of nodes at each tree depth level.
        depth_limit: Maximum depth of the thought tree.
        evaluation_method: Method for evaluating thought quality.
        pruning_strategy: Strategy for pruning low-quality branches.
        selection_strategy: Strategy for selecting nodes during exploration.
        search_algorithm: Primary search algorithm (mcts, beam, bfs, dfs).
        exploration_constant: UCB exploration constant (higher = more exploration).
        temperature: Temperature for thought generation.
        num_simulations: Number of MCTS simulations to run.
        beam_width: Width for beam search.
        min_score_threshold: Minimum score for a thought to be considered valid.
        parallel_evaluation: Whether to evaluate thoughts in parallel.
        max_tree_size: Maximum total number of nodes in the tree.
        value_function_type: Type of value function for node evaluation.
        rollout_depth: Depth of random rollouts in MCTS.
        discount_factor: Discount factor for value backpropagation.
    """

    num_thoughts: int = 5
    breadth_limit: int = 5
    depth_limit: int = 8
    evaluation_method: EvaluationMethod = EvaluationMethod.STATELESS
    pruning_strategy: PruningStrategy = PruningStrategy.TOP_K
    selection_strategy: SelectionStrategy = SelectionStrategy.UCB
    search_algorithm: str = "mcts"
    exploration_constant: float = 1.414
    temperature: float = 0.7
    num_simulations: int = 20
    beam_width: int = 3
    min_score_threshold: float = 0.1
    parallel_evaluation: bool = False
    max_tree_size: int = 1000
    value_function_type: str = "linear"
    rollout_depth: int = 5
    discount_factor: float = 0.95

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        validate_positive_integer("num_thoughts", self.num_thoughts, max_val=50)
        validate_positive_integer("breadth_limit", self.breadth_limit, max_val=100)
        validate_positive_integer("depth_limit", self.depth_limit, max_val=50)
        validate_positive_integer("num_simulations", self.num_simulations, max_val=500)
        validate_positive_integer("beam_width", self.beam_width, max_val=50)
        validate_range("exploration_constant", self.exploration_constant, 0.0, 10.0)
        validate_range("temperature", self.temperature, 0.0, 2.0)
        validate_range("min_score_threshold", self.min_score_threshold, 0.0, 1.0)
        validate_positive_integer("max_tree_size", self.max_tree_size, max_val=100000)
        validate_range("discount_factor", self.discount_factor, 0.0, 1.0)
        validate_positive_integer("rollout_depth", self.rollout_depth, max_val=50)
        valid_algorithms = {"mcts", "beam", "bfs", "dfs", "astar"}
        if self.search_algorithm not in valid_algorithms:
            raise ValueError(
                f"search_algorithm must be one of {valid_algorithms}, "
                f"got '{self.search_algorithm}'"
            )
        valid_value_functions = {"linear", "exponential", "sigmoid", "step"}
        if self.value_function_type not in valid_value_functions:
            raise ValueError(
                f"value_function_type must be one of {valid_value_functions}, "
                f"got '{self.value_function_type}'"
            )

    def compute_ucb_score(
        self,
        node_value: float,
        node_visits: int,
        parent_visits: int,
    ) -> float:
        """Compute the Upper Confidence Bound score for a node.

        The UCB formula balances exploitation (high node value) with
        exploration (low visit count):

            UCB = value + C * sqrt(ln(parent_visits) / node_visits)

        Args:
            node_value: Average value/score of the node.
            node_visits: Number of times this node has been visited.
            parent_visits: Number of times the parent has been visited.

        Returns:
            UCB score combining exploitation and exploration.
        """
        if node_visits == 0:
            return float("inf")
        exploitation = node_value
        exploration = self.exploration_constant * (
            (parent_visits ** 0.5) / (node_visits ** 0.5)
        )
        return exploitation + exploration

    def should_prune(self, score: float, scores_at_level: List[float]) -> bool:
        """Determine whether a thought branch should be pruned based on its score.

        Args:
            score: The score of the candidate branch.
            scores_at_level: Scores of all branches at this tree level.

        Returns:
            True if the branch should be pruned.
        """
        if self.pruning_strategy == PruningStrategy.NONE:
            return False
        if self.pruning_strategy == PruningStrategy.THRESHOLD:
            return score < self.min_score_threshold
        if self.pruning_strategy == PruningStrategy.TOP_K:
            sorted_scores = sorted(scores_at_level, reverse=True)
            if len(sorted_scores) > self.breadth_limit:
                cutoff = sorted_scores[self.breadth_limit - 1]
                return score < cutoff
        if self.pruning_strategy == PruningStrategy.PROPORTIONAL:
            if not scores_at_level:
                return False
            mean_score = sum(scores_at_level) / len(scores_at_level)
            return score < mean_score * 0.5
        if self.pruning_strategy == PruningStrategy.CONFIDENCE_BASED:
            if len(scores_at_level) < 2:
                return False
            mean_score = sum(scores_at_level) / len(scores_at_level)
            std_score = (sum((s - mean_score) ** 2 for s in scores_at_level) /
                         len(scores_at_level)) ** 0.5
            return score < mean_score - 1.5 * std_score
        return False

    def effective_discount(self, depth: int) -> float:
        """Compute the effective discount factor for a given tree depth.

        Deeper nodes receive lower effective value through discounting.

        Args:
            depth: Current depth in the thought tree.

        Returns:
            Discounted value multiplier for this depth.
        """
        return self.discount_factor ** depth

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                result[f.name] = value.value
            else:
                result[f.name] = value
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation of the configuration.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TreeOfThoughtConfig:
        """Deserialize configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            A TreeOfThoughtConfig instance.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {}
        for key, value in data.items():
            if key not in valid_fields:
                raise TypeError(f"Unexpected configuration key: {key}")
            if key == "evaluation_method" and isinstance(value, str):
                value = EvaluationMethod.from_string(value)
            if key == "pruning_strategy" and isinstance(value, str):
                value = PruningStrategy.from_string(value)
            if key == "selection_strategy" and isinstance(value, str):
                value = SelectionStrategy.from_string(value)
            filtered_data[key] = value
        return cls(**filtered_data)

    def config_hash(self) -> str:
        """Compute a stable hash of this configuration for caching.

        Returns:
            SHA-256 hash hex string prefix of the configuration.
        """
        serialized = self.to_json()
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# PlanningConfig
# =============================================================================

@dataclass(frozen=True)
class PlanningConfig:
    """Configuration for goal-oriented planning reasoning.

    Controls how complex goals are decomposed into subgoals, ordered, executed,
    and monitored during reasoning.

    Attributes:
        max_plan_steps: Maximum number of steps in a generated plan.
        replanning_enabled: Whether to allow dynamic replanning during execution.
        execution_budget: Total resource budget for plan execution (in tokens).
        subgoal_decomposition: Maximum depth of subgoal nesting.
        replanning_triggers: Set of conditions that trigger replanning.
        confidence_threshold_for_replan: Confidence below which to trigger replanning.
        max_replan_attempts: Maximum number of replanning attempts.
        parallel_execution: Whether to execute independent subgoals in parallel.
        dependency_resolution: Method for resolving dependencies between subgoals.
        plan_validation: Whether to validate plans before execution.
        timeout_per_step: Maximum time per subgoal execution (in seconds).
        progress_monitoring_interval: How often to check plan progress (in steps).
        plan_complexity_limit: Maximum complexity score allowed for plans.
        failure_recovery: Whether to attempt recovery when a subgoal fails.
        recovery_strategies: Ordered list of recovery strategies to try.
    """

    max_plan_steps: int = 15
    replanning_enabled: bool = True
    execution_budget: int = 10000
    subgoal_decomposition: int = 3
    replanning_triggers: Tuple[ReplanningTrigger, ...] = (
        ReplanningTrigger.STEP_FAILURE,
        ReplanningTrigger.LOW_CONFIDENCE,
        ReplanningTrigger.BUDGET_EXCEEDED,
    )
    confidence_threshold_for_replan: float = 0.3
    max_replan_attempts: int = 3
    parallel_execution: bool = True
    dependency_resolution: str = "topological"
    plan_validation: bool = True
    timeout_per_step: float = 30.0
    progress_monitoring_interval: int = 2
    plan_complexity_limit: float = 100.0
    failure_recovery: bool = True
    recovery_strategies: Tuple[str, ...] = (
        "retry",
        "skip",
        "replan",
        "escalate",
    )

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        validate_positive_integer("max_plan_steps", self.max_plan_steps, max_val=100)
        validate_positive_integer("execution_budget", self.execution_budget, max_val=1000000)
        validate_positive_integer("subgoal_decomposition", self.subgoal_decomposition, max_val=10)
        validate_range(
            "confidence_threshold_for_replan",
            self.confidence_threshold_for_replan,
            0.0, 1.0,
        )
        validate_positive_integer("max_replan_attempts", self.max_replan_attempts, max_val=20)
        validate_range("timeout_per_step", self.timeout_per_step, 0.1, 600.0)
        validate_positive_integer("progress_monitoring_interval", self.progress_monitoring_interval)
        validate_range("plan_complexity_limit", self.plan_complexity_limit, 1.0, 10000.0)
        valid_resolutions = {"topological", "priority", "manual"}
        if self.dependency_resolution not in valid_resolutions:
            raise ValueError(
                f"dependency_resolution must be one of {valid_resolutions}, "
                f"got '{self.dependency_resolution}'"
            )

    def has_replan_trigger(self, trigger: ReplanningTrigger) -> bool:
        """Check if a specific replanning trigger is enabled.

        Args:
            trigger: The trigger to check.

        Returns:
            True if the trigger is enabled.
        """
        return trigger in self.replanning_triggers

    def budget_remaining(
        self,
        tokens_used: int,
        steps_completed: int,
    ) -> Tuple[int, float]:
        """Calculate remaining budget as tokens and percentage.

        Args:
            tokens_used: Tokens consumed so far.
            steps_completed: Number of plan steps completed.

        Returns:
            Tuple of (remaining tokens, percentage remaining).
        """
        remaining = max(0, self.execution_budget - tokens_used)
        percentage = remaining / self.execution_budget if self.execution_budget > 0 else 0.0
        return remaining, percentage

    def is_over_budget(self, tokens_used: int) -> bool:
        """Check if the plan has exceeded its execution budget.

        Args:
            tokens_used: Tokens consumed so far.

        Returns:
            True if the budget has been exceeded.
        """
        return tokens_used >= self.execution_budget

    def get_recovery_strategy(self, attempt_index: int) -> str:
        """Get the recovery strategy to use for a given attempt.

        Args:
            attempt_index: Zero-indexed recovery attempt number.

        Returns:
            Name of the recovery strategy to use.
        """
        if not self.failure_recovery:
            return "escalate"
        if attempt_index < len(self.recovery_strategies):
            return self.recovery_strategies[attempt_index]
        return self.recovery_strategies[-1] if self.recovery_strategies else "escalate"

    def complexity_of_plan(self, num_subgoals: int, num_dependencies: int) -> float:
        """Estimate the complexity of a plan.

        Complexity is based on the number of subgoals, dependencies,
        and the decomposition depth.

        Args:
            num_subgoals: Number of subgoals in the plan.
            num_dependencies: Number of dependency edges between subgoals.

        Returns:
            Estimated complexity score.
        """
        goal_complexity = num_subgoals * 1.0
        dep_complexity = num_dependencies * 0.5
        depth_factor = self.subgoal_decomposition * 2.0
        return goal_complexity + dep_complexity + depth_factor

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                result[f.name] = value.value
            elif isinstance(value, tuple):
                result[f.name] = [v.value if isinstance(v, Enum) else v for v in value]
            else:
                result[f.name] = value
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanningConfig:
        """Deserialize configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            A PlanningConfig instance.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {}
        for key, value in data.items():
            if key not in valid_fields:
                raise TypeError(f"Unexpected configuration key: {key}")
            if key == "replanning_triggers" and isinstance(value, list):
                converted = []
                for v in value:
                    if isinstance(v, str):
                        converted.append(ReplanningTrigger.from_string(v))
                    else:
                        converted.append(v)
                value = tuple(converted)
            if key == "recovery_strategies" and isinstance(value, list):
                value = tuple(value)
            filtered_data[key] = value
        return cls(**filtered_data)

    def config_hash(self) -> str:
        """Compute a stable hash of this configuration.

        Returns:
            SHA-256 hash hex string prefix.
        """
        serialized = self.to_json()
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# VerificationConfig
# =============================================================================

@dataclass(frozen=True)
class VerificationConfig:
    """Configuration for answer verification systems.

    Controls how answers are verified across multiple methods and how
    consensus is reached among different verifiers.

    Attributes:
        num_verifiers: Number of independent verification methods to use.
        consensus_threshold: Agreement level required for consensus (0-1).
        backtracking_enabled: Whether to allow backtracking after failed verification.
        verification_methods: Ordered list of verification methods to apply.
        max_verification_attempts: Maximum attempts to verify an answer.
        verification_timeout: Maximum time for each verification attempt (seconds).
        strict_mode: Whether to reject answers with any verification failure.
        detailed_reporting: Whether to generate detailed verification reports.
        cross_checking: Whether to cross-check between different verification methods.
        fallback_strategy: Strategy when all verification methods fail.
        confidence_calibration: Whether to calibrate confidence scores.
        parallel_verification: Whether to run verifiers in parallel.
        min_evidence_threshold: Minimum evidence strength to accept verification.
    """

    num_verifiers: int = 3
    consensus_threshold: float = 0.7
    backtracking_enabled: bool = True
    verification_methods: Tuple[str, ...] = (
        "self_verification",
        "cross_verification",
        "backward_verification",
    )
    max_verification_attempts: int = 3
    verification_timeout: float = 10.0
    strict_mode: bool = False
    detailed_reporting: bool = True
    cross_checking: bool = True
    fallback_strategy: str = "accept_with_low_confidence"
    confidence_calibration: bool = True
    parallel_verification: bool = True
    min_evidence_threshold: float = 0.3

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        validate_positive_integer("num_verifiers", self.num_verifiers, max_val=20)
        validate_range("consensus_threshold", self.consensus_threshold, 0.0, 1.0)
        validate_positive_integer("max_verification_attempts", self.max_verification_attempts, max_val=10)
        validate_range("verification_timeout", self.verification_timeout, 0.1, 300.0)
        validate_range("min_evidence_threshold", self.min_evidence_threshold, 0.0, 1.0)
        valid_methods = {
            "self_verification",
            "cross_verification",
            "backward_verification",
            "formal_verification",
            "execution_verification",
            "consensus_verification",
        }
        for method in self.verification_methods:
            if method not in valid_methods:
                raise ValueError(
                    f"Unknown verification method '{method}'. "
                    f"Valid methods: {valid_methods}"
                )
        valid_fallbacks = {
            "accept_with_low_confidence",
            "reject",
            "retry",
            "escalate",
        }
        if self.fallback_strategy not in valid_fallbacks:
            raise ValueError(
                f"fallback_strategy must be one of {valid_fallbacks}, "
                f"got '{self.fallback_strategy}'"
            )

    def has_verification_method(self, method_name: str) -> bool:
        """Check if a specific verification method is enabled.

        Args:
            method_name: Name of the verification method.

        Returns:
            True if the method is enabled.
        """
        return method_name in self.verification_methods

    def compute_consensus(self, results: List[bool]) -> Tuple[bool, float]:
        """Compute consensus from a list of verification results.

        Args:
            results: List of boolean verification results.

        Returns:
            Tuple of (is_consensus_reached, agreement_ratio).
        """
        if not results:
            return False, 0.0
        positive_count = sum(1 for r in results if r)
        agreement_ratio = positive_count / len(results)
        is_consensus = agreement_ratio >= self.consensus_threshold
        return is_consensus, agreement_ratio

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                result[f.name] = list(value)
            else:
                result[f.name] = value
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationConfig:
        """Deserialize configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            A VerificationConfig instance.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {}
        for key, value in data.items():
            if key not in valid_fields:
                raise TypeError(f"Unexpected configuration key: {key}")
            if key == "verification_methods" and isinstance(value, list):
                value = tuple(value)
            filtered_data[key] = value
        return cls(**filtered_data)

    def config_hash(self) -> str:
        """Compute a stable hash of this configuration.

        Returns:
            SHA-256 hash hex string prefix.
        """
        serialized = self.to_json()
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# SelfConsistencyConfig
# =============================================================================

@dataclass(frozen=True)
class SelfConsistencyConfig:
    """Configuration for self-consistency decoding.

    Controls how multiple diverse solutions are generated and aggregated
    to produce more reliable final answers.

    Attributes:
        num_samples: Number of independent solution samples to generate.
        temperature_range: Range of temperatures for diverse sampling.
        aggregation_method: Method for aggregating solutions into a final answer.
        min_agreement_ratio: Minimum ratio of samples that must agree.
        outlier_detection: Whether to detect and exclude outlier solutions.
        cluster_threshold: Similarity threshold for clustering solutions.
        max_temperature: Maximum temperature for sampling diversity.
        min_temperature: Minimum temperature for sampling.
        include_reasoning_traces: Whether to include reasoning traces in samples.
        normalize_solutions: Whether to normalize solutions before comparison.
        diversity_penalty: Penalty for similar solutions (encourages diversity).
        quality_weighting: Whether to weight solutions by estimated quality.
        early_stopping_ratio: Stop sampling early if this ratio agrees.
        seed_selection_method: How to select the initial reasoning seed.
    """

    num_samples: int = 10
    temperature_range: Tuple[float, float] = (0.3, 1.0)
    aggregation_method: AggregationMethod = AggregationMethod.MAJORITY_VOTE
    min_agreement_ratio: float = 0.5
    outlier_detection: bool = True
    cluster_threshold: float = 0.7
    max_temperature: float = 1.0
    min_temperature: float = 0.3
    include_reasoning_traces: bool = True
    normalize_solutions: bool = True
    diversity_penalty: float = 0.1
    quality_weighting: bool = True
    early_stopping_ratio: float = 0.8
    seed_selection_method: str = "diverse"

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        validate_positive_integer("num_samples", self.num_samples, max_val=100)
        validate_range("min_agreement_ratio", self.min_agreement_ratio, 0.0, 1.0)
        validate_range("cluster_threshold", self.cluster_threshold, 0.0, 1.0)
        validate_range("max_temperature", self.max_temperature, 0.0, 2.0)
        validate_range("min_temperature", self.min_temperature, 0.0, 2.0)
        if self.min_temperature >= self.max_temperature:
            raise ValueError(
                f"min_temperature ({self.min_temperature}) must be less than "
                f"max_temperature ({self.max_temperature})"
            )
        validate_range("diversity_penalty", self.diversity_penalty, 0.0, 1.0)
        validate_range("early_stopping_ratio", self.early_stopping_ratio, 0.0, 1.0)
        valid_seeds = {"diverse", "random", "best", "centroid"}
        if self.seed_selection_method not in valid_seeds:
            raise ValueError(
                f"seed_selection_method must be one of {valid_seeds}, "
                f"got '{self.seed_selection_method}'"
            )

    def get_temperatures(self) -> List[float]:
        """Generate a list of temperatures spread across the configured range.

        Creates evenly spaced temperatures from min to max, with the number
        of distinct values equal to num_samples.

        Returns:
            List of temperature values for sampling.
        """
        if self.num_samples == 1:
            return [(self.min_temperature + self.max_temperature) / 2.0]
        step = (self.max_temperature - self.min_temperature) / (self.num_samples - 1)
        temperatures = []
        for i in range(self.num_samples):
            temp = self.min_temperature + i * step
            temperatures.append(round(temp, 4))
        return temperatures

    def check_early_stopping(self, agreement_ratio: float, samples_collected: int) -> bool:
        """Check if early stopping criteria are met.

        Args:
            agreement_ratio: Current ratio of agreement among collected samples.
            samples_collected: Number of samples collected so far.

        Returns:
            True if early stopping should be triggered.
        """
        min_samples = max(3, self.num_samples // 3)
        if samples_collected < min_samples:
            return False
        return agreement_ratio >= self.early_stopping_ratio

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                result[f.name] = value.value
            elif isinstance(value, tuple):
                result[f.name] = list(value)
            else:
                result[f.name] = value
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SelfConsistencyConfig:
        """Deserialize configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            A SelfConsistencyConfig instance.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {}
        for key, value in data.items():
            if key not in valid_fields:
                raise TypeError(f"Unexpected configuration key: {key}")
            if key == "aggregation_method" and isinstance(value, str):
                value = AggregationMethod.from_string(value)
            if key == "temperature_range" and isinstance(value, list):
                value = tuple(value)
            filtered_data[key] = value
        return cls(**filtered_data)

    def config_hash(self) -> str:
        """Compute a stable hash of this configuration.

        Returns:
            SHA-256 hash hex string prefix.
        """
        serialized = self.to_json()
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# ReasoningConfig (Top-Level)
# =============================================================================

@dataclass(frozen=True)
class ReasoningConfig:
    """Top-level configuration for the Nexus reasoning module.

    Orchestrates all reasoning strategies and provides a unified interface
    for configuring the reasoning pipeline. Individual strategy configs
    can be overridden independently.

    Attributes:
        default_strategy: The default reasoning strategy to use.
        max_reasoning_tokens: Maximum total tokens for reasoning across all steps.
        reasoning_budget: Total compute budget allocated to reasoning (in tokens).
        tools_enabled: Whether external tools (calculator, search) are available.
        memory_enabled: Whether to use episodic memory across reasoning steps.
        chain_of_thought: Configuration for chain-of-thought reasoning.
        tree_of_thought: Configuration for tree-of-thought reasoning.
        planning: Configuration for planning-based reasoning.
        verification: Configuration for answer verification.
        self_consistency: Configuration for self-consistency decoding.
        enable_caching: Whether to cache intermediate reasoning results.
        cache_ttl: Time-to-live for cached reasoning results (seconds).
        logging_level: Logging verbosity for reasoning operations.
        max_concurrent_strategies: Maximum strategies that can run concurrently.
        fallback_strategy: Strategy to use when the primary strategy fails.
        retry_on_failure: Whether to retry failed reasoning attempts.
        max_retries: Maximum number of retry attempts.
        output_format: Format for reasoning output (text, json, structured).
        include_metadata: Whether to include reasoning metadata in output.
    """

    default_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    max_reasoning_tokens: int = 8192
    reasoning_budget: int = 16384
    tools_enabled: bool = False
    memory_enabled: bool = False
    chain_of_thought: ChainOfThoughtConfig = field(default_factory=ChainOfThoughtConfig)
    tree_of_thought: TreeOfThoughtConfig = field(default_factory=TreeOfThoughtConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    self_consistency: SelfConsistencyConfig = field(default_factory=SelfConsistencyConfig)
    enable_caching: bool = True
    cache_ttl: float = 300.0
    logging_level: str = "INFO"
    max_concurrent_strategies: int = 3
    fallback_strategy: Optional[ReasoningStrategy] = None
    retry_on_failure: bool = True
    max_retries: int = 2
    output_format: str = "text"
    include_metadata: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        validate_enum_value("default_strategy", self.default_strategy, ReasoningStrategy)
        validate_positive_integer("max_reasoning_tokens", self.max_reasoning_tokens, max_val=1000000)
        validate_positive_integer("reasoning_budget", self.reasoning_budget, max_val=10000000)
        validate_range("cache_ttl", self.cache_ttl, 0.0, 86400.0)
        validate_positive_integer("max_concurrent_strategies", self.max_concurrent_strategies, max_val=10)
        validate_positive_integer("max_retries", self.max_retries, max_val=10)
        valid_formats = {"text", "json", "structured", "verbose"}
        if self.output_format not in valid_formats:
            raise ValueError(
                f"output_format must be one of {valid_formats}, "
                f"got '{self.output_format}'"
            )
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.logging_level not in valid_log_levels:
            raise ValueError(
                f"logging_level must be one of {valid_log_levels}, "
                f"got '{self.logging_level}'"
            )
        if self.fallback_strategy is not None:
            validate_enum_value("fallback_strategy", self.fallback_strategy, ReasoningStrategy)
        if self.max_reasoning_tokens > self.reasoning_budget:
            raise ValueError(
                f"max_reasoning_tokens ({self.max_reasoning_tokens}) cannot exceed "
                f"reasoning_budget ({self.reasoning_budget})"
            )

    def get_strategy_config(self, strategy: Optional[ReasoningStrategy] = None) -> Any:
        """Get the configuration for a specific reasoning strategy.

        Args:
            strategy: The strategy to get config for. Uses default if None.

        Returns:
            The configuration object for the specified strategy.

        Raises:
            ValueError: If the strategy is not recognized.
        """
        target = strategy or self.default_strategy
        config_map = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: self.chain_of_thought,
            ReasoningStrategy.TREE_OF_THOUGHT: self.tree_of_thought,
            ReasoningStrategy.PLANNING: self.planning,
            ReasoningStrategy.SELF_CONSISTENCY: self.self_consistency,
            ReasoningStrategy.RETRIEVAL_AUGMENTED: None,
            ReasoningStrategy.HYBRID: None,
        }
        if target not in config_map:
            raise ValueError(f"Unknown strategy: {target}")
        return config_map[target]

    def budget_remaining(self, tokens_used: int) -> Tuple[int, float]:
        """Calculate remaining reasoning budget.

        Args:
            tokens_used: Tokens consumed so far.

        Returns:
            Tuple of (remaining tokens, percentage remaining).
        """
        remaining = max(0, self.reasoning_budget - tokens_used)
        percentage = remaining / self.reasoning_budget if self.reasoning_budget > 0 else 0.0
        return remaining, percentage

    def is_over_budget(self, tokens_used: int) -> bool:
        """Check if reasoning has exceeded the token budget.

        Args:
            tokens_used: Tokens consumed so far.

        Returns:
            True if the budget has been exceeded.
        """
        return tokens_used >= self.reasoning_budget

    def should_retry(self, attempt_count: int, error_type: Optional[str] = None) -> bool:
        """Determine whether to retry a failed reasoning attempt.

        Args:
            attempt_count: Number of attempts made so far.
            error_type: Type of error that caused the failure.

        Returns:
            True if retrying is recommended.
        """
        if not self.retry_on_failure:
            return False
        if attempt_count >= self.max_retries:
            return False
        if error_type == "budget_exceeded":
            return False
        return True

    def get_effective_strategy(self, task_complexity: float) -> ReasoningStrategy:
        """Select the most appropriate strategy based on task complexity.

        Uses heuristic thresholds to select a strategy that balances accuracy
        and computational cost.

        Args:
            task_complexity: Estimated complexity score for the task (0.0-1.0).

        Returns:
            Recommended reasoning strategy.
        """
        if task_complexity < 0.2:
            return ReasoningStrategy.CHAIN_OF_THOUGHT
        if task_complexity < 0.5:
            return ReasoningStrategy.SELF_CONSISTENCY
        if task_complexity < 0.7:
            return ReasoningStrategy.TREE_OF_THOUGHT
        if task_complexity < 0.9:
            return ReasoningStrategy.PLANNING
        return ReasoningStrategy.HYBRID

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full configuration to a nested dictionary.

        Returns:
            Dictionary representation of all configuration values.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                result[f.name] = value.value
            elif hasattr(value, "to_dict"):
                result[f.name] = value.to_dict()
            else:
                result[f.name] = value
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation of the full configuration.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningConfig:
        """Deserialize configuration from a dictionary.

        Handles nested configuration objects and enum conversions.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            A ReasoningConfig instance.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {}
        for key, value in data.items():
            if key not in valid_fields:
                raise TypeError(f"Unexpected configuration key: {key}")
            if key == "default_strategy" and isinstance(value, str):
                value = ReasoningStrategy.from_string(value)
            elif key == "fallback_strategy" and isinstance(value, str):
                value = ReasoningStrategy.from_string(value)
            elif key == "chain_of_thought" and isinstance(value, dict):
                value = ChainOfThoughtConfig.from_dict(value)
            elif key == "tree_of_thought" and isinstance(value, dict):
                value = TreeOfThoughtConfig.from_dict(value)
            elif key == "planning" and isinstance(value, dict):
                value = PlanningConfig.from_dict(value)
            elif key == "verification" and isinstance(value, dict):
                value = VerificationConfig.from_dict(value)
            elif key == "self_consistency" and isinstance(value, dict):
                value = SelfConsistencyConfig.from_dict(value)
            filtered_data[key] = value
        return cls(**filtered_data)

    @classmethod
    def from_json(cls, json_string: str) -> ReasoningConfig:
        """Deserialize configuration from a JSON string.

        Args:
            json_string: JSON string containing configuration.

        Returns:
            A ReasoningConfig instance.
        """
        data = json.loads(json_string)
        return cls.from_dict(data)

    def config_hash(self) -> str:
        """Compute a stable hash of this entire configuration.

        Returns:
            SHA-256 hash hex string prefix.
        """
        serialized = self.to_json()
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save configuration to a JSON file.

        Args:
            filepath: Path to the output JSON file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=2))

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> ReasoningConfig:
        """Load configuration from a JSON file.

        Args:
            filepath: Path to the input JSON file.

        Returns:
            A ReasoningConfig instance.
        """
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            json_string = f.read()
        return cls.from_json(json_string)


# =============================================================================
# Configuration Utilities
# =============================================================================

def merge_configs(
    base: ReasoningConfig,
    override: Dict[str, Any],
) -> ReasoningConfig:
    """Merge an override dictionary into a base ReasoningConfig.

    Only overrides values that are explicitly provided. Nested configs are
    deep-merged rather than replaced entirely.

    Args:
        base: The base configuration to merge into.
        override: Dictionary of override values.

    Returns:
        A new ReasoningConfig with overrides applied.
    """
    base_dict = base.to_dict()

    def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge source dict into target dict."""
        result = copy.deepcopy(target)
        for key, value in source.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged_dict = deep_merge(base_dict, override)
    return ReasoningConfig.from_dict(merged_dict)


def validate_config_compatibility(config: ReasoningConfig) -> List[str]:
    """Validate that all sub-configurations are mutually compatible.

    Checks for logical inconsistencies between different strategy
    configurations that could lead to unexpected behavior.

    Args:
        config: The ReasoningConfig to validate.

    Returns:
        List of warning messages for any incompatibilities found.
    """
    warnings: List[str] = []

    if config.max_reasoning_tokens > config.reasoning_budget * 0.9:
        warnings.append(
            "max_reasoning_tokens is close to or exceeds reasoning_budget, "
            "leaving insufficient budget for other operations"
        )

    if config.chain_of_thought.max_steps > 20 and config.tree_of_thought.depth_limit > 10:
        warnings.append(
            "Both chain_of_thought.max_steps and tree_of_thought.depth_limit "
            "are set high, which may lead to excessive resource usage"
        )

    if (config.self_consistency.num_samples > 20
            and config.verification.parallel_verification):
        warnings.append(
            "High self_consistency.num_samples with parallel verification "
            "may cause resource contention"
        )

    if (config.planning.max_plan_steps > config.chain_of_thought.max_steps
            and config.default_strategy == ReasoningStrategy.PLANNING):
        warnings.append(
            "planning.max_plan_steps exceeds chain_of_thought.max_steps, "
            "which may cause incomplete reasoning in fallback scenarios"
        )

    if config.tools_enabled and config.memory_enabled:
        if config.max_concurrent_strategies > 3:
            warnings.append(
                "Both tools and memory are enabled with high concurrency, "
                "which may lead to memory pressure"
            )

    return warnings


def get_default_config() -> ReasoningConfig:
    """Create a ReasoningConfig with sensible production defaults.

    Returns:
        A ReasoningConfig instance with default values optimized for
        general-purpose reasoning tasks.
    """
    return ReasoningConfig()


def get_high_accuracy_config() -> ReasoningConfig:
    """Create a ReasoningConfig optimized for maximum accuracy.

    Uses more aggressive reasoning strategies at the cost of
    increased computational resources.

    Returns:
        A ReasoningConfig optimized for accuracy.
    """
    return ReasoningConfig(
        default_strategy=ReasoningStrategy.HYBRID,
        max_reasoning_tokens=16384,
        reasoning_budget=32768,
        chain_of_thought=ChainOfThoughtConfig(
            max_steps=20,
            temperature=0.2,
            enable_verification=True,
            enable_pruning=True,
            backtracking_mode=BacktrackingMode.SMART,
            max_backtracks=5,
        ),
        tree_of_thought=TreeOfThoughtConfig(
            num_thoughts=8,
            breadth_limit=8,
            depth_limit=12,
            num_simulations=50,
            search_algorithm="mcts",
        ),
        verification=VerificationConfig(
            num_verifiers=5,
            consensus_threshold=0.8,
            strict_mode=True,
            verification_methods=(
                "self_verification",
                "cross_verification",
                "backward_verification",
                "formal_verification",
                "execution_verification",
            ),
        ),
        self_consistency=SelfConsistencyConfig(
            num_samples=20,
            min_agreement_ratio=0.6,
            quality_weighting=True,
        ),
    )


def get_fast_config() -> ReasoningConfig:
    """Create a ReasoningConfig optimized for speed.

    Uses minimal reasoning for quick responses.

    Returns:
        A ReasoningConfig optimized for low latency.
    """
    return ReasoningConfig(
        default_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        max_reasoning_tokens=2048,
        reasoning_budget=4096,
        chain_of_thought=ChainOfThoughtConfig(
            max_steps=5,
            temperature=0.1,
            enable_verification=False,
            enable_pruning=False,
            backtracking_mode=BacktrackingMode.DISABLED,
        ),
        verification=VerificationConfig(
            num_verifiers=1,
            consensus_threshold=0.5,
            strict_mode=False,
            parallel_verification=False,
        ),
        self_consistency=SelfConsistencyConfig(
            num_samples=3,
            min_agreement_ratio=0.5,
        ),
    )


def get_balanced_config() -> ReasoningConfig:
    """Create a ReasoningConfig balanced between accuracy and speed.

    Returns:
        A ReasoningConfig with balanced settings.
    """
    return ReasoningConfig(
        default_strategy=ReasoningStrategy.SELF_CONSISTENCY,
        max_reasoning_tokens=8192,
        reasoning_budget=16384,
        chain_of_thought=ChainOfThoughtConfig(
            max_steps=10,
            temperature=0.3,
            enable_verification=True,
            enable_pruning=True,
            backtracking_mode=BacktrackingMode.SMART,
            max_backtracks=2,
        ),
        tree_of_thought=TreeOfThoughtConfig(
            num_thoughts=5,
            breadth_limit=5,
            depth_limit=8,
            num_simulations=20,
        ),
        verification=VerificationConfig(
            num_verifiers=3,
            consensus_threshold=0.7,
        ),
        self_consistency=SelfConsistencyConfig(
            num_samples=8,
            min_agreement_ratio=0.5,
        ),
    )
