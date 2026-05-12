"""
Chain-of-Thought Reasoning Module
===================================

Implements comprehensive chain-of-thought (CoT) reasoning systems for the Nexus LLM
framework. Chain-of-thought reasoning enables language models to solve complex problems
by breaking them down into intermediate reasoning steps, mimicking human cognitive
processes.

This module provides multiple CoT variants:
- ZeroShotCoT: Zero-shot prompting with "Let's think step by step"
- FewShotCoT: Few-shot examples to guide reasoning
- AutoCoT: Automatic construction of reasoning examples via clustering
- StructuredCoT: Structured output with explicit Given/Reasoning/Conclusion fields

The core CoTReasoner class provides advanced capabilities including step verification,
backtracking, trace compression, pattern detection, and confidence estimation.

References:
    - Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    - Kojima et al. (2022) "Large Language Models are Zero-Shot Reasoners"
    - Zhang et al. (2022) "Automatic Chain of Thought Prompting in Large Language Models"
"""

from __future__ import annotations

import math
import re
import copy
import json
import time
import hashlib
import logging
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, Counter, OrderedDict
from dataclasses import dataclass, field, asdict, fields
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ReasoningStepType(Enum):
    """Classification of individual reasoning steps.

    Each step in a reasoning chain can be classified by its function
    in the overall reasoning process.

    Attributes:
        THOUGHT: A reasoning thought or deliberation step.
        ACTION: An action taken based on reasoning (e.g., calculation, lookup).
        OBSERVATION: An observation or fact noted during reasoning.
        CONCLUSION: A conclusion drawn from the reasoning chain.
        ASSUMPTION: An explicit assumption made during reasoning.
        CORRECTION: A correction to a previous reasoning step.
        VERIFICATION: A step where a previous result is verified.
    """
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    CONCLUSION = "conclusion"
    ASSUMPTION = "assumption"
    CORRECTION = "correction"
    VERIFICATION = "verification"


class ReasoningPattern(Enum):
    """Identified patterns in reasoning chains.

    Different reasoning approaches follow distinct logical patterns
    that can be detected and classified.

    Attributes:
        DIRECT: Direct reasoning from premises to conclusion.
        DEDUCTIVE: Top-down reasoning from general to specific (syllogistic).
        INDUCTIVE: Bottom-up reasoning from specific to general.
        ABDUCTIVE: Inference to the best explanation.
        ANALOGICAL: Reasoning by analogy from similar cases.
        CAUSAL: Cause-and-effect reasoning chain.
        CONDITIONAL: Hypothetical/conditional reasoning.
        MIXED: Combination of multiple reasoning patterns.
        UNKNOWN: Pattern could not be determined.
    """
    DIRECT = "direct"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    CONDITIONAL = "conditional"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class ReasoningStep:
    """A single step in a chain-of-thought reasoning trace.

    Each step represents one unit of reasoning, including the thought process,
    any actions taken, observations made, and confidence in the step.

    Attributes:
        thought: The reasoning thought or deliberation content.
        action: Optional action taken based on the reasoning.
        observation: Optional observation resulting from the action.
        confidence: Confidence score for this reasoning step (0.0 to 1.0).
        timestamp: Unix timestamp when this step was created.
        step_type: Classification of this reasoning step.
        step_index: Zero-indexed position in the reasoning chain.
        metadata: Additional metadata about this step.
        token_count: Estimated number of tokens in this step.
        dependencies: Indices of steps this step depends on.
        verification_status: Whether this step has been verified.
        correction_of: Index of step being corrected, if this is a correction.
        source: Source of this step (model, user, external).
    """
    thought: str = ""
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    step_type: ReasoningStepType = ReasoningStepType.THOUGHT
    step_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    dependencies: List[int] = field(default_factory=list)
    verification_status: bool = False
    correction_of: Optional[int] = None
    source: str = "model"

    def __post_init__(self) -> None:
        """Initialize derived fields after construction."""
        if self.token_count == 0:
            self.token_count = self._estimate_tokens()

    def _estimate_tokens(self) -> int:
        """Estimate the number of tokens in this step.

        Uses a simple heuristic of approximately 4 characters per token.

        Returns:
            Estimated token count.
        """
        total_chars = len(self.thought)
        if self.action:
            total_chars += len(self.action)
        if self.observation:
            total_chars += len(self.observation)
        return max(1, total_chars // 4)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this step to a dictionary.

        Returns:
            Dictionary representation of the reasoning step.
        """
        return {
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "step_type": self.step_type.value,
            "step_index": self.step_index,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "dependencies": self.dependencies,
            "verification_status": self.verification_status,
            "correction_of": self.correction_of,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningStep:
        """Deserialize a reasoning step from a dictionary.

        Args:
            data: Dictionary containing step data.

        Returns:
            A ReasoningStep instance.
        """
        step_type_str = data.get("step_type", "thought")
        step_type = ReasoningStepType(step_type_str)
        return cls(
            thought=data.get("thought", ""),
            action=data.get("action"),
            observation=data.get("observation"),
            confidence=data.get("confidence", 0.5),
            timestamp=data.get("timestamp", time.time()),
            step_type=step_type,
            step_index=data.get("step_index", 0),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count", 0),
            dependencies=data.get("dependencies", []),
            verification_status=data.get("verification_status", False),
            correction_of=data.get("correction_of"),
            source=data.get("source", "model"),
        )

    def format(self, include_metadata: bool = False) -> str:
        """Format this step as a human-readable string.

        Args:
            include_metadata: Whether to include metadata in the output.

        Returns:
            Formatted string representation of the step.
        """
        parts = [f"Step {self.step_index + 1} [{self.step_type.value}]: {self.thought}"]
        if self.action:
            parts.append(f"  Action: {self.action}")
        if self.observation:
            parts.append(f"  Observation: {self.observation}")
        parts.append(f"  Confidence: {self.confidence:.2f}")
        if include_metadata and self.metadata:
            parts.append(f"  Metadata: {json.dumps(self.metadata, default=str)}")
        return "\n".join(parts)

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if this step has high confidence.

        Args:
            threshold: Confidence threshold.

        Returns:
            True if confidence exceeds the threshold.
        """
        return self.confidence >= threshold

    def is_low_confidence(self, threshold: float = 0.3) -> bool:
        """Check if this step has low confidence.

        Args:
            threshold: Low confidence threshold.

        Returns:
            True if confidence is below the threshold.
        """
        return self.confidence < threshold

    def clone(self) -> ReasoningStep:
        """Create a deep copy of this reasoning step.

        Returns:
            A new ReasoningStep with identical data.
        """
        return ReasoningStep(
            thought=self.thought,
            action=copy.deepcopy(self.action),
            observation=copy.deepcopy(self.observation),
            confidence=self.confidence,
            timestamp=self.timestamp,
            step_type=self.step_type,
            step_index=self.step_index,
            metadata=copy.deepcopy(self.metadata),
            token_count=self.token_count,
            dependencies=list(self.dependencies),
            verification_status=self.verification_status,
            correction_of=self.correction_of,
            source=self.source,
        )


@dataclass
class ReasoningTrace:
    """A complete chain of reasoning steps from a single reasoning process.

    Attributes:
        steps: Ordered list of reasoning steps.
        prompt: The original prompt that initiated reasoning.
        solution: The final solution or answer derived.
        total_confidence: Aggregate confidence score for the entire trace.
        pattern: Detected reasoning pattern.
        tokens_used: Total tokens consumed by this trace.
        duration_ms: Time taken to generate the trace in milliseconds.
        metadata: Additional metadata about the reasoning process.
    """
    steps: List[ReasoningStep] = field(default_factory=list)
    prompt: str = ""
    solution: str = ""
    total_confidence: float = 0.0
    pattern: ReasoningPattern = ReasoningPattern.UNKNOWN
    tokens_used: int = 0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the trace.

        Args:
            step: The reasoning step to add.
        """
        step.step_index = len(self.steps)
        self.steps.append(step)
        self.tokens_used += step.token_count
        self._update_confidence()

    def _update_confidence(self) -> None:
        """Recalculate the aggregate confidence from all steps."""
        if not self.steps:
            self.total_confidence = 0.0
            return
        confidences = [s.confidence for s in self.steps]
        self.total_confidence = statistics.mean(confidences)

    def get_step(self, index: int) -> Optional[ReasoningStep]:
        """Get a reasoning step by index.

        Args:
            index: Zero-indexed step position.

        Returns:
            The reasoning step, or None if index is out of range.
        """
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def truncate(self, max_steps: int) -> ReasoningTrace:
        """Create a truncated copy of this trace.

        Args:
            max_steps: Maximum number of steps to keep.

        Returns:
            A new ReasoningTrace with at most max_steps.
        """
        truncated = ReasoningTrace(
            steps=[s.clone() for s in self.steps[:max_steps]],
            prompt=self.prompt,
            solution=self.solution,
            pattern=self.pattern,
            metadata=copy.deepcopy(self.metadata),
        )
        truncated.tokens_used = sum(s.token_count for s in truncated.steps)
        truncated._update_confidence()
        return truncated

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the reasoning trace to a dictionary.

        Returns:
            Dictionary representation of the trace.
        """
        return {
            "steps": [s.to_dict() for s in self.steps],
            "prompt": self.prompt,
            "solution": self.solution,
            "total_confidence": self.total_confidence,
            "pattern": self.pattern.value,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningTrace:
        """Deserialize a reasoning trace from a dictionary.

        Args:
            data: Dictionary containing trace data.

        Returns:
            A ReasoningTrace instance.
        """
        pattern_str = data.get("pattern", "unknown")
        pattern = ReasoningPattern(pattern_str)
        steps = [ReasoningStep.from_dict(s) for s in data.get("steps", [])]
        trace = cls(
            steps=steps,
            prompt=data.get("prompt", ""),
            solution=data.get("solution", ""),
            total_confidence=data.get("total_confidence", 0.0),
            pattern=pattern,
            tokens_used=data.get("tokens_used", 0),
            duration_ms=data.get("duration_ms", 0.0),
            metadata=data.get("metadata", {}),
        )
        return trace

    def clone(self) -> ReasoningTrace:
        """Create a deep copy of this reasoning trace.

        Returns:
            A new ReasoningTrace with identical data.
        """
        return ReasoningTrace(
            steps=[s.clone() for s in self.steps],
            prompt=self.prompt,
            solution=self.solution,
            total_confidence=self.total_confidence,
            pattern=self.pattern,
            tokens_used=self.tokens_used,
            duration_ms=self.duration_ms,
            metadata=copy.deepcopy(self.metadata),
        )

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[ReasoningStep]:
        return iter(self.steps)

    def __getitem__(self, index: int) -> ReasoningStep:
        return self.steps[index]


# =============================================================================
# Model Interface Protocol
# =============================================================================

class ModelInterface(ABC):
    """Abstract interface that any LLM model must implement for reasoning.

    This protocol defines the minimal interface required by reasoning modules
    to interact with language models.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text from the model.

        Args:
            prompt: Input prompt for generation.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop_sequences: Optional stop sequences.

        Returns:
            Generated text string.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_with_logprobs(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Tuple[str, List[float]]:
        """Generate text with per-token log probabilities.

        Args:
            prompt: Input prompt for generation.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (generated_text, list_of_log_probabilities).
        """
        raise NotImplementedError

    @abstractmethod
    def score_text(self, text: str, context: str = "") -> float:
        """Score a piece of text for quality or relevance.

        Args:
            text: Text to score.
            context: Optional context for scoring.

        Returns:
            A score value.
        """
        raise NotImplementedError


class MockModel(ModelInterface):
    """Mock model implementation for testing and demonstration purposes.

    Simulates LLM behavior with pattern-based responses that provide
    reasonable facsimiles of chain-of-thought reasoning.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize the mock model.

        Args:
            seed: Random seed for reproducible behavior.
        """
        self._seed = seed
        self._call_count = 0
        self._responses: Dict[str, str] = {}

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate a mock response.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature (affects response length).
            max_tokens: Maximum tokens (affects response length).
            stop_sequences: Optional stop sequences.

        Returns:
            Mock generated text.
        """
        self._call_count += 1
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        if prompt_hash in self._responses:
            response = self._responses[prompt_hash]
        else:
            response = self._generate_mock_response(prompt, temperature)
            self._responses[prompt_hash] = response

        if stop_sequences:
            for seq in stop_sequences:
                idx = response.find(seq)
                if idx >= 0:
                    response = response[:idx]

        estimated_tokens = len(response) // 4
        if estimated_tokens > max_tokens:
            response = response[:max_tokens * 4]
        return response

    def _generate_mock_response(self, prompt: str, temperature: float) -> str:
        """Generate a mock reasoning response based on the prompt.

        Args:
            prompt: The input prompt.
            temperature: Temperature parameter.

        Returns:
            A mock reasoning response string.
        """
        prompt_lower = prompt.lower()
        length_factor = max(1, int(temperature * 3))

        if any(kw in prompt_lower for kw in ["calculate", "compute", "math", "how many"]):
            steps = [
                "Let me break this down step by step.",
                "First, I need to identify the key numbers and operations in the problem.",
                "I'll extract the numerical values and determine the correct mathematical operations.",
                "Let me perform the calculations carefully.",
                "I'll verify my intermediate results to make sure there are no errors.",
                "Now I can combine these results to reach the final answer.",
                "Therefore, based on my calculations, I can conclude the answer.",
            ]
        elif any(kw in prompt_lower for kw in ["why", "explain", "reason", "because"]):
            steps = [
                "Let me think through this systematically.",
                "I need to consider the key factors and their relationships.",
                "Looking at the evidence and logical connections,",
                "I can identify the primary cause or reason.",
                "There are several supporting arguments that reinforce this conclusion.",
                "Therefore, the explanation is as follows.",
            ]
        elif any(kw in prompt_lower for kw in ["compare", "difference", "similar", "versus"]):
            steps = [
                "Let me compare these systematically.",
                "First, I'll examine the key characteristics of each item.",
                "Looking at their similarities, I notice several common features.",
                "Now examining the differences, the key distinctions are:",
                "Based on this comparison, I can draw the following conclusions.",
            ]
        else:
            steps = [
                "Let me think about this step by step.",
                "I need to carefully analyze the given information.",
                "Breaking down the problem into manageable parts,",
                "I can identify the key elements that matter.",
                "Considering the relationships between these elements,",
                "I can form a reasoned conclusion.",
                "Therefore, my answer is based on the reasoning above.",
            ]

        selected = steps[:min(len(steps), 3 + length_factor)]
        return " ".join(selected)

    def generate_with_logprobs(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Tuple[str, List[float]]:
        """Generate text with mock log probabilities.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            Tuple of (text, log probabilities).
        """
        text = self.generate(prompt, temperature, max_tokens)
        num_tokens = max(1, len(text) // 4)
        base_logprob = -0.5 - temperature * 0.3
        logprobs = [base_logprob + (i % 5) * 0.1 for i in range(num_tokens)]
        return text, logprobs

    def score_text(self, text: str, context: str = "") -> float:
        """Score text with a mock scoring function.

        Args:
            text: Text to score.
            context: Optional context.

        Returns:
            A score between 0.0 and 1.0.
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_int = int(text_hash[:8], 16)
        score = (hash_int % 100) / 100.0
        if context:
            context_hash = hashlib.md5(context.encode()).hexdigest()
            context_int = int(context_hash[:8], 16)
            relevance = (context_int % 50) / 100.0
            score = score * 0.6 + relevance * 0.4
        return min(1.0, max(0.0, score))


# =============================================================================
# CoT Reasoner - Main Chain-of-Thought Engine
# =============================================================================

class CoTReasoner:
    """Main chain-of-thought reasoning engine.

    Manages the complete lifecycle of chain-of-thought reasoning including
    trace generation, step verification, backtracking, pattern detection,
    and quality evaluation.

    The reasoner supports multiple CoT variants and provides advanced features
    like automatic pruning, confidence estimation, and reasoning pattern detection.

    Attributes:
        config: Configuration for chain-of-thought reasoning.
        model: The language model interface used for generation.
        step_verifier: Optional custom step verification function.
        trace_history: History of all generated reasoning traces.
    """

    # Keywords that signal different reasoning patterns
    DEDUCTIVE_KEYWORDS = frozenset({
        "therefore", "thus", "hence", "consequently", "it follows",
        "we can deduce", "must be", "necessarily", "logically",
    })
    INDUCTIVE_KEYWORDS = frozenset({
        "in general", "typically", "usually", "pattern", "observation",
        "we observe", "tends to", "frequently", "commonly",
    })
    ABDUCTIVE_KEYWORDS = frozenset({
        "the best explanation", "most likely", "probably because",
        "suggests that", "hypothesis", "would explain",
    })
    ANALOGICAL_KEYWORDS = frozenset({
        "similar to", "analogous to", "just like", "in the same way",
        "comparable", "parallel", "likewise", "by analogy",
    })
    CAUSAL_KEYWORDS = frozenset({
        "because", "causes", "leads to", "results in", "due to",
        "contributes to", "effect", "impact", "influences",
    })
    CONDITIONAL_KEYWORDS = frozenset({
        "if", "assuming", "suppose", "what if", "in case",
        "provided that", "given that", "conditionally", "hypothetically",
    })

    def __init__(
        self,
        config: Any = None,
        model: Optional[ModelInterface] = None,
    ) -> None:
        """Initialize the chain-of-thought reasoner.

        Args:
            config: ChainOfThoughtConfig instance. Uses defaults if None.
            model: Language model interface. Uses MockModel if None.
        """
        if config is None:
            from nexus.reasoning.reasoning_config import ChainOfThoughtConfig
            config = ChainOfThoughtConfig()
        self.config = config
        self.model = model or MockModel()
        self.step_verifier: Optional[Callable[[str, str], float]] = None
        self.trace_history: List[ReasoningTrace] = []
        self._backtrack_count: int = 0
        self._total_tokens_used: int = 0
        self._cache: Dict[str, ReasoningTrace] = {}

    def generate_reasoning_trace(
        self,
        model: Optional[ModelInterface] = None,
        prompt: str = "",
        max_steps: Optional[int] = None,
    ) -> ReasoningTrace:
        """Generate a complete reasoning trace for a given prompt.

        Produces a full chain of reasoning steps from prompt to solution,
        with verification and backtracking when needed.

        Args:
            model: Optional model override for this generation.
            prompt: The input prompt or question to reason about.
            max_steps: Optional override for maximum reasoning steps.

        Returns:
            A complete ReasoningTrace with all steps and metadata.
        """
        active_model = model or self.model
        steps_limit = max_steps or self.config.max_steps
        trace = ReasoningTrace(prompt=prompt)
        start_time = time.time()
        context = prompt

        for step_num in range(steps_limit):
            effective_temp = self.config.get_effective_temperature(step_num)
            step_prompt = self._build_step_prompt(context, step_num, trace)

            response = active_model.generate(
                prompt=step_prompt,
                temperature=effective_temp,
                max_tokens=self.config.max_tokens_per_step,
                stop_sequences=list(self.config.stop_sequences),
            )

            thoughts = self.extract_thoughts(response)
            if not thoughts:
                break

            for thought_text in thoughts:
                step = ReasoningStep(
                    thought=thought_text,
                    confidence=0.5,
                    step_type=ReasoningStepType.THOUGHT,
                    step_index=len(trace.steps),
                    source="model",
                )

                if self.config.enable_verification:
                    verification_score = self.verify_step(thought_text, context)
                    step.confidence = verification_score
                    step.verification_status = verification_score >= self.config.confidence_threshold

                trace.add_step(step)
                context = self.format_trace_for_model(trace)

                if self.config.should_backtrack(step.confidence, self._backtrack_count):
                    if self.config.backtracking_mode.value != "disabled":
                        backtracked = self.backtrack(trace, len(trace.steps) - 1)
                        if backtracked:
                            self._backtrack_count += 1

            if self._is_conclusion(response):
                trace.solution = self._extract_solution(response)
                break

        trace.duration_ms = (time.time() - start_time) * 1000.0
        trace.pattern = self.detect_reasoning_pattern(trace)

        if self.config.enable_pruning:
            trace = self._prune_trace(trace, prompt)

        self.trace_history.append(trace)
        self._total_tokens_used += trace.tokens_used
        trace.total_confidence = self.estimate_confidence(trace)

        return trace

    def _build_step_prompt(
        self,
        context: str,
        step_num: int,
        trace: ReasoningTrace,
    ) -> str:
        """Build the prompt for generating the next reasoning step.

        Constructs a prompt that includes the original context and all
        previous reasoning steps to maintain coherence.

        Args:
            context: The current reasoning context.
            step_num: Current step number.
            trace: The reasoning trace so far.

        Returns:
            The constructed prompt string.
        """
        prefix = self.config.reasoning_prefix
        parts = [f"Context: {context}"]
        if step_num == 0:
            parts.insert(0, prefix)
        if trace.steps:
            recent = trace.steps[-3:]
            parts.append("\nPrevious reasoning:")
            for s in recent:
                parts.append(f"  Step {s.step_index + 1}: {s.thought}")
        parts.append(f"\nContinue reasoning (Step {step_num + 1}):")
        return "\n".join(parts)

    def _is_conclusion(self, response: str) -> bool:
        """Detect whether a response contains a conclusion.

        Looks for conclusion-indicating phrases in the model output.

        Args:
            response: The model's response text.

        Returns:
            True if the response appears to contain a conclusion.
        """
        conclusion_markers = [
            "therefore", "thus", "the answer is", "in conclusion",
            "so the answer is", "final answer", "the result is",
            "we conclude that", "hence the answer",
        ]
        response_lower = response.lower()
        for marker in conclusion_markers:
            if marker in response_lower:
                return True
        return False

    def _extract_solution(self, response: str) -> str:
        """Extract the solution/answer from a concluding response.

        Parses the model output to find the explicit answer.

        Args:
            response: The model's response containing a conclusion.

        Returns:
            Extracted solution string.
        """
        patterns = [
            r"(?:the answer is|therefore|thus|so the answer is)[:\s]*(.+?)(?:\.|$)",
            r"(?:final answer|result is)[:\s]*(.+?)(?:\.|$)",
            r"(?:we conclude that)[:\s]*(.+?)(?:\.|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        sentences = response.split(".")
        if sentences:
            return sentences[-1].strip()
        return response.strip()

    def _prune_trace(self, trace: ReasoningTrace, goal: str) -> ReasoningTrace:
        """Prune irrelevant steps from a reasoning trace.

        Removes steps that are identified as off-topic or redundant
        based on relevance scoring.

        Args:
            trace: The reasoning trace to prune.
            goal: The original goal or question.

        Returns:
            A new ReasoningTrace with irrelevant steps removed.
        """
        goal_words = set(goal.lower().split())
        goal_words.discard("")
        if not goal_words:
            return trace

        relevant_steps = []
        for step in trace.steps:
            step_words = set(step.thought.lower().split())
            if not step_words:
                relevant_steps.append(step)
                continue
            overlap = len(step_words & goal_words) / max(len(step_words), 1)
            if step.step_type == ReasoningStepType.CONCLUSION:
                relevant_steps.append(step)
            elif overlap >= 0.1 or step.is_high_confidence(0.6):
                relevant_steps.append(step)
            elif relevant_steps and step.dependencies:
                has_relevant_dep = any(
                    d < len(relevant_steps) for d in step.dependencies
                )
                if has_relevant_dep:
                    relevant_steps.append(step)

        pruned = ReasoningTrace(
            steps=relevant_steps,
            prompt=trace.prompt,
            solution=trace.solution,
            pattern=trace.pattern,
            metadata=copy.deepcopy(trace.metadata),
        )
        pruned.tokens_used = sum(s.token_count for s in pruned.steps)
        pruned._update_confidence()
        return pruned

    def extract_thoughts(self, response: str) -> List[str]:
        """Parse model output into a list of structured thought strings.

        Splits the response into individual reasoning steps based on
        sentence boundaries, step markers, and paragraph breaks.

        Args:
            response: Raw model output text.

        Returns:
            List of individual thought strings extracted from the response.
        """
        thoughts: List[str] = []
        if not response or not response.strip():
            return thoughts

        step_markers = [
            r"step\s+\d+[:.]",
            r"\d+[\.\)]\s",
            r"(?:first|second|third|next|then|finally|lastly)[,:]",
        ]
        split_pattern = r"(?:" + "|".join(step_markers) + r")"

        segments = re.split(split_pattern, response, flags=re.IGNORECASE)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            sentences = re.split(r'(?<=[.!?])\s+', segment)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) >= 5:
                    thoughts.append(sentence)

        if not thoughts:
            sentences = re.split(r'(?<=[.!?])\s+', response)
            thoughts = [s.strip() for s in sentences if len(s.strip()) >= 5]

        return thoughts[:5]

    def verify_step(self, thought: str, context: str) -> float:
        """Verify if a reasoning step is logically sound.

        Evaluates the coherence and relevance of a thought within the
        current reasoning context.

        Args:
            thought: The reasoning thought to verify.
            context: The current reasoning context.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if self.step_verifier is not None:
            return self.step_verifier(thought, context)

        score = 0.5
        thought_lower = thought.lower()

        logical_connectives = [
            "because", "therefore", "since", "thus", "implies",
            "follows", "means", "shows", "indicates", "proves",
        ]
        connective_count = sum(1 for c in logical_connectives if c in thought_lower)
        score += min(0.15, connective_count * 0.05)

        context_words = set(context.lower().split())
        thought_words = set(thought_lower.split())
        if context_words and thought_words:
            overlap = len(context_words & thought_words) / len(thought_words)
            score += min(0.15, overlap * 0.2)

        numerical_patterns = r'\d+[\.\,]?\d*\s*[+\-\*/x×÷=<>]'
        if re.search(numerical_patterns, thought):
            score += 0.1

        negation_patterns = ["not", "no", "never", "cannot", "incorrect"]
        has_negation = any(neg in thought_lower for neg in negation_patterns)
        if has_negation:
            score -= 0.05

        quantitative_words = ["approximately", "exactly", "roughly", "about", "precisely"]
        has_quantification = any(qw in thought_lower for qw in quantitative_words)
        if has_quantification:
            score += 0.05

        if thought.endswith("?"):
            score += 0.05

        length_score = min(0.1, len(thought) / 500.0)
        score += length_score

        return max(0.0, min(1.0, score))

    def backtrack(
        self,
        trace: ReasoningTrace,
        step_index: int,
    ) -> bool:
        """Backtrack to a previous reasoning step and attempt an alternative.

        When a reasoning step has low confidence or leads to an error,
        this method removes the problematic step and subsequent steps,
        then generates an alternative reasoning path.

        Args:
            trace: The current reasoning trace.
            step_index: Index of the step to backtrack to.

        Returns:
            True if backtracking was successful, False otherwise.
        """
        if step_index < 0 or step_index >= len(trace.steps):
            return False

        if self.config.backtracking_mode.value == "disabled":
            return False

        target_index = step_index
        if self.config.backtracking_mode.value == "immediate":
            target_index = max(0, step_index - 1)
        elif self.config.backtracking_mode.value == "smart":
            target_index = self._find_best_backtrack_point(trace, step_index)
        elif self.config.backtracking_mode.value == "full":
            target_index = self._find_optimal_backtrack_point(trace, step_index)

        if target_index >= step_index:
            return False

        removed_steps = trace.steps[target_index:]
        trace.steps = trace.steps[:target_index]
        trace.tokens_used = sum(s.token_count for s in trace.steps)
        trace._update_confidence()

        alternative = self.generate_alternative(trace, target_index)
        if alternative:
            trace.steps.append(alternative)
            trace.tokens_used += alternative.token_count
            trace._update_confidence()
            return True

        return False

    def _find_best_backtrack_point(
        self,
        trace: ReasoningTrace,
        current_index: int,
    ) -> int:
        """Find the best point to backtrack to using heuristic scoring.

        Evaluates each previous step to find the most appropriate point
        to resume reasoning from, preferring high-confidence steps that
        are close to the current position.

        Args:
            trace: The current reasoning trace.
            current_index: Index of the problematic step.

        Returns:
            Index of the recommended backtrack point.
        """
        best_index = 0
        best_score = -1.0

        for i in range(min(current_index, len(trace.steps))):
            step = trace.steps[i]
            proximity_score = 1.0 - (current_index - i) / max(current_index, 1)
            confidence_score = step.confidence
            type_bonus = 0.1 if step.step_type in (
                ReasoningStepType.THOUGHT,
                ReasoningStepType.OBSERVATION,
            ) else 0.0
            combined = proximity_score * 0.4 + confidence_score * 0.5 + type_bonus

            if combined > best_score:
                best_score = combined
                best_index = i

        return best_index

    def _find_optimal_backtrack_point(
        self,
        trace: ReasoningTrace,
        current_index: int,
    ) -> int:
        """Find the globally optimal backtrack point.

        Evaluates all possible backtrack points considering the full
        context of the reasoning chain.

        Args:
            trace: The current reasoning trace.
            current_index: Index of the problematic step.

        Returns:
            Index of the optimal backtrack point.
        """
        if current_index == 0:
            return 0

        scores = []
        for i in range(current_index):
            prefix_trace = trace.truncate(i + 1)
            prefix_confidence = self.estimate_confidence(prefix_trace)
            distance_penalty = (current_index - i) / current_index
            score = prefix_confidence - distance_penalty * 0.3
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores else 0

    def generate_alternative(
        self,
        trace: ReasoningTrace,
        step_index: int,
    ) -> Optional[ReasoningStep]:
        """Generate an alternative reasoning step at the given position.

        Creates a new reasoning step that takes a different approach from
        the original step at the specified index.

        Args:
            trace: The current reasoning trace (up to step_index).
            step_index: Index where the alternative should be inserted.

        Returns:
            A new ReasoningStep with alternative reasoning, or None.
        """
        context = self.format_trace_for_model(trace)
        alternative_prompt = (
            f"Given the reasoning so far:\n{context}\n\n"
            f"Provide an alternative approach for the next step that "
            f"differs from the previous attempt. Think differently."
        )

        response = self.model.generate(
            prompt=alternative_prompt,
            temperature=self.config.temperature * 1.5,
            max_tokens=self.config.max_tokens_per_step,
        )

        thoughts = self.extract_thoughts(response)
        if thoughts:
            thought_text = thoughts[0]
            confidence = self.verify_step(thought_text, context)
            return ReasoningStep(
                thought=thought_text,
                confidence=confidence,
                step_type=ReasoningStepType.CORRECTION,
                step_index=step_index,
                correction_of=step_index - 1,
                source="model",
                metadata={"backtrack": True, "original_index": step_index},
            )
        return None

    def evaluate_reasoning(
        self,
        trace: ReasoningTrace,
        solution: str,
    ) -> Dict[str, float]:
        """Score the quality of a reasoning chain against a known solution.

        Evaluates multiple dimensions of reasoning quality including
        coherence, correctness, efficiency, and completeness.

        Args:
            trace: The reasoning trace to evaluate.
            solution: The ground-truth solution for comparison.

        Returns:
            Dictionary of quality scores for each evaluation dimension.
        """
        scores: Dict[str, float] = {}

        solution_similarity = self._compute_solution_similarity(
            trace.solution, solution
        )
        scores["solution_accuracy"] = solution_similarity

        if trace.steps:
            step_confidences = [s.confidence for s in trace.steps]
            scores["avg_step_confidence"] = statistics.mean(step_confidences)
            scores["min_step_confidence"] = min(step_confidences)
            scores["confidence_std"] = statistics.stdev(step_confidences) if len(step_confidences) > 1 else 0.0
        else:
            scores["avg_step_confidence"] = 0.0
            scores["min_step_confidence"] = 0.0
            scores["confidence_std"] = 0.0

        scores["coherence"] = self._evaluate_coherence(trace)
        scores["completeness"] = self._evaluate_completeness(trace)
        scores["efficiency"] = self._evaluate_efficiency(trace)
        scores["relevance"] = self._evaluate_relevance(trace)

        total_weight = sum(w for _, w in [
            ("solution_accuracy", 0.30),
            ("avg_step_confidence", 0.20),
            ("coherence", 0.15),
            ("completeness", 0.15),
            ("efficiency", 0.10),
            ("relevance", 0.10),
        ])
        weighted_sum = (
            scores["solution_accuracy"] * 0.30
            + scores["avg_step_confidence"] * 0.20
            + scores["coherence"] * 0.15
            + scores["completeness"] * 0.15
            + scores["efficiency"] * 0.10
            + scores["relevance"] * 0.10
        )
        scores["overall"] = weighted_sum / total_weight

        return scores

    def _compute_solution_similarity(self, predicted: str, expected: str) -> float:
        """Compute similarity between predicted and expected solutions.

        Uses a combination of exact matching, word overlap, and character
        similarity to provide a robust similarity score.

        Args:
            predicted: The predicted solution.
            expected: The expected solution.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not predicted and not expected:
            return 1.0
        if not predicted or not expected:
            return 0.0

        pred_lower = predicted.lower().strip()
        exp_lower = expected.lower().strip()

        if pred_lower == exp_lower:
            return 1.0

        pred_words = set(pred_lower.split())
        exp_words = set(exp_lower.split())

        if pred_words and exp_words:
            jaccard = len(pred_words & exp_words) / len(pred_words | exp_words)
        else:
            jaccard = 0.0

        pred_numbers = set(re.findall(r'\d+\.?\d*', pred_lower))
        exp_numbers = set(re.findall(r'\d+\.?\d*', exp_lower))
        number_match = 0.0
        if pred_numbers and exp_numbers:
            number_match = len(pred_numbers & exp_numbers) / len(pred_numbers | exp_numbers)

        longer = max(len(pred_lower), len(exp_lower), 1)
        edit_dist = self._levenshtein_distance(pred_lower, exp_lower)
        char_similarity = 1.0 - (edit_dist / longer)

        combined = jaccard * 0.4 + number_match * 0.3 + char_similarity * 0.3
        return min(1.0, max(0.0, combined))

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute the Levenshtein edit distance between two strings.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            The edit distance as an integer.
        """
        if len(s1) < len(s2):
            return CoTReasoner._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (0 if c1 == c2 else 1)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _evaluate_coherence(self, trace: ReasoningTrace) -> float:
        """Evaluate the logical coherence of a reasoning chain.

        Measures how well consecutive reasoning steps flow together
        and build upon each other logically.

        Args:
            trace: The reasoning trace to evaluate.

        Returns:
            Coherence score between 0.0 and 1.0.
        """
        if len(trace.steps) < 2:
            return 0.8 if len(trace.steps) == 1 else 0.0

        coherence_scores = []
        for i in range(1, len(trace.steps)):
            prev_words = set(trace.steps[i - 1].thought.lower().split())
            curr_words = set(trace.steps[i].thought.lower().split())

            if not prev_words or not curr_words:
                continue

            overlap = len(prev_words & curr_words) / len(curr_words)

            prev_thought = trace.steps[i - 1].thought.lower()
            curr_thought = trace.steps[i].thought.lower()
            has_connector = any(
                connector in curr_thought
                for connector in ["therefore", "thus", "then", "so", "next", "this"]
            )

            connector_bonus = 0.15 if has_connector else 0.0
            step_coherence = min(1.0, overlap * 0.7 + connector_bonus + 0.1)
            coherence_scores.append(step_coherence)

        return statistics.mean(coherence_scores) if coherence_scores else 0.5

    def _evaluate_completeness(self, trace: ReasoningTrace) -> float:
        """Evaluate the completeness of a reasoning chain.

        Checks whether the trace has a clear structure with beginning,
        middle, and conclusion, and addresses the original question.

        Args:
            trace: The reasoning trace to evaluate.

        Returns:
            Completeness score between 0.0 and 1.0.
        """
        if not trace.steps:
            return 0.0

        has_start = any(
            s.step_type == ReasoningStepType.THOUGHT and s.step_index == 0
            for s in trace.steps
        )
        has_conclusion = any(
            s.step_type in (
                ReasoningStepType.CONCLUSION,
                ReasoningStepType.VERIFICATION,
            )
            for s in trace.steps
        )
        has_solution = bool(trace.solution.strip())

        step_types = set(s.step_type for s in trace.steps)
        diversity = min(1.0, len(step_types) / 4.0)

        start_score = 0.25 if has_start else 0.0
        conclusion_score = 0.3 if has_conclusion else 0.15
        solution_score = 0.25 if has_solution else 0.0
        diversity_score = diversity * 0.2

        return start_score + conclusion_score + solution_score + diversity_score

    def _evaluate_efficiency(self, trace: ReasoningTrace) -> float:
        """Evaluate the efficiency of a reasoning chain.

        Measures how many steps and tokens were used relative to
        the apparent complexity of the problem.

        Args:
            trace: The reasoning trace to evaluate.

        Returns:
            Efficiency score between 0.0 and 1.0.
        """
        if not trace.steps:
            return 0.5

        num_steps = len(trace.steps)
        ideal_steps = 5
        step_ratio = ideal_steps / max(num_steps, 1)
        step_efficiency = min(1.0, step_ratio)

        verified_count = sum(1 for s in trace.steps if s.verification_status)
        verification_ratio = verified_count / num_steps if num_steps > 0 else 0.0

        backtrack_count = sum(1 for s in trace.steps if s.correction_of is not None)
        backtrack_penalty = backtrack_count * 0.1

        avg_confidence = statistics.mean(s.confidence for s in trace.steps)

        efficiency = (
            step_efficiency * 0.3
            + verification_ratio * 0.2
            + avg_confidence * 0.3
            + 0.2
        ) - backtrack_penalty

        return max(0.0, min(1.0, efficiency))

    def _evaluate_relevance(self, trace: ReasoningTrace) -> float:
        """Evaluate the relevance of reasoning steps to the original question.

        Args:
            trace: The reasoning trace to evaluate.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        if not trace.prompt or not trace.steps:
            return 0.5

        prompt_words = set(trace.prompt.lower().split())
        prompt_words.discard("")
        if not prompt_words:
            return 0.5

        relevance_scores = []
        for step in trace.steps:
            step_words = set(step.thought.lower().split())
            if step_words:
                overlap = len(prompt_words & step_words) / len(step_words)
                relevance_scores.append(overlap)

        return statistics.mean(relevance_scores) if relevance_scores else 0.5

    def merge_traces(self, traces: List[ReasoningTrace]) -> ReasoningTrace:
        """Merge multiple reasoning traces into a single consolidated trace.

        Combines the best steps from each trace based on confidence scores,
        removing duplicates and maintaining logical coherence.

        Args:
            traces: List of reasoning traces to merge.

        Returns:
            A merged ReasoningTrace containing the best steps.
        """
        if not traces:
            return ReasoningTrace()
        if len(traces) == 1:
            return traces[0].clone()

        prompt = traces[0].prompt
        all_steps: List[Tuple[ReasoningStep, int]] = []
        for trace_idx, trace in enumerate(traces):
            for step in trace.steps:
                all_steps.append((step.clone(), trace_idx))

        seen_thoughts: Set[str] = set()
        unique_steps: List[ReasoningStep] = []
        for step, trace_idx in all_steps:
            normalized = step.thought.lower().strip()
            is_duplicate = False
            for seen in seen_thoughts:
                if self._levenshtein_distance(normalized, seen) < len(normalized) * 0.3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                seen_thoughts.add(normalized)
                unique_steps.append(step)

        unique_steps.sort(key=lambda s: s.confidence, reverse=True)

        deduped_steps: List[ReasoningStep] = []
        for step in unique_steps:
            is_redundant = False
            for existing in deduped_steps:
                sim = self._compute_solution_similarity(
                    step.thought, existing.thought
                )
                if sim > 0.8:
                    is_redundant = True
                    break
            if not is_redundant:
                deduped_steps.append(step)

        deduped_steps.sort(key=lambda s: s.step_index)

        solutions = [t.solution for t in traces if t.solution.strip()]
        best_solution = ""
        if solutions:
            solution_counts = Counter(solutions)
            best_solution = solution_counts.most_common(1)[0][0]

        merged = ReasoningTrace(
            steps=deduped_steps,
            prompt=prompt,
            solution=best_solution,
            metadata={"merged_from": len(traces)},
        )
        for i, step in enumerate(merged.steps):
            step.step_index = i
        merged.tokens_used = sum(s.token_count for s in merged.steps)
        merged._update_confidence()
        merged.pattern = self.detect_reasoning_pattern(merged)

        return merged

    def summarize_trace(self, trace: ReasoningTrace) -> ReasoningTrace:
        """Compress a long reasoning trace into a more concise version.

        Preserves the key reasoning steps while reducing redundancy
        and removing low-value intermediate steps.

        Args:
            trace: The reasoning trace to compress.

        Returns:
            A compressed ReasoningTrace preserving key steps.
        """
        if len(trace.steps) <= 3:
            return trace.clone()

        importance_scores = []
        for i, step in enumerate(trace.steps):
            importance = step.confidence

            if i == 0:
                importance += 0.3
            elif i == len(trace.steps) - 1:
                importance += 0.3

            if step.step_type == ReasoningStepType.CONCLUSION:
                importance += 0.2
            elif step.step_type == ReasoningStepType.VERIFICATION:
                importance += 0.15
            elif step.step_type == ReasoningStepType.ASSUMPTION:
                importance += 0.1

            if i > 0:
                prev = trace.steps[i - 1]
                sim = self._compute_solution_similarity(step.thought, prev.thought)
                if sim > 0.7:
                    importance -= 0.2

            importance_scores.append((i, importance))

        importance_scores.sort(key=lambda x: x[1], reverse=True)

        target_size = max(3, len(trace.steps) // 2)
        selected_indices = set()
        for idx, _ in importance_scores[:target_size]:
            selected_indices.add(idx)

        selected_indices.add(0)
        if trace.steps:
            selected_indices.add(len(trace.steps) - 1)

        sorted_indices = sorted(selected_indices)
        compressed_steps = [trace.steps[i].clone() for i in sorted_indices]
        for i, step in enumerate(compressed_steps):
            step.step_index = i

        compressed = ReasoningTrace(
            steps=compressed_steps,
            prompt=trace.prompt,
            solution=trace.solution,
            pattern=trace.pattern,
            metadata={
                **copy.deepcopy(trace.metadata),
                "compressed_from": len(trace.steps),
            },
        )
        compressed.tokens_used = sum(s.token_count for s in compressed.steps)
        compressed._update_confidence()
        return compressed

    def format_trace_for_model(self, trace: ReasoningTrace) -> str:
        """Format a reasoning trace for input to the language model.

        Converts the structured trace into a natural language format
        suitable for inclusion in model prompts.

        Args:
            trace: The reasoning trace to format.

        Returns:
            Formatted string representation of the trace.
        """
        if not trace.steps:
            return trace.prompt

        parts = [f"Question: {trace.prompt}"]
        parts.append("Reasoning:")
        for step in trace.steps:
            type_label = step.step_type.value.capitalize()
            parts.append(f"  [{type_label}] {step.thought}")
            if step.observation:
                parts.append(f"    Observation: {step.observation}")
        if trace.solution:
            parts.append(f"Answer: {trace.solution}")
        return "\n".join(parts)

    def detect_reasoning_pattern(self, trace: ReasoningTrace) -> ReasoningPattern:
        """Identify the reasoning pattern used in a trace.

        Analyzes the language and structure of reasoning steps to classify
        the overall reasoning approach.

        Args:
            trace: The reasoning trace to analyze.

        Returns:
            Detected ReasoningPattern enum value.
        """
        if not trace.steps:
            return ReasoningPattern.UNKNOWN

        all_text = " ".join(step.thought.lower() for step in trace.steps)
        all_words = set(all_text.split())

        pattern_scores: Dict[ReasoningPattern, float] = {
            ReasoningPattern.DEDUCTIVE: 0.0,
            ReasoningPattern.INDUCTIVE: 0.0,
            ReasoningPattern.ABDUCTIVE: 0.0,
            ReasoningPattern.ANALOGICAL: 0.0,
            ReasoningPattern.CAUSAL: 0.0,
            ReasoningPattern.CONDITIONAL: 0.0,
            ReasoningPattern.DIRECT: 0.0,
        }

        keyword_sets = {
            ReasoningPattern.DEDUCTIVE: self.DEDUCTIVE_KEYWORDS,
            ReasoningPattern.INDUCTIVE: self.INDUCTIVE_KEYWORDS,
            ReasoningPattern.ABDUCTIVE: self.ABDUCTIVE_KEYWORDS,
            ReasoningPattern.ANALOGICAL: self.ANALOGICAL_KEYWORDS,
            ReasoningPattern.CAUSAL: self.CAUSAL_KEYWORDS,
            ReasoningPattern.CONDITIONAL: self.CONDITIONAL_KEYWORDS,
        }

        for pattern, keywords in keyword_sets.items():
            matches = sum(1 for kw in keywords if kw in all_text)
            total_possible = len(keywords)
            pattern_scores[pattern] = matches / max(total_possible, 1)

        if all(s.step_index < 5 for s in trace.steps):
            pattern_scores[ReasoningPattern.DIRECT] += 0.3

        type_distribution = Counter(s.step_type for s in trace.steps)
        if type_distribution.get(ReasoningStepType.VERIFICATION, 0) > len(trace.steps) * 0.3:
            pattern_scores[ReasoningPattern.DEDUCTIVE] += 0.2

        sorted_patterns = sorted(
            pattern_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        top_pattern, top_score = sorted_patterns[0]
        second_pattern, second_score = sorted_patterns[1] if len(sorted_patterns) > 1 else (top_pattern, 0.0)

        if top_score < 0.05:
            return ReasoningPattern.DIRECT

        if top_score > 0 and second_score > 0:
            ratio = second_score / top_score if top_score > 0 else 0
            if ratio > 0.7:
                return ReasoningPattern.MIXED

        if top_score >= 0.1:
            return top_pattern

        return ReasoningPattern.DIRECT

    def prune_irrelevant_steps(
        self,
        trace: ReasoningTrace,
        goal: str,
    ) -> ReasoningTrace:
        """Remove off-topic or irrelevant steps from a reasoning trace.

        Analyzes each step's relevance to the original goal and removes
        steps that do not contribute meaningfully to the solution.

        Args:
            trace: The reasoning trace to prune.
            goal: The original goal or question.

        Returns:
            A new ReasoningTrace with irrelevant steps removed.
        """
        if not trace.steps:
            return trace.clone()

        goal_keywords = self._extract_keywords(goal)
        if not goal_keywords:
            return trace.clone()

        relevant_steps: List[ReasoningStep] = []
        for step in trace.steps:
            step_keywords = self._extract_keywords(step.thought)
            if not step_keywords:
                if relevant_steps:
                    prev_relevance = self._keyword_overlap(
                        self._extract_keywords(relevant_steps[-1].thought),
                        goal_keywords,
                    )
                    if prev_relevance > 0.2:
                        relevant_steps.append(step.clone())
                else:
                    relevant_steps.append(step.clone())
                continue

            overlap = self._keyword_overlap(step_keywords, goal_keywords)
            semantic_relevance = self._compute_solution_similarity(
                step.thought, goal
            )
            combined_relevance = overlap * 0.5 + semantic_relevance * 0.5

            if (step.step_type in (
                ReasoningStepType.CONCLUSION,
                ReasoningStepType.VERIFICATION,
            ) or combined_relevance >= 0.15 or step.is_high_confidence(0.7)):
                relevant_steps.append(step.clone())

        if not relevant_steps and trace.steps:
            relevant_steps = [trace.steps[0].clone(), trace.steps[-1].clone()]

        pruned = ReasoningTrace(
            steps=relevant_steps,
            prompt=trace.prompt,
            solution=trace.solution,
            pattern=trace.pattern,
            metadata={
                **copy.deepcopy(trace.metadata),
                "pruned_from": len(trace.steps),
            },
        )
        for i, step in enumerate(pruned.steps):
            step.step_index = i
        pruned.tokens_used = sum(s.token_count for s in pruned.steps)
        pruned._update_confidence()
        return pruned

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text.

        Removes common stop words and returns a set of unique keywords.

        Args:
            text: Input text.

        Returns:
            Set of keyword strings.
        """
        stop_words = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more", "most",
            "other", "some", "such", "no", "only", "own", "same", "than",
            "too", "very", "just", "because", "if", "when", "where", "how",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
            "she", "her", "it", "its", "they", "them", "their",
        })
        words = re.findall(r'\b[a-z]{2,}\b', text.lower())
        return set(w for w in words if w not in stop_words)

    def _keyword_overlap(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute the overlap ratio between two keyword sets.

        Args:
            set1: First keyword set.
            set2: Second keyword set.

        Returns:
            Jaccard-like overlap ratio between 0.0 and 1.0.
        """
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def estimate_confidence(self, trace: ReasoningTrace) -> float:
        """Estimate the overall confidence of a reasoning trace.

        Combines multiple confidence signals including per-step confidence,
        trace coherence, and solution certainty.

        Args:
            trace: The reasoning trace to estimate confidence for.

        Returns:
            Overall confidence score between 0.0 and 1.0.
        """
        if not trace.steps:
            return 0.0

        step_confidences = [s.confidence for s in trace.steps]
        mean_confidence = statistics.mean(step_confidences)

        if len(step_confidences) > 1:
            std_confidence = statistics.stdev(step_confidences)
            consistency_factor = max(0.0, 1.0 - std_confidence * 2)
        else:
            consistency_factor = 0.8

        verified_ratio = sum(
            1 for s in trace.steps if s.verification_status
        ) / len(trace.steps)

        backtrack_ratio = sum(
            1 for s in trace.steps if s.correction_of is not None
        ) / len(trace.steps)
        backtrack_penalty = backtrack_ratio * 0.2

        has_conclusion = any(
            s.step_type == ReasoningStepType.CONCLUSION for s in trace.steps
        )
        conclusion_bonus = 0.1 if has_conclusion else 0.0

        length_factor = min(1.0, len(trace.steps) / 5.0) * 0.1

        confidence = (
            mean_confidence * 0.4
            + consistency_factor * 0.2
            + verified_ratio * 0.15
            + conclusion_bonus
            + length_factor
            - backtrack_penalty
        )

        return max(0.0, min(1.0, confidence))

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics about the reasoner's usage.

        Returns:
            Dictionary containing usage statistics.
        """
        if not self.trace_history:
            return {
                "total_traces": 0,
                "total_steps": 0,
                "total_tokens": self._total_tokens_used,
                "avg_confidence": 0.0,
                "avg_steps_per_trace": 0.0,
                "backtrack_count": self._backtrack_count,
            }

        total_steps = sum(len(t) for t in self.trace_history)
        confidences = [t.total_confidence for t in self.trace_history]
        patterns = Counter(t.pattern for t in self.trace_history)

        return {
            "total_traces": len(self.trace_history),
            "total_steps": total_steps,
            "total_tokens": self._total_tokens_used,
            "avg_confidence": statistics.mean(confidences),
            "avg_steps_per_trace": total_steps / len(self.trace_history),
            "backtrack_count": self._backtrack_count,
            "pattern_distribution": {p.value: c for p, c in patterns.items()},
            "cache_size": len(self._cache),
        }

    def reset(self) -> None:
        """Reset the reasoner's state, clearing all history and caches."""
        self.trace_history.clear()
        self._backtrack_count = 0
        self._total_tokens_used = 0
        self._cache.clear()


# =============================================================================
# ZeroShotCoT - Zero-Shot Chain-of-Thought
# =============================================================================

class ZeroShotCoT:
    """Zero-shot chain-of-thought reasoning.

    Appends a reasoning prompt (e.g., "Let's think step by step") to the
    input question and generates a complete reasoning trace in a single
    model call. No examples are provided.

    This approach relies on the model's inherent ability to perform
    step-by-step reasoning when prompted appropriately.

    Attributes:
        reasoner: The underlying CoT reasoner for trace processing.
        reasoning_prompt: The prompt suffix that triggers reasoning.
    """

    DEFAULT_PROMPTS = [
        "Let's think step by step.",
        "Let's work through this step by step.",
        "I'll think about this carefully, step by step.",
        "Let me reason through this systematically.",
        "Approaching this step by step:",
    ]

    def __init__(
        self,
        config: Optional[Any] = None,
        model: Optional[ModelInterface] = None,
        reasoning_prompt: Optional[str] = None,
    ) -> None:
        """Initialize the zero-shot CoT reasoner.

        Args:
            config: ChainOfThoughtConfig instance.
            model: Language model interface.
            reasoning_prompt: Custom reasoning trigger prompt.
        """
        if config is None:
            from nexus.reasoning.reasoning_config import ChainOfThoughtConfig
            config = ChainOfThoughtConfig(strategy="zero_shot")
        self.config = config
        self.model = model or MockModel()
        self.reasoning_prompt = reasoning_prompt or self.DEFAULT_PROMPTS[0]
        self.reasoner = CoTReasoner(config=config, model=model)

    def reason(self, question: str) -> ReasoningTrace:
        """Perform zero-shot chain-of-thought reasoning on a question.

        Constructs a prompt with the reasoning trigger and generates
        a complete reasoning trace.

        Args:
            question: The question or problem to reason about.

        Returns:
            A ReasoningTrace with the complete reasoning process.
        """
        prompt = f"{question}\n\n{self.reasoning_prompt}"
        return self.reasoner.generate_reasoning_trace(
            model=self.model,
            prompt=prompt,
        )

    def reason_batch(
        self,
        questions: List[str],
    ) -> List[ReasoningTrace]:
        """Perform zero-shot reasoning on a batch of questions.

        Args:
            questions: List of questions to reason about.

        Returns:
            List of ReasoningTraces, one per question.
        """
        return [self.reason(q) for q in questions]

    def evaluate_prompts(
        self,
        questions: List[str],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """Evaluate different reasoning prompts on a set of questions.

        Tests each default prompt and returns accuracy metrics.

        Args:
            questions: List of test questions.
            ground_truths: List of correct answers.

        Returns:
            Dictionary mapping prompt to accuracy score.
        """
        results: Dict[str, float] = {}
        for prompt in self.DEFAULT_PROMPTS:
            original = self.reasoning_prompt
            self.reasoning_prompt = prompt
            correct = 0
            for question, truth in zip(questions, ground_truths):
                trace = self.reason(question)
                similarity = self.reasoner._compute_solution_similarity(
                    trace.solution, truth
                )
                if similarity >= 0.7:
                    correct += 1
            accuracy = correct / max(len(questions), 1)
            results[prompt] = accuracy
            self.reasoning_prompt = original
        return results


# =============================================================================
# FewShotCoT - Few-Shot Chain-of-Thought
# =============================================================================

class FewShotCoT:
    """Few-shot chain-of-thought reasoning with example-based guidance.

    Provides worked examples of reasoning processes to guide the model's
    chain-of-thought generation. Examples are selected based on similarity
    to the input question.

    Attributes:
        reasoner: The underlying CoT reasoner.
        examples: List of example (question, reasoning, answer) tuples.
        max_examples: Maximum number of examples to include.
        selection_method: How to select examples from the pool.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        model: Optional[ModelInterface] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        max_examples: int = 3,
        selection_method: str = "similarity",
    ) -> None:
        """Initialize the few-shot CoT reasoner.

        Args:
            config: ChainOfThoughtConfig instance.
            model: Language model interface.
            examples: List of example dicts with 'question', 'reasoning', 'answer' keys.
            max_examples: Maximum examples to include in the prompt.
            selection_method: Example selection strategy ('similarity', 'random', 'first').
        """
        if config is None:
            from nexus.reasoning.reasoning_config import ChainOfThoughtConfig
            config = ChainOfThoughtConfig(strategy="few_shot")
        self.config = config
        self.model = model or MockModel()
        self.max_examples = max_examples
        self.selection_method = selection_method
        self.examples = examples or self._default_examples()
        self.reasoner = CoTReasoner(config=config, model=model)

    def _default_examples(self) -> List[Dict[str, str]]:
        """Provide default few-shot reasoning examples.

        Returns:
            List of example dictionaries.
        """
        return [
            {
                "question": "If a train travels 120 miles in 2 hours, what is its average speed?",
                "reasoning": (
                    "Let me think about this step by step. "
                    "The train travels 120 miles in 2 hours. "
                    "Speed is calculated as distance divided by time. "
                    "So speed = 120 miles / 2 hours = 60 miles per hour."
                ),
                "answer": "60 miles per hour",
            },
            {
                "question": "A store has a 20% off sale. If an item originally costs $50, what is the sale price?",
                "reasoning": (
                    "Let's work through this step by step. "
                    "The original price is $50. "
                    "The discount is 20% of $50 = 0.20 × $50 = $10. "
                    "The sale price is $50 - $10 = $40."
                ),
                "answer": "$40",
            },
            {
                "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "reasoning": (
                    "Let me reason through this carefully. "
                    "If 5 machines take 5 minutes to make 5 widgets, "
                    "then each machine takes 5 minutes to make 1 widget. "
                    "So 100 machines each making 1 widget would also take 5 minutes."
                ),
                "answer": "5 minutes",
            },
            {
                "question": "What is the next number in the sequence: 2, 6, 18, 54, ...?",
                "reasoning": (
                    "Let me look at the pattern step by step. "
                    "2 × 3 = 6, 6 × 3 = 18, 18 × 3 = 54. "
                    "Each number is multiplied by 3 to get the next number. "
                    "So the next number is 54 × 3 = 162."
                ),
                "answer": "162",
            },
            {
                "question": "A farmer has 17 sheep. All but 9 run away. How many are left?",
                "reasoning": (
                    "Let me read this carefully. "
                    "'All but 9 run away' means 9 sheep did NOT run away. "
                    "So 9 sheep remain with the farmer."
                ),
                "answer": "9",
            },
        ]

    def add_example(self, example: Dict[str, str]) -> None:
        """Add a new reasoning example to the example pool.

        Args:
            example: Dictionary with 'question', 'reasoning', and 'answer' keys.
        """
        required_keys = {"question", "reasoning", "answer"}
        if not required_keys.issubset(example.keys()):
            missing = required_keys - set(example.keys())
            raise ValueError(f"Example missing required keys: {missing}")
        self.examples.append(example)

    def select_examples(self, question: str) -> List[Dict[str, str]]:
        """Select the most relevant examples for a given question.

        Args:
            question: The input question.

        Returns:
            List of selected example dictionaries.
        """
        if not self.examples:
            return []

        if self.selection_method == "first":
            return self.examples[:self.max_examples]

        if self.selection_method == "random":
            import random
            sampled = random.sample(
                self.examples,
                min(self.max_examples, len(self.examples)),
            )
            return sampled

        if self.selection_method == "similarity":
            scored_examples = []
            for example in self.examples:
                similarity = self.reasoner._compute_solution_similarity(
                    question, example["question"]
                )
                scored_examples.append((similarity, example))
            scored_examples.sort(key=lambda x: x[0], reverse=True)
            return [ex for _, ex in scored_examples[:self.max_examples]]

        return self.examples[:self.max_examples]

    def reason(self, question: str) -> ReasoningTrace:
        """Perform few-shot chain-of-thought reasoning.

        Selects relevant examples, constructs a prompt with examples,
        and generates a reasoning trace.

        Args:
            question: The question to reason about.

        Returns:
            A ReasoningTrace with the reasoning process.
        """
        selected = self.select_examples(question)
        prompt = self._build_few_shot_prompt(question, selected)
        return self.reasoner.generate_reasoning_trace(
            model=self.model,
            prompt=prompt,
        )

    def _build_few_shot_prompt(
        self,
        question: str,
        examples: List[Dict[str, str]],
    ) -> str:
        """Build the few-shot prompt with examples.

        Args:
            question: The target question.
            examples: Selected reasoning examples.

        Returns:
            The constructed prompt string.
        """
        parts = ["Solve the following problems step by step.\n"]
        for i, example in enumerate(examples, 1):
            parts.append(f"Example {i}:")
            parts.append(f"Q: {example['question']}")
            parts.append(f"Reasoning: {example['reasoning']}")
            parts.append(f"A: {example['answer']}")
            parts.append("")

        parts.append(f"Now solve this problem:")
        parts.append(f"Q: {question}")
        parts.append("Reasoning:")
        return "\n".join(parts)

    def reason_batch(self, questions: List[str]) -> List[ReasoningTrace]:
        """Perform few-shot reasoning on a batch of questions.

        Args:
            questions: List of questions to reason about.

        Returns:
            List of ReasoningTraces.
        """
        return [self.reason(q) for q in questions]


# =============================================================================
# AutoCoT - Automatic Chain-of-Thought
# =============================================================================

class AutoCoT:
    """Automatic chain-of-thought with clustered example construction.

    Automatically constructs reasoning examples by clustering questions
    and generating reasoning traces for representative questions from
    each cluster. This eliminates the need for manually written examples.

    The process involves:
    1. Cluster input questions by topic/similarity
    2. Select representative questions from each cluster
    3. Generate reasoning traces for representatives using zero-shot CoT
    4. Use the generated traces as few-shot examples

    Attributes:
        reasoner: The underlying CoT reasoner.
        model: Language model interface.
        num_clusters: Number of clusters for example grouping.
        examples_per_cluster: Number of examples to select per cluster.
        constructed_examples: Automatically constructed examples.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        model: Optional[ModelInterface] = None,
        num_clusters: int = 5,
        examples_per_cluster: int = 1,
    ) -> None:
        """Initialize the AutoCoT reasoner.

        Args:
            config: ChainOfThoughtConfig instance.
            model: Language model interface.
            num_clusters: Number of question clusters.
            examples_per_cluster: Examples to generate per cluster.
        """
        if config is None:
            from nexus.reasoning.reasoning_config import ChainOfThoughtConfig
            config = ChainOfThoughtConfig(strategy="auto")
        self.config = config
        self.model = model or MockModel()
        self.num_clusters = num_clusters
        self.examples_per_cluster = examples_per_cluster
        self.constructed_examples: List[Dict[str, str]] = []
        self.reasoner = CoTReasoner(config=config, model=model)

    def construct_examples(
        self,
        questions: List[str],
    ) -> List[Dict[str, str]]:
        """Automatically construct reasoning examples from a question pool.

        Clusters the questions, selects representatives, generates
        reasoning traces, and stores them as examples.

        Args:
            questions: Pool of questions to construct examples from.

        Returns:
            List of constructed example dictionaries.
        """
        if len(questions) < self.num_clusters:
            self.num_clusters = max(1, len(questions))

        clusters = self._cluster_questions(questions)
        self.constructed_examples = []

        for cluster in clusters:
            representatives = self._select_representatives(cluster)
            for rep_question in representatives[:self.examples_per_cluster]:
                example = self._generate_example(rep_question)
                if example:
                    self.constructed_examples.append(example)

        return self.constructed_examples

    def _cluster_questions(
        self,
        questions: List[str],
    ) -> List[List[str]]:
        """Cluster questions by topic similarity.

        Uses a simple keyword-based clustering approach that groups
        questions with similar vocabulary.

        Args:
            questions: List of questions to cluster.

        Returns:
            List of clusters, where each cluster is a list of questions.
        """
        if not questions:
            return []

        if len(questions) <= self.num_clusters:
            return [[q] for q in questions]

        question_vectors = []
        for q in questions:
            words = set(q.lower().split())
            question_vectors.append(words)

        n = len(questions)
        distance_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if question_vectors[i] and question_vectors[j]:
                    intersection = len(question_vectors[i] & question_vectors[j])
                    union = len(question_vectors[i] | question_vectors[j])
                    similarity = intersection / union if union > 0 else 0.0
                else:
                    similarity = 0.0
                distance = 1.0 - similarity
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        assignments = list(range(n))
        for _ in range(n - self.num_clusters):
            min_dist = float("inf")
            merge_i, merge_j = 0, 1
            for i in range(n):
                for j in range(i + 1, n):
                    if assignments[i] != assignments[j]:
                        if distance_matrix[i][j] < min_dist:
                            min_dist = distance_matrix[i][j]
                            merge_i, merge_j = i, j

            old_cluster = assignments[merge_j]
            for k in range(n):
                if assignments[k] == old_cluster:
                    assignments[k] = assignments[merge_i]

        clusters: Dict[int, List[str]] = {}
        for i, cluster_id in enumerate(assignments):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(questions[i])

        return list(clusters.values())

    def _select_representatives(
        self,
        cluster: List[str],
    ) -> List[str]:
        """Select representative questions from a cluster.

        Chooses questions that are most central to the cluster based
        on average similarity to other cluster members.

        Args:
            cluster: List of questions in the cluster.

        Returns:
            List of representative questions, ordered by centrality.
        """
        if not cluster:
            return []
        if len(cluster) <= self.examples_per_cluster:
            return list(cluster)

        centrality_scores = []
        for i, q in enumerate(cluster):
            total_sim = 0.0
            for j, other_q in enumerate(cluster):
                if i != j:
                    sim = self.reasoner._compute_solution_similarity(q, other_q)
                    total_sim += sim
            avg_sim = total_sim / (len(cluster) - 1) if len(cluster) > 1 else 0.5
            centrality_scores.append((avg_sim, q))

        centrality_scores.sort(key=lambda x: x[0], reverse=True)
        return [q for _, q in centrality_scores[:self.examples_per_cluster]]

    def _generate_example(self, question: str) -> Optional[Dict[str, str]]:
        """Generate a reasoning example for a representative question.

        Uses zero-shot CoT to generate the reasoning trace, then
        extracts the answer.

        Args:
            question: The representative question.

        Returns:
            Example dictionary, or None if generation failed.
        """
        zero_shot = ZeroShotCoT(config=self.config, model=self.model)
        trace = zero_shot.reason(question)

        if trace.steps and trace.solution:
            reasoning = " ".join(s.thought for s in trace.steps)
            return {
                "question": question,
                "reasoning": reasoning,
                "answer": trace.solution,
            }
        return None

    def reason(self, question: str) -> ReasoningTrace:
        """Perform automatic chain-of-thought reasoning.

        If examples haven't been constructed, uses zero-shot reasoning.
        Otherwise, uses constructed examples as few-shot guidance.

        Args:
            question: The question to reason about.

        Returns:
            A ReasoningTrace with the reasoning process.
        """
        if not self.constructed_examples:
            zero_shot = ZeroShotCoT(config=self.config, model=self.model)
            return zero_shot.reason(question)

        few_shot = FewShotCoT(
            config=self.config,
            model=self.model,
            examples=self.constructed_examples,
            max_examples=self.num_clusters,
            selection_method="similarity",
        )
        return few_shot.reason(question)


# =============================================================================
# StructuredCoT - Structured Output Chain-of-Thought
# =============================================================================

class StructuredCoT:
    """Structured chain-of-thought with explicit field-based output.

    Enforces a specific output structure with fields for Given (premises),
    Reasoning (step-by-step logic), and Conclusion (final answer).

    This format makes reasoning traces more interpretable and easier to
    verify, as each component of the reasoning process is explicitly labeled.

    Attributes:
        reasoner: The underlying CoT reasoner.
        model: Language model interface.
        fields: The required output fields for the structured format.
    """

    REQUIRED_FIELDS = ["given", "reasoning", "conclusion"]
    OPTIONAL_FIELDS = ["assumptions", "constraints", "verification"]

    def __init__(
        self,
        config: Optional[Any] = None,
        model: Optional[ModelInterface] = None,
        additional_fields: Optional[List[str]] = None,
    ) -> None:
        """Initialize the structured CoT reasoner.

        Args:
            config: ChainOfThoughtConfig instance.
            model: Language model interface.
            additional_fields: Additional fields to include in the output.
        """
        if config is None:
            from nexus.reasoning.reasoning_config import ChainOfThoughtConfig
            config = ChainOfThoughtConfig(strategy="structured")
        self.config = config
        self.model = model or MockModel()
        self.fields = self.REQUIRED_FIELDS + (additional_fields or [])
        self.reasoner = CoTReasoner(config=config, model=model)

    def reason(self, question: str) -> ReasoningTrace:
        """Perform structured chain-of-thought reasoning.

        Generates a response with explicit Given/Reasoning/Conclusion
        sections and parses them into a structured trace.

        Args:
            question: The question to reason about.

        Returns:
            A ReasoningTrace with structured reasoning steps.
        """
        prompt = self._build_structured_prompt(question)
        response = self.model.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_per_step * 3,
        )

        parsed = self._parse_structured_response(response)
        trace = ReasoningTrace(prompt=question)

        given_steps = parsed.get("given", [])
        for i, text in enumerate(given_steps):
            step = ReasoningStep(
                thought=text,
                confidence=0.8,
                step_type=ReasoningStepType.OBSERVATION,
                step_index=len(trace.steps),
                source="model",
                metadata={"field": "given"},
            )
            trace.add_step(step)

        reasoning_steps = parsed.get("reasoning", [])
        for i, text in enumerate(reasoning_steps):
            confidence = self.reasoner.verify_step(text, question)
            step = ReasoningStep(
                thought=text,
                confidence=confidence,
                step_type=ReasoningStepType.THOUGHT,
                step_index=len(trace.steps),
                source="model",
                metadata={"field": "reasoning"},
            )
            trace.add_step(step)

        assumptions = parsed.get("assumptions", [])
        for text in assumptions:
            step = ReasoningStep(
                thought=text,
                confidence=0.6,
                step_type=ReasoningStepType.ASSUMPTION,
                step_index=len(trace.steps),
                source="model",
                metadata={"field": "assumptions"},
            )
            trace.add_step(step)

        conclusions = parsed.get("conclusion", [])
        for text in conclusions:
            step = ReasoningStep(
                thought=text,
                confidence=0.9,
                step_type=ReasoningStepType.CONCLUSION,
                step_index=len(trace.steps),
                source="model",
                metadata={"field": "conclusion"},
            )
            trace.add_step(step)
            trace.solution = text

        verifications = parsed.get("verification", [])
        for text in verifications:
            step = ReasoningStep(
                thought=text,
                confidence=0.85,
                step_type=ReasoningStepType.VERIFICATION,
                step_index=len(trace.steps),
                source="model",
                metadata={"field": "verification"},
            )
            trace.add_step(step)

        trace.pattern = self.reasoner.detect_reasoning_pattern(trace)
        trace.total_confidence = self.reasoner.estimate_confidence(trace)
        return trace

    def _build_structured_prompt(self, question: str) -> str:
        """Build a prompt that enforces structured output format.

        Args:
            question: The input question.

        Returns:
            Structured prompt string.
        """
        field_descriptions = {
            "given": "List the given information and known facts.",
            "reasoning": "Show your step-by-step reasoning process.",
            "conclusion": "State your final answer or conclusion.",
            "assumptions": "List any assumptions you are making.",
            "constraints": "Note any constraints or limitations.",
            "verification": "Verify your answer through an alternative method.",
        }

        parts = [
            "Please answer the following question using this structured format:",
            "",
            f"Question: {question}",
            "",
        ]

        for field_name in self.fields:
            description = field_descriptions.get(field_name, f"Provide {field_name}.")
            parts.append(f"[{field_name.upper()}]")
            parts.append(f"{description}")
            parts.append("")

        return "\n".join(parts)

    def _parse_structured_response(self, response: str) -> Dict[str, List[str]]:
        """Parse a structured response into field-specific content.

        Extracts text between field markers and organizes it by field name.

        Args:
            response: The model's structured response.

        Returns:
            Dictionary mapping field names to lists of content strings.
        """
        parsed: Dict[str, List[str]] = {f: [] for f in self.fields}

        field_pattern = r'\[(' + '|'.join(re.escape(f.upper()) for f in self.fields) + r')\]'
        splits = re.split(field_pattern, response, flags=re.IGNORECASE)

        current_field = None
        for segment in splits:
            segment_stripped = segment.strip()
            if not segment_stripped:
                continue

            is_field_name = False
            for f in self.fields:
                if segment_stripped.upper() == f.upper():
                    current_field = f
                    is_field_name = True
                    break

            if is_field_name:
                continue

            if current_field is not None:
                sentences = re.split(r'(?<=[.!?])\s+', segment_stripped)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) >= 3:
                        parsed[current_field].append(sentence)

        return parsed

    def validate_trace(self, trace: ReasoningTrace) -> Dict[str, bool]:
        """Validate that a reasoning trace has all required structured fields.

        Args:
            trace: The reasoning trace to validate.

        Returns:
            Dictionary mapping field names to presence status.
        """
        field_present: Dict[str, bool] = {}
        trace_fields = set()
        for step in trace.steps:
            if "field" in step.metadata:
                trace_fields.add(step.metadata["field"])

        for field in self.fields:
            field_present[field] = field in trace_fields

        return field_present

    def reason_batch(self, questions: List[str]) -> List[ReasoningTrace]:
        """Perform structured reasoning on a batch of questions.

        Args:
            questions: List of questions.

        Returns:
            List of ReasoningTraces.
        """
        return [self.reason(q) for q in questions]
