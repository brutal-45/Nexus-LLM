"""
Verification Module
=====================

Implements multiple answer verification systems for the Nexus LLM framework.
Verification ensures reasoning outputs are correct, consistent, and reliable
before being presented as final answers.

Verification Methods:
- SelfVerification: Model verifies its own answer by re-solving independently
- CrossVerification: Multiple independent solutions compared for consensus
- BackwardVerification: Verifies answer by working backward from conclusion
- FormalVerification: Formal logic verification for mathematical/code reasoning
- ExecutionVerification: Executes code or calculations to verify correctness
- ConsensusVerifier: Aggregates multiple verification methods

Each verifier produces a VerificationResult with confidence scores and
detailed reasoning about the verification process.
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
import subprocess
import tempfile
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Structures
# =============================================================================

class VerificationMethod(Enum):
    """Types of verification methods available.

    Attributes:
        SELF: Model re-solves and compares with its own answer.
        CROSS: Multiple independent solutions compared.
        BACKWARD: Works backward from conclusion to premises.
        FORMAL: Formal logic/mathematical verification.
        EXECUTION: Code execution or calculation verification.
        CONSENSUS: Aggregated verification from multiple methods.
    """
    SELF = "self_verification"
    CROSS = "cross_verification"
    BACKWARD = "backward_verification"
    FORMAL = "formal_verification"
    EXECUTION = "execution_verification"
    CONSENSUS = "consensus_verification"


@dataclass
class VerificationResult:
    """Result of a single verification attempt.

    Attributes:
        is_correct: Whether the answer was verified as correct.
        confidence: Confidence in the verification result (0.0 to 1.0).
        method: The verification method used.
        answer: The answer that was verified.
        expected: The expected/correct answer (if known).
        details: Detailed reasoning about the verification.
        verification_time_ms: Time taken for verification in milliseconds.
        tokens_used: Tokens consumed by verification.
        error: Error message if verification encountered an error.
        metadata: Additional metadata.
    """
    is_correct: bool = False
    confidence: float = 0.0
    method: str = ""
    answer: str = ""
    expected: str = ""
    details: str = ""
    verification_time_ms: float = 0.0
    tokens_used: int = 0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "method": self.method,
            "answer": self.answer[:200],
            "expected": self.expected[:200] if self.expected else "",
            "details": self.details[:500],
            "verification_time_ms": self.verification_time_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
        }


@dataclass
class VerificationReport:
    """Aggregated verification results from multiple methods.

    Combines results from different verifiers, identifies disagreements,
    and provides an overall assessment.

    Attributes:
        results: Individual verification results.
        overall_correct: Whether the answer is considered correct overall.
        overall_confidence: Aggregated confidence score.
        agreement_ratio: Ratio of verifiers that agree.
        disagreements: List of method pairs that disagree.
        recommendation: Final recommendation (accept/reject/review).
        answer: The answer being verified.
    """
    results: List[VerificationResult] = field(default_factory=list)
    overall_correct: bool = False
    overall_confidence: float = 0.0
    agreement_ratio: float = 0.0
    disagreements: List[str] = field(default_factory=list)
    recommendation: str = "review"
    answer: str = ""

    def add_result(self, result: VerificationResult) -> None:
        """Add a verification result to the report.

        Args:
            result: The verification result to add.
        """
        self.results.append(result)
        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate aggregate metrics from all results."""
        if not self.results:
            self.overall_correct = False
            self.overall_confidence = 0.0
            self.agreement_ratio = 0.0
            self.recommendation = "review"
            return

        correct_count = sum(1 for r in self.results if r.is_correct)
        total = len(self.results)
        self.agreement_ratio = correct_count / total
        self.overall_confidence = statistics.mean(
            r.confidence for r in self.results
        )

        confidences = [r.confidence for r in self.results if r.is_correct]
        if confidences:
            weighted_conf = statistics.mean(confidences)
            self.overall_confidence = weighted_conf * 0.6 + self.agreement_ratio * 0.4

        self.overall_correct = self.agreement_ratio >= 0.5

        for i, r1 in enumerate(self.results):
            for j, r2 in enumerate(self.results):
                if i < j and r1.is_correct != r2.is_correct:
                    self.disagreements.append(
                        f"{r1.method} vs {r2.method}"
                    )

        if self.agreement_ratio >= 0.8:
            self.recommendation = "accept"
        elif self.agreement_ratio >= 0.5:
            self.recommendation = "review"
        else:
            self.recommendation = "reject"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "overall_correct": self.overall_correct,
            "overall_confidence": self.overall_confidence,
            "agreement_ratio": self.agreement_ratio,
            "recommendation": self.recommendation,
            "answer": self.answer[:200],
            "num_verifiers": len(self.results),
            "disagreements": self.disagreements,
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Base Verifier
# =============================================================================

class Verifier(ABC):
    """Abstract base class for all verification methods.

    Provides common interface and utilities for answer verification.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        """Initialize the verifier.

        Args:
            model: Language model interface.
        """
        self.model = model or self._default_model()
        self._verification_count: int = 0
        self._total_time_ms: float = 0.0

    def _default_model(self) -> Any:
        """Create a default mock model.

        Returns:
            A mock model interface.
        """
        from nexus.reasoning.chain_of_thought import MockModel
        return MockModel()

    @abstractmethod
    def verify(
        self,
        answer: str,
        question: str,
        context: str = "",
        expected: str = "",
    ) -> VerificationResult:
        """Verify an answer to a question.

        Args:
            answer: The answer to verify.
            question: The original question.
            context: Additional context.
            expected: Known correct answer (optional).

        Returns:
            A VerificationResult with the outcome.
        """
        raise NotImplementedError

    def normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison.

        Removes whitespace, normalizes case, strips articles, etc.

        Args:
            answer: The answer to normalize.

        Returns:
            Normalized answer string.
        """
        normalized = answer.strip().lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[.,;:!?]+$', '', normalized)
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        normalized = normalized.strip()
        return normalized

    def extract_numeric_answer(self, answer: str) -> Optional[float]:
        """Extract a numeric value from an answer string.

        Args:
            answer: The answer string.

        Returns:
            Extracted float value, or None if no number found.
        """
        patterns = [
            r'(?:answer|result|is|equals?|=)\s*[-+]?\d+\.?\d*',
            r'[-+]?\d+\.?\d*\s*(?:dollars?|percent?%?|degrees?|units?|meters?|seconds?|feet?|miles?|kg|lbs?)?',
            r'[-+]?\d+\.?\d*',
        ]
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                try:
                    value_str = match.group(0)
                    value_str = re.sub(r'[^\d.\-+]', '', value_str)
                    return float(value_str)
                except (ValueError, IndexError):
                    continue
        return None

    def compute_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Compute similarity between two answers.

        Uses a combination of text similarity and numeric comparison.

        Args:
            answer1: First answer.
            answer2: Second answer.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        norm1 = self.normalize_answer(answer1)
        norm2 = self.normalize_answer(answer2)

        if norm1 == norm2:
            return 1.0

        num1 = self.extract_numeric_answer(answer1)
        num2 = self.extract_numeric_answer(answer2)

        if num1 is not None and num2 is not None:
            if num1 == num2:
                return 1.0
            if num2 > 0:
                relative_error = abs(num1 - num2) / abs(num2)
                return max(0.0, 1.0 - relative_error)
            return 0.5 if abs(num1 - num2) < 1e-6 else 0.0

        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if words1 and words2:
            jaccard = len(words1 & words2) / len(words1 | words2)
            return jaccard

        return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics.

        Returns:
            Dictionary of usage statistics.
        """
        return {
            "verification_count": self._verification_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": (
                self._total_time_ms / self._verification_count
                if self._verification_count > 0 else 0.0
            ),
        }


# =============================================================================
# Self-Verification
# =============================================================================

class SelfVerification(Verifier):
    """Model verifies its own answer by re-solving the problem.

    The model is asked to solve the same question again independently,
    and the two answers are compared for consistency.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        num_resolutions: int = 3,
    ) -> None:
        """Initialize self-verification.

        Args:
            model: Language model interface.
            num_resolutions: Number of independent re-solutions.
        """
        super().__init__(model=model)
        self.num_resolutions = num_resolutions

    def verify(
        self,
        answer: str,
        question: str,
        context: str = "",
        expected: str = "",
    ) -> VerificationResult:
        """Verify by having the model re-solve the problem.

        Args:
            answer: The answer to verify.
            question: The original question.
            context: Additional context.
            expected: Known correct answer.

        Returns:
            VerificationResult from self-verification.
        """
        start_time = time.time()
        self._verification_count += 1

        resolutions: List[str] = []
        for i in range(self.num_resolutions):
            prompt = self._build_resolution_prompt(question, context)
            response = self.model.generate(
                prompt=prompt,
                temperature=0.2 + i * 0.1,
                max_tokens=500,
            )
            resolved = self._extract_answer(response)
            resolutions.append(resolved)

        similarities = [
            self.compute_answer_similarity(answer, res)
            for res in resolutions
        ]
        avg_similarity = statistics.mean(similarities) if similarities else 0.0

        max_similarity = max(similarities) if similarities else 0.0
        agreement_count = sum(1 for s in similarities if s >= 0.8)

        if expected:
            expected_similarity = self.compute_answer_similarity(answer, expected)
            is_correct = expected_similarity >= 0.8
            confidence = expected_similarity * 0.6 + avg_similarity * 0.4
        else:
            is_correct = avg_similarity >= 0.7 and agreement_count >= self.num_resolutions // 2
            confidence = avg_similarity

        verification_time = (time.time() - start_time) * 1000.0
        self._total_time_ms += verification_time

        details_parts = [
            f"Re-solved {self.num_resolutions} times.",
            f"Similarities: {[f'{s:.2f}' for s in similarities]}",
            f"Average similarity: {avg_similarity:.2f}",
            f"Agreement count: {agreement_count}/{self.num_resolutions}",
        ]
        if expected:
            details_parts.append(f"Expected answer similarity: {expected_similarity:.2f}")

        return VerificationResult(
            is_correct=is_correct,
            confidence=min(1.0, max(0.0, confidence)),
            method=VerificationMethod.SELF.value,
            answer=answer,
            expected=expected,
            details=". ".join(details_parts),
            verification_time_ms=verification_time,
            tokens_used=sum(len(r) // 4 for r in resolutions),
        )

    def _build_resolution_prompt(self, question: str, context: str) -> str:
        """Build prompt for independent re-resolution.

        Args:
            question: The original question.
            context: Additional context.

        Returns:
            Resolution prompt.
        """
        parts = [f"Solve this problem independently and provide only the final answer."]
        if context:
            parts.append(f"Context: {context[:200]}")
        parts.append(f"Problem: {question}")
        parts.append("Provide a clear, concise answer:")
        return "\n".join(parts)

    def _extract_answer(self, response: str) -> str:
        """Extract the answer from a model response.

        Args:
            response: Raw model output.

        Returns:
            Extracted answer string.
        """
        answer_patterns = [
            r"(?:the answer is|answer:|therefore|thus|result:)[:\s]*([^.\n]+)",
            r"(?:final answer|final result)[:\s]*([^.\n]+)",
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        sentences = response.strip().split(".")
        if sentences:
            return sentences[-1].strip()
        return response.strip()[:100]


# =============================================================================
# Cross-Verification
# =============================================================================

class CrossVerification(Verifier):
    """Multiple independent solutions compared for consensus.

    Generates multiple independent solutions using different prompts
    and temperatures, then checks for agreement.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        num_solutions: int = 5,
        consensus_threshold: float = 0.6,
    ) -> None:
        """Initialize cross-verification.

        Args:
            model: Language model interface.
            num_solutions: Number of independent solutions to generate.
            consensus_threshold: Minimum agreement ratio for acceptance.
        """
        super().__init__(model=model)
        self.num_solutions = num_solutions
        self.consensus_threshold = consensus_threshold

    def verify(
        self,
        answer: str,
        question: str,
        context: str = "",
        expected: str = "",
    ) -> VerificationResult:
        """Verify by generating multiple independent solutions.

        Args:
            answer: The answer to verify.
            question: The original question.
            context: Additional context.
            expected: Known correct answer.

        Returns:
            VerificationResult from cross-verification.
        """
        start_time = time.time()
        self._verification_count += 1

        solutions = self._generate_solutions(question, context)
        all_answers = [answer] + solutions

        agreement_groups = self._find_agreement_groups(all_answers)
        largest_group = max(agreement_groups, key=len) if agreement_groups else [answer]
        group_size = len(largest_group)

        answer_in_largest = any(
            self.compute_answer_similarity(answer, ga) >= 0.8
            for ga in largest_group
        )
        consensus_ratio = group_size / len(all_answers)

        if expected:
            expected_match = any(
                self.compute_answer_similarity(expected, ga) >= 0.8
                for ga in largest_group
            )
            is_correct = expected_match and consensus_ratio >= self.consensus_threshold
            confidence = consensus_ratio
        else:
            is_correct = (
                answer_in_largest
                and consensus_ratio >= self.consensus_threshold
            )
            confidence = consensus_ratio

        verification_time = (time.time() - start_time) * 1000.0
        self._total_time_ms += verification_time

        details_parts = [
            f"Generated {len(solutions)} independent solutions.",
            f"Found {len(agreement_groups)} agreement groups.",
            f"Largest group size: {group_size}/{len(all_answers)}",
            f"Consensus ratio: {consensus_ratio:.2f}",
            f"Answer in largest group: {answer_in_largest}",
        ]

        return VerificationResult(
            is_correct=is_correct,
            confidence=min(1.0, max(0.0, confidence)),
            method=VerificationMethod.CROSS.value,
            answer=answer,
            expected=expected,
            details=". ".join(details_parts),
            verification_time_ms=verification_time,
            tokens_used=sum(len(s) // 4 for s in solutions),
        )

    def _generate_solutions(self, question: str, context: str) -> List[str]:
        """Generate multiple independent solutions.

        Uses diverse prompts and temperatures.

        Args:
            question: The original question.
            context: Additional context.

        Returns:
            List of solution strings.
        """
        solutions: List[str] = []
        prompt_variations = [
            f"Solve this problem: {question}",
            f"What is the answer to: {question} Show your work.",
            f"Please provide the solution to: {question}",
            f"Determine the correct answer: {question}",
            f"Think carefully and answer: {question}",
        ]

        temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]

        for i in range(self.num_solutions):
            temp = temperatures[i % len(temperatures)]
            prompt = prompt_variations[i % len(prompt_variations)]
            if context:
                prompt = f"Context: {context[:200]}\n{prompt}"

            response = self.model.generate(
                prompt=prompt,
                temperature=temp,
                max_tokens=500,
            )
            extracted = self._extract_answer_from_response(response)
            solutions.append(extracted)

        return solutions

    def _extract_answer_from_response(self, response: str) -> str:
        """Extract the answer from a response.

        Args:
            response: Raw model output.

        Returns:
            Extracted answer string.
        """
        patterns = [
            r"(?:the answer is|answer:|result:|therefore|thus)[:\s]*([^.\n]+)",
            r"(?:final answer|final result)[:\s]*([^.\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        sentences = response.strip().split(".")
        return sentences[-1].strip()[:100] if sentences else response.strip()[:100]

    def _find_agreement_groups(self, answers: List[str]) -> List[List[str]]:
        """Find groups of answers that agree with each other.

        Args:
            answers: List of answer strings.

        Returns:
            List of agreement groups.
        """
        if not answers:
            return []

        visited: List[bool] = [False] * len(answers)
        groups: List[List[str]] = []

        for i, ans in enumerate(answers):
            if visited[i]:
                continue
            group = [ans]
            visited[i] = True
            for j, other in enumerate(answers):
                if not visited[j]:
                    sim = self.compute_answer_similarity(ans, other)
                    if sim >= 0.8:
                        group.append(other)
                        visited[j] = True
            groups.append(group)

        groups.sort(key=len, reverse=True)
        return groups


# =============================================================================
# Backward Verification
# =============================================================================

class BackwardVerification(Verifier):
    """Verifies answer by working backward from the conclusion.

    Takes the proposed answer and checks if it logically leads back
    to the original problem conditions.
    """

    def verify(
        self,
        answer: str,
        question: str,
        context: str = "",
        expected: str = "",
    ) -> VerificationResult:
        """Verify by working backward from the answer.

        Args:
            answer: The answer to verify.
            question: The original question.
            context: Additional context.
            expected: Known correct answer.

        Returns:
            VerificationResult from backward verification.
        """
        start_time = time.time()
        self._verification_count += 1

        prompt = self._build_backward_prompt(question, answer, context)
        response = self.model.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=500,
        )

        verification_score = self._parse_verification_response(response)
        is_correct = verification_score >= 0.7

        if expected:
            expected_sim = self.compute_answer_similarity(answer, expected)
            combined_conf = verification_score * 0.6 + expected_sim * 0.4
            is_correct = combined_conf >= 0.7 or expected_sim >= 0.9
        else:
            combined_conf = verification_score

        verification_time = (time.time() - start_time) * 1000.0
        self._total_time_ms += verification_time

        return VerificationResult(
            is_correct=is_correct,
            confidence=min(1.0, max(0.0, combined_conf)),
            method=VerificationMethod.BACKWARD.value,
            answer=answer,
            expected=expected,
            details=f"Backward verification score: {verification_score:.2f}. Response: {response[:200]}",
            verification_time_ms=verification_time,
            tokens_used=len(response) // 4,
        )

    def _build_backward_prompt(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> str:
        """Build prompt for backward verification.

        Args:
            question: The original question.
            answer: The proposed answer.
            context: Additional context.

        Returns:
            Backward verification prompt.
        """
        return (
            f"Given the answer '{answer}' to the question '{question}', "
            f"verify this answer by working backward. Start from the answer "
            f"and show that it logically leads back to the problem conditions. "
            f"Rate your confidence in the answer's correctness from 0 to 1. "
            f"If context is available: {context[:200]}\n\n"
            f"Provide your verification and final confidence score:"
        )

    def _parse_verification_response(self, response: str) -> float:
        """Parse the confidence score from a backward verification response.

        Args:
            response: Raw model output.

        Returns:
            Parsed confidence score between 0.0 and 1.0.
        """
        score_patterns = [
            r'(?:confidence|certainty|score|rating)[:\s]*([0-9]*\.?[0-9]+)',
            r'(?:\b)([01]?\.\d+)(?:\s*/\s*1)?',
            r'(?:\b)(\d+)(?:\s*%|percent)',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if "%" in match.group(0):
                        value = value / 100.0
                    return max(0.0, min(1.0, value))
                except (ValueError, IndexError):
                    continue

        positive_words = ["correct", "verified", "confirmed", "valid", "right", "true"]
        negative_words = ["incorrect", "wrong", "invalid", "false", "error", "impossible"]
        response_lower = response.lower()
        pos_count = sum(1 for w in positive_words if w in response_lower)
        neg_count = sum(1 for w in negative_words if w in response_lower)
        total = pos_count + neg_count
        if total > 0:
            return pos_count / total
        return 0.5


# =============================================================================
# Formal Verification
# =============================================================================

class FormalVerification(Verifier):
    """Formal logic verification for mathematical and code reasoning.

    Attempts to verify answers using formal methods including:
    - Mathematical equation verification
    - Logical consistency checking
    - Unit consistency
    """

    def verify(
        self,
        answer: str,
        question: str,
        context: str = "",
        expected: str = "",
    ) -> VerificationResult:
        """Verify using formal methods.

        Args:
            answer: The answer to verify.
            question: The original question.
            context: Additional context.
            expected: Known correct answer.

        Returns:
            VerificationResult from formal verification.
        """
        start_time = time.time()
        self._verification_count += 1

        scores: List[float] = []
        details_parts: List[str] = []

        numeric_score = self._verify_numeric(answer, question)
        scores.append(numeric_score)
        details_parts.append(f"Numeric verification: {numeric_score:.2f}")

        logic_score = self._verify_logic(answer, question, context)
        scores.append(logic_score)
        details_parts.append(f"Logic verification: {logic_score:.2f}")

        unit_score = self._verify_units(answer, question)
        scores.append(unit_score)
        details_parts.append(f"Unit verification: {unit_score:.2f}")

        if expected:
            expected_score = self.compute_answer_similarity(answer, expected)
            scores.append(expected_score)
            details_parts.append(f"Expected match: {expected_score:.2f}")

        avg_score = statistics.mean(scores)
        is_correct = avg_score >= 0.7

        verification_time = (time.time() - start_time) * 1000.0
        self._total_time_ms += verification_time

        return VerificationResult(
            is_correct=is_correct,
            confidence=min(1.0, max(0.0, avg_score)),
            method=VerificationMethod.FORMAL.value,
            answer=answer,
            expected=expected,
            details=". ".join(details_parts),
            verification_time_ms=verification_time,
            tokens_used=0,
        )

    def _verify_numeric(self, answer: str, question: str) -> float:
        """Verify numerical answer using calculation.

        Attempts to extract and verify mathematical operations.

        Args:
            answer: The proposed answer.
            question: The question.

        Returns:
            Verification score between 0.0 and 1.0.
        """
        question_numbers = re.findall(r'\d+\.?\d*', question)
        answer_numbers = re.findall(r'\d+\.?\d*', answer)

        if not question_numbers or not answer_numbers:
            return 0.5

        operations = re.findall(
            r'(\d+\.?\d*)\s*([+\-*/x×÷])\s*(\d+\.?\d*)',
            question,
        )
        if operations:
            correct_results = []
            for op1, op, op2 in operations:
                try:
                    a, b = float(op1), float(op2)
                    if op in ('+',):
                        correct_results.append(a + b)
                    elif op in ('-', '−'):
                        correct_results.append(a - b)
                    elif op in ('*', 'x', '×'):
                        correct_results.append(a * b)
                    elif op in ('/', '÷'):
                        if b != 0:
                            correct_results.append(a / b)
                except (ValueError, ZeroDivisionError):
                    continue

            answer_num = self.extract_numeric_answer(answer)
            if correct_results and answer_num is not None:
                for correct in correct_results:
                    if abs(correct - answer_num) < 1e-6:
                        return 1.0
                    relative_error = abs(correct - answer_num) / max(abs(correct), 1e-6)
                    if relative_error < 0.05:
                        return 0.9

        return 0.5

    def _verify_logic(self, answer: str, question: str, context: str) -> float:
        """Verify logical consistency of the answer.

        Args:
            answer: The proposed answer.
            question: The question.
            context: Additional context.

        Returns:
            Verification score.
        """
        question_lower = question.lower()
        answer_lower = answer.lower()

        negative_questions = ["not", "no", "never", "false", "incorrect", "impossible"]
        is_negative_q = any(nq in question_lower for nq in negative_questions)

        negative_answers = ["no", "not", "never", "false", "none", "zero", "impossible"]
        is_negative_a = any(na in answer_lower for na in negative_answers)

        if is_negative_q and not is_negative_a:
            return 0.3
        if not is_negative_q and is_negative_a:
            if "cannot" not in question_lower and "impossible" not in question_lower:
                return 0.4

        logical_connectives = ["therefore", "because", "since", "thus", "implies"]
        has_reasoning = any(lc in answer_lower for lc in logical_connectives)
        if has_reasoning and context:
            return 0.7

        return 0.5

    def _verify_units(self, answer: str, question: str) -> float:
        """Verify that units in the answer match the question.

        Args:
            answer: The proposed answer.
            question: The question.

        Returns:
            Unit consistency score.
        """
        question_units = set(re.findall(
            r'\b(dollars?|USD|percent?%?|degrees?|meters?|cm|km|seconds?|minutes?|hours?|feet?|miles?|kg|lbs?|pounds?|inches?|gallons?|liters?)\b',
            question, re.IGNORECASE,
        ))
        answer_units = set(re.findall(
            r'\b(dollars?|USD|percent?%?|degrees?|meters?|cm|km|seconds?|minutes?|hours?|feet?|miles?|kg|lbs?|pounds?|inches?|gallons?|liters?)\b',
            answer, re.IGNORECASE,
        ))

        if not question_units:
            return 0.8

        if question_units and answer_units:
            overlap = question_units & answer_units
            if overlap:
                return 0.9
            return 0.3

        if question_units and not answer_units:
            return 0.4

        return 0.8


# =============================================================================
# Execution Verification
# =============================================================================

class ExecutionVerification(Verifier):
    """Verifies answers by executing code or calculations.

    Extracts code from answers and executes it to verify correctness,
    or performs direct arithmetic verification.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        timeout: float = 5.0,
        allowed_modules: Optional[Set[str]] = None,
    ) -> None:
        """Initialize execution verification.

        Args:
            model: Language model interface.
            timeout: Execution timeout in seconds.
            allowed_modules: Set of allowed Python modules for execution.
        """
        super().__init__(model=model)
        self.timeout = timeout
        self.allowed_modules = allowed_modules or {
            "math", "statistics", "json", "re", "collections",
            "itertools", "functools", "operator", "decimal", "fractions",
        }

    def verify(
        self,
        answer: str,
        question: str,
        context: str = "",
        expected: str = "",
    ) -> VerificationResult:
        """Verify by executing code or calculations.

        Args:
            answer: The answer to verify.
            question: The original question.
            context: Additional context.
            expected: Known correct answer.

        Returns:
            VerificationResult from execution verification.
        """
        start_time = time.time()
        self._verification_count += 1

        code_blocks = self._extract_code(answer)
        if code_blocks:
            code_result = self._execute_code_blocks(code_blocks)
            if code_result is not None:
                verification_time = (time.time() - start_time) * 1000.0
                self._total_time_ms += verification_time

                if expected:
                    match = self.compute_answer_similarity(str(code_result), expected)
                    return VerificationResult(
                        is_correct=match >= 0.8,
                        confidence=match,
                        method=VerificationMethod.EXECUTION.value,
                        answer=answer,
                        expected=expected,
                        details=f"Code execution result: {code_result}. Expected match: {match:.2f}",
                        verification_time_ms=verification_time,
                    )
                else:
                    return VerificationResult(
                        is_correct=True,
                        confidence=0.7,
                        method=VerificationMethod.EXECUTION.value,
                        answer=answer,
                        details=f"Code executed successfully. Result: {code_result}",
                        verification_time_ms=verification_time,
                    )

        calc_result = self._verify_calculation(answer, question)
        verification_time = (time.time() - start_time) * 1000.0
        self._total_time_ms += verification_time

        if expected:
            expected_match = self.compute_answer_similarity(answer, expected)
            combined = calc_result * 0.6 + expected_match * 0.4
            is_correct = combined >= 0.7
        else:
            combined = calc_result
            is_correct = calc_result >= 0.8

        return VerificationResult(
            is_correct=is_correct,
            confidence=min(1.0, max(0.0, combined)),
            method=VerificationMethod.EXECUTION.value,
            answer=answer,
            expected=expected,
            details=f"Calculation verification: {calc_result:.2f}",
            verification_time_ms=verification_time,
        )

    def _extract_code(self, text: str) -> List[str]:
        """Extract code blocks from text.

        Args:
            text: Text potentially containing code.

        Returns:
            List of code strings.
        """
        code_blocks: List[str] = []
        python_pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(python_pattern, text, re.DOTALL)
        code_blocks.extend(matches)

        inline_pattern = r'`([^`]+)`'
        inline_matches = re.findall(inline_pattern, text)
        for match in inline_matches:
            if any(c in match for c in ['=', '+', '-', '*', '/', '(', ')']):
                code_blocks.append(match)

        return [c.strip() for c in code_blocks if c.strip()]

    def _execute_code_blocks(self, code_blocks: List[str]) -> Optional[str]:
        """Execute code blocks safely and return results.

        Args:
            code_blocks: List of code strings to execute.

        Returns:
            Execution result string, or None if execution failed.
        """
        for code in code_blocks:
            try:
                safe_globals: Dict[str, Any] = {
                    "__builtins__": {
                        "print": lambda *a: None,
                        "range": range,
                        "len": len,
                        "int": int,
                        "float": float,
                        "str": str,
                        "list": list,
                        "dict": dict,
                        "set": set,
                        "tuple": tuple,
                        "abs": abs,
                        "min": min,
                        "max": max,
                        "sum": sum,
                        "round": round,
                        "sorted": sorted,
                        "enumerate": enumerate,
                        "zip": zip,
                        "map": map,
                        "filter": filter,
                        "True": True,
                        "False": False,
                        "None": None,
                    },
                }

                for module_name in self.allowed_modules:
                    try:
                        safe_globals[module_name] = __import__(module_name)
                    except ImportError:
                        pass

                result = eval(code, safe_globals, {})  # noqa: S307
                return str(result)
            except Exception:
                try:
                    local_vars: Dict[str, Any] = {}
                    safe_globals = {
                        "__builtins__": {
                            "print": lambda *a: None,
                            "range": range,
                            "len": len,
                            "int": int,
                            "float": float,
                            "str": str,
                            "abs": abs,
                            "min": min,
                            "max": max,
                            "sum": sum,
                            "round": round,
                            "sorted": sorted,
                            "True": True,
                            "False": False,
                            "None": None,
                        },
                    }
                    exec(code, safe_globals, local_vars)  # noqa: S102
                    if "result" in local_vars:
                        return str(local_vars["result"])
                    if "answer" in local_vars:
                        return str(local_vars["answer"])
                    if "output" in local_vars:
                        return str(local_vars["output"])
                except Exception:
                    continue
        return None

    def _verify_calculation(self, answer: str, question: str) -> float:
        """Verify arithmetic calculations in the answer.

        Args:
            answer: The proposed answer.
            question: The question.

        Returns:
            Verification score.
        """
        expressions = re.findall(
            r'(\d+\.?\d*)\s*([+\-*/x×÷])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',
            answer,
        )
        if expressions:
            correct = 0
            for a_str, op, b_str, result_str in expressions:
                try:
                    a, b, result = float(a_str), float(b_str), float(result_str)
                    expected = self._compute(a, op, b)
                    if expected is not None and abs(expected - result) < 1e-6:
                        correct += 1
                except (ValueError, ZeroDivisionError):
                    continue
            if expressions:
                return correct / len(expressions)

        return 0.5

    @staticmethod
    def _compute(a: float, op: str, b: float) -> Optional[float]:
        """Perform a basic arithmetic operation.

        Args:
            a: First operand.
            op: Operator string.
            b: Second operand.

        Returns:
            Result, or None for division by zero.
        """
        if op in ('+',):
            return a + b
        if op in ('-', '−'):
            return a - b
        if op in ('*', 'x', '×'):
            return a * b
        if op in ('/', '÷'):
            if b == 0:
                return None
            return a / b
        return None


# =============================================================================
# Consensus Verifier
# =============================================================================

class ConsensusVerifier(Verifier):
    """Aggregates multiple verification methods for robust verification.

    Runs multiple verifiers and combines their results using a weighted
    voting scheme to produce a more reliable final verdict.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        verifiers: Optional[List[Verifier]] = None,
        weights: Optional[Dict[str, float]] = None,
        consensus_threshold: float = 0.6,
    ) -> None:
        """Initialize the consensus verifier.

        Args:
            model: Language model interface.
            verifiers: List of verification methods to use.
            weights: Custom weights for each verification method.
            consensus_threshold: Minimum agreement for acceptance.
        """
        super().__init__(model=model)
        self.consensus_threshold = consensus_threshold

        if verifiers is not None:
            self.verifiers = verifiers
        else:
            self.verifiers = [
                SelfVerification(model=model, num_resolutions=2),
                CrossVerification(model=model, num_solutions=3),
                BackwardVerification(model=model),
                FormalVerification(model=model),
            ]

        self.weights = weights or {
            VerificationMethod.SELF.value: 0.20,
            VerificationMethod.CROSS.value: 0.30,
            VerificationMethod.BACKWARD.value: 0.20,
            VerificationMethod.FORMAL.value: 0.20,
            VerificationMethod.EXECUTION.value: 0.10,
        }

    def verify(
        self,
        answer: str,
        question: str,
        context: str = "",
        expected: str = "",
    ) -> VerificationResult:
        """Verify using consensus from multiple methods.

        Args:
            answer: The answer to verify.
            question: The original question.
            context: Additional context.
            expected: Known correct answer.

        Returns:
            Aggregated VerificationResult.
        """
        start_time = time.time()
        self._verification_count += 1

        report = VerificationReport(answer=answer)

        for verifier in self.verifiers:
            try:
                result = verifier.verify(answer, question, context, expected)
                report.add_result(result)
            except Exception as e:
                error_result = VerificationResult(
                    is_correct=False,
                    confidence=0.0,
                    method=type(verifier).__name__,
                    answer=answer,
                    error=str(e),
                )
                report.add_result(error_result)

        total_weight = 0.0
        weighted_correct = 0.0
        weighted_confidence = 0.0

        for result in report.results:
            weight = self.weights.get(result.method, 0.1)
            total_weight += weight
            if result.is_correct:
                weighted_correct += weight * result.confidence
            weighted_confidence += weight * result.confidence

        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        correct_ratio = weighted_correct / total_weight if total_weight > 0 else 0.0

        is_correct = report.overall_correct
        verification_time = (time.time() - start_time) * 1000.0
        self._total_time_ms += verification_time

        details_parts = [
            f"Ran {len(report.results)} verifiers.",
            f"Agreement ratio: {report.agreement_ratio:.2f}",
            f"Recommendation: {report.recommendation}",
        ]
        if report.disagreements:
            details_parts.append(f"Disagreements: {', '.join(report.disagreements[:3])}")

        return VerificationResult(
            is_correct=is_correct,
            confidence=min(1.0, max(0.0, final_confidence)),
            method=VerificationMethod.CONSENSUS.value,
            answer=answer,
            expected=expected,
            details=". ".join(details_parts),
            verification_time_ms=verification_time,
            tokens_used=sum(r.tokens_used for r in report.results),
            metadata={"report": report.to_dict()},
        )

    def verify_with_report(
        self,
        answer: str,
        question: str,
        context: str = "",
        expected: str = "",
    ) -> Tuple[VerificationResult, VerificationReport]:
        """Verify and return both result and detailed report.

        Args:
            answer: The answer to verify.
            question: The question.
            context: Additional context.
            expected: Known correct answer.

        Returns:
            Tuple of (result, detailed report).
        """
        result = self.verify(answer, question, context, expected)
        report = result.metadata.get("report")
        if report:
            report_obj = VerificationReport(
                answer=answer,
            )
            report_obj.overall_correct = result.is_correct
            report_obj.overall_confidence = result.confidence
            report_obj.recommendation = "accept" if result.is_correct else "reject"
            return result, report_obj
        return result, VerificationReport(answer=answer)
