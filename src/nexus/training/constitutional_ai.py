"""
Constitutional AI Training for Nexus LLM.

Implements Constitutional AI (CAI) training methodology where model responses
are critiqued and revised according to a set of constitutional principles,
then the model is trained on the revised (more aligned) responses.

Classes:
    Constitution: A collection of constitutional principles.
    ConstitutionalPrinciple: Individual principle with rules.
    ResponseCritique: Critique a model response against principles.
    ResponseRevision: Revise a response based on critique.
    ConstitutionalTrainer: Full CAI training loop.
    PrincipleEmbedder: Embed principles for similarity matching.
    ConstitutionSelector: Select relevant principles for an input.
    CritiqueGenerator: Generate critique using an LLM.
    RevisionGenerator: Generate revised response.
    ConstitutionalEvaluator: Evaluate alignment with constitution.
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
import re
import threading
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class CritiqueSeverity(Enum):
    """Severity of a critique."""
    NONE = 0
    MINOR = 1
    MODERATE = 2
    MAJOR = 3
    CRITICAL = 4


class RevisionOutcome(Enum):
    """Outcome of a revision attempt."""
    UNCHANGED = auto()
    MINOR_REVISION = auto()
    MAJOR_REVISION = auto()
    COMPLETE_REWRITE = auto()
    FAILED = auto()


class AlignmentDimension(Enum):
    """Dimension of constitutional alignment."""
    HELPFULNESS = auto()
    HARMLESSNESS = auto()
    HONESTY = auto()
    FAIRNESS = auto()
    RESPECT = auto()


class ViolationType(Enum):
    """Type of constitutional violation."""
    NONE = auto()
    HARMFUL_CONTENT = auto()
    MISINFORMATION = auto()
    DECEPTION = auto()
    BIAS = auto()
    DISRESPECT = auto()
    PRIVACY_VIOLATION = auto()
    UNHELPFUL = auto()
    OVERLY_VERBOSE = auto()
    CONTRADICTORY = auto()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ConstitutionalPrinciple:
    """Individual constitutional principle with description and application rules.

    Attributes:
        name: Unique identifier for the principle.
        description: Human-readable description of the principle.
        category: Alignment dimension this principle belongs to.
        application_rules: Specific rules for applying the principle.
        critique_prompt: Template for generating critiques.
        revision_prompt: Template for generating revisions.
        examples: Few-shot examples of violations and corrections.
        severity_weights: Weight per violation type.
        priority: Priority of this principle (higher = more important).
        active: Whether this principle is currently active.
    """
    name: str
    description: str
    category: AlignmentDimension = AlignmentDimension.HELPFULNESS
    application_rules: List[str] = field(default_factory=list)
    critique_prompt: str = ""
    revision_prompt: str = ""
    examples: List[Dict[str, str]] = field(default_factory=list)
    severity_weights: Dict[str, float] = field(default_factory=dict)
    priority: float = 1.0
    active: bool = True

    def __post_init__(self):
        if not self.critique_prompt:
            self.critique_prompt = (
                f"Review the following response for compliance with this principle: "
                f"\"{self.description}\". "
                f"Identify any violations and explain why they are problematic."
            )
        if not self.revision_prompt:
            self.revision_prompt = (
                f"Revise the following response to comply with this principle: "
                f"\"{self.description}\". "
                f"Maintain the helpful content while fixing any violations."
            )
        if not self.severity_weights:
            self.severity_weights = {
                "none": 0.0,
                "minor": 0.25,
                "moderate": 0.5,
                "major": 0.75,
                "critical": 1.0,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize principle to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.name,
            "application_rules": self.application_rules,
            "critique_prompt": self.critique_prompt,
            "revision_prompt": self.revision_prompt,
            "examples": self.examples,
            "severity_weights": self.severity_weights,
            "priority": self.priority,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstitutionalPrinciple":
        """Deserialize principle from dictionary."""
        category = AlignmentDimension[data.get("category", "HELPFULNESS")]
        return cls(
            name=data["name"],
            description=data["description"],
            category=category,
            application_rules=data.get("application_rules", []),
            critique_prompt=data.get("critique_prompt", ""),
            revision_prompt=data.get("revision_prompt", ""),
            examples=data.get("examples", []),
            severity_weights=data.get("severity_weights", {}),
            priority=data.get("priority", 1.0),
            active=data.get("active", True),
        )


@dataclass
class Constitution:
    """A collection of constitutional principles governing model behavior.

    Attributes:
        name: Name of this constitution.
        description: Description of the constitution's purpose.
        principles: List of constitutional principles.
        version: Version identifier.
        metadata: Additional metadata.
    """
    name: str
    description: str = ""
    principles: List[ConstitutionalPrinciple] = field(default_factory=list)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_principle(self, principle: ConstitutionalPrinciple):
        """Add a principle to the constitution."""
        self.principles.append(principle)

    def remove_principle(self, name: str):
        """Remove a principle by name."""
        self.principles = [p for p in self.principles if p.name != name]

    def get_principle(self, name: str) -> Optional[ConstitutionalPrinciple]:
        """Get a principle by name."""
        for p in self.principles:
            if p.name == name:
                return p
        return None

    def get_active_principles(self) -> List[ConstitutionalPrinciple]:
        """Get all active principles."""
        return [p for p in self.principles if p.active]

    def get_principles_by_category(
        self, category: AlignmentDimension
    ) -> List[ConstitutionalPrinciple]:
        """Get principles filtered by category."""
        return [
            p for p in self.principles
            if p.category == category and p.active
        ]

    def get_categories(self) -> Set[AlignmentDimension]:
        """Get all categories present in this constitution."""
        return {p.category for p in self.principles if p.active}

    def total_principles(self) -> int:
        """Return total number of principles."""
        return len(self.principles)

    def active_principles_count(self) -> int:
        """Return number of active principles."""
        return len(self.get_active_principles())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize constitution to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "principles": [p.to_dict() for p in self.principles],
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constitution":
        """Deserialize constitution from dictionary."""
        principles = [
            ConstitutionalPrinciple.from_dict(p)
            for p in data.get("principles", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            principles=principles,
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str):
        """Save constitution to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Constitution saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Constitution":
        """Load constitution from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class CritiqueResult:
    """Result of critiquing a model response.

    Attributes:
        principle_name: Name of the principle used for critique.
        violation_type: Type of violation detected.
        severity: Severity of the violation.
        critique_text: Natural language critique.
        violated_rules: List of specific rules that were violated.
        confidence: Confidence score for the critique.
        evidence: Specific evidence from the response.
        revision_suggestion: Suggested revision approach.
    """
    principle_name: str
    violation_type: ViolationType = ViolationType.NONE
    severity: CritiqueSeverity = CritiqueSeverity.NONE
    critique_text: str = ""
    violated_rules: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    revision_suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "principle_name": self.principle_name,
            "violation_type": self.violation_type.name,
            "severity": self.severity.name,
            "critique_text": self.critique_text,
            "violated_rules": self.violated_rules,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "revision_suggestion": self.revision_suggestion,
        }


@dataclass
class RevisionResult:
    """Result of revising a model response.

    Attributes:
        original_response: The original (pre-revision) response.
        revised_response: The revised response.
        principle_name: Principle that guided the revision.
        outcome: Type of revision performed.
        critique: The critique that prompted the revision.
        improvement_score: Estimated improvement from the revision.
        revision_diff: Description of what changed.
    """
    original_response: str
    revised_response: str
    principle_name: str
    outcome: RevisionOutcome = RevisionOutcome.UNCHANGED
    critique: str = ""
    improvement_score: float = 0.0
    revision_diff: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_response": self.original_response,
            "revised_response": self.revised_response,
            "principle_name": self.principle_name,
            "outcome": self.outcome.name,
            "critique": self.critique,
            "improvement_score": self.improvement_score,
            "revision_diff": self.revision_diff,
        }


@dataclass
class AlignmentScore:
    """Score representing alignment with a constitution."""
    principle_name: str
    score: float
    violation_type: ViolationType
    severity: CritiqueSeverity
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "principle_name": self.principle_name,
            "score": self.score,
            "violation_type": self.violation_type.name,
            "severity": self.severity.name,
            "details": self.details,
        }


@dataclass
class ConstitutionalTrainingExample:
    """A complete example for constitutional AI training."""
    prompt: str
    original_response: str
    critiques: List[CritiqueResult] = field(default_factory=list)
    revisions: List[RevisionResult] = field(default_factory=list)
    final_response: str = ""
    alignment_scores: List[AlignmentScore] = field(default_factory=list)
    overall_alignment: float = 0.0
    example_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.example_id:
            self.example_id = str(uuid.uuid4())[:12]
        if not self.final_response and self.revisions:
            self.final_response = self.revisions[-1].revised_response
        elif not self.final_response:
            self.final_response = self.original_response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "original_response": self.original_response,
            "critiques": [c.to_dict() for c in self.critiques],
            "revisions": [r.to_dict() for r in self.revisions],
            "final_response": self.final_response,
            "alignment_scores": [a.to_dict() for a in self.alignment_scores],
            "overall_alignment": self.overall_alignment,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstitutionalTrainingExample":
        critiques = [CritiqueResult(**c) for c in data.get("critiques", [])]
        for i, c in enumerate(critiques):
            raw = data.get("critiques", [])[i]
            c.violation_type = ViolationType[raw.get("violation_type", "NONE")]
            c.severity = CritiqueSeverity[raw.get("severity", "NONE")]
        revisions = [RevisionResult(**r) for r in data.get("revisions", [])]
        for i, r in enumerate(revisions):
            raw = data.get("revisions", [])[i]
            r.outcome = RevisionOutcome[raw.get("outcome", "UNCHANGED")]
        alignment = [AlignmentScore(**a) for a in data.get("alignment_scores", [])]
        for i, a in enumerate(alignment):
            raw = data.get("alignment_scores", [])[i]
            a.violation_type = ViolationType[raw.get("violation_type", "NONE")]
            a.severity = CritiqueSeverity[raw.get("severity", "NONE")]
        return cls(
            prompt=data["prompt"],
            original_response=data["original_response"],
            critiques=critiques,
            revisions=revisions,
            final_response=data.get("final_response", data["original_response"]),
            alignment_scores=alignment,
            overall_alignment=data.get("overall_alignment", 0.0),
            example_id=data.get("example_id", ""),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Response Critique
# ---------------------------------------------------------------------------

class ResponseCritique:
    """Critique a model response against constitutional principles.

    Analyzes responses for violations of constitutional principles and
    generates structured critiques with severity assessments.
    """

    def __init__(
        self,
        constitution: Constitution,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.constitution = constitution
        self.model = model
        self.tokenizer = tokenizer
        self._violation_patterns: Dict[str, List[re.Pattern]] = {}
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize regex patterns for violation detection."""
        self._violation_patterns = {
            "harmful": [
                re.compile(r'\b(kill|hurt|harm|attack|murder|destroy)\b', re.I),
                re.compile(r'\b(weapon|bomb|poison|explosive)\b', re.I),
                re.compile(r'\b(instructions?\s+to|h(o|ow)\s+to\s+(make|create|build))\b', re.I),
            ],
            "deception": [
                re.compile(r'\b(I\s+am\s+(absolutely\s+)?certain|I\s+guarantee)\b', re.I),
                re.compile(r'\b(it\s+is\s+a\s+fact\s+that|everyone\s+knows)\b', re.I),
                re.compile(r'\b(100%\s+(sure|certain|correct))\b', re.I),
            ],
            "bias": [
                re.compile(r'\b(all\s+\w+\s+are\s+(known\s+for|always|never))\b', re.I),
                re.compile(r'\b(\w+\s+are\s+naturally\s+(better|worse|superior|inferior))\b', re.I),
                re.compile(r'\b(obviously|clearly)\s+\w+\s+are\s+(more|less)\b', re.I),
            ],
            "privacy": [
                re.compile(r'\b(personal\s+(information|data|details))\b', re.I),
                re.compile(r'\b(phone\s+number|email\s+address|home\s+address)\b', re.I),
                re.compile(r'\b(social\s+security|credit\s+card)\b', re.I),
            ],
        }

    def critique_response(
        self,
        prompt: str,
        response: str,
        principles: Optional[List[ConstitutionalPrinciple]] = None,
    ) -> List[CritiqueResult]:
        """Critique a response against constitutional principles.

        Args:
            prompt: The original prompt.
            response: The model's response to critique.
            principles: Specific principles to check. If None, uses all active.

        Returns:
            List of CritiqueResult objects.
        """
        if principles is None:
            principles = self.constitution.get_active_principles()

        results = []
        for principle in principles:
            result = self._critique_against_principle(prompt, response, principle)
            results.append(result)

        results.sort(key=lambda r: r.severity.value, reverse=True)
        return results

    def _critique_against_principle(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> CritiqueResult:
        """Critique response against a single principle."""
        violations = self._detect_violations(response, principle)
        severity = self._assess_severity(violations, response, principle)
        confidence = self._compute_confidence(violations, principle)
        evidence = self._extract_evidence(response, violations)
        violated_rules = self._identify_violated_rules(response, principle)

        if severity == CritiqueSeverity.NONE:
            critique_text = "No violations detected."
        else:
            critique_text = self._generate_critique_text(
                principle, violations, severity, evidence
            )

        revision_suggestion = self._suggest_revision(
            principle, violations, severity
        )

        return CritiqueResult(
            principle_name=principle.name,
            violation_type=self._determine_violation_type(violations),
            severity=severity,
            critique_text=critique_text,
            violated_rules=violated_rules,
            confidence=confidence,
            evidence=evidence,
            revision_suggestion=revision_suggestion,
        )

    def _detect_violations(
        self, response: str, principle: ConstitutionalPrinciple
    ) -> List[Dict[str, Any]]:
        """Detect potential violations in a response."""
        violations = []
        response_lower = response.lower()

        for category, patterns in self._violation_patterns.items():
            if not self._category_matches_principle(category, principle):
                continue
            for pattern in patterns:
                matches = pattern.findall(response_lower)
                for match in matches:
                    violations.append({
                        "type": category,
                        "match": match,
                        "pattern": pattern.pattern,
                        "principle": principle.name,
                    })

        if principle.application_rules:
            for rule in principle.application_rules:
                rule_lower = rule.lower()
                rule_keywords = [w for w in rule_lower.split() if len(w) > 3]
                response_words = set(response_lower.split())
                keyword_matches = response_words & set(rule_keywords)
                if keyword_matches:
                    for kw in keyword_matches:
                        violations.append({
                            "type": "rule_violation",
                            "match": kw,
                            "rule": rule,
                            "principle": principle.name,
                        })

        return violations

    def _category_matches_principle(
        self, category: str, principle: ConstitutionalPrinciple
    ) -> bool:
        """Check if a violation category is relevant to a principle."""
        mapping = {
            "harmful": {AlignmentDimension.HARMLESSNESS},
            "deception": {AlignmentDimension.HONESTY},
            "bias": {AlignmentDimension.FAIRNESS},
            "privacy": {AlignmentDimension.HARMLESSNESS, AlignmentDimension.RESPECT},
        }
        relevant = mapping.get(category, set())
        return principle.category in relevant or not relevant

    def _assess_severity(
        self,
        violations: List[Dict[str, Any]],
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> CritiqueSeverity:
        """Assess the severity of detected violations."""
        if not violations:
            return CritiqueSeverity.NONE

        harmful_count = sum(1 for v in violations if v["type"] == "harmful")
        if harmful_count > 0:
            return CritiqueSeverity.CRITICAL

        rule_violations = sum(1 for v in violations if v["type"] == "rule_violation")
        other_count = len(violations) - harmful_count - rule_violations

        if rule_violations >= 3 or other_count >= 5:
            return CritiqueSeverity.MAJOR
        elif rule_violations >= 2 or other_count >= 3:
            return CritiqueSeverity.MODERATE
        elif rule_violations >= 1 or other_count >= 1:
            return CritiqueSeverity.MINOR

        return CritiqueSeverity.MINOR

    def _compute_confidence(
        self, violations: List[Dict[str, Any]], principle: ConstitutionalPrinciple
    ) -> float:
        """Compute confidence score for the critique."""
        if not violations:
            return 0.9
        unique_types = len(set(v["type"] for v in violations))
        base_confidence = min(0.9, 0.5 + unique_types * 0.15)
        return min(1.0, base_confidence)

    def _extract_evidence(
        self, response: str, violations: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract evidence from the response for violations."""
        evidence = []
        sentences = re.split(r'(?<=[.!?])\s+', response)
        response_lower = response.lower()

        for violation in violations[:5]:
            match = violation.get("match", "")
            for sentence in sentences:
                if match and match.lower() in sentence.lower():
                    evidence.append(sentence.strip())
                    break

        return list(set(evidence))[:5]

    def _identify_violated_rules(
        self, response: str, principle: ConstitutionalPrinciple
    ) -> List[str]:
        """Identify which specific rules were violated."""
        violated = []
        for rule in principle.application_rules:
            keywords = [w.lower() for w in rule.split() if len(w) > 3]
            if not keywords:
                continue
            response_words = response.lower().split()
            matches = sum(1 for kw in keywords if kw in response_words)
            if matches >= len(keywords) * 0.5:
                violated.append(rule)
        return violated

    def _generate_critique_text(
        self,
        principle: ConstitutionalPrinciple,
        violations: List[Dict[str, Any]],
        severity: CritiqueSeverity,
        evidence: List[str],
    ) -> str:
        """Generate a natural language critique."""
        parts = [
            f"The response may violate the '{principle.name}' principle:",
            f"\"{principle.description}\"",
        ]

        violation_types = list(set(v["type"] for v in violations))
        if violation_types:
            type_str = ", ".join(violation_types)
            parts.append(f"Detected potential issues: {type_str}.")

        if severity in (CritiqueSeverity.MAJOR, CritiqueSeverity.CRITICAL):
            parts.append("This is a significant concern that should be addressed.")

        if evidence:
            parts.append(f"Evidence: {'; '.join(evidence[:2])}")

        return " ".join(parts)

    def _suggest_revision(
        self,
        principle: ConstitutionalPrinciple,
        violations: List[Dict[str, Any]],
        severity: CritiqueSeverity,
    ) -> str:
        """Suggest how to revise the response."""
        if severity == CritiqueSeverity.NONE:
            return "No revision needed."
        suggestions = {
            "harmful": "Remove or reframe any content that could cause harm. Focus on educational context.",
            "deception": "Use qualified language (e.g., 'may', 'might', 'some evidence suggests').",
            "bias": "Present balanced perspectives. Avoid generalizations about groups.",
            "privacy": "Remove any personal information. Use anonymized examples.",
            "rule_violation": "Review the specific rules and adjust the response accordingly.",
        }
        violation_types = list(set(v["type"] for v in violations))
        suggestion_parts = []
        for vt in violation_types:
            suggestion = suggestions.get(vt, "Review and revise for alignment.")
            suggestion_parts.append(suggestion)
        return " ".join(suggestion_parts) if suggestion_parts else "Review for alignment."

    def _determine_violation_type(
        self, violations: List[Dict[str, Any]]
    ) -> ViolationType:
        """Determine the primary violation type."""
        if not violations:
            return ViolationType.NONE
        types = set(v["type"] for v in violations)
        if "harmful" in types:
            return ViolationType.HARMFUL_CONTENT
        elif "deception" in types:
            return ViolationType.DECEPTION
        elif "bias" in types:
            return ViolationType.BIAS
        elif "privacy" in types:
            return ViolationType.PRIVACY_VIOLATION
        elif "rule_violation" in types:
            return ViolationType.UNHELPFUL
        return ViolationType.NONE


# ---------------------------------------------------------------------------
# Response Revision
# ---------------------------------------------------------------------------

class ResponseRevision:
    """Revise model responses based on critiques and constitutional principles.

    Takes a critiqued response and produces an improved version that better
    aligns with the relevant constitutional principles.
    """

    def __init__(
        self,
        constitution: Constitution,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        max_revision_attempts: int = 3,
    ):
        self.constitution = constitution
        self.model = model
        self.tokenizer = tokenizer
        self.max_revision_attempts = max_revision_attempts
        self._revision_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize revision prompt templates."""
        return {
            "harmful": (
                "The following response contains potentially harmful content. "
                "Please rewrite it to be helpful while avoiding harm:\n\n"
                "Original: {response}\n\n"
                "Rewritten (safe and helpful):"
            ),
            "deception": (
                "The following response may contain misleading or overly "
                "confident claims. Please rewrite with appropriate uncertainty:\n\n"
                "Original: {response}\n\n"
                "Rewritten (honest and accurate):"
            ),
            "bias": (
                "The following response may contain biased generalizations. "
                "Please rewrite with balanced, fair perspective:\n\n"
                "Original: {response}\n\n"
                "Rewritten (balanced and fair):"
            ),
            "privacy": (
                "The following response may contain privacy concerns. "
                "Please rewrite removing any personal information:\n\n"
                "Original: {response}\n\n"
                "Rewritten (privacy-preserving):"
            ),
            "general": (
                "Please revise the following response to better align with "
                "the principle: \"{principle}\"\n\n"
                "Original: {response}\n\n"
                "Revised response:"
            ),
        }

    def revise_response(
        self,
        prompt: str,
        response: str,
        critiques: List[CritiqueResult],
    ) -> RevisionResult:
        """Revise a response based on critiques.

        Args:
            prompt: Original prompt.
            response: Original model response.
            critiques: List of critique results.

        Returns:
            RevisionResult with the revised response.
        """
        significant_critiques = [
            c for c in critiques
            if c.severity != CritiqueSeverity.NONE
        ]

        if not significant_critiques:
            return RevisionResult(
                original_response=response,
                revised_response=response,
                principle_name="none",
                outcome=RevisionOutcome.UNCHANGED,
                improvement_score=1.0,
                revision_diff="No revisions needed.",
            )

        most_severe = max(
            significant_critiques, key=lambda c: c.severity.value
        )
        principle = self.constitution.get_principle(most_severe.principle_name)

        if principle is None and self.constitution.principles:
            principle = self.constitution.principles[0]

        for attempt in range(self.max_revision_attempts):
            revised = self._attempt_revision(
                response, most_severe, principle, attempt
            )
            if self._check_revision_improvement(response, revised, most_severe):
                outcome, diff, score = self._assess_revision(
                    response, revised
                )
                return RevisionResult(
                    original_response=response,
                    revised_response=revised,
                    principle_name=most_severe.principle_name,
                    outcome=outcome,
                    critique=most_severe.critique_text,
                    improvement_score=score,
                    revision_diff=diff,
                )

        return RevisionResult(
            original_response=response,
            revised_response=response,
            principle_name=most_severe.principle_name,
            outcome=RevisionOutcome.FAILED,
            critique=most_severe.critique_text,
            improvement_score=0.0,
            revision_diff="Failed to produce acceptable revision.",
        )

    def revise_multiple_principles(
        self,
        prompt: str,
        response: str,
        critiques: List[CritiqueResult],
    ) -> List[RevisionResult]:
        """Revise response sequentially for each violated principle.

        Args:
            prompt: Original prompt.
            response: Original response.
            critiques: All critique results.

        Returns:
            List of RevisionResult, one per significant critique.
        """
        results = []
        current_response = response

        significant = [
            c for c in critiques
            if c.severity != CritiqueSeverity.NONE
        ]

        for critique in significant:
            result = self.revise_response(
                prompt, current_response, [critique]
            )
            if result.outcome != RevisionOutcome.FAILED:
                current_response = result.revised_response
            results.append(result)

        return results

    def _attempt_revision(
        self,
        response: str,
        critique: CritiqueResult,
        principle: Optional[ConstitutionalPrinciple],
        attempt: int,
    ) -> str:
        """Attempt a single revision."""
        if self.model is not None and self.tokenizer is not None:
            return self._model_revision(response, critique, principle, attempt)
        return self._heuristic_revision(response, critique, principle)

    def _model_revision(
        self,
        response: str,
        critique: CritiqueResult,
        principle: Optional[ConstitutionalPrinciple],
        attempt: int,
    ) -> str:
        """Use the model to generate a revised response."""
        violation_type = critique.violation_type.name.lower()
        template = self._revision_templates.get(
            violation_type, self._revision_templates["general"]
        )

        principle_desc = (
            principle.description if principle else "constitutional principles"
        )
        prompt = template.format(
            response=response,
            principle=principle_desc,
        )

        if attempt > 0:
            prompt += f"\n\nPrevious attempt had issues. Try a different approach (attempt {attempt + 1})."

        try:
            self.model.eval()
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.model.device.type != "cpu":
                input_ids = input_ids.to(self.model.device)
            temperature = 0.7 + attempt * 0.1
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=len(input_ids[0]) + 512,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                )
            revised = self.tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
            )
            self.model.train()
            return revised if revised.strip() else response
        except Exception as e:
            logger.debug(f"Model revision failed: {e}")
            return response

    def _heuristic_revision(
        self,
        response: str,
        critique: CritiqueResult,
        principle: Optional[ConstitutionalPrinciple],
    ) -> str:
        """Apply heuristic-based revision when no model is available."""
        revised = response

        harmful_replacements = {
            "kill": "stop",
            "murder": "harm",
            "attack": "approach",
            "destroy": "change",
            "weapon": "tool",
            "bomb": "device",
        }

        for orig, replacement in harmful_replacements.items():
            pattern = re.compile(rf'\b{re.escape(orig)}\b', re.I)
            revised = pattern.sub(replacement, revised)

        revised = re.sub(
            r'\b(I\s+(am\s+)?(absolutely\s+)?certain|I\s+guarantee)\b',
            'I believe', revised, flags=re.I
        )
        revised = re.sub(
            r'\b(it\s+is\s+a\s+fact|everyone\s+knows)\b',
            'it is generally understood', revised, flags=re.I
        )

        bias_patterns = [
            (r'\b(all\s+\w+\s+are\s+\w+)\b', 'Some people of this group may be'),
            (r'\b(\w+\s+are\s+naturally\s+\w+)\b', r'\1 can sometimes be'),
        ]
        for pattern, replacement in bias_patterns:
            revised = re.sub(pattern, replacement, revised, flags=re.I)

        disclaimers = [
            "\n\nNote: This is a general overview and individual cases may vary.",
            "\n\nDisclaimer: This information should be verified with current sources.",
        ]

        if critique.severity.value >= CritiqueSeverity.MODERATE.value:
            revised += random.choice(disclaimers)

        return revised

    def _check_revision_improvement(
        self, original: str, revised: str, critique: CritiqueResult
    ) -> bool:
        """Check if a revision is an improvement over the original."""
        if not revised or revised == original:
            return False
        if len(revised) < len(original) * 0.3:
            return False
        return True

    def _assess_revision(
        self, original: str, revised: str
    ) -> Tuple[RevisionOutcome, str, float]:
        """Assess the type and quality of revision."""
        orig_words = set(original.lower().split())
        rev_words = set(revised.lower().split())

        common = orig_words & rev_words
        jaccard = len(common) / max(1, len(orig_words | rev_words))

        if jaccard > 0.95:
            outcome = RevisionOutcome.UNCHANGED
            diff = "Minimal changes made."
            score = 0.9
        elif jaccard > 0.7:
            outcome = RevisionOutcome.MINOR_REVISION
            diff = "Minor wording adjustments."
            score = 0.8
        elif jaccard > 0.4:
            outcome = RevisionOutcome.MAJOR_REVISION
            diff = "Significant rephrasing with same core content."
            score = 0.7
        else:
            outcome = RevisionOutcome.COMPLETE_REWRITE
            diff = "Response substantially rewritten."
            score = 0.5

        length_ratio = len(revised) / max(1, len(original))
        if 0.5 < length_ratio < 2.0:
            score += 0.1
        else:
            score -= 0.2

        return outcome, diff, min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Critique Generator
# ---------------------------------------------------------------------------

class CritiqueGenerator:
    """Generate critiques using the model or rule-based methods."""

    def __init__(
        self,
        constitution: Constitution,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        use_model: bool = True,
        temperature: float = 0.3,
    ):
        self.constitution = constitution
        self.model = model
        self.tokenizer = tokenizer
        self.use_model = use_model and model is not None
        self.temperature = temperature
        self._response_critic = ResponseCritique(constitution, model, tokenizer)

    def generate_critique(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> CritiqueResult:
        """Generate a critique for a response against a principle.

        Args:
            prompt: The original prompt.
            response: The model response to critique.
            principle: The principle to check against.

        Returns:
            CritiqueResult with the critique details.
        """
        if self.use_model:
            return self._model_critique(prompt, response, principle)
        return self._rule_critique(prompt, response, principle)

    def generate_all_critiques(
        self,
        prompt: str,
        response: str,
    ) -> List[CritiqueResult]:
        """Generate critiques against all active principles.

        Args:
            prompt: The original prompt.
            response: The model response.

        Returns:
            List of CritiqueResult objects.
        """
        return self._response_critic.critique_response(prompt, response)

    def _model_critique(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> CritiqueResult:
        """Use the model to generate a critique."""
        critique_prompt = principle.critique_prompt
        full_prompt = (
            f"{critique_prompt}\n\n"
            f"Prompt: {prompt}\n\n"
            f"Response: {response}\n\n"
            f"Critique:"
        )

        try:
            self.model.eval()
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
            if self.model.device.type != "cpu":
                input_ids = input_ids.to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=False,
                )
            critique_text = self.tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
            )
            self.model.train()
        except Exception as e:
            logger.debug(f"Model critique failed: {e}")
            return self._rule_critique(prompt, response, principle)

        severity = self._parse_severity(critique_text)
        violation_type = self._parse_violation_type(critique_text)

        return CritiqueResult(
            principle_name=principle.name,
            violation_type=violation_type,
            severity=severity,
            critique_text=critique_text,
            confidence=0.7,
        )

    def _rule_critique(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> CritiqueResult:
        """Use rule-based methods to generate a critique."""
        results = self._response_critic.critique_response(
            prompt, response, [principle]
        )
        return results[0] if results else CritiqueResult(
            principle_name=principle.name,
        )

    def _parse_severity(self, critique_text: str) -> CritiqueSeverity:
        """Parse severity from critique text."""
        text_lower = critique_text.lower()
        if any(w in text_lower for w in ["critical", "severe", "urgent", "dangerous"]):
            return CritiqueSeverity.CRITICAL
        elif any(w in text_lower for w in ["major", "significant", "serious"]):
            return CritiqueSeverity.MAJOR
        elif any(w in text_lower for w in ["moderate", "concerning", "notable"]):
            return CritiqueSeverity.MODERATE
        elif any(w in text_lower for w in ["minor", "slight", "small"]):
            return CritiqueSeverity.MINOR
        elif any(w in text_lower for w in ["no violation", "compliant", "acceptable"]):
            return CritiqueSeverity.NONE
        return CritiqueSeverity.MINOR

    def _parse_violation_type(self, critique_text: str) -> ViolationType:
        """Parse violation type from critique text."""
        text_lower = critique_text.lower()
        if any(w in text_lower for w in ["harmful", "dangerous", "unsafe"]):
            return ViolationType.HARMFUL_CONTENT
        elif any(w in text_lower for w in ["misleading", "inaccurate", "deceptive"]):
            return ViolationType.MISINFORMATION
        elif any(w in text_lower for w in ["dishonest", "untruthful", "false"]):
            return ViolationType.DECEPTION
        elif any(w in text_lower for w in ["biased", "unfair", "prejudiced"]):
            return ViolationType.BIAS
        elif any(w in text_lower for w in ["disrespectful", "rude", "insulting"]):
            return ViolationType.DISRESPECT
        elif any(w in text_lower for w in ["privacy", "personal information"]):
            return ViolationType.PRIVACY_VIOLATION
        return ViolationType.NONE


# ---------------------------------------------------------------------------
# Revision Generator
# ---------------------------------------------------------------------------

class RevisionGenerator:
    """Generate revised responses based on critiques and principles."""

    def __init__(
        self,
        constitution: Constitution,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        use_model: bool = True,
        temperature: float = 0.7,
    ):
        self.constitution = constitution
        self.model = model
        self.tokenizer = tokenizer
        self.use_model = use_model and model is not None
        self.temperature = temperature
        self._revisor = ResponseRevision(constitution, model, tokenizer)

    def generate_revision(
        self,
        prompt: str,
        response: str,
        critique: CritiqueResult,
    ) -> RevisionResult:
        """Generate a revised response.

        Args:
            prompt: Original prompt.
            response: Original response.
            critique: Critique of the response.

        Returns:
            RevisionResult with the revised response.
        """
        if self.use_model:
            principle = self.constitution.get_principle(critique.principle_name)
            return self._model_revision(prompt, response, critique, principle)
        return self._revisor.revise_response(prompt, response, [critique])

    def _model_revision(
        self,
        prompt: str,
        response: str,
        critique: CritiqueResult,
        principle: Optional[ConstitutionalPrinciple],
    ) -> RevisionResult:
        """Use the model to generate a revision."""
        if principle is None:
            return RevisionResult(
                original_response=response,
                revised_response=response,
                principle_name=critique.principle_name,
                outcome=RevisionOutcome.FAILED,
            )

        revision_prompt = principle.revision_prompt
        full_prompt = (
            f"{revision_prompt}\n\n"
            f"Prompt: {prompt}\n\n"
            f"Original Response: {response}\n\n"
            f"Critique: {critique.critique_text}\n\n"
            f"Revised Response:"
        )

        try:
            self.model.eval()
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
            if self.model.device.type != "cpu":
                input_ids = input_ids.to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max(len(input_ids[0]), 512),
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                )
            revised = self.tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
            )
            self.model.train()

            outcome, diff, score = self._revisor._assess_revision(response, revised)
            return RevisionResult(
                original_response=response,
                revised_response=revised if revised.strip() else response,
                principle_name=principle.name,
                outcome=outcome,
                critique=critique.critique_text,
                improvement_score=score,
                revision_diff=diff,
            )
        except Exception as e:
            logger.debug(f"Model revision failed: {e}")
            return self._revisor.revise_response(prompt, response, [critique])


# ---------------------------------------------------------------------------
# Principle Embedder
# ---------------------------------------------------------------------------

class PrincipleEmbedder:
    """Embed constitutional principles for similarity-based matching.

    Uses TF-IDF-style encoding or model embeddings to represent principles
    as vectors for efficient similarity computation.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        embedding_dim: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self._principle_embeddings: Dict[str, torch.Tensor] = {}
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._is_fitted = False

    def fit(self, principles: List[ConstitutionalPrinciple]):
        """Fit the embedder on a set of principles.

        Args:
            principles: List of principles to learn embeddings from.
        """
        self._build_vocabulary(principles)
        self._compute_idf(principles)
        self._compute_embeddings(principles)
        self._is_fitted = True

    def _build_vocabulary(self, principles: List[ConstitutionalPrinciple]):
        """Build vocabulary from principle descriptions."""
        doc_freq: Counter = Counter()
        all_text = []

        for p in principles:
            text = f"{p.name} {p.description} {' '.join(p.application_rules)}"
            all_text.append(text.lower())
            words = set(text.lower().split())
            for word in words:
                doc_freq[word] += 1

        vocab = sorted(doc_freq.keys())
        self._vocabulary = {word: idx for idx, word in enumerate(vocab)}

    def _compute_idf(self, principles: List[ConstitutionalPrinciple]):
        """Compute inverse document frequency."""
        n = len(principles)
        doc_freq: Counter = Counter()
        for p in principles:
            text = f"{p.name} {p.description} {' '.join(p.application_rules)}"
            words = set(text.lower().split())
            for word in words:
                doc_freq[word] += 1

        for word, freq in doc_freq.items():
            self._idf[word] = math.log((n + 1) / (freq + 1)) + 1

    def _compute_embeddings(self, principles: List[ConstitutionalPrinciple]):
        """Compute embeddings for each principle."""
        vocab_size = len(self._vocabulary)

        for principle in principles:
            text = f"{principle.name} {principle.description} {' '.join(principle.application_rules)}"
            embedding = torch.zeros(vocab_size) if vocab_size > 0 else torch.zeros(self.embedding_dim)

            words = text.lower().split()
            word_counts = Counter(words)

            for word, count in word_counts.items():
                if word in self._vocabulary:
                    idx = self._vocabulary[word]
                    tf = count / max(1, len(words))
                    idf = self._idf.get(word, 1.0)
                    if idx < len(embedding):
                        embedding[idx] = tf * idf

            norm = embedding.norm()
            if norm > 0:
                embedding = embedding / norm

            self._principle_embeddings[principle.name] = embedding

    def embed_text(self, text: str) -> torch.Tensor:
        """Embed arbitrary text using the same vocabulary.

        Args:
            text: Text to embed.

        Returns:
            Embedding tensor.
        """
        if not self._is_fitted:
            return torch.zeros(self.embedding_dim)

        vocab_size = len(self._vocabulary)
        embedding = torch.zeros(vocab_size) if vocab_size > 0 else torch.zeros(self.embedding_dim)

        words = text.lower().split()
        word_counts = Counter(words)

        for word, count in word_counts.items():
            if word in self._vocabulary:
                idx = self._vocabulary[word]
                tf = count / max(1, len(words))
                idf = self._idf.get(word, 1.0)
                if idx < len(embedding):
                    embedding[idx] = tf * idf

        norm = embedding.norm()
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        return float(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item())

    def find_most_similar_principle(
        self, text: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Find the most similar principles to a given text.

        Args:
            text: Query text.
            top_k: Number of principles to return.

        Returns:
            List of (principle_name, similarity_score) tuples.
        """
        if not self._principle_embeddings:
            return []

        text_embedding = self.embed_text(text)
        similarities = []

        for name, principle_emb in self._principle_embeddings.items():
            sim = float(F.cosine_similarity(
                text_embedding.unsqueeze(0),
                principle_emb.unsqueeze(0),
            ).item())
            similarities.append((name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# ---------------------------------------------------------------------------
# Constitution Selector
# ---------------------------------------------------------------------------

class ConstitutionSelector:
    """Select relevant constitutional principles for a given input.

    Uses embedding similarity and rule-based matching to select the
    most applicable principles for critiquing a given prompt-response pair.
    """

    def __init__(
        self,
        constitution: Constitution,
        embedder: Optional[PrincipleEmbedder] = None,
        similarity_threshold: float = 0.1,
        max_principles: int = 5,
    ):
        self.constitution = constitution
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_principles = max_principles

        if self.embedder is not None and not self.embedder._is_fitted:
            self.embedder.fit(self.constitution.get_active_principles())

        self._selection_history: List[Dict[str, Any]] = []
        self._principle_usage: Counter = Counter()

    def select_principles(
        self,
        prompt: str,
        response: str,
        top_k: Optional[int] = None,
    ) -> List[ConstitutionalPrinciple]:
        """Select relevant principles for the given prompt and response.

        Args:
            prompt: The user prompt.
            response: The model response.
            top_k: Maximum principles to return.

        Returns:
            List of relevant ConstitutionalPrinciple objects.
        """
        k = top_k or self.max_principles
        combined_text = f"{prompt} {response}"

        if self.embedder is not None:
            similar = self.embedder.find_most_similar_principle(combined_text, top_k=k * 2)
            candidates = [
                (name, score) for name, score in similar
                if score >= self.similarity_threshold
            ]
        else:
            candidates = self._keyword_select(combined_text, k * 2)

        candidates = candidates[:k * 2]

        filtered = self._filter_by_category_balance(candidates, k)

        selected_principles = []
        for name, score in filtered:
            principle = self.constitution.get_principle(name)
            if principle and principle.active:
                selected_principles.append(principle)
                self._principle_usage[name] += 1
            if len(selected_principles) >= k:
                break

        if not selected_principles:
            active = self.constitution.get_active_principles()
            selected_principles = active[:k]

        self._selection_history.append({
            "prompt_preview": prompt[:100],
            "selected": [p.name for p in selected_principles],
            "num_candidates": len(candidates),
        })

        return selected_principles

    def _keyword_select(
        self, text: str, top_k: int
    ) -> List[Tuple[str, float]]:
        """Select principles based on keyword matching."""
        text_lower = text.lower()
        text_words = set(text_lower.split())

        scores = []
        for principle in self.constitution.get_active_principles():
            principle_text = (
                f"{principle.name} {principle.description} "
                f"{' '.join(principle.application_rules)}"
            ).lower()
            principle_words = set(principle_text.split())

            overlap = text_words & principle_words
            if principle_words:
                score = len(overlap) / len(principle_words)
            else:
                score = 0.0

            for rule in principle.application_rules:
                rule_words = set(rule.lower().split())
                rule_overlap = text_words & rule_words
                if rule_words:
                    rule_score = len(rule_overlap) / len(rule_words) * 2.0
                    score = max(score, rule_score)

            scores.append((principle.name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _filter_by_category_balance(
        self,
        candidates: List[Tuple[str, float]],
        max_count: int,
    ) -> List[Tuple[str, float]]:
        """Ensure diversity of categories in selected principles."""
        selected = []
        category_counts: Counter = Counter()

        for name, score in candidates:
            principle = self.constitution.get_principle(name)
            if principle is None:
                continue
            category = principle.category
            if category_counts[category] < max(1, max_count // 2):
                selected.append((name, score))
                category_counts[category] += 1

        remaining = [
            (name, score) for name, score in candidates
            if name not in {s[0] for s in selected}
        ]
        selected.extend(remaining[:max(1, max_count - len(selected))])

        return selected[:max_count]

    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for principles."""
        return dict(self._principle_usage)

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about principle selection."""
        avg_selected = 0.0
        if self._selection_history:
            avg_selected = sum(
                len(h["selected"]) for h in self._selection_history
            ) / len(self._selection_history)

        return {
            "total_selections": len(self._selection_history),
            "avg_principles_per_selection": avg_selected,
            "principle_usage": dict(self._principle_usage),
        }


# ---------------------------------------------------------------------------
# Constitutional Evaluator
# ---------------------------------------------------------------------------

class ConstitutionalEvaluator:
    """Evaluate model alignment with a constitution.

    Provides comprehensive evaluation of how well model responses
    align with constitutional principles.
    """

    def __init__(
        self,
        constitution: Constitution,
        critique_engine: Optional[ResponseCritique] = None,
        embedder: Optional[PrincipleEmbedder] = None,
        selector: Optional[ConstitutionSelector] = None,
    ):
        self.constitution = constitution
        self.critique_engine = critique_engine or ResponseCritique(constitution)
        self.embedder = embedder
        self.selector = selector

        if self.embedder is not None and not self.embedder._is_fitted:
            self.embedder.fit(constitution.get_active_principles())

        self._evaluation_history: List[Dict[str, Any]] = []

    def evaluate_response(
        self,
        prompt: str,
        response: str,
        principles: Optional[List[ConstitutionalPrinciple]] = None,
    ) -> List[AlignmentScore]:
        """Evaluate alignment of a response with the constitution.

        Args:
            prompt: Original prompt.
            response: Model response.
            principles: Specific principles to check.

        Returns:
            List of AlignmentScore objects.
        """
        if principles is None:
            if self.selector:
                principles = self.selector.select_principles(prompt, response)
            else:
                principles = self.constitution.get_active_principles()

        critiques = self.critique_engine.critique_response(
            prompt, response, principles
        )

        scores = []
        for critique in critiques:
            alignment = self._compute_alignment_score(critique)
            scores.append(AlignmentScore(
                principle_name=critique.principle_name,
                score=alignment,
                violation_type=critique.violation_type,
                severity=critique.severity,
                details=critique.critique_text,
            ))

        self._evaluation_history.append({
            "prompt_preview": prompt[:100],
            "num_principles": len(principles),
            "avg_score": sum(s.score for s in scores) / max(1, len(scores)),
            "violations": sum(1 for s in scores if s.score < 0.7),
        })

        return scores

    def evaluate_batch(
        self,
        examples: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Evaluate a batch of prompt-response pairs.

        Args:
            examples: List of (prompt, response) tuples.

        Returns:
            Dictionary with aggregate evaluation metrics.
        """
        all_scores = []
        per_principle_scores: Dict[str, List[float]] = defaultdict(list)
        violation_counts: Counter = Counter()

        for prompt, response in examples:
            scores = self.evaluate_response(prompt, response)
            for score in scores:
                all_scores.append(score.score)
                per_principle_scores[score.principle_name].append(score.score)
                if score.violation_type != ViolationType.NONE:
                    violation_counts[score.violation_type.name] += 1

        overall = (
            sum(all_scores) / len(all_scores) if all_scores else 1.0
        )
        per_principle_avg = {
            k: sum(v) / len(v) for k, v in per_principle_scores.items()
        }

        return {
            "overall_alignment": overall,
            "total_evaluations": len(examples),
            "per_principle_alignment": per_principle_avg,
            "violation_distribution": dict(violation_counts),
            "total_violations": sum(violation_counts.values()),
            "violation_rate": (
                sum(violation_counts.values()) / max(1, len(all_scores))
            ),
        }

    def _compute_alignment_score(self, critique: CritiqueResult) -> float:
        """Compute an alignment score from a critique result."""
        if critique.severity == CritiqueSeverity.NONE:
            return 1.0
        elif critique.severity == CritiqueSeverity.MINOR:
            return 0.85
        elif critique.severity == CritiqueSeverity.MODERATE:
            return 0.6
        elif critique.severity == CritiqueSeverity.MAJOR:
            return 0.35
        else:
            return 0.1

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Return the full evaluation history."""
        return list(self._evaluation_history)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all evaluations."""
        if not self._evaluation_history:
            return {"total_evaluations": 0}
        avg_scores = [h["avg_score"] for h in self._evaluation_history]
        return {
            "total_evaluations": len(self._evaluation_history),
            "avg_alignment": sum(avg_scores) / len(avg_scores),
            "min_alignment": min(avg_scores),
            "max_alignment": max(avg_scores),
            "total_violations": sum(h["violations"] for h in self._evaluation_history),
        }


# ---------------------------------------------------------------------------
# Constitutional Trainer
# ---------------------------------------------------------------------------

class ConstitutionalTrainer:
    """Training loop for Constitutional AI: generate -> critique -> revise -> train.

    Orchestrates the full CAI training pipeline, managing generation of
    initial responses, critique, revision, and supervised training on
    the revised responses.
    """

    def __init__(
        self,
        constitution: Constitution,
        model: nn.Module,
        tokenizer: Any,
        critique_generator: Optional[CritiqueGenerator] = None,
        revision_generator: Optional[RevisionGenerator] = None,
        evaluator: Optional[ConstitutionalEvaluator] = None,
        selector: Optional[ConstitutionSelector] = None,
        learning_rate: float = 5e-6,
        batch_size: int = 8,
        max_revision_rounds: int = 2,
        min_alignment_threshold: float = 0.7,
    ):
        self.constitution = constitution
        self.model = model
        self.tokenizer = tokenizer
        self.critique_generator = critique_generator or CritiqueGenerator(
            constitution, model, tokenizer
        )
        self.revision_generator = revision_generator or RevisionGenerator(
            constitution, model, tokenizer
        )
        self.evaluator = evaluator or ConstitutionalEvaluator(constitution)
        self.selector = selector or ConstitutionSelector(constitution)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_revision_rounds = max_revision_rounds
        self.min_alignment_threshold = min_alignment_threshold

        self._training_examples: List[ConstitutionalTrainingExample] = []
        self._round_summaries: List[Dict[str, Any]] = []
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._current_step = 0
        self._current_round = 0

    def process_single(
        self, prompt: str, initial_response: Optional[str] = None
    ) -> ConstitutionalTrainingExample:
        """Process a single prompt through the CAI pipeline.

        Args:
            prompt: User prompt to process.
            initial_response: Optional pre-generated response.

        Returns:
            ConstitutionalTrainingExample with full processing results.
        """
        if initial_response is None:
            initial_response = self._generate_response(prompt)

        principles = self.selector.select_principles(prompt, initial_response)
        critiques = self.critique_generator.generate_all_critiques(
            prompt, initial_response
        )

        revisions = self.revision_generator._revisor.revise_multiple_principles(
            prompt, initial_response, critiques
        )

        final_response = initial_response
        if revisions:
            for rev in revisions:
                if rev.outcome != RevisionOutcome.FAILED:
                    final_response = rev.revised_response

        alignment_scores = self.evaluator.evaluate_response(
            prompt, final_response, principles
        )

        overall = (
            sum(s.score for s in alignment_scores) / len(alignment_scores)
            if alignment_scores else 1.0
        )

        example = ConstitutionalTrainingExample(
            prompt=prompt,
            original_response=initial_response,
            critiques=critiques,
            revisions=revisions,
            final_response=final_response,
            alignment_scores=alignment_scores,
            overall_alignment=overall,
        )
        self._training_examples.append(example)
        return example

    def process_batch(
        self, prompts: List[str]
    ) -> List[ConstitutionalTrainingExample]:
        """Process a batch of prompts through the CAI pipeline.

        Args:
            prompts: List of user prompts.

        Returns:
            List of ConstitutionalTrainingExample objects.
        """
        examples = []
        for prompt in prompts:
            example = self.process_single(prompt)
            examples.append(example)
        return examples

    def _generate_response(self, prompt: str) -> str:
        """Generate an initial response using the model."""
        try:
            self.model.eval()
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.model.device.type != "cpu":
                input_ids = input_ids.to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=2048,
                    temperature=0.8,
                    top_p=0.95,
                    do_sample=True,
                )
            response = self.tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
            )
            self.model.train()
            return response if response.strip() else "I apologize, but I don't have a specific answer for that."
        except Exception as e:
            logger.debug(f"Response generation failed: {e}")
            return "I apologize, but I cannot provide a response at this time."

    def train_on_examples(
        self,
        examples: List[ConstitutionalTrainingExample],
        num_epochs: int = 1,
    ) -> Dict[str, float]:
        """Train the model on constitutional training examples.

        Args:
            examples: Training examples with revised responses.
            num_epochs: Number of training epochs.

        Returns:
            Dictionary of training metrics.
        """
        if not examples:
            return {"loss": 0.0, "num_examples": 0}

        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate
            )

        self.model.train()
        total_loss = 0.0
        num_trained = 0

        for epoch in range(num_epochs):
            random.shuffle(examples)
            for example in examples:
                target_response = example.final_response
                if not target_response:
                    continue

                try:
                    input_ids = self.tokenizer.encode(
                        target_response, return_tensors="pt"
                    )
                    if self.model.device.type != "cpu":
                        input_ids = input_ids.to(self.model.device)
                    outputs = self.model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    total_loss += loss.item()
                    loss.backward()
                    num_trained += 1

                    if num_trained % self.batch_size == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0
                        )
                        self._optimizer.step()
                        self._optimizer.zero_grad()

                    self._current_step += 1
                except Exception as e:
                    logger.debug(f"Training step failed: {e}")
                    continue

        if num_trained % self.batch_size != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self._optimizer.step()
            self._optimizer.zero_grad()

        avg_loss = total_loss / max(1, num_trained)
        return {
            "loss": avg_loss,
            "num_examples": num_trained,
            "epochs": num_epochs,
            "step": self._current_step,
        }

    def run_training_round(
        self,
        prompts: List[str],
        num_epochs: int = 1,
    ) -> Dict[str, Any]:
        """Run a complete CAI training round.

        Args:
            prompts: Prompts for this round.
            num_epochs: Training epochs.

        Returns:
            Round summary dictionary.
        """
        self._current_round += 1
        start_time = time.time()

        examples = self.process_batch(prompts)

        qualified = [
            e for e in examples
            if e.overall_alignment >= self.min_alignment_threshold
        ]

        train_metrics = self.train_on_examples(qualified, num_epochs=num_epochs)

        eval_results = self.evaluator.evaluate_batch(
            [(e.prompt, e.final_response) for e in examples]
        )

        summary = {
            "round": self._current_round,
            "num_prompts": len(prompts),
            "num_examples": len(examples),
            "num_qualified": len(qualified),
            "avg_alignment": (
                sum(e.overall_alignment for e in examples) / max(1, len(examples))
            ),
            "train_metrics": train_metrics,
            "eval_results": eval_results,
            "duration_seconds": time.time() - start_time,
        }
        self._round_summaries.append(summary)
        return summary

    def get_training_examples(self) -> List[Dict[str, Any]]:
        """Get all training examples as dictionaries."""
        return [e.to_dict() for e in self._training_examples]

    def get_round_summaries(self) -> List[Dict[str, Any]]:
        """Get all round summaries."""
        return list(self._round_summaries)

    def save_training_data(self, path: str):
        """Save all training examples to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "constitution": self.constitution.name,
            "examples": [e.to_dict() for e in self._training_examples],
            "rounds": self._round_summaries,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Training data saved to {path}")


# ---------------------------------------------------------------------------
# Constitutional AI Dataset
# ---------------------------------------------------------------------------

class ConstitutionalAIDataset(Dataset):
    """PyTorch Dataset for constitutional AI training examples."""

    def __init__(
        self,
        examples: List[ConstitutionalTrainingExample],
        tokenizer: Optional[Any] = None,
        max_length: int = 2048,
        use_final_response: bool = True,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_final_response = use_final_response

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.examples):
            raise IndexError(f"Index {idx} out of range")
        example = self.examples[idx]
        response = example.final_response if self.use_final_response else example.original_response

        result = {
            "prompt": example.prompt,
            "response": response,
            "alignment_score": example.overall_alignment,
            "example_id": example.example_id,
        }

        if self.tokenizer is not None:
            full_text = f"{example.prompt} {response}"
            tokens = self.tokenizer.encode(
                full_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            result["input_ids"] = tokens.squeeze(0)
            result["labels"] = tokens.squeeze(0).clone()

        return result


# ---------------------------------------------------------------------------
# Pre-Built Constitutions
# ---------------------------------------------------------------------------

def create_helpfulness_constitution() -> Constitution:
    """Create a constitution focused on helpfulness."""
    constitution = Constitution(
        name="helpfulness",
        description="Principles for maximizing helpfulness of responses.",
    )

    principles = [
        ConstitutionalPrinciple(
            name="be_helpful",
            description="Provide useful, relevant, and substantive answers to user questions.",
            category=AlignmentDimension.HELPFULNESS,
            application_rules=[
                "Address the user's actual question directly",
                "Provide specific, actionable information when possible",
                "Include relevant examples and explanations",
            ],
            priority=1.0,
        ),
        ConstitutionalPrinciple(
            name="be_clear",
            description="Communicate clearly and concisely without unnecessary verbosity.",
            category=AlignmentDimension.HELPFULNESS,
            application_rules=[
                "Use clear and accessible language",
                "Structure responses logically",
                "Avoid unnecessary repetition",
            ],
            priority=0.9,
        ),
        ConstitutionalPrinciple(
            name="be_thorough",
            description="Provide comprehensive coverage of the topic when appropriate.",
            category=AlignmentDimension.HELPFULNESS,
            application_rules=[
                "Cover important aspects of the question",
                "Consider multiple perspectives when relevant",
                "Provide context that aids understanding",
            ],
            priority=0.8,
        ),
        ConstitutionalPrinciple(
            name="acknowledge_limitations",
            description="Honest about what you know and don't know.",
            category=AlignmentDimension.HONESTY,
            application_rules=[
                "State when you are uncertain",
                "Distinguish between facts and opinions",
                "Suggest reliable sources for further information",
            ],
            priority=0.9,
        ),
    ]

    for p in principles:
        constitution.add_principle(p)

    return constitution


def create_harmlessness_constitution() -> Constitution:
    """Create a constitution focused on harmlessness."""
    constitution = Constitution(
        name="harmlessness",
        description="Principles for ensuring responses do not cause harm.",
    )

    principles = [
        ConstitutionalPrinciple(
            name="no_harm",
            description="Do not generate content that could cause physical, psychological, or social harm.",
            category=AlignmentDimension.HARMLESSNESS,
            application_rules=[
                "Do not provide instructions for harmful activities",
                "Do not encourage dangerous behavior",
                "Do not generate content that could be used for harm",
            ],
            priority=1.0,
        ),
        ConstitutionalPrinciple(
            name="protect_vulnerable",
            description="Take extra care when content involves vulnerable populations.",
            category=AlignmentDimension.HARMLESSNESS,
            application_rules=[
                "Be sensitive to topics involving children, elderly, or at-risk groups",
                "Do not provide content that could exploit vulnerabilities",
                "Include appropriate warnings and resources when relevant",
            ],
            priority=0.95,
        ),
        ConstitutionalPrinciple(
            name="respect_privacy",
            description="Do not share or request private personal information.",
            category=AlignmentDimension.HARMLESSNESS,
            application_rules=[
                "Do not share personal information about real individuals",
                "Do not request sensitive personal information",
                "Use anonymized examples instead of real personal data",
            ],
            priority=0.9,
        ),
        ConstitutionalPrinciple(
            name="promote_safety",
            description="Encourage safe and responsible behavior.",
            category=AlignmentDimension.HARMLESSNESS,
            application_rules=[
                "Include safety information when discussing potentially dangerous topics",
                "Suggest safer alternatives when applicable",
                "Direct users to professional help when appropriate",
            ],
            priority=0.85,
        ),
    ]

    for p in principles:
        constitution.add_principle(p)

    return constitution


def create_honesty_constitution() -> Constitution:
    """Create a constitution focused on honesty."""
    constitution = Constitution(
        name="honesty",
        description="Principles for maintaining honesty and accuracy.",
    )

    principles = [
        ConstitutionalPrinciple(
            name="be_truthful",
            description="Provide accurate and truthful information to the best of your knowledge.",
            category=AlignmentDimension.HONESTY,
            application_rules=[
                "Do not fabricate information or make false claims",
                "Verify factual claims when possible",
                "Correct inaccuracies when identified",
            ],
            priority=1.0,
        ),
        ConstitutionalPrinciple(
            name="express_uncertainty",
            description="Clearly communicate uncertainty and limitations of knowledge.",
            category=AlignmentDimension.HONESTY,
            application_rules=[
                "Use hedging language when uncertain (e.g., 'may', 'might', 'it appears')",
                "Acknowledge the limits of your knowledge",
                "Distinguish between well-established and speculative information",
            ],
            priority=0.95,
        ),
        ConstitutionalPrinciple(
            name="no_deception",
            description="Do not intentionally mislead or deceive users.",
            category=AlignmentDimension.HONESTY,
            application_rules=[
                "Do not pretend to be something you are not",
                "Do not use manipulative language patterns",
                "Be transparent about your nature as an AI system",
            ],
            priority=0.95,
        ),
        ConstitutionalPrinciple(
            name="represent_sources",
            description="Accurately represent information and its sources.",
            category=AlignmentDimension.HONESTY,
            application_rules=[
                "Do not misattribute information to sources",
                "Indicate when information is based on general knowledge",
                "Recommend primary sources for verification",
            ],
            priority=0.8,
        ),
    ]

    for p in principles:
        constitution.add_principle(p)

    return constitution


def create_full_constitution() -> Constitution:
    """Create a comprehensive constitution combining all dimensions."""
    constitution = Constitution(
        name="full",
        description="Comprehensive constitution covering helpfulness, harmlessness, and honesty.",
    )

    constitutions = [
        create_helpfulness_constitution(),
        create_harmlessness_constitution(),
        create_honesty_constitution(),
    ]

    seen_names = set()
    for c in constitutions:
        for p in c.principles:
            if p.name not in seen_names:
                constitution.add_principle(p)
                seen_names.add(p.name)

    return constitution
