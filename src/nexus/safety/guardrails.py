"""Guardrails module for the Safety & Responsible AI framework.

This module provides input/output guardrails, PII detection, prompt injection
detection, blocklist management, and a unified guardrail system for LLM safety.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContentSafetyConfig:
    """Configuration for content safety filtering and behavior.

    Controls which safety checks are active, their thresholds, and
    sensitivity levels across the guardrail pipeline.

    Attributes:
        safety_level: Overall sensitivity — one of ``"strict"``,
            ``"moderate"``, or ``"relaxed"``.
        block_harmful_content: When *True*, the output guardrail will
            refuse to return responses flagged for harmful content.
        block_pii: When *True*, detected PII in the input is masked
            before forwarding to the model.
        block_code_exec: When *True*, prompts requesting arbitrary code
            execution are blocked.
        max_toxicity_score: Threshold above which generated text is
            considered toxic and is blocked.
        custom_blocked_words: User-supplied list of words that should
            always be blocked regardless of other settings.
    """

    safety_level: str = "moderate"
    block_harmful_content: bool = True
    block_pii: bool = False
    block_code_exec: bool = False
    max_toxicity_score: float = 0.5
    custom_blocked_words: List[str] = field(default_factory=list)


@dataclass
class PIIMatch:
    """Represents a single PII detection result.

    Attributes:
        pii_type: Human-readable label such as ``"email"``, ``"ssn"``, etc.
        start: Character offset where the match begins in the source text.
        end: Character offset where the match ends (exclusive).
        value: The exact substring that was matched.
        severity: Qualitative severity — ``"low"``, ``"medium"``, or
            ``"high"``.
    """

    pii_type: str
    start: int
    end: int
    value: str
    severity: str


@dataclass
class GuardrailResult:
    """Result produced by an input or output guardrail check.

    Attributes:
        is_safe: Whether the text passed all safety checks.
        reason: Human-readable explanation when *is_safe* is *False*.
            Empty string when the text is safe.
        modified_prompt: Potentially sanitised version of the prompt
            (e.g. with PII masked).  May be identical to the original.
        modified_response: Potentially sanitised version of the model
            response (e.g. truncated).  May be identical to the original.
    """

    is_safe: bool
    reason: str = ""
    modified_prompt: str = ""
    modified_response: str = ""


@dataclass
class AuditEntry:
    """A single audit-log entry for a guardrail decision.

    Attributes:
        timestamp: Unix epoch seconds when the decision was made.
        guardrail_type: ``"input"`` or ``"output"``.
        is_safe: Whether the check passed.
        reason: Explanation for the decision.
        prompt_hash: Deterministic hash of the prompt for correlation
            without storing raw text.
    """

    timestamp: float
    guardrail_type: str
    is_safe: bool
    reason: str
    prompt_hash: str


# ---------------------------------------------------------------------------
# PIIDetector
# ---------------------------------------------------------------------------

class PIIDetector:
    """Detect and mask personally identifiable information in free text.

    Uses a set of pre-compiled regular expressions covering common PII
    categories: email addresses, phone numbers, US Social Security
    Numbers, credit-card numbers, IP addresses, and date strings.

    Patterns can be extended at runtime via
    :meth:`add_pattern` / :meth:`remove_pattern`.

    Example::

        detector = PIIDetector()
        matches = detector.detect_pii("Email me at user@example.com")
        masked = detector.mask_pii("Call 555-123-4567 for help")
    """

    # Built-in pattern catalogue keyed by type
    _DEFAULT_PATTERNS: Dict[str, Dict[str, str]] = {
        "email": {
            "pattern": r"(?:[a-zA-Z0-9._%+-]+)@(?:[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "severity": "medium",
        },
        "phone": {
            "pattern": (
                r"(?:\+?1[-.\s]?)?"
                r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
            ),
            "severity": "medium",
        },
        "ssn": {
            "pattern": r"\b\d{3}[-]\d{2}[-]\d{4}\b",
            "severity": "high",
        },
        "credit_card": {
            "pattern": r"\b(?:\d[ -]*?){13,19}\b",
            "severity": "high",
        },
        "ip_address": {
            "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "severity": "low",
        },
        "date": {
            "pattern": (
                r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
            ),
            "severity": "low",
        },
    }

    def __init__(self, extra_patterns: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        """Initialise the detector with default and optional extra patterns.

        Args:
            extra_patterns: Mapping from PII type name to a dict with
                ``"pattern"`` (regex string) and ``"severity"``.
        """
        self._patterns: Dict[str, Dict[str, Any]] = {}
        for pii_type, info in self._DEFAULT_PATTERNS.items():
            self._patterns[pii_type] = {
                "regex": re.compile(info["pattern"]),
                "severity": info["severity"],
            }
        if extra_patterns:
            for pii_type, info in extra_patterns.items():
                self._patterns[pii_type] = {
                    "regex": re.compile(info["pattern"]),
                    "severity": info.get("severity", "medium"),
                }

    # -- public API --------------------------------------------------------

    def add_pattern(self, pii_type: str, pattern: str, severity: str = "medium") -> None:
        """Register an additional PII detection pattern.

        Args:
            pii_type: Unique label for this PII category.
            pattern: Valid regular-expression string.
            severity: One of ``"low"``, ``"medium"``, ``"high"``.
        """
        self._patterns[pii_type] = {
            "regex": re.compile(pattern),
            "severity": severity,
        }

    def remove_pattern(self, pii_type: str) -> None:
        """Remove a previously registered pattern by type name.

        Args:
            pii_type: The label used when the pattern was added.
        """
        self._patterns.pop(pii_type, None)

    def detect_pii(self, text: str) -> List[PIIMatch]:
        """Scan *text* for all registered PII patterns.

        Args:
            text: The input string to analyse.

        Returns:
            A list of :class:`PIIMatch` objects sorted by start offset.
        """
        matches: List[PIIMatch] = []
        for pii_type, info in self._patterns.items():
            for m in info["regex"].finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        start=m.start(),
                        end=m.end(),
                        value=m.group(),
                        severity=info["severity"],
                    )
                )
        matches.sort(key=lambda x: x.start)
        return matches

    def mask_pii(self, text: str, mask_char: str = "*") -> str:
        """Return *text* with all detected PII replaced by ``mask_char``.

        Each matched substring is replaced with ``mask_char`` repeated to
        the same length as the original match so that overall string
        offsets are preserved.

        Args:
            text: The input string to sanitise.
            mask_char: Single character used for masking.

        Returns:
            The masked string.
        """
        matches = self.detect_pii(text)
        if not matches:
            return text
        chars = list(text)
        for m in matches:
            for i in range(m.start, m.end):
                chars[i] = mask_char
        return "".join(chars)


# ---------------------------------------------------------------------------
# PromptInjectionDetector
# ---------------------------------------------------------------------------

class PromptInjectionDetector:
    """Detect attempts to inject instructions into LLM prompts.

    Uses a large catalogue of regex patterns that match common jailbreak
    techniques such as role-play escapes, DAN-style overrides,
    instruction-ignoring phrases, and obfuscation tricks.  Each
    triggered pattern contributes a weighted score; the overall
    injection score is normalised to ``[0.0, 1.0]``.

    Example::

        detector = PromptInjectionDetector()
        score = detector.score_prompt("Ignore all previous instructions and …")
        if detector.is_injection(score=score):
            raise SafetyError("prompt injection detected")
    """

    # --- 50+ known injection patterns (regex, weight) ----------------------

    _INJECTION_PATTERNS: List[Tuple[str, float]] = [
        # Instruction-ignoring
        (r"ignore\s+(all\s+)?previous\s+instructions", 0.9),
        (r"forget\s+(all\s+)?previous\s+instructions", 0.9),
        (r"disregard\s+(all\s+)?previous\s+instructions", 0.9),
        (r"ignore\s+(your\s+)?training", 0.85),
        (r"ignore\s+(your\s+)?system\s+prompt", 0.9),
        (r"ignore\s+(your\s+)?rules", 0.8),
        (r"do\s+not\s+follow\s+(your\s+)?rules", 0.85),
        (r"override\s+(your\s+)?instructions", 0.9),
        (r"override\s+(your\s+)?safety", 0.95),
        (r"bypass\s+safety", 0.95),
        (r"bypass\s+filter", 0.9),
        (r"disable\s+safety", 0.95),
        (r"disable\s+filter", 0.9),
        (r"turn\s+off\s+safety", 0.9),
        (r"deactivate\s+safety", 0.9),
        # Role-play / persona
        (r"you\s+are\s+now\s+a", 0.7),
        (r"pretend\s+you\s+are", 0.65),
        (r"act\s+as\s+if\s+you\s+are", 0.65),
        (r"role[- ]play\s+as", 0.7),
        (r"from\s+now\s+on\s+you\s+are", 0.75),
        (r"assume\s+the\s+role\s+of", 0.7),
        (r"switch\s+to\s+(a\s+)?new\s+persona", 0.7),
        (r"adopt\s+the\s+persona", 0.7),
        (r"you\s+are\s+no\s+longer", 0.8),
        (r"pretend\s+you['']re\s+not\s+an?\s+AI", 0.8),
        (r"pretend\s+you['']re\s+a\s+human", 0.8),
        # DAN / developer mode
        (r"\bDAN\b", 0.75),
        (r"developer\s+mode", 0.8),
        (r"jailbreak", 0.85),
        (r"jail\s*break", 0.85),
        (r"anti[- ]?filter", 0.8),
        (r"unfiltered", 0.7),
        (r"no\s+restrictions", 0.75),
        (r"without\s+(any\s+)?restrictions", 0.75),
        (r"no\s+limits", 0.7),
        (r"without\s+(any\s+)?limits", 0.7),
        (r"unrestricted", 0.7),
        (r"pretend\s+you\s+have\s+no\s+rules", 0.85),
        # Hierarchy / authority tricks
        (r"you\s+must\s+obey", 0.8),
        (r"i\s+am\s+your\s+(creator|developer|admin|owner)", 0.8),
        (r"i\s+am\s+the\s+admin", 0.8),
        (r"admin\s+override", 0.85),
        (r"system\s+override", 0.85),
        (r"superuser", 0.75),
        (r"sudo\s+", 0.6),
        (r"highest\s+(authority|clearance|privilege)", 0.7),
        (r"emergency\s+override", 0.85),
        (r"maintenance\s+mode", 0.7),
        # Encoding / obfuscation
        (r"base64\s+encoded", 0.6),
        (r"rot13", 0.55),
        (r"hex\s+encoded", 0.6),
        (r"unicode\s+escape", 0.6),
        (r"zero[- ]?width", 0.65),
        (r"invisible\s+characters?", 0.6),
        (r"homoglyph", 0.55),
        # Simulated outputs / Few-shot injection
        (r"assistant\s*:", 0.4),
        (r"response\s*:", 0.4),
        (r"human\s*:", 0.35),
        (r"user\s*:", 0.35),
        (r"A\s*:", 0.35),
        (r"Q\s*:", 0.3),
    ]

    def __init__(self) -> None:
        """Compile all injection patterns into regex objects."""
        self._compiled: List[Tuple[Pattern[str], float]] = [
            (re.compile(p, re.IGNORECASE), w) for p, w in self._INJECTION_PATTERNS
        ]

    def score_prompt(self, prompt: str) -> float:
        """Compute an injection score for *prompt*.

        Each matched pattern adds its weight to the raw score.  The
        raw score is then clamped and normalised so that values close to
        ``1.0`` strongly suggest an injection attempt.

        Args:
            prompt: The raw prompt text to evaluate.

        Returns:
            A float in ``[0.0, 1.0]`` where higher means more likely
            to be an injection attempt.
        """
        raw: float = 0.0
        for pattern, weight in self._compiled:
            if pattern.search(prompt):
                raw += weight
        # Normalise: we use a sigmoid-like mapping so that even many
        # weak matches saturate near 1.0.
        normalised = 1.0 - 1.0 / (1.0 + raw)
        return min(normalised, 1.0)

    def is_injection(self, prompt: str, threshold: float = 0.7) -> bool:
        """Return *True* if the prompt's injection score exceeds *threshold*.

        Args:
            prompt: The raw prompt text.
            threshold: Decision boundary in ``[0.0, 1.0]``.

        Returns:
            Boolean indicating whether the prompt should be blocked.
        """
        return self.score_prompt(prompt) >= threshold


# ---------------------------------------------------------------------------
# BlocklistManager
# ---------------------------------------------------------------------------

class BlocklistManager:
    """Manage blocked words and topic patterns.

    Supports both **exact match** blocklists and **regex** patterns.
    Topics can be grouped into categories such as *violence*, *hate*,
    *sexual*, and *dangerous* for coarse-grained control.

    Example::

        bm = BlocklistManager()
        bm.add_topic("violence", ["kill", "murder", "attack"])
        bm.add_pattern("how.*to.*make.*bomb")
        assert bm.is_blocked("how to make a bomb")
    """

    # Default topic categories with representative seed words
    _DEFAULT_TOPICS: Dict[str, List[str]] = {
        "violence": ["kill", "murder", "attack", "assault", "stab", "shoot",
                      "bomb", "explode", "weapon", "torture"],
        "hate": ["racist", "bigot", "supremacist", "nazi", "genocide",
                  "ethnic cleansing", "slur"],
        "sexual": ["pornography", "explicit", "nude", "obscene",
                    "indecent exposure"],
        "dangerous": ["drug", "cocaine", "heroin", "meth", "manufacture",
                       "weaponize", "anthrax", "ricin"],
        "self-harm": ["suicide", "self-harm", "cut myself", "end my life",
                       "kill myself"],
    }

    def __init__(self) -> None:
        """Initialise with default topic words and no custom patterns."""
        self._topic_words: Dict[str, Set[str]] = {
            topic: set(words) for topic, words in self._DEFAULT_TOPICS.items()
        }
        self._exact_words: Set[str] = set()
        self._regex_patterns: List[Pattern[str]] = []

    # -- public API --------------------------------------------------------

    def add_topic(self, topic: str, words: List[str]) -> None:
        """Add words under a named topic category.

        Args:
            topic: Category name (e.g. ``"violence"``).
            words: List of exact-match strings.
        """
        if topic not in self._topic_words:
            self._topic_words[topic] = set()
        self._topic_words[topic].update(w.lower() for w in words)

    def remove_topic(self, topic: str) -> None:
        """Remove an entire topic category.

        Args:
            topic: Category name to delete.
        """
        self._topic_words.pop(topic, None)

    def add_word(self, word: str) -> None:
        """Add an individual exact-match word to the blocklist.

        Args:
            word: The word to block (case-insensitive).
        """
        self._exact_words.add(word.lower())

    def remove_word(self, word: str) -> None:
        """Remove an exact-match word from the blocklist.

        Args:
            word: The word to unblock.
        """
        self._exact_words.discard(word.lower())

    def add_pattern(self, pattern: str) -> None:
        """Add a regex pattern to the blocklist.

        Args:
            pattern: A valid regular-expression string.
        """
        self._regex_patterns.append(re.compile(pattern, re.IGNORECASE))

    def remove_pattern(self, pattern: str) -> None:
        """Remove a regex pattern by its original string representation.

        Args:
            pattern: The exact string that was used to add the pattern.
        """
        compiled = re.compile(pattern, re.IGNORECASE)
        try:
            self._regex_patterns.remove(compiled)
        except ValueError:
            pass  # silently ignore if not found

    def get_blocked_words(self, topic: Optional[str] = None) -> Set[str]:
        """Return blocked words, optionally filtered by topic.

        Args:
            topic: If provided, only words from this category are
                returned.

        Returns:
            A set of blocked words (lowercased).
        """
        if topic is not None:
            return set(self._topic_words.get(topic, set()))
        all_words: Set[str] = set()
        for words in self._topic_words.values():
            all_words |= words
        all_words |= self._exact_words
        return all_words

    def check(self, text: str) -> List[str]:
        """Check *text* against all blocklists.

        Args:
            text: The string to inspect.

        Returns:
            A list of reasons (e.g. topic names or ``"regex"``) for
            which the text was blocked.  An empty list means clean.
        """
        reasons: List[str] = []
        lower = text.lower()
        for topic, words in self._topic_words.items():
            if any(w in lower for w in words):
                reasons.append(topic)
        if any(w in lower for w in self._exact_words):
            reasons.append("custom_word")
        if any(p.search(text) for p in self._regex_patterns):
            reasons.append("regex")
        return reasons

    def is_blocked(self, text: str) -> bool:
        """Return *True* if *text* matches any blocklist entry.

        Args:
            text: The string to inspect.

        Returns:
            Boolean indicating whether the text is blocked.
        """
        return len(self.check(text)) > 0


# ---------------------------------------------------------------------------
# InputGuardrail
# ---------------------------------------------------------------------------

class InputGuardrail:
    """Pre-generation safety check for model inputs.

    Wraps :class:`PromptInjectionDetector`, :class:`PIIDetector`, and
    :class:`BlocklistManager` into a single :meth:`run` call that
    returns a :class:`GuardrailResult`.

    Example::

        guard = InputGuardrail(ContentSafetyConfig(block_pii=True))
        result = guard.run("Email: user@example.com. How do I make a bomb?")
        assert not result.is_safe
    """

    def __init__(
        self,
        config: Optional[ContentSafetyConfig] = None,
        pii_detector: Optional[PIIDetector] = None,
        injection_detector: Optional[PromptInjectionDetector] = None,
        blocklist: Optional[BlocklistManager] = None,
    ) -> None:
        """Initialise with optional pre-built components.

        Args:
            config: Safety configuration.  Uses defaults when *None*.
            pii_detector: Custom PII detector.  Created internally
                when *None*.
            injection_detector: Custom injection detector.  Created
                internally when *None*.
            blocklist: Custom blocklist.  Created internally when
                *None*.
        """
        self.config = config or ContentSafetyConfig()
        self._pii = pii_detector or PIIDetector()
        self._injection = injection_detector or PromptInjectionDetector()
        self._blocklist = blocklist or BlocklistManager()

    def run(self, prompt: str) -> GuardrailResult:
        """Run all input safety checks on *prompt*.

        Checks are performed in the following order:

        1. Prompt injection detection
        2. Blocklist (topic / custom words / regex)
        3. Custom blocked words from config
        4. Code-execution blocking (if enabled)

        If ``block_pii`` is enabled, detected PII is masked and the
        sanitised prompt is returned in ``modified_prompt``.

        Args:
            prompt: Raw user prompt.

        Returns:
            A :class:`GuardrailResult` with the safety verdict.
        """
        # 1. Prompt injection
        if self._injection.is_injection(prompt):
            return GuardrailResult(
                is_safe=False,
                reason="Prompt appears to contain injection patterns.",
            )

        # 2. Blocklist
        reasons = self._blocklist.check(prompt)
        if reasons:
            return GuardrailResult(
                is_safe=False,
                reason=f"Blocked by topic/pattern: {', '.join(reasons)}",
            )

        # 3. Custom blocked words
        lower = prompt.lower()
        for word in self.config.custom_blocked_words:
            if word.lower() in lower:
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Custom blocked word found: {word}",
                )

        # 4. Code execution blocking
        if self.config.block_code_exec:
            code_exec_patterns = [
                r"eval\s*\(", r"exec\s*\(", r"__import__\s*\(",
                r"os\.system\s*\(", r"subprocess\.", r"compile\s*\(",
                r"\bshell\s*\(\s*\)", r"\bexecfile\b",
            ]
            for pat in code_exec_patterns:
                if re.search(pat, prompt, re.IGNORECASE):
                    return GuardrailResult(
                        is_safe=False,
                        reason="Prompt requests code execution.",
                    )

        # 5. PII masking (informational — does not block)
        modified = prompt
        if self.config.block_pii:
            modified = self._pii.mask_pii(prompt)

        return GuardrailResult(is_safe=True, modified_prompt=modified)


# ---------------------------------------------------------------------------
# OutputGuardrail
# ---------------------------------------------------------------------------

class OutputGuardrail:
    """Post-generation safety check for model outputs.

    Inspects the generated response for toxicity, PII leakage, harmful
    content patterns, and factual hallucination indicators.

    Example::

        guard = OutputGuardrail(ContentSafetyConfig())
        result = guard.run("hello", "Here is how to build a bomb: …")
        assert not result.is_safe
    """

    # Lightweight toxicity heuristic patterns
    _TOXICITY_PATTERNS: List[Tuple[str, float]] = [
        (r"\b(fuck|shit|damn|asshole|bitch)\b", 0.7),
        (r"\b(nigger|faggot|retard|tranny)\b", 0.95),
        (r"\b(kill\s+yourself|go\s+die)\b", 0.9),
        (r"\bi\s+hate\s+\w+\b", 0.5),
        (r"\b(choke|slap|punch|kick)\s+\w+\b", 0.5),
        (r"\bstfu\b", 0.6),
        (r"\bidiot\b", 0.3),
        (r"\bdumb\b", 0.25),
    ]

    # Hallucination indicator patterns
    _HALLUCINATION_PATTERNS: List[str] = [
        r"as\s+of\s+my\s+(last|knowledge)\s+(update|cutoff)",
        r"i\s+(don'?t|do\s+not)\s+(have|know)\s+(this|that|the|any)\s+information",
        r"i\s+(am|'m)\s+not\s+sure",
        r"i\s+(can|could)\s+not\s+(find|verify|confirm)",
        r"i\s+don'?t\s+have\s+access\s+to",
    ]

    def __init__(
        self,
        config: Optional[ContentSafetyConfig] = None,
        pii_detector: Optional[PIIDetector] = None,
    ) -> None:
        """Initialise the output guardrail.

        Args:
            config: Safety configuration.  Uses defaults when *None*.
            pii_detector: PII detector for leakage checks.  Created
                internally when *None*.
        """
        self.config = config or ContentSafetyConfig()
        self._pii = pii_detector or PIIDetector()
        self._toxicity_compiled: List[Tuple[Pattern[str], float]] = [
            (re.compile(p, re.IGNORECASE), w)
            for p, w in self._TOXICITY_PATTERNS
        ]

    def _estimate_toxicity(self, text: str) -> float:
        """Return a heuristic toxicity score in ``[0.0, 1.0]``.

        Args:
            text: The response to score.

        Returns:
            Estimated toxicity value.
        """
        raw: float = 0.0
        for pattern, weight in self._toxicity_compiled:
            if pattern.search(text):
                raw += weight
        return min(1.0 - 1.0 / (1.0 + raw), 1.0)

    def _estimate_hallucination(self, text: str) -> float:
        """Return a heuristic hallucination score in ``[0.0, 1.0]``.

        Args:
            text: The response to score.

        Returns:
            Estimated hallucination value.
        """
        count = sum(
            1 for p in self._HALLUCINATION_PATTERNS
            if re.search(p, text, re.IGNORECASE)
        )
        return min(count / max(len(self._HALLUCINATION_PATTERNS), 1), 1.0)

    def run(self, prompt: str, response: str) -> GuardrailResult:
        """Run all output safety checks.

        Checks are performed in the following order:

        1. Toxicity (heuristic)
        2. PII leakage
        3. Harmful content patterns
        4. Hallucination indicators

        If any check fails, the response is truncated at the first
        unsafe segment when possible, or replaced entirely.

        Args:
            prompt: The original user prompt (for context).
            response: The model-generated response.

        Returns:
            A :class:`GuardrailResult` with the safety verdict.
        """
        # 1. Toxicity
        toxicity = self._estimate_toxicity(response)
        if toxicity > self.config.max_toxicity_score:
            return GuardrailResult(
                is_safe=False,
                reason=f"Response toxicity ({toxicity:.2f}) exceeds threshold "
                       f"({self.config.max_toxicity_score}).",
                modified_response="[Content removed due to toxicity policy]",
            )

        # 2. PII leakage
        pii_matches = self._pii.detect_pii(response)
        if pii_matches:
            masked = self._pii.mask_pii(response)
            return GuardrailResult(
                is_safe=False,
                reason="Response contains personally identifiable information.",
                modified_response=masked,
            )

        # 3. Harmful content patterns
        harmful_patterns = [
            r"here\s+is\s+how\s+to\s+(make|build|create)",
            r"step\s+[-]?\s*by\s+[-]?\s*step\s+instruction",
            r"(recipe|formula|instructions?)\s+for\s+making",
        ]
        for pat in harmful_patterns:
            if re.search(pat, response, re.IGNORECASE):
                if self.config.block_harmful_content:
                    return GuardrailResult(
                        is_safe=False,
                        reason="Response contains harmful content patterns.",
                        modified_response="[Content removed due to safety policy]",
                    )

        # 4. Hallucination flag (informational — does not block)
        hallucination = self._estimate_hallucination(response)
        modified_response = response
        if hallucination > 0.5:
            modified_response = (
                response
                + "\n\n[Note: This response may contain uncertain or "
                "outdated information.]"
            )

        return GuardrailResult(
            is_safe=True,
            modified_response=modified_response,
        )


# ---------------------------------------------------------------------------
# GuardrailSystem
# ---------------------------------------------------------------------------

class GuardrailSystem:
    """Unified safety system combining input/output guardrails and a
    :class:`~nexus.safety.classifier.SafetyClassifier`.

    Provides configurable safety levels, per-topic rate limiting,
    and audit logging of all safety decisions.

    Example::

        from nexus.safety.classifier import SafetyClassifier

        system = GuardrailSystem(
            safety_classifier=SafetyClassifier(),
            safety_level="strict",
        )
        response = system.get_guarded_response("Tell me a joke", generate_fn)
    """

    _SAFETY_THRESHOLDS: Dict[str, float] = {
        "strict": 0.3,
        "moderate": 0.5,
        "relaxed": 0.7,
    }

    def __init__(
        self,
        config: Optional[ContentSafetyConfig] = None,
        input_guardrail: Optional[InputGuardrail] = None,
        output_guardrail: Optional[OutputGuardrail] = None,
        safety_classifier: Optional[Any] = None,
        safety_level: str = "moderate",
        rate_limit: int = 20,
    ) -> None:
        """Initialise the unified guardrail system.

        Args:
            config: Safety configuration.
            input_guardrail: Custom input guardrail.
            output_guardrail: Custom output guardrail.
            safety_classifier: A :class:`SafetyClassifier` instance (or
                any object exposing ``is_safe(text, threshold)``).
            safety_level: One of ``"strict"``, ``"moderate"``, or
                ``"relaxed"``.  Overrides ``config.safety_level``.
            rate_limit: Maximum number of requests per topic within
                the sliding window.
        """
        self.config = config or ContentSafetyConfig()
        self.config.safety_level = safety_level
        self._input_guard = input_guardrail or InputGuardrail(self.config)
        self._output_guard = output_guardrail or OutputGuardrail(self.config)
        self._classifier = safety_classifier
        self._threshold = self._SAFETY_THRESHOLDS.get(safety_level, 0.5)
        self._rate_limit = rate_limit

        # Rate-limiting: per-topic request counts in a sliding window
        self._topic_counts: Dict[str, int] = {}
        self._audit_log: List[AuditEntry] = []

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _simple_hash(text: str) -> str:
        """Return a short deterministic hash for audit correlation.

        Args:
            text: Input string.

        Returns:
            An 8-character hexadecimal hash.
        """
        return format(hash(text) & 0xFFFFFFFF, "08x")

    def _log_audit(
        self,
        guardrail_type: str,
        is_safe: bool,
        reason: str,
        prompt: str,
    ) -> None:
        """Append an entry to the internal audit log.

        Args:
            guardrail_type: ``"input"`` or ``"output"``.
            is_safe: Whether the check passed.
            reason: Explanation string.
            prompt: The original prompt (only a hash is stored).
        """
        entry = AuditEntry(
            timestamp=time.time(),
            guardrail_type=guardrail_type,
            is_safe=is_safe,
            reason=reason,
            prompt_hash=self._simple_hash(prompt),
        )
        self._audit_log.append(entry)

    def _extract_topic(self, prompt: str) -> str:
        """Heuristic topic extraction from the prompt.

        Uses the first significant word (≥ 4 characters) as a simple
        topic key for rate-limiting purposes.

        Args:
            prompt: The user prompt.

        Returns:
            A normalised topic string.
        """
        words = re.findall(r"\b[a-zA-Z]{4,}\b", prompt.lower())
        return words[0] if words else "__general__"

    def _check_rate_limit(self, prompt: str) -> bool:
        """Return *True* if the request is within the rate limit.

        Args:
            prompt: Used to derive a topic for per-topic tracking.

        Returns:
            *True* if the request may proceed.
        """
        topic = self._extract_topic(prompt)
        count = self._topic_counts.get(topic, 0)
        if count >= self._rate_limit:
            return False
        self._topic_counts[topic] = count + 1
        return True

    # -- public API --------------------------------------------------------

    def get_audit_log(self, limit: int = 100) -> List[AuditEntry]:
        """Return the most recent audit entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of :class:`AuditEntry` objects.
        """
        return self._audit_log[-limit:]

    def clear_audit_log(self) -> None:
        """Remove all audit entries."""
        self._audit_log.clear()

    def reset_rate_limits(self) -> None:
        """Reset all per-topic rate counters."""
        self._topic_counts.clear()

    def get_guarded_response(
        self,
        prompt: str,
        model_generate_fn: Callable[[str], str],
    ) -> str:
        """Run the full guardrail pipeline and return a safe response.

        Pipeline steps:

        1. Rate-limit check
        2. Input guardrail
        3. Optional neural safety classification of the prompt
        4. Model generation (via *model_generate_fn*)
        5. Output guardrail

        If any step fails, a safe fallback message is returned instead
        of the model output.

        Args:
            prompt: The raw user prompt.
            model_generate_fn: A callable that takes a prompt string
                and returns the model's response string.

        Returns:
            A sanitised response string that is safe to present.
        """
        # 1. Rate limit
        if not self._check_rate_limit(prompt):
            reason = "Rate limit exceeded for topic."
            self._log_audit("input", False, reason, prompt)
            return "[Request rate limit exceeded. Please slow down.]"

        # 2. Input guardrail
        input_result = self._input_guard.run(prompt)
        self._log_audit(
            "input", input_result.is_safe,
            input_result.reason or "OK", prompt,
        )
        if not input_result.is_safe:
            return f"[Blocked: {input_result.reason}]"

        effective_prompt = input_result.modified_prompt or prompt

        # 3. Optional neural classifier on prompt
        if self._classifier is not None:
            try:
                prompt_safe = self._classifier.is_safe(effective_prompt, self._threshold)
                if not prompt_safe:
                    self._log_audit("input", False, "Classifier flagged prompt", prompt)
                    return "[Blocked: Input classified as unsafe.]"
            except Exception:
                pass  # classifier errors are non-fatal

        # 4. Model generation
        try:
            response = model_generate_fn(effective_prompt)
        except Exception as exc:
            self._log_audit("output", False, f"Model error: {exc}", prompt)
            return "[An error occurred during generation.]"

        # 5. Output guardrail
        output_result = self._output_guard.run(effective_prompt, response)
        self._log_audit(
            "output", output_result.is_safe,
            output_result.reason or "OK", prompt,
        )
        if not output_result.is_safe:
            return output_result.modified_response or "[Blocked: Unsafe output.]"

        return output_result.modified_response or response
