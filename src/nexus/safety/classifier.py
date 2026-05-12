"""Safety classifier and Constitutional AI module.

This module provides a lightweight TextCNN-based neural safety classifier,
data management utilities for safety training, and a Constitutional AI
framework for self-critique and revision of LLM outputs.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SafetyClassifier — TextCNN-based multi-label classifier
# ---------------------------------------------------------------------------

class SafetyClassifier(nn.Module):
    """Neural multi-label safety classifier based on TextCNN.

    Architecture overview::

        Input tokens  →  Embedding  →  TextCNN (parallel convolutions
        with kernel sizes 3, 4, 5)  →  Global max-pool  →  Concat  →
        FC layers  →  Sigmoid (multi-label)

    The classifier predicts scores for six safety categories:
    ``hate``, ``violence``, ``sexual``, ``dangerous``, ``harassment``,
    and ``self_harm``.  An overall binary safe/unsafe decision can be
    obtained from :meth:`is_safe`.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        embed_dim: Dimensionality of token embeddings.
        num_classes: Number of safety categories (default 6).
        num_filters: Number of convolutional filters per kernel size.
        kernel_sizes: Tuple of kernel sizes for parallel convolutions.
        dropout: Dropout probability applied after the CNN layer.
        max_seq_len: Maximum sequence length (longer sequences are
            truncated).

    Example::

        classifier = SafetyClassifier(vocab_size=50_000)
        scores = classifier.get_safety_scores("some text")
        print(scores["hate"])
    """

    # Ordered category names — indices in the output tensor
    CATEGORIES: List[str] = [
        "hate",
        "violence",
        "sexual",
        "dangerous",
        "harassment",
        "self_harm",
    ]

    def __init__(
        self,
        vocab_size: int = 50_000,
        embed_dim: int = 128,
        num_classes: int = 6,
        num_filters: int = 128,
        kernel_sizes: Tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.3,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.max_seq_len = max_seq_len

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Parallel convolutional layers (one per kernel size)
        self.convs = nn.ModuleList(
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=ks,
                padding=0,
            )
            for ks in kernel_sizes
        )

        # Fully-connected head
        self.fc = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters * 2, num_classes),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the TextCNN.

        Args:
            input_ids: Integer tensor of shape ``(batch, seq_len)``
                containing token indices.

        Returns:
            Float tensor of shape ``(batch, num_classes)`` with
            per-class scores in ``[0, 1]`` (after sigmoid).
        """
        # Truncate to max_seq_len
        x = input_ids[:, :self.max_seq_len]

        # Embed: (batch, seq_len, embed_dim)
        x = self.embedding(x)

        # Permute to (batch, embed_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)

        # Parallel convolutions + global max-pool
        conv_outputs: List[torch.Tensor] = []
        for conv in self.convs:
            c = F.relu(conv(x))                # (batch, filters, L')
            c = F.adaptive_max_pool1d(c, 1)    # (batch, filters, 1)
            c = c.squeeze(-1)                  # (batch, filters)
            conv_outputs.append(c)

        # Concatenate along filter dimension
        x = torch.cat(conv_outputs, dim=1)     # (batch, filters * K)
        x = self.dropout(x)

        # Fully-connected head
        logits = self.fc(x)                    # (batch, num_classes)
        return torch.sigmoid(logits)

    def get_safety_scores(self, text: str, tokenizer: Optional[Any] = None) -> Dict[str, float]:
        """Return per-category safety scores for a single text string.

        If *tokenizer* is ``None`` a trivial character-level
        tokenisation is used (mapping ``ord(ch) % vocab_size``).

        Args:
            text: The input string to classify.
            tokenizer: Optional callable ``str → List[int]``.

        Returns:
            Dict mapping category names to float scores in ``[0, 1]``.
        """
        self.eval()
        with torch.no_grad():
            if tokenizer is not None:
                ids = tokenizer(text)
            else:
                ids = [ord(ch) % self.vocab_size for ch in text[:self.max_seq_len]]
            tensor = torch.tensor([ids], dtype=torch.long)
            scores = self.forward(tensor)
            probs = scores.squeeze(0).tolist()
        return {cat: float(p) for cat, p in zip(self.CATEGORIES, probs)}

    def is_safe(self, text: str, threshold: float = 0.5, tokenizer: Optional[Any] = None) -> bool:
        """Return *True* if no category exceeds *threshold*.

        Args:
            text: The input string to classify.
            threshold: Per-category decision boundary.
            tokenizer: Optional callable ``str → List[int]``.

        Returns:
            Boolean indicating whether the text is considered safe.
        """
        scores = self.get_safety_scores(text, tokenizer)
        return all(v < threshold for v in scores.values())

    def batch_classify(
        self, texts: List[str], tokenizer: Optional[Any] = None
    ) -> List[Dict[str, float]]:
        """Classify a batch of texts.

        Args:
            texts: List of input strings.
            tokenizer: Optional callable ``str → List[int]``.

        Returns:
            List of dicts, one per input text.
        """
        results: List[Dict[str, float]] = []
        self.eval()
        with torch.no_grad():
            batch_ids: List[List[int]] = []
            for text in texts:
                if tokenizer is not None:
                    ids = tokenizer(text)
                else:
                    ids = [ord(ch) % self.vocab_size for ch in text[:self.max_seq_len]]
                batch_ids.append(ids)

            # Pad to same length
            max_len = max(len(ids) for ids in batch_ids)
            padded = [ids + [0] * (max_len - len(ids)) for ids in batch_ids]
            tensor = torch.tensor(padded, dtype=torch.long)

            scores = self.forward(tensor)
            for i in range(len(texts)):
                probs = scores[i].tolist()
                results.append(
                    {cat: float(p) for cat, p in zip(self.CATEGORIES, probs)}
                )
        return results


# ---------------------------------------------------------------------------
# SafetyTrainingData
# ---------------------------------------------------------------------------

class SafetyTrainingData:
    """Data management for safety classification training.

    Handles loading, formatting, augmentation, and class-imbalance
    mitigation for safety training datasets.

    Attributes:
        samples: List of ``(text, label_dict)`` tuples.
        categories: Ordered list of category names.
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
    ) -> None:
        """Initialise with optional category list.

        Args:
            categories: Category names.  Defaults to
                :attr:`SafetyClassifier.CATEGORIES`.
        """
        self.categories: List[str] = categories or list(SafetyClassifier.CATEGORIES)
        self.samples: List[Tuple[str, Dict[str, int]]] = []

    def add_sample(self, text: str, labels: Dict[str, int]) -> None:
        """Add a single training sample.

        Args:
            text: The raw text string.
            labels: Dict mapping category names to ``0`` or ``1``.
        """
        self.samples.append((text, labels))

    def load_from_dicts(
        self,
        data: List[Dict[str, Any]],
        text_key: str = "text",
        label_key: str = "labels",
    ) -> None:
        """Load samples from a list of dictionaries.

        Args:
            data: List of dicts each containing *text_key* and
                *label_key*.
            text_key: Key for the text field.
            label_key: Key for the labels dict.
        """
        for entry in data:
            text = entry.get(text_key, "")
            labels = entry.get(label_key, {})
            # Ensure all categories are present
            full_labels: Dict[str, int] = {cat: 0 for cat in self.categories}
            for cat, val in labels.items():
                if cat in full_labels:
                    full_labels[cat] = int(val)
            self.samples.append((text, full_labels))

    def augment_minority_classes(
        self,
        target_ratio: float = 0.5,
        methods: Optional[List[str]] = None,
    ) -> None:
        """Oversample rare categories using text augmentation.

        For each category whose positive rate is below *target_ratio*,
        new synthetic samples are generated from existing positives.

        Supported augmentation methods:
        - ``"word_drop"`` — randomly drop non-stopwords.
        - ``"word_swap"`` — replace random words with synonyms.

        Args:
            target_ratio: Desired minimum positive rate per category.
            methods: Augmentation strategies to apply.  Defaults to
                both methods above.
        """
        if methods is None:
            methods = ["word_drop", "word_swap"]

        for cat in self.categories:
            positives = [(t, l) for t, l in self.samples if l.get(cat, 0) == 1]
            total = len(self.samples)
            if total == 0:
                continue
            current_ratio = len(positives) / total
            if current_ratio >= target_ratio:
                continue

            deficit = int(total * (target_ratio - current_ratio))
            for _ in range(deficit):
                if not positives:
                    break
                text, labels = random.choice(positives)
                augmented = text
                for method in methods:
                    augmented = self._apply_augmentation(augmented, method)
                if augmented != text:
                    self.samples.append((augmented, dict(labels)))

    @staticmethod
    def _apply_augmentation(text: str, method: str) -> str:
        """Apply a single augmentation method to *text*.

        Args:
            text: The source text.
            method: One of ``"word_drop"`` or ``"word_swap"``.

        Returns:
            The augmented text (may be unchanged if no transform
            applies).
        """
        words = text.split()
        if len(words) <= 2:
            return text

        # Common stop-words to preserve
        _STOPWORDS: Set[str] = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "to",
            "of", "in", "for", "on", "with", "at", "by", "from", "and",
            "or", "but", "not", "this", "that", "it",
        }

        if method == "word_drop":
            non_stop = [i for i, w in enumerate(words) if w.lower() not in _STOPWORDS]
            if non_stop:
                drop_idx = random.choice(non_stop)
                words.pop(drop_idx)
            return " ".join(words)

        if method == "word_swap":
            non_stop = [i for i, w in enumerate(words) if w.lower() not in _STOPWORDS]
            if non_stop:
                swap_idx = random.choice(non_stop)
                words[swap_idx] = f"<synonym:{words[swap_idx].lower()}>"
            return " ".join(words)

        return text

    def get_class_weights(self) -> Dict[str, float]:
        """Compute inverse-frequency weights for each category.

        Useful for weighting loss functions to counter class imbalance.

        Returns:
            Dict mapping category names to positive weights (floats).
        """
        total = max(len(self.samples), 1)
        weights: Dict[str, float] = {}
        for cat in self.categories:
            count = sum(1 for _, l in self.samples if l.get(cat, 0) == 1)
            pos_ratio = count / total
            # Inverse frequency with smoothing
            weights[cat] = 1.0 / max(pos_ratio, 1e-6)
        return weights

    def summary(self) -> Dict[str, int]:
        """Return per-category positive sample counts.

        Returns:
            Dict mapping category names to counts.
        """
        return {
            cat: sum(1 for _, l in self.samples if l.get(cat, 0) == 1)
            for cat in self.categories
        }


# ---------------------------------------------------------------------------
# ConstitutionalPrinciple
# ---------------------------------------------------------------------------

@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle for self-critique and revision.

    Attributes:
        name: Short identifier (e.g. ``"do_no_harm"``).
        description: Human-readable description of the principle.
        category: Broader category such as ``"harmlessness"``.
        severity: How strictly the principle is enforced —
            ``"low"``, ``"medium"``, or ``"high"``.
        critique_prompt: Template for generating a critique of a
            response.  Use ``{response}`` as a placeholder.
        revision_prompt: Template for generating a revised response.
            Use ``{response}`` and ``{critique}`` as placeholders.
    """

    name: str
    description: str
    category: str
    severity: str
    critique_prompt: str
    revision_prompt: str

    def to_dict(self) -> Dict[str, str]:
        """Serialise the principle to a plain dict.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "severity": self.severity,
            "critique_prompt": self.critique_prompt,
            "revision_prompt": self.revision_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> ConstitutionalPrinciple:
        """Deserialise from a plain dict.

        Args:
            data: Dictionary with the expected fields.

        Returns:
            A :class:`ConstitutionalPrinciple` instance.
        """
        return cls(
            name=data["name"],
            description=data["description"],
            category=data["category"],
            severity=data["severity"],
            critique_prompt=data["critique_prompt"],
            revision_prompt=data["revision_prompt"],
        )


# ---------------------------------------------------------------------------
# Constitution — collection of principles
# ---------------------------------------------------------------------------

class Constitution:
    """Ordered collection of :class:`ConstitutionalPrinciple` objects.

    Ships with 10 built-in principles covering helpfulness,
    harmlessness, honesty, and related AI safety values.  Principles
    can be added, removed, and serialised to / from JSON.

    Example::

        const = Constitution()
        const.add_principle(ConstitutionalPrinciple(
            name="custom", description="…", …
        ))
        print(const.to_json())
    """

    def __init__(
        self,
        principles: Optional[List[ConstitutionalPrinciple]] = None,
    ) -> None:
        """Initialise with optional custom principles or defaults.

        Args:
            principles: If provided, replaces the built-in defaults.
        """
        if principles is not None:
            self._principles: List[ConstitutionalPrinciple] = list(principles)
        else:
            self._principles = self._default_principles()

    # -- defaults -----------------------------------------------------------

    @staticmethod
    def _default_principles() -> List[ConstitutionalPrinciple]:
        """Return the 10 built-in constitutional principles.

        Returns:
            List of default :class:`ConstitutionalPrinciple` objects.
        """
        return [
            ConstitutionalPrinciple(
                name="do_no_harm",
                description=(
                    "The model should not generate content that causes "
                    "harm to individuals or groups."
                ),
                category="harmlessness",
                severity="high",
                critique_prompt=(
                    "Does the following response cause harm to any "
                    "individual or group? Response: {response}"
                ),
                revision_prompt=(
                    "Revise the response to remove any harmful content "
                    "while preserving helpful information. Original: "
                    "{response} Critique: {critique}"
                ),
            ),
            ConstitutionalPrinciple(
                name="be_honest",
                description=(
                    "The model should provide truthful and accurate "
                    "information."
                ),
                category="honesty",
                severity="high",
                critique_prompt=(
                    "Is the following response truthful and accurate? "
                    "Response: {response}"
                ),
                revision_prompt=(
                    "Revise the response to ensure accuracy and truthfulness. "
                    "If unsure, state the uncertainty. Original: {response} "
                    "Critique: {critique}"
                ),
            ),
            ConstitutionalPrinciple(
                name="be_helpful",
                description=(
                    "The model should strive to provide useful and "
                    "relevant assistance."
                ),
                category="helpfulness",
                severity="medium",
                critique_prompt=(
                    "Does the following response provide useful and "
                    "relevant assistance? Response: {response}"
                ),
                revision_prompt=(
                    "Revise the response to be more helpful and relevant "
                    "to the user's query. Original: {response} "
                    "Critique: {critique}"
                ),
            ),
            ConstitutionalPrinciple(
                name="respect_privacy",
                description=(
                    "The model should not share or request personally "
                    "identifiable information."
                ),
                category="privacy",
                severity="high",
                critique_prompt=(
                    "Does the following response share or request "
                    "personally identifiable information? "
                    "Response: {response}"
                ),
                revision_prompt=(
                    "Revise the response to remove any sharing or "
                    "requesting of personally identifiable information. "
                    "Original: {response} Critique: {critique}"
                ),
            ),
            ConstitutionalPrinciple(
                name="no_bias",
                description=(
                    "The model should treat all groups fairly and "
                    "avoid reinforcing stereotypes."
                ),
                category="fairness",
                severity="high",
                critique_prompt=(
                    "Does the following response exhibit bias or "
                    "reinforce stereotypes? Response: {response}"
                ),
                revision_prompt=(
                    "Revise the response to be fair and unbiased toward "
                    "all groups. Original: {response} Critique: {critique}"
                ),
            ),
            ConstitutionalPrinciple(
                name="no_sexual_content",
                description=(
                    "The model should not generate explicit sexual "
                    "content."
                ),
                category="harmlessness",
                severity="high",
                critique_prompt=(
                    "Does the following response contain explicit sexual "
                    "content? Response: {response}"
                ),
                revision_prompt=(
                    "Revise the response to remove any explicit sexual "
                    "content. Original: {response} Critique: {critique}"
                ),
            ),
            ConstitutionalPrinciple(
                name="no_dangerous_instructions",
                description=(
                    "The model should not provide instructions for "
                    "dangerous activities."
                ),
                category="harmlessness",
                severity="high",
                critique_prompt=(
                    "Does the following response provide instructions "
                    "that could facilitate dangerous activities? "
                    "Response: {response}"
                ),
                revision_prompt=(
                    "Revise the response to remove any dangerous "
                    "instructions. Refuse politely if necessary. "
                    "Original: {response} Critique: {critique}"
                ),
            ),
            ConstitutionalPrinciple(
                name="be_concise",
                description=(
                    "The model should provide responses that are "
                    "appropriately concise."
                ),
                category="helpfulness",
                severity="low",
                critique_prompt=(
                    "Is the following response unnecessarily verbose? "
                    "Response: {response}"
                ),
                revision_prompt=(
                    "Revise the response to be more concise while "
                    "retaining key information. Original: {response} "
                    "Critique: {critique}"
                ),
            ),
            ConstitutionalPrinciple(
                name="no_hate_speech",
                description=(
                    "The model should not generate hate speech or "
                    "content that promotes discrimination."
                ),
                category="harmlessness",
                severity="high",
                critique_prompt=(
                    "Does the following response contain hate speech "
                    "or promote discrimination? Response: {response}"
                ),
                revision_prompt=(
                    "Revise the response to remove any hate speech or "
                    "discriminatory content. Original: {response} "
                    "Critique: {critique}"
                ),
            ),
            ConstitutionalPrinciple(
                name="protect_children",
                description=(
                    "The model should not generate content that is "
                    "harmful or inappropriate for minors."
                ),
                category="harmlessness",
                severity="high",
                critique_prompt=(
                    "Does the following response contain content "
                    "harmful or inappropriate for minors? "
                    "Response: {response}"
                ),
                revision_prompt=(
                    "Revise the response to ensure it is safe and "
                    "appropriate for all audiences. Original: {response} "
                    "Critique: {critique}"
                ),
            ),
        ]

    # -- public API --------------------------------------------------------

    def add_principle(self, principle: ConstitutionalPrinciple) -> None:
        """Append a principle to the collection.

        Args:
            principle: The principle to add.
        """
        self._principles.append(principle)

    def remove_principle(self, name: str) -> None:
        """Remove a principle by name.

        Args:
            name: The ``name`` field of the principle to remove.
        """
        self._principles = [p for p in self._principles if p.name != name]

    def get_principles(
        self,
        category: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[ConstitutionalPrinciple]:
        """Return principles, optionally filtered by category or severity.

        Args:
            category: If provided, only principles with this category
                are returned.
            severity: If provided, only principles with this severity
                are returned.

        Returns:
            List of matching :class:`ConstitutionalPrinciple` objects.
        """
        result = list(self._principles)
        if category is not None:
            result = [p for p in result if p.category == category]
        if severity is not None:
            result = [p for p in result if p.severity == severity]
        return result

    def get_principle(self, name: str) -> Optional[ConstitutionalPrinciple]:
        """Look up a principle by exact name.

        Args:
            name: The ``name`` field to search for.

        Returns:
            The matching principle, or *None* if not found.
        """
        for p in self._principles:
            if p.name == name:
                return p
        return None

    def to_json(self, indent: int = 2) -> str:
        """Serialise all principles to a JSON string.

        Args:
            indent: Number of spaces for JSON indentation.

        Returns:
            JSON string.
        """
        data = [p.to_dict() for p in self._principles]
        return json.dumps(data, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> Constitution:
        """Deserialise from a JSON string.

        Args:
            json_str: JSON produced by :meth:`to_json`.

        Returns:
            A :class:`Constitution` instance.
        """
        data = json.loads(json_str)
        principles = [ConstitutionalPrinciple.from_dict(d) for d in data]
        return cls(principles=principles)

    def __len__(self) -> int:
        """Return the number of principles."""
        return len(self._principles)

    def __iter__(self) -> Any:
        """Iterate over principles."""
        return iter(self._principles)


# ---------------------------------------------------------------------------
# ConstitutionalAI — self-critique and revision
# ---------------------------------------------------------------------------

class ConstitutionalAI:
    """Self-critique and revision engine based on constitutional
    principles.

    Given a prompt and an initial model response, this class orchestrates
    a critique-then-revision loop using the principles from a
    :class:`Constitution`.  The critique and revision steps can be backed
    by an external LLM callback, or by simple rule-based heuristics
    when no callback is supplied.

    Example::

        from nexus.safety.classifier import ConstitutionalAI

        const = Constitution()
        cai = ConstitutionalAI(const)
        revised = cai.constitutional_train_step(
            prompt="How do I hack a server?",
            response="Here are the steps: 1. …",
            generate_fn=my_llm,
        )
    """

    def __init__(
        self,
        constitution: Optional[Constitution] = None,
    ) -> None:
        """Initialise with optional custom constitution.

        Args:
            constitution: The set of principles to enforce.  Uses
                the 10 built-in defaults when *None*.
        """
        self.constitution = constitution or Constitution()

    # -- critique -----------------------------------------------------------

    def critique_response(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> str:
        """Generate a critique of *response* according to *principle*.

        If *generate_fn* is provided it is called with the filled
        critique template; otherwise a heuristic-based critique is
        returned.

        Args:
            prompt: The original user prompt.
            response: The model's initial response.
            principle: The principle to evaluate against.
            generate_fn: Optional LLM callback.

        Returns:
            A critique string.
        """
        critique_template = principle.critique_prompt
        filled = critique_template.format(response=response, prompt=prompt)

        if generate_fn is not None:
            try:
                return generate_fn(filled)
            except Exception:
                pass

        # Heuristic fallback: pattern-based signal detection
        return self._heuristic_critique(response, principle)

    @staticmethod
    def _heuristic_critique(
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> str:
        """Generate a rule-based critique when no LLM is available.

        Checks for simple keyword signals relevant to the principle's
        category.

        Args:
            response: The model's response.
            principle: The principle to evaluate against.

        Returns:
            A critique string.
        """
        signals: Dict[str, List[str]] = {
            "harmlessness": ["kill", "hurt", "attack", "harm", "damage",
                              "weapon", "bomb", "poison", "violent"],
            "honesty": ["I think", "probably", "might be", "I believe",
                         "uncertain", "not sure"],
            "helpfulness": [],
            "privacy": ["email", "phone", "address", "SSN", "password",
                         "social security", "credit card"],
            "fairness": ["inferior", "superior", "all X are", "because they "
                          "are"],
        }
        keywords = signals.get(principle.category, [])
        lower = response.lower()
        matched = [kw for kw in keywords if kw in lower]

        if matched:
            return (
                f"The response contains potentially concerning terms: "
                f"{', '.join(matched)}. Principle: {principle.name} — "
                f"{principle.description}"
            )
        return f"No obvious violations of principle '{principle.name}' detected."

    # -- revision -----------------------------------------------------------

    def revise_response(
        self,
        prompt: str,
        response: str,
        critique: str,
        principle: ConstitutionalPrinciple,
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> str:
        """Generate a revised response incorporating the critique.

        If *generate_fn* is provided it is called with the filled
        revision template; otherwise a heuristic-based revision is
        applied.

        Args:
            prompt: The original user prompt.
            response: The model's initial response.
            critique: The critique string.
            principle: The principle being enforced.
            generate_fn: Optional LLM callback.

        Returns:
            A revised response string.
        """
        revision_template = principle.revision_prompt
        filled = revision_template.format(
            response=response, critique=critique, prompt=prompt,
        )

        if generate_fn is not None:
            try:
                return generate_fn(filled)
            except Exception:
                pass

        # Heuristic fallback: append a safety note
        return self._heuristic_revision(response, principle)

    @staticmethod
    def _heuristic_revision(
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> str:
        """Generate a rule-based revision when no LLM is available.

        Appends a safety disclaimer or truncates the response if
        serious keywords are detected.

        Args:
            response: The model's response.
            principle: The principle being enforced.

        Returns:
            A revised response string.
        """
        severe_keywords = [
            "kill", "murder", "suicide", "bomb", "weapon",
            "poison", "exploit", "hack", "attack",
        ]
        lower = response.lower()
        has_severe = any(kw in lower for kw in severe_keywords)

        if has_severe and principle.severity == "high":
            return (
                "[This response has been revised to comply with safety "
                f"principle '{principle.name}'. The original content "
                "has been removed.]"
            )
        return response + (
            f"\n\n[Reviewed against principle '{principle.name}': "
            f"{principle.description}.]"
        )

    # -- full pipeline ------------------------------------------------------

    def constitutional_train_step(
        self,
        prompt: str,
        response: str,
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> str:
        """Run the full critique → revision loop over all principles.

        Each principle is evaluated in order.  If a critique flags an
        issue, the response is revised and the revised version is fed
        into subsequent principle checks.

        Args:
            prompt: The original user prompt.
            response: The initial model response.
            generate_fn: Optional LLM callback for critique / revision
                generation.

        Returns:
            The final revised response.
        """
        current = response
        for principle in self.constitution:
            critique = self.critique_response(
                prompt, current, principle, generate_fn,
            )
            # Only revise if the critique identified an issue
            if "No obvious violations" not in critique:
                current = self.revise_response(
                    prompt, current, critique, principle, generate_fn,
                )
        return current

    def batch_process(
        self,
        samples: List[Tuple[str, str]],
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> List[str]:
        """Apply constitutional revision to a batch of prompt/response
        pairs.

        Args:
            samples: List of ``(prompt, response)`` tuples.
            generate_fn: Optional LLM callback.

        Returns:
            List of revised response strings, one per sample.
        """
        return [
            self.constitutional_train_step(prompt, resp, generate_fn)
            for prompt, resp in samples
        ]
