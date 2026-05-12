"""Nexus Safety & Responsible AI Framework.

This package provides a comprehensive set of tools for ensuring safe,
responsible LLM behaviour.  It is organised into two sub-modules:

* **guardrails** — Input / output guardrails, PII detection, prompt
  injection detection, blocklist management, and a unified
  :class:`GuardrailSystem`.

* **classifier** — Neural safety classification (TextCNN), training
  data management, and a Constitutional AI framework for self-critique
  and revision.

Public API
----------
The following classes and data classes are re-exported here for
convenient top-level imports:

.. autoclass:: ContentSafetyConfig
.. autoclass:: PIIMatch
.. autoclass:: GuardrailResult
.. autoclass:: AuditEntry
.. autoclass:: PIIDetector
.. autoclass:: PromptInjectionDetector
.. autoclass:: InputGuardrail
.. autoclass:: OutputGuardrail
.. autoclass:: GuardrailSystem
.. autoclass:: BlocklistManager
.. autoclass:: SafetyClassifier
.. autoclass:: SafetyTrainingData
.. autoclass:: ConstitutionalPrinciple
.. autoclass:: Constitution
.. autoclass:: ConstitutionalAI
"""

from __future__ import annotations

# -- guardrails.py exports -------------------------------------------------
from nexus.safety.guardrails import (
    AuditEntry,
    BlocklistManager,
    ContentSafetyConfig,
    GuardrailResult,
    GuardrailSystem,
    InputGuardrail,
    OutputGuardrail,
    PIIDetector,
    PIIMatch,
    PromptInjectionDetector,
)

# -- classifier.py exports -------------------------------------------------
from nexus.safety.classifier import (
    ConstitutionalAI,
    ConstitutionalPrinciple,
    Constitution,
    SafetyClassifier,
    SafetyTrainingData,
)

__all__: list[str] = [
    # Data classes
    "ContentSafetyConfig",
    "PIIMatch",
    "GuardrailResult",
    "AuditEntry",
    # Guardrails
    "PIIDetector",
    "PromptInjectionDetector",
    "InputGuardrail",
    "OutputGuardrail",
    "GuardrailSystem",
    "BlocklistManager",
    # Classifier & Constitutional AI
    "SafetyClassifier",
    "SafetyTrainingData",
    "ConstitutionalPrinciple",
    "Constitution",
    "ConstitutionalAI",
]
