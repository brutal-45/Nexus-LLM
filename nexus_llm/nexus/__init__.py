"""Nexus-LLM Nexus Module.

The nexus subpackage provides the core orchestration layer that ties together
engine execution, runtime environments, request dispatching, task coordination,
performance optimization, text analysis, transformation, and response composition.
"""

from nexus_llm.nexus.core import (
    NexusCore,
    create_nexus,
    get_nexus_instance,
    shutdown_nexus,
)
from nexus_llm.nexus.engine import Engine, EngineState
from nexus_llm.nexus.runtime import Runtime, RuntimeConfig
from nexus_llm.nexus.dispatcher import Dispatcher, DispatchResult
from nexus_llm.nexus.coordinator import Coordinator, TaskPriority
from nexus_llm.nexus.optimizer import Optimizer, OptimizationLevel
from nexus_llm.nexus.analyzer import Analyzer, AnalysisResult
from nexus_llm.nexus.transformer import Transformer, TransformResult
from nexus_llm.nexus.composer import Composer, CompositionResult

__all__ = [
    "NexusCore",
    "create_nexus",
    "get_nexus_instance",
    "shutdown_nexus",
    "Engine",
    "EngineState",
    "Runtime",
    "RuntimeConfig",
    "Dispatcher",
    "DispatchResult",
    "Coordinator",
    "TaskPriority",
    "Optimizer",
    "OptimizationLevel",
    "Analyzer",
    "AnalysisResult",
    "Transformer",
    "TransformResult",
    "Composer",
    "CompositionResult",
]
