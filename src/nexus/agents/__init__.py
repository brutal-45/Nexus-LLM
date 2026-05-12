"""
Nexus LLM Agents Module
=======================

Production-grade agent framework for building intelligent LLM-powered agents
with tool use, multi-agent coordination, planning, memory, and evaluation.

This module provides a comprehensive toolkit for constructing autonomous agents
capable of complex reasoning, tool integration, multi-agent collaboration,
and self-improvement through reflection and evaluation.

Core Architecture
-----------------
- ``agent_config``: Configuration dataclasses for all agent components
- ``agent_framework``: Base agent classes and the core think-plan-act loop
- ``tool_use``: Tool definition, execution, selection, and built-in tools
- ``multi_agent``: Multi-agent orchestration, debate, collaboration, hierarchy
- ``planning_agent``: Advanced planning with graph-based plan representation
- ``memory_agent``: Multi-tier memory (short-term, long-term, episodic, semantic)
- ``code_agent``: Software engineering agent with code analysis/generation
- ``agent_eval``: Comprehensive evaluation suite and benchmarks
- ``agent_utils``: Shared utilities (prompts, parsing, retry, rate limiting)

Quick Start
-----------
>>> from nexus.agents import AgentConfig, TaskAgent, ToolDefinition
>>> config = AgentConfig(name="assistant", role="helpful assistant")
>>> agent = TaskAgent(config)
>>> result = agent.run("What is the capital of France?")
>>> print(result.content)

>>> # Multi-agent example
>>> from nexus.agents import AgentOrchestrator, BroadcastProtocol
>>> orchestrator = AgentOrchestrator(protocol=BroadcastProtocol())
>>> orchestrator.register_agent(researcher_agent)
>>> orchestrator.register_agent(writer_agent)
>>> result = orchestrator.run_pipeline("Research and write about quantum computing")
"""

from nexus.agents.agent_config import (
    AgentConfig,
    AgentMemoryConfig,
    MultiAgentConfig,
    ReasoningStrategy,
    TaskAllocationStrategy,
    ToolConfig,
    CommunicationProtocolType,
    MemoryRetrievalStrategy,
)
from nexus.agents.agent_framework import (
    AgentState,
    BaseAgent,
    ConversationalAgent,
    Message,
    ReflectiveAgent,
    TaskAgent,
)
from nexus.agents.tool_use import (
    BuiltinTools,
    Calculator,
    CodeExecutor,
    DateTimeTool,
    FileManager,
    SystemInfoTool,
    ToolChain,
    ToolDefinition,
    ToolExecutor,
    ToolResult,
    ToolSelector,
    ToolUsePolicy,
    WebSearch,
)
from nexus.agents.multi_agent import (
    AgentMessage,
    AgentOrchestrator,
    BlackboardProtocol,
    BroadcastProtocol,
    CollaborativeAgent,
    CommunicationProtocol,
    DebateAgent,
    DirectProtocol,
    EnsembleAgent,
    HierarchicalAgent,
    MessageType,
)
from nexus.agents.planning_agent import (
    ConditionalBranch,
    ParallelPlanGroup,
    PlanExecutor,
    PlanNode,
    PlanOptimizationResult,
    PlanOptimizer,
    PlanRepresentation,
    PlanStatus,
    PlanningAgent,
    PlanStep,
    ReplanningStrategy,
    ReplanningTrigger,
)
from nexus.agents.memory_agent import (
    AgentMemory,
    EpisodicMemory,
    LongTermMemory,
    MemoryConsolidationPolicy,
    MemoryEntry,
    MemoryManager,
    MemoryRetriever,
    MemoryVisualizer,
    MemoryVisualizerConfig,
    SemanticMemory,
    ShortTermMemory,
    WorkingMemory,
)
from nexus.agents.code_agent import (
    CodeAgent,
    CodeContext,
    CodeQualityAnalyzer,
    CodeQualityReport,
    ComplexityMetric,
    ComplexityReport,
    TestCase,
    TestCaseGenerator,
    TestCaseResult,
    TestSuite,
)
from nexus.agents.agent_eval import (
    AgentBenchmark,
    AgentBenchmarkResult,
    AgentEvaluator,
    AgentEvaluationReport,
    BenchmarkCategory,
    BenchmarkDataset,
    BenchmarkMetric,
    BenchmarkResult,
    LeaderboardEntry,
    LeaderboardTracker,
    MemoryAccuracy,
    MultiAgentCoordination,
    PlanningQuality,
    SafetyCompliance,
    TaskCompletionMetric,
    ToolUseEfficiency,
)
from nexus.agents.agent_utils import (
    CircuitBreaker,
    CostEstimator,
    FormatConverter,
    LoggingUtils,
    OutputParser,
    PromptBuilder,
    RateLimiter,
    RetryHandler,
    RetryResult,
    TimeTracker,
    TokenCounter,
    TokenUsageRecord,
    ValidationUtils,
)

__all__ = [
    # --- agent_config exports ---
    "AgentConfig",
    "AgentMemoryConfig",
    "MultiAgentConfig",
    "ReasoningStrategy",
    "TaskAllocationStrategy",
    "ToolConfig",
    "CommunicationProtocolType",
    "MemoryRetrievalStrategy",
    # --- agent_framework exports ---
    "Message",
    "AgentState",
    "BaseAgent",
    "ConversationalAgent",
    "TaskAgent",
    "ReflectiveAgent",
    # --- tool_use exports ---
    "ToolDefinition",
    "ToolResult",
    "ToolUsePolicy",
    "ToolExecutor",
    "ToolSelector",
    "ToolChain",
    "BuiltinTools",
    "FileManager",
    "Calculator",
    "CodeExecutor",
    "WebSearch",
    "DateTimeTool",
    "SystemInfoTool",
    # --- multi_agent exports ---
    "AgentMessage",
    "MessageType",
    "CommunicationProtocol",
    "BroadcastProtocol",
    "DirectProtocol",
    "BlackboardProtocol",
    "AgentOrchestrator",
    "DebateAgent",
    "CollaborativeAgent",
    "HierarchicalAgent",
    "EnsembleAgent",
    # --- planning_agent exports ---
    "PlanningAgent",
    "PlanRepresentation",
    "PlanNode",
    "PlanStep",
    "PlanStatus",
    "PlanOptimizationResult",
    "PlanOptimizer",
    "PlanExecutor",
    "ReplanningStrategy",
    "ReplanningTrigger",
    "ConditionalBranch",
    "ParallelPlanGroup",
    # --- memory_agent exports ---
    "AgentMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "WorkingMemory",
    "MemoryManager",
    "MemoryRetriever",
    "MemoryVisualizer",
    "MemoryVisualizerConfig",
    "MemoryEntry",
    "MemoryConsolidationPolicy",
    # --- code_agent exports ---
    "CodeAgent",
    "CodeContext",
    "CodeQualityAnalyzer",
    "CodeQualityReport",
    "ComplexityMetric",
    "ComplexityReport",
    "TestCaseGenerator",
    "TestCase",
    "TestCaseResult",
    "TestSuite",
    # --- agent_eval exports ---
    "AgentBenchmark",
    "AgentBenchmarkResult",
    "AgentEvaluator",
    "AgentEvaluationReport",
    "BenchmarkCategory",
    "BenchmarkDataset",
    "BenchmarkMetric",
    "BenchmarkResult",
    "LeaderboardEntry",
    "LeaderboardTracker",
    "MemoryAccuracy",
    "MultiAgentCoordination",
    "PlanningQuality",
    "SafetyCompliance",
    "TaskCompletionMetric",
    "ToolUseEfficiency",
    # --- agent_utils exports ---
    "PromptBuilder",
    "OutputParser",
    "TokenCounter",
    "TokenUsageRecord",
    "RetryHandler",
    "RetryResult",
    "CircuitBreaker",
    "RateLimiter",
    "TimeTracker",
    "CostEstimator",
    "FormatConverter",
    "ValidationUtils",
    "LoggingUtils",
]

__version__ = "1.0.0"
__author__ = "Nexus LLM Team"
