"""
Nexus LLM Reasoning Module
===========================
Production-grade reasoning framework for the Nexus LLM framework.

Provides chain-of-thought, tree-of-thought, planning, verification,
self-consistency, and evaluation capabilities for enhanced LLM reasoning.

Chain-of-Thought Components:
    ReasoningStep, ReasoningTrace, CoTReasoner, ZeroShotCoT, FewShotCoT,
    AutoCoT, StructuredCoT

Tree-of-Thought Components:
    ThoughtNode, ThoughtTree, ToTReasoner

Planning Components:
    Plan, SubGoal, Planner, HierarchicalPlanner, ReactivePlanner, PlanVerifier

Verification Components:
    SelfVerification, CrossVerification, BackwardVerification,
    ExecutionVerification, ConsensusVerifier, VerificationResult

Self-Consistency Components:
    SelfConsistencySampler, MajorityVoter, WeightedVoter,
    ClusterBasedSelector, ConsistencyScorer

Evaluation Components:
    ReasoningEvaluator, StepAccuracy, ChainConsistency,
    GoalAchievement, EfficiencyScorer

Example:
    >>> from nexus.reasoning import CoTReasoner, ToTReasoner, Planner
    >>> reasoner = CoTReasoner(max_steps=10, temperature=0.7)
    >>> trace = reasoner.generate_trace(model, prompt="What is 15 * 23?")
"""

from nexus.reasoning.chain_of_thought import (
    # Data Classes
    ReasoningStep,
    ReasoningStepType,
    ReasoningPattern,
    ReasoningTrace,
    # Chain-of-Thought Reasoners
    CoTReasoner,
    CoTReasonerConfig,
    ZeroShotCoT,
    ZeroShotCoTConfig,
    FewShotCoT,
    FewShotCoTConfig,
    FewShotExample,
    AutoCoT,
    AutoCoTConfig,
    StructuredCoT,
    StructuredCoTConfig,
    StructuredOutput,
    # Tree-of-Thought
    ThoughtNode,
    ThoughtTree,
    ThoughtTreeConfig,
    ToTReasoner,
    ToTReasonerConfig,
    SelectionMethod,
    SearchMethod,
    # Planning
    Plan,
    PlanStatus,
    SubGoal,
    SubGoalStatus,
    Planner,
    PlannerConfig,
    HierarchicalPlanner,
    HierarchicalPlanLevel,
    ReactivePlanner,
    ReactivePlannerConfig,
    PlanVerifier,
    PlanVerificationResult,
    PlanVerificationStatus,
    # Verification
    SelfVerification,
    SelfVerificationConfig,
    CrossVerification,
    CrossVerificationConfig,
    BackwardVerification,
    BackwardVerificationConfig,
    ExecutionVerification,
    ExecutionVerificationConfig,
    ConsensusVerifier,
    ConsensusVerifierConfig,
    VerificationResult,
    VerificationMethod,
    VerificationStatus,
    # Self-Consistency
    SelfConsistencySampler,
    SelfConsistencyConfig,
    MajorityVoter,
    MajorityVoterConfig,
    WeightedVoter,
    WeightedVoterConfig,
    ClusterBasedSelector,
    ClusterBasedSelectorConfig,
    ConsistencyScorer,
    ConsistencyScorerConfig,
    SolutionCluster,
    ConsistencyReport,
    # Evaluation
    ReasoningEvaluator,
    ReasoningEvaluatorConfig,
    StepAccuracy,
    StepAccuracyConfig,
    ChainConsistency,
    ChainConsistencyConfig,
    GoalAchievement,
    GoalAchievementConfig,
    EfficiencyScorer,
    EfficiencyScorerConfig,
    BenchmarkType,
    EvaluationReport,
)

__all__ = [
    # Data Classes
    "ReasoningStep",
    "ReasoningStepType",
    "ReasoningPattern",
    "ReasoningTrace",
    # Chain-of-Thought Reasoners
    "CoTReasoner",
    "CoTReasonerConfig",
    "ZeroShotCoT",
    "ZeroShotCoTConfig",
    "FewShotCoT",
    "FewShotCoTConfig",
    "FewShotExample",
    "AutoCoT",
    "AutoCoTConfig",
    "StructuredCoT",
    "StructuredCoTConfig",
    "StructuredOutput",
    # Tree-of-Thought
    "ThoughtNode",
    "ThoughtTree",
    "ThoughtTreeConfig",
    "ToTReasoner",
    "ToTReasonerConfig",
    "SelectionMethod",
    "SearchMethod",
    # Planning
    "Plan",
    "PlanStatus",
    "SubGoal",
    "SubGoalStatus",
    "Planner",
    "PlannerConfig",
    "HierarchicalPlanner",
    "HierarchicalPlanLevel",
    "ReactivePlanner",
    "ReactivePlannerConfig",
    "PlanVerifier",
    "PlanVerificationResult",
    "PlanVerificationStatus",
    # Verification
    "SelfVerification",
    "SelfVerificationConfig",
    "CrossVerification",
    "CrossVerificationConfig",
    "BackwardVerification",
    "BackwardVerificationConfig",
    "ExecutionVerification",
    "ExecutionVerificationConfig",
    "ConsensusVerifier",
    "ConsensusVerifierConfig",
    "VerificationResult",
    "VerificationMethod",
    "VerificationStatus",
    # Self-Consistency
    "SelfConsistencySampler",
    "SelfConsistencyConfig",
    "MajorityVoter",
    "MajorityVoterConfig",
    "WeightedVoter",
    "WeightedVoterConfig",
    "ClusterBasedSelector",
    "ClusterBasedSelectorConfig",
    "ConsistencyScorer",
    "ConsistencyScorerConfig",
    "SolutionCluster",
    "ConsistencyReport",
    # Evaluation
    "ReasoningEvaluator",
    "ReasoningEvaluatorConfig",
    "StepAccuracy",
    "StepAccuracyConfig",
    "ChainConsistency",
    "ChainConsistencyConfig",
    "GoalAchievement",
    "GoalAchievementConfig",
    "EfficiencyScorer",
    "EfficiencyScorerConfig",
    "BenchmarkType",
    "EvaluationReport",
]

__version__ = "2.0.0"
__author__ = "Nexus LLM Team"
