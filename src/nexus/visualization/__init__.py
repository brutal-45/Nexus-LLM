"""
Nexus LLM Visualization Module
===============================

Comprehensive visualization and monitoring tools for LLM training,
model analysis, attention pattern inspection, and dataset profiling.

All implementations use Python stdlib only - no external dependencies
like matplotlib or pandas. Visualizations are text-based using Unicode
characters for terminal display.

Classes:
    TrainingTracker - Track training metrics per step and epoch
    LossVisualizer - Plot loss curves with smoothing and multi-run comparison
    ThroughputMonitor - Monitor tokens/sec, samples/sec, GPU utilization
    GradientMonitor - Track gradient norms, detect exploding/vanishing gradients
    CheckpointVisualizer - Compare checkpoints and visualize parameter drift
    DashboardFormatter - Text-based terminal dashboard using Unicode box-drawing
    ExperimentComparison - Compare multiple experiment runs
    ModelProfiler - Count parameters, estimate FLOPs and memory usage
    ArchitecturePrinter - Text-based model architecture tree
    ParameterDistributionAnalyzer - Analyze weight distributions and sparsity
    ModelDiff - Compare two model architectures and parameters
    LayerInspector - Detailed analysis of individual layers
    AttentionMapExtractor - Extract attention weights from transformer models
    AttentionPatternAnalyzer - Identify and classify attention patterns
    HeadRoleAnalyzer - Analyze head specialization and find duplicates
    AttentionEvolutionTracker - Track attention pattern changes during training
    TextAttentionRenderer - Render attention as text heatmap
    DataProfiler - Analyze dataset statistics
    QualityAnalyzer - Detect data quality issues
    DataBalanceChecker - Class/category balance analysis

Usage:
    from nexus.visualization import TrainingTracker, LossVisualizer
    from nexus.visualization import ModelProfiler, ArchitecturePrinter
    from nexus.visualization import AttentionMapExtractor, AttentionPatternAnalyzer
    from nexus.visualization import DataProfiler, QualityAnalyzer
"""

from nexus.visualization.training_dashboard import (
    TrainingTracker,
    LossVisualizer,
    ThroughputMonitor,
    GradientMonitor,
    CheckpointVisualizer,
    DashboardFormatter,
    ExperimentComparison,
)

from nexus.visualization.model_visualizer import (
    ModelProfiler,
    ArchitecturePrinter,
    ParameterDistributionAnalyzer,
    ModelDiff,
    LayerInspector,
)

from nexus.visualization.attention_visualizer import (
    AttentionMapExtractor,
    AttentionPatternAnalyzer,
    HeadRoleAnalyzer,
    AttentionEvolutionTracker,
    TextAttentionRenderer,
)

from nexus.visualization.data_visualizer import (
    DataProfiler,
    QualityAnalyzer,
    DataBalanceChecker,
)

__all__ = [
    # Training Dashboard
    "TrainingTracker",
    "LossVisualizer",
    "ThroughputMonitor",
    "GradientMonitor",
    "CheckpointVisualizer",
    "DashboardFormatter",
    "ExperimentComparison",
    # Model Visualization
    "ModelProfiler",
    "ArchitecturePrinter",
    "ParameterDistributionAnalyzer",
    "ModelDiff",
    "LayerInspector",
    # Attention Visualization
    "AttentionMapExtractor",
    "AttentionPatternAnalyzer",
    "HeadRoleAnalyzer",
    "AttentionEvolutionTracker",
    "TextAttentionRenderer",
    # Data Visualization
    "DataProfiler",
    "QualityAnalyzer",
    "DataBalanceChecker",
]

__version__ = "0.1.0"
__author__ = "Nexus LLM Team"
