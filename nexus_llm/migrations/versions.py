"""Pre-built migration versions for Nexus-LLM.

Each migration class modifies a shared context dict that represents
the application's configuration / directory state.
"""

from typing import Any, Dict

from nexus_llm.migrations.migration import Migration


class V1_InitialSetup(Migration):
    """Create default directories and base configuration."""

    version = "20250101_000001"
    description = "Initial setup: default directories and configs"

    def up(self, context: Dict[str, Any]) -> None:
        context.setdefault("directories", {})
        context["directories"].update({
            "models": "./models",
            "data": "./data",
            "logs": "./logs",
            "cache": "./cache",
        })
        context.setdefault("config", {})
        context["config"].update({
            "app_name": "Nexus-LLM",
            "version": "1.0.0",
            "debug": False,
        })

    def down(self, context: Dict[str, Any]) -> None:
        context.pop("directories", None)
        context.pop("config", None)


class V2_AddTrainingConfig(Migration):
    """Add training configuration section."""

    version = "20250102_000001"
    description = "Add training configuration"

    def up(self, context: Dict[str, Any]) -> None:
        config = context.setdefault("config", {})
        config["training"] = {
            "default_epochs": 3,
            "default_batch_size": 32,
            "default_learning_rate": 5e-5,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
        }
        context["directories"].setdefault("checkpoints", "./checkpoints")

    def down(self, context: Dict[str, Any]) -> None:
        config = context.get("config", {})
        config.pop("training", None)
        dirs = context.get("directories", {})
        dirs.pop("checkpoints", None)


class V3_AddSafetyConfig(Migration):
    """Add safety configuration section."""

    version = "20250103_000001"
    description = "Add safety configuration"

    def up(self, context: Dict[str, Any]) -> None:
        config = context.setdefault("config", {})
        config["safety"] = {
            "content_filter_enabled": True,
            "pii_filter_enabled": True,
            "toxicity_threshold": 0.7,
            "max_input_length": 4096,
            "safety_check_on_input": True,
            "safety_check_on_output": True,
        }

    def down(self, context: Dict[str, Any]) -> None:
        config = context.get("config", {})
        config.pop("safety", None)


class V4_AddMonitoringConfig(Migration):
    """Add monitoring configuration section."""

    version = "20250104_000001"
    description = "Add monitoring configuration"

    def up(self, context: Dict[str, Any]) -> None:
        config = context.setdefault("config", {})
        config["monitoring"] = {
            "metrics_enabled": True,
            "health_check_interval": 30,
            "performance_tracking": True,
            "alert_thresholds": {
                "latency_ms": 5000,
                "error_rate": 0.05,
                "memory_pct": 90,
            },
            "log_retention_days": 30,
        }

    def down(self, context: Dict[str, Any]) -> None:
        config = context.get("config", {})
        config.pop("monitoring", None)


class V5_AddRAGConfig(Migration):
    """Add RAG (Retrieval-Augmented Generation) configuration."""

    version = "20250105_000001"
    description = "Add RAG configuration"

    def up(self, context: Dict[str, Any]) -> None:
        config = context.setdefault("config", {})
        config["rag"] = {
            "enabled": True,
            "chunk_size": 512,
            "chunk_overlap": 64,
            "retriever_top_k": 5,
            "similarity_threshold": 0.7,
            "embedding_model": "all-MiniLM-L6-v2",
        }
        dirs = context.setdefault("directories", {})
        dirs["index"] = "./data/index"
        dirs["documents"] = "./data/documents"

    def down(self, context: Dict[str, Any]) -> None:
        config = context.get("config", {})
        config.pop("rag", None)
        dirs = context.get("directories", {})
        dirs.pop("index", None)
        dirs.pop("documents", None)
