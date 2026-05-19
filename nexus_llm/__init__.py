"""Nexus-LLM: A powerful LLM framework for training, serving, and chatting.

This package provides a comprehensive framework for working with Large Language
Models, including interactive chat, inference serving, fine-tuning, evaluation,
and benchmarking capabilities.

Public API Exports:
    - NexusLLMApp: Main application class for orchestrating all operations
    - NexusLLMError: Base exception for all Nexus-LLM errors
    - ModelNotFoundError: Raised when a requested model is not found
    - ModelLoadError: Raised when a model fails to load
    - InferenceError: Raised during inference failures
    - ConfigError: Raised for configuration errors
    - TrainingError: Raised during training failures
    - ServerError: Raised for server-related errors
    - ChatError: Raised during chat operations
    - Message: Dataclass for chat messages
    - Conversation: Dataclass for conversation state
    - GenerationConfig: Dataclass for generation parameters
    - ModelInfo: Dataclass for model metadata
    - TrainingConfig: Dataclass for training configuration
    - ModelType: Enum for model types
    - DeviceType: Enum for device types
    - PrecisionType: Enum for precision types
    - TaskType: Enum for task types
    - ChatRole: Enum for chat message roles
    - Registry: Component registry for models, plugins, and commands
    - EventBus: Event system for inter-component communication
    - PluginInterface: Base class for plugins
    - PluginManager: Manager for loading and managing plugins
"""

from nexus_llm.__version__ import __version__, __author__, __email__, __license__
from nexus_llm.app import NexusLLMApp
from nexus_llm.exceptions import (
    NexusLLMError,
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    TokenizerError,
    ConfigError,
    TrainingError,
    ServerError,
    ChatError,
    PluginError,
)
from nexus_llm.types import (
    Message,
    Conversation,
    GenerationConfig,
    ModelInfo,
    TrainingConfig,
    EvalConfig,
    BenchmarkConfig,
    ServerConfig,
    DownloadConfig,
    ChatConfig,
)
from nexus_llm.enums import (
    ModelType,
    DeviceType,
    PrecisionType,
    TaskType,
    ChatRole,
    MessageType,
    TrainingStage,
)
from nexus_llm.registry import Registry
from nexus_llm.events import EventBus, Event, EventHandler
from nexus_llm.plugins import PluginManager as _PkgPluginManager
from nexus_llm.plugins.hook import HookManager, HookPriority
from nexus_llm.plugins.manager import PluginManager as PluginManager
from nexus_llm.plugins.loader import PluginLoader

# Import PluginInterface from the top-level plugins module
import importlib.util
import os as _os
_plugins_py = _os.path.join(_os.path.dirname(__file__), "plugins.py")
if _os.path.exists(_plugins_py):
    # Import PluginInterface from the standalone plugins.py
    try:
        _spec = importlib.util.spec_from_file_location(
            "nexus_llm._plugins_standalone", _plugins_py
        )
        _plugins_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_plugins_mod)
        PluginInterface = _plugins_mod.PluginInterface
    except Exception:
        PluginInterface = None
else:
    PluginInterface = None
from nexus_llm.constants import APP_NAME, VERSION, DEFAULT_MODEL

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Application
    "NexusLLMApp",
    # Constants
    "APP_NAME",
    "VERSION",
    "DEFAULT_MODEL",
    # Exceptions
    "NexusLLMError",
    "ModelNotFoundError",
    "ModelLoadError",
    "InferenceError",
    "TokenizerError",
    "ConfigError",
    "TrainingError",
    "ServerError",
    "ChatError",
    "PluginError",
    # Types
    "Message",
    "Conversation",
    "GenerationConfig",
    "ModelInfo",
    "TrainingConfig",
    "EvalConfig",
    "BenchmarkConfig",
    "ServerConfig",
    "DownloadConfig",
    "ChatConfig",
    # Enums
    "ModelType",
    "DeviceType",
    "PrecisionType",
    "TaskType",
    "ChatRole",
    "MessageType",
    "TrainingStage",
    # Infrastructure
    "Registry",
    "EventBus",
    "Event",
    "EventHandler",
    "PluginInterface",
    "PluginManager",
]
