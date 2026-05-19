"""Nexus-LLM Application Context Module.

Provides the ApplicationContext class for managing the lifecycle and
resources of the Nexus-LLM application. Handles initialization,
cleanup, and provides access to shared components.
"""

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from nexus_llm.constants import CACHE_DIR, CONFIG_DIR, LOG_DIR, MODELS_DIR
from nexus_llm.events import EventBus, get_event_bus
from nexus_llm.plugins import PluginManager
from nexus_llm.registry import GlobalRegistry
from nexus_llm.signals import SignalHandler
from nexus_llm.state import StateManager

logger = logging.getLogger(__name__)


class ApplicationContext:
    """Application context for managing shared resources and lifecycle.

    Provides centralized access to all major components of the Nexus-LLM
    application and ensures proper initialization and cleanup.

    Example:
        >>> with ApplicationContext() as ctx:
        >>>     ctx.event_bus.publish(event)
        >>>     ctx.plugin_manager.load_plugin("my_plugin")
    """

    _instance: Optional["ApplicationContext"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        config_path: Optional[str] = None,
        verbose: bool = False,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the application context.

        Args:
            config_path: Path to a configuration file.
            verbose: Enable verbose logging.
            log_level: Logging level string.
        """
        self._config_path = config_path
        self._verbose = verbose
        self._log_level = log_level
        self._initialized = False
        self._shutting_down = False

        # Core components (lazy initialization)
        self._event_bus: Optional[EventBus] = None
        self._registry: Optional[GlobalRegistry] = None
        self._plugin_manager: Optional[PluginManager] = None
        self._state_manager: Optional[StateManager] = None
        self._signal_handler: Optional[SignalHandler] = None

        # Shared resources
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "ApplicationContext":
        """Get or create the singleton application context.

        Returns:
            The global ApplicationContext instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (primarily for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None

    def initialize(self) -> None:
        """Initialize all application components.

        Sets up directories, event bus, registry, plugin manager,
        state manager, and signal handler.
        """
        if self._initialized:
            logger.warning("ApplicationContext already initialized.")
            return

        logger.info("Initializing ApplicationContext...")

        # Setup directories
        self._ensure_directories()

        # Initialize core components
        self._event_bus = get_event_bus()
        self._registry = GlobalRegistry()
        self._state_manager = StateManager()
        self._signal_handler = SignalHandler(event_bus=self._event_bus)
        self._plugin_manager = PluginManager(event_bus=self._event_bus)

        # Register shutdown callback
        self._signal_handler.register_shutdown_callback(self.shutdown)

        # Install signal handlers
        self._signal_handler.install()

        # Load configuration
        if self._config_path:
            self._load_config(self._config_path)

        # Discover plugins
        self._plugin_manager.discover_plugins()

        self._initialized = True
        logger.info("ApplicationContext initialized successfully.")

    def shutdown(self) -> None:
        """Shut down all application components gracefully.

        Unloads plugins, clears resources, and restores signal handlers.
        """
        if self._shutting_down:
            return

        self._shutting_down = True
        logger.info("Shutting down ApplicationContext...")

        # Unload all plugins
        if self._plugin_manager is not None:
            try:
                self._plugin_manager.disable_all()
                self._plugin_manager.unload_all()
            except Exception as exc:
                logger.error("Error unloading plugins: %s", exc)

        # Clear models and tokenizers
        self._models.clear()
        self._tokenizers.clear()

        # Uninstall signal handlers
        if self._signal_handler is not None:
            try:
                self._signal_handler.uninstall()
            except Exception as exc:
                logger.error("Error uninstalling signal handlers: %s", exc)

        self._initialized = False
        logger.info("ApplicationContext shut down complete.")

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for dir_path in [CACHE_DIR, CONFIG_DIR, LOG_DIR, MODELS_DIR]:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Could not create directory %s: %s", dir_path, exc)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from a file.

        Args:
            config_path: Path to the configuration file.
        """
        try:
            from nexus_llm.config_loader import ConfigLoader
            loader = ConfigLoader()
            self._config = loader.load(config_path)
            logger.info("Loaded configuration from: %s", config_path)
        except Exception as exc:
            logger.warning("Failed to load config from %s: %s", config_path, exc)

    @property
    def event_bus(self) -> EventBus:
        """Get the event bus."""
        if self._event_bus is None:
            self._event_bus = get_event_bus()
        return self._event_bus

    @property
    def registry(self) -> GlobalRegistry:
        """Get the global registry."""
        if self._registry is None:
            self._registry = GlobalRegistry()
        return self._registry

    @property
    def plugin_manager(self) -> PluginManager:
        """Get the plugin manager."""
        if self._plugin_manager is None:
            self._plugin_manager = PluginManager(event_bus=self.event_bus)
        return self._plugin_manager

    @property
    def state_manager(self) -> StateManager:
        """Get the state manager."""
        if self._state_manager is None:
            self._state_manager = StateManager()
        return self._state_manager

    @property
    def signal_handler(self) -> SignalHandler:
        """Get the signal handler."""
        if self._signal_handler is None:
            self._signal_handler = SignalHandler(event_bus=self.event_bus)
        return self._signal_handler

    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()

    @property
    def is_initialized(self) -> bool:
        """Check if the context has been initialized."""
        return self._initialized

    @property
    def is_shutting_down(self) -> bool:
        """Check if the context is shutting down."""
        return self._shutting_down

    def register_model(self, name: str, model: Any, tokenizer: Any = None) -> None:
        """Register a loaded model in the context.

        Args:
            name: Name to register the model under.
            model: The loaded model instance.
            tokenizer: Optional tokenizer instance.
        """
        self._models[name] = model
        if tokenizer is not None:
            self._tokenizers[name] = tokenizer

    def get_model(self, name: str) -> Any:
        """Get a registered model.

        Args:
            name: Name of the model.

        Returns:
            The model instance.

        Raises:
            KeyError: If the model is not registered.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in context.")
        return self._models[name]

    def get_tokenizer(self, name: str) -> Any:
        """Get a registered tokenizer.

        Args:
            name: Name of the model/tokenizer.

        Returns:
            The tokenizer instance.

        Raises:
            KeyError: If the tokenizer is not registered.
        """
        if name not in self._tokenizers:
            raise KeyError(f"Tokenizer for '{name}' not found in context.")
        return self._tokenizers[name]

    def list_models(self) -> list:
        """List all registered model names.

        Returns:
            List of model name strings.
        """
        return list(self._models.keys())

    def __enter__(self) -> "ApplicationContext":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"ApplicationContext(initialized={self._initialized}, "
            f"models={len(self._models)}, "
            f"shutting_down={self._shutting_down})"
        )


@contextmanager
def app_context(
    config_path: Optional[str] = None,
    verbose: bool = False,
    log_level: str = "INFO",
) -> Generator[ApplicationContext, None, None]:
    """Context manager for the application context.

    Provides a convenient way to manage the application lifecycle
    with proper initialization and cleanup.

    Args:
        config_path: Path to a configuration file.
        verbose: Enable verbose logging.
        log_level: Logging level string.

    Yields:
        The initialized ApplicationContext.

    Example:
        >>> with app_context() as ctx:
        >>>     ctx.plugin_manager.load_all()
    """
    ctx = ApplicationContext(
        config_path=config_path,
        verbose=verbose,
        log_level=log_level,
    )
    ctx.initialize()
    try:
        yield ctx
    finally:
        ctx.shutdown()
