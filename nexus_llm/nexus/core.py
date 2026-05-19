"""Nexus-LLM Core Nexus Functions.

Provides the central NexusCore class that manages the lifecycle of the nexus
orchestration layer, including initialization, configuration, health checks,
and graceful shutdown of all sub-components.
"""

import logging
import threading
from typing import Any, Dict, Optional

from nexus_llm.exceptions import NexusLLMError, ConfigError

logger = logging.getLogger(__name__)

_global_nexus: Optional["NexusCore"] = None
_nexus_lock = threading.Lock()


class NexusCore:
    """Central orchestration core for the Nexus-LLM framework.

    The NexusCore manages the lifecycle of all nexus sub-components
    (engine, runtime, dispatcher, coordinator, optimizer) and provides
    a unified interface for starting, stopping, and querying the
    system state.

    Attributes:
        config: Active configuration dictionary.
        is_running: Whether the core is currently active.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._is_running = False
        self._components: Dict[str, Any] = {}
        self._lock = threading.RLock()
        logger.info("NexusCore initialized with config keys: %s", list(self._config.keys()))

    @property
    def config(self) -> Dict[str, Any]:
        """Return a copy of the active configuration."""
        return dict(self._config)

    @property
    def is_running(self) -> bool:
        """Whether the core is currently active."""
        return self._is_running

    def start(self) -> None:
        """Start all managed components.

        Raises:
            NexusLLMError: If the core is already running.
        """
        with self._lock:
            if self._is_running:
                raise NexusLLMError("NexusCore is already running", error_code="NEXUS_ALREADY_RUNNING")
            logger.info("Starting NexusCore components...")
            for name, component in self._components.items():
                if hasattr(component, "start"):
                    component.start()
                    logger.debug("Started component: %s", name)
            self._is_running = True
            logger.info("NexusCore started successfully")

    def stop(self) -> None:
        """Stop all managed components gracefully.

        Components are stopped in reverse registration order.
        """
        with self._lock:
            if not self._is_running:
                return
            logger.info("Stopping NexusCore components...")
            for name in reversed(list(self._components.keys())):
                component = self._components[name]
                if hasattr(component, "stop"):
                    try:
                        component.stop()
                        logger.debug("Stopped component: %s", name)
                    except Exception as exc:
                        logger.warning("Error stopping component %s: %s", name, exc)
            self._is_running = False
            logger.info("NexusCore stopped")

    def register_component(self, name: str, component: Any) -> None:
        """Register a sub-component with the core.

        Args:
            name: Unique name for the component.
            component: The component instance to register.

        Raises:
            ConfigError: If a component with the same name already exists.
        """
        with self._lock:
            if name in self._components:
                raise ConfigError(
                    f"Component '{name}' is already registered",
                    config_key=f"components.{name}",
                )
            self._components[name] = component
            logger.debug("Registered component: %s", name)

    def unregister_component(self, name: str) -> Optional[Any]:
        """Unregister and return a sub-component.

        Args:
            name: Name of the component to unregister.

        Returns:
            The unregistered component, or None if not found.
        """
        with self._lock:
            return self._components.pop(name, None)

    def get_component(self, name: str) -> Optional[Any]:
        """Retrieve a registered component by name.

        Args:
            name: Component name.

        Returns:
            The component instance, or None if not found.
        """
        return self._components.get(name)

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all managed components.

        Returns:
            Dictionary mapping component names to their health status.
        """
        results: Dict[str, Any] = {
            "nexus_core": "healthy" if self._is_running else "stopped",
            "components": {},
        }
        for name, component in self._components.items():
            if hasattr(component, "health_check"):
                try:
                    results["components"][name] = component.health_check()
                except Exception as exc:
                    results["components"][name] = {"status": "error", "error": str(exc)}
            else:
                results["components"][name] = {"status": "unknown"}
        return results

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Merge configuration updates into the active config.

        Args:
            updates: Dictionary of configuration updates.
        """
        with self._lock:
            self._config.update(updates)
            logger.info("Configuration updated with keys: %s", list(updates.keys()))

    def __enter__(self) -> "NexusCore":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def create_nexus(config: Optional[Dict[str, Any]] = None) -> NexusCore:
    """Create and return a new NexusCore instance.

    Also sets it as the global instance for singleton access.

    Args:
        config: Optional configuration dictionary.

    Returns:
        A new NexusCore instance.
    """
    global _global_nexus
    with _nexus_lock:
        nexus = NexusCore(config=config)
        _global_nexus = nexus
        return nexus


def get_nexus_instance() -> Optional[NexusCore]:
    """Return the global NexusCore instance if one exists.

    Returns:
        The global NexusCore, or None.
    """
    return _global_nexus


def shutdown_nexus() -> None:
    """Shut down and remove the global NexusCore instance."""
    global _global_nexus
    with _nexus_lock:
        if _global_nexus is not None:
            _global_nexus.stop()
            _global_nexus = None
