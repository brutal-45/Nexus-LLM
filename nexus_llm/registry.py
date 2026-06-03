"""Nexus-LLM Component Registry Module.

Provides a registry pattern for managing models, plugins, commands,
and other components. Supports registration, lookup, and iteration
over registered components with type safety and conflict detection.
"""

import threading
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Set, TypeVar, Union

from nexus_llm.exceptions import PluginError

T = TypeVar("T")


class RegistryEntry(Generic[T]):
    """A single entry in the registry.

    Attributes:
        name: The unique name of the registered component.
        component: The registered component instance or class.
        metadata: Additional metadata about the component.
        tags: Tags for categorizing the component.
    """

    def __init__(
        self,
        name: str,
        component: T,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
    ) -> None:
        self.name = name
        self.component = component
        self.metadata = metadata or {}
        self.tags = tags or set()

    def __repr__(self) -> str:
        return f"RegistryEntry(name={self.name!r}, type={type(self.component).__name__})"


class Registry(Generic[T]):
    """Thread-safe component registry.

    A generic registry that allows registration, lookup, and management
    of components by name. Supports tagging, metadata, and conflict
    resolution strategies.

    Example:
        >>> model_registry = Registry[ModelInfo]()
        >>> model_registry.register("gpt2", model_info)
        >>> info = model_registry.get("gpt2")
        >>> all_models = model_registry.list()
    """

    def __init__(self, name: str = "default", allow_overwrite: bool = False) -> None:
        """Initialize the registry.

        Args:
            name: A descriptive name for this registry.
            allow_overwrite: Whether to allow overwriting existing entries.
        """
        self.name = name
        self.allow_overwrite = allow_overwrite
        self._entries: Dict[str, RegistryEntry[T]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        component: T,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
    ) -> None:
        """Register a component.

        Args:
            name: Unique name for the component.
            component: The component to register.
            metadata: Optional metadata dictionary.
            tags: Optional set of tags for categorization.

        Raises:
            ValueError: If a component with the same name exists and
                       allow_overwrite is False.
        """
        with self._lock:
            if name in self._entries and not self.allow_overwrite:
                raise ValueError(
                    f"Component '{name}' is already registered in registry '{self.name}'. "
                    f"Set allow_overwrite=True to replace it."
                )
            self._entries[name] = RegistryEntry(
                name=name,
                component=component,
                metadata=metadata or {},
                tags=tags or set(),
            )

    def unregister(self, name: str) -> RegistryEntry[T]:
        """Remove a component from the registry.

        Args:
            name: Name of the component to remove.

        Returns:
            The removed RegistryEntry.

        Raises:
            KeyError: If the component is not found.
        """
        with self._lock:
            if name not in self._entries:
                raise KeyError(f"Component '{name}' not found in registry '{self.name}'.")
            return self._entries.pop(name)

    def get(self, name: str) -> T:
        """Get a registered component by name.

        Args:
            name: Name of the component to retrieve.

        Returns:
            The registered component.

        Raises:
            KeyError: If the component is not found.
        """
        with self._lock:
            if name not in self._entries:
                raise KeyError(f"Component '{name}' not found in registry '{self.name}'.")
            return self._entries[name].component

    def get_entry(self, name: str) -> RegistryEntry[T]:
        """Get a full registry entry by name.

        Args:
            name: Name of the component.

        Returns:
            The RegistryEntry including metadata and tags.

        Raises:
            KeyError: If the component is not found.
        """
        with self._lock:
            if name not in self._entries:
                raise KeyError(f"Component '{name}' not found in registry '{self.name}'.")
            return self._entries[name]

    def has(self, name: str) -> bool:
        """Check if a component is registered.

        Args:
            name: Name to check.

        Returns:
            True if the component is registered.
        """
        with self._lock:
            return name in self._entries

    def list(self) -> List[str]:
        """List all registered component names.

        Returns:
            List of registered component names.
        """
        with self._lock:
            return list(self._entries.keys())

    def list_entries(self) -> List[RegistryEntry[T]]:
        """List all registry entries.

        Returns:
            List of all RegistryEntry objects.
        """
        with self._lock:
            return list(self._entries.values())

    def list_by_tag(self, tag: str) -> List[T]:
        """List all components with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of components with the given tag.
        """
        with self._lock:
            return [
                entry.component
                for entry in self._entries.values()
                if tag in entry.tags
            ]

    def search(self, pattern: str) -> List[T]:
        """Search for components by name pattern.

        Args:
            pattern: Substring to search for in component names.

        Returns:
            List of matching components.
        """
        pattern_lower = pattern.lower()
        with self._lock:
            return [
                entry.component
                for name, entry in self._entries.items()
                if pattern_lower in name.lower()
            ]

    def size(self) -> int:
        """Get the number of registered components.

        Returns:
            Number of registered components.
        """
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Remove all registered components."""
        with self._lock:
            self._entries.clear()

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __getitem__(self, name: str) -> T:
        return self.get(name)

    def __setitem__(self, name: str, component: T) -> None:
        self.register(name, component)

    def __delitem__(self, name: str) -> None:
        self.unregister(name)

    def __len__(self) -> int:
        return self.size()

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._entries.keys()))

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, entries={self.size()})"


class GlobalRegistry:
    """Global registry manager for all Nexus-LLM component registries.

    Provides a centralized point of access for all registries in the
    application, including models, plugins, commands, and custom registries.
    """

    _instance: Optional["GlobalRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "GlobalRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._registries: Dict[str, Registry] = {}
                    cls._instance._init_default_registries()
        return cls._instance

    def _init_default_registries(self) -> None:
        """Initialize the default registries."""
        self._registries["models"] = Registry[Dict[str, Any]](name="models")
        self._registries["plugins"] = Registry[Any](name="plugins")
        self._registries["commands"] = Registry[Any](name="commands")
        self._registries["downloaders"] = Registry[Any](name="downloaders")
        self._registries["evaluators"] = Registry[Any](name="evaluators")
        self._registries["benchmarks"] = Registry[Any](name="benchmarks")
        self._registries["formatters"] = Registry[Any](name="formatters")

    def get_registry(self, name: str) -> Registry:
        """Get a registry by name.

        Args:
            name: Name of the registry.

        Returns:
            The requested Registry.

        Raises:
            KeyError: If the registry doesn't exist.
        """
        if name not in self._registries:
            raise KeyError(f"Registry '{name}' not found. Available: {list(self._registries.keys())}")
        return self._registries[name]

    def create_registry(self, name: str, allow_overwrite: bool = False) -> Registry:
        """Create a new registry.

        Args:
            name: Name for the new registry.
            allow_overwrite: Whether to allow overwriting entries.

        Returns:
            The newly created Registry.

        Raises:
            ValueError: If a registry with the same name already exists.
        """
        if name in self._registries:
            raise ValueError(f"Registry '{name}' already exists.")
        registry = Registry(name=name, allow_overwrite=allow_overwrite)
        self._registries[name] = registry
        return registry

    def list_registries(self) -> List[str]:
        """List all registry names.

        Returns:
            List of registry names.
        """
        return list(self._registries.keys())

    @property
    def models(self) -> Registry:
        """Access the models registry."""
        return self._registries["models"]

    @property
    def plugins(self) -> Registry:
        """Access the plugins registry."""
        return self._registries["plugins"]

    @property
    def commands(self) -> Registry:
        """Access the commands registry."""
        return self._registries["commands"]

    @property
    def downloaders(self) -> Registry:
        """Access the downloaders registry."""
        return self._registries["downloaders"]

    @property
    def evaluators(self) -> Registry:
        """Access the evaluators registry."""
        return self._registries["evaluators"]

    @property
    def benchmarks(self) -> Registry:
        """Access the benchmarks registry."""
        return self._registries["benchmarks"]

    @classmethod
    def reset(cls) -> None:
        """Reset the global registry (primarily for testing)."""
        cls._instance = None
