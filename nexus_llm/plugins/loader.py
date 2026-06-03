"""Plugin discovery and loading for Nexus-LLM."""

import importlib
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

from nexus_llm.plugins.plugin import Plugin

logger = logging.getLogger(__name__)


class PluginLoadError(Exception):
    """Raised when a plugin cannot be loaded."""


class PluginValidationError(Exception):
    """Raised when a plugin fails validation."""


class PluginLoader:
    """Discovers, loads, and validates plugin classes from the filesystem.

    Plugins are Python modules that contain one or more concrete subclasses
    of :class:`Plugin`.  The loader scans a directory for ``.py`` files
    (and packages) and extracts the first concrete ``Plugin`` subclass it
    finds in each module.

    Example::

        loader = PluginLoader()
        paths = loader.discover_plugins("/path/to/plugins")
        for p in paths:
            plugin = loader.load_from_file(p)
            if loader.validate_plugin(plugin):
                plugin.on_load()
    """

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_plugins(self, directory: str) -> List[str]:
        """Scan *directory* for Python files that may contain plugins.

        Returns a list of absolute file paths.  Sub-directories that
        contain an ``__init__.py`` are also included (loaded as packages).

        Args:
            directory: Path to scan.

        Returns:
            List of absolute paths to candidate plugin files/packages.
        """
        dir_path = Path(directory).resolve()
        if not dir_path.is_dir():
            logger.warning("Plugin directory %s does not exist", dir_path)
            return []

        candidates: List[str] = []
        for entry in sorted(dir_path.iterdir()):
            # Skip private / dunder files
            if entry.name.startswith("_") or entry.name.startswith("."):
                continue
            # Regular .py file
            if entry.is_file() and entry.suffix == ".py":
                candidates.append(str(entry))
            # Package directory
            elif entry.is_dir() and (entry / "__init__.py").exists():
                candidates.append(str(entry / "__init__.py"))
        logger.info(
            "Discovered %d candidate plugin(s) in %s",
            len(candidates),
            dir_path,
        )
        return candidates

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_file(self, path: str) -> Plugin:
        """Load a :class:`Plugin` instance from a Python file.

        The file is imported into a unique module name so that reloading
        the same path does not hit the module cache.  The first **concrete**
        ``Plugin`` subclass found in the module is instantiated and
        returned.

        Args:
            path: Absolute or relative path to a Python file.

        Returns:
            An instance of the discovered Plugin subclass.

        Raises:
            PluginLoadError: If the file cannot be loaded or contains no
                             suitable Plugin subclass.
        """
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise PluginLoadError(f"Plugin file not found: {file_path}")

        module_name = f"nexus_plugin_{file_path.stem}_{id(file_path)}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, str(file_path))
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Cannot create module spec from {file_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            # Clean up half-loaded module
            sys.modules.pop(module_name, None)
            raise PluginLoadError(
                f"Failed to load plugin from {file_path}: {exc}"
            ) from exc

        plugin_cls = self._find_plugin_class(module)
        if plugin_cls is None:
            sys.modules.pop(module_name, None)
            raise PluginLoadError(
                f"No concrete Plugin subclass found in {file_path}"
            )

        try:
            plugin_instance = plugin_cls()
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise PluginLoadError(
                f"Failed to instantiate {plugin_cls.__name__} from {file_path}: {exc}"
            ) from exc

        logger.info("Loaded plugin %r from %s", plugin_instance.name, file_path)
        return plugin_instance

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_plugin(self, plugin: Plugin) -> bool:
        """Perform sanity checks on a loaded plugin instance.

        Returns:
            True if the plugin passes all checks.

        Raises:
            PluginValidationError: If a check fails.
        """
        errors: List[str] = []

        if not plugin.name or plugin.name == "unnamed_plugin":
            errors.append("Plugin must have a non-empty 'name' attribute")

        if not plugin.version:
            errors.append("Plugin must specify a 'version'")

        # Ensure lifecycle methods are present (they may be no-ops)
        for method_name in ("on_load", "on_unload", "register_hooks"):
            if not hasattr(plugin, method_name):
                errors.append(f"Plugin missing required method: {method_name}")

        if errors:
            msg = "; ".join(errors)
            raise PluginValidationError(f"Plugin {plugin.name!r} validation failed: {msg}")

        logger.debug("Plugin %r passed validation", plugin.name)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_plugin_class(module) -> Optional[Type[Plugin]]:
        """Return the first concrete Plugin subclass found in *module*."""
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Plugin) and obj is not Plugin and not inspect.isabstract(obj):
                return obj
        return None
