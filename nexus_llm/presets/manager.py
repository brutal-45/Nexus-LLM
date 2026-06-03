"""Preset manager for Nexus-LLM.

Provides CRUD operations, import/export, and category filtering
for user-defined and built-in presets.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from nexus_llm.presets.preset import Preset
from nexus_llm.presets.library import PresetLibrary
from nexus_llm.presets.categories import PresetCategory

logger = logging.getLogger(__name__)


class PresetManager:
    """Manage user and built-in presets.

    The manager wraps a :class:`PresetLibrary` for built-in presets and
    maintains a separate dict for user-defined presets.  It supports
    loading, saving, listing, deleting, and importing/exporting presets.

    Example::

        mgr = PresetManager()
        p = mgr.load_preset("chat_assistant")
        mgr.save_preset("my_custom", Preset(name="my_custom", ...))
        mgr.export_preset("my_custom", "./my_custom.json")
    """

    def __init__(self, user_presets_dir: Optional[str] = None) -> None:
        self._library = PresetLibrary()
        self._user_presets: Dict[str, Preset] = {}
        self._user_presets_dir = user_presets_dir or os.path.join(
            os.path.expanduser("~"), ".nexus_llm", "presets"
        )

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load_preset(self, name: str) -> Preset:
        """Load a preset by name.

        User presets take precedence over built-in presets.

        Args:
            name: Preset name.

        Returns:
            The matching :class:`Preset`.

        Raises:
            KeyError: If no preset with the given name exists.
        """
        if name in self._user_presets:
            return self._user_presets[name]
        return self._library.get(name)

    def save_preset(self, name: str, preset: Preset) -> None:
        """Save a user-defined preset.

        If a preset with the same name already exists among user
        presets, it is overwritten.

        Args:
            name: Preset name (also stored in ``preset.name``).
            preset: The :class:`Preset` to save.

        Raises:
            ValueError: If the preset fails validation.
        """
        preset.name = name
        preset.validate()
        self._user_presets[name] = preset
        logger.info("Saved user preset %s", name)

    # ------------------------------------------------------------------
    # List / Delete
    # ------------------------------------------------------------------

    def list_presets(
        self, category: Optional[str] = None
    ) -> List[Preset]:
        """List all presets, optionally filtered by category.

        Built-in presets are listed first, followed by user presets.

        Args:
            category: If provided, only return presets in this category.

        Returns:
            List of :class:`Preset` objects.
        """
        builtins = self._library.list_presets(category=category)
        user = list(self._user_presets.values())
        if category is not None:
            user = [p for p in user if p.category == category]
        return builtins + user

    def delete_preset(self, name: str) -> None:
        """Delete a user-defined preset.

        Built-in presets cannot be deleted.

        Args:
            name: Preset name.

        Raises:
            KeyError: If the preset is not a user-defined preset.
        """
        if name not in self._user_presets:
            raise KeyError(
                f"User preset {name!r} not found; "
                f"built-in presets cannot be deleted."
            )
        del self._user_presets[name]
        logger.info("Deleted user preset %s", name)

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def import_preset(self, path: str) -> Preset:
        """Import a preset from a JSON file.

        Args:
            path: Filesystem path to the JSON preset file.

        Returns:
            The imported :class:`Preset`.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid JSON or fails validation.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Preset file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        preset = Preset.from_dict(data)
        preset.validate()
        self._user_presets[preset.name] = preset
        logger.info("Imported preset %s from %s", preset.name, path)
        return preset

    def export_preset(self, name: str, path: str) -> None:
        """Export a preset to a JSON file.

        Args:
            name: Preset name to export.
            path: Destination filesystem path.

        Raises:
            KeyError: If no preset with the given name exists.
        """
        preset = self.load_preset(name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(preset.to_dict(), fh, indent=2)
        logger.info("Exported preset %s to %s", name, path)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<PresetManager builtins={len(self._library.names)} "
            f"user={len(self._user_presets)}>"
        )
