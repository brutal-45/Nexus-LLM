"""
Nexus-LLM Terminal Themes Module

Provides a theme management system with built-in themes (dark, light,
monokai, solarized, dracula, nord) and support for custom themes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from rich.theme import Theme as RichTheme
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


DEFAULT_THEMES_DIR = os.path.expanduser("~/.nexus_llm/themes")


@dataclass
class ThemeColors:
    """Color definitions for a terminal theme."""
    # Text colors
    primary: str = "#ffffff"
    secondary: str = "#aaaaaa"
    muted: str = "#666666"
    accent: str = "#00aaff"

    # Semantic colors
    success: str = "#00ff00"
    warning: str = "#ffaa00"
    error: str = "#ff0000"
    info: str = "#00aaff"

    # Chat role colors
    user_color: str = "#00aaff"
    assistant_color: str = "#00ff88"
    system_color: str = "#ffaa00"

    # UI element colors
    border: str = "#444444"
    panel_bg: str = "#1a1a2e"
    panel_border: str = "#00aaff"
    table_header: str = "#00aaff"
    table_border: str = "#333333"
    table_row_alt: str = "#1a1a2e"

    # Syntax highlighting colors
    keyword: str = "#ff79c6"
    string: str = "#f1fa8c"
    comment: str = "#6272a4"
    number: str = "#bd93f9"
    function: str = "#50fa7b"
    class_name: str = "#8be9fd"
    decorator: str = "#ffb86c"
    operator: str = "#ff79c6"
    variable: str = "#f8f8f2"
    constant: str = "#bd93f9"

    # Background
    background: str = "#0d0d1a"
    surface: str = "#1a1a2e"

    def to_dict(self) -> dict[str, str]:
        """Serialize to a dictionary."""
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "muted": self.muted,
            "accent": self.accent,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
            "info": self.info,
            "user_color": self.user_color,
            "assistant_color": self.assistant_color,
            "system_color": self.system_color,
            "border": self.border,
            "panel_bg": self.panel_bg,
            "panel_border": self.panel_border,
            "table_header": self.table_header,
            "table_border": self.table_border,
            "table_row_alt": self.table_row_alt,
            "keyword": self.keyword,
            "string": self.string,
            "comment": self.comment,
            "number": self.number,
            "function": self.function,
            "class_name": self.class_name,
            "decorator": self.decorator,
            "operator": self.operator,
            "variable": self.variable,
            "constant": self.constant,
            "background": self.background,
            "surface": self.surface,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> ThemeColors:
        """Deserialize from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Theme:
    """A complete terminal theme with colors and metadata."""
    name: str
    display_name: str = ""
    description: str = ""
    colors: ThemeColors = field(default_factory=ThemeColors)
    is_dark: bool = True
    author: str = ""
    version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "colors": self.colors.to_dict(),
            "is_dark": self.is_dark,
            "author": self.author,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Theme:
        """Deserialize from a dictionary."""
        colors_data = data.get("colors", {})
        return cls(
            name=data.get("name", "custom"),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            colors=ThemeColors.from_dict(colors_data),
            is_dark=data.get("is_dark", True),
            author=data.get("author", ""),
            version=data.get("version", "1.0"),
        )

    def to_rich_theme(self) -> Any:
        """Convert to a Rich Theme object for use with Rich console.

        Returns:
            A Rich Theme instance, or None if Rich is unavailable.
        """
        if not HAS_RICH:
            return None
        c = self.colors
        return RichTheme({
            "primary": c.primary,
            "secondary": c.secondary,
            "muted": c.muted,
            "accent": c.accent,
            "success": c.success,
            "warning": c.warning,
            "error": c.error,
            "info": c.info,
            "user": c.user_color,
            "assistant": c.assistant_color,
            "system": c.system_color,
            "border": c.border,
            "panel": c.panel_border,
            "table.header": c.table_header,
            "table.border": c.table_border,
        })


def _make_dark_theme() -> Theme:
    """Create the built-in dark theme."""
    return Theme(
        name="dark",
        display_name="Dark",
        description="Default dark theme with blue accents",
        is_dark=True,
        author="Nexus-LLM",
        colors=ThemeColors(
            primary="#e0e0e0",
            secondary="#a0a0a0",
            muted="#606060",
            accent="#4488ff",
            success="#44ff88",
            warning="#ffaa44",
            error="#ff4444",
            info="#4488ff",
            user_color="#4488ff",
            assistant_color="#44ff88",
            system_color="#ffaa44",
            border="#333344",
            panel_bg="#1a1a2e",
            panel_border="#4488ff",
            table_header="#4488ff",
            table_border="#333344",
            table_row_alt="#1e1e32",
            keyword="#ff79c6",
            string="#f1fa8c",
            comment="#6272a4",
            number="#bd93f9",
            function="#50fa7b",
            class_name="#8be9fd",
            decorator="#ffb86c",
            operator="#ff79c6",
            variable="#f8f8f2",
            constant="#bd93f9",
            background="#0d0d1a",
            surface="#1a1a2e",
        ),
    )


def _make_light_theme() -> Theme:
    """Create the built-in light theme."""
    return Theme(
        name="light",
        display_name="Light",
        description="Clean light theme for bright environments",
        is_dark=False,
        author="Nexus-LLM",
        colors=ThemeColors(
            primary="#1a1a2e",
            secondary="#444444",
            muted="#888888",
            accent="#0055aa",
            success="#008844",
            warning="#aa6600",
            error="#cc0000",
            info="#0055aa",
            user_color="#0055aa",
            assistant_color="#008844",
            system_color="#aa6600",
            border="#cccccc",
            panel_bg="#f5f5f5",
            panel_border="#0055aa",
            table_header="#0055aa",
            table_border="#cccccc",
            table_row_alt="#f0f0f0",
            keyword="#a626a4",
            string="#50a14f",
            comment="#a0a1a7",
            number="#986801",
            function="#4078f2",
            class_name="#c18401",
            decorator="#986801",
            operator="#a626a4",
            variable="#383a42",
            constant="#986801",
            background="#ffffff",
            surface="#f5f5f5",
        ),
    )


def _make_monokai_theme() -> Theme:
    """Create the built-in Monokai theme."""
    return Theme(
        name="monokai",
        display_name="Monokai",
        description="Classic Monokai color scheme",
        is_dark=True,
        author="Wimer Hazenberg",
        colors=ThemeColors(
            primary="#f8f8f2",
            secondary="#a6a6a6",
            muted="#75715e",
            accent="#a6e22e",
            success="#a6e22e",
            warning="#e6db74",
            error="#f92672",
            info="#66d9ef",
            user_color="#66d9ef",
            assistant_color="#a6e22e",
            system_color="#e6db74",
            border="#49483e",
            panel_bg="#272822",
            panel_border="#a6e22e",
            table_header="#a6e22e",
            table_border="#49483e",
            table_row_alt="#2d2e27",
            keyword="#f92672",
            string="#e6db74",
            comment="#75715e",
            number="#ae81ff",
            function="#a6e22e",
            class_name="#66d9ef",
            decorator="#fd971f",
            operator="#f92672",
            variable="#f8f8f2",
            constant="#ae81ff",
            background="#272822",
            surface="#3e3d32",
        ),
    )


def _make_solarized_theme() -> Theme:
    """Create the built-in Solarized Dark theme."""
    return Theme(
        name="solarized",
        display_name="Solarized Dark",
        description="Ethan Schoonover's Solarized Dark",
        is_dark=True,
        author="Ethan Schoonover",
        colors=ThemeColors(
            primary="#839496",
            secondary="#657b83",
            muted="#586e75",
            accent="#268bd2",
            success="#859900",
            warning="#b58900",
            error="#dc322f",
            info="#268bd2",
            user_color="#268bd2",
            assistant_color="#859900",
            system_color="#b58900",
            border="#073642",
            panel_bg="#002b36",
            panel_border="#268bd2",
            table_header="#268bd2",
            table_border="#073642",
            table_row_alt="#073642",
            keyword="#859900",
            string="#2aa198",
            comment="#586e75",
            number="#d33682",
            function="#268bd2",
            class_name="#b58900",
            decorator="#cb4b16",
            operator="#859900",
            variable="#839496",
            constant="#d33682",
            background="#002b36",
            surface="#073642",
        ),
    )


def _make_dracula_theme() -> Theme:
    """Create the built-in Dracula theme."""
    return Theme(
        name="dracula",
        display_name="Dracula",
        description="Dracula color scheme by Zeno Rocha",
        is_dark=True,
        author="Zeno Rocha",
        colors=ThemeColors(
            primary="#f8f8f2",
            secondary="#bd93f9",
            muted="#6272a4",
            accent="#bd93f9",
            success="#50fa7b",
            warning="#f1fa8c",
            error="#ff5555",
            info="#8be9fd",
            user_color="#8be9fd",
            assistant_color="#50fa7b",
            system_color="#f1fa8c",
            border="#44475a",
            panel_bg="#282a36",
            panel_border="#bd93f9",
            table_header="#bd93f9",
            table_border="#44475a",
            table_row_alt="#2c2e3a",
            keyword="#ff79c6",
            string="#f1fa8c",
            comment="#6272a4",
            number="#bd93f9",
            function="#50fa7b",
            class_name="#8be9fd",
            decorator="#ffb86c",
            operator="#ff79c6",
            variable="#f8f8f2",
            constant="#bd93f9",
            background="#282a36",
            surface="#44475a",
        ),
    )


def _make_nord_theme() -> Theme:
    """Create the built-in Nord theme."""
    return Theme(
        name="nord",
        display_name="Nord",
        description="Nord color scheme by Arctic Ice Studio",
        is_dark=True,
        author="Arctic Ice Studio",
        colors=ThemeColors(
            primary="#d8dee9",
            secondary="#81a1c1",
            muted="#4c566a",
            accent="#88c0d0",
            success="#a3be8c",
            warning="#ebcb8b",
            error="#bf616a",
            info="#88c0d0",
            user_color="#88c0d0",
            assistant_color="#a3be8c",
            system_color="#ebcb8b",
            border="#3b4252",
            panel_bg="#2e3440",
            panel_border="#88c0d0",
            table_header="#88c0d0",
            table_border="#3b4252",
            table_row_alt="#3b4252",
            keyword="#81a1c1",
            string="#a3be8c",
            comment="#4c566a",
            number="#b48ead",
            function="#88c0d0",
            class_name="#8fbcbb",
            decorator="#ebcb8b",
            operator="#81a1c1",
            variable="#d8dee9",
            constant="#b48ead",
            background="#2e3440",
            surface="#3b4252",
        ),
    )


class ThemeManager:
    """Manages terminal themes with loading, switching, and customization.

    Supports built-in themes, custom themes from JSON files, and runtime
    theme modification. Integrates with Rich console theming.
    """

    def __init__(self, themes_dir: str | None = None) -> None:
        self._themes: dict[str, Theme] = {}
        self._current_theme_name: str = "dark"
        self._themes_dir = Path(themes_dir or DEFAULT_THEMES_DIR)

        # Register built-in themes
        self._register_builtins()
        # Load custom themes from disk
        self._load_custom_themes()

    def _register_builtins(self) -> None:
        """Register all built-in themes."""
        builtins = [
            _make_dark_theme(),
            _make_light_theme(),
            _make_monokai_theme(),
            _make_solarized_theme(),
            _make_dracula_theme(),
            _make_nord_theme(),
        ]
        for theme in builtins:
            self._themes[theme.name] = theme

    def _load_custom_themes(self) -> None:
        """Load custom themes from the themes directory."""
        if not self._themes_dir.exists():
            return
        for theme_file in self._themes_dir.glob("*.json"):
            try:
                with open(theme_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                theme = Theme.from_dict(data)
                self._themes[theme.name] = theme
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    @property
    def current_theme(self) -> Theme:
        """Get the currently active theme."""
        return self._themes.get(self._current_theme_name, self._themes["dark"])

    @property
    def current_theme_name(self) -> str:
        """Get the name of the current theme."""
        return self._current_theme_name

    def get_theme(self, name: str) -> Theme | None:
        """Get a theme by name.

        Args:
            name: Theme name.

        Returns:
            The Theme object, or None if not found.
        """
        return self._themes.get(name)

    def list_themes(self) -> list[Theme]:
        """List all available themes."""
        return list(self._themes.values())

    def set_theme(self, name: str) -> bool:
        """Switch to a different theme.

        Args:
            name: Name of the theme to activate.

        Returns:
            True if the theme was found and activated.
        """
        if name in self._themes:
            self._current_theme_name = name
            return True
        return False

    def register_theme(self, theme: Theme) -> None:
        """Register a new theme.

        Args:
            theme: The Theme object to register.
        """
        self._themes[theme.name] = theme

    def create_custom_theme(
        self,
        name: str,
        base_theme: str = "dark",
        overrides: dict[str, str] | None = None,
        display_name: str = "",
        description: str = "",
    ) -> Theme:
        """Create a custom theme based on an existing one with color overrides.

        Args:
            name: Unique name for the new theme.
            base_theme: Name of the base theme to derive from.
            overrides: Dictionary of color name → color value overrides.
            display_name: Human-readable display name.
            description: Theme description.

        Returns:
            The newly created Theme object.
        """
        base = self._themes.get(base_theme)
        if not base:
            raise ValueError(f"Base theme '{base_theme}' not found")

        import copy
        new_colors = copy.deepcopy(base.colors)
        if overrides:
            for key, value in overrides.items():
                if hasattr(new_colors, key):
                    setattr(new_colors, key, value)

        theme = Theme(
            name=name,
            display_name=display_name or name.replace("_", " ").title(),
            description=description or f"Custom theme based on {base_theme}",
            colors=new_colors,
            is_dark=base.is_dark,
            author="Custom",
        )
        self._themes[name] = theme
        return theme

    def save_custom_theme(self, name: str, path: str | None = None) -> str:
        """Save a theme to a JSON file.

        Args:
            name: Name of the theme to save.
            path: Custom file path. Defaults to themes_dir/<name>.json.

        Returns:
            The path the theme was saved to.
        """
        theme = self._themes.get(name)
        if not theme:
            raise ValueError(f"Theme '{name}' not found")

        save_path = Path(path) if path else self._themes_dir / f"{name}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(theme.to_dict(), f, indent=2, ensure_ascii=False)

        return str(save_path)

    def delete_custom_theme(self, name: str) -> bool:
        """Delete a custom theme (built-in themes cannot be deleted).

        Args:
            name: Name of the theme to delete.

        Returns:
            True if the theme was deleted.
        """
        builtin_names = {"dark", "light", "monokai", "solarized", "dracula", "nord"}
        if name in builtin_names:
            return False
        if name not in self._themes:
            return False
        del self._themes[name]
        # Also delete the file
        theme_path = self._themes_dir / f"{name}.json"
        if theme_path.exists():
            theme_path.unlink()
        if self._current_theme_name == name:
            self._current_theme_name = "dark"
        return True

    def export_theme(self, name: str) -> str:
        """Export a theme as a JSON string.

        Args:
            name: Theme name to export.

        Returns:
            JSON string of the theme definition.
        """
        theme = self._themes.get(name)
        if not theme:
            raise ValueError(f"Theme '{name}' not found")
        return json.dumps(theme.to_dict(), indent=2, ensure_ascii=False)

    def import_theme(self, json_str: str) -> Theme:
        """Import a theme from a JSON string.

        Args:
            json_str: JSON string containing the theme definition.

        Returns:
            The imported Theme object.
        """
        data = json.loads(json_str)
        theme = Theme.from_dict(data)
        self._themes[theme.name] = theme
        return theme
