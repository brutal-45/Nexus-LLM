"""Terminal themes for Nexus-LLM."""

from dataclasses import dataclass, field
from typing import Dict, Optional
from rich.style import Style
from rich.text import Text


@dataclass
class Theme:
    """A terminal color theme."""
    name: str
    description: str
    user_prompt: str = "bold cyan"
    assistant_text: str = "white"
    system_text: str = "dim yellow"
    error_text: str = "bold red"
    success_text: str = "bold green"
    info_text: str = "dim"
    code_text: str = "green"
    code_bg: str = "default"
    border_color: str = "blue"
    title_color: str = "bold magenta"
    highlight: str = "bold yellow"


THEMES: Dict[str, Theme] = {
    "dark": Theme(
        name="dark", description="Dark theme (default)",
        user_prompt="bold cyan", assistant_text="white",
        system_text="dim yellow", error_text="bold red",
        success_text="bold green", info_text="dim",
        code_text="green", code_bg="default",
        border_color="blue", title_color="bold magenta",
        highlight="bold yellow",
    ),
    "light": Theme(
        name="light", description="Light theme",
        user_prompt="bold blue", assistant_text="black",
        system_text="dim magenta", error_text="bold red",
        success_text="bold green", info_text="dim",
        code_text="dark_green", code_bg="default",
        border_color="dark_blue", title_color="bold dark_red",
        highlight="bold dark_red",
    ),
    "ocean": Theme(
        name="ocean", description="Ocean blue theme",
        user_prompt="bold bright_blue", assistant_text="bright_white",
        system_text="dim bright_cyan", error_text="bold bright_red",
        success_text="bold bright_green", info_text="dim bright_blue",
        code_text="bright_cyan", code_bg="default",
        border_color="bright_blue", title_color="bold bright_magenta",
        highlight="bold bright_yellow",
    ),
    "forest": Theme(
        name="forest", description="Forest green theme",
        user_prompt="bold bright_green", assistant_text="bright_white",
        system_text="dim yellow", error_text="bold red",
        success_text="bold green", info_text="dim green",
        code_text="green", code_bg="default",
        border_color="green", title_color="bold bright_green",
        highlight="bold yellow",
    ),
    "sunset": Theme(
        name="sunset", description="Warm sunset theme",
        user_prompt="bold bright_yellow", assistant_text="bright_white",
        system_text="dim bright_red", error_text="bold red",
        success_text="bold green", info_text="dim yellow",
        code_text="bright_yellow", code_bg="default",
        border_color="bright_red", title_color="bold bright_red",
        highlight="bold bright_white",
    ),
}


def get_theme(name: str = "dark") -> Theme:
    """Get a theme by name. Returns dark theme if not found."""
    return THEMES.get(name, THEMES["dark"])


def list_themes() -> Dict[str, Theme]:
    """List all available themes."""
    return THEMES.copy()
