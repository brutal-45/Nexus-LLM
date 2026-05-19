"""
Nexus-LLM Layout Manager Module

Provides terminal layout management with split panes, tabs,
and resizable layouts for building complex terminal UIs.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class SplitDirection(str, Enum):
    """Direction for pane splitting."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class PaneType(str, Enum):
    """Type of content a pane displays."""
    CHAT = "chat"
    HISTORY = "history"
    CONFIG = "config"
    STATUS = "status"
    CUSTOM = "custom"


@dataclass
class Pane:
    """A single pane in a layout.

    Represents a rectangular area of the terminal that can contain
    content and be resized, focused, or split.
    """
    name: str = ""
    pane_type: PaneType = PaneType.CUSTOM
    width: int = 0
    height: int = 0
    x: int = 0
    y: int = 0
    weight: float = 1.0
    min_width: int = 20
    min_height: int = 5
    visible: bool = True
    focused: bool = False
    content: str = ""
    border: bool = True
    scroll_offset: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def inner_width(self) -> int:
        """Get the inner content width (excluding borders)."""
        return max(0, self.width - 2) if self.border else self.width

    @property
    def inner_height(self) -> int:
        """Get the inner content height (excluding borders)."""
        return max(0, self.height - 2) if self.border else self.height

    def content_lines(self) -> list[str]:
        """Get content split into lines fitting the pane width."""
        if not self.content:
            return []
        width = self.inner_width
        lines = []
        for line in self.content.split("\n"):
            if len(line) <= width:
                lines.append(line)
            else:
                # Word-wrap long lines
                while len(line) > width:
                    lines.append(line[:width])
                    line = line[width:]
                if line:
                    lines.append(line)
        return lines

    def visible_content(self) -> list[str]:
        """Get content lines visible within the pane height."""
        all_lines = self.content_lines()
        height = self.inner_height
        if len(all_lines) <= height:
            return all_lines
        start = max(0, min(self.scroll_offset, len(all_lines) - height))
        return all_lines[start:start + height]

    def render(self) -> str:
        """Render the pane with borders and content.

        Returns:
            ANSI-formatted string representation of the pane.
        """
        if not self.visible:
            return ""

        lines = []
        if self.border:
            # Top border with optional title
            title = f" {self.name} " if self.name else ""
            inner_width = self.width - 2
            if title:
                pad_len = max(0, inner_width - len(title))
                left_pad = pad_len // 2
                right_pad = pad_len - left_pad
                top = f"┌{'─' * left_pad}{title}{'─' * right_pad}┐"
            else:
                top = f"┌{'─' * inner_width}┐"
            border_color = "\033[38;5;117m" if self.focused else "\033[38;5;67m"
            reset = "\033[0m"
            lines.append(f"{border_color}{top}{reset}")

            # Content lines
            visible = self.visible_content()
            for i in range(self.inner_height):
                if i < len(visible):
                    line = visible[i]
                    padding = self.inner_width - len(line)
                    lines.append(f"│ {line}{' ' * max(0, padding)} │")
                else:
                    lines.append(f"│{' ' * self.inner_width}│")

            # Bottom border
            bottom = f"└{'─' * inner_width}┘"
            lines.append(f"{border_color}{bottom}{reset}")
        else:
            visible = self.visible_content()
            for line in visible:
                lines.append(line)

        return "\n".join(lines)


@dataclass
class Tab:
    """A single tab in a tab bar."""
    name: str
    active: bool = False
    pane: Pane | None = None
    icon: str = ""
    closable: bool = True

    @property
    def display_name(self) -> str:
        """Get the tab display name with optional icon."""
        if self.icon:
            return f"{self.icon} {self.name}"
        return self.name


class TabBar:
    """A horizontal tab bar for switching between panes.

    Renders a row of tabs with the active tab highlighted,
    supporting tab switching, creation, and closure.
    """

    def __init__(self, width: int = 80) -> None:
        self._tabs: list[Tab] = []
        self._active_index = 0
        self._width = width

    @property
    def active_tab(self) -> Tab | None:
        """Get the currently active tab."""
        if 0 <= self._active_index < len(self._tabs):
            return self._tabs[self._active_index]
        return None

    @property
    def active_index(self) -> int:
        """Get the index of the active tab."""
        return self._active_index

    @property
    def tabs(self) -> list[Tab]:
        """Get all tabs."""
        return list(self._tabs)

    def add_tab(self, name: str, pane: Pane | None = None, icon: str = "") -> Tab:
        """Add a new tab.

        Args:
            name: Tab name.
            pane: Optional pane associated with the tab.
            icon: Optional icon character.

        Returns:
            The created Tab object.
        """
        tab = Tab(name=name, pane=pane, icon=icon)
        self._tabs.append(tab)
        if len(self._tabs) == 1:
            tab.active = True
        return tab

    def remove_tab(self, index: int) -> Tab | None:
        """Remove a tab by index.

        Args:
            index: Tab index to remove.

        Returns:
            The removed Tab, or None if index is invalid.
        """
        if 0 <= index < len(self._tabs):
            tab = self._tabs.pop(index)
            if self._active_index >= len(self._tabs):
                self._active_index = max(0, len(self._tabs) - 1)
            if self._tabs:
                self._tabs[self._active_index].active = True
            return tab
        return None

    def switch_to(self, index: int) -> bool:
        """Switch to a tab by index.

        Args:
            index: Tab index to activate.

        Returns:
            True if the switch was successful.
        """
        if 0 <= index < len(self._tabs):
            for tab in self._tabs:
                tab.active = False
            self._tabs[index].active = True
            self._active_index = index
            return True
        return False

    def switch_next(self) -> None:
        """Switch to the next tab (wraps around)."""
        if self._tabs:
            self.switch_to((self._active_index + 1) % len(self._tabs))

    def switch_prev(self) -> None:
        """Switch to the previous tab (wraps around)."""
        if self._tabs:
            self.switch_to((self._active_index - 1) % len(self._tabs))

    def render(self) -> str:
        """Render the tab bar.

        Returns:
            ANSI-formatted string representation.
        """
        if not self._tabs:
            return ""

        parts = []
        for tab in self._tabs:
            name = tab.display_name
            if tab.active:
                styled = f"\033[48;5;26;38;5;231m {name} \033[0m"
            else:
                styled = f"\033[48;5;236;38;5;145m {name} \033[0m"
            parts.append(styled)

        bar = "".join(parts)

        # Pad to width
        visible_len = sum(len(t.display_name) + 2 for t in self._tabs)
        remaining = max(0, self._width - visible_len)
        if remaining > 0:
            bar += f"\033[48;5;236m{' ' * remaining}\033[0m"

        return bar


class LayoutManager:
    """Manages terminal layout with split panes, tabs, and resizing.

    Provides a hierarchical layout system where the terminal area can be
    divided into horizontal or vertical splits, each containing panes
    that can be independently resized, focused, and scrolled.
    """

    def __init__(self, width: int | None = None, height: int | None = None) -> None:
        self._width = width or shutil.get_terminal_size().columns
        self._height = height or shutil.get_terminal_size().lines
        self._panes: list[Pane] = []
        self._tab_bar = TabBar(width=self._width)
        self._split_stack: list[SplitDirection] = []
        self._focused_index = 0

    @property
    def width(self) -> int:
        """Get the layout width."""
        return self._width

    @property
    def height(self) -> int:
        """Get the layout height."""
        return self._height

    @property
    def panes(self) -> list[Pane]:
        """Get all panes."""
        return list(self._panes)

    @property
    def focused_pane(self) -> Pane | None:
        """Get the currently focused pane."""
        if 0 <= self._focused_index < len(self._panes):
            return self._panes[self._focused_index]
        return None

    @property
    def tab_bar(self) -> TabBar:
        """Get the tab bar."""
        return self._tab_bar

    def add_pane(
        self,
        name: str = "",
        pane_type: PaneType = PaneType.CUSTOM,
        weight: float = 1.0,
        border: bool = True,
        content: str = "",
    ) -> Pane:
        """Add a new pane to the layout.

        Args:
            name: Pane name displayed in the title bar.
            pane_type: Type of content the pane holds.
            weight: Relative size weight for splitting.
            border: Whether to draw borders.
            content: Initial content.

        Returns:
            The created Pane object.
        """
        pane = Pane(
            name=name,
            pane_type=pane_type,
            weight=weight,
            border=border,
            content=content,
        )
        self._panes.append(pane)
        if len(self._panes) == 1:
            pane.focused = True
        self._recalculate()
        return pane

    def remove_pane(self, index: int) -> Pane | None:
        """Remove a pane by index.

        Args:
            index: Pane index to remove.

        Returns:
            The removed Pane, or None if invalid index.
        """
        if 0 <= index < len(self._panes):
            pane = self._panes.pop(index)
            if self._focused_index >= len(self._panes):
                self._focused_index = max(0, len(self._panes) - 1)
            if self._panes:
                self._panes[self._focused_index].focused = True
            self._recalculate()
            return pane
        return None

    def focus_pane(self, index: int) -> bool:
        """Set focus to a specific pane.

        Args:
            index: Pane index to focus.

        Returns:
            True if the pane was found and focused.
        """
        if 0 <= index < len(self._panes):
            for pane in self._panes:
                pane.focused = False
            self._panes[index].focused = True
            self._focused_index = index
            return True
        return False

    def focus_next(self) -> None:
        """Move focus to the next pane."""
        if self._panes:
            self.focus_pane((self._focused_index + 1) % len(self._panes))

    def focus_prev(self) -> None:
        """Move focus to the previous pane."""
        if self._panes:
            self.focus_pane((self._focused_index - 1) % len(self._panes))

    def resize_pane(self, index: int, weight: float) -> None:
        """Resize a pane by adjusting its weight.

        Args:
            index: Pane index to resize.
            weight: New weight value.
        """
        if 0 <= index < len(self._panes):
            self._panes[index].weight = max(0.1, weight)
            self._recalculate()

    def split(self, direction: SplitDirection = SplitDirection.HORIZONTAL) -> None:
        """Set the split direction for the layout.

        Args:
            direction: Horizontal or vertical split.
        """
        self._split_stack.append(direction)
        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate pane positions and sizes based on weights."""
        if not self._panes:
            return

        total_weight = sum(p.weight for p in self._panes if p.visible)
        if total_weight <= 0:
            return

        visible_panes = [p for p in self._panes if p.visible]
        if not visible_panes:
            return

        # Determine current split direction
        direction = self._split_stack[-1] if self._split_stack else SplitDirection.HORIZONTAL

        # Reserve space for tab bar
        tab_height = 1 if self._tab_bar._tabs else 0
        available_height = self._height - tab_height

        if direction == SplitDirection.HORIZONTAL:
            # Panes side by side
            available_width = self._width
            x_offset = 0
            for pane in visible_panes:
                pane.width = int(available_width * (pane.weight / total_weight))
                pane.height = available_height
                pane.x = x_offset
                pane.y = tab_height
                x_offset += pane.width
            # Adjust last pane to fill remaining space
            if visible_panes:
                visible_panes[-1].width = available_width - sum(p.width for p in visible_panes[:-1])
        else:
            # Panes stacked vertically
            y_offset = tab_height
            for pane in visible_panes:
                pane.width = self._width
                pane.height = int(available_height * (pane.weight / total_weight))
                pane.x = 0
                pane.y = y_offset
                y_offset += pane.height
            # Adjust last pane
            if visible_panes:
                visible_panes[-1].height = available_height - sum(p.height for p in visible_panes[:-1])

    def render(self) -> str:
        """Render the entire layout.

        Returns:
            ANSI-formatted string of the complete layout.
        """
        self._recalculate()
        parts = []

        # Tab bar
        if self._tab_bar._tabs:
            parts.append(self._tab_bar.render())

        # Panes
        for pane in self._panes:
            if pane.visible:
                parts.append(pane.render())

        return "\n".join(parts)

    def set_pane_content(self, index: int, content: str) -> bool:
        """Update a pane's content.

        Args:
            index: Pane index.
            content: New content string.

        Returns:
            True if the pane was updated.
        """
        if 0 <= index < len(self._panes):
            self._panes[index].content = content
            return True
        return False

    def scroll_pane(self, index: int, lines: int) -> None:
        """Scroll a pane's content.

        Args:
            index: Pane index.
            lines: Number of lines to scroll (negative for up).
        """
        if 0 <= index < len(self._panes):
            pane = self._panes[index]
            all_lines = pane.content_lines()
            max_scroll = max(0, len(all_lines) - pane.inner_height)
            pane.scroll_offset = max(0, min(pane.scroll_offset + lines, max_scroll))
