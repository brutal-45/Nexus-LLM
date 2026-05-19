"""
Nexus-LLM Terminal UI Module

Provides a rich terminal interface with chat, formatting, theming,
syntax highlighting, interactive widgets, and advanced input handling.
"""

from nexus_llm.terminal.ansi import AnsiFormatter, AnsiColor, AnsiStyle
from nexus_llm.terminal.autocomplete import AutoCompleter, CommandCompleter, PathCompleter, ModelCompleter
from nexus_llm.terminal.chat import ChatSession, ChatLoop
from nexus_llm.terminal.commands import CommandRegistry, CommandContext
from nexus_llm.terminal.formatter import RichFormatter
from nexus_llm.terminal.history import ChatHistory, HistoryEntry
from nexus_llm.terminal.keybinds import KeyBindingManager, BindingMode
from nexus_llm.terminal.layout import LayoutManager, Pane, TabBar
from nexus_llm.terminal.markdown_ext import MarkdownRenderer
from nexus_llm.terminal.multiline import MultilineInput
from nexus_llm.terminal.panel import PanelRenderer, CollapsiblePanel
from nexus_llm.terminal.progress import ProgressBar, ProgressTracker
from nexus_llm.terminal.prompts import PromptSession
from nexus_llm.terminal.renderer import TextRenderer
from nexus_llm.terminal.spinner import Spinner, SpinnerStyle
from nexus_llm.terminal.status import StatusBar, StatusField
from nexus_llm.terminal.syntax import SyntaxHighlighter
from nexus_llm.terminal.table import TableBuilder
from nexus_llm.terminal.themes import ThemeManager, Theme
from nexus_llm.terminal.widgets import TextBox, SelectBox, ProgressBarWidget, ConfirmDialog, InputBox

__all__ = [
    "AnsiColor",
    "AnsiFormatter",
    "AnsiStyle",
    "AutoCompleter",
    "ChatLoop",
    "ChatSession",
    "CollapsiblePanel",
    "CommandCompleter",
    "CommandContext",
    "CommandRegistry",
    "ConfirmDialog",
    "HistoryEntry",
    "InputBox",
    "KeyBindingManager",
    "LayoutManager",
    "MarkdownRenderer",
    "ModelCompleter",
    "MultilineInput",
    "Pane",
    "PanelRenderer",
    "PathCompleter",
    "ProgressBar",
    "ProgressBarWidget",
    "ProgressTracker",
    "PromptSession",
    "RichFormatter",
    "SelectBox",
    "Spinner",
    "SpinnerStyle",
    "StatusBar",
    "StatusField",
    "SyntaxHighlighter",
    "TabBar",
    "TableBuilder",
    "TextRenderer",
    "TextBox",
    "Theme",
    "ThemeManager",
    "BindingMode",
]
