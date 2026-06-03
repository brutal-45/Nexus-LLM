"""Terminal module for Nexus-LLM - Claude-like terminal chat interface."""
from nexus_llm.terminal.chat import TerminalChat
from nexus_llm.terminal.formatter import OutputFormatter
from nexus_llm.terminal.commands import CommandHandler
from nexus_llm.terminal.history import ChatHistory
from nexus_llm.terminal.themes import Theme, get_theme

__all__ = ["TerminalChat", "OutputFormatter", "CommandHandler", "ChatHistory", "Theme", "get_theme"]
