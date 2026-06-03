"""Terminal module for Nexus-LLM - Claude-like terminal chat interface."""

from terminal.chat import TerminalChat
from terminal.formatter import OutputFormatter
from terminal.commands import CommandHandler
from terminal.history import ChatHistory

__all__ = ["TerminalChat", "OutputFormatter", "CommandHandler", "ChatHistory"]
