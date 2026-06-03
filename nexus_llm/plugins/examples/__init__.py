"""Example plugins package for Nexus-LLM."""

from nexus_llm.plugins.examples.custom_greet import CustomGreetPlugin
from nexus_llm.plugins.examples.echo import EchoPlugin

__all__ = [
    "CustomGreetPlugin",
    "EchoPlugin",
]
