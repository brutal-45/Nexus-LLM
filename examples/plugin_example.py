#!/usr/bin/env python3
"""
Custom Plugin Example - Nexus-LLM
===================================
Demonstrates how to create and register custom plugins
that extend Nexus-LLM's functionality.
"""

from nexus_llm import InferenceEngine, Conversation
from nexus_llm.plugins import (
    Plugin,
    PluginManager,
    HookType,
    PluginContext,
)


# --- Define custom plugins ---

class LoggingPlugin(Plugin):
    """Plugin that logs all conversations to a file."""

    name = "conversation_logger"
    version = "1.0.0"
    description = "Logs all conversations to a structured log file"

    def __init__(self, log_path: str = "./logs/conversations.log"):
        self.log_path = log_path
        self.log_file = None

    def on_load(self, context: PluginContext):
        """Called when the plugin is loaded."""
        import os
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log_file = open(self.log_path, "a")
        self.log_file.write(f"# Plugin loaded at {self._timestamp()}\n")
        self.log_file.flush()

    def on_unload(self, context: PluginContext):
        """Called when the plugin is unloaded."""
        if self.log_file:
            self.log_file.write(f"# Plugin unloaded at {self._timestamp()}\n")
            self.log_file.close()

    @HookType.PRE_INFERENCE
    def on_pre_inference(self, context: PluginContext):
        """Log the user's input before inference."""
        conversation = context.conversation
        last_message = conversation.messages[-1]
        self.log_file.write(
            f"[{self._timestamp()}] USER: {last_message.content}\n"
        )
        self.log_file.flush()

    @HookType.POST_INFERENCE
    def on_post_inference(self, context: PluginContext):
        """Log the model's response after inference."""
        response = context.response
        self.log_file.write(
            f"[{self._timestamp()}] ASSISTANT: {response.text}\n"
        )
        self.log_file.write(
            f"[{self._timestamp()}] META: tokens={response.token_count}, "
            f"latency={response.elapsed_time:.3f}s\n"
        )
        self.log_file.flush()

    def _timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()


class TokenBudgetPlugin(Plugin):
    """Plugin that enforces a token budget per conversation."""

    name = "token_budget"
    version = "1.0.0"
    description = "Enforces maximum token usage per conversation"

    def __init__(self, max_tokens: int = 10000):
        self.max_tokens = max_tokens

    @HookType.PRE_INFERENCE
    def on_pre_inference(self, context: PluginContext):
        """Check if the conversation has exceeded the token budget."""
        conversation = context.conversation
        if conversation.total_tokens >= self.max_tokens:
            raise TokenBudgetExceeded(
                f"Token budget exceeded: {conversation.total_tokens}/{self.max_tokens}"
            )
        # Set max_tokens for this request based on remaining budget
        remaining = self.max_tokens - conversation.total_tokens
        context.generation_kwargs["max_tokens"] = min(
            context.generation_kwargs.get("max_tokens", 512),
            remaining,
        )


class ResponseCachingPlugin(Plugin):
    """Plugin that caches responses to avoid redundant computation."""

    name = "response_cache"
    version = "1.0.0"
    description = "Caches model responses for identical queries"

    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0

    @HookType.PRE_INFERENCE
    def on_pre_inference(self, context: PluginContext):
        """Check cache for a matching query."""
        cache_key = self._make_key(context)
        if cache_key in self.cache:
            self.hits += 1
            context.cached_response = self.cache[cache_key]
            context.skip_inference = True  # Skip the actual model call
        else:
            self.misses += 1

    @HookType.POST_INFERENCE
    def on_post_inference(self, context: PluginContext):
        """Store the response in cache."""
        if not hasattr(context, 'cached_response'):
            cache_key = self._make_key(context)
            if len(self.cache) >= self.max_cache_size:
                # Evict oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = context.response

    def _make_key(self, context: PluginContext) -> str:
        """Create a cache key from the conversation state."""
        import hashlib
        last_msg = context.conversation.messages[-1].content
        return hashlib.md5(last_msg.encode()).hexdigest()

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }


class TokenBudgetExceeded(Exception):
    """Raised when the token budget is exceeded."""
    pass


def main():
    # --- Create the plugin manager ---
    plugin_manager = PluginManager()

    # Register plugins
    plugin_manager.register(LoggingPlugin(log_path="./logs/conversations.log"))
    plugin_manager.register(TokenBudgetPlugin(max_tokens=50000))
    plugin_manager.register(ResponseCachingPlugin(max_cache_size=500))

    # --- Attach plugins to the engine ---
    engine = InferenceEngine(
        model_name="nexus-7b-chat",
        device="auto",
        plugin_manager=plugin_manager,
    )

    # --- Use the engine with plugins active ---
    conversation = Conversation(
        system_prompt="You are a helpful assistant."
    )

    # First query (cache miss)
    conversation.add_user_message("What is machine learning?")
    response = engine.chat(conversation)
    print(f"Response: {response.text[:100]}...")

    # Same query again (cache hit)
    conversation2 = Conversation(system_prompt="You are a helpful assistant.")
    conversation2.add_user_message("What is machine learning?")
    response2 = engine.chat(conversation2)
    print(f"Cached response: {response2.text[:100]}...")

    # Print cache stats
    cache_plugin = plugin_manager.get_plugin("response_cache")
    stats = cache_plugin.get_stats()
    print(f"\nCache stats: {stats}")

    # --- Load plugins from directory ---
    plugin_manager.load_from_directory("./plugins/")

    # --- List active plugins ---
    print("\nActive plugins:")
    for plugin in plugin_manager.list_plugins():
        print(f"  - {plugin.name} v{plugin.version}: {plugin.description}")


if __name__ == "__main__":
    main()
