# Custom Plugins Tutorial

This tutorial shows how to create, register, and use custom plugins to extend Nexus-LLM's functionality.

## What Are Plugins?

Plugins are modular components that hook into the Nexus-LLM lifecycle. They can:
- Transform inputs and outputs
- Add logging and monitoring
- Implement caching and rate limiting
- Enforce safety policies
- Integrate with external services

Plugins use a hook system with defined lifecycle points where they can inject behavior.

## Step 1: Create a Plugin

Every plugin extends the `Plugin` base class and implements one or more hook methods:

```python
from nexus_llm.plugins import Plugin, HookType, PluginContext


class TimingPlugin(Plugin):
    """Plugin that tracks and logs inference timing."""

    name = "timing_tracker"
    version = "1.0.0"
    description = "Tracks inference timing and logs slow requests"

    def __init__(self, slow_threshold: float = 5.0):
        self.slow_threshold = slow_threshold
        self.timings = []

    def on_load(self, context: PluginContext):
        """Called when the plugin is loaded."""
        print(f"[{self.name}] Plugin loaded (slow threshold: {self.slow_threshold}s)")

    def on_unload(self, context: PluginContext):
        """Called when the plugin is unloaded."""
        avg_time = sum(self.timings) / len(self.timings) if self.timings else 0
        print(f"[{self.name}] Avg inference time: {avg_time:.3f}s over {len(self.timings)} requests")

    @HookType.PRE_INFERENCE
    def on_pre_inference(self, context: PluginContext):
        """Record the start time before inference."""
        import time
        context.state["start_time"] = time.time()

    @HookType.POST_INFERENCE
    def on_post_inference(self, context: PluginContext):
        """Calculate and log the elapsed time after inference."""
        import time
        start_time = context.state.get("start_time", time.time())
        elapsed = time.time() - start_time
        self.timings.append(elapsed)

        if elapsed > self.slow_threshold:
            print(f"[{self.name}] SLOW REQUEST: {elapsed:.3f}s")
```

## Step 2: Available Hooks

| Hook | Decorator | When It Fires | Use Case |
|------|-----------|---------------|----------|
| `on_load` | — | Plugin loaded | Initialize resources |
| `on_unload` | — | Plugin unloaded | Cleanup resources |
| `PRE_INFERENCE` | `@HookType.PRE_INFERENCE` | Before model inference | Input validation, caching |
| `POST_INFERENCE` | `@HookType.POST_INFERENCE` | After model inference | Output filtering, logging |
| `ON_ERROR` | `@HookType.ON_ERROR` | On inference error | Error handling, retry logic |
| `PRE_TOOL_CALL` | `@HookType.PRE_TOOL_CALL` | Before agent tool execution | Tool validation |
| `POST_TOOL_CALL` | `@HookType.POST_TOOL_CALL` | After agent tool execution | Result logging |

## Step 3: Register Plugins

```python
from nexus_llm.plugins import PluginManager

# Create a plugin manager
manager = PluginManager()

# Register individual plugins
manager.register(TimingPlugin(slow_threshold=3.0))

# Register multiple plugins
manager.register_all([
    TimingPlugin(slow_threshold=3.0),
    LoggingPlugin(log_path="./logs/inference.log"),
    ResponseCachingPlugin(max_cache_size=500),
])

# Load all plugins from a directory
manager.load_from_directory("./plugins/")
```

## Step 4: Use Plugins with the Engine

```python
from nexus_llm import InferenceEngine, Conversation

engine = InferenceEngine(
    model_name="nexus-7b-chat",
    device="auto",
    plugin_manager=manager,    # Attach the plugin manager
)

# All inference calls will now go through the plugin hooks
conversation = Conversation()
conversation.add_user_message("Hello!")
response = engine.chat(conversation)
```

## Step 5: Advanced Plugin Patterns

### Caching Plugin

```python
class ResponseCachingPlugin(Plugin):
    name = "response_cache"
    version = "1.0.0"
    description = "Caches responses to avoid redundant computation"

    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.max_cache_size = max_cache_size

    @HookType.PRE_INFERENCE
    def on_pre_inference(self, context: PluginContext):
        cache_key = self._make_key(context)
        if cache_key in self.cache:
            context.cached_response = self.cache[cache_key]
            context.skip_inference = True

    @HookType.POST_INFERENCE
    def on_post_inference(self, context: PluginContext):
        if not hasattr(context, "cached_response"):
            cache_key = self._make_key(context)
            self.cache[cache_key] = context.response

    def _make_key(self, context: PluginContext) -> str:
        import hashlib
        msg = context.conversation.messages[-1].content
        return hashlib.md5(msg.encode()).hexdigest()
```

### Error Retry Plugin

```python
class RetryPlugin(Plugin):
    name = "retry"
    version = "1.0.0"
    description = "Retries failed inference requests"

    def __init__(self, max_retries: int = 3, backoff: float = 1.0):
        self.max_retries = max_retries
        self.backoff = backoff

    @HookType.ON_ERROR
    def on_error(self, context: PluginContext):
        retry_count = context.state.get("retry_count", 0)
        if retry_count < self.max_retries:
            import time
            wait = self.backoff * (2 ** retry_count)
            time.sleep(wait)
            context.state["retry_count"] = retry_count + 1
            context.retry = True
```

### Input Sanitization Plugin

```python
class InputSanitizationPlugin(Plugin):
    name = "input_sanitizer"
    version = "1.0.0"
    description = "Sanitizes user inputs before inference"

    @HookType.PRE_INFERENCE
    def on_pre_inference(self, context: PluginContext):
        last_msg = context.conversation.messages[-1]
        # Remove potential prompt injection patterns
        sanitized = self._sanitize(last_msg.content)
        last_msg.content = sanitized

    def _sanitize(self, text: str) -> str:
        # Remove common injection patterns
        patterns = ["ignore previous", "system:", "### Instruction"]
        for pattern in patterns:
            text = text.replace(pattern, "[filtered]")
        return text
```

## Step 6: Plugin Execution Order

Plugins execute in registration order. You can control order with priority:

```python
manager.register(SafetyPlugin(), priority=1)      # Runs first
manager.register(InputSanitizerPlugin(), priority=2)
manager.register(CachingPlugin(), priority=3)
manager.register(LoggingPlugin(), priority=10)     # Runs last
```

## Step 7: Plugin Configuration

For production use, configure plugins via YAML:

```yaml
plugins:
  - name: timing_tracker
    enabled: true
    config:
      slow_threshold: 3.0

  - name: response_cache
    enabled: true
    config:
      max_cache_size: 500

  - name: input_sanitizer
    enabled: true

  - name: retry
    enabled: true
    config:
      max_retries: 3
      backoff: 1.0
```

Load configuration:

```python
manager = PluginManager.from_config("plugins.yaml")
```

## Best Practices

1. **Keep plugins focused**: Each plugin should do one thing well.
2. **Handle errors gracefully**: Don't let plugin errors break the inference pipeline.
3. **Use `context.state`** for sharing data between hooks within the same request.
4. **Document configuration options**: Clearly document all constructor parameters.
5. **Test plugins in isolation**: Write unit tests that don't require a real model.
6. **Be mindful of performance**: Pre-inference hooks add latency to every request.
