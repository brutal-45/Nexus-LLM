# Plugin Guide

Learn how to create, register, and use plugins with Nexus-LLM. Plugins extend the system's functionality by hooking into the processing pipeline.

---

## Overview

Nexus-LLM's plugin system allows you to:

- **Extend the chat pipeline** — Preprocess prompts, postprocess responses, filter content
- **Add custom commands** — Register new slash commands in the chat interface
- **Integrate external tools** — Connect to databases, APIs, and services
- **Add model providers** — Support new model backends
- **Customize inference** — Add custom stopping criteria, sampling strategies

Plugins are Python modules that register hooks at specific points in the Nexus-LLM pipeline.

---

## Plugin Structure

A plugin is a directory containing at minimum a `plugin.py` file:

```
plugins/
└── my_plugin/
    ├── __init__.py       # Optional: package init
    ├── plugin.py         # Required: plugin definition and hooks
    ├── config.yaml       # Optional: default configuration
    └── README.md         # Optional: documentation
```

### Minimal Plugin

```python
# plugins/my_plugin/plugin.py
from nexus_llm.plugins import Plugin, hook

class MyPlugin(Plugin):
    """A simple example plugin."""

    name = "my_plugin"
    version = "1.0.0"
    description = "My custom Nexus-LLM plugin"

    @hook("pre_inference")
    def on_pre_inference(self, context):
        """Called before the model generates a response."""
        self.logger.info(f"Processing prompt: {context.prompt[:50]}...")
        return context

    @hook("post_inference")
    def on_post_inference(self, context):
        """Called after the model generates a response."""
        self.logger.info(f"Generated {len(context.response)} characters")
        return context
```

### Plugin with Configuration

```python
# plugins/my_plugin/plugin.py
from nexus_llm.plugins import Plugin, hook, command

class SentimentPlugin(Plugin):
    """Adds sentiment analysis to responses."""

    name = "sentiment"
    version = "1.0.0"
    description = "Analyzes and tags response sentiment"

    # Default configuration (overridable by user)
    default_config = {
        "enabled": True,
        "threshold": 0.5,
        "tag_in_response": False,
    }

    def on_load(self):
        """Called when the plugin is loaded."""
        self.threshold = self.config.get("threshold", 0.5)
        self.logger.info(f"Sentiment plugin loaded (threshold: {self.threshold})")

    def on_unload(self):
        """Called when the plugin is unloaded."""
        self.logger.info("Sentiment plugin unloaded")

    @hook("post_inference")
    def analyze_sentiment(self, context):
        """Analyze the sentiment of the model's response."""
        from transformers import pipeline

        if not self._sentiment_pipeline:
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )

        result = self._sentiment_pipeline(context.response[:512])[0]
        context.metadata["sentiment"] = {
            "label": result["label"],
            "score": result["score"],
        }

        if self.config.get("tag_in_response", False):
            context.response += f"\n\n[Sentiment: {result['label']} ({result['score']:.2f})]"

        return context

    @command("/sentiment")
    def sentiment_command(self, args, context):
        """Toggle sentiment analysis. Usage: /sentiment [on|off]"""
        if args.strip() == "off":
            self.config["enabled"] = False
            return "Sentiment analysis disabled."
        elif args.strip() == "on":
            self.config["enabled"] = True
            return "Sentiment analysis enabled."
        else:
            current = context.get("sentiment", "N/A")
            return f"Sentiment analysis: {'enabled' if self.config['enabled'] else 'disabled'}\nLast result: {current}"
```

---

## Available Hooks

Hooks are extension points in the Nexus-LLM pipeline where plugins can inject custom logic.

### Inference Hooks

| Hook | Timing | Context Fields | Description |
|------|--------|---------------|-------------|
| `pre_inference` | Before model inference | `prompt`, `messages`, `params`, `model` | Modify prompts, add context, enforce limits |
| `post_inference` | After model inference | `response`, `usage`, `metadata` | Transform responses, add metadata, log |
| `on_token` | Each generated token | `token`, `token_id`, `logprob`, `index` | Real-time token filtering, custom stopping |
| `on_stream_start` | Stream begins | `request_id`, `model`, `params` | Initialize streaming state |
| `on_stream_end` | Stream ends | `request_id`, `full_response`, `usage` | Finalize streaming, aggregate metrics |

### Chat Hooks

| Hook | Timing | Context Fields | Description |
|------|--------|---------------|-------------|
| `on_message_sent` | User sends a message | `message`, `conversation_id` | Log, preprocess, or modify user messages |
| `on_message_received` | Response received | `response`, `conversation_id` | Post-process responses, update UI |
| `on_conversation_start` | New conversation | `conversation_id`, `system_prompt` | Initialize conversation state |
| `on_conversation_end` | Conversation closed | `conversation_id`, `message_count` | Save, summarize, or clean up |

### Model Hooks

| Hook | Timing | Context Fields | Description |
|------|--------|---------------|-------------|
| `on_model_load` | Model loading | `model_id`, `device`, `config` | Customize model loading |
| `on_model_unload` | Model unloading | `model_id`, `memory_freed` | Clean up resources |
| `on_model_error` | Model error | `model_id`, `error`, `step` | Handle errors gracefully |

### System Hooks

| Hook | Timing | Context Fields | Description |
|------|--------|---------------|-------------|
| `on_startup` | Server starts | `config`, `version` | Initialize resources |
| `on_shutdown` | Server stops | — | Clean up resources |
| `on_config_change` | Config updated | `key`, `old_value`, `new_value` | React to configuration changes |

### Hook Priority

Hooks are executed in priority order (lower number = runs first):

```python
@hook("post_inference", priority=10)  # Runs first
def early_postprocess(self, context):
    return context

@hook("post_inference", priority=50)  # Runs second
def late_postprocess(self, context):
    return context
```

---

## Custom Commands

Register slash commands that users can invoke in the chat interface.

```python
class MyPlugin(Plugin):
    name = "my_plugin"

    @command("/translate", help="Translate text. Usage: /translate <lang> <text>")
    def translate_command(self, args, context):
        """Translate text to a specified language."""
        parts = args.strip().split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /translate <language> <text>"

        target_lang = parts[0]
        text = parts[1]

        # Use the current model to translate
        prompt = f"Translate the following text to {target_lang}:\n\n{text}"
        result = self.inference(prompt, max_tokens=1024)
        return result

    @command("/calc", help="Evaluate a math expression. Usage: /calc <expression>")
    def calc_command(self, args, context):
        """Evaluate a mathematical expression."""
        try:
            result = eval(args.strip(), {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    @command("/wordcount", help="Count words in the last response")
    def wordcount_command(self, args, context):
        """Count words in the last assistant response."""
        last_response = context.get("last_response", "")
        word_count = len(last_response.split())
        char_count = len(last_response)
        return f"Words: {word_count} | Characters: {char_count}"
```

---

## Registering Plugins

### Configuration File

Add plugins to your configuration:

```yaml
# config/default.yaml
plugins:
  directories:
    - "./plugins"           # Local plugin directory
    - "~/.nexus/plugins"   # User-level plugins

  enabled:
    - name: "sentiment"
      config:
        threshold: 0.7
        tag_in_response: true

    - name: "my_plugin"
      config:
        custom_setting: "value"

  disabled:
    - "example_plugin"
```

### CLI

```bash
# List installed plugins
nexus plugin list

# Enable a plugin
nexus plugin enable sentiment

# Disable a plugin
nexus plugin disable sentiment

# Install a plugin from a directory
nexus plugin install ./path/to/plugin

# Install from a git repository
nexus plugin install https://github.com/user/nexus-plugin-example

# Show plugin details
nexus plugin info sentiment
```

### API

```bash
# List plugins
curl http://localhost:8000/api/v1/plugins \
  -H "Authorization: Bearer nexus_your_api_key"

# Enable a plugin
curl -X POST http://localhost:8000/api/v1/plugins/sentiment/enable \
  -H "Authorization: Bearer nexus_your_api_key"

# Disable a plugin
curl -X POST http://localhost:8000/api/v1/plugins/sentiment/disable \
  -H "Authorization: Bearer nexus_your_api_key"
```

---

## Built-in Plugins

Nexus-LLM ships with several built-in plugins:

### Content Filter

Filters inappropriate content from prompts and responses.

```yaml
plugins:
  enabled:
    - name: "content_filter"
      config:
        filter_prompts: true
        filter_responses: true
        blocked_words: []
        custom_patterns: []
```

### Token Counter

Tracks and reports token usage per conversation.

```yaml
plugins:
  enabled:
    - name: "token_counter"
      config:
        show_in_prompt: true
        warn_at_tokens: 3000
```

### Conversation Logger

Saves all conversations to a database or file.

```yaml
plugins:
  enabled:
    - name: "conversation_logger"
      config:
        backend: "sqlite"          # sqlite, jsonl, or postgres
        database_url: "sqlite:///./logs/conversations.db"
        log_system_prompts: false
```

### Auto-Summary

Automatically summarizes long conversations to manage context length.

```yaml
plugins:
  enabled:
    - name: "auto_summary"
      config:
        max_messages_before_summary: 20
        summary_model: null        # Uses current model
        keep_recent_messages: 5
```

### Code Execution

Safely executes code blocks from model responses.

```yaml
plugins:
  enabled:
    - name: "code_execution"
      config:
        enabled_languages: ["python", "javascript"]
        timeout_seconds: 30
        sandbox: true
        max_output_length: 10000
```

---

## Plugin Development Best Practices

1. **Keep plugins lightweight.** Heavy computation should be offloaded or cached.
2. **Handle errors gracefully.** Never let a plugin crash bring down the server.
3. **Use async when possible.** Long-running hooks should use `async def`.
4. **Provide sensible defaults.** Plugins should work without configuration.
5. **Document your hooks.** Clearly describe what each hook does and when it runs.
6. **Test with the plugin tester:**

```bash
nexus plugin test ./plugins/my_plugin
```

7. **Version your plugins** using semantic versioning (MAJOR.MINOR.PATCH).
8. **Don't modify context in breaking ways.** Add new fields; don't remove existing ones.
