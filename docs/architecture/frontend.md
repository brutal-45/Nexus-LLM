# Frontend Architecture

Deep dive into the terminal UI, command system, and output rendering that powers the Nexus-LLM user interface.

---

## Overview

Nexus-LLM provides a rich terminal user interface (TUI) built with the Python `rich` library. The frontend layer handles user input, command parsing, conversation management, and formatted output rendering. It also provides the API server interface for programmatic access.

```
┌──────────────────────────────────────────────────┐
│                  Frontend Layer                    │
│                                                    │
│  ┌─────────────────┐  ┌───────────────────────┐  │
│  │  Terminal UI     │  │  API Server Interface  │  │
│  │  (Rich TUI)      │  │  (FastAPI + OpenAPI)   │  │
│  └────────┬─────────┘  └───────────┬───────────┘  │
│           │                         │              │
│           ▼                         ▼              │
│  ┌──────────────────────────────────────────────┐ │
│  │              Command System                    │ │
│  │  ┌─────────┐ ┌─────────┐ ┌────────────────┐ │ │
│  │  │ Parser  │ │ Router  │ │  Executor       │ │ │
│  │  └─────────┘ └─────────┘ └────────────────┘ │ │
│  └──────────────────────────────────────────────┘ │
│                         │                          │
│                         ▼                          │
│  ┌──────────────────────────────────────────────┐ │
│  │            Conversation Manager                │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │ │
│  │  │ History  │ │ Context  │ │  Persistence  │ │ │
│  │  │ Manager  │ │ Window   │ │  Layer        │ │ │
│  │  └──────────┘ └──────────┘ └──────────────┘ │ │
│  └──────────────────────────────────────────────┘ │
│                         │                          │
│                         ▼                          │
│  ┌──────────────────────────────────────────────┐ │
│  │             Output Renderer                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │ │
│  │  │ Markdown │ │ Syntax   │ │  Streaming    │ │ │
│  │  │ Renderer │ │Highlight │ │  Display      │ │ │
│  │  └──────────┘ └──────────┘ └──────────────┘ │ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

---

## Terminal UI

### Component Architecture

The TUI is built with `rich` and provides a modern, interactive terminal experience:

```python
class ChatTUI:
    """Main terminal UI controller for interactive chat mode."""

    def __init__(self, config: ChatConfig):
        self.console = Console(theme=nexus_theme)
        self.session = PromptSession(history=FileHistory(".nexus_history"))
        self.renderer = OutputRenderer(self.console)
        self.command_router = CommandRouter()
        self.conversation = ConversationManager()

    async def run(self):
        """Main event loop for the chat TUI."""
        self._show_banner()
        self._show_model_info()

        while True:
            try:
                # Get user input with rich prompt
                user_input = await self.session.prompt_async(
                    HTML("<ansicyan>You></ansicyan> "),
                    multiline=False,
                    auto_suggest=AutoSuggestFromHistory(),
                )

                if not user_input.strip():
                    continue

                # Check if it's a command
                if user_input.startswith("/"):
                    result = self.command_router.execute(user_input)
                    self.renderer.render_command_result(result)
                    continue

                # Send to model and stream response
                self.renderer.start_response()
                async for token in self.inference_engine.stream_chat(
                    messages=self.conversation.get_messages(),
                    params=self.conversation.get_params(),
                ):
                    self.renderer.render_token(token)
                self.renderer.end_response()

                # Update conversation
                self.conversation.add_exchange(user_input, self.renderer.get_response())

            except KeyboardInterrupt:
                continue
            except EOFError:
                break
```

### Banner and Model Info

```
╔═══════════════════════════════════════════════╗
║              Nexus-LLM v1.2.0                  ║
║     Local LLM Framework by Nexus Labs          ║
╠═══════════════════════════════════════════════╣
║  Model  : Llama-3.1-8B-Instruct               ║
║  Device : NVIDIA RTX 4090 (cuda:0)            ║
║  Memory : 15.1 / 24.0 GB (63%)                ║
║  RAG    : Enabled (company_docs, 312 chunks)  ║
║  Plugins: sentiment, token_counter             ║
╠═══════════════════════════════════════════════╣
║  Type /help for commands, /quit to exit        ║
╚═══════════════════════════════════════════════╝
```

### Input Handling

The input system supports:

- **Single-line input** — Standard prompt for quick messages
- **Multi-line input** — `Shift+Enter` or `\\` at end of line to continue
- **Auto-suggestions** — History-based suggestions as you type
- **Tab completion** — Complete commands and model names
- **Paste support** — Multi-line paste detection

```python
class InputHandler:
    """Handles user input with advanced editing features."""

    def __init__(self, session: PromptSession):
        self.session = session
        self.completer = NexusCompleter()

    async def get_input(self) -> str:
        """Get user input with completion and history."""
        return await self.session.prompt_async(
            HTML("<ansicyan>You></ansicyan> "),
            completer=self.completer,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self._get_key_bindings(),
        )

class NexusCompleter(Completer):
    """Tab completion for commands, model names, and file paths."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        if text.startswith("/"):
            # Command completion
            yield from self._complete_command(text)
        elif text.startswith("@"):
            # Model name completion
            yield from self._complete_model(text)
        elif text.startswith("~"):
            # File path completion
            yield from self._complete_path(text)
```

---

## Command System

### Architecture

```
User types: "/model mistralai/Mistral-7B"
                    │
                    ▼
            ┌──────────────┐
            │   Parser      │  Splits into: command="model", args="mistralai/Mistral-7B"
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │   Router      │  Looks up "model" → ModelCommand handler
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │   Validator   │  Validates args (model exists?)
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │   Executor    │  Executes: load model, update session
            └──────┬───────┘
                   │
                   ▼
            Render result to user
```

### Command Registry

```python
class CommandRouter:
    """Routes slash commands to their handlers."""

    def __init__(self):
        self.commands: dict[str, CommandHandler] = {}
        self._register_builtin_commands()

    def register(self, name: str, handler: CommandHandler, aliases: list[str] = None):
        """Register a command handler."""
        self.commands[name] = handler
        for alias in (aliases or []):
            self.commands[alias] = handler

    def execute(self, input_str: str) -> CommandResult:
        """Parse and execute a command."""
        parsed = self._parse(input_str)
        handler = self.commands.get(parsed.command)

        if not handler:
            return CommandResult(
                success=False,
                message=f"Unknown command: /{parsed.command}. Type /help for available commands."
            )

        return handler.execute(parsed.args)
```

### Built-in Commands

| Command | Aliases | Handler | Description |
|---------|---------|---------|-------------|
| `/help` | `/?` | HelpCommand | Show available commands |
| `/quit` | `/exit`, `/q` | QuitCommand | Exit the application |
| `/model` | `/m` | ModelCommand | Switch models |
| `/system` | `/sys` | SystemCommand | Set system prompt |
| `/clear` | `/cls` | ClearCommand | Clear conversation history |
| `/save` | `/s` | SaveCommand | Save conversation |
| `/load` | `/l` | LoadCommand | Load conversation |
| `/temp` | `/t` | TempCommand | Change temperature |
| `/topp` | | ToppCommand | Change top_p |
| `/maxtokens` | `/mt` | MaxTokensCommand | Change max tokens |
| `/rag` | | RagCommand | Toggle RAG |
| `/agent` | `/a` | AgentCommand | Switch agent |
| `/info` | `/i` | InfoCommand | Show current settings |
| `/copy` | `/c` | CopyCommand | Copy last response |
| `/redo` | `/r` | RedoCommand | Regenerate last response |
| `/edit` | `/e` | EditCommand | Edit last user message |

### Plugin Commands

Plugins can register custom commands that appear alongside built-in commands:

```python
class PluginCommandHandler:
    """Handles commands registered by plugins."""

    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager

    def register_plugin_commands(self):
        """Register all commands from loaded plugins."""
        for plugin in self.plugin_manager.loaded_plugins:
            for cmd_name, cmd_func in plugin.get_commands().items():
                self.router.register(
                    name=f"/{cmd_name}",
                    handler=PluginCommandAdapter(plugin, cmd_func),
                )
```

---

## Conversation Manager

### Message History

```python
class ConversationManager:
    """Manages conversation state, history, and context window."""

    def __init__(self, max_context_tokens: int = 4096):
        self.messages: list[Message] = []
        self.system_prompt: Optional[str] = None
        self.max_context_tokens = max_context_tokens
        self.metadata: dict = {}

    def add_exchange(self, user_msg: str, assistant_msg: str):
        """Add a user-assistant exchange."""
        self.messages.append(Message(role="user", content=user_msg))
        self.messages.append(Message(role="assistant", content=assistant_msg))

    def get_messages(self) -> list[dict]:
        """Get messages formatted for the model, respecting context limits."""
        # Always include system prompt
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})

        # Add messages from newest to oldest until context is full
        token_count = self._count_tokens(result)
        included = []

        for msg in reversed(self.messages):
            msg_tokens = self._count_tokens([msg.to_dict()])
            if token_count + msg_tokens > self.max_context_tokens:
                break
            included.append(msg)
            token_count += msg_tokens

        # Reverse to get chronological order
        result.extend(m.to_dict() for m in reversed(included))
        return result
```

### Context Window Management

When conversations grow beyond the model's context window, the manager applies one of several strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `truncate_oldest` | Remove oldest messages first | General chat |
| `summarize` | Summarize older messages into a condensed form | Long conversations |
| `sliding_window` | Keep only the last N messages | Simple, predictable |
| `rag_fallback` | Move old messages to RAG, retrieve as needed | Research conversations |

```python
class ContextWindowManager:
    """Manages which messages fit within the context window."""

    def apply_strategy(self, messages: list[Message], strategy: str) -> list[Message]:
        if strategy == "truncate_oldest":
            return self._truncate_oldest(messages)
        elif strategy == "summarize":
            return self._summarize_old(messages)
        elif strategy == "sliding_window":
            return self._sliding_window(messages)
        elif strategy == "rag_fallback":
            return self._rag_fallback(messages)
```

### Conversation Persistence

```python
class ConversationPersistence:
    """Save and load conversations to disk."""

    def save(self, conversation: Conversation, path: str) -> None:
        """Save a conversation to a JSON file."""
        data = {
            "id": conversation.id,
            "model": conversation.model,
            "system_prompt": conversation.system_prompt,
            "messages": [m.to_dict() for m in conversation.messages],
            "params": conversation.params.to_dict(),
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "metadata": conversation.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> Conversation:
        """Load a conversation from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return Conversation.from_dict(data)
```

---

## Output Renderer

### Markdown Rendering

The renderer converts model output to rich terminal formatting:

```python
class OutputRenderer:
    """Renders model output with syntax highlighting and formatting."""

    def __init__(self, console: Console):
        self.console = console
        self.markdown = Markdown()
        self.current_response = ""

    def render_token(self, token: str):
        """Render a streaming token."""
        self.current_response += token

        # Detect and render code blocks in real-time
        if self._in_code_block():
            self._render_code_token(token)
        else:
            self.console.print(token, end="")

    def end_response(self):
        """Finalize the response with full markdown rendering."""
        self.console.print()
        md = Markdown(self.current_response)
        self.console.print(md)
```

### Syntax Highlighting

Code blocks are highlighted using `pygments`:

```python
class CodeHighlighter:
    """Syntax highlights code blocks in responses."""

    LANGUAGE_MAP = {
        "python": "python",
        "py": "python",
        "javascript": "javascript",
        "js": "javascript",
        "typescript": "typescript",
        "ts": "typescript",
        "bash": "bash",
        "sh": "bash",
        "sql": "sql",
        "json": "json",
        "yaml": "yaml",
        "html": "html",
        "css": "css",
        "rust": "rust",
        "go": "go",
        "java": "java",
        "cpp": "cpp",
        "c": "c",
    }

    def highlight(self, code: str, language: str) -> str:
        """Return syntax-highlighted code for terminal output."""
        lexer_name = self.LANGUAGE_MAP.get(language, "text")
        lexer = get_lexer_by_name(lexer_name)
        formatter = Terminal256Formatter(style="monokai")
        return highlight(code, lexer, formatter)
```

### Streaming Display

The streaming renderer provides smooth, real-time output:

```python
class StreamingRenderer:
    """Handles real-time streaming display with cursor management."""

    def __init__(self, console: Console):
        self.console = console
        self.buffer = ""
        self.line_count = 0

    def append(self, token: str):
        """Append a token to the streaming display."""
        self.buffer += token
        self.console.print(token, end="", highlight=False)

    def flush(self):
        """Flush the display buffer."""
        self.console.file.flush()

    def show_cursor(self):
        """Show the typing cursor."""
        self.console.print("▌", style="dim", end="")

    def hide_cursor(self):
        """Hide the typing cursor."""
        self.console.print("\b \b", end="")
```

---

## Theme and Styling

The Nexus-LLM terminal theme provides a consistent visual identity:

```python
from rich.theme import Theme

nexus_theme = Theme({
    # Primary colors
    "nexus.primary": "#7C3AED",       # Purple
    "nexus.secondary": "#2563EB",     # Blue
    "nexus.accent": "#10B981",        # Green

    # Semantic colors
    "nexus.success": "#22C55E",
    "nexus.warning": "#F59E0B",
    "nexus.error": "#EF4444",
    "nexus.info": "#3B82F6",

    # UI elements
    "nexus.user_prompt": "bold cyan",
    "nexus.assistant": "white",
    "nexus.system": "dim yellow",
    "nexus.command": "bold magenta",
    "nexus.timestamp": "dim",
    "nexus.token_count": "dim blue",
    "nexus.divider": "dim",
})
```
