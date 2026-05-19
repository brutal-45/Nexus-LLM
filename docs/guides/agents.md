# Agents Guide

Learn how to configure and use AI agents with Nexus-LLM. Agents combine language models with tools, planning, and memory to perform complex tasks autonomously.

---

## What Are Agents?

An agent is an LLM-powered system that can:

1. **Reason** about what steps to take
2. **Use tools** to interact with the outside world (web search, code execution, file access)
3. **Plan** multi-step workflows
4. **Remember** past interactions and context
5. **Self-correct** when things go wrong

### Agent Architecture

```
┌──────────────────────────────────────────────┐
│                  Agent                        │
│                                               │
│  ┌─────────┐  ┌─────────┐  ┌──────────────┐ │
│  │  LLM     │  │ Memory  │  │  Planner     │ │
│  │  (Brain) │  │ (State) │  │  (Strategy)  │ │
│  └────┬─────┘  └────┬────┘  └──────┬───────┘ │
│       │             │              │          │
│       └─────────────┼──────────────┘          │
│                     │                         │
│              ┌──────┴───────┐                 │
│              │  Tool Router  │                │
│              └──────┬───────┘                 │
│                     │                         │
│    ┌────────┬───────┼────────┬──────────┐     │
│    ▼        ▼       ▼        ▼          ▼     │
│  Search  Execute  Read     API      Database  │
│  Tool    Code     Files    Call     Query     │
└──────────────────────────────────────────────┘
```

---

## Built-in Agents

Nexus-LLM comes with several pre-configured agents:

### Research Agent

Searches the web and synthesizes information on a topic.

```yaml
agents:
  research:
    model: "meta-llama/Llama-3.1-8B-Instruct"
    system_prompt: |
      You are a research assistant. When given a topic:
      1. Break it down into key questions
      2. Search for relevant information
      3. Synthesize findings into a comprehensive answer
      4. Cite your sources
    tools:
      - web_search
      - web_reader
    max_iterations: 5
    memory:
      type: "conversation"
      max_messages: 20
```

```bash
# Use the research agent
./scripts/run.sh --mode chat --agent research

# Or in chat mode
/agent research
```

### Code Agent

Writes, executes, and debugs code.

```yaml
agents:
  coder:
    model: "meta-llama/Llama-3.1-8B-Instruct"
    system_prompt: |
      You are a coding assistant. You can write, execute, and debug code.
      Always explain your approach before writing code.
      When code fails, analyze the error and fix it.
    tools:
      - code_execute
      - file_read
      - file_write
      - shell_execute
    max_iterations: 10
    code_execution:
      languages: ["python", "javascript", "bash"]
      timeout_seconds: 30
      sandbox: true
```

### Data Analysis Agent

Analyzes datasets and generates insights.

```yaml
agents:
  analyst:
    model: "meta-llama/Llama-3.1-8B-Instruct"
    system_prompt: |
      You are a data analyst. When given data:
      1. Explore and understand the data structure
      2. Compute summary statistics
      3. Generate visualizations when appropriate
      4. Provide actionable insights
    tools:
      - code_execute
      - file_read
      - shell_execute
    max_iterations: 15
```

### Writing Agent

Creates and refines written content.

```yaml
agents:
  writer:
    model: "meta-llama/Llama-3.1-8B-Instruct"
    system_prompt: |
      You are a skilled writer. You can:
      - Draft articles, emails, and documents
      - Edit and improve existing text
      - Adapt tone and style to the audience
      - Research topics when needed
    tools:
      - web_search
      - file_read
      - file_write
    max_iterations: 5
```

---

## Using Tools

Tools give agents the ability to interact with the outside world.

### Available Tools

| Tool | Description | Required Config |
|------|-------------|-----------------|
| `web_search` | Search the web for information | `NEXUS_SEARCH_API_KEY` |
| `web_reader` | Read and extract content from URLs | — |
| `code_execute` | Execute code in a sandbox | — |
| `file_read` | Read files from the filesystem | Allowed paths |
| `file_write` | Write files to the filesystem | Allowed paths |
| `shell_execute` | Run shell commands | Allowed commands |
| `database_query` | Query SQL databases | Database connection |
| `api_call` | Make HTTP API calls | — |
| `calculator` | Evaluate mathematical expressions | — |
| `rag_search` | Search RAG document collections | RAG enabled |

### Tool Configuration

```yaml
tools:
  web_search:
    provider: "serper"        # serper, google, bing
    api_key: "${NEXUS_SEARCH_API_KEY}"
    max_results: 10

  code_execute:
    languages: ["python", "javascript"]
    timeout_seconds: 30
    sandbox: true
    max_output_length: 10000
    allowed_imports:
      - numpy
      - pandas
      - matplotlib
      - sklearn

  file_read:
    allowed_paths:
      - "./data"
      - "./documents"
    max_file_size_mb: 10

  file_write:
    allowed_paths:
      - "./output"
    max_file_size_mb: 10

  shell_execute:
    allowed_commands:
      - "ls"
      - "cat"
      - "head"
      - "tail"
      - "wc"
      - "python3"
    blocked_commands:
      - "rm -rf /"
      - "sudo"
    timeout_seconds: 30

  database_query:
    url: "sqlite:///./data/app.db"
    read_only: true
    max_rows: 1000
```

### Function Calling

The agent uses the model's function calling capability to invoke tools:

```python
# Example: agent deciding to use a tool
# User: "What's the weather in Tokyo?"

# Model's internal reasoning:
# "I need to look up the current weather. I'll use the web_search tool."

# Function call:
{
    "name": "web_search",
    "arguments": {
        "query": "current weather Tokyo"
    }
}

# Tool result:
{
    "results": [
        "Tokyo: 18°C, Partly Cloudy, Humidity 65%, Wind 12 km/h"
    ]
}

# Model's final response:
"The current weather in Tokyo is 18°C (64°F) with partly cloudy skies.
Humidity is at 65% with winds of 12 km/h."
```

---

## Creating Custom Agents

### YAML Configuration

Create a custom agent by adding it to `config/agents/`:

```yaml
# config/agents/support_bot.yaml
name: "support_bot"
description: "Customer support agent that answers questions and creates tickets"

model: "meta-llama/Llama-3.1-8B-Instruct"

system_prompt: |
  You are a customer support agent for Acme Corp. Your responsibilities:
  1. Answer customer questions using the knowledge base
  2. Create support tickets for issues you can't resolve
  3. Escalate urgent issues to human agents
  4. Always be polite, professional, and helpful

  When you don't know an answer, search the knowledge base first.
  If you still can't find the answer, create a support ticket.

tools:
  - rag_search
  - web_search
  - api_call

max_iterations: 8

tool_config:
  rag_search:
    collection: "support_kb"
    top_k: 5
  api_call:
    base_url: "https://api.acme.com"
    headers:
      Authorization: "Bearer ${ACME_API_KEY}"

memory:
  type: "conversation"
  max_messages: 30

guardrails:
  max_tokens_per_response: 2048
  blocked_topics: ["competitors", "pricing_internal"]
  require_citation: true
```

### Python Agent Class

For more complex agent logic, create a Python agent:

```python
# config/agents/custom_agent.py
from nexus_llm.agents import Agent, tool, step

class SupportAgent(Agent):
    """Custom customer support agent."""

    name = "support_bot"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    system_prompt = """You are a customer support agent. Help users with their questions."""

    def __init__(self, config):
        super().__init__(config)
        self.ticket_counter = 0

    @tool(description="Search the knowledge base for answers")
    def search_kb(self, query: str) -> str:
        """Search the support knowledge base."""
        results = self.rag.search(query, collection="support_kb", top_k=3)
        return "\n".join([r.content for r in results])

    @tool(description="Create a support ticket for unresolved issues")
    def create_ticket(self, title: str, description: str, priority: str = "normal") -> str:
        """Create a support ticket in the system."""
        self.ticket_counter += 1
        ticket_id = f"TKT-{self.ticket_counter:04d}"
        # In production, this would call your ticketing API
        self.logger.info(f"Created ticket {ticket_id}: {title} [{priority}]")
        return f"Support ticket {ticket_id} created. A team member will follow up within 24 hours."

    @tool(description="Escalate an issue to a human agent")
    def escalate(self, reason: str, context: str) -> str:
        """Escalate to a human support agent."""
        self.logger.warning(f"Escalation requested: {reason}")
        return "Your issue has been escalated to a senior support agent. They will contact you shortly."

    @step(after="response")
    def add_disclaimer(self, context):
        """Append a disclaimer to every response."""
        context.response += "\n\n---\n*This response was generated by Acme Support AI. For urgent issues, call 1-800-ACME.*"
        return context
```

Register the Python agent:

```yaml
# config/default.yaml
agents:
  custom:
    module: "config.agents.custom_agent"
    class: "SupportAgent"
    config:
      rag_collection: "support_kb"
```

---

## Agent Memory

### Conversation Memory

Short-term memory within a conversation:

```yaml
memory:
  type: "conversation"
  max_messages: 30         # Keep last N messages
  summarize_at: 25         # Summarize when approaching limit
```

### Long-Term Memory

Persistent memory across conversations:

```yaml
memory:
  type: "persistent"
  backend: "sqlite"        # sqlite, postgres, redis
  database_url: "sqlite:///./cache/agent_memory.db"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  max_memories: 1000
  relevance_threshold: 0.7
```

### Working Memory

Scratchpad for the agent to track progress:

```yaml
memory:
  working_memory:
    enabled: true
    max_items: 20
```

---

## Multi-Agent Systems

Orchestrate multiple agents working together:

```yaml
# config/agents/team.yaml
name: "engineering_team"
type: "orchestrator"

orchestrator:
  model: "meta-llama/Llama-3.1-8B-Instruct"
  system_prompt: |
    You are an engineering team lead. Delegate tasks to the appropriate
    specialist and synthesize their work into a final deliverable.

agents:
  - name: "architect"
    role: "Design system architecture and APIs"
    model: "meta-llama/Llama-3.1-8B-Instruct"

  - name: "developer"
    role: "Write and test code"
    model: "meta-llama/Llama-3.1-8B-Instruct"
    tools: ["code_execute", "file_write"]

  - name: "reviewer"
    role: "Review code and suggest improvements"
    model: "meta-llama/Llama-3.1-8B-Instruct"

workflow:
  - step: "architect"
    input: "${user_request}"
    output: "architecture_plan"

  - step: "developer"
    input: "${architecture_plan}"
    output: "code_implementation"

  - step: "reviewer"
    input: "${code_implementation}"
    output: "review_feedback"

  - step: "developer"
    input: "${review_feedback}"
    output: "final_implementation"
```

---

## Guardrails

Configure safety and behavior guardrails for agents:

```yaml
guardrails:
  # Token limits
  max_tokens_per_response: 2048
  max_total_tokens: 100000

  # Content filtering
  blocked_topics: []
  blocked_words: []
  content_filter_model: null

  # Behavior constraints
  require_citation: false        # Must cite sources
  no_self_harm: true             # Block self-harm content
  no_illegal: true               # Block illegal content

  # Tool restrictions
  max_tool_calls: 20
  max_consecutive_failures: 3    # Stop after N consecutive tool failures

  # Human-in-the-loop
  human_approval:
    enabled: false
    for_actions: ["shell_execute", "file_write"]
    timeout_seconds: 300
```
