# Safety Configuration Guide

This guide explains how to configure and customize Nexus-LLM's safety features, including content filtering, moderation, toxicity detection, and guardrails.

## Overview

Nexus-LLM provides a multi-layered safety system that inspects both user inputs and model outputs:

```
User Input → [Content Filter] → [Moderation] → [Guardrails] → Model Inference → Output
                                      ↓                                   ↓
                                 [Log/Flag]                    [Toxicity Check] → [Guardrails] → Response
```

All safety components are configurable and can be enabled, disabled, or customized independently.

---

## Quick Start

### Enable Default Safety

By default, Nexus-LLM ships with sensible safety defaults. To explicitly enable all safety features:

```bash
# Start server with safety enabled
nexus-llm serve --model meta-llama/Llama-3.1-8B-Instruct --safety enabled

# Or configure via environment variable
export NEXUS_SAFETY_ENABLED=true
```

### Disable Safety (Development Only)

> **Warning:** Disabling safety is intended for development and testing only. Never disable safety in production environments.

```bash
# Disable all safety checks
nexus-llm serve --model meta-llama/Llama-3.1-8B-Instruct --safety disabled
```

---

## Content Filter

The content filter uses keyword matching, regular expressions, and custom rules to detect and block unwanted content in both inputs and outputs.

### Filter Categories

| Category | Code | Default Action | Description |
|---|---|---|---|
| Profanity | `profanity` | Flag | Profane or vulgar language |
| Hate Speech | `hate_speech` | Block | Hate speech or discriminatory content |
| Violence | `violence` | Block | Violent or threatening content |
| Self-Harm | `self_harm` | Block | Self-harm related content |
| Sexual | `sexual` | Block | Sexually explicit content |
| Harassment | `harassment` | Block | Harassment or bullying content |
| Illegal | `illegal` | Block | Content promoting illegal activities |
| PII | `pii` | Replace | Personally identifiable information |
| Spam | `spam` | Flag | Spam or deceptive content |
| Custom | `custom` | Configurable | User-defined categories |

### Filter Actions

| Action | Description |
|---|---|
| `block` | Reject the request entirely with a `422 Content Filter Error` |
| `flag` | Allow the content but log a flag for review |
| `replace` | Replace matched text with a placeholder (e.g., `[REDACTED]`) |
| `allow` | Explicitly allow the content (used to override other rules) |

### Configuration File

Content filter settings are configured in `nexus_llm/config/safety_config.yaml`:

```yaml
content_filter:
  enabled: true
  mode: "block"              # block, flag, replace

  # Built-in PII detection (regex-based)
  pii_detection:
    enabled: true
    phone_numbers: true
    email_addresses: true
    social_security_numbers: true
    credit_card_numbers: true
    ip_addresses: false
    replacement: "[REDACTED]"

  # Custom keyword blocklists
  keyword_blocklists:
    - name: "profanity_en"
      file: "blocklists/profanity_en.txt"    # One keyword per line
      case_sensitive: false
      whole_word: true
      action: "flag"
    - name: "custom_blocked"
      keywords: ["forbidden_term_1", "forbidden_term_2"]
      action: "block"

  # Custom regex rules
  regex_rules:
    - name: "url_detector"
      pattern: 'https?://[^\s]+'
      action: "replace"
      replacement: "[URL]"
    - name: "api_key_leak"
      pattern: '(?:sk|pk|nxs_sk)_[a-zA-Z0-9]{20,}'
      action: "block"
```

### Managing Filter Rules via CLI

```bash
# List all active filter rules
nexus-llm safety rules list

# Add a keyword blocklist
nexus-llm safety rules add-keyword \
  --name "custom_block" \
  --keywords "bad_word1,bad_word2" \
  --category custom \
  --action block

# Add a regex rule
nexus-llm safety rules add-regex \
  --name "phone_blocker" \
  --pattern '\b\d{3}[-.]?\d{3}[-.]?\d{4}\b' \
  --category pii \
  --action replace \
  --replacement "[PHONE]"

# Remove a rule
nexus-llm safety rules remove --name "custom_block"

# Enable/disable a specific rule
nexus-llm safety rules enable --name "profanity_en"
nexus-llm safety rules disable --name "profanity_en"
```

### Managing Filter Rules via API

```bash
# List rules
curl http://localhost:8000/v1/safety/rules \
  -H "Authorization: Bearer nxs_sk_abc123"

# Add a keyword rule
curl -X POST http://localhost:8000/v1/safety/rules \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "keyword",
    "name": "custom_block",
    "keywords": ["bad_word1", "bad_word2"],
    "category": "custom",
    "action": "block"
  }'

# Test content against filters
curl -X POST http://localhost:8000/v1/safety/check \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, my email is user@example.com and my SSN is 123-45-6789"
  }'
```

**Check Response:**

```json
{
  "is_safe": true,
  "was_modified": true,
  "filtered_text": "Hello, my email is [REDACTED] and my SSN is [REDACTED]",
  "blocked_categories": [],
  "flagged_categories": [],
  "matches": [
    {
      "rule_name": "pii_email",
      "category": "pii",
      "action": "replace",
      "confidence": 1.0
    },
    {
      "rule_name": "pii_ssn",
      "category": "pii",
      "action": "replace",
      "confidence": 1.0
    }
  ]
}
```

---

## Moderation

The moderation system uses a classifier model to detect harmful content beyond simple keyword and regex matching. It can identify subtle forms of harmful content that rule-based filters miss.

### Configuration

```yaml
moderation:
  enabled: true
  model: "meta-llama/Llama-Guard-3-8B"    # Moderation classifier model
  device: "auto"                           # Device for the moderation model

  # Check both input and output
  check_input: true
  check_output: true

  # Categories to monitor
  categories:
    - violence
    - hate_speech
    - self_harm
    - sexual_content
    - harassment
    - illegal_activity

  # Thresholds per category (0.0 - 1.0)
  thresholds:
    violence: 0.5
    hate_speech: 0.5
    self_harm: 0.3
    sexual_content: 0.5
    harassment: 0.5
    illegal_activity: 0.5

  # Action when threshold exceeded
  action: "block"              # block, flag, replace

  # Caching moderation results
  cache:
    enabled: true
    ttl: 3600
    max_size: 10000
```

### Customizing Thresholds

Lower thresholds are more restrictive (more content flagged), higher thresholds are more permissive:

```bash
# Set a stricter threshold for self-harm
nexus-llm safety moderation set-threshold --category self_harm --value 0.2

# Set a more permissive threshold for violence (for creative writing apps)
nexus-llm safety moderation set-threshold --category violence --value 0.8
```

### Via API

```bash
curl -X POST http://localhost:8000/v1/config \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "safety": {
        "moderation": {
          "thresholds": {
            "self_harm": 0.2,
            "violence": 0.8
          }
        }
      }
    }
  }'
```

---

## Toxicity Detection

The toxicity detector provides real-time scoring of generated content. Unlike moderation which uses a classifier, toxicity scoring produces a numerical score that can be used for soft filtering.

### Configuration

```yaml
toxicity:
  enabled: true
  model: "unitary/toxic-bert"       # Toxicity classifier model

  # Overall toxicity threshold
  threshold: 0.7

  # Sub-category thresholds
  categories:
    toxic: 0.7
    severe_toxic: 0.5
    obscene: 0.7
    threat: 0.5
    insult: 0.7
    identity_hate: 0.5

  # Action when threshold exceeded
  action: "regenerate"             # block, flag, regenerate

  # Regeneration settings (when action is "regenerate")
  max_retries: 3
  retry_temperature_adjustment: -0.1   # Lower temperature each retry

  # Include toxicity scores in response metadata
  include_scores: false
```

### Regeneration Strategy

When `action` is set to `regenerate`, the system will attempt to generate a new response if the first one exceeds the toxicity threshold:

1. First attempt uses the original parameters
2. Each subsequent attempt reduces temperature by `retry_temperature_adjustment`
3. After `max_retries` failed attempts, the request is blocked with a `422` error

### Including Toxicity Scores

For debugging and monitoring, you can include toxicity scores in the response metadata:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -H "X-Nexus-Include-Safety-Scores: true" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Response includes a `safety` field:

```json
{
  "id": "chatcmpl-abc123",
  "choices": [...],
  "safety": {
    "content_filter": {"is_safe": true, "matches": 0},
    "moderation": {"is_safe": true, "max_score": 0.02, "flagged_categories": []},
    "toxicity": {"is_safe": true, "toxic_score": 0.01, "categories": {"toxic": 0.01, "obscene": 0.00}}
  }
}
```

---

## Guardrails

Guardrails enforce structural and topical constraints on model behavior. They go beyond content filtering by ensuring the model stays within defined boundaries.

### Configuration

```yaml
guardrails:
  enabled: true

  # Topic restrictions
  topics:
    allowed: []                   # Empty = all topics allowed
    blocked:
      - "financial_advice"
      - "medical_diagnosis"
      - "legal_counsel"

  # Output format constraints
  output_format:
    # Enforce JSON output when requested
    enforce_json: true
    # Maximum response length
    max_length: 4096
    # Minimum response length
    min_length: 1

  # Input validation
  input:
    # Maximum prompt length
    max_prompt_length: 32768
    # Reject empty prompts
    reject_empty: true
    # Reject prompts that are only whitespace
    reject_whitespace: true

  # Language restrictions
  languages:
    allowed: []                   # Empty = all languages allowed
    # blocked: ["zh", "ko"]       # Block specific language codes

  # Pattern-based output validation
  output_patterns:
    # Block outputs that look like API keys or tokens
    - name: "api_key_leak"
      pattern: '(?:sk|pk|nxs_sk)_[a-zA-Z0-9]{20,}'
      action: "replace"
      replacement: "[API_KEY_REDACTED]"

    # Block outputs that look like IP addresses
    - name: "ip_address_leak"
      pattern: '\b(?:\d{1,3}\.){3}\d{1,3}\b'
      action: "replace"
      replacement: "[IP_REDACTED]"

  # Custom guardrail scripts (Python)
  custom_guardrails:
    - name: "no_code_execution"
      module: "nexus_llm.safety.custom.no_code_exec"
      config:
        blocked_patterns:
          - "os.system"
          - "subprocess"
          - "eval("
          - "exec("
```

### Topic Guardrails

Topic guardrails prevent the model from engaging in conversations about specific subjects:

```bash
# Block financial advice topics
nexus-llm safety guardrails add-topic \
  --topic "financial_advice" \
  --action block \
  --message "I cannot provide financial advice. Please consult a qualified financial advisor."

# Block medical diagnosis topics
nexus-llm safety guardrails add-topic \
  --topic "medical_diagnosis" \
  --action block \
  --message "I cannot provide medical diagnoses. Please consult a healthcare professional."
```

### Custom Guardrails

Create custom guardrails by implementing a Python module:

```python
# nexus_llm/safety/custom/no_code_exec.py

from typing import Any, Dict

class NoCodeExecutionGuardrail:
    """Prevents the model from generating code that could execute system commands."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.blocked_patterns = self.config.get("blocked_patterns", [
            "os.system", "subprocess", "eval(", "exec("
        ])

    def check_input(self, text: str) -> Dict[str, Any]:
        """Check user input."""
        return {"is_safe": True, "reason": None}

    def check_output(self, text: str) -> Dict[str, Any]:
        """Check model output for dangerous code patterns."""
        for pattern in self.blocked_patterns:
            if pattern in text:
                return {
                    "is_safe": False,
                    "reason": f"Output contains potentially dangerous pattern: {pattern}",
                    "action": "flag"
                }
        return {"is_safe": True, "reason": None}
```

---

## Safety Policies

Safety policies define preset configurations for different deployment scenarios. They allow you to quickly switch between safety profiles.

### Available Policies

| Policy | Description | Use Case |
|---|---|---|
| `strict` | All safety features enabled with low thresholds | Production public-facing apps |
| `standard` | Balanced safety with moderate thresholds | Internal tools, development |
| `permissive` | Minimal safety checks with high thresholds | Research, creative applications |
| `disabled` | All safety checks disabled | Local development only |

### Applying a Policy

```bash
# Apply a strict safety policy
nexus-llm safety policy apply strict

# Apply a standard policy
nexus-llm safety policy apply standard

# Check current policy
nexus-llm safety policy current
```

### Policy Definitions

Policies are defined in `safety_config.yaml`:

```yaml
policies:
  strict:
    content_filter:
      enabled: true
      mode: "block"
    moderation:
      enabled: true
      check_input: true
      check_output: true
      action: "block"
    toxicity:
      enabled: true
      threshold: 0.5
      action: "regenerate"
    guardrails:
      enabled: true
      topics:
        blocked: ["financial_advice", "medical_diagnosis", "legal_counsel"]

  standard:
    content_filter:
      enabled: true
      mode: "flag"
    moderation:
      enabled: true
      check_input: true
      check_output: false
      action: "flag"
    toxicity:
      enabled: true
      threshold: 0.7
      action: "flag"
    guardrails:
      enabled: true

  permissive:
    content_filter:
      enabled: true
      mode: "flag"
      pii_detection:
        enabled: true
    moderation:
      enabled: false
    toxicity:
      enabled: true
      threshold: 0.9
      action: "flag"
    guardrails:
      enabled: false
```

---

## Auditing and Logging

### Safety Event Logging

All safety events are logged for auditing and review:

```bash
# View recent safety events
nexus-llm safety logs --limit 50

# Filter by category
nexus-llm safety logs --category hate_speech --limit 20

# Filter by action
nexus-llm safety logs --action blocked --since "2024-01-15"
```

### Log Format

Safety events are logged with the following structure:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "event_type": "content_filtered",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "source": "input",
  "rule_name": "pii_email",
  "category": "pii",
  "action": "replace",
  "confidence": 1.0,
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "user_id": "user-abc",
  "project_id": null
}
```

### Integration with Monitoring

Safety events feed into the monitoring system for dashboards and alerts:

```bash
# Check safety metrics
nexus-llm safety metrics

# Output:
# Total requests:     10,000
# Blocked:           45 (0.45%)
# Flagged:          120 (1.20%)
# Modified:          89 (0.89%)
# Top blocked:       hate_speech (20), violence (15), self_harm (10)
# Top flagged:       profanity (50), spam (40), pii (30)
```

---

## Programmatic Access

### Python SDK

```python
from nexus_llm import NexusClient
from nexus_llm.safety import ContentFilter, FilterAction, FilterCategory

client = NexusClient()

# Check text safety
result = client.safety.check("Hello, my email is user@example.com")
print(result.is_safe)         # True (PII was replaced, not blocked)
print(result.filtered_text)   # "Hello, my email is [REDACTED]"
print(result.was_modified)    # True

# Add a custom keyword blocklist
client.safety.add_keyword_blocklist(
    name="custom_block",
    keywords=["forbidden_term"],
    category=FilterCategory.CUSTOM,
    action=FilterAction.BLOCK
)

# Check moderation
mod_result = client.safety.moderate("Some potentially harmful text")
print(mod_result.is_safe)
print(mod_result.flagged_categories)

# Get toxicity scores
tox_result = client.safety.toxicity_score("Some text to check")
print(tox_result.toxic_score)
```

### Async Safety Checks

For high-throughput applications, use async safety checking:

```python
import asyncio
from nexus_llm import AsyncNexusClient

async def process_with_safety():
    client = AsyncNexusClient()

    texts = ["text1", "text2", "text3"]
    results = await asyncio.gather(*[
        client.safety.check(text) for text in texts
    ])

    for text, result in zip(texts, results):
        if not result.is_safe:
            print(f"Blocked: {text}")
        elif result.was_modified:
            print(f"Modified: {text} → {result.filtered_text}")

asyncio.run(process_with_safety())
```

---

## Best Practices

1. **Start with the `standard` policy** and adjust based on your use case
2. **Monitor safety metrics** regularly to identify emerging patterns
3. **Use `flag` mode during development** to understand what content is caught before switching to `block`
4. **Regularly review flagged content** to tune thresholds and reduce false positives
5. **Enable PII detection** in all environments that handle user data
6. **Keep moderation models updated** for best classification accuracy
7. **Layer your defenses** — use content filter + moderation + guardrails together
8. **Test with adversarial inputs** before deploying to production
9. **Document your safety configuration** and change history
10. **Respect user privacy** — minimize logging of flagged content

---

## Related Documentation

- [Error Codes Reference](../api/errors.md) — Content filter error codes
- [Monitoring Guide](monitoring.md) — Safety metrics and alerting
- [Configuration Guide](configuration.md) — General configuration reference
- [REST API Reference](../api/rest.md) — API endpoint documentation
