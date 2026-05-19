---
name: Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] "
labels: bug, triage
assignees: ""
---

## Bug Description

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Start the server with `...`
2. Send a request to `...`
3. See error

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

A clear and concise description of what actually happened.

## Error Logs

```
Paste relevant error logs here. Include full stack traces if available.
```

## Environment

- **Nexus-LLM Version**: [e.g., 1.0.0]
- **Python Version**: [e.g., 3.11.5]
- **OS**: [e.g., Ubuntu 22.04]
- **CUDA Version**: [e.g., 12.1]
- **GPU**: [e.g., NVIDIA A100 80GB]
- **Model**: [e.g., meta-llama/Llama-3.1-8B-Instruct]
- **Quantization**: [e.g., AWQ 4-bit / None]
- **Installation Method**: [pip / Docker / from source]

## Configuration

```yaml
# Paste your nexus_config.yaml or relevant configuration here
```

## Additional Context

Add any other context about the problem here.

- Is this a regression? (Did it work in a previous version?)
- Does it happen consistently or intermittently?
- Have you tried any workarounds?

## Minimal Reproducible Example

```python
# If possible, provide a minimal code example that reproduces the bug
from nexus_llm import NexusClient

client = NexusClient(...)
# ...
```
