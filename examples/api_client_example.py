#!/usr/bin/env python3
"""API client example."""
from nexus_llm.api.client import NexusClient

client = NexusClient('http://localhost:8000')
result = client.chat([{'role': 'user', 'content': 'Hello!'}])
print(result)
