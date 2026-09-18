# Agents Tutorial

How to build agents with tools.

## Creating an Agent

```python
from nexus_llm.agents import Agent, ToolRegistry

agent = Agent(tools=ToolRegistry())
result = agent.run('Calculate 2+2')
```
