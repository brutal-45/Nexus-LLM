#!/usr/bin/env python3
"""Custom tool example."""
from nexus_llm.tools import ToolBuilder, ToolManager

def weather(city: str) -> str:
    return f'Weather in {city}: Sunny, 22C'

tool = ToolBuilder().name('weather').description('Get weather').function(weather).param('city', str, True).build()

mgr = ToolManager()
mgr.register(tool)
result = mgr.execute('weather', city='London')
print(result)
