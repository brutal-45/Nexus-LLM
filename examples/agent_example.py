#!/usr/bin/env python3
"""
Agent with Tools Example - Nexus-LLM
======================================
Demonstrates how to build an AI agent that can use external tools
to accomplish tasks through multi-step reasoning.
"""

import json
from nexus_llm import InferenceEngine
from nexus_llm.agents import Agent, Tool, ToolRegistry, AgentConfig


# --- Define custom tools ---

class CalculatorTool(Tool):
    """A simple calculator tool for mathematical expressions."""

    name = "calculator"
    description = "Evaluates mathematical expressions. Input should be a valid Python math expression."

    def run(self, expression: str) -> str:
        try:
            # Safe evaluation with limited builtins
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "pow": pow, "sum": sum,
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return json.dumps({"result": result, "expression": expression})
        except Exception as e:
            return json.dumps({"error": str(e)})


class WeatherTool(Tool):
    """Simulated weather lookup tool."""

    name = "weather"
    description = "Gets the current weather for a given city. Input should be a city name."

    # Simulated weather data
    WEATHER_DATA = {
        "new york": {"temp": 72, "condition": "Partly Cloudy", "humidity": 65},
        "london": {"temp": 59, "condition": "Rainy", "humidity": 80},
        "tokyo": {"temp": 77, "condition": "Sunny", "humidity": 55},
        "paris": {"temp": 64, "condition": "Overcast", "humidity": 72},
    }

    def run(self, city: str) -> str:
        city_lower = city.lower()
        if city_lower in self.WEATHER_DATA:
            data = self.WEATHER_DATA[city_lower]
            return json.dumps({"city": city, **data})
        return json.dumps({"error": f"Weather data not available for {city}"})


class SearchTool(Tool):
    """Simulated web search tool."""

    name = "web_search"
    description = "Searches the web for information. Input should be a search query string."

    def run(self, query: str) -> str:
        # Simulated search results
        return json.dumps({
            "query": query,
            "results": [
                {"title": f"Result for: {query}", "snippet": "This is a simulated search result."},
            ]
        })


def main():
    # Register tools
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WeatherTool())
    registry.register(SearchTool())

    # Configure the agent
    agent_config = AgentConfig(
        max_iterations=10,          # Maximum reasoning steps
        max_tokens_per_step=512,    # Max tokens per agent step
        verbose=True,               # Print reasoning steps
        allow_parallel_tools=False, # Execute tools sequentially
        stop_on_error=True,         # Stop if a tool errors
    )

    # Create the agent
    engine = InferenceEngine(model_name="nexus-7b-chat", device="auto")
    agent = Agent(
        inference_engine=engine,
        tool_registry=registry,
        config=agent_config,
    )

    # --- Run agent tasks ---

    print("=" * 60)
    print("Task 1: Weather-based calculation")
    print("=" * 60)
    result = agent.run(
        "What is the temperature in Tokyo? Then calculate the temperature "
        "in Celsius (subtract 32, multiply by 5/9)."
    )
    print(f"\nFinal Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    print(f"Tools used: {result.tools_used}")
    print()

    print("=" * 60)
    print("Task 2: Multi-step research")
    print("=" * 60)
    result = agent.run(
        "Search for information about quantum computing, then explain "
        "it in simple terms using an analogy."
    )
    print(f"\nFinal Answer: {result.answer}")
    print(f"Steps taken: {result.num_steps}")
    print(f"Tools used: {result.tools_used}")
    print()

    # --- Streaming agent output ---
    print("=" * 60)
    print("Task 3: Streaming agent")
    print("=" * 60)
    for step in agent.run_stream("Calculate 15^3 and tell me the weather in London."):
        if step.type == "thinking":
            print(f"[Thinking] {step.content}")
        elif step.type == "tool_call":
            print(f"[Tool Call] {step.tool_name}({step.input})")
        elif step.type == "tool_result":
            print(f"[Tool Result] {step.output}")
        elif step.type == "answer":
            print(f"[Answer] {step.content}")


if __name__ == "__main__":
    main()
