"""
Nexus Tools System
===================
Function calling system for LLM-based agents.

This module provides a comprehensive framework for defining, parsing, validating,
and executing function calls from LLM outputs. It supports:

- **Function Registration**: Register functions with JSON Schema parameter definitions
- **Call Parsing**: Parse structured function calls from LLM text output
- **Execution**: Execute calls with timeout, retry, and error handling
- **Parallel Execution**: Execute multiple independent calls concurrently
- **Prompt Building**: Construct system/user messages with function schemas
- **Tool Choice Policies**: Control when and how tools are selected (auto/required/none)
- **Built-in Functions**: Calculator, date/time, unit conversion, text processing,
  JSON utilities, and safe code evaluation

Architecture
------------
The function calling pipeline follows these stages:

1. **Schema Definition**: Functions are defined with FunctionSchema objects
   containing name, description, JSON Schema parameters, and examples.

2. **Registration**: Schemas are registered in a FunctionRegistry, which
   validates schemas and provides lookup capabilities.

3. **Prompt Construction**: FunctionCallingPromptBuilder generates system and
   user messages that describe available functions to the LLM.

4. **LLM Generation**: The LLM produces output that may contain function calls.

5. **Parsing**: FunctionParser extracts structured FunctionCall objects from
   the LLM's text output.

6. **Validation**: Each parsed call is validated against its registered schema.

7. **Execution**: FunctionExecutor runs validated calls with timeout and retry.

8. **Result Collection**: FunctionResult objects are returned with success/error
   status and execution metadata.

Key Classes
-----------
- FunctionSchema — Describes a callable function
- FunctionCall — A parsed function call request
- FunctionResult — Result of executing a function call
- FunctionRegistry — Registry of available functions
- FunctionParser — Parses LLM output into FunctionCalls
- FunctionExecutor — Executes function calls safely
- ParallelExecutor — Concurrent execution of independent calls
- FunctionCallingPromptBuilder — Builds prompts with function schemas
- ToolChoicePolicy — Controls tool selection behavior
- BuiltinFunctions — Collection of useful built-in tools

Quick Start
-----------
    from nexus.tools_system import (
        FunctionRegistry, FunctionSchema, FunctionParser,
        FunctionExecutor, FunctionCallingPromptBuilder,
        BuiltinFunctions,
    )

    # Create registry and register built-in functions
    registry = FunctionRegistry()
    builtins = BuiltinFunctions(registry)
    builtins.register_all()

    # Also register a custom function
    registry.register(FunctionSchema(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            },
            "required": ["location"],
        },
    ))

    # Build prompt
    builder = FunctionCallingPromptBuilder(registry)
    messages = builder.build_system_message()

    # Parse LLM output
    parser = FunctionParser(registry)
    calls = parser.parse('{"name": "get_weather", "arguments": {"location": "Tokyo"}}')

    # Execute
    executor = FunctionExecutor(registry)
    results = executor.execute(calls[0])
"""

__version__ = "1.0.0"
__author__ = "Nexus LLM Team"

# ─── Core Classes ─────────────────────────────────────────────────────────────
from nexus.tools_system.function_calling import (
    # Schema & Data Classes
    FunctionSchema,
    FunctionCall,
    FunctionResult,
    # Registry
    FunctionRegistry,
    # Parsing
    FunctionParser,
    # Execution
    FunctionExecutor,
    ParallelExecutor,
    # Prompt Building
    FunctionCallingPromptBuilder,
    # Tool Choice
    ToolChoicePolicy,
    ToolChoice,
    # Built-in Functions
    BuiltinFunctions,
    CalculatorFunction,
    DateTimeFunction,
    UnitConversionFunction,
    TextProcessingFunction,
    JsonUtilsFunction,
    CodeEvalFunction,
)

# ─── Public API ───────────────────────────────────────────────────────────────
__all__ = [
    # Schema & Data Classes
    "FunctionSchema",
    "FunctionCall",
    "FunctionResult",
    # Registry
    "FunctionRegistry",
    # Parsing
    "FunctionParser",
    # Execution
    "FunctionExecutor",
    "ParallelExecutor",
    # Prompt Building
    "FunctionCallingPromptBuilder",
    # Tool Choice
    "ToolChoicePolicy",
    "ToolChoice",
    # Built-in Functions
    "BuiltinFunctions",
    "CalculatorFunction",
    "DateTimeFunction",
    "UnitConversionFunction",
    "TextProcessingFunction",
    "JsonUtilsFunction",
    "CodeEvalFunction",
]


# ─── Factory Helpers ─────────────────────────────────────────────────────────

def create_function_registry(with_builtins: bool = True) -> FunctionRegistry:
    """Create a function registry with optional built-in functions.

    Args:
        with_builtins: Whether to register all built-in functions.

    Returns:
        Configured FunctionRegistry instance.

    Example:
        >>> registry = create_function_registry(with_builtins=True)
        >>> registry.list_functions()
        ['calculator', 'date_time', 'unit_conversion', ...]
    """
    registry = FunctionRegistry()
    if with_builtins:
        builtins = BuiltinFunctions(registry)
        builtins.register_all()
    return registry


def create_executor(
    registry: Optional[FunctionRegistry] = None,
    timeout: float = 30.0,
    max_retries: int = 2,
    with_builtins: bool = True,
) -> FunctionExecutor:
    """Create a configured function executor.

    Args:
        registry: Function registry. Created with builtins if None.
        timeout: Default timeout per function call in seconds.
        max_retries: Default maximum retry attempts on failure.
        with_builtins: Whether to include built-in functions.

    Returns:
        Configured FunctionExecutor instance.

    Example:
        >>> executor = create_executor(timeout=10.0)
        >>> result = executor.execute_by_name("calculator", {"expression": "2 + 2"})
        >>> result.output
        '4'
    """
    if registry is None:
        registry = create_function_registry(with_builtins=with_builtins)
    return FunctionExecutor(
        registry=registry,
        default_timeout=timeout,
        max_retries=max_retries,
    )


def create_prompt_builder(
    registry: Optional[FunctionRegistry] = None,
    with_builtins: bool = True,
) -> FunctionCallingPromptBuilder:
    """Create a prompt builder for function calling.

    Args:
        registry: Function registry. Created with builtins if None.
        with_builtins: Whether to include built-in functions.

    Returns:
        Configured FunctionCallingPromptBuilder instance.

    Example:
        >>> builder = create_prompt_builder()
        >>> messages = builder.build_system_message()
        >>> print(messages)
    """
    if registry is None:
        registry = create_function_registry(with_builtins=with_builtins)
    return FunctionCallingPromptBuilder(registry=registry)


# ─── Version Info ────────────────────────────────────────────────────────────

def get_version() -> str:
    """Return the current version of the tools system module."""
    return __version__


def get_module_info() -> dict:
    """Return detailed information about the tools system module.

    Returns:
        Dictionary containing version, components, and exports.
    """
    return {
        "version": __version__,
        "author": __author__,
        "exports": sorted(__all__),
        "factory_functions": [
            "create_function_registry",
            "create_executor",
            "create_prompt_builder",
        ],
    }
