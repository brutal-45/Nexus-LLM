# Tools Guide

This guide covers how to use and create tools in the Nexus-LLM framework.

## Overview

Tools are the building blocks of agent capabilities in Nexus-LLM. Each tool encapsulates a specific functionality - from simple calculations to complex API integrations.

## Built-in Tools

### Calculator

```python
from nexus_llm.tools.calculator import CalculatorTool

tool = CalculatorTool()
result = tool.run(expression="2 + 3 * 4")
print(result.output)  # 14
```

### JSON Tool

```python
from nexus_llm.tools.json_tool import JsonTool

tool = JsonTool()
result = tool.run(operation="parse", data='{"key": "value"}')
print(result.output)  # {"key": "value"}
```

### Text Tool

```python
from nexus_llm.tools.text_tool import TextTool

tool = TextTool()
result = tool.run(operation="upper", text="hello")
print(result.output)  # "HELLO"
```

### Search Tool

```python
from nexus_llm.tools.search import SearchTool

tool = SearchTool()
result = tool.run(query="machine learning", max_results=5)
```

### File Operations

```python
from nexus_llm.tools.file_ops import FileOpsTool

tool = FileOpsTool()
result = tool.run(operation="read", path="/path/to/file.txt")
```

### DateTime Tool

```python
from nexus_llm.tools.datetime_tool import DatetimeTool

tool = DatetimeTool()
result = tool.run(operation="now")
```

### Math Tool

```python
from nexus_llm.tools.math_tool import MathTool

tool = MathTool()
result = tool.run(operation="sqrt", value=16)
print(result.output)  # 4.0
```

### CSV Tool

```python
from nexus_llm.tools.csv_tool import CsvTool

tool = CsvTool()
result = tool.run(operation="read", data="name,age\nAlice,30")
```

### Translator Tool

```python
from nexus_llm.tools.translator import TranslatorTool

tool = TranslatorTool()
result = tool.run(text="Hello", source_lang="en", target_lang="es")
```

### Code Linter

```python
from nexus_llm.tools.code_linter import CodeLinterTool

tool = CodeLinterTool()
result = tool.run(code="def hello():\n    print('hi')", language="python")
```

### Regex Tool

```python
from nexus_llm.tools.regex_tool import RegexTool

tool = RegexTool()
result = tool.run(operation="findall", text="Hello 123", pattern=r"\d+")
```

### Statistics Tool

```python
from nexus_llm.tools.stats_tool import StatsTool

tool = StatsTool()
result = tool.run(operation="describe", data=[1, 2, 3, 4, 5])
```

### Unit Converter

```python
from nexus_llm.tools.unit_converter import UnitConverterTool

tool = UnitConverterTool()
result = tool.run(value=100, from_unit="km", to_unit="mi", category="length")
```

### Diff Tool

```python
from nexus_llm.tools.diff_tool import DiffTool

tool = DiffTool()
result = tool.run(text1="hello world", text2="hello earth")
```

### XML Tool

```python
from nexus_llm.tools.xml_tool import XmlTool

tool = XmlTool()
result = tool.run(operation="parse", data="<root><item>hello</item></root>")
```

### YAML Tool

```python
from nexus_llm.tools.yaml_tool import YamlTool

tool = YamlTool()
result = tool.run(operation="parse", data="key: value")
```

## Creating Custom Tools

All tools extend the `BaseTool` abstract class:

```python
from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

class MyCustomTool(BaseTool):
    @property
    def parameters(self):
        return [
            ToolParameter(name="input", type=ParameterType.STRING, required=True),
            ToolParameter(name="count", type=ParameterType.INTEGER, required=False, default=1),
        ]

    def execute(self, **kwargs):
        text = kwargs["input"]
        count = kwargs.get("count", 1)
        return ToolResult(output=text * count, success=True)

# Use it
tool = MyCustomTool(name="custom")
result = tool.run(input="hello", count=3)
print(result.output)  # "hellohellohello"
```

## Tool Registry

Register and discover tools using the registry:

```python
from nexus_llm.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(CalculatorTool())
registry.register(MyCustomTool())

# List available tools
tools = registry.list_tools()

# Execute by name
result = registry.execute("calculator", expression="2+2")
```

## Tool Results

Every tool returns a `ToolResult` with:

| Field | Description |
|-------|-------------|
| `success` | Whether execution succeeded |
| `output` | The result data |
| `error` | Error message if failed |
| `duration_ms` | Execution time |
| `metadata` | Additional info |

## Best Practices

1. **Always validate inputs** in your `execute` method
2. **Use typed parameters** with appropriate `ParameterType`
3. **Handle errors gracefully** - return failed `ToolResult` instead of raising
4. **Document parameters** with clear descriptions
5. **Keep tools focused** - one responsibility per tool
6. **Use the `run()` method** (not `execute()`) for automatic validation and timing
