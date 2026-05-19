# Workflow Guide

This guide covers how to create, configure, and execute workflows in Nexus-LLM.

## Overview

The Nexus-LLM workflow system allows you to define directed acyclic graphs (DAGs) of processing steps, connect them with edges, and execute them with built-in validation, error handling, and monitoring.

## Quick Start

### Creating a Simple Workflow

```python
from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType

# Create a workflow engine
engine = WorkflowEngine(WorkflowConfig(name="my_workflow"))

# Add nodes
engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: "input data"))
engine.add_node(WorkflowNode(id="process", type=NodeType.PROCESS, fn=lambda x: x.upper()))
engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))

# Connect nodes with edges
engine.add_edge("start", "process")
engine.add_edge("process", "end")

# Validate the workflow
errors = engine.validate()
if errors:
    for error in errors:
        print(f"Error: {error}")

# Execute the workflow
from nexus_llm.workflow.executor import WorkflowExecutor
executor = WorkflowExecutor()
result = executor.execute(engine)
```

## Node Types

| Type | Description |
|------|-------------|
| `START` | Entry point of the workflow |
| `END` | Exit point of the workflow |
| `PROCESS` | Standard processing node |
| `DECISION` | Branching/conditional node |

## Conditional Edges

Edges can have conditions that determine whether they are followed during execution:

```python
from nexus_llm.workflow.edges import WorkflowEdge, EdgeCondition

condition = EdgeCondition(predicate=lambda ctx: ctx.get("score", 0) > 0.5)
engine.add_edge("check", "pass_path", condition=condition)
```

## Using Templates

Pre-built workflow templates simplify common patterns:

```python
from nexus_llm.workflow.templates import get_template

# Sequential pipeline
template = get_template("sequential")
engine = template.build(steps=[step1, step2, step3])

# Fan-out/fan-in
template = get_template("fan_out_fan_in")
engine = template.build(branches=[branch1, branch2, branch3], merge_fn=merge)

# Conditional branching
template = get_template("conditional_branch")
engine = template.build(condition_fn=check, true_fn=handle_true, false_fn=handle_false)
```

## Scheduling Workflows

The scheduler enables automated, recurring workflow execution:

```python
from nexus_llm.workflow.scheduler import WorkflowScheduler, ScheduleConfig, ScheduleType

scheduler = WorkflowScheduler()

# One-time execution after delay
config = ScheduleConfig(schedule_type=ScheduleType.ONCE, delay_seconds=60)
scheduler.schedule(engine, config)

# Recurring execution
config = ScheduleConfig(schedule_type=ScheduleType.INTERVAL, interval_seconds=300)
scheduler.schedule(engine, config)

scheduler.start()
```

## Serialization

Save and load workflows:

```python
from nexus_llm.workflow.serializer import WorkflowSerializer

serializer = WorkflowSerializer()

# Save
serializer.save(engine, "my_workflow.json")

# Load
engine = serializer.load("my_workflow.json")
```

## Validation

Use the workflow validator to check for common issues:

```python
from nexus_llm.workflow.validators import WorkflowValidator

validator = WorkflowValidator()
errors = validator.validate(engine)

for error in errors:
    print(f"[{error.severity}] {error.path}: {error.message}")
```

## Error Handling

Configure error handling per workflow:

```python
config = WorkflowConfig(
    name="resilient_workflow",
    max_retries=3,
    retry_delay_seconds=2.0,
    continue_on_error=True,
)
engine = WorkflowEngine(config=config)
```

## Best Practices

1. **Always validate** your workflow before execution
2. **Use meaningful node IDs** for easier debugging
3. **Keep workflows simple** - break complex logic into sub-workflows
4. **Use conditions** to handle branching logic declaratively
5. **Monitor execution** with the built-in metrics and logging
6. **Save workflow definitions** for reproducibility
