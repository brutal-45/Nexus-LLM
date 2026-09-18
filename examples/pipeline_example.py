#!/usr/bin/env python3
"""Pipeline example."""
from nexus_llm.pipeline import Pipeline, PipelineBuilder

pipeline = PipelineBuilder() \
    .step('generate', lambda x: f'Generated: {x}') \
    .step('format', lambda x: x.upper()) \
    .build()
result = pipeline.run('hello world')
print(result)
