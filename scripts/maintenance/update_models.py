#!/usr/bin/env python3
"""Update model catalog."""
from nexus_llm.core.model_catalog import MODEL_CATALOG
for m in MODEL_CATALOG.values():
    print(f'{m.id}: {m.name} ({m.size})')
