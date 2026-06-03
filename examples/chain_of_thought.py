#!/usr/bin/env python3
"""Chain of thought example."""
from nexus_llm.chains import SequentialChain

chain = SequentialChain()
chain.add_step(lambda x: f'Step 1: Analyze - {x}')
chain.add_step(lambda x: f'Step 2: Reason - {x}')
chain.add_step(lambda x: f'Step 3: Conclude - {x}')
result = chain.run('What is 2+2?')
print(result)
