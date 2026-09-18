# Chains Tutorial

How to build processing chains.

```python
from nexus_llm.chains import SequentialChain, ParallelChain

chain = SequentialChain()
chain.add_step(lambda x: x.upper())
```
