# Safety Tutorial

How to use safety features.

```python
from nexus_llm.safety import ContentFilter, PIIFilter

filter = ContentFilter()
safe, reason = filter.check_prompt('Hello')
```
