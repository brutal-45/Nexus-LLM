# Plugin Development

How to create custom plugins.

## Creating a Plugin

```python
from nexus_llm.plugins import Plugin

class MyPlugin(Plugin):
    name = "my_plugin"
    version = "1.0.0"
    
    def on_load(self):
        print('Plugin loaded!')
    
    def on_unload(self):
        print('Plugin unloaded!')
```
