# Project: GDB Frontend Extension

## Description
Extend an existing GDB frontend with new features and improve visualization of complex data structures.

## Example: VS Code GDB Extension Snippet
```json
{
    "contributes": {
        "commands": [
            {
                "command": "extension.gdbPrettyPrint",
                "title": "GDB: Pretty Print Data Structure"
            }
        ]
    }
}
```

## Example: Custom Visualization (Python)
```python
# In a GDB Python script
import gdb
class PrettyPrintVector(gdb.Command):
    def __init__(self):
        super().__init__("ppvector", gdb.COMMAND_USER)
    def invoke(self, arg, from_tty):
        gdb.execute(f"print {arg}")
PrettyPrintVector()
```
