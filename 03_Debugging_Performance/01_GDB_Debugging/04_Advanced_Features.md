# Advanced GDB Features

## Overview
Explore scripting, reverse debugging, remote debugging, and core dump analysis.

## Example: Python Scripting in GDB
```python
def hello_gdb():
    print("Hello from GDB Python!")
end
define pyhello
    python hello_gdb()
end
```

## Example: Reverse Debugging
```
record
reverse-continue
reverse-step
```
