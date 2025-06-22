# Project: Thread Analysis Tool

## Description
Implement a GDB script for analyzing thread interactions and detecting potential deadlocks or race conditions.

## Example: Thread Info Script
```gdb
info threads
thread apply all bt
```

## Example: Python GDB Script
```python
import gdb
class ThreadLister(gdb.Command):
    def __init__(self):
        super(ThreadLister, self).__init__("listthreads", gdb.COMMAND_USER)
    def invoke(self, arg, from_tty):
        gdb.execute("info threads")
ThreadLister()
```
