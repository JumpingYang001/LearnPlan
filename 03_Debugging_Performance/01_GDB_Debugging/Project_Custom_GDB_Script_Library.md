# Project: Custom GDB Script Library

## Description
Create a collection of useful GDB scripts and pretty printers for common C/C++ data structures.

## Example: Pretty Printer for Linked List
```python
# gdb_printers.py
class ListNodePrinter:
    def __init__(self, val):
        self.val = val
    def to_string(self):
        return f"ListNode(val={self.val['val']})"

def register_printers(obj):
    obj.pretty_printers.append(ListNodePrinter)
```

## Usage in GDB
```
(gdb) source gdb_printers.py
(gdb) print node
```
