# Project: Automated Debugging Tool

## Description
Build a tool that automates common debugging tasks with GDB and supports batch analysis of programs.

## Example: Batch Debugging Script
```sh
#!/bin/bash
for prog in prog1 prog2 prog3; do
    gdb -batch -ex "run" -ex "bt" -ex "quit" $prog > $prog.gdb.log
done
```

## Example: Python Automation
```python
import subprocess
progs = ["prog1", "prog2"]
for p in progs:
    subprocess.run(["gdb", "-batch", "-ex", "run", "-ex", "bt", p])
```
