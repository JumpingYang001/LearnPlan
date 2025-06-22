# Project: Memory Leak Detector

## Description
Create a GDB-based tool to identify memory leaks and generate reports of allocation sites.

## Example: GDB Script for Leaks
```gdb
set pagination off
break malloc
commands
  silent
  bt
  continue
end
run
```

## Example: Integration with Valgrind
```
valgrind --leak-check=full ./a.out
```
