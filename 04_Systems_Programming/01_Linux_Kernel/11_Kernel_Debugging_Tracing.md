# Kernel Debugging and Tracing

## Overview
Describes kernel debugging tools, tracing, oops/panic, and dynamic debugging.

### Kernel Debugging Tools
- kgdb, kdump, crash utility

### Kernel Tracing
- ftrace, perf, eBPF

#### Example: printk Debugging (C)
```c
printk(KERN_DEBUG "Debug message\n");
```

---
