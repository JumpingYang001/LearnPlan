# Process Management

## Overview
Discusses process creation, scheduling, signals, and IPC.

### Process Creation
- fork, exec, exit internals

### Scheduling
- Scheduler classes, CFS

#### Example: Access task_struct (C)
```c
#include <linux/sched.h>
struct task_struct *task = current;
```

---
