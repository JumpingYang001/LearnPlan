# Linux IPC Mechanisms

## Overview
Covers System V IPC, POSIX IPC, pipes, Unix domain sockets, futexes, and shared memory.

### System V IPC
- Message queues, semaphores, shared memory

### POSIX IPC
- POSIX message queues, semaphores

#### Example: POSIX Shared Memory (C)
```c
#include <sys/mman.h>
void *addr = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
```

---
