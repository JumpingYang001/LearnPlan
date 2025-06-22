# Kernel Synchronization

## Overview
Explains atomic operations, spinlocks, mutexes, RCU, and other synchronization methods.

### Atomic Operations
- Atomic integers, bitops

#### Example: Atomic Variable (C)
```c
#include <linux/atomic.h>
atomic_t v = ATOMIC_INIT(0);
atomic_inc(&v);
```

### Spinlocks and Mutexes
- Spinlock, mutex, semaphores

---
