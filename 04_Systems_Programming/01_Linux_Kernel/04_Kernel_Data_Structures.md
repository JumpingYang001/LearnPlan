# Kernel Data Structures

## Overview
Discusses common and specialized kernel data structures and memory allocation.

### Common Data Structures
- Lists, queues, maps
- Radix trees, RB-trees

#### Example: Kernel Linked List (C)
```c
#include <linux/list.h>
struct my_struct {
    int data;
    struct list_head list;
};
```

### Memory Allocation
- kmalloc, vmalloc, slab allocator

---
