# Memory Management

## Overview
Explains physical and virtual memory management, kernel memory allocation, and OOM killer.

### Physical Memory Management
- Page frame allocation, buddy system

### Virtual Memory System
- Page tables, address spaces

#### Example: kmalloc Usage (C)
```c
#include <linux/slab.h>
void *ptr = kmalloc(128, GFP_KERNEL);
```

---
