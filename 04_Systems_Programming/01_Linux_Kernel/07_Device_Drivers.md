# Device Drivers

## Overview
Covers character, block, network, platform, PCI, USB, input, and network device drivers.

### Character Device Drivers
- File operations, cdev interface

#### Example: Simple Char Device (C)
```c
#include <linux/fs.h>
static int my_open(struct inode *inode, struct file *file) { return 0; }
static struct file_operations fops = {
    .open = my_open,
};
```

---
