# Kernel-User Space Interface

## Overview
Describes system calls, procfs, sysfs, ioctl, and netlink sockets.

### System Calls
- Mechanism, table, wrappers

#### Example: Custom System Call (C)
```c
// Kernel: Add entry to syscall table
// User: syscall(SYS_mycall, ...);
```

### procfs and sysfs
- Creating proc/sys entries

---
