# Project 4: System Call Addition

## Description
Add a new system call to the kernel and create user-space programs to test it.

### Example: Kernel and User Code (C)
```c
// Kernel: Add entry to syscall table and implement function
// User: syscall(SYS_mycall, ...);
#include <unistd.h>
#include <sys/syscall.h>
long res = syscall(333, "test"); // 333: example syscall number
```
