# System Call Interface

## Description
Describes glibc's system call wrappers, conventions, vDSO mechanism, and direct system calls.

## Example: Direct syscall
```c
#include <unistd.h>
#include <sys/syscall.h>
#include <stdio.h>

int main() {
    long res = syscall(SYS_write, 1, "Hello via syscall!\n", 19);
    printf("syscall returned %ld\n", res);
    return 0;
}
```
