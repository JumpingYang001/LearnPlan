# Futexes and Synchronization Primitives

## Overview

Covers futex system call, building mutexes, and performance considerations.

### Example: Futex Wait/Wake (Pseudo-C)
```c
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

int futex_wait(volatile int *addr, int val) {
    return syscall(SYS_futex, addr, FUTEX_WAIT, val, NULL, NULL, 0);
}
int futex_wake(volatile int *addr, int n) {
    return syscall(SYS_futex, addr, FUTEX_WAKE, n, NULL, NULL, 0);
}

int main() {
    volatile int futex_var = 0;
    // Example usage: futex_wait(&futex_var, 0); futex_wake(&futex_var, 1);
    printf("Futex example (see code for usage)\n");
    return 0;
}
```
