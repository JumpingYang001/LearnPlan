# WinDbg Fundamentals

## Overview
WinDbg is a powerful debugger for Windows applications and operating system components. This section covers the basics of using WinDbg, including its interface, symbol management, and debugging modes.

## Example: Attaching to a Process
```c
#include <windows.h>
#include <stdio.h>

int main() {
    printf("Process started. PID: %lu\n", GetCurrentProcessId());
    getchar(); // Wait for debugger to attach
    return 0;
}
```

*Compile and run this program, then attach WinDbg to the process using its PID.*
