# Basic WinDbg Commands

## Overview
Learn essential WinDbg commands for navigation, breakpoints, and memory examination.

## Example: Setting a Breakpoint
```c
#include <stdio.h>

int main() {
    int x = 42;
    printf("x = %d\n", x); // Set a breakpoint here
    return 0;
}
```

*Use WinDbg command `bp main+0xXX` to set a breakpoint at the printf line.*
