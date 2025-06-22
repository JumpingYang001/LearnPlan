# Memory Debugging with GDB

## Overview
Find memory leaks, detect buffer overflows, and inspect heap allocations using GDB. Integrate with Valgrind and AddressSanitizer.

## Example: Detecting Buffer Overflow
```c
#include <stdio.h>
#include <string.h>
int main() {
    char buf[8];
    strcpy(buf, "This is too long!");
    return 0;
}
```

GDB commands:
```
run
backtrace
```

## Integration Example
Run with Valgrind:
```
valgrind ./a.out
```
