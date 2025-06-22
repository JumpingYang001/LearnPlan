# Basic Debugging Techniques

## Overview
Covers running programs, setting breakpoints/watchpoints, and examining program state in GDB.

## Example: Breakpoints and Watchpoints
```c
#include <stdio.h>
int main() {
    int x = 0;
    for (int i = 0; i < 5; ++i) {
        x += i;
        printf("i=%d, x=%d\n", i, x);
    }
    return 0;
}
```

Set a breakpoint and watchpoint in GDB:
```
break main
watch x
run
```
