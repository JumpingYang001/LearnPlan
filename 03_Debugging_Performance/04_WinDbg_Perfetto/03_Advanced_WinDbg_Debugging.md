# Advanced WinDbg Debugging

## Overview
Explore advanced debugging techniques such as stack tracing, thread analysis, and exception handling.

## Example: Stack Trace
```c
#include <stdio.h>

void funcB() {
    int y = 2;
    printf("In funcB: y = %d\n", y);
}

void funcA() {
    funcB();
}

int main() {
    funcA();
    return 0;
}
```

*Use WinDbg command `k` to view the call stack when paused in funcB.*
