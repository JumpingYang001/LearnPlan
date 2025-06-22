# WinDbg Preview and Time Travel Debugging (TTD)

## Overview
Explore the modern WinDbg Preview interface and Time Travel Debugging for advanced analysis.

## Example: Recording a TTD Trace
```c
#include <stdio.h>

int main() {
    printf("Start TTD trace\n");
    for (int i = 0; i < 5; ++i) {
        printf("Step %d\n", i);
    }
    return 0;
}
```

*Use WinDbg Preview to record and replay execution with TTD features.*
