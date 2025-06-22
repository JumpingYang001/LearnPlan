# Controlling Program Execution

## Overview
Learn to step through code, manage call stacks, select threads/frames, and handle signals in GDB.

## Example: Stepping and Stack Navigation
```c
#include <stdio.h>
void foo() { printf("In foo\n"); }
int main() {
    foo();
    printf("Back in main\n");
    return 0;
}
```

GDB commands:
```
break main
run
step
next
backtrace
frame 0
```
