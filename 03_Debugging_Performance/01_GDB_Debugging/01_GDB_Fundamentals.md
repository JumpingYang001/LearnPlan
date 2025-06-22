# GDB Fundamentals

## Overview
Learn how to install, configure, and start using GDB for C/C++ debugging. This section covers basic commands, compiling with debug info, and navigating the GDB interface.

## Example: Compiling and Running with GDB
```c
#include <stdio.h>
int main() {
    printf("Hello, GDB!\n");
    return 0;
}
```

Compile with debug info:
```sh
gcc -g hello.c -o hello
```

Start GDB:
```sh
gdb ./hello
```

Basic commands:
- `run`, `break`, `next`, `step`, `print`, `quit`
