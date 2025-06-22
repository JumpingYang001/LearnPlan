# Compiler Flags and Optimizations

## Basic Compilation Flags
- Debug info (-g), warnings (-Wall, -Wextra), standard (-std=), ABI control.

**C Example:**
```c
// Compile with: gcc -Wall -g -std=c11 example.c
#include <stdio.h>
int main() {
    printf("Flags example\n");
    return 0;
}
```

## Optimization Levels
- -O0, -O1, -O2, -O3, -Os, -Oz: trade-offs and use cases.

## Advanced Optimization Flags
- Inlining, loop optimizations, vectorization, LTO, PGO, FDO.

## Architecture-Specific Optimizations
- Target features, instruction sets, tuning.
