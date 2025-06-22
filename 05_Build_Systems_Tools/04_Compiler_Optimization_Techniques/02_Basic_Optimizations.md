# Basic Optimizations

## Overview
Covers constant folding, constant propagation, common subexpression elimination, dead code elimination, and unreachable code.

## Example: Constant Folding and Dead Code Elimination (C/C++)
```c
int foo() {
    int x = 2 * 3; // Constant folding
    int y = x + 0; // Constant propagation
    return y;
    int z = 5; // Dead code (eliminated)
}
```
// Compile with -O2 and check that dead code is removed and constants are folded.
