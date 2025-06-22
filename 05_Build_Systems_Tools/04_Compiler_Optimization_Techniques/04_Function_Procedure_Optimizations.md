# Function and Procedure Optimizations

## Overview
Discusses inlining, tail call optimization, and interprocedural optimization.

## Example: Function Inlining (C/C++)
```c
inline int square(int x) {
    return x * x;
}
int main() {
    int y = square(5); // May be inlined
    return y;
}
```
// Compile with -O2 or -O3 to see inlining in effect.
