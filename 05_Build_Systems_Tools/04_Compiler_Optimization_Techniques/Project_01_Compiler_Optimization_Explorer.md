# Project: Compiler Optimization Explorer

## Description
Build a tool to visualize different optimization levels, compare assembly output, and benchmark optimization strategies.

## Example (C/C++)
```c
// test.c
int add(int a, int b) { return a + b; }
```
// Compile and compare:
// gcc -O0 -S test.c -o test_O0.s
// gcc -O2 -S test.c -o test_O2.s
// Use diff or a custom script to compare assembly outputs.
// Benchmark with time or perf tools.
