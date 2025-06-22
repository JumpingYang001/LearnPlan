# Instruction-Level Parallelism

## Overview
Covers instruction scheduling, software pipelining, superscalar, and VLIW architectures.

## Example: Instruction Scheduling (C/C++)
```c
void example(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = b[i] + c[i];
    }
}
```
// Compilers may reorder instructions for better parallelism. Use -O2/-O3 and inspect assembly.
