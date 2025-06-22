# Compiler Optimization Fundamentals

## Overview
This section introduces the basics of compiler optimization, including the compilation pipeline, optimization phases, intermediate representations (IR), and the relationship between source code and generated code.

## Example: Compilation Pipeline (C/C++)
```c
// Example: Simple C code
int add(int a, int b) {
    return a + b;
}
```

- Compile with different optimization levels:
  ```sh
  gcc -O0 -S add.c -o add_O0.s
  gcc -O2 -S add.c -o add_O2.s
  ```
- Compare the generated assembly to see optimization effects.
