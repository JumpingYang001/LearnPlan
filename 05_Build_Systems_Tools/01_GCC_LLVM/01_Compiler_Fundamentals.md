# Compiler Fundamentals

## Compilation Process Overview

**C Example:**
```c
#include <stdio.h>
#define SQUARE(x) ((x)*(x))
int main() {
    int a = 5;
    printf("%d\n", SQUARE(a));
    return 0;
}
```
// Preprocessing, parsing, semantic analysis, optimization, code generation steps explained in comments.

## Compiler vs. Interpreter

- Compilers translate source code to machine code before execution.
- Interpreters execute code line by line.

## Compiler Design Patterns

- Front-end/middle-end/back-end separation
- Intermediate representations (IR)
- Optimization passes
- Target abstraction
