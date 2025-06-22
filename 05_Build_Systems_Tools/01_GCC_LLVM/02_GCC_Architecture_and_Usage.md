# GCC Architecture and Usage

## GCC History and Evolution
- Major versions, language support, and features.

## GCC Architecture
- Front-end, GIMPLE/RTL IR, pass managers, back-end code generators.

**C Example:**
```c
// Compile with: gcc -S -fdump-tree-all example.c
#include <stdio.h>
int main() {
    printf("Hello, GCC!\n");
    return 0;
}
```

## Basic GCC Usage
- Command-line interface, compilation phases, file types, standard library integration.

## GCC Configuration
- Target specification, installation, multilib, sysroot.
