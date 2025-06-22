# Memory Profiling Tools

## Overview
Explore tools for detecting and analyzing memory leaks in C/C++.

### Valgrind Example
```sh
valgrind --leak-check=full ./a.out
```

### AddressSanitizer Example
```cpp
// Compile with -fsanitize=address
int *leak() {
    int *p = new int[10];
    return p; // not deleted
}
```

### Dr. Memory Example
```sh
drmemory -- ./a.out
```
