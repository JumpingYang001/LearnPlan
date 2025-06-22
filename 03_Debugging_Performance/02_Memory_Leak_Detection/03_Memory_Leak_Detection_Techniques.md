# Memory Leak Detection Techniques

## Overview
Static and dynamic analysis, and manual review for memory issues.

### Static Analysis Example
```cpp
// Use clang-analyzer or Coverity for static checks
int* foo() {
    int* p = new int[10];
    // forgot delete[] p;
    return p;
}
```

### Dynamic Analysis Example
```cpp
#include <stdlib.h>
void test() {
    char* buf = (char*)malloc(100);
    // forgot free(buf);
}
```
