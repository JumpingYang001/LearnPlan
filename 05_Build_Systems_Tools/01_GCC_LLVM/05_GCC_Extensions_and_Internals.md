# GCC Extensions and Internals

## GCC Language Extensions
- Statement expressions, attributes, built-ins, nested functions, type discovery.

**C Example:**
```c
#include <stdio.h>
#define max(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })
int main() {
    printf("%d\n", max(3, 7));
    return 0;
}
```

## GCC Intrinsics
- SIMD, atomics, memory barriers, bit manipulation.

## GCC Plugins
- Plugin architecture, writing/using plugins, build system integration.

## GCC Internals
- Data structures, pass implementation, AST manipulation, extending GCC.
