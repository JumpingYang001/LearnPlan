# Static Analysis and Sanitizers

## Clang Static Analyzer
- Checker types, running analyzer, interpreting results, custom checkers.

## GCC Static Analysis
- -fanalyzer, warning systems, custom plugins.

## Sanitizers in GCC/Clang
- AddressSanitizer, UBSan, ThreadSanitizer, MemorySanitizer, LeakSanitizer.

**C Example:**
```c
// Compile with: gcc -fsanitize=address -g example.c
#include <stdlib.h>
int main() {
    int *p = malloc(10);
    p[10] = 1; // Out-of-bounds
    return 0;
}
```

## Integration with Build Systems
- CI, automated analysis, result filtering/reporting.
