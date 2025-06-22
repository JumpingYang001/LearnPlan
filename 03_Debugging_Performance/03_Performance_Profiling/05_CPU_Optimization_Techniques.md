# CPU Optimization Techniques

Discusses algorithm, code, data structure, and compiler optimizations for CPU performance.

## Loop Optimization Example (C)
```c
// Loop unrolling
for (int i = 0; i < n; i += 4) {
    a[i] = b[i] + c[i];
    a[i+1] = b[i+1] + c[i+1];
    a[i+2] = b[i+2] + c[i+2];
    a[i+3] = b[i+3] + c[i+3];
}
```

## SIMD Vectorization Example (C++)
```cpp
#include <immintrin.h>
void add(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        __m256 va = _mm256_add_ps(vb, vc);
        _mm256_storeu_ps(&a[i], va);
    }
}
```
