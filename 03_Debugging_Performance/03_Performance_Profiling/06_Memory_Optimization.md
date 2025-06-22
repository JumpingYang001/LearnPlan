# Memory Optimization

Covers cache optimization, memory access patterns, and dynamic memory optimization.

## Cache Optimization Example (C)
```c
// Access array sequentially for better cache usage
for (int i = 0; i < n; ++i) {
    sum += arr[i];
}
```

## Custom Allocator Example (C++)
```cpp
#include <vector>
#include <memory>
std::vector<int, MyCustomAllocator<int>> v;
```
