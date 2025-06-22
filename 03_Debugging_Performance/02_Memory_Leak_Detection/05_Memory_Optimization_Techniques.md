# Memory Optimization Techniques

## Overview
Techniques to reduce and optimize memory usage in C/C++.

### Compact Data Structures Example
```cpp
struct Packed {
    unsigned int a : 4;
    unsigned int b : 4;
};
```

### Object Pooling Example
```cpp
#include <vector>
std::vector<int*> pool;
int* getObj() {
    if (!pool.empty()) {
        int* obj = pool.back(); pool.pop_back(); return obj;
    }
    return new int;
}
```
