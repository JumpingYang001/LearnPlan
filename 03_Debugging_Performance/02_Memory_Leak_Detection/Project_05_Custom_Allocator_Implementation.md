# Project 5: Custom Allocator Implementation

## Description
Design a specialized allocator for a specific use case and benchmark performance against standard allocators.

### Example: Simple Pool Allocator
```cpp
#include <vector>
class Pool {
    std::vector<void*> pool;
public:
    void* alloc() {
        if (!pool.empty()) {
            void* p = pool.back(); pool.pop_back(); return p;
        }
        return ::operator new(64);
    }
    void free(void* p) { pool.push_back(p); }
};
```
