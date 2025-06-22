# Project 3: Memory Optimization Case Study

## Description
Analyze and optimize a memory-intensive application. Document improvements and techniques used.

### Example: Before and After Optimization
```cpp
// Before: Frequent allocations
for (int i = 0; i < N; ++i) {
    int* arr = new int[1000];
    // ...
    delete[] arr;
}
// After: Object pool
std::vector<int*> pool;
for (int i = 0; i < N; ++i) {
    int* arr = pool.empty() ? new int[1000] : pool.back();
    if (!pool.empty()) pool.pop_back();
    // ...
    pool.push_back(arr);
}
```
