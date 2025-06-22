# Multithreading Optimization

Explains thread scaling, concurrency patterns, and synchronization optimization.

## Thread Scaling Example (C++)
```cpp
#include <thread>
#include <vector>
void worker(int id) { /* ... */ }
int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i)
        threads.emplace_back(worker, i);
    for (auto& t : threads) t.join();
}
```

## Lock-Free Example (C++)
```cpp
#include <atomic>
std::atomic<int> counter{0};
void increment() { counter.fetch_add(1, std::memory_order_relaxed); }
```
