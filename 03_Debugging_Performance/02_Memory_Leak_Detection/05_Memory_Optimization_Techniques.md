# Memory Optimization Techniques

*Duration: 2 weeks*

## Overview

Memory optimization is crucial for creating efficient, scalable applications. This comprehensive guide covers techniques to reduce memory usage, improve cache performance, and eliminate memory-related bottlenecks in C/C++ programs.

### Why Memory Optimization Matters

1. **Performance Impact**: Poor memory usage leads to cache misses and slower execution
2. **Resource Constraints**: Limited memory in embedded systems and mobile devices
3. **Scalability**: Memory-efficient code can handle larger datasets
4. **Cost Reduction**: Less memory usage reduces cloud computing costs
5. **Battery Life**: Optimized memory access improves energy efficiency

### Memory Hierarchy Understanding

```
CPU Registers    (~1 cycle,    ~1KB)
    ↓
L1 Cache        (~4 cycles,   ~32KB)
    ↓
L2 Cache        (~10 cycles,  ~256KB)
    ↓
L3 Cache        (~40 cycles,  ~8MB)
    ↓
Main Memory     (~100 cycles, ~16GB)
    ↓
SSD Storage     (~25,000 cycles, ~1TB)
    ↓
HDD Storage     (~10,000,000 cycles, ~4TB)
```

## 1. Data Structure Optimization

### 1.1 Compact Data Structures with Bit Fields

**Problem:** Wasteful memory alignment can increase structure size significantly.

```cpp
// Inefficient structure (12 bytes on 64-bit systems)
struct Inefficient {
    bool flag1;        // 1 byte + 3 bytes padding
    int value;         // 4 bytes
    bool flag2;        // 1 byte + 3 bytes padding
};

// Optimized structure with bit fields (4 bytes)
struct Optimized {
    unsigned int flag1 : 1;    // 1 bit
    unsigned int flag2 : 1;    // 1 bit
    unsigned int value : 30;   // 30 bits (supports values up to ~1 billion)
};

// Advanced bit field example for network packet
struct NetworkPacket {
    unsigned int version : 4;        // IP version (4 or 6)
    unsigned int header_length : 4;  // Header length
    unsigned int type_of_service : 8; // QoS information
    unsigned int total_length : 16;  // Packet length
    unsigned int identification : 16; // Fragment identification
    unsigned int flags : 3;          // Control flags
    unsigned int fragment_offset : 13; // Fragment position
    unsigned int ttl : 8;            // Time to live
    unsigned int protocol : 8;       // Next protocol
    unsigned int checksum : 16;      // Header checksum
};

void demonstrate_size_difference() {
    std::cout << "Inefficient struct size: " << sizeof(Inefficient) << " bytes\n";
    std::cout << "Optimized struct size: " << sizeof(Optimized) << " bytes\n";
    std::cout << "Network packet size: " << sizeof(NetworkPacket) << " bytes\n";
    
    // Calculate memory savings for 1 million objects
    size_t inefficient_total = sizeof(Inefficient) * 1000000;
    size_t optimized_total = sizeof(Optimized) * 1000000;
    size_t savings = inefficient_total - optimized_total;
    
    std::cout << "Memory savings for 1M objects: " << savings / 1024 / 1024 << " MB\n";
}
```

### 1.2 Structure Packing and Alignment

```cpp
#include <iostream>

// Default alignment (typically 8 bytes on 64-bit)
struct DefaultAlignment {
    char c;      // 1 byte + 7 bytes padding
    double d;    // 8 bytes
    int i;       // 4 bytes + 4 bytes padding
}; // Total: 24 bytes

// Packed structure (no padding)
#pragma pack(push, 1)
struct PackedStructure {
    char c;      // 1 byte
    double d;    // 8 bytes
    int i;       // 4 bytes
}; // Total: 13 bytes
#pragma pack(pop)

// Manually optimized structure (reorder fields)
struct OptimizedAlignment {
    double d;    // 8 bytes
    int i;       // 4 bytes
    char c;      // 1 byte + 3 bytes padding
}; // Total: 16 bytes

void analyze_structure_sizes() {
    std::cout << "Default alignment: " << sizeof(DefaultAlignment) << " bytes\n";
    std::cout << "Packed structure: " << sizeof(PackedStructure) << " bytes\n";
    std::cout << "Optimized alignment: " << sizeof(OptimizedAlignment) << " bytes\n";
    
    // Performance consideration
    std::cout << "\nNote: Packed structures may have slower access due to unaligned memory reads\n";
}
```

### 1.3 Cache-Friendly Data Layouts

```cpp
// Cache-unfriendly: Array of Structures (AoS)
struct Particle {
    float x, y, z;     // Position
    float vx, vy, vz;  // Velocity
    float mass;
    int id;
};

std::vector<Particle> particles_aos(10000);

// Cache-friendly: Structure of Arrays (SoA)
struct ParticleSystem {
    std::vector<float> x, y, z;        // Positions
    std::vector<float> vx, vy, vz;     // Velocities
    std::vector<float> mass;
    std::vector<int> id;
    
    void resize(size_t count) {
        x.resize(count); y.resize(count); z.resize(count);
        vx.resize(count); vy.resize(count); vz.resize(count);
        mass.resize(count); id.resize(count);
    }
};

// Performance comparison
void update_positions_aos(std::vector<Particle>& particles, float dt) {
    for (auto& p : particles) {
        p.x += p.vx * dt;  // Cache miss likely here
        p.y += p.vy * dt;  // And here
        p.z += p.vz * dt;  // And here
    }
}

void update_positions_soa(ParticleSystem& system, float dt) {
    size_t count = system.x.size();
    for (size_t i = 0; i < count; ++i) {
        system.x[i] += system.vx[i] * dt;  // Sequential memory access
        system.y[i] += system.vy[i] * dt;  // Better cache utilization
        system.z[i] += system.vz[i] * dt;  // Vectorization friendly
    }
}
```

## 2. Object Pooling and Resource Management

### 2.1 Basic Object Pool Implementation

Object pooling reduces allocation overhead by reusing objects instead of constantly creating and destroying them.

```cpp
#include <vector>
#include <memory>
#include <iostream>
#include <chrono>

template<typename T>
class ObjectPool {
private:
    std::vector<std::unique_ptr<T>> pool;
    std::vector<T*> available;
    
public:
    ObjectPool(size_t initial_size = 10) {
        pool.reserve(initial_size);
        available.reserve(initial_size);
        
        for (size_t i = 0; i < initial_size; ++i) {
            pool.push_back(std::make_unique<T>());
            available.push_back(pool.back().get());
        }
    }
    
    T* acquire() {
        if (available.empty()) {
            // Pool exhausted, create new object
            pool.push_back(std::make_unique<T>());
            return pool.back().get();
        }
        
        T* obj = available.back();
        available.pop_back();
        return obj;
    }
    
    void release(T* obj) {
        if (obj) {
            // Reset object state if needed
            // obj->reset(); // Implement reset() in your class
            available.push_back(obj);
        }
    }
    
    size_t pool_size() const { return pool.size(); }
    size_t available_count() const { return available.size(); }
};

// Example usage with a complex object
class ExpensiveObject {
private:
    std::vector<double> data;
    std::string buffer;
    
public:
    ExpensiveObject() : data(1000), buffer(1024, 'x') {
        std::cout << "ExpensiveObject created\n";
    }
    
    ~ExpensiveObject() {
        std::cout << "ExpensiveObject destroyed\n";
    }
    
    void reset() {
        std::fill(data.begin(), data.end(), 0.0);
        buffer.clear();
        buffer.resize(1024, 'x');
    }
    
    void do_work(int work_id) {
        // Simulate work
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = work_id * i;
        }
    }
};

void demonstrate_object_pooling() {
    const int iterations = 1000;
    
    // Test without pool (frequent allocation/deallocation)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto obj = std::make_unique<ExpensiveObject>();
        obj->do_work(i);
        // obj destroyed here
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto without_pool = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test with pool (object reuse)
    ObjectPool<ExpensiveObject> pool(10);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ExpensiveObject* obj = pool.acquire();
        obj->do_work(i);
        pool.release(obj);
    }
    end = std::chrono::high_resolution_clock::now();
    auto with_pool = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Without pool: " << without_pool.count() << " microseconds\n";
    std::cout << "With pool: " << with_pool.count() << " microseconds\n";
    std::cout << "Speedup: " << (double)without_pool.count() / with_pool.count() << "x\n";
}
```

### 2.2 Advanced Memory Pool with Custom Allocator

```cpp
#include <cstddef>
#include <new>

class MemoryPool {
private:
    struct Block {
        Block* next;
    };
    
    Block* free_blocks;
    char* pool_start;
    size_t block_size;
    size_t block_count;
    
public:
    MemoryPool(size_t block_size, size_t block_count) 
        : free_blocks(nullptr), block_size(block_size), block_count(block_count) {
        
        // Ensure block size can hold a pointer
        if (block_size < sizeof(Block*)) {
            this->block_size = sizeof(Block*);
        }
        
        // Allocate the entire pool
        pool_start = new char[this->block_size * block_count];
        
        // Initialize free list
        char* current = pool_start;
        for (size_t i = 0; i < block_count - 1; ++i) {
            reinterpret_cast<Block*>(current)->next = 
                reinterpret_cast<Block*>(current + this->block_size);
            current += this->block_size;
        }
        reinterpret_cast<Block*>(current)->next = nullptr;
        free_blocks = reinterpret_cast<Block*>(pool_start);
    }
    
    ~MemoryPool() {
        delete[] pool_start;
    }
    
    void* allocate() {
        if (!free_blocks) {
            throw std::bad_alloc(); // Pool exhausted
        }
        
        void* result = free_blocks;
        free_blocks = free_blocks->next;
        return result;
    }
    
    void deallocate(void* ptr) {
        if (ptr) {
            Block* block = static_cast<Block*>(ptr);
            block->next = free_blocks;
            free_blocks = block;
        }
    }
    
    bool owns(void* ptr) const {
        char* p = static_cast<char*>(ptr);
        return p >= pool_start && p < pool_start + (block_size * block_count);
    }
};

// Custom allocator using memory pool
template<typename T>
class PoolAllocator {
private:
    static MemoryPool pool;
    
public:
    using value_type = T;
    
    PoolAllocator() = default;
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>&) {}
    
    T* allocate(size_t n) {
        if (n != 1) {
            throw std::bad_alloc(); // This simple pool only handles single objects
        }
        return static_cast<T*>(pool.allocate());
    }
    
    void deallocate(T* ptr, size_t) {
        pool.deallocate(ptr);
    }
    
    template<typename U>
    bool operator==(const PoolAllocator<U>&) const { return true; }
    
    template<typename U>
    bool operator!=(const PoolAllocator<U>&) const { return false; }
};

// Initialize static pool
template<typename T>
MemoryPool PoolAllocator<T>::pool(sizeof(T), 1000);

// Usage example
void demonstrate_custom_allocator() {
    std::vector<int, PoolAllocator<int>> pool_vector;
    
    // This vector uses our custom memory pool
    for (int i = 0; i < 100; ++i) {
        pool_vector.push_back(i);
    }
}
```

### 2.3 RAII and Smart Pointer Optimization

```cpp
#include <memory>
#include <vector>
#include <iostream>

// Custom deleter for pool objects
template<typename T>
class PoolDeleter {
private:
    ObjectPool<T>* pool;
    
public:
    PoolDeleter(ObjectPool<T>* p) : pool(p) {}
    
    void operator()(T* ptr) {
        if (pool && ptr) {
            pool->release(ptr);
        }
    }
};

// Pool-aware smart pointer
template<typename T>
using PoolPtr = std::unique_ptr<T, PoolDeleter<T>>;

template<typename T>
PoolPtr<T> make_pool_ptr(ObjectPool<T>& pool) {
    T* obj = pool.acquire();
    return PoolPtr<T>(obj, PoolDeleter<T>(&pool));
}

void demonstrate_pool_smart_pointers() {
    ObjectPool<ExpensiveObject> pool(5);
    
    {
        auto obj1 = make_pool_ptr(pool);
        auto obj2 = make_pool_ptr(pool);
        
        obj1->do_work(1);
        obj2->do_work(2);
        
        std::cout << "Available objects: " << pool.available_count() << "\n";
    } // Objects automatically returned to pool here
    
    std::cout << "Available objects after scope: " << pool.available_count() << "\n";
}
```

## 3. Memory Allocation Strategies

### 3.1 Stack vs Heap Allocation

```cpp
#include <chrono>
#include <vector>
#include <memory>

class LargeObject {
    int data[1000]; // 4KB object
public:
    LargeObject() { std::fill(data, data + 1000, 42); }
    void process() { 
        for (int& val : data) val *= 2; 
    }
};

void stack_allocation_demo() {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Stack allocation (very fast)
    for (int i = 0; i < 10000; ++i) {
        LargeObject obj;  // Allocated on stack
        obj.process();
        // Automatically destroyed when out of scope
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto stack_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Stack allocation time: " << stack_time.count() << " μs\n";
}

void heap_allocation_demo() {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Heap allocation (slower due to malloc/free overhead)
    for (int i = 0; i < 10000; ++i) {
        auto obj = std::make_unique<LargeObject>();  // Allocated on heap
        obj->process();
        // Destroyed when unique_ptr goes out of scope
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto heap_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Heap allocation time: " << heap_time.count() << " μs\n";
}
```

### 3.2 Small Object Optimization (SOO)

```cpp
#include <string>
#include <iostream>

// Example: std::string uses SOO
void demonstrate_small_string_optimization() {
    // Small strings (typically <= 23 chars) are stored inline
    std::string small_str = "Hello";
    
    // Large strings are heap-allocated
    std::string large_str = "This is a very long string that exceeds the small string optimization threshold";
    
    std::cout << "Small string capacity: " << small_str.capacity() << "\n";
    std::cout << "Large string capacity: " << large_str.capacity() << "\n";
    
    // Check if string is using heap allocation
    auto check_heap_allocation = [](const std::string& str) {
        return str.capacity() > sizeof(std::string);
    };
    
    std::cout << "Small string uses heap: " << check_heap_allocation(small_str) << "\n";
    std::cout << "Large string uses heap: " << check_heap_allocation(large_str) << "\n";
}

// Custom class with SOO
template<typename T, size_t N = 16>
class SmallVector {
private:
    union {
        T local_storage[N];
        T* heap_storage;
    };
    size_t size_;
    size_t capacity_;
    bool using_heap;
    
public:
    SmallVector() : size_(0), capacity_(N), using_heap(false) {}
    
    ~SmallVector() {
        if (using_heap) {
            delete[] heap_storage;
        }
    }
    
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            grow();
        }
        
        if (using_heap) {
            heap_storage[size_] = value;
        } else {
            local_storage[size_] = value;
        }
        ++size_;
    }
    
    T& operator[](size_t index) {
        return using_heap ? heap_storage[index] : local_storage[index];
    }
    
    size_t size() const { return size_; }
    bool is_using_heap() const { return using_heap; }
    
private:
    void grow() {
        size_t new_capacity = capacity_ * 2;
        T* new_storage = new T[new_capacity];
        
        // Copy existing elements
        if (using_heap) {
            std::copy(heap_storage, heap_storage + size_, new_storage);
            delete[] heap_storage;
        } else {
            std::copy(local_storage, local_storage + size_, new_storage);
            using_heap = true;
        }
        
        heap_storage = new_storage;
        capacity_ = new_capacity;
    }
};
```

### 3.3 Memory Arenas and Linear Allocators

```cpp
class MemoryArena {
private:
    char* memory;
    size_t size;
    size_t offset;
    
public:
    MemoryArena(size_t arena_size) : size(arena_size), offset(0) {
        memory = new char[arena_size];
    }
    
    ~MemoryArena() {
        delete[] memory;
    }
    
    template<typename T>
    T* allocate(size_t count = 1) {
        size_t bytes_needed = sizeof(T) * count;
        size_t aligned_offset = align(offset, alignof(T));
        
        if (aligned_offset + bytes_needed > size) {
            throw std::bad_alloc(); // Arena exhausted
        }
        
        T* result = reinterpret_cast<T*>(memory + aligned_offset);
        offset = aligned_offset + bytes_needed;
        return result;
    }
    
    void reset() {
        offset = 0; // Reset arena (doesn't call destructors!)
    }
    
    size_t bytes_used() const { return offset; }
    size_t bytes_remaining() const { return size - offset; }
    
private:
    size_t align(size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }
};

// Usage example: Game frame allocator
class GameFrameAllocator {
private:
    MemoryArena arena;
    
public:
    GameFrameAllocator() : arena(1024 * 1024) {} // 1MB arena
    
    template<typename T>
    T* allocate_temporary(size_t count = 1) {
        return arena.allocate<T>(count);
    }
    
    void end_frame() {
        arena.reset(); // Clear all temporary allocations
        std::cout << "Frame memory reset\n";
    }
};

void demonstrate_arena_allocator() {
    GameFrameAllocator frame_alloc;
    
    // Simulate game loop
    for (int frame = 0; frame < 3; ++frame) {
        std::cout << "Frame " << frame << ":\n";
        
        // Allocate temporary objects for this frame
        int* temp_array = frame_alloc.allocate_temporary<int>(100);
        float* temp_positions = frame_alloc.allocate_temporary<float>(300);
        
        // Use the temporary objects...
        std::fill(temp_array, temp_array + 100, frame);
        
        frame_alloc.end_frame(); // Clear all temporary allocations
    }
}
```

## 4. Copy Optimization Techniques

### 4.1 Move Semantics and Perfect Forwarding

```cpp
#include <vector>
#include <string>
#include <utility>

class ExpensiveResource {
private:
    std::vector<double> data;
    std::string name;
    
public:
    // Constructor
    ExpensiveResource(const std::string& n, size_t size) 
        : data(size), name(n) {
        std::cout << "ExpensiveResource constructed: " << name << "\n";
    }
    
    // Copy constructor (expensive)
    ExpensiveResource(const ExpensiveResource& other) 
        : data(other.data), name(other.name) {
        std::cout << "ExpensiveResource copied: " << name << "\n";
    }
    
    // Move constructor (cheap)
    ExpensiveResource(ExpensiveResource&& other) noexcept
        : data(std::move(other.data)), name(std::move(other.name)) {
        std::cout << "ExpensiveResource moved: " << name << "\n";
    }
    
    // Copy assignment (expensive)
    ExpensiveResource& operator=(const ExpensiveResource& other) {
        if (this != &other) {
            data = other.data;
            name = other.name;
            std::cout << "ExpensiveResource copy-assigned: " << name << "\n";
        }
        return *this;
    }
    
    // Move assignment (cheap)
    ExpensiveResource& operator=(ExpensiveResource&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            name = std::move(other.name);
            std::cout << "ExpensiveResource move-assigned: " << name << "\n";
        }
        return *this;
    }
    
    const std::string& get_name() const { return name; }
};

// Factory function that returns by value (move-enabled)
ExpensiveResource create_resource(const std::string& name, size_t size) {
    return ExpensiveResource(name, size); // Return value optimization (RVO)
}

void demonstrate_move_semantics() {
    std::cout << "=== Move Semantics Demo ===\n";
    
    // 1. RVO (Return Value Optimization)
    auto resource1 = create_resource("Resource1", 1000000);
    
    // 2. Move construction
    auto resource2 = std::move(resource1);
    
    // 3. Container with move semantics
    std::vector<ExpensiveResource> resources;
    resources.reserve(3); // Avoid reallocations
    
    resources.emplace_back("Resource3", 500000); // Direct construction
    resources.push_back(create_resource("Resource4", 300000)); // Move from temporary
    
    std::cout << "Container size: " << resources.size() << "\n";
}
```

### 4.2 Copy Elision and Return Value Optimization

```cpp
#include <iostream>
#include <vector>

class TrackingObject {
private:
    static int construction_count;
    static int copy_count;
    static int move_count;
    int id;
    
public:
    TrackingObject(int i = 0) : id(i) {
        ++construction_count;
        std::cout << "Constructed object " << id << "\n";
    }
    
    TrackingObject(const TrackingObject& other) : id(other.id) {
        ++copy_count;
        std::cout << "Copied object " << id << "\n";
    }
    
    TrackingObject(TrackingObject&& other) noexcept : id(other.id) {
        ++move_count;
        std::cout << "Moved object " << id << "\n";
    }
    
    static void print_stats() {
        std::cout << "Constructions: " << construction_count << "\n";
        std::cout << "Copies: " << copy_count << "\n";
        std::cout << "Moves: " << move_count << "\n";
    }
    
    static void reset_stats() {
        construction_count = copy_count = move_count = 0;
    }
};

int TrackingObject::construction_count = 0;
int TrackingObject::copy_count = 0;
int TrackingObject::move_count = 0;

// RVO example
TrackingObject create_object_rvo(int id) {
    return TrackingObject(id); // RVO eliminates copy/move
}

// NRVO (Named Return Value Optimization) example
TrackingObject create_object_nrvo(int id) {
    TrackingObject obj(id);
    // ... some processing ...
    return obj; // NRVO eliminates copy/move
}

void demonstrate_copy_elision() {
    std::cout << "=== Copy Elision Demo ===\n";
    
    TrackingObject::reset_stats();
    
    std::cout << "1. RVO:\n";
    auto obj1 = create_object_rvo(1);
    
    std::cout << "\n2. NRVO:\n";
    auto obj2 = create_object_nrvo(2);
    
    std::cout << "\n3. Direct initialization:\n";
    TrackingObject obj3 = TrackingObject(3);
    
    std::cout << "\nFinal stats:\n";
    TrackingObject::print_stats();
}
```

### 4.3 String and Container Optimizations

```cpp
#include <string>
#include <vector>
#include <sstream>

void string_optimization_techniques() {
    std::cout << "=== String Optimization ===\n";
    
    // 1. Reserve capacity to avoid reallocations
    std::string result;
    result.reserve(10000); // Pre-allocate space
    
    for (int i = 0; i < 1000; ++i) {
        result += std::to_string(i) + " ";
    }
    
    // 2. Use string_view for non-owning references
    auto process_substring = [](std::string_view sv) {
        return sv.substr(0, 10); // No allocation, returns another string_view
    };
    
    std::string data = "Hello, World! This is a long string.";
    auto sub = process_substring(data); // No copy made
    
    // 3. Use stringstream for complex string building
    std::ostringstream oss;
    oss.str().reserve(10000); // Reserve capacity
    
    for (int i = 0; i < 1000; ++i) {
        oss << "Item " << i << ": " << (i * i) << "\n";
    }
    std::string formatted = oss.str();
    
    // 4. Move semantics with strings
    std::vector<std::string> strings;
    strings.reserve(1000);
    
    for (int i = 0; i < 1000; ++i) {
        std::string temp = "String number " + std::to_string(i);
        strings.push_back(std::move(temp)); // Move instead of copy
    }
}

// Container optimization techniques
template<typename T>
void container_optimization_demo() {
    std::cout << "=== Container Optimization ===\n";
    
    // 1. Reserve capacity
    std::vector<T> vec;
    vec.reserve(1000); // Avoid multiple reallocations
    
    // 2. Use emplace_back instead of push_back for in-place construction
    for (int i = 0; i < 1000; ++i) {
        vec.emplace_back(i); // Construct in-place
        // vs vec.push_back(T(i)); // Construct temporary then copy/move
    }
    
    // 3. Use shrink_to_fit to reclaim unused capacity
    vec.erase(vec.begin() + 500, vec.end()); // Remove half the elements
    std::cout << "Capacity after erase: " << vec.capacity() << "\n";
    
    vec.shrink_to_fit(); // Reclaim unused memory
    std::cout << "Capacity after shrink_to_fit: " << vec.capacity() << "\n";
    
    // 4. Use appropriate container for use case
    // - std::array for fixed-size, stack-allocated
    // - std::vector for dynamic arrays
    // - std::deque for double-ended queues
    // - std::list for frequent insertions/deletions
    // - std::unordered_map for hash tables
}
```

## 5. Cache Optimization Techniques

### 5.1 Data Locality and Cache-Friendly Programming

```cpp
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>

class CacheOptimizationDemo {
private:
    static constexpr size_t ARRAY_SIZE = 64 * 1024 * 1024; // 64MB
    
public:
    // Poor cache locality: random access pattern
    static void poor_cache_locality() {
        std::vector<int> data(ARRAY_SIZE);
        std::iota(data.begin(), data.end(), 0);
        
        // Shuffle to create random access pattern
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(data.begin(), data.end(), gen);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        long long sum = 0;
        for (size_t i = 0; i < ARRAY_SIZE; ++i) {
            sum += data[data[i] % ARRAY_SIZE]; // Random memory access
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Poor cache locality - Sum: " << sum 
                  << ", Time: " << duration.count() << "ms\n";
    }
    
    // Good cache locality: sequential access pattern
    static void good_cache_locality() {
        std::vector<int> data(ARRAY_SIZE);
        std::iota(data.begin(), data.end(), 0);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        long long sum = 0;
        for (size_t i = 0; i < ARRAY_SIZE; ++i) {
            sum += data[i]; // Sequential memory access
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Good cache locality - Sum: " << sum 
                  << ", Time: " << duration.count() << "ms\n";
    }
    
    // Matrix multiplication: compare row-major vs column-major access
    static void matrix_access_patterns() {
        constexpr size_t SIZE = 1024;
        std::vector<std::vector<double>> matrix(SIZE, std::vector<double>(SIZE, 1.0));
        
        // Row-major access (cache-friendly)
        auto start = std::chrono::high_resolution_clock::now();
        double sum1 = 0;
        for (size_t i = 0; i < SIZE; ++i) {
            for (size_t j = 0; j < SIZE; ++j) {
                sum1 += matrix[i][j]; // Access by rows
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto row_major_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Column-major access (cache-unfriendly)
        start = std::chrono::high_resolution_clock::now();
        double sum2 = 0;
        for (size_t j = 0; j < SIZE; ++j) {
            for (size_t i = 0; i < SIZE; ++i) {
                sum2 += matrix[i][j]; // Access by columns
            }
        }
        end = std::chrono::high_resolution_clock::now();
        auto col_major_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Row-major access: " << row_major_time.count() << "ms\n";
        std::cout << "Column-major access: " << col_major_time.count() << "ms\n";
        std::cout << "Speedup: " << (double)col_major_time.count() / row_major_time.count() << "x\n";
    }
};
```

### 5.2 Memory Prefetching and Cache Line Optimization

```cpp
#include <immintrin.h> // For prefetch intrinsics

class CacheLineOptimization {
private:
    static constexpr size_t CACHE_LINE_SIZE = 64; // Typical cache line size
    
public:
    // Struct with false sharing problem
    struct BadCacheLineUsage {
        alignas(CACHE_LINE_SIZE) int counter1;
        int counter2; // Same cache line as counter1
        int counter3; // Same cache line as counter1
    };
    
    // Struct avoiding false sharing
    struct GoodCacheLineUsage {
        alignas(CACHE_LINE_SIZE) int counter1;
        alignas(CACHE_LINE_SIZE) int counter2; // Different cache line
        alignas(CACHE_LINE_SIZE) int counter3; // Different cache line
    };
    
    // Demonstrate false sharing
    static void demonstrate_false_sharing() {
        constexpr int ITERATIONS = 10000000;
        
        // Test with false sharing
        BadCacheLineUsage bad_data{};
        auto start = std::chrono::high_resolution_clock::now();
        
        std::thread t1([&]() {
            for (int i = 0; i < ITERATIONS; ++i) {
                ++bad_data.counter1; // Modifies cache line
            }
        });
        
        std::thread t2([&]() {
            for (int i = 0; i < ITERATIONS; ++i) {
                ++bad_data.counter2; // Same cache line, causes invalidation
            }
        });
        
        t1.join();
        t2.join();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto bad_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Test without false sharing
        GoodCacheLineUsage good_data{};
        start = std::chrono::high_resolution_clock::now();
        
        std::thread t3([&]() {
            for (int i = 0; i < ITERATIONS; ++i) {
                ++good_data.counter1; // Different cache line
            }
        });
        
        std::thread t4([&]() {
            for (int i = 0; i < ITERATIONS; ++i) {
                ++good_data.counter2; // Different cache line
            }
        });
        
        t3.join();
        t4.join();
        
        end = std::chrono::high_resolution_clock::now();
        auto good_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "With false sharing: " << bad_time.count() << "ms\n";
        std::cout << "Without false sharing: " << good_time.count() << "ms\n";
        std::cout << "Speedup: " << (double)bad_time.count() / good_time.count() << "x\n";
    }
    
    // Manual prefetching example
    static void prefetch_demo() {
        constexpr size_t SIZE = 1024 * 1024;
        std::vector<int> data(SIZE);
        std::iota(data.begin(), data.end(), 0);
        
        // Without prefetching
        auto start = std::chrono::high_resolution_clock::now();
        long long sum1 = 0;
        for (size_t i = 0; i < SIZE - 8; i += 8) {
            sum1 += data[i] + data[i+1] + data[i+2] + data[i+3] + 
                   data[i+4] + data[i+5] + data[i+6] + data[i+7];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto without_prefetch = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // With prefetching
        start = std::chrono::high_resolution_clock::now();
        long long sum2 = 0;
        for (size_t i = 0; i < SIZE - 8; i += 8) {
            // Prefetch next cache line
            _mm_prefetch(reinterpret_cast<const char*>(&data[i + 16]), _MM_HINT_T0);
            
            sum2 += data[i] + data[i+1] + data[i+2] + data[i+3] + 
                   data[i+4] + data[i+5] + data[i+6] + data[i+7];
        }
        end = std::chrono::high_resolution_clock::now();
        auto with_prefetch = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Without prefetch: " << without_prefetch.count() << " μs\n";
        std::cout << "With prefetch: " << with_prefetch.count() << " μs\n";
        std::cout << "Prefetch improvement: " 
                  << (double)without_prefetch.count() / with_prefetch.count() << "x\n";
    }
};
```

## 6. Memory Profiling and Analysis Tools

### 6.1 Built-in Memory Tracking

```cpp
#include <new>
#include <iostream>
#include <cstdlib>

class MemoryTracker {
private:
    static size_t total_allocated;
    static size_t allocation_count;
    
public:
    static void* allocate(size_t size) {
        total_allocated += size;
        ++allocation_count;
        
        // Store size before the actual data
        void* ptr = std::malloc(size + sizeof(size_t));
        if (!ptr) throw std::bad_alloc();
        
        *static_cast<size_t*>(ptr) = size;
        return static_cast<char*>(ptr) + sizeof(size_t);
    }
    
    static void deallocate(void* ptr) {
        if (!ptr) return;
        
        // Get the stored size
        char* actual_ptr = static_cast<char*>(ptr) - sizeof(size_t);
        size_t size = *reinterpret_cast<size_t*>(actual_ptr);
        
        total_allocated -= size;
        --allocation_count;
        
        std::free(actual_ptr);
    }
    
    static void print_stats() {
        std::cout << "Total allocated: " << total_allocated << " bytes\n";
        std::cout << "Active allocations: " << allocation_count << "\n";
    }
    
    static size_t get_total_allocated() { return total_allocated; }
    static size_t get_allocation_count() { return allocation_count; }
};

size_t MemoryTracker::total_allocated = 0;
size_t MemoryTracker::allocation_count = 0;

// Override global new/delete operators
void* operator new(size_t size) {
    return MemoryTracker::allocate(size);
}

void operator delete(void* ptr) noexcept {
    MemoryTracker::deallocate(ptr);
}

void* operator new[](size_t size) {
    return MemoryTracker::allocate(size);
}

void operator delete[](void* ptr) noexcept {
    MemoryTracker::deallocate(ptr);
}

void demonstrate_memory_tracking() {
    std::cout << "=== Memory Tracking Demo ===\n";
    
    MemoryTracker::print_stats();
    
    {
        std::vector<int> vec(1000);
        std::string str = "Hello, Memory Tracking!";
        auto ptr = std::make_unique<double[]>(500);
        
        MemoryTracker::print_stats();
    } // Objects destroyed here
    
    MemoryTracker::print_stats();
}
```

### 6.2 Performance Benchmarking Framework

```cpp
#include <chrono>
#include <functional>
#include <string>
#include <vector>

class MemoryBenchmark {
public:
    struct BenchmarkResult {
        std::string name;
        std::chrono::microseconds execution_time;
        size_t memory_used;
        size_t allocations;
    };
    
    template<typename Func>
    static BenchmarkResult run_benchmark(const std::string& name, Func&& func) {
        // Record initial memory state
        size_t initial_memory = MemoryTracker::get_total_allocated();
        size_t initial_allocations = MemoryTracker::get_allocation_count();
        
        // Run the benchmark
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate results
        auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        size_t final_memory = MemoryTracker::get_total_allocated();
        size_t final_allocations = MemoryTracker::get_allocation_count();
        
        return {
            name,
            execution_time,
            final_memory - initial_memory,
            final_allocations - initial_allocations
        };
    }
    
    static void print_result(const BenchmarkResult& result) {
        std::cout << "Benchmark: " << result.name << "\n";
        std::cout << "  Execution time: " << result.execution_time.count() << " μs\n";
        std::cout << "  Memory used: " << result.memory_used << " bytes\n";
        std::cout << "  Allocations: " << result.allocations << "\n";
        std::cout << "  Avg allocation size: " 
                  << (result.allocations > 0 ? result.memory_used / result.allocations : 0) 
                  << " bytes\n\n";
    }
    
    static void compare_results(const std::vector<BenchmarkResult>& results) {
        if (results.size() < 2) return;
        
        std::cout << "=== Comparison ===\n";
        const auto& baseline = results[0];
        
        for (size_t i = 1; i < results.size(); ++i) {
            const auto& current = results[i];
            
            double time_ratio = (double)current.execution_time.count() / baseline.execution_time.count();
            double memory_ratio = (double)current.memory_used / baseline.memory_used;
            
            std::cout << current.name << " vs " << baseline.name << ":\n";
            std::cout << "  Time: " << time_ratio << "x (" 
                      << (time_ratio < 1.0 ? "faster" : "slower") << ")\n";
            std::cout << "  Memory: " << memory_ratio << "x (" 
                      << (memory_ratio < 1.0 ? "less" : "more") << ")\n\n";
        }
    }
};

// Example usage
void run_memory_benchmarks() {
    std::vector<MemoryBenchmark::BenchmarkResult> results;
    
    // Benchmark 1: Vector with reserve
    results.push_back(MemoryBenchmark::run_benchmark("Vector with reserve", []() {
        std::vector<int> vec;
        vec.reserve(10000);
        for (int i = 0; i < 10000; ++i) {
            vec.push_back(i);
        }
    }));
    
    // Benchmark 2: Vector without reserve
    results.push_back(MemoryBenchmark::run_benchmark("Vector without reserve", []() {
        std::vector<int> vec;
        for (int i = 0; i < 10000; ++i) {
            vec.push_back(i);
        }
    }));
    
    // Benchmark 3: Object pool
    ObjectPool<ExpensiveObject> pool(100);
    results.push_back(MemoryBenchmark::run_benchmark("Object pool", [&pool]() {
        std::vector<ExpensiveObject*> objects;
        for (int i = 0; i < 1000; ++i) {
            objects.push_back(pool.acquire());
        }
        for (auto* obj : objects) {
            pool.release(obj);
        }
    }));
    
    // Print all results
    for (const auto& result : results) {
        MemoryBenchmark::print_result(result);
    }
    
    // Compare results
    MemoryBenchmark::compare_results(results);
}
```

## Learning Objectives and Assessment

### Learning Objectives

By the end of this section, you should be able to:

- **Analyze memory usage patterns** and identify optimization opportunities
- **Implement various data structure optimizations** including bit fields, packing, and cache-friendly layouts
- **Design and implement object pools** and custom memory allocators
- **Apply move semantics and copy elision** to reduce unnecessary copying
- **Optimize cache performance** through data locality and prefetching techniques
- **Use profiling tools** to measure and validate memory optimizations
- **Choose appropriate allocation strategies** for different scenarios

### Self-Assessment Checklist

□ Can identify memory waste in data structures and fix alignment issues  
□ Can implement a working object pool with proper resource management  
□ Understands when and how to use move semantics vs copy semantics  
□ Can design cache-friendly data layouts (AoS vs SoA)  
□ Can measure memory usage and allocation patterns  
□ Knows how to avoid false sharing in multi-threaded code  
□ Can implement custom allocators for specific use cases  
□ Understands the trade-offs between different optimization techniques  

### Practical Exercises

**Exercise 1: Data Structure Optimization**
```cpp
// TODO: Optimize this structure to minimize memory usage
struct GameEntity {
    bool is_active;
    float position_x, position_y, position_z;
    bool is_visible;
    int health;
    bool can_move;
    float velocity_x, velocity_y, velocity_z;
    short level;
};
```

**Exercise 2: Custom Allocator Implementation**
```cpp
// TODO: Implement a stack allocator that allocates from a fixed buffer
template<size_t BufferSize>
class StackAllocator {
    // Your implementation here
};
```

**Exercise 3: Cache Optimization**
```cpp
// TODO: Optimize this matrix multiplication for better cache performance
void matrix_multiply(const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    std::vector<std::vector<double>>& C);
```

## Tools and Resources

### Profiling Tools
- **Valgrind Massif**: Heap profiler for detailed memory usage analysis
- **AddressSanitizer**: Detects memory errors and leaks
- **Intel VTune**: Performance profiler with memory analysis
- **Google Benchmark**: Microbenchmarking library
- **Heaptrack**: Heap memory profiler for Linux

### Useful Commands
```bash
# Compile with memory debugging
g++ -g -fsanitize=address -fsanitize=leak program.cpp

# Run with Valgrind
valgrind --tool=massif --detailed-freq=1 ./program
valgrind --tool=memcheck --leak-check=full ./program

# Profile with perf (Linux)
perf record -g ./program
perf report
```

### Recommended Reading
- "Optimized C++" by Kurt Guntheroth
- "High Performance C++ Cookbook" by Dmitri Nesteruk
- "Memory Management: Algorithms and Implementation" by Donald Knuth
- Intel Optimization Reference Manual
- "What Every Programmer Should Know About Memory" by Ulrich Drepper
