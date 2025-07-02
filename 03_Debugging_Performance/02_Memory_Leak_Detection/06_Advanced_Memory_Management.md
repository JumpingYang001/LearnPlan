# Advanced Memory Management

*Duration: 2 weeks*

## Overview

Advanced memory management encompasses sophisticated techniques for managing memory in high-performance, concurrent, and security-critical applications. This module covers lock-free memory management, garbage collection strategies, secure memory handling, custom allocators, and memory optimization techniques in C/C++.

### Learning Goals
- Master lock-free data structures and memory reclamation
- Understand garbage collection principles and implementation
- Implement secure memory handling for sensitive data
- Design custom memory allocators for performance optimization
- Apply advanced debugging techniques for complex memory issues

## Lock-Free Memory Management

Lock-free programming enables multiple threads to access shared data structures without traditional locking mechanisms, providing better performance and avoiding deadlocks.

### Understanding the ABA Problem

The ABA problem occurs when a memory location changes from A to B and back to A between atomic operations, potentially causing corruption.

```cpp
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

template<typename T>
class LockFreeStack {
private:
    struct Node {
        T data;
        Node* next;
        Node(T data) : data(data), next(nullptr) {}
    };
    
    std::atomic<Node*> head;

public:
    LockFreeStack() : head(nullptr) {}
    
    void push(T item) {
        Node* new_node = new Node(item);
        new_node->next = head.load();
        
        // Compare-and-swap loop to handle concurrent modifications
        while (!head.compare_exchange_weak(new_node->next, new_node)) {
            // If CAS fails, new_node->next is updated with current head value
            // Retry the operation
        }
    }
    
    bool pop(T& result) {
        Node* old_head = head.load();
        
        while (old_head && !head.compare_exchange_weak(old_head, old_head->next)) {
            // Retry if another thread modified head
        }
        
        if (old_head) {
            result = old_head->data;
            delete old_head;  // Potential ABA problem here!
            return true;
        }
        return false;
    }
};

// Demonstration of ABA problem
void demonstrate_aba_problem() {
    LockFreeStack<int> stack;
    stack.push(1);
    stack.push(2);
    
    // Thread 1: pop() loads head (pointing to node with value 2)
    // Thread 2: pop() twice, then push(2) - head points to new node with value 2
    // Thread 1: CAS succeeds because head still "looks" the same
    // But it's actually pointing to a different node!
}
```

### Hazard Pointers for Safe Memory Reclamation

Hazard pointers solve the ABA problem by ensuring that memory isn't reclaimed while other threads might be accessing it.

```cpp
#include <atomic>
#include <array>
#include <vector>
#include <algorithm>

template<int MAX_THREADS = 8>
class HazardPointer {
private:
    struct HazardRecord {
        std::atomic<std::thread::id> id;
        std::atomic<void*> pointer;
    };
    
    static std::array<HazardRecord, MAX_THREADS> hazard_records;
    static thread_local std::vector<void*> retired_list;
    static constexpr int RETIRED_THRESHOLD = 10;

public:
    class Guard {
    private:
        HazardRecord* record;
        
    public:
        Guard() : record(nullptr) {
            // Find an available hazard record
            for (auto& hr : hazard_records) {
                std::thread::id null_id{};
                if (hr.id.compare_exchange_strong(null_id, std::this_thread::get_id())) {
                    record = &hr;
                    break;
                }
            }
        }
        
        ~Guard() {
            if (record) {
                record->pointer.store(nullptr);
                record->id.store(std::thread::id{});
            }
        }
        
        template<typename T>
        T* protect(const std::atomic<T*>& atomic_ptr) {
            T* ptr;
            do {
                ptr = atomic_ptr.load();
                record->pointer.store(ptr);
            } while (ptr != atomic_ptr.load()); // Ensure pointer didn't change
            
            return ptr;
        }
    };
    
    static void retire(void* ptr) {
        retired_list.push_back(ptr);
        
        if (retired_list.size() >= RETIRED_THRESHOLD) {
            scan_and_reclaim();
        }
    }

private:
    static void scan_and_reclaim() {
        // Collect all currently protected pointers
        std::vector<void*> protected_ptrs;
        for (const auto& hr : hazard_records) {
            void* ptr = hr.pointer.load();
            if (ptr) {
                protected_ptrs.push_back(ptr);
            }
        }
        
        // Sort for efficient searching
        std::sort(protected_ptrs.begin(), protected_ptrs.end());
        
        // Reclaim non-protected pointers
        auto it = retired_list.begin();
        while (it != retired_list.end()) {
            if (!std::binary_search(protected_ptrs.begin(), protected_ptrs.end(), *it)) {
                delete static_cast<char*>(*it); // Safe to reclaim
                it = retired_list.erase(it);
            } else {
                ++it;
            }
        }
    }
};

// Safe lock-free stack using hazard pointers
template<typename T>
class SafeLockFreeStack {
private:
    struct Node {
        T data;
        std::atomic<Node*> next;
        Node(T data) : data(data), next(nullptr) {}
    };
    
    std::atomic<Node*> head;

public:
    SafeLockFreeStack() : head(nullptr) {}
    
    void push(T item) {
        Node* new_node = new Node(item);
        new_node->next = head.load();
        
        while (!head.compare_exchange_weak(new_node->next, new_node));
    }
    
    bool pop(T& result) {
        HazardPointer<>::Guard guard;
        Node* old_head;
        
        do {
            old_head = guard.protect(head);
            if (!old_head) return false;
        } while (!head.compare_exchange_weak(old_head, old_head->next.load()));
        
        result = old_head->data;
        HazardPointer<>::retire(old_head); // Safe deferred deletion
        return true;
    }
};
```

### RCU (Read-Copy-Update) Pattern

RCU allows readers to access data without locks while writers create new versions of data structures.

```cpp
#include <atomic>
#include <memory>
#include <vector>

template<typename T>
class RCUProtectedData {
private:
    std::atomic<T*> data_ptr;
    std::vector<std::unique_ptr<T>> old_versions;
    mutable std::atomic<int> reader_count{0};

public:
    RCUProtectedData(T initial_data) {
        data_ptr.store(new T(std::move(initial_data)));
    }
    
    // Read operation - lock-free
    class ReadGuard {
    private:
        const RCUProtectedData* rcu;
        const T* data;
        
    public:
        ReadGuard(const RCUProtectedData* rcu) : rcu(rcu) {
            rcu->reader_count.fetch_add(1);
            data = rcu->data_ptr.load();
        }
        
        ~ReadGuard() {
            rcu->reader_count.fetch_sub(1);
        }
        
        const T* operator->() const { return data; }
        const T& operator*() const { return *data; }
    };
    
    ReadGuard read() const {
        return ReadGuard(this);
    }
    
    // Write operation - creates new version
    void update(T new_data) {
        T* new_ptr = new T(std::move(new_data));
        T* old_ptr = data_ptr.exchange(new_ptr);
        
        // Wait for all readers to finish with old data
        wait_for_readers();
        
        // Safe to delete old version
        delete old_ptr;
    }

private:
    void wait_for_readers() const {
        // Simple spin-wait - in production, use more sophisticated approach
        while (reader_count.load() > 0) {
            std::this_thread::yield();
        }
    }
};

// Usage example
void rcu_example() {
    RCUProtectedData<std::vector<int>> rcu_data({1, 2, 3, 4, 5});
    
    // Multiple readers can access simultaneously
    auto reader1 = rcu_data.read();
    auto reader2 = rcu_data.read();
    
    // Readers see consistent snapshot
    for (int val : *reader1) {
        std::cout << val << " ";
    }
    
    // Writer updates data
    rcu_data.update({10, 20, 30});
    
    // New readers see updated data
    auto reader3 = rcu_data.read();
    for (int val : *reader3) {
        std::cout << val << " ";
    }
}
```

## Secure Memory Handling

Security-critical applications require careful handling of sensitive data in memory to prevent information leakage through memory dumps, swap files, or compiler optimizations.

### Secure Memory Wiping

Simple memory clearing can be optimized away by compilers. Here are several robust approaches:

```c
#include <string.h>
#include <stddef.h>

// Basic secure wipe - may be optimized away by smart compilers
void insecure_wipe(void* ptr, size_t len) {
    memset(ptr, 0, len);  // Compiler might optimize this away!
}

// Secure wipe using volatile pointer - prevents optimization
void secure_wipe_volatile(void* ptr, size_t len) {
    volatile char* p = (volatile char*)ptr;
    while (len--) *p++ = 0;
}

// More robust secure wipe with memory barrier
void secure_wipe_barrier(void* ptr, size_t len) {
    memset(ptr, 0, len);
    __asm__ __volatile__("" ::: "memory");  // Memory barrier
}

// Platform-specific secure wipe functions
#ifdef _WIN32
#include <windows.h>
void secure_wipe_windows(void* ptr, size_t len) {
    SecureZeroMemory(ptr, len);  // Windows API - guaranteed not to be optimized
}
#endif

#ifdef __linux__
#include <sys/mman.h>
void secure_wipe_linux(void* ptr, size_t len) {
    memset(ptr, 0, len);
    madvise(ptr, len, MADV_DONTDUMP);  // Exclude from core dumps
}
#endif

// Cross-platform secure wipe implementation
void secure_wipe(void* ptr, size_t len) {
    if (!ptr || len == 0) return;
    
#ifdef _WIN32
    SecureZeroMemory(ptr, len);
#elif defined(__OpenBSD__)
    explicit_bzero(ptr, len);
#elif defined(__GLIBC__) && __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 25
    explicit_bzero(ptr, len);
#else
    // Fallback implementation
    volatile char* p = (volatile char*)ptr;
    while (len--) *p++ = 0;
    __asm__ __volatile__("" ::: "memory");
#endif
}
```

### Secure Memory Allocation

```cpp
#include <memory>
#include <new>
#include <cstdlib>

template<typename T>
class SecureAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = SecureAllocator<U>;
    };

    SecureAllocator() = default;
    template<typename U>
    SecureAllocator(const SecureAllocator<U>&) {}

    pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }

        // Allocate memory that won't be swapped to disk
        void* ptr = nullptr;
        size_t size = n * sizeof(T);

#ifdef _WIN32
        ptr = VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (ptr) {
            VirtualLock(ptr, size);  // Lock in physical memory
        }
#elif defined(__linux__) || defined(__APPLE__)
        if (posix_memalign(&ptr, alignof(T), size) == 0) {
            mlock(ptr, size);  // Lock in physical memory
            madvise(ptr, size, MADV_DONTDUMP);  // Exclude from core dumps
        }
#else
        ptr = aligned_alloc(alignof(T), size);
#endif

        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type n) {
        if (p) {
            size_t size = n * sizeof(T);
            
            // Securely wipe memory before deallocation
            secure_wipe(p, size);

#ifdef _WIN32
            VirtualUnlock(p, size);
            VirtualFree(p, 0, MEM_RELEASE);
#elif defined(__linux__) || defined(__APPLE__)
            munlock(p, size);
            free(p);
#else
            free(p);
#endif
        }
    }

    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* p) {
        // Secure wipe before destruction
        secure_wipe(p, sizeof(U));
        p->~U();
    }
};

// Secure string type using secure allocator
using SecureString = std::basic_string<char, std::char_traits<char>, SecureAllocator<char>>;

// Example usage
void secure_password_handling() {
    SecureString password("my_secret_password");
    
    // Use password...
    
    // Password memory will be securely wiped when SecureString is destroyed
}
```

### Memory Protection Techniques

```cpp
#include <sys/mman.h>
#include <unistd.h>

class ProtectedMemory {
private:
    void* memory;
    size_t size;
    size_t page_size;

public:
    ProtectedMemory(size_t requested_size) {
        page_size = getpagesize();
        
        // Round up to page boundary
        size = ((requested_size + page_size - 1) / page_size) * page_size;
        
        // Allocate 3 pages: guard | data | guard
        void* allocation = mmap(nullptr, size + 2 * page_size,
                              PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        
        if (allocation == MAP_FAILED) {
            throw std::bad_alloc();
        }
        
        // Set up guard pages (no access)
        mprotect(allocation, page_size, PROT_NONE);  // Front guard
        mprotect(static_cast<char*>(allocation) + page_size + size, 
                page_size, PROT_NONE);  // Back guard
        
        memory = static_cast<char*>(allocation) + page_size;
    }
    
    ~ProtectedMemory() {
        if (memory) {
            // Secure wipe
            secure_wipe(memory, size);
            
            // Unmap entire allocation including guard pages
            munmap(static_cast<char*>(memory) - page_size, size + 2 * page_size);
        }
    }
    
    void* get() { return memory; }
    size_t get_size() const { return size; }
    
    // Make memory read-only
    void make_readonly() {
        mprotect(memory, size, PROT_READ);
    }
    
    // Make memory executable (for JIT scenarios)
    void make_executable() {
        mprotect(memory, size, PROT_READ | PROT_EXEC);
    }
    
    // Restore read-write access
    void make_writable() {
        mprotect(memory, size, PROT_READ | PROT_WRITE);
    }
};

// Stack protection example
class StackProtector {
private:
    void* guard_page;
    size_t page_size;

public:
    StackProtector() {
        page_size = getpagesize();
        
        // Allocate a guard page near stack
        guard_page = mmap(nullptr, page_size, PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        
        if (guard_page == MAP_FAILED) {
            throw std::runtime_error("Failed to create stack guard");
        }
    }
    
    ~StackProtector() {
        if (guard_page) {
            munmap(guard_page, page_size);
        }
    }
    
    // Check if address is in guard page (indicates stack overflow)
    bool is_stack_overflow(void* addr) {
        return addr >= guard_page && 
               addr < static_cast<char*>(guard_page) + page_size;
    }
};
```

## Garbage Collection Implementation

While C++ doesn't have built-in garbage collection, we can implement various GC strategies for specific use cases.

### Reference Counting GC

```cpp
#include <atomic>
#include <memory>

template<typename T>
class RefCountedPtr {
private:
    struct ControlBlock {
        std::atomic<int> ref_count{1};
        T* ptr;
        
        ControlBlock(T* p) : ptr(p) {}
        
        ~ControlBlock() {
            delete ptr;
        }
    };
    
    ControlBlock* control;

public:
    explicit RefCountedPtr(T* ptr = nullptr) 
        : control(ptr ? new ControlBlock(ptr) : nullptr) {}
    
    RefCountedPtr(const RefCountedPtr& other) : control(other.control) {
        if (control) {
            control->ref_count.fetch_add(1);
        }
    }
    
    RefCountedPtr& operator=(const RefCountedPtr& other) {
        if (this != &other) {
            reset();
            control = other.control;
            if (control) {
                control->ref_count.fetch_add(1);
            }
        }
        return *this;
    }
    
    ~RefCountedPtr() {
        reset();
    }
    
    void reset() {
        if (control && control->ref_count.fetch_sub(1) == 1) {
            delete control;
        }
        control = nullptr;
    }
    
    T* get() const { return control ? control->ptr : nullptr; }
    T& operator*() const { return *get(); }
    T* operator->() const { return get(); }
    
    int use_count() const {
        return control ? control->ref_count.load() : 0;
    }
};

// Cycle-breaking weak reference
template<typename T>
class WeakRefPtr {
private:
    RefCountedPtr<T>::ControlBlock* control;

public:
    WeakRefPtr() : control(nullptr) {}
    
    WeakRefPtr(const RefCountedPtr<T>& strong_ref) 
        : control(strong_ref.control) {}
    
    RefCountedPtr<T> lock() {
        if (control && control->ref_count.load() > 0) {
            return RefCountedPtr<T>(control->ptr);
        }
        return RefCountedPtr<T>();
    }
    
    bool expired() const {
        return !control || control->ref_count.load() == 0;
    }
};
```

### Mark-and-Sweep GC

```cpp
#include <vector>
#include <unordered_set>
#include <queue>

class GarbageCollector {
private:
    std::unordered_set<void*> allocated_objects;
    std::unordered_set<void*> root_objects;
    std::vector<std::pair<void*, std::vector<void*>>> object_references;

public:
    void* allocate(size_t size) {
        void* ptr = malloc(size);
        allocated_objects.insert(ptr);
        return ptr;
    }
    
    void add_root(void* ptr) {
        root_objects.insert(ptr);
    }
    
    void remove_root(void* ptr) {
        root_objects.erase(ptr);
    }
    
    void add_reference(void* from, void* to) {
        // Find existing entry or create new one
        auto it = std::find_if(object_references.begin(), object_references.end(),
                              [from](const auto& pair) { return pair.first == from; });
        
        if (it != object_references.end()) {
            it->second.push_back(to);
        } else {
            object_references.emplace_back(from, std::vector<void*>{to});
        }
    }
    
    void collect() {
        // Mark phase
        std::unordered_set<void*> marked;
        std::queue<void*> work_queue;
        
        // Start with root objects
        for (void* root : root_objects) {
            if (allocated_objects.count(root)) {
                marked.insert(root);
                work_queue.push(root);
            }
        }
        
        // Mark all reachable objects
        while (!work_queue.empty()) {
            void* current = work_queue.front();
            work_queue.pop();
            
            // Find references from current object
            auto it = std::find_if(object_references.begin(), object_references.end(),
                                  [current](const auto& pair) { return pair.first == current; });
            
            if (it != object_references.end()) {
                for (void* referenced : it->second) {
                    if (allocated_objects.count(referenced) && !marked.count(referenced)) {
                        marked.insert(referenced);
                        work_queue.push(referenced);
                    }
                }
            }
        }
        
        // Sweep phase - deallocate unmarked objects
        auto it = allocated_objects.begin();
        while (it != allocated_objects.end()) {
            if (!marked.count(*it)) {
                free(*it);
                it = allocated_objects.erase(it);
            } else {
                ++it;
            }
        }
        
        // Clean up references to deallocated objects
        object_references.erase(
            std::remove_if(object_references.begin(), object_references.end(),
                          [this](const auto& pair) {
                              return !allocated_objects.count(pair.first);
                          }),
            object_references.end());
    }
    
    size_t get_allocated_count() const {
        return allocated_objects.size();
    }
};

// RAII wrapper for GC-managed objects
template<typename T>
class GCPtr {
private:
    static GarbageCollector* gc;
    T* ptr;

public:
    explicit GCPtr(T* p = nullptr) : ptr(p) {
        if (ptr && gc) {
            gc->add_root(ptr);
        }
    }
    
    ~GCPtr() {
        if (ptr && gc) {
            gc->remove_root(ptr);
        }
    }
    
    T* get() const { return ptr; }
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
    
    static void set_collector(GarbageCollector* collector) {
        gc = collector;
    }
    
    static void collect() {
        if (gc) gc->collect();
    }
};

template<typename T>
GarbageCollector* GCPtr<T>::gc = nullptr;
```

## Custom Memory Allocators

Custom allocators can provide better performance, memory tracking, or specialized behavior for specific use cases.

### Pool Allocator

```cpp
#include <memory>
#include <vector>
#include <cassert>

template<typename T, size_t BlockSize = 4096>
class PoolAllocator {
private:
    struct Block {
        alignas(T) char data[sizeof(T)];
        Block* next;
    };
    
    struct Chunk {
        static constexpr size_t blocks_per_chunk = (BlockSize - sizeof(Chunk*)) / sizeof(Block);
        Chunk* next_chunk;
        Block blocks[blocks_per_chunk];
        
        Chunk() : next_chunk(nullptr) {
            // Initialize free list within this chunk
            for (size_t i = 0; i < blocks_per_chunk - 1; ++i) {
                blocks[i].next = &blocks[i + 1];
            }
            blocks[blocks_per_chunk - 1].next = nullptr;
        }
    };
    
    Chunk* first_chunk;
    Block* free_list;
    size_t allocated_count;

public:
    PoolAllocator() : first_chunk(nullptr), free_list(nullptr), allocated_count(0) {}
    
    ~PoolAllocator() {
        clear();
    }
    
    T* allocate() {
        if (!free_list) {
            allocate_new_chunk();
        }
        
        assert(free_list != nullptr);
        
        Block* block = free_list;
        free_list = free_list->next;
        ++allocated_count;
        
        return reinterpret_cast<T*>(block);
    }
    
    void deallocate(T* ptr) {
        if (!ptr) return;
        
        Block* block = reinterpret_cast<Block*>(ptr);
        block->next = free_list;
        free_list = block;
        --allocated_count;
    }
    
    template<typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate();
        try {
            new(ptr) T(std::forward<Args>(args)...);
            return ptr;
        } catch (...) {
            deallocate(ptr);
            throw;
        }
    }
    
    void destroy(T* ptr) {
        if (ptr) {
            ptr->~T();
            deallocate(ptr);
        }
    }
    
    size_t get_allocated_count() const { return allocated_count; }
    
    void clear() {
        Chunk* current = first_chunk;
        while (current) {
            Chunk* next = current->next_chunk;
            delete current;
            current = next;
        }
        first_chunk = nullptr;
        free_list = nullptr;
        allocated_count = 0;
    }

private:
    void allocate_new_chunk() {
        Chunk* new_chunk = new Chunk();
        new_chunk->next_chunk = first_chunk;
        first_chunk = new_chunk;
        
        // Add all blocks from new chunk to free list
        for (size_t i = 0; i < Chunk::blocks_per_chunk; ++i) {
            new_chunk->blocks[i].next = free_list;
            free_list = &new_chunk->blocks[i];
        }
    }
};

// Usage example
void pool_allocator_example() {
    PoolAllocator<int> pool;
    
    std::vector<int*> allocated;
    
    // Fast allocation from pool
    for (int i = 0; i < 1000; ++i) {
        allocated.push_back(pool.construct(i));
    }
    
    std::cout << "Allocated objects: " << pool.get_allocated_count() << std::endl;
    
    // Fast deallocation
    for (int* ptr : allocated) {
        pool.destroy(ptr);
    }
    
    std::cout << "After cleanup: " << pool.get_allocated_count() << std::endl;
}
```

### Stack Allocator

```cpp
#include <cstddef>
#include <cassert>

class StackAllocator {
private:
    char* memory;
    size_t size;
    size_t offset;
    
    struct Marker {
        size_t offset;
    };

public:
    explicit StackAllocator(size_t size) 
        : memory(new char[size]), size(size), offset(0) {}
    
    ~StackAllocator() {
        delete[] memory;
    }
    
    template<typename T>
    T* allocate(size_t count = 1) {
        size_t bytes = count * sizeof(T);
        size_t aligned_bytes = align_size(bytes, alignof(T));
        
        assert(offset + aligned_bytes <= size);
        
        void* ptr = memory + offset;
        offset += aligned_bytes;
        
        return static_cast<T*>(ptr);
    }
    
    Marker get_marker() const {
        return {offset};
    }
    
    void reset_to_marker(const Marker& marker) {
        assert(marker.offset <= offset);
        offset = marker.offset;
    }
    
    void clear() {
        offset = 0;
    }
    
    size_t get_used_memory() const { return offset; }
    size_t get_free_memory() const { return size - offset; }

private:
    size_t align_size(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
};

// RAII wrapper for stack allocator markers
class StackAllocatorScope {
private:
    StackAllocator& allocator;
    StackAllocator::Marker marker;

public:
    explicit StackAllocatorScope(StackAllocator& alloc) 
        : allocator(alloc), marker(alloc.get_marker()) {}
    
    ~StackAllocatorScope() {
        allocator.reset_to_marker(marker);
    }
};

void stack_allocator_example() {
    StackAllocator stack_alloc(1024 * 1024);  // 1MB stack
    
    {
        StackAllocatorScope scope(stack_alloc);
        
        // Allocate temporary arrays
        int* temp_array1 = stack_alloc.allocate<int>(1000);
        float* temp_array2 = stack_alloc.allocate<float>(500);
        
        // Use arrays...
        
        // Automatically freed when scope ends
    }
    
    // Memory is available again
    assert(stack_alloc.get_used_memory() == 0);
}
```

### Memory Tracking Allocator

```cpp
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <iomanip>

class TrackingAllocator {
private:
    struct AllocationInfo {
        size_t size;
        const char* file;
        int line;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
        
        AllocationInfo(size_t s, const char* f, int l)
            : size(s), file(f), line(l), timestamp(std::chrono::steady_clock::now()) {}
    };
    
    std::unordered_map<void*, AllocationInfo> allocations;
    mutable std::mutex mutex;
    size_t total_allocated;
    size_t peak_allocated;
    size_t allocation_count;

public:
    TrackingAllocator() : total_allocated(0), peak_allocated(0), allocation_count(0) {}
    
    void* allocate(size_t size, const char* file = __FILE__, int line = __LINE__) {
        void* ptr = malloc(size);
        if (!ptr) return nullptr;
        
        std::lock_guard<std::mutex> lock(mutex);
        allocations.emplace(ptr, AllocationInfo(size, file, line));
        total_allocated += size;
        peak_allocated = std::max(peak_allocated, total_allocated);
        ++allocation_count;
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex);
        auto it = allocations.find(ptr);
        if (it != allocations.end()) {
            total_allocated -= it->second.size;
            allocations.erase(it);
        }
        
        free(ptr);
    }
    
    void print_stats() const {
        std::lock_guard<std::mutex> lock(mutex);
        
        std::cout << "=== Memory Allocation Statistics ===" << std::endl;
        std::cout << "Current allocations: " << allocations.size() << std::endl;
        std::cout << "Current memory used: " << format_bytes(total_allocated) << std::endl;
        std::cout << "Peak memory used: " << format_bytes(peak_allocated) << std::endl;
        std::cout << "Total allocations made: " << allocation_count << std::endl;
        
        if (!allocations.empty()) {
            std::cout << "\n=== Active Allocations ===" << std::endl;
            for (const auto& [ptr, info] : allocations) {
                auto duration = std::chrono::steady_clock::now() - info.timestamp;
                auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
                
                std::cout << "0x" << std::hex << ptr << std::dec 
                         << " | " << std::setw(8) << format_bytes(info.size)
                         << " | " << info.file << ":" << info.line
                         << " | alive for " << seconds << "s" << std::endl;
            }
        }
    }
    
    bool has_leaks() const {
        std::lock_guard<std::mutex> lock(mutex);
        return !allocations.empty();
    }

private:
    std::string format_bytes(size_t bytes) const {
        const char* units[] = {"B", "KB", "MB", "GB"};
        double size = static_cast<double>(bytes);
        int unit = 0;
        
        while (size >= 1024.0 && unit < 3) {
            size /= 1024.0;
            ++unit;
        }
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
        return oss.str();
    }
};

// Global instance
TrackingAllocator g_tracking_allocator;

// Macros for convenient usage
#define TRACKED_MALLOC(size) g_tracking_allocator.allocate(size, __FILE__, __LINE__)
#define TRACKED_FREE(ptr) g_tracking_allocator.deallocate(ptr)

// RAII wrapper
template<typename T>
class TrackedPtr {
private:
    T* ptr;

public:
    explicit TrackedPtr(size_t count = 1) {
        ptr = static_cast<T*>(TRACKED_MALLOC(count * sizeof(T)));
        if (ptr) {
            for (size_t i = 0; i < count; ++i) {
                new(ptr + i) T();
            }
        }
    }
    
    ~TrackedPtr() {
        if (ptr) {
            ptr->~T();
            TRACKED_FREE(ptr);
        }
    }
    
    T* get() const { return ptr; }
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
};
```

## Memory Optimization Techniques

### Cache-Friendly Data Structures

```cpp
#include <vector>
#include <array>

// Structure of Arrays (SoA) vs Array of Structures (AoS)
namespace performance_comparison {

// Array of Structures (AoS) - poor cache locality
struct Particle_AoS {
    float x, y, z;     // position
    float vx, vy, vz;  // velocity
    float mass;
    int id;
};

// Structure of Arrays (SoA) - better cache locality
struct ParticleSystem_SoA {
    std::vector<float> x, y, z;        // positions
    std::vector<float> vx, vy, vz;     // velocities
    std::vector<float> mass;
    std::vector<int> id;
    
    void add_particle(float px, float py, float pz, 
                     float pvx, float pvy, float pvz, 
                     float pmass, int pid) {
        x.push_back(px);   y.push_back(py);   z.push_back(pz);
        vx.push_back(pvx); vy.push_back(pvy); vz.push_back(pvz);
        mass.push_back(pmass);
        id.push_back(pid);
    }
    
    size_t size() const { return x.size(); }
};

// Benchmark: Update positions
void update_positions_aos(std::vector<Particle_AoS>& particles, float dt) {
    for (auto& p : particles) {
        p.x += p.vx * dt;  // Cache miss likely here
        p.y += p.vy * dt;  // And here
        p.z += p.vz * dt;  // And here
    }
}

void update_positions_soa(ParticleSystem_SoA& system, float dt) {
    const size_t count = system.size();
    
    // All x updates together - much better cache locality
    for (size_t i = 0; i < count; ++i) {
        system.x[i] += system.vx[i] * dt;
    }
    for (size_t i = 0; i < count; ++i) {
        system.y[i] += system.vy[i] * dt;
    }
    for (size_t i = 0; i < count; ++i) {
        system.z[i] += system.vz[i] * dt;
    }
}

}  // namespace performance_comparison

// Cache-oblivious data structures
template<typename T, size_t BlockSize = 64>
class CacheObliviousArray {
private:
    struct Block {
        alignas(64) std::array<T, BlockSize / sizeof(T)> data;
    };
    
    std::vector<Block> blocks;
    size_t element_count;

public:
    CacheObliviousArray() : element_count(0) {}
    
    void push_back(const T& value) {
        constexpr size_t elements_per_block = BlockSize / sizeof(T);
        
        if (element_count % elements_per_block == 0) {
            blocks.emplace_back();
        }
        
        size_t block_index = element_count / elements_per_block;
        size_t element_index = element_count % elements_per_block;
        
        blocks[block_index].data[element_index] = value;
        ++element_count;
    }
    
    T& operator[](size_t index) {
        constexpr size_t elements_per_block = BlockSize / sizeof(T);
        size_t block_index = index / elements_per_block;
        size_t element_index = index % elements_per_block;
        return blocks[block_index].data[element_index];
    }
    
    size_t size() const { return element_count; }
};
```

### Memory Prefetching

```cpp
#include <xmmintrin.h>  // For _mm_prefetch

class PrefetchingVector {
private:
    std::vector<int> data;
    static constexpr int PREFETCH_DISTANCE = 64;  // Cache lines ahead

public:
    void process_with_prefetch() {
        const size_t size = data.size();
        
        for (size_t i = 0; i < size; ++i) {
            // Prefetch future data while processing current
            if (i + PREFETCH_DISTANCE < size) {
                _mm_prefetch(reinterpret_cast<const char*>(&data[i + PREFETCH_DISTANCE]), 
                           _MM_HINT_T0);
            }
            
            // Process current element
            data[i] = data[i] * 2 + 1;  // Some computation
        }
    }
    
    void process_matrix_with_blocking(std::vector<std::vector<int>>& matrix) {
        const size_t N = matrix.size();
        constexpr size_t BLOCK_SIZE = 64;  // Optimize for cache
        
        // Process in blocks to improve cache locality
        for (size_t bi = 0; bi < N; bi += BLOCK_SIZE) {
            for (size_t bj = 0; bj < N; bj += BLOCK_SIZE) {
                // Process block
                for (size_t i = bi; i < std::min(bi + BLOCK_SIZE, N); ++i) {
                    for (size_t j = bj; j < std::min(bj + BLOCK_SIZE, N); ++j) {
                        matrix[i][j] = matrix[i][j] * 2;
                    }
                }
            }
        }
    }
};
```

## Learning Objectives & Assessment

By the end of this section, you should be able to:

### Technical Mastery
- **Implement lock-free data structures** using atomic operations and memory ordering
- **Design and implement hazard pointers** for safe memory reclamation in concurrent environments
- **Build custom allocators** optimized for specific use cases (pool, stack, tracking)
- **Apply secure memory handling** techniques to protect sensitive data
- **Implement garbage collection** strategies (reference counting, mark-and-sweep)
- **Optimize memory layout** for cache performance and reduced fragmentation

### Practical Skills
- **Debug complex memory issues** in multi-threaded applications
- **Profile memory usage** and identify optimization opportunities
- **Choose appropriate memory management** strategies for different scenarios
- **Handle memory security** concerns in production systems

### Self-Assessment Checklist

□ Can explain the ABA problem and implement solutions  
□ Can design a lock-free data structure with proper memory reclamation  
□ Can implement secure memory wiping that won't be optimized away  
□ Can create a custom allocator for a specific use case  
□ Can identify cache-unfriendly memory patterns and fix them  
□ Can implement a simple garbage collector  
□ Can debug memory corruption in concurrent programs  
□ Can apply memory protection techniques using mmap/VirtualAlloc  

### Practical Exercises

**Exercise 1: Lock-Free Queue Implementation**
```cpp
// TODO: Implement a lock-free queue using hazard pointers
template<typename T>
class LockFreeQueue {
    // Your implementation here
public:
    void enqueue(T item);
    bool dequeue(T& result);
};
```

**Exercise 2: Custom Memory Pool**
```cpp
// TODO: Create a memory pool for a game engine that:
// - Manages objects of different sizes
// - Provides fast allocation/deallocation
// - Tracks memory usage statistics
class GameMemoryPool {
    // Your implementation here
};
```

**Exercise 3: Secure String Class**
```cpp
// TODO: Implement a secure string class that:
// - Stores passwords safely in memory
// - Prevents memory dumps
// - Securely wipes data on destruction
class SecureString {
    // Your implementation here
};
```

## Study Materials & Resources

### Essential Reading
- **"The Art of Multiprocessor Programming"** by Maurice Herlihy - Lock-free algorithms
- **"Memory Management: Algorithms and Implementation in C/C++"** by Bill Blunden
- **"Effective Modern C++"** by Scott Meyers - Smart pointers and memory management
- **Intel Software Developer Manual** - Memory ordering and atomics

### Online Resources
- [Lock-Free Programming](https://preshing.com/20120612/an-introduction-to-lock-free-programming/)
- [Memory Ordering](https://en.cppreference.com/w/cpp/atomic/memory_order)
- [Secure Coding Practices](https://wiki.sei.cmu.edu/confluence/display/c/MEM)

### Tools and Libraries
- **AddressSanitizer** - Memory error detection
- **Valgrind/Memcheck** - Memory profiling and leak detection
- **Intel Inspector** - Threading and memory error detection
- **Microsoft Application Verifier** - Windows memory debugging

### Development Environment
```bash
# Install required tools
sudo apt-get install valgrind
sudo apt-get install libc6-dbg

# Compile with sanitizers
g++ -fsanitize=address -fsanitize=thread -g -O1 program.cpp

# Memory debugging
valgrind --tool=memcheck --leak-check=full ./program
valgrind --tool=helgrind ./program  # Thread safety
```

