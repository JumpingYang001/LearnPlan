# STL Allocators

*Part of STL Learning Track - 1 week*

## Overview

STL allocators are objects responsible for encapsulating memory allocation and deallocation for STL containers. They provide a standardized interface for memory management, allowing containers to be decoupled from specific memory allocation strategies. This enables custom memory management policies and optimizations.

## Allocator Concept

### Basic Allocator Requirements
```cpp
#include <memory>
#include <vector>
#include <iostream>

void basic_allocator_example() {
    // Default allocator
    std::vector<int> vec1;
    
    // Explicit allocator specification
    std::vector<int, std::allocator<int>> vec2;
    
    // Both are equivalent
    std::cout << "vec1 and vec2 use the same allocator type" << std::endl;
    
    // Get allocator from container
    auto alloc = vec1.get_allocator();
    
    // Allocate raw memory
    int* ptr = alloc.allocate(10); // Allocate space for 10 integers
    
    // Construct objects in allocated memory
    for (int i = 0; i < 10; ++i) {
        std::allocator_traits<decltype(alloc)>::construct(alloc, ptr + i, i * i);
    }
    
    // Use the objects
    std::cout << "Constructed values: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << ptr[i] << " ";
    }
    std::cout << std::endl;
    
    // Destroy objects
    for (int i = 0; i < 10; ++i) {
        std::allocator_traits<decltype(alloc)>::destroy(alloc, ptr + i);
    }
    
    // Deallocate memory
    alloc.deallocate(ptr, 10);
}
```

## std::allocator

### Using std::allocator Directly
```cpp
#include <memory>
#include <string>
#include <iostream>

void std_allocator_examples() {
    std::allocator<std::string> alloc;
    
    // Allocate memory for 5 strings
    std::string* ptr = alloc.allocate(5);
    
    try {
        // Construct strings
        std::allocator_traits<std::allocator<std::string>>::construct(
            alloc, ptr + 0, "Hello");
        std::allocator_traits<std::allocator<std::string>>::construct(
            alloc, ptr + 1, "World");
        std::allocator_traits<std::allocator<std::string>>::construct(
            alloc, ptr + 2, "From");
        std::allocator_traits<std::allocator<std::string>>::construct(
            alloc, ptr + 3, "STL");
        std::allocator_traits<std::allocator<std::string>>::construct(
            alloc, ptr + 4, "Allocator");
        
        // Use the strings
        std::cout << "Constructed strings: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << ptr[i] << " ";
        }
        std::cout << std::endl;
        
        // Destroy strings (important for proper cleanup)
        for (int i = 0; i < 5; ++i) {
            std::allocator_traits<std::allocator<std::string>>::destroy(alloc, ptr + i);
        }
    }
    catch (...) {
        // Make sure to deallocate even if construction fails
        alloc.deallocate(ptr, 5);
        throw;
    }
    
    // Deallocate memory
    alloc.deallocate(ptr, 5);
}
```

### Allocator with Containers
```cpp
#include <vector>
#include <list>
#include <memory>
#include <iostream>

void allocator_with_containers() {
    // Vector with custom allocator instance  
    std::allocator<int> custom_alloc;
    std::vector<int, std::allocator<int>> vec(custom_alloc);
    
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    
    std::cout << "Vector contents: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // List with allocator
    std::list<std::string> str_list({"apple", "banana", "cherry"});
    
    std::cout << "List contents: ";
    for (const auto& item : str_list) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Check allocator equality
    auto vec_alloc = vec.get_allocator();
    if (vec_alloc == custom_alloc) {
        std::cout << "Allocators are equal" << std::endl;
    }
}
```

## Custom Allocator Implementation

### Basic Custom Allocator
```cpp
#include <memory>
#include <vector>
#include <iostream>
#include <cstdlib>

template<typename T>
class DebugAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    // Rebind for different types
    template<typename U>
    struct rebind {
        using other = DebugAllocator<U>;
    };
    
    DebugAllocator() = default;
    
    template<typename U>
    DebugAllocator(const DebugAllocator<U>&) {}
    
    pointer allocate(size_type n) {
        std::cout << "Allocating " << n << " objects of size " 
                  << sizeof(T) << " bytes each" << std::endl;
        
        pointer ptr = static_cast<pointer>(std::malloc(n * sizeof(T)));
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        std::cout << "Allocated at address: " << ptr << std::endl;
        return ptr;
    }
    
    void deallocate(pointer ptr, size_type n) {
        std::cout << "Deallocating " << n << " objects at address: " 
                  << ptr << std::endl;
        std::free(ptr);
    }
    
    // Optional: construct and destroy methods
    template<typename U, typename... Args>
    void construct(U* ptr, Args&&... args) {
        std::cout << "Constructing object at: " << ptr << std::endl;
        new(ptr) U(std::forward<Args>(args)...);
    }
    
    template<typename U>
    void destroy(U* ptr) {
        std::cout << "Destroying object at: " << ptr << std::endl;
        ptr->~U();
    }
    
    // Equality comparison
    template<typename U>
    bool operator==(const DebugAllocator<U>&) const { return true; }
    
    template<typename U>
    bool operator!=(const DebugAllocator<U>&) const { return false; }
};

void custom_allocator_example() {
    std::cout << "=== Custom Debug Allocator ===" << std::endl;
    
    std::vector<int, DebugAllocator<int>> vec;
    
    std::cout << "\nPushing elements:" << std::endl;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    vec.push_back(4);
    vec.push_back(5);
    
    std::cout << "\nVector contents: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nVector being destroyed..." << std::endl;
}
```

### Memory Pool Allocator
```cpp
#include <memory>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>

template<typename T, std::size_t PoolSize = 1024>
class PoolAllocator {
private:
    struct Block {
        alignas(T) char data[sizeof(T)];
        Block* next;
    };
    
    Block* free_list_;
    std::unique_ptr<Block[]> pool_;
    std::size_t pool_size_;
    
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = PoolAllocator<U, PoolSize>;
    };
    
    PoolAllocator() : pool_size_(PoolSize) {
        pool_ = std::make_unique<Block[]>(pool_size_);
        
        // Initialize free list
        free_list_ = &pool_[0];
        for (std::size_t i = 0; i < pool_size_ - 1; ++i) {
            pool_[i].next = &pool_[i + 1];
        }
        pool_[pool_size_ - 1].next = nullptr;
        
        std::cout << "Pool allocator initialized with " << pool_size_ 
                  << " blocks" << std::endl;
    }
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U, PoolSize>& other) 
        : PoolAllocator() {}
    
    T* allocate(size_type n) {
        if (n != 1) {
            throw std::bad_alloc(); // Pool allocator only supports single objects
        }
        
        if (!free_list_) {
            throw std::bad_alloc(); // Pool exhausted
        }
        
        Block* block = free_list_;
        free_list_ = free_list_->next;
        
        std::cout << "Allocated block from pool" << std::endl;
        return reinterpret_cast<T*>(block);
    }
    
    void deallocate(T* ptr, size_type n) {
        if (n != 1) return;
        
        Block* block = reinterpret_cast<Block*>(ptr);
        block->next = free_list_;
        free_list_ = block;
        
        std::cout << "Returned block to pool" << std::endl;
    }
    
    template<typename U>
    bool operator==(const PoolAllocator<U, PoolSize>&) const { return true; }
    
    template<typename U>
    bool operator!=(const PoolAllocator<U, PoolSize>&) const { return false; }
};

void pool_allocator_example() {
    std::cout << "\n=== Pool Allocator Example ===" << std::endl;
    
    {
        std::vector<int, PoolAllocator<int, 100>> vec;
        
        // This will use pool allocation for individual elements
        // Note: vector may allocate multiple elements at once,
        // so this is mainly for demonstration
        std::cout << "Creating vector with pool allocator" << std::endl;
        
        // For better demonstration, let's use the allocator directly
        PoolAllocator<int, 10> pool_alloc;
        
        std::vector<int*> pointers;
        
        // Allocate several single objects
        for (int i = 0; i < 5; ++i) {
            int* ptr = pool_alloc.allocate(1);
            std::allocator_traits<decltype(pool_alloc)>::construct(pool_alloc, ptr, i * 10);
            pointers.push_back(ptr);
        }
        
        std::cout << "Allocated values: ";
        for (auto ptr : pointers) {
            std::cout << *ptr << " ";
        }
        std::cout << std::endl;
        
        // Deallocate
        for (auto ptr : pointers) {
            std::allocator_traits<decltype(pool_alloc)>::destroy(pool_alloc, ptr);
            pool_alloc.deallocate(ptr, 1);
        }
    }
    
    std::cout << "Pool allocator example completed" << std::endl;
}
```

### Tracking Allocator
```cpp
#include <memory>
#include <vector>
#include <iostream>
#include <atomic>

template<typename T>
class TrackingAllocator {
private:
    static std::atomic<std::size_t> total_allocated_;
    static std::atomic<std::size_t> total_deallocated_;
    static std::atomic<std::size_t> current_usage_;
    static std::atomic<std::size_t> peak_usage_;
    
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = TrackingAllocator<U>;
    };
    
    TrackingAllocator() = default;
    
    template<typename U>
    TrackingAllocator(const TrackingAllocator<U>&) {}
    
    T* allocate(size_type n) {
        std::size_t bytes = n * sizeof(T);
        T* ptr = static_cast<T*>(std::malloc(bytes));
        
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        total_allocated_ += bytes;
        current_usage_ += bytes;
        
        std::size_t current = current_usage_.load();
        std::size_t peak = peak_usage_.load();
        while (current > peak && !peak_usage_.compare_exchange_weak(peak, current)) {
            peak = peak_usage_.load();
        }
        
        return ptr;
    }
    
    void deallocate(T* ptr, size_type n) {
        std::size_t bytes = n * sizeof(T);
        total_deallocated_ += bytes;
        current_usage_ -= bytes;
        std::free(ptr);
    }
    
    template<typename U>
    bool operator==(const TrackingAllocator<U>&) const { return true; }
    
    template<typename U>
    bool operator!=(const TrackingAllocator<U>&) const { return false; }
    
    static void print_stats() {
        std::cout << "Memory Statistics:" << std::endl;
        std::cout << "  Total allocated: " << total_allocated_ << " bytes" << std::endl;
        std::cout << "  Total deallocated: " << total_deallocated_ << " bytes" << std::endl;
        std::cout << "  Current usage: " << current_usage_ << " bytes" << std::endl;
        std::cout << "  Peak usage: " << peak_usage_ << " bytes" << std::endl;
    }
    
    static void reset_stats() {
        total_allocated_ = 0;
        total_deallocated_ = 0;
        current_usage_ = 0;
        peak_usage_ = 0;
    }
};

// Static member definitions
template<typename T>
std::atomic<std::size_t> TrackingAllocator<T>::total_allocated_{0};

template<typename T>
std::atomic<std::size_t> TrackingAllocator<T>::total_deallocated_{0};

template<typename T>
std::atomic<std::size_t> TrackingAllocator<T>::current_usage_{0};

template<typename T>
std::atomic<std::size_t> TrackingAllocator<T>::peak_usage_{0};

void tracking_allocator_example() {
    std::cout << "\n=== Tracking Allocator Example ===" << std::endl;
    
    TrackingAllocator<int>::reset_stats();
    
    {
        std::vector<int, TrackingAllocator<int>> vec;
        
        std::cout << "Initial state:" << std::endl;
        TrackingAllocator<int>::print_stats();
        
        // Add elements to trigger allocations
        for (int i = 0; i < 1000; ++i) {
            vec.push_back(i);
        }
        
        std::cout << "\nAfter adding 1000 elements:" << std::endl;
        TrackingAllocator<int>::print_stats();
        
        // Create another vector
        std::vector<int, TrackingAllocator<int>> vec2(500, 42);
        
        std::cout << "\nAfter creating second vector with 500 elements:" << std::endl;
        TrackingAllocator<int>::print_stats();
        
        // Clear first vector
        vec.clear();
        vec.shrink_to_fit();
        
        std::cout << "\nAfter clearing and shrinking first vector:" << std::endl;
        TrackingAllocator<int>::print_stats();
    }
    
    std::cout << "\nAfter all vectors destroyed:" << std::endl;
    TrackingAllocator<int>::print_stats();
}
```

## Stateful Allocators

### Allocator with State
```cpp
#include <memory>
#include <vector>
#include <iostream>
#include <string>

template<typename T>
class StatefulAllocator {
private:
    std::string name_;
    int id_;
    
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = StatefulAllocator<U>;
    };
    
    StatefulAllocator(const std::string& name, int id) 
        : name_(name), id_(id) {}
    
    template<typename U>
    StatefulAllocator(const StatefulAllocator<U>& other)
        : name_(other.name_), id_(other.id_) {}
    
    T* allocate(size_type n) {
        std::cout << "Allocator '" << name_ << "' (id: " << id_ 
                  << ") allocating " << n << " objects" << std::endl;
        
        T* ptr = static_cast<T*>(std::malloc(n * sizeof(T)));
        if (!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }
    
    void deallocate(T* ptr, size_type n) {
        std::cout << "Allocator '" << name_ << "' (id: " << id_ 
                  << ") deallocating " << n << " objects" << std::endl;
        std::free(ptr);
    }
    
    // Stateful allocators must define proper equality
    bool operator==(const StatefulAllocator& other) const {
        return name_ == other.name_ && id_ == other.id_;
    }
    
    bool operator!=(const StatefulAllocator& other) const {
        return !(*this == other);
    }
    
    const std::string& name() const { return name_; }
    int id() const { return id_; }
    
    // Allow access to private members for rebind
    template<typename U>
    friend class StatefulAllocator;
};

void stateful_allocator_example() {
    std::cout << "\n=== Stateful Allocator Example ===" << std::endl;
    
    StatefulAllocator<int> alloc1("Primary", 1);
    StatefulAllocator<int> alloc2("Secondary", 2);
    
    std::vector<int, StatefulAllocator<int>> vec1(alloc1);
    std::vector<int, StatefulAllocator<int>> vec2(alloc2);
    
    vec1.push_back(10);
    vec1.push_back(20);
    
    vec2.push_back(30);
    vec2.push_back(40);
    
    std::cout << "vec1 allocator: " << vec1.get_allocator().name() 
              << " (id: " << vec1.get_allocator().id() << ")" << std::endl;
    std::cout << "vec2 allocator: " << vec2.get_allocator().name() 
              << " (id: " << vec2.get_allocator().id() << ")" << std::endl;
    
    // Test allocator equality
    if (vec1.get_allocator() == vec2.get_allocator()) {
        std::cout << "Allocators are equal" << std::endl;
    } else {
        std::cout << "Allocators are different" << std::endl;
    }
}
```

## Polymorphic Allocators (C++17)

### std::pmr::memory_resource
```cpp
#ifdef __cpp_lib_memory_resource
#include <memory_resource>
#include <vector>
#include <iostream>
#include <array>

void pmr_basic_example() {
    std::cout << "\n=== Polymorphic Memory Resources (C++17) ===" << std::endl;
    
    // Use default memory resource
    std::pmr::vector<int> vec1;
    vec1.push_back(1);
    vec1.push_back(2);
    vec1.push_back(3);
    
    std::cout << "Default PMR vector size: " << vec1.size() << std::endl;
    
    // Use monotonic buffer resource
    std::array<char, 1024> buffer;
    std::pmr::monotonic_buffer_resource mbr(buffer.data(), buffer.size());
    
    std::pmr::vector<int> vec2(&mbr);
    for (int i = 0; i < 10; ++i) {
        vec2.push_back(i * i);
    }
    
    std::cout << "Monotonic buffer vector contents: ";
    for (const auto& item : vec2) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Use unsynchronized pool resource
    std::pmr::unsynchronized_pool_resource pool;
    std::pmr::vector<std::string> vec3(&pool);
    
    vec3.emplace_back("Hello");
    vec3.emplace_back("Polymorphic");
    vec3.emplace_back("Memory");
    vec3.emplace_back("Resources");
    
    std::cout << "Pool resource vector contents: ";
    for (const auto& str : vec3) {
        std::cout << str << " ";
    }
    std::cout << std::endl;
}

// Custom memory resource
class LoggingMemoryResource : public std::pmr::memory_resource {
private:
    std::pmr::memory_resource* upstream_;
    
public:
    explicit LoggingMemoryResource(std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
        : upstream_(upstream) {}
    
protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        std::cout << "Allocating " << bytes << " bytes with alignment " << alignment << std::endl;
        return upstream_->allocate(bytes, alignment);
    }
    
    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override {
        std::cout << "Deallocating " << bytes << " bytes with alignment " << alignment << std::endl;
        upstream_->deallocate(ptr, bytes, alignment);
    }
    
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }
};

void custom_pmr_example() {
    std::cout << "\n=== Custom PMR Example ===" << std::endl;
    
    LoggingMemoryResource logging_resource;
    
    std::pmr::vector<int> vec(&logging_resource);
    
    std::cout << "Adding elements to PMR vector:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        vec.push_back(i * 10);
    }
    
    std::cout << "Vector contents: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

#endif // __cpp_lib_memory_resource
```

## Allocator Performance Comparison

### Benchmarking Different Allocators
```cpp
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <random>

template<typename Allocator>
void benchmark_allocator(const std::string& name, int iterations = 10000) {
    auto start = std::chrono::high_resolution_clock::now();
    
    {
        std::vector<int, Allocator> vec;
        
        // Random insertions and deletions
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 1000);
        
        for (int i = 0; i < iterations; ++i) {
            vec.push_back(dis(gen));
            
            if (i % 100 == 0 && !vec.empty()) {
                vec.erase(vec.begin() + vec.size() / 2);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << name << " took: " << duration.count() << " microseconds" << std::endl;
}

void allocator_performance_comparison() {
    std::cout << "\n=== Allocator Performance Comparison ===" << std::endl;
    
    const int iterations = 50000;
    
    // Benchmark different allocators
    benchmark_allocator<std::allocator<int>>("std::allocator", iterations);
    benchmark_allocator<DebugAllocator<int>>("DebugAllocator", iterations);
    benchmark_allocator<TrackingAllocator<int>>("TrackingAllocator", iterations);
    
    std::cout << "\nNote: Performance may vary based on usage patterns and system" << std::endl;
}
```

## Allocator Best Practices

### RAII Allocator Wrapper
```cpp
#include <memory>
#include <iostream>

template<typename T, typename Allocator = std::allocator<T>>
class AllocatorRAII {
private:
    Allocator alloc_;
    T* ptr_;
    std::size_t size_;
    
public:
    explicit AllocatorRAII(std::size_t size, const Allocator& alloc = Allocator())
        : alloc_(alloc), size_(size) {
        ptr_ = alloc_.allocate(size_);
        
        // Construct objects
        for (std::size_t i = 0; i < size_; ++i) {
            std::allocator_traits<Allocator>::construct(alloc_, ptr_ + i);
        }
    }
    
    ~AllocatorRAII() {
        // Destroy objects
        for (std::size_t i = 0; i < size_; ++i) {
            std::allocator_traits<Allocator>::destroy(alloc_, ptr_ + i);
        }
        
        // Deallocate memory
        alloc_.deallocate(ptr_, size_);
    }
    
    // Non-copyable
    AllocatorRAII(const AllocatorRAII&) = delete;
    AllocatorRAII& operator=(const AllocatorRAII&) = delete;
    
    // Moveable
    AllocatorRAII(AllocatorRAII&& other) noexcept
        : alloc_(std::move(other.alloc_)), ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    AllocatorRAII& operator=(AllocatorRAII&& other) noexcept {
        if (this != &other) {
            // Cleanup current resources
            this->~AllocatorRAII();
            
            // Move from other
            alloc_ = std::move(other.alloc_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T& operator[](std::size_t index) { return ptr_[index]; }
    const T& operator[](std::size_t index) const { return ptr_[index]; }
    
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    
    std::size_t size() const { return size_; }
};

void raii_allocator_example() {
    std::cout << "\n=== RAII Allocator Wrapper Example ===" << std::endl;
    
    {
        AllocatorRAII<int, TrackingAllocator<int>> array(10);
        
        // Initialize array
        for (std::size_t i = 0; i < array.size(); ++i) {
            array[i] = static_cast<int>(i * i);
        }
        
        std::cout << "Array contents: ";
        for (std::size_t i = 0; i < array.size(); ++i) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
        
        TrackingAllocator<int>::print_stats();
    } // Automatic cleanup
    
    std::cout << "\nAfter RAII cleanup:" << std::endl;
    TrackingAllocator<int>::print_stats();
}
```

## Complete Example Program

```cpp
#include <iostream>
#include <memory>
#include <vector>

int main() {
    std::cout << "=== STL Allocators Examples ===" << std::endl;
    
    std::cout << "\n--- Basic Allocator Usage ---" << std::endl;
    basic_allocator_example();
    std_allocator_examples();
    allocator_with_containers();
    
    std::cout << "\n--- Custom Allocators ---" << std::endl;
    custom_allocator_example();
    pool_allocator_example();
    tracking_allocator_example();
    
    std::cout << "\n--- Stateful Allocators ---" << std::endl;
    stateful_allocator_example();
    
#ifdef __cpp_lib_memory_resource
    std::cout << "\n--- Polymorphic Memory Resources ---" << std::endl;
    pmr_basic_example();
    custom_pmr_example();
#endif
    
    std::cout << "\n--- Performance Comparison ---" << std::endl;
    allocator_performance_comparison();
    
    std::cout << "\n--- Best Practices ---" << std::endl;
    raii_allocator_example();
    
    return 0;
}
```

## Key Concepts Summary

1. **Allocator Concept**: Interface for memory management in STL containers
2. **std::allocator**: Default allocator using new/delete
3. **Custom Allocators**: Implement specific memory management strategies
4. **Stateful Allocators**: Carry configuration or state information
5. **Polymorphic Allocators**: Runtime polymorphic memory resources (C++17)
6. **Performance**: Different allocators have different performance characteristics

## Best Practices

1. **Use std::allocator_traits**: Instead of calling allocator methods directly
2. **Handle exceptions**: Ensure proper cleanup in custom allocators
3. **Implement all required members**: For custom allocators to work with containers
4. **Consider stateful vs stateless**: Affects container copying and assignment
5. **Profile before optimizing**: Custom allocators should solve real performance problems
6. **RAII principles**: Ensure proper resource management

## When to Use Custom Allocators

1. **Memory pools**: For frequent small allocations
2. **Tracking**: Monitor memory usage patterns
3. **Special memory**: GPU memory, shared memory, etc.
4. **Performance**: Reduce fragmentation or allocation overhead
5. **Debugging**: Add debugging capabilities to memory operations

## Exercises

1. Implement a stack allocator for temporary objects
2. Create a thread-local memory pool allocator
3. Build a garbage-collecting allocator with reference counting
4. Implement an allocator that uses memory mapping
5. Create a debugging allocator that detects memory leaks and buffer overruns
6. Write an allocator that aligns all allocations to cache line boundaries
7. Implement a hierarchical allocator system with fallback strategies
