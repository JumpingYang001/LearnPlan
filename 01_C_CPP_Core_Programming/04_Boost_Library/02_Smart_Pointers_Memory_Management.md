# Smart Pointers and Memory Management

*Duration: 1 week*

## Overview

This section covers Boost's smart pointer implementations and memory management strategies, including comparisons with modern C++ standard library equivalents.

## Learning Topics

### boost::shared_ptr (pre-C++11)
- Reference-counted smart pointer
- Thread safety considerations
- Custom deleters
- Weak pointer interactions

### boost::intrusive_ptr
- Intrusive reference counting
- Performance benefits
- Custom reference counting implementations
- Use cases and trade-offs

### boost::scoped_ptr
- Non-copyable, non-transferable smart pointer
- RAII principles
- Comparison with std::unique_ptr
- Legacy code considerations

### Object Pools and Memory Management Strategies
- boost::pool library
- Object pool patterns
- Memory pool allocators
- Performance optimization techniques

### Comparison with std Smart Pointers
- Migration strategies from Boost to std
- Performance comparisons
- Feature differences
- When to use Boost vs std

## Code Examples

### Basic shared_ptr Usage
```cpp
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <iostream>

class Resource {
public:
    Resource(int id) : id_(id) {
        std::cout << "Resource " << id_ << " created\n";
    }
    
    ~Resource() {
        std::cout << "Resource " << id_ << " destroyed\n";
    }
    
    void use() const {
        std::cout << "Using resource " << id_ << "\n";
    }
    
private:
    int id_;
};

void demonstrate_shared_ptr() {
    boost::shared_ptr<Resource> ptr1 = boost::make_shared<Resource>(1);
    
    {
        boost::shared_ptr<Resource> ptr2 = ptr1;
        std::cout << "Reference count: " << ptr1.use_count() << "\n";
        ptr2->use();
    } // ptr2 goes out of scope
    
    std::cout << "Reference count: " << ptr1.use_count() << "\n";
    ptr1->use();
} // ptr1 goes out of scope, Resource destroyed
```

### Custom Deleter Example
```cpp
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <cstdlib>

void custom_deleter(int* ptr) {
    std::cout << "Custom deleter called for value: " << *ptr << "\n";
    delete ptr;
}

void demonstrate_custom_deleter() {
    boost::shared_ptr<int> ptr(new int(42), custom_deleter);
    std::cout << "Value: " << *ptr << "\n";
} // custom_deleter will be called
```

### intrusive_ptr Example
```cpp
#include <boost/intrusive_ptr.hpp>
#include <iostream>

class IntrusiveResource {
public:
    IntrusiveResource() : ref_count_(0) {
        std::cout << "IntrusiveResource created\n";
    }
    
    ~IntrusiveResource() {
        std::cout << "IntrusiveResource destroyed\n";
    }
    
    void use() const {
        std::cout << "Using intrusive resource\n";
    }
    
private:
    friend void intrusive_ptr_add_ref(IntrusiveResource* p);
    friend void intrusive_ptr_release(IntrusiveResource* p);
    
    mutable int ref_count_;
};

void intrusive_ptr_add_ref(IntrusiveResource* p) {
    ++p->ref_count_;
}

void intrusive_ptr_release(IntrusiveResource* p) {
    if (--p->ref_count_ == 0) {
        delete p;
    }
}

void demonstrate_intrusive_ptr() {
    boost::intrusive_ptr<IntrusiveResource> ptr(new IntrusiveResource);
    ptr->use();
}
```

### scoped_ptr Example
```cpp
#include <boost/scoped_ptr.hpp>
#include <iostream>

class ScopedResource {
public:
    ScopedResource(const std::string& name) : name_(name) {
        std::cout << "ScopedResource " << name_ << " created\n";
    }
    
    ~ScopedResource() {
        std::cout << "ScopedResource " << name_ << " destroyed\n";
    }
    
    void use() const {
        std::cout << "Using " << name_ << "\n";
    }
    
private:
    std::string name_;
};

void demonstrate_scoped_ptr() {
    boost::scoped_ptr<ScopedResource> ptr(new ScopedResource("test"));
    ptr->use();
    
    // ptr.reset(new ScopedResource("test2")); // Replace resource
    // Cannot copy or assign scoped_ptr
} // Automatic cleanup
```

### Object Pool Example
```cpp
#include <boost/pool/pool.hpp>
#include <boost/pool/object_pool.hpp>
#include <iostream>
#include <vector>

class PooledObject {
public:
    PooledObject(int value) : value_(value) {
        std::cout << "PooledObject " << value_ << " constructed\n";
    }
    
    ~PooledObject() {
        std::cout << "PooledObject " << value_ << " destructed\n";
    }
    
    int getValue() const { return value_; }
    
private:
    int value_;
};

void demonstrate_object_pool() {
    boost::object_pool<PooledObject> pool;
    
    // Allocate objects from pool
    std::vector<PooledObject*> objects;
    for (int i = 0; i < 5; ++i) {
        objects.push_back(pool.construct(i));
    }
    
    // Use objects
    for (auto* obj : objects) {
        std::cout << "Object value: " << obj->getValue() << "\n";
    }
    
    // Return objects to pool
    for (auto* obj : objects) {
        pool.destroy(obj);
    }
    
    // Pool automatically manages memory
}
```

## Practical Exercises

1. **Smart Pointer Comparison**
   - Implement the same functionality using boost::shared_ptr and std::shared_ptr
   - Measure performance differences
   - Compare memory usage

2. **Custom Reference Counting**
   - Create a class with intrusive reference counting
   - Implement thread-safe reference counting
   - Compare with shared_ptr performance

3. **Memory Pool Implementation**
   - Create a custom object pool for a specific class
   - Measure allocation performance vs standard new/delete
   - Implement pool size tuning

4. **Legacy Code Migration**
   - Convert code using raw pointers to smart pointers
   - Handle circular reference scenarios
   - Implement proper RAII patterns

## Performance Considerations

### Reference Counting Overhead
- Atomic operations cost
- Cache line bouncing in multithreaded scenarios
- Memory overhead of control blocks

### Pool Allocation Benefits
- Reduced memory fragmentation
- Faster allocation/deallocation
- Better cache locality

### Migration Strategies
- Gradual replacement of raw pointers
- Interface compatibility considerations
- Testing strategies for smart pointer adoption

## Best Practices

1. **Prefer make_shared over new**
   - Single allocation for object and control block
   - Exception safety
   - Better performance

2. **Use weak_ptr to break cycles**
   - Identify potential circular references
   - Design with ownership hierarchies
   - Implement observer patterns safely

3. **Choose the right smart pointer**
   - shared_ptr for shared ownership
   - scoped_ptr/unique_ptr for exclusive ownership
   - intrusive_ptr for performance-critical scenarios

## Assessment

- Understands different smart pointer types and their use cases
- Can implement custom deleters and reference counting
- Knows when to use object pools for performance
- Can migrate legacy code to use smart pointers safely

## Next Steps

Move on to [Containers and Data Structures](03_Containers_Data_Structures.md) to explore Boost's container libraries.
