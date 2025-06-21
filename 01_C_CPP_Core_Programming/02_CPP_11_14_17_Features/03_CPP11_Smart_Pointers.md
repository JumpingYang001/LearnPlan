# C++11 Smart Pointers

## Overview

Smart pointers are one of the most important features introduced in C++11. They provide automatic memory management, helping to prevent memory leaks, dangling pointers, and other common memory-related bugs. The three main smart pointers are `std::unique_ptr`, `std::shared_ptr`, and `std::weak_ptr`.

## Key Smart Pointer Types

### 1. std::unique_ptr

`std::unique_ptr` represents exclusive ownership of a resource. It cannot be copied but can be moved.

#### Basic Usage

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Resource {
private:
    std::string name;
    int id;
    
public:
    Resource(const std::string& n, int i) : name(n), id(i) {
        std::cout << "Resource created: " << name << " (ID: " << id << ")" << std::endl;
    }
    
    ~Resource() {
        std::cout << "Resource destroyed: " << name << " (ID: " << id << ")" << std::endl;
    }
    
    void use() const {
        std::cout << "Using resource: " << name << " (ID: " << id << ")" << std::endl;
    }
    
    const std::string& get_name() const { return name; }
    int get_id() const { return id; }
};

void demonstrate_unique_ptr_basics() {
    std::cout << "\n=== std::unique_ptr Basics ===" << std::endl;
    
    // Creating unique_ptr
    std::unique_ptr<Resource> ptr1 = std::make_unique<Resource>("Resource1", 1);
    
    // Alternative creation (less preferred)
    std::unique_ptr<Resource> ptr2(new Resource("Resource2", 2));
    
    // Using the pointer
    ptr1->use();
    (*ptr1).use();  // Alternative syntax
    
    // Checking if pointer is valid
    if (ptr1) {
        std::cout << "ptr1 is valid" << std::endl;
    }
    
    // Getting raw pointer (be careful!)
    Resource* raw_ptr = ptr1.get();
    raw_ptr->use();
    
    // Releasing ownership
    Resource* released = ptr1.release();
    if (!ptr1) {
        std::cout << "ptr1 is now empty" << std::endl;
    }
    
    // Manual cleanup (since we released)
    delete released;
    
    // Reset with new resource
    ptr2.reset(new Resource("Resource3", 3));
    
    // ptr2 will be automatically destroyed when going out of scope
}

// unique_ptr cannot be copied but can be moved
std::unique_ptr<Resource> create_resource(const std::string& name, int id) {
    return std::make_unique<Resource>(name, id);
}

void transfer_ownership(std::unique_ptr<Resource> ptr) {
    std::cout << "Received ownership of: " << ptr->get_name() << std::endl;
    ptr->use();
    // ptr is destroyed here, resource is cleaned up
}

void demonstrate_unique_ptr_move() {
    std::cout << "\n=== std::unique_ptr Move Semantics ===" << std::endl;
    
    // Create through factory function
    auto ptr1 = create_resource("MovableResource", 10);
    
    // Move to another unique_ptr
    auto ptr2 = std::move(ptr1);
    
    if (!ptr1 && ptr2) {
        std::cout << "Ownership successfully transferred" << std::endl;
    }
    
    // Transfer ownership to function
    transfer_ownership(std::move(ptr2));
    
    if (!ptr2) {
        std::cout << "ptr2 is now empty after transfer" << std::endl;
    }
}
```

#### Arrays with unique_ptr

```cpp
#include <iostream>
#include <memory>

void demonstrate_unique_ptr_arrays() {
    std::cout << "\n=== std::unique_ptr with Arrays ===" << std::endl;
    
    // Array of built-in types
    std::unique_ptr<int[]> int_array = std::make_unique<int[]>(10);
    
    // Initialize array
    for (int i = 0; i < 10; ++i) {
        int_array[i] = i * i;
    }
    
    // Use array
    for (int i = 0; i < 10; ++i) {
        std::cout << "int_array[" << i << "] = " << int_array[i] << std::endl;
    }
    
    // Array of objects
    std::unique_ptr<Resource[]> resource_array = std::make_unique<Resource[]>(3);
    
    // Note: Default constructor must be available for object arrays
    // For non-default constructible types, consider std::vector instead
}
```

#### Custom Deleters

```cpp
#include <iostream>
#include <memory>
#include <cstdio>

// Custom deleter for FILE*
struct FileDeleter {
    void operator()(FILE* file) const {
        if (file) {
            std::cout << "Closing file" << std::endl;
            std::fclose(file);
        }
    }
};

// Custom deleter for arrays allocated with malloc
struct MallocDeleter {
    void operator()(void* ptr) const {
        std::cout << "Freeing malloc'd memory" << std::endl;
        std::free(ptr);
    }
};

void demonstrate_custom_deleters() {
    std::cout << "\n=== Custom Deleters ===" << std::endl;
    
    // File pointer with custom deleter
    std::unique_ptr<FILE, FileDeleter> file_ptr(std::fopen("test.txt", "w"));
    if (file_ptr) {
        std::fprintf(file_ptr.get(), "Hello, custom deleter!\n");
        // File will be automatically closed when file_ptr goes out of scope
    }
    
    // Malloc'd memory with custom deleter
    std::unique_ptr<int, MallocDeleter> malloc_ptr(
        static_cast<int*>(std::malloc(sizeof(int) * 10))
    );
    
    if (malloc_ptr) {
        for (int i = 0; i < 10; ++i) {
            malloc_ptr.get()[i] = i;
        }
        // Memory will be freed with free() when malloc_ptr goes out of scope
    }
    
    // Lambda as custom deleter
    auto lambda_deleter = [](Resource* ptr) {
        std::cout << "Lambda deleter called" << std::endl;
        delete ptr;
    };
    
    std::unique_ptr<Resource, decltype(lambda_deleter)> lambda_ptr(
        new Resource("LambdaDeleted", 99), lambda_deleter
    );
}
```

### 2. std::shared_ptr

`std::shared_ptr` allows shared ownership of a resource through reference counting.

#### Basic Usage

```cpp
#include <iostream>
#include <memory>
#include <vector>

void demonstrate_shared_ptr_basics() {
    std::cout << "\n=== std::shared_ptr Basics ===" << std::endl;
    
    // Create shared_ptr
    std::shared_ptr<Resource> ptr1 = std::make_shared<Resource>("SharedResource", 20);
    std::cout << "Reference count: " << ptr1.use_count() << std::endl;
    
    // Share ownership
    std::shared_ptr<Resource> ptr2 = ptr1;
    std::cout << "Reference count after sharing: " << ptr1.use_count() << std::endl;
    
    // Another way to share
    std::shared_ptr<Resource> ptr3(ptr1);
    std::cout << "Reference count with 3 owners: " << ptr1.use_count() << std::endl;
    
    // Use the resource
    ptr1->use();
    ptr2->use();
    ptr3->use();
    
    // Reset one pointer
    ptr2.reset();
    std::cout << "Reference count after reset: " << ptr1.use_count() << std::endl;
    
    // Create new resource for ptr2
    ptr2 = std::make_shared<Resource>("AnotherShared", 21);
    std::cout << "ptr1 reference count: " << ptr1.use_count() << std::endl;
    std::cout << "ptr2 reference count: " << ptr2.use_count() << std::endl;
    
    // Check uniqueness
    if (ptr2.unique()) {
        std::cout << "ptr2 is the unique owner" << std::endl;
    }
}

// Sharing across functions
void use_shared_resource(std::shared_ptr<Resource> ptr) {
    std::cout << "Function received shared resource, ref count: " 
              << ptr.use_count() << std::endl;
    ptr->use();
    // Reference count decreases when function exits
}

void demonstrate_shared_ptr_sharing() {
    std::cout << "\n=== std::shared_ptr Sharing ===" << std::endl;
    
    auto shared_resource = std::make_shared<Resource>("GlobalShared", 30);
    std::cout << "Initial ref count: " << shared_resource.use_count() << std::endl;
    
    // Pass to function (copy)
    use_shared_resource(shared_resource);
    std::cout << "After function call: " << shared_resource.use_count() << std::endl;
    
    // Store in container
    std::vector<std::shared_ptr<Resource>> resources;
    resources.push_back(shared_resource);  // Another reference
    resources.push_back(shared_resource);  // Yet another reference
    
    std::cout << "After adding to vector: " << shared_resource.use_count() << std::endl;
    
    // Clear container
    resources.clear();
    std::cout << "After clearing vector: " << shared_resource.use_count() << std::endl;
}
```

#### Thread Safety

```cpp
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <atomic>

class ThreadSafeCounter {
private:
    std::atomic<int> count{0};
    std::string name;
    
public:
    ThreadSafeCounter(const std::string& n) : name(n) {
        std::cout << "Counter created: " << name << std::endl;
    }
    
    ~ThreadSafeCounter() {
        std::cout << "Counter destroyed: " << name << " (final count: " 
                  << count.load() << ")" << std::endl;
    }
    
    void increment() {
        ++count;
    }
    
    int get_count() const {
        return count.load();
    }
    
    const std::string& get_name() const { return name; }
};

void worker_thread(std::shared_ptr<ThreadSafeCounter> counter, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        counter->increment();
    }
    std::cout << "Thread finished, counter: " << counter->get_name() 
              << ", count: " << counter->get_count() << std::endl;
}

void demonstrate_shared_ptr_threading() {
    std::cout << "\n=== std::shared_ptr Thread Safety ===" << std::endl;
    
    auto counter = std::make_shared<ThreadSafeCounter>("ThreadCounter");
    std::cout << "Initial ref count: " << counter.use_count() << std::endl;
    
    // Create multiple threads sharing the same resource
    std::vector<std::thread> threads;
    const int num_threads = 4;
    const int iterations_per_thread = 1000;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker_thread, counter, iterations_per_thread);
    }
    
    std::cout << "Threads created, ref count: " << counter.use_count() << std::endl;
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "All threads completed" << std::endl;
    std::cout << "Final count: " << counter->get_count() << std::endl;
    std::cout << "Expected count: " << num_threads * iterations_per_thread << std::endl;
    std::cout << "Final ref count: " << counter.use_count() << std::endl;
}
```

### 3. std::weak_ptr

`std::weak_ptr` provides non-owning access to an object managed by `std::shared_ptr`, helping to break circular references.

#### Breaking Circular References

```cpp
#include <iostream>
#include <memory>
#include <string>

class Child;

class Parent {
private:
    std::string name;
    std::vector<std::shared_ptr<Child>> children;
    
public:
    Parent(const std::string& n) : name(n) {
        std::cout << "Parent created: " << name << std::endl;
    }
    
    ~Parent() {
        std::cout << "Parent destroyed: " << name << std::endl;
    }
    
    void add_child(std::shared_ptr<Child> child) {
        children.push_back(child);
    }
    
    const std::string& get_name() const { return name; }
    
    void list_children() const;
};

class Child {
private:
    std::string name;
    std::weak_ptr<Parent> parent;  // weak_ptr to break circular reference
    
public:
    Child(const std::string& n) : name(n) {
        std::cout << "Child created: " << name << std::endl;
    }
    
    ~Child() {
        std::cout << "Child destroyed: " << name << std::endl;
    }
    
    void set_parent(std::shared_ptr<Parent> p) {
        parent = p;
    }
    
    void visit_parent() {
        // Convert weak_ptr to shared_ptr
        if (auto p = parent.lock()) {
            std::cout << "Child " << name << " visiting parent " 
                      << p->get_name() << std::endl;
        } else {
            std::cout << "Child " << name << " has no parent (or parent destroyed)" 
                      << std::endl;
        }
    }
    
    const std::string& get_name() const { return name; }
};

void Parent::list_children() const {
    std::cout << "Parent " << name << " has children: ";
    for (const auto& child : children) {
        std::cout << child->get_name() << " ";
    }
    std::cout << std::endl;
}

void demonstrate_weak_ptr() {
    std::cout << "\n=== std::weak_ptr and Circular References ===" << std::endl;
    
    // Create parent and children
    auto parent = std::make_shared<Parent>("Dad");
    auto child1 = std::make_shared<Child>("Alice");
    auto child2 = std::make_shared<Child>("Bob");
    
    // Set up relationships
    parent->add_child(child1);
    parent->add_child(child2);
    child1->set_parent(parent);
    child2->set_parent(parent);
    
    // Use the relationships
    parent->list_children();
    child1->visit_parent();
    child2->visit_parent();
    
    std::cout << "Parent ref count: " << parent.use_count() << std::endl;  // Should be 1
    
    // Demonstrate weak_ptr expiration
    std::weak_ptr<Parent> weak_parent = parent;
    std::cout << "Weak pointer valid: " << !weak_parent.expired() << std::endl;
    
    // Reset parent (this will destroy the parent)
    parent.reset();
    
    std::cout << "After parent reset:" << std::endl;
    std::cout << "Weak pointer valid: " << !weak_parent.expired() << std::endl;
    
    // Children try to visit parent
    child1->visit_parent();
    child2->visit_parent();
    
    // Children will be destroyed when they go out of scope
}
```

#### Observer Pattern with weak_ptr

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <string>

class Observer {
public:
    virtual ~Observer() = default;
    virtual void notify(const std::string& message) = 0;
    virtual std::string get_name() const = 0;
};

class ConcreteObserver : public Observer {
private:
    std::string name;
    
public:
    ConcreteObserver(const std::string& n) : name(n) {
        std::cout << "Observer created: " << name << std::endl;
    }
    
    ~ConcreteObserver() {
        std::cout << "Observer destroyed: " << name << std::endl;
    }
    
    void notify(const std::string& message) override {
        std::cout << "Observer " << name << " received: " << message << std::endl;
    }
    
    std::string get_name() const override { return name; }
};

class Subject {
private:
    std::vector<std::weak_ptr<Observer>> observers;
    std::string name;
    
public:
    Subject(const std::string& n) : name(n) {
        std::cout << "Subject created: " << name << std::endl;
    }
    
    ~Subject() {
        std::cout << "Subject destroyed: " << name << std::endl;
    }
    
    void attach(std::shared_ptr<Observer> observer) {
        observers.push_back(observer);
        std::cout << "Observer " << observer->get_name() 
                  << " attached to subject " << name << std::endl;
    }
    
    void notify_all(const std::string& message) {
        std::cout << "Subject " << name << " notifying observers: " << message << std::endl;
        
        // Remove expired weak_ptrs and notify valid ones
        observers.erase(
            std::remove_if(observers.begin(), observers.end(),
                [&message](const std::weak_ptr<Observer>& weak_obs) {
                    if (auto obs = weak_obs.lock()) {
                        obs->notify(message);
                        return false;  // Keep this observer
                    } else {
                        std::cout << "Removing expired observer" << std::endl;
                        return true;   // Remove expired observer
                    }
                }),
            observers.end()
        );
    }
    
    size_t observer_count() const {
        return std::count_if(observers.begin(), observers.end(),
            [](const std::weak_ptr<Observer>& weak_obs) {
                return !weak_obs.expired();
            });
    }
};

void demonstrate_observer_pattern() {
    std::cout << "\n=== Observer Pattern with weak_ptr ===" << std::endl;
    
    auto subject = std::make_shared<Subject>("NewsService");
    
    // Create observers
    auto observer1 = std::make_shared<ConcreteObserver>("Observer1");
    auto observer2 = std::make_shared<ConcreteObserver>("Observer2");
    auto observer3 = std::make_shared<ConcreteObserver>("Observer3");
    
    // Attach observers
    subject->attach(observer1);
    subject->attach(observer2);
    subject->attach(observer3);
    
    std::cout << "Active observers: " << subject->observer_count() << std::endl;
    
    // Notify all observers
    subject->notify_all("Breaking news!");
    
    // Destroy one observer
    std::cout << "\nDestroying observer2..." << std::endl;
    observer2.reset();
    
    // Notify again (expired observer will be cleaned up)
    subject->notify_all("Weather update");
    
    std::cout << "Active observers after cleanup: " << subject->observer_count() << std::endl;
}
```

## Best Practices

### 1. Choosing the Right Smart Pointer

```cpp
#include <iostream>
#include <memory>
#include <vector>

// Guidelines for choosing smart pointers:

// Use std::unique_ptr when:
// - You need exclusive ownership
// - You want zero overhead (same as raw pointer)
// - You're working with factory functions
std::unique_ptr<Resource> create_unique_resource() {
    return std::make_unique<Resource>("UniqueResource", 100);
}

// Use std::shared_ptr when:
// - Multiple objects need to share ownership
// - You're implementing reference-counted data structures
// - You need thread-safe reference counting
class DataManager {
private:
    std::shared_ptr<std::vector<int>> data;
    
public:
    DataManager() : data(std::make_shared<std::vector<int>>()) {}
    
    std::shared_ptr<std::vector<int>> get_data() const {
        return data;  // Share ownership
    }
    
    void add_value(int value) {
        data->push_back(value);
    }
};

// Use std::weak_ptr when:
// - You need to observe an object without owning it
// - You want to break circular references
// - You're implementing observer patterns
class Cache {
private:
    std::vector<std::weak_ptr<Resource>> cached_resources;
    
public:
    void add_to_cache(std::shared_ptr<Resource> resource) {
        cached_resources.push_back(resource);
    }
    
    std::shared_ptr<Resource> get_from_cache(int id) {
        for (auto& weak_res : cached_resources) {
            if (auto res = weak_res.lock()) {
                if (res->get_id() == id) {
                    return res;
                }
            }
        }
        return nullptr;
    }
};
```

### 2. Performance Considerations

```cpp
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>

void performance_comparison() {
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    
    const size_t num_operations = 1000000;
    
    // Raw pointer performance
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<Resource*> raw_ptrs;
    for (size_t i = 0; i < num_operations; ++i) {
        raw_ptrs.push_back(new Resource("Raw" + std::to_string(i), i));
    }
    
    for (auto ptr : raw_ptrs) {
        delete ptr;
    }
    
    auto raw_end = std::chrono::high_resolution_clock::now();
    auto raw_duration = std::chrono::duration_cast<std::chrono::milliseconds>(raw_end - start);
    
    // unique_ptr performance
    auto unique_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::unique_ptr<Resource>> unique_ptrs;
    for (size_t i = 0; i < num_operations; ++i) {
        unique_ptrs.push_back(std::make_unique<Resource>("Unique" + std::to_string(i), i));
    }
    unique_ptrs.clear();  // Automatic cleanup
    
    auto unique_end = std::chrono::high_resolution_clock::now();
    auto unique_duration = std::chrono::duration_cast<std::chrono::milliseconds>(unique_end - unique_start);
    
    // shared_ptr performance
    auto shared_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::shared_ptr<Resource>> shared_ptrs;
    for (size_t i = 0; i < num_operations; ++i) {
        shared_ptrs.push_back(std::make_shared<Resource>("Shared" + std::to_string(i), i));
    }
    shared_ptrs.clear();  // Automatic cleanup
    
    auto shared_end = std::chrono::high_resolution_clock::now();
    auto shared_duration = std::chrono::duration_cast<std::chrono::milliseconds>(shared_end - shared_start);
    
    std::cout << "Raw pointer time: " << raw_duration.count() << " ms" << std::endl;
    std::cout << "unique_ptr time: " << unique_duration.count() << " ms" << std::endl;
    std::cout << "shared_ptr time: " << shared_duration.count() << " ms" << std::endl;
    
    std::cout << "unique_ptr overhead: " 
              << ((double)unique_duration.count() / raw_duration.count() - 1) * 100 
              << "%" << std::endl;
    std::cout << "shared_ptr overhead: " 
              << ((double)shared_duration.count() / raw_duration.count() - 1) * 100 
              << "%" << std::endl;
}
```

### 3. Exception Safety

```cpp
#include <iostream>
#include <memory>
#include <stdexcept>

class RiskyResource {
public:
    RiskyResource() {
        std::cout << "RiskyResource constructor" << std::endl;
        // Simulate potential exception
        static int count = 0;
        if (++count % 3 == 0) {
            throw std::runtime_error("Construction failed");
        }
    }
    
    ~RiskyResource() {
        std::cout << "RiskyResource destructor" << std::endl;
    }
};

void demonstrate_exception_safety() {
    std::cout << "\n=== Exception Safety ===" << std::endl;
    
    // BAD: Not exception safe
    try {
        // If the second new throws, first allocation leaks
        // process_resources(new RiskyResource(), new RiskyResource());
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }
    
    // GOOD: Exception safe with smart pointers
    try {
        auto ptr1 = std::make_unique<RiskyResource>();
        auto ptr2 = std::make_unique<RiskyResource>();
        
        // Use ptr1 and ptr2...
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        // Automatic cleanup - no leaks
    }
    
    // GOOD: Exception safe function parameters
    auto safe_create = []() -> std::unique_ptr<RiskyResource> {
        return std::make_unique<RiskyResource>();
    };
    
    try {
        auto resources = std::vector<std::unique_ptr<RiskyResource>>{};
        resources.push_back(safe_create());
        resources.push_back(safe_create());
        resources.push_back(safe_create());
        
    } catch (const std::exception& e) {
        std::cout << "Exception in vector creation: " << e.what() << std::endl;
        // All successfully created resources are automatically cleaned up
    }
}
```

## Common Pitfalls

### 1. Mixing Smart Pointers and Raw Pointers

```cpp
#include <iostream>
#include <memory>

void demonstrate_common_pitfalls() {
    std::cout << "\n=== Common Pitfalls ===" << std::endl;
    
    // PITFALL 1: Creating shared_ptr from raw pointer multiple times
    {
        Resource* raw = new Resource("DangerousRaw", 999);
        
        // BAD: Both shared_ptrs think they own the resource
        // std::shared_ptr<Resource> ptr1(raw);
        // std::shared_ptr<Resource> ptr2(raw);  // Double delete!
        
        // GOOD: Create shared_ptr once and share it
        std::shared_ptr<Resource> ptr1(raw);
        std::shared_ptr<Resource> ptr2 = ptr1;
        
        std::cout << "Reference count: " << ptr1.use_count() << std::endl;
    }
    
    // PITFALL 2: Storing raw pointer from get()
    {
        auto smart_ptr = std::make_unique<Resource>("SmartResource", 888);
        Resource* raw_ptr = smart_ptr.get();
        
        // BAD: Using raw pointer after smart pointer is destroyed
        smart_ptr.reset();
        // raw_ptr->use();  // Dangling pointer!
    }
    
    // PITFALL 3: Circular references with shared_ptr
    {
        struct Node {
            std::shared_ptr<Node> next;
            std::weak_ptr<Node> parent;  // Use weak_ptr to break cycle
            int value;
            
            Node(int v) : value(v) {
                std::cout << "Node " << value << " created" << std::endl;
            }
            
            ~Node() {
                std::cout << "Node " << value << " destroyed" << std::endl;
            }
        };
        
        auto node1 = std::make_shared<Node>(1);
        auto node2 = std::make_shared<Node>(2);
        
        node1->next = node2;
        node2->parent = node1;  // weak_ptr breaks the cycle
        
        std::cout << "Node1 ref count: " << node1.use_count() << std::endl;
        std::cout << "Node2 ref count: " << node2.use_count() << std::endl;
    }
}
```

## Exercises

### Exercise 1: Resource Manager
Implement a resource manager class that uses `std::unique_ptr` to manage dynamically allocated resources.

### Exercise 2: Shared Cache
Create a cache system using `std::shared_ptr` that allows multiple clients to share cached objects.

### Exercise 3: Observer Pattern
Implement the observer pattern using `std::weak_ptr` to avoid circular dependencies.

### Exercise 4: Custom Deleter
Write a custom deleter for managing a special resource type (e.g., database connection, file handle).

## Summary

Smart pointers are essential for modern C++ memory management:

- **`std::unique_ptr`**: Exclusive ownership, zero overhead, move-only semantics
- **`std::shared_ptr`**: Shared ownership through reference counting, thread-safe
- **`std::weak_ptr`**: Non-owning observer, breaks circular references

Key benefits:
- Automatic memory cleanup
- Exception safety
- Clear ownership semantics
- Prevention of common memory bugs (leaks, double deletes, dangling pointers)

Always prefer smart pointers over raw pointers for dynamic memory management, and choose the appropriate type based on ownership requirements.
