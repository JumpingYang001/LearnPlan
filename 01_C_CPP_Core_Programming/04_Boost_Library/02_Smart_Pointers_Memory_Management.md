# Smart Pointers and Memory Management

*Duration: 1 week*

## Overview

Smart pointers are one of the most critical components in modern C++ programming, providing automatic memory management and helping prevent common memory-related bugs like memory leaks, dangling pointers, and double-delete errors. Boost was instrumental in pioneering smart pointer concepts that eventually became part of the C++ standard library.

This section covers Boost's smart pointer implementations and memory management strategies, including detailed comparisons with modern C++ standard library equivalents. Understanding both Boost and standard smart pointers is essential for:

- **Legacy Code Maintenance**: Many existing codebases still use Boost smart pointers
- **Advanced Features**: Some Boost smart pointers offer features not available in std equivalents
- **Performance Optimization**: Understanding the trade-offs between different implementations
- **Memory Management Mastery**: Learning sophisticated memory management patterns

### Why Smart Pointers Matter

**Traditional C++ Memory Management Problems:**
```cpp
// Common memory management issues in traditional C++
void problematic_function() {
    Resource* resource = new Resource();
    
    if (some_condition) {
        return; // Memory leak! Forgot to delete resource
    }
    
    process_resource(resource);
    
    if (error_occurred) {
        throw std::exception(); // Memory leak! Exception bypasses delete
    }
    
    delete resource; // This might not be reached
}

// Solution with exceptions - still problematic
void better_but_still_problematic() {
    Resource* resource = nullptr;
    try {
        resource = new Resource();
        process_resource(resource);
        delete resource;
    } catch (...) {
        delete resource; // Code duplication, easy to forget
        throw;
    }
}
```

**Smart Pointer Solution:**
```cpp
void smart_pointer_solution() {
    boost::scoped_ptr<Resource> resource(new Resource());
    
    if (some_condition) {
        return; // Automatic cleanup - no memory leak!
    }
    
    process_resource(resource.get());
    
    if (error_occurred) {
        throw std::exception(); // Automatic cleanup during stack unwinding
    }
    
    // No explicit delete needed - automatic cleanup
}
```

## Learning Topics

### boost::shared_ptr (pre-C++11)

`boost::shared_ptr` was the precursor to `std::shared_ptr` and implements shared ownership through reference counting. Multiple `shared_ptr` instances can point to the same object, and the object is automatically deleted when the last `shared_ptr` is destroyed.

#### How Reference Counting Works

```cpp
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <iostream>

class RefCountDemo {
public:
    RefCountDemo(int id) : id_(id) {
        std::cout << "RefCountDemo " << id_ << " created\n";
    }
    
    ~RefCountDemo() {
        std::cout << "RefCountDemo " << id_ << " destroyed\n";
    }
    
    void doWork() const {
        std::cout << "RefCountDemo " << id_ << " working...\n";
    }
    
private:
    int id_;
};

void demonstrate_reference_counting() {
    std::cout << "=== Reference Counting Demonstration ===\n";
    
    boost::shared_ptr<RefCountDemo> ptr1;
    std::cout << "ptr1 count: " << ptr1.use_count() << " (empty)\n";
    
    ptr1 = boost::make_shared<RefCountDemo>(1);
    std::cout << "ptr1 count after creation: " << ptr1.use_count() << "\n";
    
    {
        boost::shared_ptr<RefCountDemo> ptr2 = ptr1; // Copy constructor
        std::cout << "ptr1 count after copy: " << ptr1.use_count() << "\n";
        std::cout << "ptr2 count: " << ptr2.use_count() << "\n";
        
        boost::shared_ptr<RefCountDemo> ptr3;
        ptr3 = ptr1; // Assignment operator
        std::cout << "ptr1 count after assignment: " << ptr1.use_count() << "\n";
        
        ptr2->doWork();
        
    } // ptr2 and ptr3 go out of scope
    
    std::cout << "ptr1 count after scope exit: " << ptr1.use_count() << "\n";
    ptr1.reset(); // Explicitly release
    std::cout << "After reset, object should be destroyed\n";
}
```

#### Thread Safety Considerations

**Reference Count Operations are Thread-Safe:**
```cpp
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <thread>
#include <vector>
#include <iostream>

class ThreadSafeResource {
public:
    ThreadSafeResource(int id) : id_(id), access_count_(0) {
        std::cout << "ThreadSafeResource " << id_ << " created\n";
    }
    
    ~ThreadSafeResource() {
        std::cout << "ThreadSafeResource " << id_ << " destroyed (accessed " 
                  << access_count_ << " times)\n";
    }
    
    void access() {
        // This is NOT thread-safe! Only the reference counting is thread-safe
        ++access_count_;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
private:
    int id_;
    int access_count_; // This needs protection in multi-threaded access
};

void worker_thread(boost::shared_ptr<ThreadSafeResource> resource, int thread_id) {
    for (int i = 0; i < 100; ++i) {
        // Safe: Copy shared_ptr (reference counting is thread-safe)
        boost::shared_ptr<ThreadSafeResource> local_copy = resource;
        
        // NOT safe: Accessing object contents (needs synchronization)
        local_copy->access();
    }
    std::cout << "Thread " << thread_id << " finished\n";
}

void demonstrate_thread_safety() {
    std::cout << "=== Thread Safety Demonstration ===\n";
    
    auto resource = boost::make_shared<ThreadSafeResource>(1);
    
    std::vector<std::thread> threads;
    
    // Create multiple threads sharing the same resource
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker_thread, resource, i);
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Final reference count: " << resource.use_count() << "\n";
    
    // Resource will be destroyed when the last shared_ptr goes out of scope
}
```

#### Custom Deleters

Custom deleters allow you to specify how an object should be cleaned up:

```cpp
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <fstream>
#include <cstdio>

// Custom deleter for C-style arrays
template<typename T>
struct ArrayDeleter {
    void operator()(T* ptr) const {
        std::cout << "ArrayDeleter: Deleting array\n";
        delete[] ptr;
    }
};

// Custom deleter for FILE*
struct FileDeleter {
    void operator()(FILE* file) const {
        if (file) {
            std::cout << "FileDeleter: Closing file\n";
            fclose(file);
        }
    }
};

// Function-based custom deleter
void custom_resource_deleter(int* resource) {
    std::cout << "Custom deleter: Cleaning up resource with value " << *resource << "\n";
    // Perform special cleanup logic here
    delete resource;
}

void demonstrate_custom_deleters() {
    std::cout << "=== Custom Deleters Demonstration ===\n";
    
    // Example 1: Array deleter
    {
        boost::shared_ptr<int> array_ptr(new int[10], ArrayDeleter<int>());
        // Will use delete[] instead of delete when destroyed
    }
    
    // Example 2: FILE* deleter
    {
        boost::shared_ptr<FILE> file_ptr(fopen("test.txt", "w"), FileDeleter());
        if (file_ptr) {
            fprintf(file_ptr.get(), "Hello, World!\n");
        }
        // File will be automatically closed
    }
    
    // Example 3: Function-based deleter
    {
        boost::shared_ptr<int> resource_ptr(new int(42), custom_resource_deleter);
        std::cout << "Resource value: " << *resource_ptr << "\n";
    }
    
    // Example 4: Lambda deleter (C++11 and later)
    {
        auto lambda_deleter = [](int* ptr) {
            std::cout << "Lambda deleter: Cleaning up value " << *ptr << "\n";
            delete ptr;
        };
        
        boost::shared_ptr<int> lambda_ptr(new int(123), lambda_deleter);
    }
}
```

#### Weak Pointer Interactions

Weak pointers prevent circular dependencies and allow safe observation of shared objects:

```cpp
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <iostream>
#include <vector>

class Node {
public:
    Node(int value) : value_(value) {
        std::cout << "Node " << value_ << " created\n";
    }
    
    ~Node() {
        std::cout << "Node " << value_ << " destroyed\n";
    }
    
    void addChild(boost::shared_ptr<Node> child) {
        children_.push_back(child);
        child->parent_ = boost::weak_ptr<Node>(shared_from_this());
    }
    
    void printTree(int depth = 0) const {
        for (int i = 0; i < depth; ++i) std::cout << "  ";
        std::cout << "Node " << value_ << "\n";
        
        for (const auto& child : children_) {
            child->printTree(depth + 1);
        }
    }
    
    boost::shared_ptr<Node> getParent() const {
        return parent_.lock(); // Convert weak_ptr to shared_ptr
    }
    
    int getValue() const { return value_; }
    
    // Enable shared_from_this functionality
    boost::shared_ptr<Node> shared_from_this() {
        return boost::shared_ptr<Node>(this);
    }
    
private:
    int value_;
    std::vector<boost::shared_ptr<Node>> children_;
    boost::weak_ptr<Node> parent_; // Weak reference to avoid cycles
};

void demonstrate_weak_ptr() {
    std::cout << "=== Weak Pointer Demonstration ===\n";
    
    auto root = boost::make_shared<Node>(1);
    auto child1 = boost::make_shared<Node>(2);
    auto child2 = boost::make_shared<Node>(3);
    
    root->addChild(child1);
    root->addChild(child2);
    
    std::cout << "Tree structure:\n";
    root->printTree();
    
    // Demonstrate weak_ptr usage
    if (auto parent = child1->getParent()) {
        std::cout << "Child1's parent value: " << parent->getValue() << "\n";
    }
    
    std::cout << "Reference counts:\n";
    std::cout << "root: " << root.use_count() << "\n";
    std::cout << "child1: " << child1.use_count() << "\n";
    
    // Clear child references
    child1.reset();
    child2.reset();
    
    std::cout << "After clearing child references:\n";
    std::cout << "root: " << root.use_count() << "\n";
    
    // Tree will be properly cleaned up without cycles
}
```

#### Common Pitfalls and Solutions

**Pitfall 1: Creating shared_ptr from raw pointer multiple times**
```cpp
void dangerous_double_shared_ptr() {
    Resource* raw_ptr = new Resource();
    
    // DANGER: Creating multiple shared_ptr from same raw pointer
    boost::shared_ptr<Resource> ptr1(raw_ptr);
    boost::shared_ptr<Resource> ptr2(raw_ptr); // Double delete will occur!
    
    // Solution: Use make_shared or pass existing shared_ptr
    auto safe_ptr1 = boost::make_shared<Resource>();
    auto safe_ptr2 = safe_ptr1; // Safe copy
}
```

**Pitfall 2: Circular references causing memory leaks**
```cpp
class Parent;
class Child;

class Parent {
public:
    ~Parent() { std::cout << "Parent destroyed\n"; }
    boost::shared_ptr<Child> child;
};

class Child {
public:
    ~Child() { std::cout << "Child destroyed\n"; }
    boost::shared_ptr<Parent> parent; // This creates a cycle!
};

void demonstrate_circular_reference_problem() {
    auto parent = boost::make_shared<Parent>();
    auto child = boost::make_shared<Child>();
    
    parent->child = child;
    child->parent = parent; // Circular reference - memory leak!
    
    // Neither destructor will be called!
}

// Solution: Use weak_ptr for back-references
class ChildFixed {
public:
    ~ChildFixed() { std::cout << "ChildFixed destroyed\n"; }
    boost::weak_ptr<Parent> parent; // Breaks the cycle
};
```

### boost::intrusive_ptr

`boost::intrusive_ptr` is a smart pointer that relies on intrusive reference counting, meaning the reference count is stored within the pointed-to object itself. This approach offers better performance and memory efficiency compared to `shared_ptr` in certain scenarios.

#### How Intrusive Reference Counting Works

Unlike `shared_ptr` which maintains a separate control block, `intrusive_ptr` requires the managed object to maintain its own reference count:

```cpp
#include <boost/intrusive_ptr.hpp>
#include <iostream>
#include <atomic>

// Basic intrusive reference counting implementation
class IntrusiveRefCountBase {
public:
    IntrusiveRefCountBase() : ref_count_(0) {
        std::cout << "IntrusiveRefCountBase created (ref_count = 0)\n";
    }
    
    virtual ~IntrusiveRefCountBase() {
        std::cout << "IntrusiveRefCountBase destroyed\n";
    }
    
    // Thread-safe reference counting
    void add_ref() const {
        ++ref_count_;
        std::cout << "Reference added (count = " << ref_count_ << ")\n";
    }
    
    void release() const {
        std::cout << "Reference released (count = " << ref_count_ - 1 << ")\n";
        if (--ref_count_ == 0) {
            delete this;
        }
    }
    
    long use_count() const {
        return ref_count_;
    }
    
private:
    mutable std::atomic<long> ref_count_;
};

// Required global functions for boost::intrusive_ptr
void intrusive_ptr_add_ref(const IntrusiveRefCountBase* p) {
    p->add_ref();
}

void intrusive_ptr_release(const IntrusiveRefCountBase* p) {
    p->release();
}

// Example class using intrusive reference counting
class IntrusiveResource : public IntrusiveRefCountBase {
public:
    IntrusiveResource(int id) : id_(id) {
        std::cout << "IntrusiveResource " << id_ << " created\n";
    }
    
    ~IntrusiveResource() override {
        std::cout << "IntrusiveResource " << id_ << " destroyed\n";
    }
    
    void doWork() const {
        std::cout << "IntrusiveResource " << id_ << " working (ref_count = " 
                  << use_count() << ")\n";
    }
    
    int getId() const { return id_; }
    
private:
    int id_;
};

void demonstrate_intrusive_ptr_basics() {
    std::cout << "=== Intrusive Pointer Basics ===\n";
    
    {
        // Create intrusive_ptr - initial reference count will be 1
        boost::intrusive_ptr<IntrusiveResource> ptr1(new IntrusiveResource(1));
        ptr1->doWork();
        
        {
            // Copy constructor - reference count becomes 2
            boost::intrusive_ptr<IntrusiveResource> ptr2 = ptr1;
            ptr2->doWork();
            
            // Assignment - reference count becomes 3
            boost::intrusive_ptr<IntrusiveResource> ptr3;
            ptr3 = ptr1;
            ptr3->doWork();
            
        } // ptr2 and ptr3 go out of scope - reference count becomes 1
        
        ptr1->doWork();
        
    } // ptr1 goes out of scope - reference count becomes 0, object destroyed
}
```

#### Performance Benefits

**Memory Overhead Comparison:**
```cpp
#include <boost/shared_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/make_shared.hpp>
#include <chrono>
#include <iostream>

class SharedPtrObject {
public:
    SharedPtrObject(int value) : value_(value) {}
    int getValue() const { return value_; }
private:
    int value_;
};

class IntrusivePtrObject : public IntrusiveRefCountBase {
public:
    IntrusivePtrObject(int value) : value_(value) {}
    int getValue() const { return value_; }
private:
    int value_;
};

void performance_comparison() {
    std::cout << "=== Performance Comparison ===\n";
    
    const int NUM_OBJECTS = 1000000;
    
    // Measure shared_ptr performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        auto shared_obj = boost::make_shared<SharedPtrObject>(i);
        auto shared_copy = shared_obj;
        // shared_ptr maintains separate control block with reference count
    }
    
    auto shared_time = std::chrono::high_resolution_clock::now() - start;
    
    // Measure intrusive_ptr performance
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        boost::intrusive_ptr<IntrusivePtrObject> intrusive_obj(new IntrusivePtrObject(i));
        auto intrusive_copy = intrusive_obj;
        // intrusive_ptr uses object's own reference count
    }
    
    auto intrusive_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "shared_ptr time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(shared_time).count() 
              << " ms\n";
    std::cout << "intrusive_ptr time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(intrusive_time).count() 
              << " ms\n";
    
    // Memory usage comparison
    std::cout << "\nMemory overhead per object:\n";
    std::cout << "SharedPtrObject: " << sizeof(SharedPtrObject) << " bytes\n";
    std::cout << "IntrusivePtrObject: " << sizeof(IntrusivePtrObject) << " bytes\n";
    std::cout << "shared_ptr control block overhead: ~24 bytes (approximate)\n";
    std::cout << "intrusive_ptr overhead: 0 bytes (reference count in object)\n";
}
```

#### Custom Reference Counting Implementations

You can implement custom reference counting strategies:

```cpp
#include <boost/intrusive_ptr.hpp>
#include <atomic>
#include <iostream>
#include <mutex>

// Thread-safe reference counting with custom behavior
class CustomRefCount {
public:
    CustomRefCount() : ref_count_(0), destruction_callback_(nullptr) {
        std::cout << "CustomRefCount created\n";
    }
    
    virtual ~CustomRefCount() {
        std::cout << "CustomRefCount destroyed\n";
        if (destruction_callback_) {
            destruction_callback_(this);
        }
    }
    
    void add_ref() const {
        long old_count = ref_count_.fetch_add(1, std::memory_order_relaxed);
        std::cout << "Reference added: " << old_count << " -> " << (old_count + 1) << "\n";
    }
    
    void release() const {
        long old_count = ref_count_.fetch_sub(1, std::memory_order_acq_rel);
        std::cout << "Reference released: " << old_count << " -> " << (old_count - 1) << "\n";
        
        if (old_count == 1) {
            delete this;
        }
    }
    
    long use_count() const {
        return ref_count_.load(std::memory_order_acquire);
    }
    
    // Custom behavior: Set destruction callback
    void setDestructionCallback(std::function<void(CustomRefCount*)> callback) {
        destruction_callback_ = callback;
    }
    
private:
    mutable std::atomic<long> ref_count_;
    std::function<void(CustomRefCount*)> destruction_callback_;
};

void intrusive_ptr_add_ref(const CustomRefCount* p) {
    p->add_ref();
}

void intrusive_ptr_release(const CustomRefCount* p) {
    p->release();
}

class CustomResource : public CustomRefCount {
public:
    CustomResource(const std::string& name) : name_(name) {
        std::cout << "CustomResource '" << name_ << "' created\n";
        
        // Set up destruction callback
        setDestructionCallback([](CustomRefCount* obj) {
            CustomResource* resource = static_cast<CustomResource*>(obj);
            std::cout << "Destruction callback: Resource '" << resource->name_ 
                      << "' is being destroyed\n";
        });
    }
    
    ~CustomResource() override {
        std::cout << "CustomResource '" << name_ << "' destroyed\n";
    }
    
    void use() const {
        std::cout << "Using resource '" << name_ << "' (ref_count = " 
                  << use_count() << ")\n";
    }
    
private:
    std::string name_;
};

void demonstrate_custom_refcount() {
    std::cout << "=== Custom Reference Counting ===\n";
    
    boost::intrusive_ptr<CustomResource> ptr(new CustomResource("TestResource"));
    ptr->use();
    
    {
        auto ptr_copy = ptr;
        ptr_copy->use();
    }
    
    ptr->use();
}
```

#### Use Cases and Trade-offs

**When to Use intrusive_ptr:**

✅ **Advantages:**
1. **Better Performance**: No separate control block allocation
2. **Lower Memory Overhead**: Reference count is part of the object
3. **Cache Efficiency**: Better cache locality (object + ref count together)
4. **Existing Codebase**: When you already have reference-counted objects

❌ **Disadvantages:**
1. **Intrusive**: Requires modifying the target class
2. **Virtual Destructor**: Often requires virtual destructor (adds vtable overhead)
3. **Less Flexible**: Cannot use custom deleters as easily
4. **Legacy**: Not part of C++ standard library

**Practical Use Case Example:**
```cpp
// Example: Game engine entity system with intrusive reference counting
class GameObject : public IntrusiveRefCountBase {
public:
    GameObject(int id) : id_(id), position_{0, 0, 0} {
        std::cout << "GameObject " << id_ << " created\n";
    }
    
    ~GameObject() override {
        std::cout << "GameObject " << id_ << " destroyed\n";
    }
    
    void setPosition(float x, float y, float z) {
        position_[0] = x;
        position_[1] = y;
        position_[2] = z;
    }
    
    void render() const {
        std::cout << "Rendering GameObject " << id_ 
                  << " at (" << position_[0] << ", " << position_[1] << ", " << position_[2] << ")\n";
    }
    
    int getId() const { return id_; }
    
private:
    int id_;
    float position_[3];
};

using GameObjectPtr = boost::intrusive_ptr<GameObject>;

class Scene {
public:
    void addObject(GameObjectPtr obj) {
        objects_.push_back(obj);
    }
    
    void removeObject(int id) {
        objects_.erase(
            std::remove_if(objects_.begin(), objects_.end(),
                [id](const GameObjectPtr& obj) { return obj->getId() == id; }),
            objects_.end());
    }
    
    void renderAll() const {
        for (const auto& obj : objects_) {
            obj->render();
        }
    }
    
private:
    std::vector<GameObjectPtr> objects_;
};

void demonstrate_game_engine_usage() {
    std::cout << "=== Game Engine Usage Example ===\n";
    
    Scene scene;
    
    // Create game objects
    auto player = GameObjectPtr(new GameObject(1));
    auto enemy1 = GameObjectPtr(new GameObject(2));
    auto enemy2 = GameObjectPtr(new GameObject(3));
    
    // Position objects
    player->setPosition(0, 0, 0);
    enemy1->setPosition(10, 0, 5);
    enemy2->setPosition(-5, 0, 8);
    
    // Add to scene
    scene.addObject(player);
    scene.addObject(enemy1);
    scene.addObject(enemy2);
    
    // Render scene
    scene.renderAll();
    
    // Remove an enemy
    scene.removeObject(2);
    
    std::cout << "After removing enemy 2:\n";
    scene.renderAll();
    
    // Objects will be automatically cleaned up when no longer referenced
}
```

### boost::scoped_ptr

`boost::scoped_ptr` is a simple smart pointer that provides exclusive ownership of a dynamically allocated object. It's the Boost equivalent of `std::unique_ptr` but with fewer features and stricter semantics. The key characteristic is that it cannot be copied or transferred - it's "scoped" to its declaration block.

#### RAII Principles and Basic Usage

RAII (Resource Acquisition Is Initialization) is the fundamental principle behind `scoped_ptr`:

```cpp
#include <boost/scoped_ptr.hpp>
#include <iostream>
#include <stdexcept>

class ScopedResource {
public:
    ScopedResource(const std::string& name) : name_(name) {
        std::cout << "ScopedResource '" << name_ << "' created\n";
        // Simulate resource acquisition that might fail
        if (name_.empty()) {
            throw std::invalid_argument("Name cannot be empty");
        }
    }
    
    ~ScopedResource() {
        std::cout << "ScopedResource '" << name_ << "' destroyed\n";
        // Automatic cleanup happens here
    }
    
    void doWork() const {
        std::cout << "ScopedResource '" << name_ << "' working...\n";
    }
    
    const std::string& getName() const { return name_; }
    
private:
    std::string name_;
};

void demonstrate_scoped_ptr_basics() {
    std::cout << "=== Scoped Pointer Basics ===\n";
    
    {
        // Basic usage - automatic cleanup
        boost::scoped_ptr<ScopedResource> ptr(new ScopedResource("Resource1"));
        ptr->doWork();
        
        // Access patterns
        if (ptr) {
            std::cout << "Resource name: " << ptr->getName() << "\n";
        }
        
        // Get raw pointer (use with caution)
        ScopedResource* raw_ptr = ptr.get();
        if (raw_ptr) {
            raw_ptr->doWork();
        }
        
    } // Automatic cleanup here - destructor called
    
    std::cout << "After scope exit - resource cleaned up\n";
}

void demonstrate_exception_safety() {
    std::cout << "=== Exception Safety with scoped_ptr ===\n";
    
    try {
        boost::scoped_ptr<ScopedResource> ptr1(new ScopedResource("SafeResource"));
        
        // Even if an exception occurs, ptr1 will be cleaned up
        boost::scoped_ptr<ScopedResource> ptr2(new ScopedResource(""));  // This will throw
        
        // This line won't be reached
        ptr1->doWork();
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << "\n";
        std::cout << "Resources automatically cleaned up during stack unwinding\n";
    }
}

void demonstrate_reset_and_swap() {
    std::cout << "=== Reset and Swap Operations ===\n";
    
    boost::scoped_ptr<ScopedResource> ptr(new ScopedResource("Original"));
    ptr->doWork();
    
    // Reset with new resource
    ptr.reset(new ScopedResource("Replacement"));
    ptr->doWork();
    
    // Reset to empty (delete current resource)
    ptr.reset();
    if (!ptr) {
        std::cout << "Pointer is now empty\n";
    }
    
    // Swap operation
    boost::scoped_ptr<ScopedResource> ptr1(new ScopedResource("First"));
    boost::scoped_ptr<ScopedResource> ptr2(new ScopedResource("Second"));
    
    std::cout << "Before swap:\n";
    std::cout << "  ptr1: " << (ptr1 ? ptr1->getName() : "null") << "\n";
    std::cout << "  ptr2: " << (ptr2 ? ptr2->getName() : "null") << "\n";
    
    ptr1.swap(ptr2);
    
    std::cout << "After swap:\n";
    std::cout << "  ptr1: " << (ptr1 ? ptr1->getName() : "null") << "\n";
    std::cout << "  ptr2: " << (ptr2 ? ptr2->getName() : "null") << "\n";
}
```

#### Comparison with std::unique_ptr

Understanding the differences helps with migration and choosing the right tool:

```cpp
#include <boost/scoped_ptr.hpp>
#include <memory>  // For std::unique_ptr
#include <iostream>
#include <vector>

class MoveableResource {
public:
    MoveableResource(int id) : id_(id) {
        std::cout << "MoveableResource " << id_ << " created\n";
    }
    
    ~MoveableResource() {
        std::cout << "MoveableResource " << id_ << " destroyed\n";
    }
    
    // Move constructor for std::unique_ptr compatibility
    MoveableResource(MoveableResource&& other) noexcept : id_(other.id_) {
        other.id_ = -1;  // Mark as moved
        std::cout << "MoveableResource " << id_ << " moved\n";
    }
    
    void use() const {
        std::cout << "Using MoveableResource " << id_ << "\n";
    }
    
    int getId() const { return id_; }
    
private:
    int id_;
};

// Comparison of capabilities
void compare_scoped_ptr_vs_unique_ptr() {
    std::cout << "=== scoped_ptr vs unique_ptr Comparison ===\n";
    
    // 1. Basic usage - similar
    {
        boost::scoped_ptr<MoveableResource> scoped(new MoveableResource(1));
        std::unique_ptr<MoveableResource> unique(new MoveableResource(2));
        
        scoped->use();
        unique->use();
    }
    
    // 2. Move semantics - unique_ptr supports, scoped_ptr doesn't
    {
        std::unique_ptr<MoveableResource> unique1(new MoveableResource(3));
        std::unique_ptr<MoveableResource> unique2 = std::move(unique1);  // OK
        
        boost::scoped_ptr<MoveableResource> scoped1(new MoveableResource(4));
        // boost::scoped_ptr<MoveableResource> scoped2 = std::move(scoped1);  // ERROR!
        
        if (unique2) unique2->use();
        if (scoped1) scoped1->use();
    }
    
    // 3. Container storage - unique_ptr works, scoped_ptr doesn't
    {
        std::vector<std::unique_ptr<MoveableResource>> unique_vector;
        unique_vector.push_back(std::make_unique<MoveableResource>(5));  // OK
        
        // std::vector<boost::scoped_ptr<MoveableResource>> scoped_vector;  // ERROR!
        // scoped_ptr cannot be stored in containers because it's not moveable
    }
    
    // 4. Function parameters - different patterns
    auto process_unique = [](std::unique_ptr<MoveableResource> resource) {
        if (resource) {
            resource->use();
        }
    };
    
    auto process_scoped = [](const boost::scoped_ptr<MoveableResource>& resource) {
        if (resource) {
            resource->use();
        }
    };
    
    std::unique_ptr<MoveableResource> unique_param(new MoveableResource(6));
    boost::scoped_ptr<MoveableResource> scoped_param(new MoveableResource(7));
    
    process_unique(std::move(unique_param));  // Transfer ownership
    process_scoped(scoped_param);             // Pass by reference (no ownership transfer)
}

// Factory function patterns
std::unique_ptr<MoveableResource> createUniqueResource(int id) {
    return std::make_unique<MoveableResource>(id);  // Can return by value
}

boost::scoped_ptr<MoveableResource> createScopedResource(int id) {
    // return boost::scoped_ptr<MoveableResource>(new MoveableResource(id));  // ERROR!
    // Cannot return scoped_ptr by value
    return boost::scoped_ptr<MoveableResource>();  // Must return empty
}

void demonstrate_factory_patterns() {
    std::cout << "=== Factory Pattern Differences ===\n";
    
    // unique_ptr allows factory patterns
    auto unique_resource = createUniqueResource(8);
    if (unique_resource) {
        unique_resource->use();
    }
    
    // scoped_ptr requires different patterns
    boost::scoped_ptr<MoveableResource> scoped_resource(new MoveableResource(9));
    if (scoped_resource) {
        scoped_resource->use();
    }
}
```

#### Legacy Code Considerations

When working with legacy codebases that use `scoped_ptr`:

```cpp
#include <boost/scoped_ptr.hpp>
#include <iostream>
#include <map>
#include <string>

// Legacy pattern: Using scoped_ptr in class members
class LegacyClass {
public:
    LegacyClass(const std::string& name) : name_(name) {
        // Initialize expensive resource lazily
        resource_.reset(new ScopedResource(name + "_resource"));
    }
    
    // Rule of Three/Five - scoped_ptr makes copying tricky
    LegacyClass(const LegacyClass& other) : name_(other.name_) {
        // Deep copy required
        if (other.resource_) {
            resource_.reset(new ScopedResource(other.resource_->getName()));
        }
    }
    
    LegacyClass& operator=(const LegacyClass& other) {
        if (this != &other) {
            name_ = other.name_;
            if (other.resource_) {
                resource_.reset(new ScopedResource(other.resource_->getName()));
            } else {
                resource_.reset();
            }
        }
        return *this;
    }
    
    ~LegacyClass() = default;  // scoped_ptr handles cleanup
    
    void useResource() {
        if (resource_) {
            resource_->doWork();
        }
    }
    
    const std::string& getName() const { return name_; }
    
private:
    std::string name_;
    boost::scoped_ptr<ScopedResource> resource_;
};

// Migration strategy: Wrapper for legacy code
template<typename T>
class MigratingPtr {
public:
    // Can be constructed from either type
    MigratingPtr(boost::scoped_ptr<T> ptr) : modern_ptr_(ptr.release()) {}
    MigratingPtr(std::unique_ptr<T> ptr) : modern_ptr_(std::move(ptr)) {}
    
    T* get() const { return modern_ptr_.get(); }
    T* operator->() const { return modern_ptr_.get(); }
    T& operator*() const { return *modern_ptr_; }
    
    explicit operator bool() const { return static_cast<bool>(modern_ptr_); }
    
    void reset(T* ptr = nullptr) { modern_ptr_.reset(ptr); }
    
private:
    std::unique_ptr<T> modern_ptr_;  // Internally use modern smart pointer
};

void demonstrate_legacy_migration() {
    std::cout << "=== Legacy Code Migration ===\n";
    
    // Legacy usage
    LegacyClass legacy_obj("LegacyObject");
    legacy_obj.useResource();
    
    // Copy behavior
    LegacyClass copied_obj = legacy_obj;
    copied_obj.useResource();
    
    // Migration wrapper usage
    boost::scoped_ptr<ScopedResource> legacy_ptr(new ScopedResource("LegacyPtr"));
    MigratingPtr<ScopedResource> migrating_ptr(std::move(legacy_ptr));
    
    if (migrating_ptr) {
        migrating_ptr->doWork();
    }
}

// Best practices for scoped_ptr usage
class BestPracticesExample {
public:
    explicit BestPracticesExample(const std::string& config) {
        // Initialize in constructor body, not initializer list for exception safety
        try {
            resource_.reset(new ScopedResource(config));
        } catch (...) {
            // Handle initialization failure
            throw;
        }
    }
    
    // Provide check methods
    bool hasResource() const {
        return static_cast<bool>(resource_);
    }
    
    // Safe access methods
    void useResourceSafely() {
        if (resource_) {
            resource_->doWork();
        } else {
            std::cout << "No resource available\n";
        }
    }
    
    // Explicit resource management
    void releaseResource() {
        resource_.reset();
        std::cout << "Resource explicitly released\n";
    }
    
    void replaceResource(const std::string& new_config) {
        boost::scoped_ptr<ScopedResource> new_resource(new ScopedResource(new_config));
        // Only assign if construction succeeds
        resource_.swap(new_resource);
    }
    
private:
    boost::scoped_ptr<ScopedResource> resource_;
};

void demonstrate_best_practices() {
    std::cout << "=== Best Practices ===\n";
    
    BestPracticesExample example("InitialConfig");
    
    if (example.hasResource()) {
        example.useResourceSafely();
    }
    
    example.replaceResource("NewConfig");
    example.useResourceSafely();
    
    example.releaseResource();
    example.useResourceSafely();  // Safe - checks for null
}
```

#### When to Use scoped_ptr vs Modern Alternatives

**Use scoped_ptr when:**
- Working with legacy Boost-based codebases
- You need simple, non-transferable ownership
- You want to prevent accidental copying of expensive resources
- Maintaining code that predates C++11

**Prefer std::unique_ptr when:**
- Starting new projects
- You need move semantics
- You want to store smart pointers in containers
- You need custom deleters with type erasure
- You're using modern C++ (C++11 and later)

**Migration Strategy:**
```cpp
// Step 1: Identify scoped_ptr usage
// boost::scoped_ptr<Resource> ptr;

// Step 2: Replace with unique_ptr
// std::unique_ptr<Resource> ptr;

// Step 3: Update move semantics if needed
// return std::move(ptr);  // Now possible

// Step 4: Leverage modern features
// auto ptr = std::make_unique<Resource>(args);
```

### Object Pools and Memory Management Strategies

Object pools are a memory management technique that pre-allocates a fixed number of objects and reuses them instead of constantly allocating and deallocating memory. This approach is particularly beneficial for frequently created/destroyed objects and can significantly improve performance by reducing memory allocation overhead and fragmentation.

#### boost::pool Library Overview

The Boost.Pool library provides several pool implementations for different use cases:

```cpp
#include <boost/pool/pool.hpp>
#include <boost/pool/object_pool.hpp>
#include <boost/pool/pool_alloc.hpp>
#include <boost/pool/singleton_pool.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>

// Demonstration class for pool usage
class PooledObject {
public:
    PooledObject(int id = 0) : id_(id), data_(new int[100]) {
        std::fill(data_, data_ + 100, id_);
        // std::cout << "PooledObject " << id_ << " constructed\n";
    }
    
    ~PooledObject() {
        delete[] data_;
        // std::cout << "PooledObject " << id_ << " destructed\n";
    }
    
    void process() {
        // Simulate some work
        for (int i = 0; i < 100; ++i) {
            data_[i] = data_[i] * 2 + 1;
        }
    }
    
    int getId() const { return id_; }
    
private:
    int id_;
    int* data_;  // Some resource that makes construction/destruction expensive
};

void demonstrate_basic_pool() {
    std::cout << "=== Basic Pool Usage ===\n";
    
    // Simple memory pool for raw memory allocation
    boost::pool<> memory_pool(sizeof(int));
    
    // Allocate integers from pool
    std::vector<int*> allocated_ints;
    
    for (int i = 0; i < 10; ++i) {
        int* ptr = static_cast<int*>(memory_pool.malloc());
        if (ptr) {
            *ptr = i * i;
            allocated_ints.push_back(ptr);
            std::cout << "Allocated int with value: " << *ptr << "\n";
        }
    }
    
    // Use the allocated memory
    for (int* ptr : allocated_ints) {
        std::cout << "Value: " << *ptr << "\n";
    }
    
    // Return memory to pool (not to system heap)
    for (int* ptr : allocated_ints) {
        memory_pool.free(ptr);
    }
    
    std::cout << "Memory returned to pool\n";
    // Pool destructor will release all memory back to system
}

void demonstrate_object_pool() {
    std::cout << "=== Object Pool Usage ===\n";
    
    boost::object_pool<PooledObject> object_pool;
    
    // Allocate objects from pool
    std::vector<PooledObject*> objects;
    
    for (int i = 0; i < 5; ++i) {
        // construct() calls constructor and returns pointer
        PooledObject* obj = object_pool.construct(i + 1);
        objects.push_back(obj);
        std::cout << "Created object with ID: " << obj->getId() << "\n";
    }
    
    // Use objects
    for (PooledObject* obj : objects) {
        obj->process();
        std::cout << "Processed object " << obj->getId() << "\n";
    }
    
    // Return objects to pool
    for (PooledObject* obj : objects) {
        object_pool.destroy(obj);  // Calls destructor and returns memory to pool
    }
    
    std::cout << "Objects returned to pool\n";
    
    // Demonstrate reuse
    std::cout << "Creating new objects (should reuse memory):\n";
    for (int i = 0; i < 3; ++i) {
        PooledObject* obj = object_pool.construct(100 + i);
        std::cout << "Reused memory for object ID: " << obj->getId() << "\n";
        object_pool.destroy(obj);
    }
}
```

#### Object Pool Patterns and Custom Implementations

```cpp
#include <boost/pool/object_pool.hpp>
#include <queue>
#include <mutex>
#include <iostream>

// Thread-safe object pool wrapper
template<typename T>
class ThreadSafeObjectPool {
public:
    explicit ThreadSafeObjectPool(size_t initial_size = 10) {
        // Pre-allocate objects
        for (size_t i = 0; i < initial_size; ++i) {
            available_objects_.push(pool_.construct());
        }
    }
    
    ~ThreadSafeObjectPool() {
        // Clean up all available objects
        std::lock_guard<std::mutex> lock(mutex_);
        while (!available_objects_.empty()) {
            pool_.destroy(available_objects_.front());
            available_objects_.pop();
        }
    }
    
    // Get object from pool
    T* acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (available_objects_.empty()) {
            // Pool exhausted, create new object
            return pool_.construct();
        }
        
        T* obj = available_objects_.front();
        available_objects_.pop();
        return obj;
    }
    
    // Return object to pool
    void release(T* obj) {
        if (!obj) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        available_objects_.push(obj);
    }
    
    size_t available_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return available_objects_.size();
    }
    
private:
    boost::object_pool<T> pool_;
    std::queue<T*> available_objects_;
    mutable std::mutex mutex_;
};

// RAII wrapper for automatic pool management
template<typename T>
class PooledObjectWrapper {
public:
    PooledObjectWrapper(ThreadSafeObjectPool<T>& pool) 
        : pool_(pool), object_(pool.acquire()) {}
    
    ~PooledObjectWrapper() {
        if (object_) {
            pool_.release(object_);
        }
    }
    
    // Non-copyable but moveable
    PooledObjectWrapper(const PooledObjectWrapper&) = delete;
    PooledObjectWrapper& operator=(const PooledObjectWrapper&) = delete;
    
    PooledObjectWrapper(PooledObjectWrapper&& other) noexcept
        : pool_(other.pool_), object_(other.object_) {
        other.object_ = nullptr;
    }
    
    PooledObjectWrapper& operator=(PooledObjectWrapper&& other) noexcept {
        if (this != &other) {
            if (object_) {
                pool_.release(object_);
            }
            object_ = other.object_;
            other.object_ = nullptr;
        }
        return *this;
    }
    
    T* get() const { return object_; }
    T* operator->() const { return object_; }
    T& operator*() const { return *object_; }
    
    explicit operator bool() const { return object_ != nullptr; }
    
private:
    ThreadSafeObjectPool<T>& pool_;
    T* object_;
};

void demonstrate_advanced_pool_patterns() {
    std::cout << "=== Advanced Pool Patterns ===\n";
    
    ThreadSafeObjectPool<PooledObject> pool(3);  // Pre-allocate 3 objects
    
    std::cout << "Initial available objects: " << pool.available_count() << "\n";
    
    {
        // Use RAII wrapper for automatic cleanup
        PooledObjectWrapper<PooledObject> wrapper1(pool);
        PooledObjectWrapper<PooledObject> wrapper2(pool);
        PooledObjectWrapper<PooledObject> wrapper3(pool);
        
        std::cout << "After acquiring 3 objects: " << pool.available_count() << "\n";
        
        if (wrapper1) {
            wrapper1->process();
            std::cout << "Processed object " << wrapper1->getId() << "\n";
        }
        
        // Test pool exhaustion
        PooledObjectWrapper<PooledObject> wrapper4(pool);  // Should create new object
        std::cout << "After pool exhaustion: " << pool.available_count() << "\n";
        
    } // All wrappers destroyed here, objects returned to pool
    
    std::cout << "After wrappers destroyed: " << pool.available_count() << "\n";
}
```

#### Memory Pool Allocators

Boost provides allocators that use memory pools for STL containers:

```cpp
#include <boost/pool/pool_alloc.hpp>
#include <vector>
#include <list>
#include <set>
#include <chrono>
#include <iostream>

void demonstrate_pool_allocators() {
    std::cout << "=== Pool Allocators ===\n";
    
    // Using pool allocator with vector
    std::vector<int, boost::pool_allocator<int>> pooled_vector;
    std::vector<int> normal_vector;
    
    const int NUM_ELEMENTS = 100000;
    
    // Measure performance with pool allocator
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        pooled_vector.push_back(i);
    }
    pooled_vector.clear();
    
    auto pool_time = std::chrono::high_resolution_clock::now() - start;
    
    // Measure performance with standard allocator
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        normal_vector.push_back(i);
    }
    normal_vector.clear();
    
    auto normal_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "Pool allocator time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(pool_time).count() 
              << " microseconds\n";
    std::cout << "Normal allocator time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(normal_time).count() 
              << " microseconds\n";
    
    // Pool allocators work well with containers that do frequent allocation/deallocation
    std::list<int, boost::pool_allocator<int>> pooled_list;
    
    for (int i = 0; i < 1000; ++i) {
        pooled_list.push_back(i);
        if (i % 3 == 0) {
            pooled_list.pop_front();  // Frequent deallocation
        }
    }
    
    std::cout << "Pool allocator with list: " << pooled_list.size() << " elements\n";
}

// Singleton pool for global memory management
void demonstrate_singleton_pool() {
    std::cout << "=== Singleton Pool ===\n";
    
    // Singleton pool provides global access to memory pool
    typedef boost::singleton_pool<struct MyTag, sizeof(int)> IntPool;
    
    std::vector<int*> allocated_ints;
    
    // Allocate from singleton pool
    for (int i = 0; i < 10; ++i) {
        int* ptr = static_cast<int*>(IntPool::malloc());
        if (ptr) {
            *ptr = i * 10;
            allocated_ints.push_back(ptr);
        }
    }
    
    // Use allocated memory
    for (int* ptr : allocated_ints) {
        std::cout << "Singleton pool value: " << *ptr << "\n";
    }
    
    // Return to pool
    for (int* ptr : allocated_ints) {
        IntPool::free(ptr);
    }
    
    // Singleton pool persists until program termination
    // or explicit release_memory() call
    std::cout << "Singleton pool manages " << IntPool::get_requested_size() << " byte objects\n";
}
```

#### Performance Optimization Techniques

```cpp
#include <boost/pool/object_pool.hpp>
#include <chrono>
#include <iostream>
#include <memory>

// Performance comparison class
class PerformanceTest {
public:
    static void compare_allocation_strategies() {
        std::cout << "=== Performance Comparison ===\n";
        
        const int NUM_ITERATIONS = 100000;
        const int OBJECTS_PER_ITERATION = 10;
        
        // Test 1: Standard new/delete
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            std::vector<PooledObject*> objects;
            
            for (int j = 0; j < OBJECTS_PER_ITERATION; ++j) {
                objects.push_back(new PooledObject(j));
            }
            
            for (PooledObject* obj : objects) {
                obj->process();
                delete obj;
            }
        }
        
        auto standard_time = std::chrono::high_resolution_clock::now() - start;
        
        // Test 2: Object pool
        start = std::chrono::high_resolution_clock::now();
        boost::object_pool<PooledObject> pool;
        
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            std::vector<PooledObject*> objects;
            
            for (int j = 0; j < OBJECTS_PER_ITERATION; ++j) {
                objects.push_back(pool.construct(j));
            }
            
            for (PooledObject* obj : objects) {
                obj->process();
                pool.destroy(obj);
            }
        }
        
        auto pool_time = std::chrono::high_resolution_clock::now() - start;
        
        // Test 3: Smart pointers with standard allocation
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            std::vector<std::unique_ptr<PooledObject>> objects;
            
            for (int j = 0; j < OBJECTS_PER_ITERATION; ++j) {
                objects.push_back(std::make_unique<PooledObject>(j));
            }
            
            for (auto& obj : objects) {
                obj->process();
            }
        }
        
        auto smart_ptr_time = std::chrono::high_resolution_clock::now() - start;
        
        // Results
        auto standard_ms = std::chrono::duration_cast<std::chrono::milliseconds>(standard_time).count();
        auto pool_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pool_time).count();
        auto smart_ptr_ms = std::chrono::duration_cast<std::chrono::milliseconds>(smart_ptr_time).count();
        
        std::cout << "Standard new/delete: " << standard_ms << " ms\n";
        std::cout << "Object pool: " << pool_ms << " ms\n";
        std::cout << "Smart pointers: " << smart_ptr_ms << " ms\n";
        
        double pool_improvement = (double)(standard_ms - pool_ms) / standard_ms * 100;
        std::cout << "Pool improvement: " << pool_improvement << "%\n";
    }
    
    static void analyze_memory_fragmentation() {
        std::cout << "=== Memory Fragmentation Analysis ===\n";
        
        // Simulate fragmentation with standard allocation
        std::vector<void*> allocations;
        
        // Allocate various sizes to cause fragmentation
        for (int i = 0; i < 1000; ++i) {
            size_t size = (i % 10 + 1) * 64;  // Varying sizes
            allocations.push_back(malloc(size));
        }
        
        // Free every other allocation (creates holes)
        for (size_t i = 0; i < allocations.size(); i += 2) {
            free(allocations[i]);
            allocations[i] = nullptr;
        }
        
        // Clean up remaining allocations
        for (void* ptr : allocations) {
            if (ptr) free(ptr);
        }
        
        std::cout << "Standard allocation can cause fragmentation\n";
        std::cout << "Pool allocation reduces fragmentation by using fixed-size blocks\n";
    }
};

void demonstrate_performance_optimization() {
    PerformanceTest::compare_allocation_strategies();
    PerformanceTest::analyze_memory_fragmentation();
}
```

#### Best Practices for Object Pools

```cpp
// Best practices implementation
class OptimalPoolUsage {
public:
    // 1. Size pools appropriately
    static void demonstrate_pool_sizing() {
        std::cout << "=== Pool Sizing Best Practices ===\n";
        
        // Too small: frequent expansion overhead
        // Too large: wasted memory
        // Rule of thumb: 2-4x expected concurrent usage
        
        boost::object_pool<PooledObject> small_pool;  // Default size
        
        // Monitor pool usage
        size_t max_concurrent = 0;
        size_t current_in_use = 0;
        
        std::vector<PooledObject*> active_objects;
        
        for (int iteration = 0; iteration < 5; ++iteration) {
            // Simulate varying load
            int objects_needed = (iteration + 1) * 3;
            
            for (int i = 0; i < objects_needed; ++i) {
                active_objects.push_back(small_pool.construct(i));
                current_in_use++;
                max_concurrent = std::max(max_concurrent, current_in_use);
            }
            
            // Process objects
            for (PooledObject* obj : active_objects) {
                obj->process();
            }
            
            // Return some objects
            size_t to_return = active_objects.size() / 2;
            for (size_t i = 0; i < to_return; ++i) {
                small_pool.destroy(active_objects.back());
                active_objects.pop_back();
                current_in_use--;
            }
        }
        
        // Clean up remaining objects
        for (PooledObject* obj : active_objects) {
            small_pool.destroy(obj);
        }
        
        std::cout << "Maximum concurrent objects: " << max_concurrent << "\n";
        std::cout << "Recommended pool size: " << max_concurrent * 2 << " to " << max_concurrent * 4 << "\n";
    }
    
    // 2. Pool object lifecycle management
    static void demonstrate_lifecycle_management() {
        std::cout << "=== Object Lifecycle Management ===\n";
        
        boost::object_pool<PooledObject> pool;
        
        // Best practice: Reset object state when returned to pool
        class ResettablePooledObject : public PooledObject {
        public:
            ResettablePooledObject(int id = 0) : PooledObject(id), is_dirty_(false) {}
            
            void reset() {
                is_dirty_ = false;
                // Reset any stateful members
            }
            
            void makeDirty() { is_dirty_ = true; }
            bool isDirty() const { return is_dirty_; }
            
        private:
            bool is_dirty_;
        };
        
        // Usage pattern with reset
        std::cout << "Using reset pattern for clean object reuse\n";
    }
    
    // 3. Exception safety with pools
    static void demonstrate_exception_safety() {
        std::cout << "=== Exception Safety ===\n";
        
        boost::object_pool<PooledObject> pool;
        
        try {
            PooledObject* obj = pool.construct(1);
            
            // Exception-safe processing
            try {
                obj->process();
                // Simulate potential exception
                if (obj->getId() == 1) {
                    // Don't actually throw in this demo
                    // throw std::runtime_error("Processing failed");
                }
            } catch (...) {
                // Always return object to pool even if processing fails
                pool.destroy(obj);
                throw;
            }
            
            // Normal path
            pool.destroy(obj);
            
        } catch (const std::exception& e) {
            std::cout << "Exception handled: " << e.what() << "\n";
        }
        
        std::cout << "Object properly returned to pool despite exception\n";
    }
};
```

#### When to Use Object Pools

**Use Object Pools When:**
✅ Objects are expensive to construct/destruct  
✅ You create/destroy objects frequently  
✅ Objects have predictable lifecycles  
✅ Memory fragmentation is a concern  
✅ You need deterministic allocation performance  

**Avoid Object Pools When:**
❌ Objects have widely varying sizes  
❌ Object lifetimes are unpredictable  
❌ Pool management overhead exceeds allocation cost  
❌ Memory usage is more important than speed  
❌ Objects hold large amounts of state that can't be reset  

```cpp
void demonstrate_usage_guidelines() {
    std::cout << "=== Usage Guidelines ===\n";
    
    OptimalPoolUsage::demonstrate_pool_sizing();
    OptimalPoolUsage::demonstrate_lifecycle_management();
    OptimalPoolUsage::demonstrate_exception_safety();
    
    demonstrate_performance_optimization();
}
```

### Comparison with std Smart Pointers

Understanding the relationship between Boost smart pointers and their standard library counterparts is crucial for modern C++ development. This section provides detailed comparisons and migration strategies.

#### Feature-by-Feature Comparison

```cpp
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

#include <memory>
#include <iostream>
#include <vector>
#include <chrono>

class ComparisonResource {
public:
    ComparisonResource(int id) : id_(id) {
        std::cout << "ComparisonResource " << id_ << " created\n";
    }
    
    ~ComparisonResource() {
        std::cout << "ComparisonResource " << id_ << " destroyed\n";
    }
    
    void use() const {
        std::cout << "Using ComparisonResource " << id_ << "\n";
    }
    
    int getId() const { return id_; }
    
private:
    int id_;
};

void compare_shared_ptr_implementations() {
    std::cout << "=== boost::shared_ptr vs std::shared_ptr ===\n";
    
    // 1. Basic functionality - nearly identical
    {
        boost::shared_ptr<ComparisonResource> boost_ptr = boost::make_shared<ComparisonResource>(1);
        std::shared_ptr<ComparisonResource> std_ptr = std::make_shared<ComparisonResource>(2);
        
        std::cout << "Boost ref count: " << boost_ptr.use_count() << "\n";
        std::cout << "Std ref count: " << std_ptr.use_count() << "\n";
        
        boost_ptr->use();
        std_ptr->use();
    }
    
    // 2. Custom deleter syntax differences
    {
        auto custom_deleter = [](ComparisonResource* ptr) {
            std::cout << "Custom deleter called\n";
            delete ptr;
        };
        
        // Both support custom deleters, but syntax is slightly different
        boost::shared_ptr<ComparisonResource> boost_custom(new ComparisonResource(3), custom_deleter);
        std::shared_ptr<ComparisonResource> std_custom(new ComparisonResource(4), custom_deleter);
    }
    
    // 3. Aliasing constructor (both support)
    {
        struct Container {
            ComparisonResource resource{5};
            int other_data = 42;
        };
        
        auto container = std::make_shared<Container>();
        
        // Aliasing: shared_ptr to member, but keeps whole container alive
        std::shared_ptr<ComparisonResource> std_alias(container, &container->resource);
        boost::shared_ptr<ComparisonResource> boost_alias(
            boost::shared_ptr<Container>(container), &container->resource);
        
        std::cout << "Aliasing ref counts: " << std_alias.use_count() << ", " << boost_alias.use_count() << "\n";
    }
}

void compare_unique_ptr_vs_scoped_ptr() {
    std::cout << "=== boost::scoped_ptr vs std::unique_ptr ===\n";
    
    // Key differences table format
    std::cout << "Feature Comparison:\n";
    std::cout << "┌─────────────────────┬─────────────────┬─────────────────┐\n";
    std::cout << "│ Feature             │ scoped_ptr      │ unique_ptr      │\n";
    std::cout << "├─────────────────────┼─────────────────┼─────────────────┤\n";
    std::cout << "│ Move Semantics      │ No              │ Yes             │\n";
    std::cout << "│ Custom Deleter      │ Limited         │ Full Support    │\n";
    std::cout << "│ Array Support       │ No              │ Yes (unique_ptr<T[]>) │\n";
    std::cout << "│ Container Storage   │ No              │ Yes             │\n";
    std::cout << "│ Function Return     │ No              │ Yes             │\n";
    std::cout << "│ Standard Library    │ No              │ Yes (C++11+)    │\n";
    std::cout << "└─────────────────────┴─────────────────┴─────────────────┘\n";
    
    // Practical examples
    {
        // scoped_ptr - simple RAII
        boost::scoped_ptr<ComparisonResource> scoped(new ComparisonResource(6));
        
        // unique_ptr - moveable RAII
        std::unique_ptr<ComparisonResource> unique = std::make_unique<ComparisonResource>(7);
        
        // This works with unique_ptr but not scoped_ptr
        std::vector<std::unique_ptr<ComparisonResource>> container;
        container.push_back(std::move(unique));
        
        // scoped_ptr cannot be moved or stored in containers
        // std::vector<boost::scoped_ptr<ComparisonResource>> bad_container; // Won't compile
    }
}

void compare_weak_ptr_implementations() {
    std::cout << "=== boost::weak_ptr vs std::weak_ptr ===\n";
    
    // Functionality is very similar
    {
        auto boost_shared = boost::make_shared<ComparisonResource>(8);
        auto std_shared = std::make_shared<ComparisonResource>(9);
        
        boost::weak_ptr<ComparisonResource> boost_weak = boost_shared;
        std::weak_ptr<ComparisonResource> std_weak = std_shared;
        
        // Both provide lock() and expired()
        if (auto locked_boost = boost_weak.lock()) {
            locked_boost->use();
        }
        
        if (auto locked_std = std_weak.lock()) {
            locked_std->use();
        }
        
        std::cout << "Boost weak expired: " << boost_weak.expired() << "\n";
        std::cout << "Std weak expired: " << std_weak.expired() << "\n";
        
        // Reset shared pointers
        boost_shared.reset();
        std_shared.reset();
        
        std::cout << "After reset - Boost weak expired: " << boost_weak.expired() << "\n";
        std::cout << "After reset - Std weak expired: " << std_weak.expired() << "\n";
    }
}

void compare_intrusive_ptr_alternatives() {
    std::cout << "=== boost::intrusive_ptr vs Standard Alternatives ===\n";
    
    std::cout << "boost::intrusive_ptr has no direct std equivalent\n";
    std::cout << "Closest alternatives:\n";
    std::cout << "1. std::shared_ptr - but with control block overhead\n";
    std::cout << "2. Custom smart pointer implementation\n";
    std::cout << "3. Raw pointers with manual reference counting\n";
    
    // No standard library equivalent exists
    // intrusive_ptr remains unique to Boost
}

// Performance comparison framework
template<typename SmartPtrType, typename MakeFunction>
auto benchmark_smart_ptr(const std::string& name, MakeFunction make_func) {
    const int NUM_ITERATIONS = 100000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto ptr1 = make_func(i);
        auto ptr2 = ptr1;  // Copy
        auto ptr3 = ptr2;  // Another copy
        
        ptr1.reset();
        ptr2.reset();
        // ptr3 will be automatically cleaned up
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << name << " time: " << duration.count() << " microseconds\n";
    return duration;
}

void performance_comparison() {
    std::cout << "=== Performance Comparison ===\n";
    
    // Compare shared_ptr implementations
    auto boost_time = benchmark_smart_ptr<boost::shared_ptr<ComparisonResource>>(
        "boost::shared_ptr",
        [](int i) { return boost::make_shared<ComparisonResource>(i); }
    );
    
    auto std_time = benchmark_smart_ptr<std::shared_ptr<ComparisonResource>>(
        "std::shared_ptr",
        [](int i) { return std::make_shared<ComparisonResource>(i); }
    );
    
    double ratio = static_cast<double>(boost_time.count()) / std_time.count();
    std::cout << "Performance ratio (boost/std): " << ratio << "\n";
    
    if (ratio < 0.95) {
        std::cout << "boost::shared_ptr is faster\n";
    } else if (ratio > 1.05) {
        std::cout << "std::shared_ptr is faster\n";
    } else {
        std::cout << "Performance is similar\n";
    }
}
```

#### Migration Strategies from Boost to std

```cpp
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <memory>
#include <iostream>

// Migration helper templates
template<typename T>
std::shared_ptr<T> migrate_shared_ptr(const boost::shared_ptr<T>& boost_ptr) {
    if (!boost_ptr) {
        return std::shared_ptr<T>();
    }
    
    // Create std::shared_ptr that shares ownership
    return std::shared_ptr<T>(boost_ptr.get(), [boost_ptr](T*) {
        // Custom deleter that holds boost::shared_ptr
        // This ensures proper cleanup when std::shared_ptr is destroyed
    });
}

template<typename T>
std::unique_ptr<T> migrate_scoped_ptr(boost::scoped_ptr<T>& boost_ptr) {
    // Transfer ownership from scoped_ptr to unique_ptr
    T* raw_ptr = boost_ptr.release();
    return std::unique_ptr<T>(raw_ptr);
}

// Step-by-step migration example
class MigrationExample {
private:
    // Legacy Boost-based members
    boost::shared_ptr<ComparisonResource> legacy_shared_;
    boost::scoped_ptr<ComparisonResource> legacy_scoped_;
    
    // Modern std-based members
    std::shared_ptr<ComparisonResource> modern_shared_;
    std::unique_ptr<ComparisonResource> modern_scoped_;
    
public:
    // Phase 1: Support both APIs during transition
    void setLegacyShared(boost::shared_ptr<ComparisonResource> ptr) {
        legacy_shared_ = ptr;
        // Also update modern version
        modern_shared_ = std::shared_ptr<ComparisonResource>(ptr.get(), [ptr](ComparisonResource*){});
    }
    
    void setModernShared(std::shared_ptr<ComparisonResource> ptr) {
        modern_shared_ = ptr;
        // Note: Cannot easily convert std::shared_ptr back to boost::shared_ptr
        // This direction should be avoided
    }
    
    // Phase 2: Provide migration utilities
    void migrateScopedToUnique() {
        if (legacy_scoped_) {
            modern_scoped_ = std::make_unique<ComparisonResource>(*legacy_scoped_);
            legacy_scoped_.reset();
        }
    }
    
    // Phase 3: Unified interface (prefer std types)
    std::shared_ptr<ComparisonResource> getSharedResource() const {
        return modern_shared_ ? modern_shared_ : 
               std::shared_ptr<ComparisonResource>(legacy_shared_.get(), [this](ComparisonResource*){});
    }
    
    // Phase 4: Deprecation warnings for legacy methods
    [[deprecated("Use setModernShared instead")]]
    void setLegacyResource(boost::shared_ptr<ComparisonResource> ptr) {
        setLegacyShared(ptr);
    }
};

void demonstrate_migration_strategy() {
    std::cout << "=== Migration Strategy ===\n";
    
    MigrationExample example;
    
    // Step 1: Identify legacy usage
    auto legacy_resource = boost::make_shared<ComparisonResource>(10);
    example.setLegacyShared(legacy_resource);
    
    // Step 2: Introduce modern equivalents alongside
    auto modern_resource = std::make_shared<ComparisonResource>(11);
    example.setModernShared(modern_resource);
    
    // Step 3: Use unified interface
    auto unified_resource = example.getSharedResource();
    if (unified_resource) {
        unified_resource->use();
    }
    
    std::cout << "Migration completed successfully\n";
}

// Comprehensive migration checklist
void migration_checklist() {
    std::cout << "=== Migration Checklist ===\n";
    
    std::cout << "□ 1. Audit existing Boost smart pointer usage\n";
    std::cout << "□ 2. Identify dependencies and interfaces\n";
    std::cout << "□ 3. Create migration plan (inside-out or outside-in)\n";
    std::cout << "□ 4. Implement adapter/wrapper functions\n";
    std::cout << "□ 5. Update unit tests to cover both versions\n";
    std::cout << "□ 6. Gradual replacement of Boost types\n";
    std::cout << "□ 7. Update build system and dependencies\n";
    std::cout << "□ 8. Performance testing and validation\n";
    std::cout << "□ 9. Remove Boost dependencies\n";
    std::cout << "□ 10. Code review and documentation update\n";
    
    std::cout << "\nCommon pitfalls to avoid:\n";
    std::cout << "❌ Don't mix reference counting systems\n";
    std::cout << "❌ Don't assume identical performance\n";
    std::cout << "❌ Don't forget about custom deleters\n";
    std::cout << "❌ Don't ignore compilation differences\n";
    std::cout << "❌ Don't rush the migration process\n";
}
```

#### When to Use Boost vs std

```cpp
void usage_decision_matrix() {
    std::cout << "=== Decision Matrix: Boost vs std ===\n";
    
    std::cout << "\nUse Boost Smart Pointers When:\n";
    std::cout << "✓ Working with legacy codebases\n";
    std::cout << "✓ Maintaining existing Boost-dependent code\n";  
    std::cout << "✓ Need intrusive_ptr (no std equivalent)\n";
    std::cout << "✓ Specific Boost features not in std\n";
    std::cout << "✓ Team expertise in Boost ecosystem\n";
    
    std::cout << "\nUse std Smart Pointers When:\n";
    std::cout << "✓ Starting new projects (C++11+)\n";
    std::cout << "✓ Want standard library consistency\n";
    std::cout << "✓ Need move semantics and modern C++ features\n";
    std::cout << "✓ Minimize external dependencies\n";
    std::cout << "✓ Working with modern libraries/frameworks\n";
    std::cout << "✓ Long-term maintainability is priority\n";
    
    std::cout << "\nMigration Priority:\n";
    std::cout << "High:    boost::scoped_ptr → std::unique_ptr\n";
    std::cout << "Medium:  boost::shared_ptr → std::shared_ptr\n";
    std::cout << "Low:     boost::weak_ptr → std::weak_ptr\n";
    std::cout << "Special: boost::intrusive_ptr → Custom solution or keep Boost\n";
}

// Feature matrix for detailed comparison
void detailed_feature_matrix() {
    std::cout << "=== Detailed Feature Matrix ===\n";
    
    const char* features[][4] = {
        {"Feature", "boost::shared_ptr", "std::shared_ptr", "Notes"},
        {"Thread Safety", "Yes", "Yes", "Reference counting only"},
        {"Custom Deleter", "Yes", "Yes", "Slightly different syntax"},
        {"make_shared", "Yes", "Yes", "Performance optimization"},
        {"Aliasing", "Yes", "Yes", "Share ownership, different pointer"},
        {"weak_ptr Support", "Yes", "Yes", "Break circular references"},
        {"Array Support", "Limited", "Limited", "Use unique_ptr<T[]> instead"},
        {"Exception Safety", "Strong", "Strong", "RAII guarantees"},
        {"Performance", "Good", "Good", "Implementation dependent"},
        {"Standard Library", "No", "Yes", "C++11 and later"},
        {"Header Only", "No", "Yes", "Boost requires linking"},
    };
    
    const int num_rows = sizeof(features) / sizeof(features[0]);
    const int col_widths[] = {16, 18, 18, 25};
    
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << std::left << std::setw(col_widths[j]) << features[i][j];
        }
        std::cout << "\n";
        
        if (i == 0) {  // Header separator
            for (int j = 0; j < 4; ++j) {
                std::cout << std::string(col_widths[j] - 1, '-') << " ";
            }
            std::cout << "\n";
        }
    }
}
```

#### Final Recommendations

```cpp
void final_recommendations() {
    std::cout << "=== Final Recommendations ===\n";
    
    std::cout << "\n📊 For New Projects:\n";
    std::cout << "   • Use std::unique_ptr for exclusive ownership\n";
    std::cout << "   • Use std::shared_ptr for shared ownership\n";
    std::cout << "   • Use std::weak_ptr to break cycles\n";
    std::cout << "   • Consider boost::intrusive_ptr only for performance-critical scenarios\n";
    
    std::cout << "\n🔄 For Legacy Projects:\n";
    std::cout << "   • Plan gradual migration strategy\n";
    std::cout << "   • Start with scoped_ptr → unique_ptr (easiest)\n";
    std::cout << "   • Test thoroughly during migration\n";
    std::cout << "   • Keep documentation updated\n";
    
    std::cout << "\n⚡ Performance Considerations:\n";
    std::cout << "   • Both Boost and std implementations are well-optimized\n";
    std::cout << "   • Use make_shared/make_unique for better performance\n";
    std::cout << "   • Consider intrusive_ptr for high-frequency allocation scenarios\n";
    std::cout << "   • Profile before and after migration\n";
    
    std::cout << "\n🎯 Key Takeaways:\n";
    std::cout << "   • std smart pointers are the future\n";
    std::cout << "   • Boost smart pointers are still valuable for legacy code\n";
    std::cout << "   • Migration should be gradual and well-tested\n";
    std::cout << "   • Choose based on project requirements, not just novelty\n";
}

void comprehensive_comparison_demo() {
    compare_shared_ptr_implementations();
    compare_unique_ptr_vs_scoped_ptr();
    compare_weak_ptr_implementations();
    compare_intrusive_ptr_alternatives();
    performance_comparison();
    demonstrate_migration_strategy();
    migration_checklist();
    usage_decision_matrix();
    detailed_feature_matrix();
    final_recommendations();
}
```

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

### Exercise 1: Smart Pointer Comparison and Analysis

**Objective:** Implement the same functionality using different smart pointer types and analyze their trade-offs.

**Task:** Create a resource management system for a simple game engine where game objects need to be shared between different systems (rendering, physics, AI).

```cpp
// TODO: Complete this implementation
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <memory>
#include <vector>
#include <iostream>

class GameObject {
    // TODO: Implement with different smart pointer approaches
    // 1. Using boost::shared_ptr
    // 2. Using std::shared_ptr  
    // 3. Using boost::intrusive_ptr
    
    int id_;
    std::string name_;
    float position_[3];
    
public:
    GameObject(int id, const std::string& name);
    ~GameObject();
    
    void update();
    void render() const;
    int getId() const { return id_; }
    const std::string& getName() const { return name_; }
};

class GameSystem {
    // TODO: Implement three versions:
    // - SharedPtrGameSystem (using boost::shared_ptr)
    // - StdGameSystem (using std::shared_ptr)
    // - IntrusiveGameSystem (using boost::intrusive_ptr)
    
public:
    virtual void addObject(/* smart pointer type */ obj) = 0;
    virtual void removeObject(int id) = 0;
    virtual void updateAll() = 0;
    virtual size_t getObjectCount() const = 0;
};

// TODO: Implement and compare:
// 1. Memory usage of each approach
// 2. Performance characteristics
// 3. API usability
// 4. Thread safety considerations

void benchmark_smart_pointer_approaches() {
    const int NUM_OBJECTS = 10000;
    const int NUM_ITERATIONS = 1000;
    
    // TODO: Measure and compare:
    // - Object creation time
    // - Copy overhead
    // - Memory consumption
    // - Cache performance
}
```

**Expected Learning Outcomes:**
- Understand performance trade-offs between different smart pointer types
- Experience the API differences firsthand
- Learn to choose appropriate smart pointer for specific use cases

### Exercise 2: Custom Reference Counting Implementation

**Objective:** Build a custom intrusive reference counting system and compare it with boost::intrusive_ptr.

```cpp
// TODO: Implement a thread-safe intrusive reference counting base class
class CustomRefCounted {
    // TODO: Implement reference counting with:
    // - Thread-safe atomic operations
    // - Custom destruction callbacks
    // - Debug reference tracking
    // - Circular reference detection
    
private:
    mutable std::atomic<long> ref_count_;
    std::function<void(CustomRefCounted*)> destruction_callback_;
    
public:
    CustomRefCounted();
    virtual ~CustomRefCounted();
    
    void add_ref() const;
    void release() const;
    long use_count() const;
    
    // TODO: Add debugging and monitoring features
    void setDestructionCallback(std::function<void(CustomRefCounted*)> callback);
    static size_t getTotalActiveObjects();
    static void printRefCountStats();
};

// TODO: Implement smart pointer wrapper
template<typename T>
class custom_intrusive_ptr {
    // TODO: Implement complete smart pointer interface
    // - Construction, destruction, assignment
    // - Comparison operators
    // - Access operators (*, ->)
    // - Boolean conversion
    // - Reset and swap operations
};

// TODO: Compare with boost::intrusive_ptr:
// - Performance benchmarks
// - Memory overhead
// - Feature completeness
// - Thread safety
```

**Expected Learning Outcomes:**
- Deep understanding of reference counting mechanics
- Appreciation for the complexity of thread-safe smart pointers
- Hands-on experience with atomic operations
- Understanding of smart pointer design patterns

### Exercise 3: Memory Pool Implementation and Optimization

**Objective:** Create a custom object pool with advanced features and optimize it for specific use cases.

```cpp
// TODO: Design and implement a high-performance object pool
template<typename T>
class AdvancedObjectPool {
    // TODO: Implement with these features:
    // - Dynamic pool sizing based on usage patterns
    // - Thread-safe allocation/deallocation
    // - Object recycling with state reset
    // - Memory statistics and monitoring
    // - Defragmentation and cleanup strategies
    
public:
    struct PoolStats {
        size_t total_allocated;
        size_t currently_in_use;
        size_t peak_usage;
        size_t allocation_count;
        size_t deallocation_count;
        double hit_rate;  // successful reuse rate
    };
    
    AdvancedObjectPool(size_t initial_size = 100, size_t max_size = 1000);
    ~AdvancedObjectPool();
    
    // TODO: Implement advanced allocation strategies
    T* acquire();
    void release(T* obj);
    
    // TODO: Implement monitoring and tuning
    PoolStats getStats() const;
    void optimize();  // Adjust pool size based on usage patterns
    void defragment();  // Reorganize memory for better cache performance
    
private:
    // TODO: Choose and implement efficient data structures
    // Consider: ring buffer, free list, segmented allocation
};

// TODO: Create specialized pools for different scenarios:
// 1. High-frequency small objects (e.g., game particles)
// 2. Variable-size objects with size categories
// 3. Objects with complex initialization/cleanup
// 4. Thread-local pools vs global pools

class GameParticle {
    // Small, frequently created/destroyed object
    float position_[3];
    float velocity_[3];
    float color_[4];
    float lifetime_;
    
public:
    // TODO: Implement with pool-friendly design
    void reset();  // Prepare for reuse
    bool isExpired() const;
    void update(float dt);
};

// TODO: Implement benchmarking suite
void benchmark_pool_strategies() {
    // TODO: Compare:
    // - Standard new/delete
    // - boost::object_pool
    // - Custom AdvancedObjectPool
    // - STL allocators with pools
    
    // Test scenarios:
    // - Steady state allocation/deallocation
    // - Burst allocation patterns
    // - Memory fragmentation stress test
    // - Multi-threaded allocation
}
```

**Expected Learning Outcomes:**
- Understanding of memory allocation patterns and their performance impact
- Experience with pool design trade-offs
- Knowledge of memory optimization techniques
- Practical experience with performance measurement and tuning

### Exercise 4: Legacy Code Migration Project

**Objective:** Migrate a legacy codebase from Boost smart pointers to standard library equivalents.

```cpp
// TODO: You're given this legacy codebase using Boost smart pointers
// Your task is to create a migration strategy and implement it

// Legacy header file (legacy_system.h)
class LegacyDataManager {
    boost::shared_ptr<Database> database_;
    boost::scoped_ptr<ConfigManager> config_;
    std::vector<boost::shared_ptr<DataProcessor>> processors_;
    boost::weak_ptr<EventSystem> event_system_;
    
public:
    bool initialize(boost::shared_ptr<Database> db);
    void addProcessor(boost::shared_ptr<DataProcessor> processor);
    void setEventSystem(boost::shared_ptr<EventSystem> events);
    void processData(boost::shared_ptr<DataSet> data);
    boost::shared_ptr<ProcessResult> getResult() const;
};

// TODO: Create migration strategy:
// 1. Analyze dependencies and create dependency graph
// 2. Design migration phases (which components to migrate first)
// 3. Create adapter layers for gradual migration
// 4. Implement automated migration tools/scripts where possible
// 5. Create testing strategy to ensure no regressions

// Phase 1: Create adapter layer
template<typename T>
class SmartPtrAdapter {
    // TODO: Allow seamless interop between Boost and std smart pointers
    // Support both boost::shared_ptr<T> and std::shared_ptr<T>
    // Provide unified interface
};

// Phase 2: Modern version
class ModernDataManager {
    // TODO: Implement using std smart pointers
    // Consider if any design improvements can be made during migration
    // Maintain API compatibility where possible
};

// Phase 3: Migration utilities
class MigrationTools {
public:
    // TODO: Implement utilities to help with migration:
    
    // Convert boost::shared_ptr to std::shared_ptr
    template<typename T>
    static std::shared_ptr<T> migrate_shared_ptr(const boost::shared_ptr<T>& boost_ptr);
    
    // Convert boost::scoped_ptr to std::unique_ptr
    template<typename T>
    static std::unique_ptr<T> migrate_scoped_ptr(boost::scoped_ptr<T>& boost_ptr);
    
    // Detect potential issues in migration
    static std::vector<std::string> analyzeMigrationRisks(const std::string& source_code);
    
    // Generate migration report
    static void generateMigrationReport(const std::string& codebase_path);
};

// TODO: Create comprehensive test suite
class MigrationTests {
    // Test compatibility between old and new implementations
    // Test performance before and after migration
    // Test memory usage patterns
    // Test thread safety (if applicable)
    // Test error handling and edge cases
};
```

**Expected Learning Outcomes:**
- Experience with real-world migration challenges
- Understanding of API compatibility considerations
- Knowledge of testing strategies for migrations
- Appreciation for gradual migration approaches
- Experience with legacy code modernization

### Exercise 5: Thread-Safe Smart Pointer Utilities

**Objective:** Implement thread-safe utilities and patterns for smart pointer usage in multi-threaded environments.

```cpp
// TODO: Implement thread-safe smart pointer utilities

// 1. Thread-safe factory with caching
template<typename T, typename... Args>
class ThreadSafeFactory {
    // TODO: Implement with these features:
    // - Thread-safe object creation
    // - Caching of frequently created objects
    // - Weak reference cleanup
    // - Object pooling integration
    
    std::mutex cache_mutex_;
    std::unordered_map<std::string, std::weak_ptr<T>> cache_;
    
public:
    std::shared_ptr<T> create(const std::string& key, Args... args);
    void clearCache();
    size_t getCacheSize() const;
};

// 2. Thread-safe observer pattern
template<typename T>
class ThreadSafeObservable {
    // TODO: Implement observer pattern with weak_ptr to avoid cycles
    // Handle observer cleanup automatically
    // Provide thread-safe notification
    
    mutable std::shared_mutex observers_mutex_;
    std::vector<std::weak_ptr<Observer<T>>> observers_;
    
public:
    void addObserver(std::shared_ptr<Observer<T>> observer);
    void removeObserver(std::shared_ptr<Observer<T>> observer);
    void notifyObservers(const T& data);
    void cleanupExpiredObservers();
};

// 3. Lock-free smart pointer operations
template<typename T>
class LockFreeSmartPtrQueue {
    // TODO: Implement lock-free queue using atomic operations
    // Use hazard pointers or similar technique for memory reclamation
    // Compare performance with mutex-based approaches
    
    struct Node {
        std::atomic<std::shared_ptr<T>> data;
        std::atomic<Node*> next;
    };
    
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
    
public:
    void enqueue(std::shared_ptr<T> item);
    std::shared_ptr<T> dequeue();
    bool empty() const;
};

// 4. Smart pointer debugging utilities
class SmartPtrDebugger {
    // TODO: Implement debugging helpers:
    // - Track smart pointer creation/destruction
    // - Detect circular references
    // - Monitor reference count changes
    // - Generate memory usage reports
    
public:
    static void enableTracking();
    static void disableTracking();
    static void printActivePointers();
    static std::vector<CircularReference> detectCircularReferences();
    static MemoryUsageReport generateReport();
};

// TODO: Implement stress testing framework
class ConcurrencyStressTester {
    // Create high-contention scenarios
    // Test smart pointer behavior under extreme load
    // Verify thread safety guarantees
    // Measure performance degradation under contention
};
```

**Expected Learning Outcomes:**
- Deep understanding of thread safety issues with smart pointers
- Experience with lock-free programming techniques
- Knowledge of debugging techniques for memory management
- Understanding of performance implications in multi-threaded environments

### Assessment Criteria

For each exercise, you should demonstrate:

**Technical Implementation (40%)**
- Correct and complete implementation
- Proper error handling
- Thread safety where applicable
- Memory safety and leak prevention

**Performance Analysis (25%)**
- Meaningful benchmarks and measurements
- Analysis of performance characteristics
- Understanding of trade-offs
- Optimization strategies

**Code Quality (20%)**
- Clean, readable code structure
- Appropriate use of modern C++ features
- Good API design
- Comprehensive testing

**Understanding and Documentation (15%)**
- Clear explanation of design decisions
- Understanding of underlying concepts
- Comparison of different approaches
- Documentation of lessons learned

### Bonus Challenges

1. **Cross-Platform Compatibility**: Ensure your implementations work correctly on Windows, Linux, and macOS
2. **Exception Safety**: Implement strong exception safety guarantees
3. **Custom Allocators**: Integrate with custom allocators for specialized use cases
4. **Serialization Support**: Add serialization capabilities to your smart pointer utilities
5. **Profiling Integration**: Add hooks for memory profilers and debugging tools

## Performance Considerations

### Reference Counting Overhead

Understanding the performance implications of reference counting is crucial for making informed decisions about smart pointer usage.

#### Atomic Operations Cost

```cpp
#include <boost/shared_ptr.hpp>
#include <memory>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>

class PerformanceTestObject {
public:
    PerformanceTestObject(int value) : data_(value) {}
    void process() { data_ = data_ * 2 + 1; }  // Simple operation
    int getData() const { return data_; }
    
private:
    int data_;
};

void demonstrate_atomic_overhead() {
    std::cout << "=== Atomic Operations Overhead ===\n";
    
    const int NUM_OPERATIONS = 1000000;
    std::atomic<int> atomic_counter{0};
    int regular_counter = 0;
    
    // Measure atomic operations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        ++atomic_counter;  // Atomic increment (used in reference counting)
    }
    auto atomic_time = std::chrono::high_resolution_clock::now() - start;
    
    // Measure regular operations
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        ++regular_counter;  // Regular increment
    }
    auto regular_time = std::chrono::high_resolution_clock::now() - start;
    
    auto atomic_ms = std::chrono::duration_cast<std::chrono::microseconds>(atomic_time).count();
    auto regular_ms = std::chrono::duration_cast<std::chrono::microseconds>(regular_time).count();
    
    std::cout << "Atomic operations: " << atomic_ms << " microseconds\n";
    std::cout << "Regular operations: " << regular_ms << " microseconds\n";
    std::cout << "Overhead factor: " << (double)atomic_ms / regular_ms << "x\n";
    
    // This overhead is multiplied in shared_ptr copy operations
    std::cout << "\nImplication: Each shared_ptr copy involves 1-2 atomic operations\n";
}

void benchmark_shared_ptr_copies() {
    std::cout << "=== shared_ptr Copy Performance ===\n";
    
    const int NUM_COPIES = 100000;
    auto original = boost::make_shared<PerformanceTestObject>(42);
    
    // Measure boost::shared_ptr copy overhead
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_COPIES; ++i) {
        boost::shared_ptr<PerformanceTestObject> copy = original;
        copy.reset();  // Immediately release
    }
    
    auto boost_time = std::chrono::high_resolution_clock::now() - start;
    
    // Measure std::shared_ptr copy overhead  
    auto std_original = std::make_shared<PerformanceTestObject>(42);
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_COPIES; ++i) {
        std::shared_ptr<PerformanceTestObject> copy = std_original;
        copy.reset();  // Immediately release
    }
    
    auto std_time = std::chrono::high_resolution_clock::now() - start;
    
    // Measure raw pointer copy (baseline)
    PerformanceTestObject* raw_ptr = new PerformanceTestObject(42);
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_COPIES; ++i) {
        PerformanceTestObject* copy = raw_ptr;  // Just pointer copy
        (void)copy;  // Use the copy to prevent optimization
    }
    
    auto raw_time = std::chrono::high_resolution_clock::now() - start;
    delete raw_ptr;
    
    auto boost_us = std::chrono::duration_cast<std::chrono::microseconds>(boost_time).count();
    auto std_us = std::chrono::duration_cast<std::chrono::microseconds>(std_time).count();
    auto raw_us = std::chrono::duration_cast<std::chrono::microseconds>(raw_time).count();
    
    std::cout << "boost::shared_ptr copies: " << boost_us << " microseconds\n";
    std::cout << "std::shared_ptr copies: " << std_us << " microseconds\n";
    std::cout << "Raw pointer copies: " << raw_us << " microseconds\n";
    std::cout << "shared_ptr overhead: " << (double)std::min(boost_us, std_us) / raw_us << "x\n";
}
```

#### Cache Line Bouncing in Multithreaded Scenarios

```cpp
#include <thread>
#include <vector>
#include <chrono>

void demonstrate_cache_line_bouncing() {
    std::cout << "=== Cache Line Bouncing ===\n";
    
    const int NUM_THREADS = 4;
    const int OPERATIONS_PER_THREAD = 100000;
    
    auto shared_object = boost::make_shared<PerformanceTestObject>(1);
    
    // Test 1: Multiple threads copying the same shared_ptr (cache bouncing)
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&shared_object, OPERATIONS_PER_THREAD]() {
            for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
                boost::shared_ptr<PerformanceTestObject> local_copy = shared_object;
                local_copy->process();  // Do some work
                // local_copy destructor decrements reference count (atomic operation)
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto contended_time = std::chrono::high_resolution_clock::now() - start;
    
    // Test 2: Each thread works with its own shared_ptr (no contention)
    std::vector<boost::shared_ptr<PerformanceTestObject>> individual_objects;
    for (int i = 0; i < NUM_THREADS; ++i) {
        individual_objects.push_back(boost::make_shared<PerformanceTestObject>(i));
    }
    
    start = std::chrono::high_resolution_clock::now();
    
    threads.clear();
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&individual_objects, t, OPERATIONS_PER_THREAD]() {
            for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
                boost::shared_ptr<PerformanceTestObject> local_copy = individual_objects[t];
                local_copy->process();
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto non_contended_time = std::chrono::high_resolution_clock::now() - start;
    
    auto contended_ms = std::chrono::duration_cast<std::chrono::milliseconds>(contended_time).count();
    auto non_contended_ms = std::chrono::duration_cast<std::chrono::milliseconds>(non_contended_time).count();
    
    std::cout << "Contended access (shared object): " << contended_ms << " ms\n";
    std::cout << "Non-contended access (individual objects): " << non_contended_ms << " ms\n";
    std::cout << "Contention overhead: " << (double)contended_ms / non_contended_ms << "x\n";
    
    std::cout << "\nMitigation strategies:\n";
    std::cout << "• Use thread-local copies when possible\n";
    std::cout << "• Minimize shared_ptr copying in hot paths\n";
    std::cout << "• Consider using unique_ptr when exclusive ownership is sufficient\n";
}
```

#### Memory Overhead of Control Blocks

```cpp
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <memory>

void analyze_memory_overhead() {
    std::cout << "=== Memory Overhead Analysis ===\n";
    
    class SmallObject {
        char data[8];  // 8 bytes of actual data
    };
    
    class LargeObject {
        char data[1024];  // 1KB of actual data
    };
    
    std::cout << "Object sizes:\n";
    std::cout << "SmallObject: " << sizeof(SmallObject) << " bytes\n";
    std::cout << "LargeObject: " << sizeof(LargeObject) << " bytes\n";
    std::cout << "shared_ptr: " << sizeof(boost::shared_ptr<SmallObject>) << " bytes\n";
    std::cout << "unique_ptr: " << sizeof(std::unique_ptr<SmallObject>) << " bytes\n";
    std::cout << "raw pointer: " << sizeof(SmallObject*) << " bytes\n";
    
    // Control block overhead analysis
    std::cout << "\nControl block overhead (approximate):\n";
    std::cout << "• Reference count: 4-8 bytes (atomic)\n";
    std::cout << "• Weak reference count: 4-8 bytes (atomic)\n";
    std::cout << "• Virtual destructor: 8 bytes (vtable pointer)\n";
    std::cout << "• Deleter storage: varies (often 8-16 bytes)\n";
    std::cout << "• Alignment padding: 0-7 bytes\n";
    std::cout << "Total control block: ~24-48 bytes\n";
    
    // Impact analysis
    std::cout << "\nRelative overhead:\n";
    std::cout << "SmallObject (8 bytes) + control block (~32 bytes) = 400% overhead\n";
    std::cout << "LargeObject (1024 bytes) + control block (~32 bytes) = 3% overhead\n";
    
    std::cout << "\nOptimization strategies:\n";
    std::cout << "• Use make_shared to combine object and control block allocation\n";
    std::cout << "• Consider intrusive_ptr for small objects with frequent copying\n";
    std::cout << "• Use unique_ptr when sharing is not needed\n";
}
```

### Pool Allocation Benefits

#### Reduced Memory Fragmentation

```cpp
#include <boost/pool/object_pool.hpp>
#include <vector>
#include <random>
#include <chrono>

class FragmentationDemo {
public:
    static void demonstrate_fragmentation_impact() {
        std::cout << "=== Memory Fragmentation Impact ===\n";
        
        const int NUM_OBJECTS = 10000;
        const int NUM_ITERATIONS = 100;
        
        // Test 1: Standard allocation with fragmentation
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            std::vector<PerformanceTestObject*> objects;
            
            // Allocate objects
            for (int i = 0; i < NUM_OBJECTS; ++i) {
                objects.push_back(new PerformanceTestObject(i));
            }
            
            // Randomly deallocate half (creates fragmentation)
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(objects.begin(), objects.end(), gen);
            
            for (size_t i = 0; i < objects.size() / 2; ++i) {
                delete objects[i];
                objects[i] = nullptr;
            }
            
            // Allocate more objects (may not fit in holes)
            for (int i = 0; i < NUM_OBJECTS / 4; ++i) {
                objects.push_back(new PerformanceTestObject(i + NUM_OBJECTS));
            }
            
            // Clean up
            for (PerformanceTestObject* obj : objects) {
                delete obj;
            }
        }
        
        auto standard_time = std::chrono::high_resolution_clock::now() - start;
        
        // Test 2: Pool allocation (no fragmentation)
        start = std::chrono::high_resolution_clock::now();
        
        boost::object_pool<PerformanceTestObject> pool;
        
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            std::vector<PerformanceTestObject*> objects;
            
            // Allocate objects from pool
            for (int i = 0; i < NUM_OBJECTS; ++i) {
                objects.push_back(pool.construct(i));
            }
            
            // Randomly return half to pool
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(objects.begin(), objects.end(), gen);
            
            for (size_t i = 0; i < objects.size() / 2; ++i) {
                pool.destroy(objects[i]);
                objects[i] = nullptr;
            }
            
            // Allocate more objects (efficiently reuses returned memory)
            for (int i = 0; i < NUM_OBJECTS / 4; ++i) {
                objects.push_back(pool.construct(i + NUM_OBJECTS));
            }
            
            // Return all to pool
            for (PerformanceTestObject* obj : objects) {
                if (obj) pool.destroy(obj);
            }
        }
        
        auto pool_time = std::chrono::high_resolution_clock::now() - start;
        
        auto standard_ms = std::chrono::duration_cast<std::chrono::milliseconds>(standard_time).count();
        auto pool_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pool_time).count();
        
        std::cout << "Standard allocation with fragmentation: " << standard_ms << " ms\n";
        std::cout << "Pool allocation (no fragmentation): " << pool_ms << " ms\n";
        std::cout << "Pool performance improvement: " << (double)standard_ms / pool_ms << "x\n";
    }
};
```

#### Faster Allocation/Deallocation

```cpp
void benchmark_allocation_speed() {
    std::cout << "=== Allocation Speed Comparison ===\n";
    
    const int NUM_ALLOCATIONS = 1000000;
    
    // Benchmark standard new/delete
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ALLOCATIONS; ++i) {
        PerformanceTestObject* obj = new PerformanceTestObject(i);
        delete obj;
    }
    
    auto standard_time = std::chrono::high_resolution_clock::now() - start;
    
    // Benchmark pool allocation
    boost::object_pool<PerformanceTestObject> pool;
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ALLOCATIONS; ++i) {
        PerformanceTestObject* obj = pool.construct(i);
        pool.destroy(obj);
    }
    
    auto pool_time = std::chrono::high_resolution_clock::now() - start;
    
    // Benchmark pre-allocated array (best case baseline)
    std::vector<PerformanceTestObject> pre_allocated(NUM_ALLOCATIONS, PerformanceTestObject(0));
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ALLOCATIONS; ++i) {
        PerformanceTestObject* obj = &pre_allocated[i];
        obj->process();  // Simulate usage
    }
    
    auto array_time = std::chrono::high_resolution_clock::now() - start;
    
    auto standard_us = std::chrono::duration_cast<std::chrono::microseconds>(standard_time).count();
    auto pool_us = std::chrono::duration_cast<std::chrono::microseconds>(pool_time).count();
    auto array_us = std::chrono::duration_cast<std::chrono::microseconds>(array_time).count();
    
    std::cout << "Standard new/delete: " << standard_us << " microseconds\n";
    std::cout << "Pool allocation: " << pool_us << " microseconds\n";
    std::cout << "Pre-allocated array: " << array_us << " microseconds\n";
    
    std::cout << "Pool vs standard: " << (double)standard_us / pool_us << "x faster\n";
    std::cout << "Pool vs optimal: " << (double)pool_us / array_us << "x slower\n";
}
```

#### Better Cache Locality

```cpp
void demonstrate_cache_locality() {
    std::cout << "=== Cache Locality Benefits ===\n";
    
    const int NUM_OBJECTS = 100000;
    const int NUM_ACCESSES = 1000000;
    
    // Test 1: Standard allocation (objects scattered in memory)
    std::vector<PerformanceTestObject*> scattered_objects;
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        scattered_objects.push_back(new PerformanceTestObject(i));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int access = 0; access < NUM_ACCESSES; ++access) {
        int index = access % NUM_OBJECTS;
        scattered_objects[index]->process();
    }
    
    auto scattered_time = std::chrono::high_resolution_clock::now() - start;
    
    // Test 2: Pool allocation (objects likely close together)
    boost::object_pool<PerformanceTestObject> pool;
    std::vector<PerformanceTestObject*> pooled_objects;
    
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        pooled_objects.push_back(pool.construct(i));
    }
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int access = 0; access < NUM_ACCESSES; ++access) {
        int index = access % NUM_OBJECTS;
        pooled_objects[index]->process();
    }
    
    auto pooled_time = std::chrono::high_resolution_clock::now() - start;
    
    // Test 3: Array allocation (optimal cache locality)
    std::vector<PerformanceTestObject> array_objects(NUM_OBJECTS, PerformanceTestObject(0));
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int access = 0; access < NUM_ACCESSES; ++access) {
        int index = access % NUM_OBJECTS;
        array_objects[index].process();
    }
    
    auto array_time = std::chrono::high_resolution_clock::now() - start;
    
    // Cleanup scattered objects
    for (PerformanceTestObject* obj : scattered_objects) {
        delete obj;
    }
    
    for (PerformanceTestObject* obj : pooled_objects) {
        pool.destroy(obj);
    }
    
    auto scattered_ms = std::chrono::duration_cast<std::chrono::milliseconds>(scattered_time).count();
    auto pooled_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pooled_time).count();
    auto array_ms = std::chrono::duration_cast<std::chrono::milliseconds>(array_time).count();
    
    std::cout << "Scattered allocation access: " << scattered_ms << " ms\n";
    std::cout << "Pooled allocation access: " << pooled_ms << " ms\n";
    std::cout << "Array allocation access: " << array_ms << " ms\n";
    
    std::cout << "Pool vs scattered: " << (double)scattered_ms / pooled_ms << "x faster\n";
    std::cout << "Pool vs optimal: " << (double)pooled_ms / array_ms << "x slower\n";
}
```

### Migration Strategies

#### Performance Impact Assessment

```cpp
class MigrationPerformanceAnalyzer {
public:
    struct PerformanceMetrics {
        double allocation_time_ms;
        double copy_time_ms;
        double memory_overhead_percent;
        double cache_miss_rate;
        size_t memory_fragmentation_score;
    };
    
    static PerformanceMetrics measureCurrentSystem() {
        std::cout << "=== Current System Performance Baseline ===\n";
        
        PerformanceMetrics metrics{};
        
        // Measure current allocation patterns
        const int NUM_OBJECTS = 10000;
        std::vector<PerformanceTestObject*> objects;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_OBJECTS; ++i) {
            objects.push_back(new PerformanceTestObject(i));
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        metrics.allocation_time_ms = 
            std::chrono::duration<double, std::milli>(end - start).count();
        
        // Measure copy operations (if using smart pointers)
        std::vector<boost::shared_ptr<PerformanceTestObject>> smart_ptrs;
        for (auto* obj : objects) {
            smart_ptrs.emplace_back(obj);
        }
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_OBJECTS; ++i) {
            auto copy = smart_ptrs[i % smart_ptrs.size()];
        }
        end = std::chrono::high_resolution_clock::now();
        
        metrics.copy_time_ms = 
            std::chrono::duration<double, std::milli>(end - start).count();
        
        // Estimate memory overhead
        size_t object_memory = NUM_OBJECTS * sizeof(PerformanceTestObject);
        size_t smart_ptr_memory = smart_ptrs.size() * sizeof(boost::shared_ptr<PerformanceTestObject>);
        size_t control_block_memory = smart_ptrs.size() * 32;  // Approximate
        
        metrics.memory_overhead_percent = 
            ((double)(smart_ptr_memory + control_block_memory) / object_memory) * 100;
        
        std::cout << "Allocation time: " << metrics.allocation_time_ms << " ms\n";
        std::cout << "Copy time: " << metrics.copy_time_ms << " ms\n";
        std::cout << "Memory overhead: " << metrics.memory_overhead_percent << "%\n";
        
        return metrics;
    }
    
    static void compareAfterMigration(const PerformanceMetrics& baseline) {
        std::cout << "=== Post-Migration Performance Comparison ===\n";
        
        auto current = measureCurrentSystem();
        
        std::cout << "Performance changes:\n";
        std::cout << "Allocation: " << (current.allocation_time_ms / baseline.allocation_time_ms) << "x\n";
        std::cout << "Copy operations: " << (current.copy_time_ms / baseline.copy_time_ms) << "x\n";
        std::cout << "Memory overhead: " << (current.memory_overhead_percent - baseline.memory_overhead_percent) << "% change\n";
        
        if (current.allocation_time_ms < baseline.allocation_time_ms * 0.9) {
            std::cout << "✓ Significant allocation improvement\n";
        } else if (current.allocation_time_ms > baseline.allocation_time_ms * 1.1) {
            std::cout << "⚠ Allocation performance regression\n";
        } else {
            std::cout << "→ Allocation performance similar\n";
        }
    }
};
```

#### Gradual Migration Strategy

```cpp
class GradualMigrationPlanner {
public:
    enum class MigrationPhase {
        ASSESSMENT,
        PREPARATION,
        PILOT_MIGRATION,
        INCREMENTAL_ROLLOUT,
        VALIDATION,
        COMPLETION
    };
    
    static void planMigration() {
        std::cout << "=== Gradual Migration Strategy ===\n";
        
        std::cout << "Phase 1: Assessment (1-2 weeks)\n";
        std::cout << "• Audit current smart pointer usage\n";
        std::cout << "• Identify performance-critical paths\n";
        std::cout << "• Measure baseline performance\n";
        std::cout << "• Identify migration risks\n";
        
        std::cout << "\nPhase 2: Preparation (2-3 weeks)\n";
        std::cout << "• Create adapter/bridge classes\n";
        std::cout << "• Update build system for dual support\n";
        std::cout << "• Enhance test coverage\n";
        std::cout << "• Train team on new APIs\n";
        
        std::cout << "\nPhase 3: Pilot Migration (1-2 weeks)\n";
        std::cout << "• Migrate non-critical components first\n";
        std::cout << "• Validate functionality and performance\n";
        std::cout << "• Refine migration process\n";
        std::cout << "• Document lessons learned\n";
        
        std::cout << "\nPhase 4: Incremental Rollout (4-8 weeks)\n";
        std::cout << "• Migrate components in dependency order\n";
        std::cout << "• Continuous integration testing\n";
        std::cout << "• Performance monitoring\n";
        std::cout << "• Rollback capability maintenance\n";
        
        std::cout << "\nPhase 5: Validation (2-3 weeks)\n";
        std::cout << "• Comprehensive system testing\n";
        std::cout << "• Performance regression testing\n";
        std::cout << "• Memory usage validation\n";
        std::cout << "• User acceptance testing\n";
        
        std::cout << "\nPhase 6: Completion (1 week)\n";
        std::cout << "• Remove deprecated Boost dependencies\n";
        std::cout << "• Clean up adapter/bridge code\n";
        std::cout << "• Update documentation\n";
        std::cout << "• Final performance validation\n";
    }
    
    static void assessMigrationRisks() {
        std::cout << "=== Migration Risk Assessment ===\n";
        
        std::cout << "High Risk Areas:\n";
        std::cout << "• Multi-threaded code with shared ownership\n";
        std::cout << "• Performance-critical hot paths\n";
        std::cout << "• Public APIs used by external clients\n";
        std::cout << "• Complex circular reference scenarios\n";
        
        std::cout << "\nMedium Risk Areas:\n";
        std::cout << "• Custom deleter usage\n";
        std::cout << "• Mixed Boost/std smart pointer usage\n";
        std::cout << "• Template code with smart pointer parameters\n";
        std::cout << "• Exception safety critical sections\n";
        
        std::cout << "\nLow Risk Areas:\n";
        std::cout << "• Simple RAII usage\n";
        std::cout << "• Single-threaded components\n";
        std::cout << "• Internal implementation details\n";
        std::cout << "• Recently written code\n";
        
        std::cout << "\nMitigation Strategies:\n";
        std::cout << "• Extensive automated testing\n";
        std::cout << "• Gradual rollout with rollback capability\n";
        std::cout << "• Performance monitoring and alerting\n";
        std::cout << "• Code review focus on smart pointer usage\n";
        std::cout << "• Memory leak detection in CI/CD\n";
    }
};

void comprehensive_performance_analysis() {
    demonstrate_atomic_overhead();
    benchmark_shared_ptr_copies();
    demonstrate_cache_line_bouncing();
    analyze_memory_overhead();
    
    FragmentationDemo::demonstrate_fragmentation_impact();
    benchmark_allocation_speed(); 
    demonstrate_cache_locality();
    
    auto baseline = MigrationPerformanceAnalyzer::measureCurrentSystem();
    GradualMigrationPlanner::planMigration();
    GradualMigrationPlanner::assessMigrationRisks();
    
    // After migration:
    // MigrationPerformanceAnalyzer::compareAfterMigration(baseline);
}
```

## Best Practices

### Smart Pointer Selection Guidelines

Choosing the right smart pointer is crucial for both correctness and performance. Here's a comprehensive guide:

```cpp
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <memory>
#include <iostream>

// Decision flowchart implementation
class SmartPointerSelector {
public:
    enum class OwnershipType {
        EXCLUSIVE,      // Only one owner at a time
        SHARED,         // Multiple owners possible
        OBSERVED,       // Non-owning observation
        INTRUSIVE       // Object manages its own reference count
    };
    
    enum class PerformanceRequirement {
        NORMAL,         // Standard performance requirements
        HIGH,           // Performance-critical code
        MEMORY_LIMITED  // Memory usage is critical
    };
    
    enum class ThreadingModel {
        SINGLE_THREADED,
        MULTI_THREADED
    };
    
    static std::string recommendSmartPointer(
        OwnershipType ownership,
        PerformanceRequirement performance,
        ThreadingModel threading,
        bool legacy_codebase = false) {
        
        std::cout << "=== Smart Pointer Recommendation ===\n";
        
        // Decision logic
        if (ownership == OwnershipType::EXCLUSIVE) {
            if (legacy_codebase && performance == PerformanceRequirement::NORMAL) {
                return "boost::scoped_ptr - Simple RAII for legacy code";
            } else {
                return "std::unique_ptr - Modern exclusive ownership with move semantics";
            }
        }
        
        if (ownership == OwnershipType::SHARED) {
            if (performance == PerformanceRequirement::HIGH && 
                threading == ThreadingModel::SINGLE_THREADED) {
                return "Consider boost::intrusive_ptr - Lower overhead for performance-critical single-threaded code";
            } else if (legacy_codebase) {
                return "boost::shared_ptr - Shared ownership for legacy codebases";
            } else {
                return "std::shared_ptr - Standard shared ownership";
            }
        }
        
        if (ownership == OwnershipType::OBSERVED) {
            if (legacy_codebase) {
                return "boost::weak_ptr - Non-owning observation for legacy code";
            } else {
                return "std::weak_ptr - Standard non-owning observation";
            }
        }
        
        if (ownership == OwnershipType::INTRUSIVE) {
            return "boost::intrusive_ptr - When objects need to manage their own reference count";
        }
        
        return "std::unique_ptr - Safe default choice";
    }
};

void demonstrate_selection_guidelines() {
    std::cout << "Example scenarios:\n\n";
    
    // Scenario 1: File handle management
    std::cout << "Scenario 1: File handle management (exclusive ownership)\n";
    auto recommendation1 = SmartPointerSelector::recommendSmartPointer(
        SmartPointerSelector::OwnershipType::EXCLUSIVE,
        SmartPointerSelector::PerformanceRequirement::NORMAL,
        SmartPointerSelector::ThreadingModel::SINGLE_THREADED
    );
    std::cout << "Recommendation: " << recommendation1 << "\n\n";
    
    // Scenario 2: Shared cache object
    std::cout << "Scenario 2: Shared cache object (multiple owners)\n";
    auto recommendation2 = SmartPointerSelector::recommendSmartPointer(
        SmartPointerSelector::OwnershipType::SHARED,
        SmartPointerSelector::PerformanceRequirement::HIGH,
        SmartPointerSelector::ThreadingModel::MULTI_THREADED
    );
    std::cout << "Recommendation: " << recommendation2 << "\n\n";
    
    // Scenario 3: Observer pattern
    std::cout << "Scenario 3: Observer pattern (non-owning reference)\n";
    auto recommendation3 = SmartPointerSelector::recommendSmartPointer(
        SmartPointerSelector::OwnershipType::OBSERVED,
        SmartPointerSelector::PerformanceRequirement::NORMAL,
        SmartPointerSelector::ThreadingModel::MULTI_THREADED
    );
    std::cout << "Recommendation: " << recommendation3 << "\n\n";
}
```

### Prefer make_shared over new

Using `make_shared` and `make_unique` provides multiple benefits:

```cpp
#include <boost/make_shared.hpp>
#include <memory>
#include <chrono>

void demonstrate_make_shared_benefits() {
    std::cout << "=== make_shared Benefits ===\n";
    
    class TestObject {
    public:
        TestObject(int a, double b, const std::string& c) 
            : a_(a), b_(b), c_(c) {
            std::cout << "TestObject(" << a << ", " << b << ", \"" << c << "\") created\n";
        }
        
        ~TestObject() {
            std::cout << "TestObject destroyed\n";
        }
        
    private:
        int a_;
        double b_;
        std::string c_;
    };
    
    // Benefit 1: Single allocation for object and control block
    std::cout << "\n1. Memory allocation efficiency:\n";
    
    // Less efficient: separate allocations
    {
        std::cout << "Using new + shared_ptr constructor:\n";
        boost::shared_ptr<TestObject> ptr1(new TestObject(1, 2.5, "separate"));
        // This creates two separate allocations:
        // 1. new TestObject(...)
        // 2. Control block allocation inside shared_ptr constructor
    }
    
    // More efficient: single allocation
    {
        std::cout << "Using make_shared:\n";
        auto ptr2 = boost::make_shared<TestObject>(2, 3.5, "combined");
        // This creates one allocation containing both object and control block
    }
    
    // Benefit 2: Exception safety
    std::cout << "\n2. Exception safety:\n";
    
    auto potentially_throwing_function = []() -> int {
        // Could throw an exception
        return 42;
    };
    
    try {
        // DANGEROUS: Not exception safe
        // process_objects(boost::shared_ptr<TestObject>(new TestObject(1, 2.0, "unsafe")),
        //                potentially_throwing_function());
        // If potentially_throwing_function() throws AFTER new but BEFORE shared_ptr 
        // construction completes, we have a memory leak!
        
        // SAFE: Exception safe
        auto safe_ptr = boost::make_shared<TestObject>(3, 4.0, "safe");
        // process_objects(safe_ptr, potentially_throwing_function());
        // No risk of memory leak
        
    } catch (...) {
        std::cout << "Exception caught safely\n";
    }
    
    // Benefit 3: Performance comparison
    std::cout << "\n3. Performance comparison:\n";
    
    const int NUM_OBJECTS = 100000;
    
    // Measure new + shared_ptr
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        boost::shared_ptr<TestObject> ptr(new TestObject(i, i * 1.5, "new"));
        // Use the object briefly
        (void)ptr;
    }
    
    auto new_time = std::chrono::high_resolution_clock::now() - start;
    
    // Measure make_shared
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        auto ptr = boost::make_shared<TestObject>(i, i * 1.5, "make");
        // Use the object briefly
        (void)ptr;
    }
    
    auto make_time = std::chrono::high_resolution_clock::now() - start;
    
    auto new_ms = std::chrono::duration_cast<std::chrono::milliseconds>(new_time).count();
    auto make_ms = std::chrono::duration_cast<std::chrono::milliseconds>(make_time).count();
    
    std::cout << "new + shared_ptr: " << new_ms << " ms\n";
    std::cout << "make_shared: " << make_ms << " ms\n";
    std::cout << "Performance improvement: " << (double)new_ms / make_ms << "x\n";
    
    // Benefit 4: Cache locality
    std::cout << "\n4. Cache locality benefits:\n";
    std::cout << "make_shared: Object and control block adjacent in memory\n";
    std::cout << "new + shared_ptr: Object and control block may be far apart\n";
    std::cout << "Result: Better cache performance with make_shared\n";
}

// Advanced make_shared patterns
void demonstrate_advanced_make_shared_patterns() {
    std::cout << "\n=== Advanced make_shared Patterns ===\n";
    
    // Pattern 1: Factory functions
    template<typename T, typename... Args>
    std::shared_ptr<T> create_shared(Args&&... args) {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }
    
    // Pattern 2: Conditional creation
    auto create_test_object_conditionally = [](bool should_create) -> std::shared_ptr<TestObject> {
        if (should_create) {
            return std::make_shared<TestObject>(1, 2.0, "conditional");
        }
        return nullptr;  // Return null shared_ptr
    };
    
    // Pattern 3: Array creation (C++20 and later, or Boost equivalent)
    // Note: Boost doesn't have make_shared for arrays, use carefully
    
    std::cout << "Factory pattern with perfect forwarding works correctly\n";
    std::cout << "Conditional creation avoids unnecessary allocations\n";
}
```

### Use weak_ptr to break cycles

Circular references are a common source of memory leaks with shared_ptr:

```cpp
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <vector>
#include <iostream>

// Example: Tree data structure with parent-child relationships
class TreeNode {
public:
    TreeNode(const std::string& name) : name_(name) {
        std::cout << "TreeNode '" << name_ << "' created\n";
    }
    
    ~TreeNode() {
        std::cout << "TreeNode '" << name_ << "' destroyed\n";
    }
    
    void addChild(boost::shared_ptr<TreeNode> child) {
        children_.push_back(child);
        child->parent_ = shared_from_this();  // Set weak reference to parent
    }
    
    void removeChild(const std::string& child_name) {
        children_.erase(
            std::remove_if(children_.begin(), children_.end(),
                [&child_name](const boost::shared_ptr<TreeNode>& child) {
                    return child->name_ == child_name;
                }),
            children_.end());
    }
    
    boost::shared_ptr<TreeNode> getParent() const {
        return parent_.lock();  // Convert weak_ptr to shared_ptr safely
    }
    
    const std::vector<boost::shared_ptr<TreeNode>>& getChildren() const {
        return children_;
    }
    
    const std::string& getName() const { return name_; }
    
    void printTree(int depth = 0) const {
        for (int i = 0; i < depth; ++i) std::cout << "  ";
        std::cout << name_ << " (refs: " << shared_from_this().use_count() << ")\n";
        
        for (const auto& child : children_) {
            child->printTree(depth + 1);
        }
    }
    
    // Enable shared_from_this
    boost::shared_ptr<TreeNode> shared_from_this() const {
        // In real code, inherit from boost::enable_shared_from_this<TreeNode>
        // This is a simplified version for demonstration
        return boost::shared_ptr<TreeNode>(const_cast<TreeNode*>(this), [](TreeNode*){});
    }
    
private:
    std::string name_;
    std::vector<boost::shared_ptr<TreeNode>> children_;  // Strong references to children
    boost::weak_ptr<TreeNode> parent_;                   // Weak reference to parent (breaks cycle)
};

void demonstrate_circular_reference_prevention() {
    std::cout << "=== Circular Reference Prevention ===\n";
    
    {
        // Create tree structure
        auto root = boost::make_shared<TreeNode>("root");
        auto child1 = boost::make_shared<TreeNode>("child1");
        auto child2 = boost::make_shared<TreeNode>("child2");
        auto grandchild = boost::make_shared<TreeNode>("grandchild");
        
        // Build tree
        root->addChild(child1);
        root->addChild(child2);
        child1->addChild(grandchild);
        
        std::cout << "Tree structure:\n";
        root->printTree();
        
        // Demonstrate parent access via weak_ptr
        if (auto parent = grandchild->getParent()) {
            std::cout << "Grandchild's parent: " << parent->getName() << "\n";
        }
        
        // Remove a child
        root->removeChild("child2");
        std::cout << "\nAfter removing child2:\n";
        root->printTree();
        
        std::cout << "\nReference counts before scope exit:\n";
        std::cout << "root: " << root.use_count() << "\n";
        std::cout << "child1: " << child1.use_count() << "\n";
        std::cout << "grandchild: " << grandchild.use_count() << "\n";
        
    } // All objects should be properly destroyed here
    
    std::cout << "All tree nodes should be destroyed now\n";
}

// Observer pattern with weak_ptr
template<typename EventType>
class Observable {
public:
    class Observer {
    public:
        virtual ~Observer() = default;
        virtual void onEvent(const EventType& event) = 0;
    };
    
    void addObserver(boost::shared_ptr<Observer> observer) {
        observers_.push_back(observer);  // Store as weak_ptr
    }
    
    void removeObserver(boost::shared_ptr<Observer> observer) {
        observers_.erase(
            std::remove_if(observers_.begin(), observers_.end(),
                [&observer](const boost::weak_ptr<Observer>& weak_obs) {
                    auto locked = weak_obs.lock();
                    return !locked || locked == observer;
                }),
            observers_.end());
    }
    
    void notifyObservers(const EventType& event) {
        // Clean up expired observers while notifying
        auto it = observers_.begin();
        while (it != observers_.end()) {
            if (auto observer = it->lock()) {
                observer->onEvent(event);
                ++it;
            } else {
                // Observer has been destroyed, remove from list
                it = observers_.erase(it);
            }
        }
    }
    
    size_t getObserverCount() const {
        // Count only active observers
        size_t count = 0;
        for (const auto& weak_obs : observers_) {
            if (!weak_obs.expired()) {
                ++count;
            }
        }
        return count;
    }
    
private:
    std::vector<boost::weak_ptr<Observer>> observers_;
};

class EventLogger : public Observable<std::string>::Observer {
public:
    EventLogger(const std::string& name) : name_(name) {
        std::cout << "EventLogger '" << name_ << "' created\n";
    }
    
    ~EventLogger() {
        std::cout << "EventLogger '" << name_ << "' destroyed\n";
    }
    
    void onEvent(const std::string& event) override {
        std::cout << "Logger '" << name_ << "' received event: " << event << "\n";
    }
    
private:
    std::string name_;
};

void demonstrate_observer_pattern() {
    std::cout << "\n=== Observer Pattern with weak_ptr ===\n";
    
    Observable<std::string> event_source;
    
    {
        auto logger1 = boost::make_shared<EventLogger>("Logger1");
        auto logger2 = boost::make_shared<EventLogger>("Logger2");
        
        event_source.addObserver(logger1);
        event_source.addObserver(logger2);
        
        std::cout << "Active observers: " << event_source.getObserverCount() << "\n";
        
        event_source.notifyObservers("First event");
        
        // logger2 goes out of scope here
    }
    
    std::cout << "After logger2 destroyed, active observers: " 
              << event_source.getObserverCount() << "\n";
    
    event_source.notifyObservers("Second event");
    // Should automatically clean up expired observer
    
    std::cout << "After cleanup, active observers: " 
              << event_source.getObserverCount() << "\n";
}
```

### Choose the right smart pointer

Decision matrix for smart pointer selection:

```cpp
void comprehensive_selection_guide() {
    std::cout << "\n=== Comprehensive Smart Pointer Selection Guide ===\n";
    
    std::cout << "\n📋 Decision Matrix:\n";
    std::cout << "┌─────────────────────┬─────────────────┬─────────────────┬─────────────────┐\n";
    std::cout << "│ Use Case            │ Recommended     │ Alternative     │ Avoid           │\n";
    std::cout << "├─────────────────────┼─────────────────┼─────────────────┼─────────────────┤\n";
    std::cout << "│ Exclusive ownership │ std::unique_ptr │ boost::scoped_ptr│ shared_ptr      │\n";
    std::cout << "│ Shared ownership    │ std::shared_ptr │ boost::shared_ptr│ raw pointers    │\n";
    std::cout << "│ Non-owning observer │ std::weak_ptr   │ boost::weak_ptr │ raw pointers    │\n";
    std::cout << "│ Performance critical│ intrusive_ptr   │ unique_ptr      │ shared_ptr      │\n";
    std::cout << "│ Legacy codebase     │ boost::*_ptr    │ gradual migration│ mixing types   │\n";
    std::cout << "│ Arrays              │ unique_ptr<T[]> │ vector<T>       │ scoped_ptr     │\n";
    std::cout << "│ Custom deleters     │ unique_ptr      │ shared_ptr      │ manual delete   │\n";
    std::cout << "└─────────────────────┴─────────────────┴─────────────────┴─────────────────┘\n";
    
    std::cout << "\n⚡ Performance Guidelines:\n";
    std::cout << "• unique_ptr: Zero overhead abstraction\n";
    std::cout << "• shared_ptr: Reference counting overhead\n";
    std::cout << "• intrusive_ptr: Lower overhead than shared_ptr\n";
    std::cout << "• weak_ptr: Similar to shared_ptr but doesn't affect lifetime\n";
    
    std::cout << "\n🧵 Thread Safety Guidelines:\n";
    std::cout << "• Reference counting is thread-safe\n";
    std::cout << "• Object access requires synchronization\n";
    std::cout << "• Prefer thread-local ownership when possible\n";
    std::cout << "• Use atomic operations for high-contention scenarios\n";
    
    std::cout << "\n💾 Memory Guidelines:\n";
    std::cout << "• Use make_shared for better memory efficiency\n";
    std::cout << "• Consider object pools for frequent allocation\n";
    std::cout << "• Use weak_ptr to break circular references\n";
    std::cout << "• Profile memory usage in realistic scenarios\n";
    
    std::cout << "\n🔧 Migration Guidelines:\n";
    std::cout << "• Start with scoped_ptr → unique_ptr (easiest)\n";
    std::cout << "• Plan shared_ptr migration carefully\n";
    std::cout << "• Test performance before and after\n";
    std::cout << "• Keep documentation updated during migration\n";
}

// Best practices summary
void best_practices_summary() {
    std::cout << "\n=== Best Practices Summary ===\n";
    
    std::cout << "\n✅ DO:\n";
    std::cout << "• Use make_shared/make_unique for creation\n";
    std::cout << "• Use weak_ptr to break circular references\n";
    std::cout << "• Choose the most restrictive ownership model that works\n";
    std::cout << "• Use RAII consistently throughout your codebase\n";
    std::cout << "• Test with memory leak detectors\n";
    std::cout << "• Profile performance in realistic scenarios\n";
    std::cout << "• Document ownership semantics clearly\n";
    
    std::cout << "\n❌ DON'T:\n";
    std::cout << "• Mix raw pointers with smart pointers unnecessarily\n";
    std::cout << "• Create shared_ptr from the same raw pointer twice\n";
    std::cout << "• Use shared_ptr when unique_ptr would suffice\n";
    std::cout << "• Ignore circular reference possibilities\n";
    std::cout << "• Assume thread safety beyond reference counting\n";
    std::cout << "• Rush migration without proper testing\n";
    std::cout << "• Forget about custom deleter requirements\n";
    
    std::cout << "\n🎯 Key Takeaways:\n";
    std::cout << "• Smart pointers are about ownership, not just memory management\n";
    std::cout << "• The 'smartest' pointer is often the simplest one that works\n";
    std::cout << "• Performance matters, but correctness matters more\n";
    std::cout << "• Good design reduces the need for shared ownership\n";
    std::cout << "• Migration should be gradual and well-tested\n";
}

void demonstrate_all_best_practices() {
    demonstrate_selection_guidelines();
    demonstrate_make_shared_benefits();
    demonstrate_advanced_make_shared_patterns();
    demonstrate_circular_reference_prevention();
    demonstrate_observer_pattern();
    comprehensive_selection_guide();
    best_practices_summary();
}
```

## Assessment

### Knowledge Assessment Checklist

Before proceeding to the next section, ensure you can demonstrate the following competencies:

#### Conceptual Understanding ✅

**Smart Pointer Fundamentals:**
□ Explain the difference between stack and heap memory management  
□ Describe the RAII principle and its importance  
□ Identify memory management problems that smart pointers solve  
□ Understand the concept of ownership in C++  

**Reference Counting:**
□ Explain how reference counting works  
□ Understand thread safety implications of atomic reference counting  
□ Identify scenarios where reference counting causes performance issues  
□ Recognize circular reference problems and solutions  

**Memory Pool Concepts:**
□ Understand when object pools improve performance  
□ Explain memory fragmentation and how pools help  
□ Describe cache locality benefits of pooled allocation  
□ Know when pools might hurt rather than help performance  

#### Practical Implementation Skills ✅

**boost::shared_ptr Mastery:**
□ Create shared_ptr using both constructor and make_shared  
□ Implement custom deleters for different resource types  
□ Use shared_ptr safely in multi-threaded environments  
□ Convert between shared_ptr and weak_ptr appropriately  

**boost::scoped_ptr Usage:**
□ Implement RAII patterns using scoped_ptr  
□ Understand the limitations compared to unique_ptr  
□ Handle exceptions safely with scoped_ptr  
□ Know when to use scoped_ptr vs alternatives  

**boost::intrusive_ptr Implementation:**
□ Create classes with intrusive reference counting  
□ Implement thread-safe reference counting  
□ Compare performance with shared_ptr  
□ Use intrusive_ptr appropriately for performance-critical code  

**Object Pool Implementation:**
□ Design and implement custom object pools  
□ Create thread-safe pool implementations  
□ Measure and optimize pool performance  
□ Integrate pools with RAII patterns  

#### Performance Analysis Skills ✅

**Benchmarking Abilities:**
□ Design meaningful performance tests for smart pointers  
□ Measure allocation/deallocation performance  
□ Analyze memory usage patterns  
□ Identify performance bottlenecks in smart pointer usage  

**Optimization Techniques:**
□ Choose appropriate smart pointer for performance requirements  
□ Minimize reference counting overhead  
□ Optimize for cache locality  
□ Reduce memory fragmentation  

#### Problem-Solving Capabilities ✅

**Debugging Skills:**
□ Identify memory leaks caused by circular references  
□ Debug smart pointer-related crashes  
□ Use debugging tools effectively with smart pointers  
□ Trace object lifetime issues  

**Design Skills:**
□ Design ownership hierarchies to minimize shared ownership  
□ Implement observer patterns without circular references  
□ Create exception-safe code with smart pointers  
□ Design APIs that express ownership semantics clearly  

### Practical Assessment Tasks

Complete these tasks to validate your understanding:

#### Task 1: Smart Pointer Comparison (30 minutes)
```cpp
// Implement the same resource management using different smart pointers
// Compare: raw pointer, boost::scoped_ptr, boost::shared_ptr, std::unique_ptr
// Measure: performance, memory usage, code complexity
class ResourceManager {
    // TODO: Implement with different smart pointer types
    // Show trade-offs of each approach
};
```

#### Task 2: Circular Reference Resolution (20 minutes)
```cpp
// Fix this circular reference problem
class Parent {
    boost::shared_ptr<Child> child_;
public:
    void setChild(boost::shared_ptr<Child> child) { child_ = child; }
};

class Child {
    boost::shared_ptr<Parent> parent_;  // Creates cycle!
public:
    void setParent(boost::shared_ptr<Parent> parent) { parent_ = parent; }
};

// TODO: Fix the circular reference while maintaining functionality
```

#### Task 3: Thread-Safe Object Pool (45 minutes)
```cpp
// Implement a thread-safe object pool with these requirements:
// - Support concurrent allocation/deallocation
// - Automatic pool size adjustment based on usage
// - Exception safety guarantees
// - Performance monitoring capabilities

template<typename T>
class ThreadSafeObjectPool {
    // TODO: Complete implementation
};
```

#### Task 4: Migration Planning (25 minutes)
```cpp
// Create a migration plan for this legacy codebase
class LegacySystem {
    Database* database_;               // Raw pointer
    boost::shared_ptr<Config> config_; // Boost smart pointer
    std::vector<Component*> components_; // Raw pointer vector
    
    // TODO: Plan migration to modern smart pointers
    // Consider: dependencies, testing, performance, compatibility
};
```

### Assessment Criteria

**Excellent (90-100%)**
- Demonstrates deep understanding of all smart pointer types
- Implements efficient, thread-safe solutions
- Shows mastery of performance optimization techniques
- Designs clean, maintainable APIs with clear ownership semantics
- Provides insightful analysis of trade-offs

**Proficient (70-89%)**
- Shows solid understanding of smart pointer concepts
- Implements correct but not necessarily optimal solutions
- Understands basic performance implications
- Creates functional code with minor design issues
- Identifies most important trade-offs

**Developing (50-69%)**
- Basic understanding of smart pointer usage
- Implements simple scenarios correctly
- Some confusion about ownership semantics
- Limited performance optimization awareness
- Needs guidance on complex scenarios

**Needs Improvement (<50%)**
- Fundamental gaps in understanding
- Incorrect implementations
- Cannot identify appropriate smart pointer for scenario
- Poor understanding of ownership and lifetime management
- Requires significant additional study

### Advanced Challenges (Optional)

For those seeking deeper mastery:

#### Challenge 1: Custom Allocator Integration
Implement smart pointers that work with custom allocators for specialized memory management scenarios.

#### Challenge 2: Lock-Free Smart Pointer Operations
Design lock-free algorithms for smart pointer operations in high-concurrency scenarios.

#### Challenge 3: Serialization Support
Add serialization capabilities to smart pointer-managed objects while maintaining ownership semantics.

#### Challenge 4: Cross-Language Interoperability
Design smart pointer wrappers that work safely across language boundaries (C++/C, C++/Python, etc.).

### Study Resources for Gaps

If you identify knowledge gaps during assessment:

**Books:**
- "Effective C++" by Scott Meyers - Items on resource management
- "C++ Concurrency in Action" by Anthony Williams - Thread safety
- "Optimized C++" by Kurt Guntheroth - Performance optimization

**Online Resources:**
- Boost.SmartPtr documentation and examples
- CppReference articles on memory management
- Performance analysis tutorials and tools

**Practice Opportunities:**
- Contribute to open-source projects using Boost
- Implement toy problems with different smart pointer approaches
- Profile real applications for smart pointer performance issues

### Self-Reflection Questions

After completing the assessment:

1. **Which smart pointer concepts do you find most challenging?**
2. **In what scenarios would you choose Boost over std smart pointers?**
3. **How has your understanding of ownership semantics evolved?**
4. **What performance insights surprised you the most?**
5. **How will you apply these concepts in your current projects?**

### Readiness Indicators

You're ready to proceed to the next section when you can:

✅ **Confidently choose** the appropriate smart pointer for any given scenario  
✅ **Implement robust** memory management solutions using smart pointers  
✅ **Debug and optimize** smart pointer-related performance issues  
✅ **Design clean APIs** that express ownership semantics clearly  
✅ **Migrate legacy code** safely from raw pointers to smart pointers  

Remember: Mastery comes through practice. Apply these concepts in real projects to solidify your understanding!

## Next Steps

Move on to [Containers and Data Structures](03_Containers_Data_Structures.md) to explore Boost's container libraries.
