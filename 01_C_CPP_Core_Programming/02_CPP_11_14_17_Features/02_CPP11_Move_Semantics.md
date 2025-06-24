# C++11 Move Semantics

## Overview

Move semantics is one of the most important features introduced in C++11. It allows objects to transfer ownership of their resources instead of copying them, leading to significant performance improvements for resource-heavy operations.

## Key Concepts

### 1. Rvalue References

Rvalue references (`T&&`) bind to temporary objects and enable move semantics.

#### Understanding Lvalues and Rvalues

```cpp
#include <iostream>
#include <string>
#include <vector>

void demonstrate_references() {
    std::string str = "Hello";
    
    // Lvalue references (traditional references)
    std::string& lref = str;           // OK: str is an lvalue
    // std::string& lref2 = "World";   // Error: "World" is an rvalue
    
    // Rvalue references (C++11)
    std::string&& rref = "World";      // OK: "World" is an rvalue
    std::string&& rref2 = std::string("C++"); // OK: temporary object
    // std::string&& rref3 = str;     // Error: str is an lvalue
    
    // std::move converts lvalue to rvalue
    std::string&& rref3 = std::move(str); // OK: explicit conversion
    
    std::cout << "lref: " << lref << std::endl;
    std::cout << "rref: " << rref << std::endl;
    std::cout << "rref2: " << rref2 << std::endl;
    std::cout << "rref3: " << rref3 << std::endl;
    
    // Note: str is now in a valid but unspecified state
    std::cout << "str after move: '" << str << "'" << std::endl;
}

// Function overloading with lvalue and rvalue references
void process(const std::string& str) {
    std::cout << "Processing lvalue: " << str << std::endl;
}

void process(std::string&& str) {
    std::cout << "Processing rvalue: " << str << std::endl;
    // Can modify str here if needed
    str += " (modified)";
    std::cout << "Modified rvalue: " << str << std::endl;
}

int main() {
    demonstrate_references();
    
    std::string name = "Alice";
    process(name);                    // Calls lvalue version
    process("Bob");                   // Calls rvalue version
    process(std::move(name));         // Calls rvalue version (name is moved)
    
    return 0;
}
```

### 2. Move Constructors and Move Assignment Operators

Classes can define special member functions to handle move operations efficiently.

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <chrono>

class ResourceManager {
private:
    std::string name;
    std::vector<int> data;
    std::unique_ptr<int[]> buffer;
    size_t buffer_size;
    
public:
    // Constructor
    ResourceManager(const std::string& name, size_t size) 
        : name(name), buffer_size(size) {
        std::cout << "Constructor: " << name << " (size: " << size << ")" << std::endl;
        
        // Allocate some data
        data.resize(size, 42);
        buffer = std::make_unique<int[]>(size);
        for (size_t i = 0; i < size; ++i) {
            buffer[i] = static_cast<int>(i);
        }
    }
    
    // Copy constructor
    ResourceManager(const ResourceManager& other) 
        : name(other.name + "_copy"), 
          data(other.data), 
          buffer_size(other.buffer_size) {
        std::cout << "Copy constructor: " << name << std::endl;
        
        // Deep copy of buffer
        buffer = std::make_unique<int[]>(buffer_size);
        for (size_t i = 0; i < buffer_size; ++i) {
            buffer[i] = other.buffer[i];
        }
    }
    
    // Move constructor
    ResourceManager(ResourceManager&& other) noexcept
        : name(std::move(other.name)),
          data(std::move(other.data)),
          buffer(std::move(other.buffer)),
          buffer_size(other.buffer_size) {
        std::cout << "Move constructor: " << name << std::endl;
        
        // Reset moved-from object
        other.buffer_size = 0;
    }
    
    // Copy assignment operator
    ResourceManager& operator=(const ResourceManager& other) {
        std::cout << "Copy assignment: " << name << " = " << other.name << std::endl;
        
        if (this != &other) {
            name = other.name + "_assigned";
            data = other.data;
            buffer_size = other.buffer_size;
            
            // Deep copy of buffer
            buffer = std::make_unique<int[]>(buffer_size);
            for (size_t i = 0; i < buffer_size; ++i) {
                buffer[i] = other.buffer[i];
            }
        }
        return *this;
    }
    
    // Move assignment operator
    ResourceManager& operator=(ResourceManager&& other) noexcept {
        std::cout << "Move assignment: " << name << " = " << other.name << std::endl;
        
        if (this != &other) {
            // Move resources
            name = std::move(other.name);
            data = std::move(other.data);
            buffer = std::move(other.buffer);
            buffer_size = other.buffer_size;
            
            // Reset moved-from object
            other.buffer_size = 0;
        }
        return *this;
    }
    
    // Destructor
    ~ResourceManager() {
        std::cout << "Destructor: " << name << std::endl;
    }
    
    // Utility functions
    const std::string& get_name() const { return name; }
    size_t get_size() const { return buffer_size; }
    
    void display_info() const {
        std::cout << "ResourceManager '" << name 
                  << "' - size: " << buffer_size 
                  << ", data elements: " << data.size() << std::endl;
    }
};

// Factory function that returns by value
ResourceManager create_resource(const std::string& name, size_t size) {
    return ResourceManager(name, size);
}

void demonstrate_move_semantics() {
    std::cout << "\n=== Move Semantics Demonstration ===" << std::endl;
    
    // 1. Constructor
    std::cout << "\n1. Creating original resource:" << std::endl;
    ResourceManager rm1("Original", 1000);
    rm1.display_info();
    
    // 2. Copy constructor
    std::cout << "\n2. Copy constructor:" << std::endl;
    ResourceManager rm2 = rm1;  // Copy constructor called
    rm2.display_info();
    
    // 3. Move constructor
    std::cout << "\n3. Move constructor:" << std::endl;
    ResourceManager rm3 = std::move(rm1);  // Move constructor called
    rm3.display_info();
    std::cout << "rm1 after move: ";
    rm1.display_info();  // rm1 is in valid but unspecified state
    
    // 4. Copy assignment
    std::cout << "\n4. Copy assignment:" << std::endl;
    ResourceManager rm4("Target", 100);
    rm4 = rm2;  // Copy assignment called
    rm4.display_info();
    
    // 5. Move assignment
    std::cout << "\n5. Move assignment:" << std::endl;
    ResourceManager rm5("AnotherTarget", 200);
    rm5 = std::move(rm3);  // Move assignment called
    rm5.display_info();
    
    // 6. Return value optimization and move
    std::cout << "\n6. Factory function (RVO/move):" << std::endl;
    ResourceManager rm6 = create_resource("Factory", 500);
    rm6.display_info();
    
    std::cout << "\n=== End of scope - destructors will be called ===" << std::endl;
}

int main() {
    demonstrate_move_semantics();
    return 0;
}
```

### 3. std::move and std::forward

Standard library utilities for working with move semantics.

#### std::move

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <utility>

void demonstrate_std_move() {
    std::cout << "\n=== std::move Demonstration ===" << std::endl;
    
    // Basic std::move usage
    std::string source = "Hello, World!";
    std::string destination = std::move(source);
    
    std::cout << "After move:" << std::endl;
    std::cout << "destination: '" << destination << "'" << std::endl;
    std::cout << "source: '" << source << "'" << std::endl;  // Valid but unspecified
    
    // Moving containers
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::cout << "vec1 size before move: " << vec1.size() << std::endl;
    
    std::vector<int> vec2 = std::move(vec1);
    std::cout << "vec2 size after move: " << vec2.size() << std::endl;
    std::cout << "vec1 size after move: " << vec1.size() << std::endl;
    
    // Moving in function calls
    auto process_string = [](std::string str) {
        return str + " processed";
    };
    
    std::string input = "Data";
    std::string result = process_string(std::move(input));
    std::cout << "Result: " << result << std::endl;
    std::cout << "Input after move: '" << input << "'" << std::endl;
}

// Example: Moving elements in containers
void demonstrate_container_moves() {
    std::cout << "\n=== Container Move Operations ===" << std::endl;
    
    std::vector<std::string> strings = {"Apple", "Banana", "Cherry"};
    std::vector<std::string> moved_strings;
    
    // Move elements from one container to another
    for (auto& str : strings) {
        moved_strings.push_back(std::move(str));
    }
    
    std::cout << "Original strings after move:" << std::endl;
    for (const auto& str : strings) {
        std::cout << "  '" << str << "'" << std::endl;
    }
    
    std::cout << "Moved strings:" << std::endl;
    for (const auto& str : moved_strings) {
        std::cout << "  '" << str << "'" << std::endl;
    }
}
```

#### std::forward (Perfect Forwarding)

```cpp
#include <iostream>
#include <string>
#include <utility>
#include <memory>

// Example class for demonstration
class Widget {
private:
    std::string name;
    int value;
    
public:
    Widget(const std::string& n, int v) : name(n), value(v) {
        std::cout << "Widget constructor (copy): " << name << std::endl;
    }
    
    Widget(std::string&& n, int v) : name(std::move(n)), value(v) {
        std::cout << "Widget constructor (move): " << name << std::endl;
    }
    
    Widget(const Widget& other) : name(other.name), value(other.value) {
        std::cout << "Widget copy constructor: " << name << std::endl;
    }
    
    Widget(Widget&& other) noexcept : name(std::move(other.name)), value(other.value) {
        std::cout << "Widget move constructor: " << name << std::endl;
    }
    
    void display() const {
        std::cout << "Widget: " << name << " (" << value << ")" << std::endl;
    }
};

// Factory function with perfect forwarding
template<typename T, typename... Args>
std::unique_ptr<T> make_unique_perfect(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

// Wrapper function demonstrating perfect forwarding
template<typename T>
void process_widget(T&& widget) {
    std::cout << "Processing widget..." << std::endl;
    
    // Forward the widget to another function
    auto ptr = make_unique_perfect<Widget>(std::forward<T>(widget), 100);
    ptr->display();
}

void demonstrate_perfect_forwarding() {
    std::cout << "\n=== Perfect Forwarding Demonstration ===" << std::endl;
    
    // Test with lvalue
    std::string name1 = "Lvalue Widget";
    std::cout << "\nForwarding lvalue:" << std::endl;
    process_widget(name1);
    
    // Test with rvalue
    std::cout << "\nForwarding rvalue:" << std::endl;
    process_widget(std::string("Rvalue Widget"));
    
    // Test with temporary
    std::cout << "\nForwarding temporary:" << std::endl;
    process_widget("Temporary Widget");
}

// Universal reference vs rvalue reference
template<typename T>
void universal_reference(T&& param) {
    // T&& is a universal reference here (depends on T)
    std::cout << "Universal reference called" << std::endl;
}

void rvalue_reference(std::string&& param) {
    // std::string&& is specifically an rvalue reference
    std::cout << "Rvalue reference called" << std::endl;
}

int main() {
    demonstrate_std_move();
    demonstrate_container_moves();
    demonstrate_perfect_forwarding();
    
    // Universal vs rvalue reference
    std::cout << "\n=== Universal vs Rvalue Reference ===" << std::endl;
    std::string str = "test";
    universal_reference(str);           // T deduced as std::string&
    universal_reference("literal");     // T deduced as const char*
    universal_reference(std::move(str)); // T deduced as std::string
    
    // rvalue_reference(str);           // Error: won't compile
    rvalue_reference(std::move(str));   // OK
    
    return 0;
}
```

### 4. Return Value Optimization (RVO) and Named Return Value Optimization (NRVO)

Compiler optimizations that eliminate unnecessary copies when returning objects.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

class ExpensiveObject {
private:
    std::vector<int> data;
    std::string name;
    
public:
    ExpensiveObject(const std::string& n, size_t size) : name(n), data(size, 42) {
        std::cout << "ExpensiveObject constructor: " << name 
                  << " (size: " << size << ")" << std::endl;
    }
    
    ExpensiveObject(const ExpensiveObject& other) 
        : data(other.data), name(other.name + "_copy") {
        std::cout << "ExpensiveObject copy constructor: " << name << std::endl;
    }
    
    ExpensiveObject(ExpensiveObject&& other) noexcept
        : data(std::move(other.data)), name(std::move(other.name)) {
        std::cout << "ExpensiveObject move constructor: " << name << std::endl;
    }
    
    ExpensiveObject& operator=(const ExpensiveObject& other) {
        if (this != &other) {
            data = other.data;
            name = other.name + "_assigned";
            std::cout << "ExpensiveObject copy assignment: " << name << std::endl;
        }
        return *this;
    }
    
    ExpensiveObject& operator=(ExpensiveObject&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            name = std::move(other.name);
            std::cout << "ExpensiveObject move assignment: " << name << std::endl;
        }
        return *this;
    }
    
    ~ExpensiveObject() {
        std::cout << "ExpensiveObject destructor: " << name << std::endl;
    }
    
    const std::string& get_name() const { return name; }
    size_t get_size() const { return data.size(); }
};

// RVO: Return Value Optimization
ExpensiveObject create_object_rvo(const std::string& name) {
    std::cout << "Creating object with RVO..." << std::endl;
    return ExpensiveObject(name, 1000);  // RVO: no copy/move constructor called
}

// NRVO: Named Return Value Optimization
ExpensiveObject create_object_nrvo(const std::string& name) {
    std::cout << "Creating object with NRVO..." << std::endl;
    ExpensiveObject obj(name, 1000);
    // Do some work with obj...
    return obj;  // NRVO: compiler may optimize away copy/move
}

// No optimization possible (multiple return paths)
ExpensiveObject create_object_no_optimization(const std::string& name, bool condition) {
    std::cout << "Creating object without optimization..." << std::endl;
    if (condition) {
        ExpensiveObject obj1(name + "_true", 500);
        return obj1;  // Move constructor will be called
    } else {
        ExpensiveObject obj2(name + "_false", 1500);
        return obj2;  // Move constructor will be called
    }
}

void demonstrate_rvo() {
    std::cout << "\n=== RVO Demonstration ===" << std::endl;
    
    std::cout << "\n1. RVO (Return Value Optimization):" << std::endl;
    ExpensiveObject obj1 = create_object_rvo("RVO_Object");
    std::cout << "Created: " << obj1.get_name() << std::endl;
    
    std::cout << "\n2. NRVO (Named Return Value Optimization):" << std::endl;
    ExpensiveObject obj2 = create_object_nrvo("NRVO_Object");
    std::cout << "Created: " << obj2.get_name() << std::endl;
    
    std::cout << "\n3. No optimization (multiple return paths):" << std::endl;
    ExpensiveObject obj3 = create_object_no_optimization("No_Opt", true);
    std::cout << "Created: " << obj3.get_name() << std::endl;
    
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create many objects to see the performance difference
    std::vector<ExpensiveObject> objects;
    objects.reserve(1000);
    
    for (int i = 0; i < 1000; ++i) {
        objects.push_back(create_object_rvo("Perf_" + std::to_string(i)));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Created 1000 objects in " << duration.count() << " microseconds" << std::endl;
}

// Demonstrating when RVO doesn't apply
ExpensiveObject conditional_return(bool use_first) {
    ExpensiveObject first("First", 100);
    ExpensiveObject second("Second", 200);
    
    // RVO cannot be applied here because compiler doesn't know
    // which object will be returned
    return use_first ? first : second;  // Move constructor will be called
}

int main() {
    demonstrate_rvo();
    
    std::cout << "\n=== Conditional Return (No RVO) ===" << std::endl;
    ExpensiveObject result = conditional_return(true);
    std::cout << "Result: " << result.get_name() << std::endl;
    
    return 0;
}
```

## Performance Benefits

Move semantics can provide significant performance improvements:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>

// Performance comparison: Copy vs Move
void performance_comparison() {
    const size_t num_operations = 100000;
    const size_t string_size = 1000;
    
    // Test with large strings
    std::vector<std::string> source_strings;
    source_strings.reserve(num_operations);
    
    for (size_t i = 0; i < num_operations; ++i) {
        source_strings.emplace_back(string_size, 'A' + (i % 26));
    }
    
    std::cout << "Performance comparison with " << num_operations 
              << " strings of size " << string_size << std::endl;
    
    // Copy performance
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::string> copied_strings;
    copied_strings.reserve(num_operations);
    
    for (const auto& str : source_strings) {
        copied_strings.push_back(str);  // Copy
    }
    
    auto copy_end = std::chrono::high_resolution_clock::now();
    auto copy_duration = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - start);
    
    // Reset source strings
    source_strings.clear();
    for (size_t i = 0; i < num_operations; ++i) {
        source_strings.emplace_back(string_size, 'A' + (i % 26));
    }
    
    // Move performance
    auto move_start = std::chrono::high_resolution_clock::now();
    std::vector<std::string> moved_strings;
    moved_strings.reserve(num_operations);
    
    for (auto& str : source_strings) {
        moved_strings.push_back(std::move(str));  // Move
    }
    
    auto move_end = std::chrono::high_resolution_clock::now();
    auto move_duration = std::chrono::duration_cast<std::chrono::milliseconds>(move_end - move_start);
    
    std::cout << "Copy time: " << copy_duration.count() << " ms" << std::endl;
    std::cout << "Move time: " << move_duration.count() << " ms" << std::endl;
    std::cout << "Speedup: " << static_cast<double>(copy_duration.count()) / move_duration.count() 
              << "x" << std::endl;
}

int main() {
    performance_comparison();
    return 0;
}
```

## Best Practices

### 1. When to Use Move Semantics

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>

class GoodMoveExample {
private:
    std::string name;
    std::vector<int> data;
    std::unique_ptr<int[]> buffer;
    
public:
    // Always mark move constructors and move assignment operators as noexcept
    GoodMoveExample(GoodMoveExample&& other) noexcept
        : name(std::move(other.name)),
          data(std::move(other.data)),
          buffer(std::move(other.buffer)) {
        // Move operations should not throw
    }
    
    GoodMoveExample& operator=(GoodMoveExample&& other) noexcept {
        if (this != &other) {
            name = std::move(other.name);
            data = std::move(other.data);
            buffer = std::move(other.buffer);
        } 
        return *this;
    }
    
    // Implement swap for exception-safe operations
    void swap(GoodMoveExample& other) noexcept {
        using std::swap;
        swap(name, other.name);
        swap(data, other.data);
        swap(buffer, other.buffer);
    }
};

// Free function swap
void swap(GoodMoveExample& lhs, GoodMoveExample& rhs) noexcept {
    lhs.swap(rhs);
}
```

### 2. Common Mistakes to Avoid

```cpp
#include <iostream>
#include <string>
#include <utility>

class BadMoveExample {
private:
    std::string data;
    
public:
    // BAD: Don't use moved-from objects
    void bad_usage() {
        std::string source = "Hello";
        std::string dest = std::move(source);
        
        // BAD: Using source after move
        // std::cout << source << std::endl;  // Undefined behavior!
        
        // GOOD: Check if you need to use the object after move
        if (!source.empty()) {  // This is safe - moved-from objects are valid
            std::cout << "Source still has data: " << source << std::endl;
        }
    }
    
    // BAD: Don't move const objects
    void bad_const_move() {
        const std::string const_str = "Cannot move";
        // std::string moved = std::move(const_str);  // This will copy, not move!
    }
    
    // BAD: Don't move local objects being returned
    std::string bad_return_move() {
        std::string local = "Local string";
        return std::move(local);  // BAD: Prevents RVO!
    }
    
    // GOOD: Let the compiler optimize
    std::string good_return() {
        std::string local = "Local string";
        return local;  // GOOD: RVO will optimize this
    }
};
```

## Exercises

### Exercise 1: Implement Move Semantics
Create a `Buffer` class that manages a dynamically allocated array and implements proper move semantics.

### Exercise 2: Perfect Forwarding Factory
Write a factory function template that uses perfect forwarding to construct objects with arbitrary arguments.

### Exercise 3: Performance Analysis
Compare the performance of copy vs move operations for different types of objects.

### Exercise 4: Move-Only Type
Design a class that can only be moved, not copied (like `std::unique_ptr`).

## Common Pitfalls

1. **Using moved-from objects**: Objects are valid but unspecified after being moved from
2. **Moving const objects**: `std::move` on const objects will copy, not move
3. **Unnecessary std::move**: Don't move when RVO can optimize
4. **Forgetting noexcept**: Move operations should be marked noexcept when possible
5. **Self-assignment**: Always check for self-assignment in move assignment operators

## Summary

Move semantics revolutionized C++ by enabling efficient transfer of resources instead of expensive copying. Understanding and properly implementing move semantics is crucial for writing high-performance modern C++ code. The key concepts are:

- Rvalue references enable binding to temporary objects
- Move constructors and move assignment operators transfer ownership
- `std::move` converts lvalues to rvalues
- `std::forward` enables perfect forwarding in templates
- RVO and NRVO optimize return value operations
- Proper implementation requires careful attention to noexcept specifications and object states
