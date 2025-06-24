# C++11 Move Semantics

*Duration: 2 weeks*

## Overview

Move semantics is one of the most revolutionary features introduced in C++11, fundamentally changing how C++ handles object ownership and resource management. It allows objects to transfer ownership of their resources instead of copying them, leading to significant performance improvements for resource-heavy operations.

### Why Move Semantics Matter

Before C++11, C++ had a fundamental inefficiency: when you passed or returned objects by value, they were always copied, even when the original object was temporary and would be destroyed immediately. Move semantics solve this by allowing "stealing" resources from temporary objects instead of duplicating them.

#### The Problem Before C++11
```cpp
// Pre-C++11 inefficiency example
std::vector<int> create_large_vector() {
    std::vector<int> v(1000000, 42);
    return v;  // Expensive copy operation!
}

std::vector<int> vec = create_large_vector();  // Another copy!
```

#### The Solution with C++11 Move Semantics
```cpp
// C++11 efficiency with move semantics
std::vector<int> create_large_vector() {
    std::vector<int> v(1000000, 42);
    return v;  // Move operation (or RVO)
}

std::vector<int> vec = create_large_vector();  // Move, not copy!
```

### Performance Impact Visualization
```
Copy Semantics (Pre-C++11):
Source Object: [Data] ────copy────> [Data] :Destination Object
                │                     │
              Slow                  Memory
             Allocation              Usage
                                    Doubled

Move Semantics (C++11+):
Source Object: [Data] ────move────> [    ] :Empty Source
                │                     │
              Fast                 [Data] :Destination Object
           Pointer Swap             Same Memory
```

## Learning Objectives

By the end of this section, you should be able to:

### Fundamental Understanding
- **Explain the difference** between lvalues and rvalues with concrete examples
- **Understand why move semantics** were introduced and what problems they solve
- **Recognize scenarios** where move semantics provide performance benefits
- **Distinguish between** copy semantics and move semantics in code

### Technical Implementation
- **Implement move constructors** and move assignment operators correctly
- **Use rvalue references (`T&&`)** appropriately in function parameters
- **Apply `std::move`** and `std::forward` in the right contexts
- **Write classes** that properly support both copy and move operations

### Advanced Concepts
- **Understand perfect forwarding** and universal references
- **Recognize Return Value Optimization (RVO)** and when it applies
- **Debug move-related issues** and avoid common pitfalls
- **Design move-only types** when appropriate (like `std::unique_ptr`)

### Performance Optimization
- **Measure performance impact** of move vs copy operations
- **Choose between copy and move** based on use case requirements
- **Optimize container operations** using move semantics
- **Write exception-safe move operations** with `noexcept`

## Conceptual Foundation

### The Evolution of Object Semantics in C++

#### Traditional Copy Semantics (Pre-C++11)
```cpp
class TraditionalString {
private:
    char* data;
    size_t length;
    
public:
    // Constructor
    TraditionalString(const char* str) {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
        std::cout << "Constructor: allocated " << length << " bytes\n";
    }
    
    // Copy constructor - EXPENSIVE
    TraditionalString(const TraditionalString& other) {
        length = other.length;
        data = new char[length + 1];           // New allocation
        strcpy(data, other.data);              // Copy all data
        std::cout << "Copy constructor: allocated " << length << " bytes\n";
    }
    
    // Copy assignment - EXPENSIVE  
    TraditionalString& operator=(const TraditionalString& other) {
        if (this != &other) {
            delete[] data;                     // Free old memory
            length = other.length;
            data = new char[length + 1];       // New allocation
            strcpy(data, other.data);          // Copy all data
            std::cout << "Copy assignment: allocated " << length << " bytes\n";
        }
        return *this;
    }
    
    ~TraditionalString() {
        delete[] data;
        std::cout << "Destructor: freed " << length << " bytes\n";
    }
};

// Problem: Expensive operations even for temporary objects
TraditionalString create_string() {
    return TraditionalString("Temporary string");  // Will be copied!
}

void demonstrate_copy_overhead() {
    std::cout << "=== Copy Overhead Demo ===\n";
    TraditionalString s1 = create_string();        // Copy constructor called
    TraditionalString s2("Another string");
    s2 = create_string();                          // Copy assignment called
    // Total: 3 allocations, 2 unnecessary!
}
```

#### Modern Move Semantics (C++11+)
```cpp
class ModernString {
private:
    char* data;
    size_t length;
    
public:
    // Constructor (same as before)
    ModernString(const char* str) {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
        std::cout << "Constructor: allocated " << length << " bytes\n";
    }
    
    // Copy constructor (same as before)
    ModernString(const ModernString& other) {
        length = other.length;
        data = new char[length + 1];
        strcpy(data, other.data);
        std::cout << "Copy constructor: allocated " << length << " bytes\n";
    }
    
    // MOVE CONSTRUCTOR - EFFICIENT!
    ModernString(ModernString&& other) noexcept {
        data = other.data;         // Steal the pointer
        length = other.length;     // Steal the length
        
        other.data = nullptr;      // Reset moved-from object
        other.length = 0;
        std::cout << "Move constructor: transferred " << length << " bytes\n";
    }
    
    // Copy assignment (same as before)
    ModernString& operator=(const ModernString& other) {
        if (this != &other) {
            delete[] data;
            length = other.length;
            data = new char[length + 1];
            strcpy(data, other.data);
            std::cout << "Copy assignment: allocated " << length << " bytes\n";
        }
        return *this;
    }
    
    // MOVE ASSIGNMENT - EFFICIENT!
    ModernString& operator=(ModernString&& other) noexcept {
        if (this != &other) {
            delete[] data;         // Free our old data
            
            data = other.data;     // Steal the pointer
            length = other.length; // Steal the length
            
            other.data = nullptr;  // Reset moved-from object
            other.length = 0;
            std::cout << "Move assignment: transferred " << length << " bytes\n";
        }
        return *this;
    }
    
    ~ModernString() {
        if (data) {  // Check for moved-from state
            delete[] data;
            std::cout << "Destructor: freed " << length << " bytes\n";
        } else {
            std::cout << "Destructor: nothing to free (moved-from)\n";
        }
    }
    
    const char* c_str() const { return data ? data : ""; }
};

// Solution: Efficient operations with temporary objects
ModernString create_modern_string() {
    return ModernString("Temporary string");  // Will be moved!
}

void demonstrate_move_efficiency() {
    std::cout << "=== Move Efficiency Demo ===\n";
    ModernString s1 = create_modern_string();     // Move constructor called
    ModernString s2("Another string");
    s2 = create_modern_string();                  // Move assignment called
    // Total: 1 allocation only!
}
```

### Value Categories: The Foundation of Move Semantics

Understanding value categories is crucial for mastering move semantics:

```cpp
// Value Categories Demonstration
void demonstrate_value_categories() {
    std::cout << "=== Value Categories ===\n";
    
    // LVALUES - have names and addresses
    int x = 42;
    int& lref = x;              // lvalue reference to lvalue
    std::cout << "x is an lvalue, address: " << &x << "\n";
    
    // RVALUES - temporary values without names
    int y = x + 10;             // (x + 10) is an rvalue
    // int& bad_ref = x + 10;   // ERROR: can't bind lvalue ref to rvalue
    
    // RVALUE REFERENCES - bind to rvalues
    int&& rref = x + 10;        // OK: rvalue reference to rvalue
    int&& rref2 = 100;          // OK: literal is rvalue
    int&& rref3 = std::move(x); // OK: std::move converts lvalue to rvalue
    
    std::cout << "rref value: " << rref << "\n";
    
    // IMPORTANT: Named rvalue references are lvalues!
    // int&& another_rref = rref;  // ERROR: rref is now an lvalue
    int&& another_rref = std::move(rref);  // OK: explicit conversion
    
    // Function parameters and value categories
    auto test_lvalue = [](int& param) { 
        std::cout << "Received lvalue\n"; 
    };
    auto test_rvalue = [](int&& param) { 
        std::cout << "Received rvalue\n"; 
    };
    
    test_lvalue(x);             // lvalue
    test_rvalue(42);            // rvalue
    test_rvalue(std::move(x));  // converted to rvalue
}
```

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

## Advanced Concepts and Best Practices

### 1. The Rule of Five/Zero

With move semantics, the "Rule of Three" became the "Rule of Five":

```cpp
class RuleOfFiveExample {
private:
    std::unique_ptr<int[]> data;
    size_t size;
    std::string name;
    
public:
    // Constructor
    RuleOfFiveExample(const std::string& n, size_t s) 
        : name(n), size(s), data(std::make_unique<int[]>(s)) {
        std::cout << "Constructor: " << name << "\n";
    }
    
    // 1. Destructor
    ~RuleOfFiveExample() {
        std::cout << "Destructor: " << name << "\n";
    }
    
    // 2. Copy Constructor
    RuleOfFiveExample(const RuleOfFiveExample& other)
        : name(other.name + "_copy"), size(other.size) {
        data = std::make_unique<int[]>(size);
        std::copy(other.data.get(), other.data.get() + size, data.get());
        std::cout << "Copy Constructor: " << name << "\n";
    }
    
    // 3. Copy Assignment
    RuleOfFiveExample& operator=(const RuleOfFiveExample& other) {
        std::cout << "Copy Assignment: " << name << " = " << other.name << "\n";
        if (this != &other) {
            // Copy-and-swap idiom for exception safety
            RuleOfFiveExample temp(other);
            swap(temp);
        }
        return *this;
    }
    
    // 4. Move Constructor
    RuleOfFiveExample(RuleOfFiveExample&& other) noexcept
        : name(std::move(other.name)), size(other.size), data(std::move(other.data)) {
        other.size = 0;
        std::cout << "Move Constructor: " << name << "\n";
    }
    
    // 5. Move Assignment
    RuleOfFiveExample& operator=(RuleOfFiveExample&& other) noexcept {
        std::cout << "Move Assignment: " << name << " = " << other.name << "\n";
        if (this != &other) {
            // Clean up current resources
            data.reset();
            
            // Move resources
            name = std::move(other.name);
            size = other.size;
            data = std::move(other.data);
            
            // Reset moved-from object
            other.size = 0;
        }
        return *this;
    }
    
    // Helper function for copy-and-swap
    void swap(RuleOfFiveExample& other) noexcept {
        using std::swap;
        swap(name, other.name);
        swap(size, other.size);
        swap(data, other.data);
    }
    
    // Rule of Zero alternative: Use smart pointers and standard containers
    // They handle resource management automatically!
};

// Rule of Zero Example - Preferred approach when possible
class RuleOfZeroExample {
private:
    std::vector<int> data;      // Automatically handles copy/move
    std::string name;           // Automatically handles copy/move
    std::unique_ptr<int> ptr;   // Automatically handles copy/move
    
public:
    RuleOfZeroExample(const std::string& n, size_t s) 
        : name(n), data(s, 42), ptr(std::make_unique<int>(100)) {}
    
    // Compiler-generated destructor, copy/move operations are correct!
    // No need to implement the Big Five manually
};
```

### 2. Exception Safety and noexcept

Move operations should be `noexcept` to enable optimizations:

```cpp
#include <type_traits>
#include <vector>

class ExceptionSafetyExample {
private:
    std::string data;
    
public:
    // Move constructor marked noexcept
    ExceptionSafetyExample(ExceptionSafetyExample&& other) noexcept
        : data(std::move(other.data)) {
        // Move operations should not throw
    }
    
    // Move assignment marked noexcept
    ExceptionSafetyExample& operator=(ExceptionSafetyExample&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
        }
        return *this;
    }
    
    // Copy operations may throw (memory allocation)
    ExceptionSafetyExample(const ExceptionSafetyExample& other)
        : data(other.data) {  // May throw std::bad_alloc
    }
    
    ExceptionSafetyExample& operator=(const ExceptionSafetyExample& other) {
        if (this != &other) {
            data = other.data;  // May throw std::bad_alloc
        }
        return *this;
    }
};

void demonstrate_noexcept_importance() {
    std::cout << "=== noexcept Demonstration ===\n";
    
    using Example = ExceptionSafetyExample;
    
    // Check type traits
    std::cout << "Move constructor noexcept: " 
              << std::is_nothrow_move_constructible_v<Example> << "\n";
    std::cout << "Move assignment noexcept: " 
              << std::is_nothrow_move_assignable_v<Example> << "\n";
    std::cout << "Copy constructor noexcept: " 
              << std::is_nothrow_copy_constructible_v<Example> << "\n";
    
    // std::vector will use move operations only if they're noexcept
    std::vector<Example> vec;
    vec.reserve(10);
    
    vec.emplace_back(Example{});  // Uses move if noexcept
    
    std::cout << "Vector uses move operations: " 
              << std::is_nothrow_move_constructible_v<Example> << "\n";
}
```

### 3. Move-Only Types

Some types should only be movable, not copyable:

```cpp
#include <memory>
#include <iostream>

class MoveOnlyResource {
private:
    std::unique_ptr<int[]> buffer;
    size_t size;
    std::string id;
    
public:
    // Constructor
    explicit MoveOnlyResource(const std::string& identifier, size_t sz = 1000)
        : id(identifier), size(sz), buffer(std::make_unique<int[]>(sz)) {
        std::cout << "Created move-only resource: " << id << "\n";
    }
    
    // Delete copy constructor and copy assignment
    MoveOnlyResource(const MoveOnlyResource&) = delete;
    MoveOnlyResource& operator=(const MoveOnlyResource&) = delete;
    
    // Move constructor
    MoveOnlyResource(MoveOnlyResource&& other) noexcept
        : buffer(std::move(other.buffer)), size(other.size), id(std::move(other.id)) {
        other.size = 0;
        std::cout << "Moved resource: " << id << "\n";
    }
    
    // Move assignment
    MoveOnlyResource& operator=(MoveOnlyResource&& other) noexcept {
        if (this != &other) {
            buffer = std::move(other.buffer);
            size = other.size;
            id = std::move(other.id);
            other.size = 0;
            std::cout << "Move-assigned resource: " << id << "\n";
        }
        return *this;
    }
    
    ~MoveOnlyResource() {
        if (!id.empty()) {
            std::cout << "Destroyed resource: " << id << "\n";
        }
    }
    
    const std::string& get_id() const { return id; }
    size_t get_size() const { return size; }
};

// Factory function for move-only types
MoveOnlyResource create_resource(const std::string& name) {
    return MoveOnlyResource(name, 2000);  // Return by value (move)
}

void demonstrate_move_only() {
    std::cout << "=== Move-Only Types ===\n";
    
    MoveOnlyResource res1("Resource1");
    
    // MoveOnlyResource res2 = res1;        // ERROR: Copy deleted
    MoveOnlyResource res2 = std::move(res1);  // OK: Move
    
    MoveOnlyResource res3 = create_resource("Resource3");  // OK: Move from factory
    
    // Store in containers
    std::vector<MoveOnlyResource> resources;
    resources.push_back(std::move(res2));     // Move into container
    resources.emplace_back("Resource4", 500); // Construct in place
    
    std::cout << "Resources in vector: " << resources.size() << "\n";
}
```

### 4. Perfect Forwarding Deep Dive

Universal references and perfect forwarding for template functions:

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <utility>

// Demonstration class
class ForwardingTarget {
public:
    ForwardingTarget(const std::string& s) {
        std::cout << "Constructed from lvalue string: " << s << "\n";
    }
    
    ForwardingTarget(std::string&& s) {
        std::cout << "Constructed from rvalue string: " << s << "\n";
    }
    
    ForwardingTarget(int x, double y) {
        std::cout << "Constructed with int=" << x << ", double=" << y << "\n";
    }
};

// Perfect forwarding factory
template<typename T, typename... Args>
std::unique_ptr<T> make_unique_perfect(Args&&... args) {
    std::cout << "Factory called with " << sizeof...(args) << " arguments\n";
    return std::make_unique<T>(std::forward<Args>(args)...);
}

// Universal reference function
template<typename T>
void universal_function(T&& param) {
    std::cout << "Universal function called\n";
    
    // Forward to another function
    auto obj = make_unique_perfect<ForwardingTarget>(std::forward<T>(param));
}

// Variadic template with perfect forwarding
template<typename... Args>
void variadic_forwarder(Args&&... args) {
    std::cout << "Variadic forwarder with " << sizeof...(args) << " args\n";
    auto obj = make_unique_perfect<ForwardingTarget>(std::forward<Args>(args)...);
}

void demonstrate_perfect_forwarding() {
    std::cout << "=== Perfect Forwarding ===\n";
    
    // Test with different value categories
    std::string lvalue_str = "lvalue";
    const std::string const_lvalue = "const lvalue";
    
    std::cout << "\n1. Forwarding lvalue:\n";
    universal_function(lvalue_str);
    
    std::cout << "\n2. Forwarding const lvalue:\n";
    universal_function(const_lvalue);
    
    std::cout << "\n3. Forwarding rvalue:\n";
    universal_function(std::string("rvalue"));
    
    std::cout << "\n4. Forwarding literal:\n";
    universal_function("literal");
    
    std::cout << "\n5. Variadic forwarding:\n";
    variadic_forwarder(42, 3.14);
}

// Reference collapsing rules
template<typename T>
void reference_collapsing_demo() {
    std::cout << "=== Reference Collapsing Rules ===\n";
    
    // T& & → T&
    // T& && → T&  
    // T&& & → T&
    // T&& && → T&&
    
    using LRef = int&;
    using RRef = int&&;
    
    // These will all collapse to int&
    using Case1 = LRef&;   // int& & → int&
    using Case2 = LRef&&;  // int& && → int&
    using Case3 = RRef&;   // int&& & → int&
    using Case4 = RRef&&;  // int&& && → int&&
    
    std::cout << "Reference collapsing enables perfect forwarding\n";
}
```

## Practical Exercises and Challenges

### Exercise 1: Implement a Smart Pointer with Move Semantics
Create your own `unique_ptr`-like class that demonstrates proper move semantics:

```cpp
template<typename T>
class MyUniquePtr {
private:
    T* ptr;
    
public:
    // TODO: Implement constructor, destructor, move operations
    // Requirements:
    // - Should be move-only (no copying)
    // - Should release ownership when moved from
    // - Should provide get(), release(), reset() methods
    // - Should support operator* and operator->
    
    explicit MyUniquePtr(T* p = nullptr) : ptr(p) {}
    
    // Your implementation here...
};

// Test your implementation
void test_my_unique_ptr() {
    MyUniquePtr<int> ptr1(new int(42));
    MyUniquePtr<int> ptr2 = std::move(ptr1);  // Should move
    // MyUniquePtr<int> ptr3 = ptr2;          // Should not compile
}
```

### Exercise 2: Optimize a Container Class
Given this inefficient container, add move semantics:

```cpp
class StringContainer {
private:
    std::vector<std::string> strings;
    
public:
    // Add proper move semantics to this class
    StringContainer() = default;
    
    void add_string(const std::string& s) {
        strings.push_back(s);  // TODO: Add overload for rvalue
    }
    
    StringContainer merge(const StringContainer& other) {
        // TODO: Optimize this operation with move semantics
        StringContainer result = *this;
        for (const auto& s : other.strings) {
            result.add_string(s);
        }
        return result;
    }
};
```

### Exercise 3: Perfect Forwarding Factory
Implement a generic factory function that perfectly forwards arguments:

```cpp
template<typename T, typename... Args>
T create_object(Args&&... args) {
    // TODO: Use perfect forwarding to construct T with args
    // Should preserve value categories of arguments
}

// Test cases
class TestClass {
public:
    TestClass(int i, const std::string& s) { /* ... */ }
    TestClass(std::string&& s, double d) { /* ... */ }
};

void test_factory() {
    std::string lvalue = "test";
    auto obj1 = create_object<TestClass>(42, lvalue);           // Forward lvalue
    auto obj2 = create_object<TestClass>("rvalue", 3.14);       // Forward rvalue
}
```

### Exercise 4: Performance Benchmark
Write a benchmark comparing copy vs move performance:

```cpp
#include <chrono>
#include <vector>

class BenchmarkTarget {
    // TODO: Create a class with expensive copy operations
    // but efficient move operations
};

void benchmark_copy_vs_move() {
    const size_t iterations = 10000;
    
    // TODO: Measure time for copy operations
    // TODO: Measure time for move operations
    // TODO: Report the performance difference
}
```

## Common Pitfalls and How to Avoid Them

### 1. The Double Move Problem
```cpp
void demonstrate_double_move() {
    std::string str = "Hello";
    
    std::string first = std::move(str);   // str is now moved-from
    std::string second = std::move(str);  // PROBLEM: Moving from moved-from object
    
    // str is still in a valid but unspecified state
    // This is legal but usually not what you want
    std::cout << "str after double move: '" << str << "'\n";  // Might be empty
}

// Solution: Check state or redesign logic
void safe_move_usage() {
    std::string str = "Hello";
    
    if (!str.empty()) {
        std::string moved = std::move(str);
        // Don't use str after this point
    }
}
```

### 2. Moving const Objects
```cpp
void const_move_pitfall() {
    const std::string const_str = "Cannot move me";
    
    // This compiles but calls copy constructor, not move!
    std::string result = std::move(const_str);  // Copy, not move
    
    // Reason: std::move(const T) returns const T&&
    // const T&& cannot bind to T&& parameter of move constructor
    // Falls back to const T& parameter of copy constructor
}
```

### 3. Return Value Move Pessimization
```cpp
// DON'T DO THIS - Prevents RVO
std::string bad_return() {
    std::string local = "local string";
    return std::move(local);  // Prevents Return Value Optimization!
}

// DO THIS - Enables RVO
std::string good_return() {
    std::string local = "local string";
    return local;  // RVO will optimize this
}

// Exception: When returning a moved parameter
template<typename T>
T forward_parameter(T&& param) {
    // Here std::move is appropriate
    return std::forward<T>(param);
}
```

### 4. Self-Move Assignment
```cpp
class SelfMoveExample {
private:
    std::unique_ptr<int> ptr;
    
public:
    // WRONG: Doesn't handle self-assignment
    SelfMoveExample& operator=(SelfMoveExample&& other) noexcept {
        ptr = std::move(other.ptr);  // If this == &other, ptr becomes nullptr!
        return *this;
    }
    
    // CORRECT: Check for self-assignment
    SelfMoveExample& operator=(SelfMoveExample&& other) noexcept {
        if (this != &other) {
            ptr = std::move(other.ptr);
        }
        return *this;
    }
};
```

## Debugging Move Semantics

### Tools and Techniques

```cpp
#include <iostream>
#include <type_traits>

// Debugging template to show what's happening
template<typename T>
void debug_move_semantics() {
    std::cout << "=== Type Analysis for " << typeid(T).name() << " ===\n";
    std::cout << "Is move constructible: " 
              << std::is_move_constructible_v<T> << "\n";
    std::cout << "Is nothrow move constructible: " 
              << std::is_nothrow_move_constructible_v<T> << "\n";
    std::cout << "Is move assignable: " 
              << std::is_move_assignable_v<T> << "\n";
    std::cout << "Is nothrow move assignable: " 
              << std::is_nothrow_move_assignable_v<T> << "\n";
    std::cout << "Is copy constructible: " 
              << std::is_copy_constructible_v<T> << "\n";
    std::cout << "Is copy assignable: " 
              << std::is_copy_assignable_v<T> << "\n";
}

// Logging wrapper to trace moves
template<typename T>
class MoveTracker {
private:
    T value;
    mutable int id;
    static int next_id;
    
public:
    MoveTracker(T val) : value(std::move(val)), id(++next_id) {
        std::cout << "[" << id << "] Constructed\n";
    }
    
    MoveTracker(const MoveTracker& other) : value(other.value), id(++next_id) {
        std::cout << "[" << id << "] Copy constructed from [" << other.id << "]\n";
    }
    
    MoveTracker(MoveTracker&& other) noexcept 
        : value(std::move(other.value)), id(other.id) {
        other.id = -1;  // Mark as moved-from
        std::cout << "[" << id << "] Move constructed\n";
    }
    
    MoveTracker& operator=(const MoveTracker& other) {
        if (this != &other) {
            value = other.value;
            std::cout << "[" << id << "] Copy assigned from [" << other.id << "]\n";
        }
        return *this;
    }
    
    MoveTracker& operator=(MoveTracker&& other) noexcept {
        if (this != &other) {
            value = std::move(other.value);
            std::cout << "[" << id << "] Move assigned from [" << other.id << "]\n";
            other.id = -1;
        }
        return *this;
    }
    
    ~MoveTracker() {
        if (id != -1) {
            std::cout << "[" << id << "] Destroyed\n";
        } else {
            std::cout << "[moved-from] Destroyed\n";
        }
    }
    
    const T& get() const { return value; }
};

template<typename T>
int MoveTracker<T>::next_id = 0;

// Usage example
void test_move_tracking() {
    std::cout << "=== Move Tracking Demo ===\n";
    
    MoveTracker<std::string> tracker1("Hello");
    MoveTracker<std::string> tracker2 = std::move(tracker1);
    
    std::vector<MoveTracker<std::string>> vec;
    vec.push_back(std::move(tracker2));
    vec.emplace_back("World");
}
```

## Self-Assessment and Study Materials

### Self-Assessment Checklist

Before proceeding to the next topic, ensure you can:

**Conceptual Understanding:**
□ Explain what problems move semantics solve  
□ Distinguish between lvalues and rvalues in code  
□ Describe when move operations are preferred over copy operations  
□ Understand the relationship between RVO and move semantics  

**Implementation Skills:**
□ Implement move constructor and move assignment operator  
□ Use `std::move` appropriately without common pitfalls  
□ Write functions with universal references and perfect forwarding  
□ Create move-only types when appropriate  

**Advanced Concepts:**
□ Apply the Rule of Five correctly  
□ Use `noexcept` specification for move operations  
□ Debug move-related performance issues  
□ Recognize when NOT to use `std::move`  

**Practical Application:**
□ Optimize existing code with move semantics  
□ Write efficient container operations  
□ Implement generic factory functions  
□ Measure performance improvements from move semantics  

### Study Materials

#### Essential Reading
- **Primary:** "Effective Modern C++" by Scott Meyers (Items 23-30)
- **Reference:** "C++ Primer" 5th Edition - Chapter 13 (Copy Control)
- **Deep Dive:** "C++ Move Semantics - The Complete Guide" by Nicolai M. Josuttis
- **Online:** [C++ Reference - Move Semantics](https://en.cppreference.com/w/cpp/language/move_constructor)

#### Video Resources
- "C++11 Move Semantics" by Bjarne Stroustrup
- "Universal References in C++11" by Scott Meyers  
- "CppCon Move Semantics" presentations
- "Back to Basics: Move Semantics" - CppCon

#### Practice Resources
- **Online Compiler:** [Compiler Explorer](https://godbolt.org/) - See assembly output
- **Practice Site:** [C++ Insights](https://cppinsights.io/) - See what compiler generates
- **Benchmarking:** [Quick Bench](http://quick-bench.com/) - Measure performance

### Comprehensive Practice Questions

#### Conceptual Questions
1. **Value Categories:** What's the difference between an lvalue and rvalue? Give 5 examples of each.

2. **Move Semantics Motivation:** Why were move semantics introduced in C++11? Provide a concrete example where they provide significant benefits.

3. **Resource Management:** How do move semantics change the way we think about resource ownership in C++?

4. **Exception Safety:** Why should move operations be `noexcept`? What happens if they're not?

#### Code Analysis Questions
```cpp
// Question 5: What will this output?
class Tracker {
    std::string name;
public:
    Tracker(std::string n) : name(std::move(n)) { 
        std::cout << "Constructed: " << name << "\n"; 
    }
    Tracker(const Tracker& other) : name(other.name) { 
        std::cout << "Copied: " << name << "\n"; 
    }
    Tracker(Tracker&& other) noexcept : name(std::move(other.name)) { 
        std::cout << "Moved: " << name << "\n"; 
    }
};

Tracker make_tracker() { return Tracker("temp"); }
Tracker t1 = make_tracker();
Tracker t2 = std::move(t1);
```

```cpp
// Question 6: Find the problems in this code
class BadExample {
    std::unique_ptr<int> ptr;
public:
    BadExample& operator=(BadExample&& other) {
        ptr = std::move(other.ptr);
        return *this;
    }
    
    std::string process() {
        std::string result = "processed";
        return std::move(result);  // Problem?
    }
};
```

#### Implementation Challenges

**Challenge 1: Thread-Safe Move**
```cpp
// Implement a thread-safe move operation
class ThreadSafeContainer {
    std::vector<int> data;
    mutable std::mutex mtx;
public:
    // TODO: Implement move constructor that properly handles the mutex
    ThreadSafeContainer(ThreadSafeContainer&& other) noexcept;
};
```

**Challenge 2: Conditional Move**
```cpp
// Implement a container that moves elements only if it's beneficial
template<typename T>
class SmartContainer {
    std::vector<T> items;
public:
    void add_item(T item) {
        // TODO: Move if T supports efficient move, otherwise copy
        // Use SFINAE or if constexpr (C++17)
    }
};
```

**Challenge 3: Move Semantics with Inheritance**
```cpp
class Base {
public:
    virtual ~Base() = default;
    Base(Base&& other) noexcept { /* base move logic */ }
    Base& operator=(Base&& other) noexcept { /* base move logic */ return *this; }
};

class Derived : public Base {
    std::unique_ptr<int> data;
public:
    // TODO: Implement move operations that properly handle base class
    Derived(Derived&& other) noexcept;
    Derived& operator=(Derived&& other) noexcept;
};
```

### Performance Measurement Exercises

#### Exercise 1: Vector Growth Performance
```cpp
// Measure performance difference between copy and move during vector reallocation
void measure_vector_growth() {
    // TODO: Create a class with expensive copy operations
    // TODO: Fill a vector and measure reallocation performance
    // TODO: Compare with and without move semantics
}
```

#### Exercise 2: String Concatenation
```cpp
// Compare different string concatenation strategies
void string_concatenation_benchmark() {
    // TODO: Compare copying vs moving strings in concatenation
    // TODO: Measure performance with different string sizes
}
```

### Debugging Exercises

#### Exercise 1: Move Semantics Detective
Given this program that's slower than expected, identify why move semantics aren't being used:

```cpp
class Resource {
    std::vector<int> data;
public:
    Resource(size_t size) : data(size, 42) {}
    
    Resource(const Resource& other) : data(other.data) {
        std::cout << "COPY!\n";
    }
    
    Resource(Resource&& other) : data(std::move(other.data)) {
        std::cout << "MOVE!\n";
    }
    
    // What's missing here?
};

std::vector<Resource> create_resources() {
    std::vector<Resource> result;
    result.push_back(Resource(1000));  // Why does this copy?
    return result;
}
```

#### Exercise 2: Perfect Forwarding Issues
Fix the perfect forwarding in this code:

```cpp
template<typename T>
void process_data(T&& data) {
    // This doesn't preserve value categories correctly
    internal_process(data);  // Should this be std::forward?
    another_process(data);   // What about here?
}
```

### Development Environment Setup

```bash
# Compiler flags for move semantics development
g++ -std=c++11 -O2 -Wall -Wextra -Wpedantic -fno-elide-constructors program.cpp

# Disable RVO to see move operations clearly
g++ -std=c++11 -fno-elide-constructors -fno-optimize-sibling-calls program.cpp

# Enable move semantics debugging
g++ -std=c++11 -DDEBUG_MOVES -fsanitize=address program.cpp
```

### Quick Reference Cheat Sheet

```cpp
// Move semantics cheat sheet

// 1. Declare rvalue reference
T&& rref = std::move(obj);

// 2. Move constructor
Class(Class&& other) noexcept : member(std::move(other.member)) {}

// 3. Move assignment
Class& operator=(Class&& other) noexcept {
    if (this != &other) {
        member = std::move(other.member);
    }
    return *this;
}

// 4. Perfect forwarding
template<typename T>
void func(T&& param) {
    other_func(std::forward<T>(param));
}

// 5. Move-only type
Class(const Class&) = delete;
Class& operator=(const Class&) = delete;

// 6. Conditional move
if constexpr (std::is_move_constructible_v<T>) {
    return std::move(obj);
} else {
    return obj;
}
```

## Summary and Next Steps

Move semantics revolutionized C++ by enabling efficient transfer of resources instead of expensive copying. This fundamental shift in how C++ handles object ownership has made modern C++ both more performant and more expressive.

### Key Takeaways

**Core Concepts Mastered:**
- **Rvalue references (`T&&`)** enable binding to temporary objects and unlock move semantics
- **Move constructors and move assignment operators** transfer ownership efficiently instead of copying
- **`std::move`** converts lvalues to rvalues, enabling explicit move operations
- **`std::forward`** enables perfect forwarding in templates, preserving value categories
- **RVO and NRVO** optimize return value operations automatically

**Best Practices Learned:**
- Always mark move operations as `noexcept` when possible
- Follow the Rule of Five (or better yet, Rule of Zero)
- Use perfect forwarding for generic code
- Avoid common pitfalls like moving const objects or preventing RVO
- Design move-only types when copying doesn't make sense

**Performance Benefits:**
- Significant speedup for resource-heavy objects (containers, strings, smart pointers)
- Reduced memory allocation and deallocation overhead
- Better cache performance through reduced copying
- Enables efficient generic programming patterns

### What's Next?

Having mastered move semantics, you're now ready to explore other C++11/14/17 features that build upon this foundation:

1. **Smart Pointers** - `std::unique_ptr`, `std::shared_ptr`, and `std::weak_ptr` all leverage move semantics
2. **Lambda Expressions** - Can capture by move for efficient closure creation
3. **Variadic Templates** - Often combined with perfect forwarding for flexible APIs
4. **Auto and Type Deduction** - Essential for writing generic code with move semantics
5. **Range-based for loops** - Work efficiently with move-aware containers

### Continuing Your Journey

**Immediate Next Steps:**
- Practice implementing move semantics in your own classes
- Refactor existing code to use move operations where beneficial
- Experiment with perfect forwarding in template functions
- Measure performance improvements in real applications

**Advanced Topics to Explore:**
- Move semantics in concurrent programming
- Custom allocators with move support
- Expression templates and move optimization
- Move semantics in embedded systems programming

Remember: Move semantics aren't just about performance—they enable new design patterns and make C++ code more expressive and safer. The investment in understanding these concepts will pay dividends throughout your C++ career.

## Next Section
[Lambda Expressions and Closures](03_CPP11_Lambda_Expressions.md)
