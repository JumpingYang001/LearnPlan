# C++ Specific Debugging with GDB

## Overview
C++ introduces unique debugging challenges due to its advanced features like templates, overloaded functions, STL containers, exceptions, inheritance, and virtual functions. This guide covers comprehensive debugging techniques specific to C++ using GDB, with practical examples and advanced debugging strategies.

## Key C++ Debugging Challenges
- **Template instantiation** - Understanding which template version is being called
- **Function overloading** - Identifying which overloaded function is executed
- **STL containers** - Inspecting complex container contents and iterators
- **Exception handling** - Tracking exception flow and catching points
- **Virtual functions** - Understanding dynamic dispatch and vtables
- **Memory management** - Debugging new/delete, smart pointers, RAII
- **Name mangling** - Dealing with C++ symbol names

## STL Container Debugging

### Basic STL Container Inspection

**Example: Vector Debugging**
```cpp
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::vector<std::string> names = {"Alice", "Bob", "Charlie"};
    
    // Modify vector
    numbers.push_back(6);
    numbers.insert(numbers.begin() + 2, 99);
    
    // Print elements
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

**Comprehensive GDB Commands for Vectors:**
```bash
# Compile with debug info
g++ -g -std=c++17 -o vector_debug vector_debug.cpp

# Start GDB
gdb ./vector_debug

# Basic vector inspection
(gdb) break main
(gdb) run
(gdb) next 5  # Move to after vector initialization

# Print entire vector
(gdb) print numbers
(gdb) print names

# Print vector size and capacity
(gdb) print numbers.size()
(gdb) print numbers.capacity()
(gdb) print numbers.empty()

# Access specific elements
(gdb) print numbers[0]
(gdb) print numbers._M_impl._M_start[0]  # Internal access
(gdb) print *numbers._M_impl._M_start@numbers.size()  # Print all elements

# Print vector using pretty printers (if available)
(gdb) set print pretty on
(gdb) print numbers

# Examine vector internals
(gdb) print &numbers
(gdb) print numbers._M_impl
(gdb) print numbers._M_impl._M_start  # Pointer to first element
(gdb) print numbers._M_impl._M_finish  # Pointer past last element
(gdb) print numbers._M_impl._M_end_of_storage  # End of allocated memory
```

### Advanced STL Container Debugging

**Example: Complex STL Containers**
```cpp
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>

struct Person {
    std::string name;
    int age;
    std::vector<std::string> hobbies;
    
    Person(const std::string& n, int a) : name(n), age(a) {}
};

int main() {
    // Map debugging
    std::map<std::string, Person> people;
    people.emplace("john", Person("John Doe", 30));
    people["jane"] = Person("Jane Smith", 25);
    people["jane"].hobbies = {"reading", "swimming", "coding"};
    
    // Set debugging
    std::set<int> unique_numbers = {5, 2, 8, 2, 1, 9};
    
    // Unordered map debugging
    std::unordered_map<int, std::string> id_to_name = {
        {101, "Alice"}, {102, "Bob"}, {103, "Charlie"}
    };
    
    std::cout << "People count: " << people.size() << std::endl;
    
    return 0;
}
```

**GDB Commands for Complex Containers:**
```bash
# Map debugging
(gdb) print people
(gdb) print people.size()
(gdb) print people["john"]
(gdb) print people["john"].name
(gdb) print people["john"].hobbies

# Iterate through map
(gdb) print people.begin()
(gdb) print people.end()
(gdb) set $it = people.begin()
(gdb) print $it->first   # Key
(gdb) print $it->second  # Value
(gdb) print $it->second.name

# Set debugging
(gdb) print unique_numbers
(gdb) print unique_numbers.size()
(gdb) set $set_it = unique_numbers.begin()
(gdb) print *$set_it

# Unordered map debugging
(gdb) print id_to_name
(gdb) print id_to_name[101]
(gdb) print id_to_name.bucket_count()
(gdb) print id_to_name.load_factor()

# Custom pretty printer for complex structures
(gdb) define print_person
print $arg0.name
print $arg0.age
print $arg0.hobbies
end

(gdb) print_person people["jane"]
```

### STL Iterator Debugging

**Example: Iterator Problems**
```cpp
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // Safe iterator usage
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << " ";
    }
    
    // Potential iterator invalidation
    auto it = vec.begin();
    vec.push_back(6);  // May invalidate iterators!
    // std::cout << *it;  // Dangerous - iterator may be invalid
    
    return 0;
}
```

**Iterator Debugging Commands:**
```bash
# Iterator inspection
(gdb) print vec.begin()
(gdb) print vec.end()
(gdb) set $it = vec.begin()
(gdb) print $it
(gdb) print *$it
(gdb) print $it._M_current  # Internal iterator pointer

# Check iterator validity
(gdb) print $it >= vec.begin()
(gdb) print $it < vec.end()

# Advance iterator
(gdb) set $it = $it + 1
(gdb) print *$it

# Distance between iterators
(gdb) print vec.end() - vec.begin()
```

## Template Debugging

### Understanding Template Instantiation

**Example: Function Templates**
```cpp
#include <iostream>
#include <string>
#include <vector>

template<typename T>
void print_value(const T& value) {
    std::cout << "Value: " << value << std::endl;
}

template<typename T>
T add(const T& a, const T& b) {
    return a + b;
}

// Template specialization
template<>
void print_value<std::vector<int>>(const std::vector<int>& vec) {
    std::cout << "Vector contents: ";
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Different template instantiations
    print_value(42);                    // T = int
    print_value(3.14);                  // T = double
    print_value(std::string("Hello"));  // T = std::string
    
    std::vector<int> numbers = {1, 2, 3};
    print_value(numbers);               // Specialized version
    
    auto result1 = add(10, 20);         // T = int
    auto result2 = add(1.5, 2.5);       // T = double
    
    return 0;
}
```

**Template Debugging Commands:**
```bash
# Compile with debug info and disable optimizations
g++ -g -O0 -std=c++17 -o template_debug template_debug.cpp

# Start debugging
(gdb) break main
(gdb) run

# Set breakpoints in template instantiations
(gdb) break print_value<int>
(gdb) break print_value<double>
(gdb) break print_value<std::string>
(gdb) break print_value<std::vector<int>>

# List all instantiations of a template
(gdb) info functions print_value
(gdb) info functions add

# Print template parameter information
(gdb) continue  # Until we hit print_value<int>
(gdb) info args
(gdb) print value
(gdb) ptype value  # Shows the actual type T

# Step through different template instantiations
(gdb) continue  # Next instantiation
(gdb) print value
(gdb) ptype value
```

### Class Template Debugging

**Example: Complex Class Templates**
```cpp
#include <iostream>
#include <vector>
#include <memory>

template<typename T, int SIZE = 10>
class Container {
private:
    T data[SIZE];
    int current_size;
    
public:
    Container() : current_size(0) {}
    
    void add(const T& item) {
        if (current_size < SIZE) {
            data[current_size++] = item;
        }
    }
    
    T& get(int index) {
        return data[index];
    }
    
    int size() const { return current_size; }
    
    // Template member function
    template<typename U>
    void print_with_prefix(const U& prefix) {
        for (int i = 0; i < current_size; ++i) {
            std::cout << prefix << ": " << data[i] << std::endl;
        }
    }
};

// Template specialization for pointers
template<typename T, int SIZE>
class Container<T*, SIZE> {
private:
    T* data[SIZE];
    int current_size;
    
public:
    Container() : current_size(0) {
        for (int i = 0; i < SIZE; ++i) {
            data[i] = nullptr;
        }
    }
    
    void add(T* item) {
        if (current_size < SIZE) {
            data[current_size++] = item;
        }
    }
    
    T* get(int index) {
        return data[index];
    }
};

int main() {
    Container<int, 5> int_container;
    int_container.add(10);
    int_container.add(20);
    int_container.print_with_prefix(std::string("Number"));
    
    Container<std::string> string_container;  // Uses default SIZE = 10
    string_container.add("Hello");
    string_container.add("World");
    
    // Specialized template for pointers
    Container<int*, 3> pointer_container;
    int value1 = 100, value2 = 200;
    pointer_container.add(&value1);
    pointer_container.add(&value2);
    
    return 0;
}
```

**Class Template Debugging Commands:**
```bash
# Set breakpoints on specific template instantiations
(gdb) break Container<int, 5>::add
(gdb) break Container<std::string, 10>::add
(gdb) break Container<int*, 3>::add

# Print template class information
(gdb) ptype Container<int, 5>
(gdb) ptype Container<std::string, 10>
(gdb) print sizeof(Container<int, 5>)

# Inspect template class instances
(gdb) print int_container
(gdb) print int_container.current_size
(gdb) print int_container.data[0]
(gdb) print int_container.data@int_container.current_size

# Debug template member functions
(gdb) break Container<int, 5>::print_with_prefix<std::string>
(gdb) continue
(gdb) print prefix
(gdb) ptype prefix

# View template specialization
(gdb) ptype Container<int*, 3>
(gdb) print pointer_container
(gdb) print *pointer_container.get(0)
```

### Template Metaprogramming Debugging

**Example: SFINAE and Type Traits**
```cpp
#include <iostream>
#include <type_traits>
#include <vector>

// SFINAE example
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
process_value(const T& value) {
    std::cout << "Processing integral value: " << value << std::endl;
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
process_value(const T& value) {
    std::cout << "Processing floating point value: " << value << std::endl;
}

// Constexpr template metaprogramming
template<int N>
constexpr int factorial() {
    if constexpr (N <= 1) {
        return 1;
    } else {
        return N * factorial<N-1>();
    }
}

int main() {
    process_value(42);        // Integral version
    process_value(3.14);      // Floating point version
    
    constexpr int fact5 = factorial<5>();
    std::cout << "5! = " << fact5 << std::endl;
    
    return 0;
}
```

**Metaprogramming Debugging Commands:**
```bash
# Compile with template instantiation debugging
g++ -g -O0 -std=c++17 -ftemplate-backtrace-limit=0 -o meta_debug meta_debug.cpp

# Debug SFINAE resolution
(gdb) info functions process_value
(gdb) break process_value  # GDB will show all overloads

# Print type trait results at compile time
# (Use static_assert for compile-time debugging)
# static_assert(std::is_integral<int>::value, "int should be integral");

# Runtime debugging of template metaprogramming
(gdb) print fact5
(gdb) print factorial<3>()  # If function is available at runtime
```

## Function Overloading Debugging

### Understanding Overload Resolution

**Example: Function Overloads**
```cpp
#include <iostream>
#include <string>

class Calculator {
public:
    // Overloaded functions
    int add(int a, int b) {
        std::cout << "Adding integers: " << a << " + " << b << std::endl;
        return a + b;
    }
    
    double add(double a, double b) {
        std::cout << "Adding doubles: " << a << " + " << b << std::endl;
        return a + b;
    }
    
    std::string add(const std::string& a, const std::string& b) {
        std::cout << "Concatenating strings: " << a << " + " << b << std::endl;
        return a + b;
    }
    
    // Overloaded with different parameter counts
    int add(int a, int b, int c) {
        std::cout << "Adding three integers" << std::endl;
        return a + b + c;
    }
};

// Free function overloads
void process(int value) {
    std::cout << "Processing int: " << value << std::endl;
}

void process(double value) {
    std::cout << "Processing double: " << value << std::endl;
}

void process(const char* value) {
    std::cout << "Processing C-string: " << value << std::endl;
}

int main() {
    Calculator calc;
    
    // Different overload calls
    auto result1 = calc.add(5, 10);          // int version
    auto result2 = calc.add(3.14, 2.86);     // double version
    auto result3 = calc.add(std::string("Hello"), std::string(" World"));  // string version
    auto result4 = calc.add(1, 2, 3);        // three parameter version
    
    // Free function overloads
    process(42);        // int version
    process(3.14);      // double version
    process("Hello");   // const char* version
    
    return 0;
}
```

**Overload Debugging Commands:**
```bash
# List all overloads of a function
(gdb) info functions Calculator::add
(gdb) info functions process

# Set breakpoints on specific overloads
(gdb) break Calculator::add(int, int)
(gdb) break Calculator::add(double, double)
(gdb) break Calculator::add(std::string const&, std::string const&)
(gdb) break Calculator::add(int, int, int)

# Use demangled names
(gdb) set demangle-style gnu-v3
(gdb) info functions Calculator::add

# Print function signatures
(gdb) ptype Calculator::add
(gdb) whatis Calculator::add

# Debug overload resolution at call site
(gdb) break main
(gdb) run
(gdb) step  # Step into each function call to see which overload is chosen

# Print arguments to understand why specific overload was chosen
(gdb) continue  # Until we hit an add function
(gdb) info args
(gdb) ptype a
(gdb) ptype b
```

## Exception Debugging

### Catching and Debugging Exceptions

**Example: Exception Handling**
```cpp
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

class CustomException : public std::exception {
private:
    std::string message;
    
public:
    CustomException(const std::string& msg) : message(msg) {}
    
    const char* what() const noexcept override {
        return message.c_str();
    }
};

class FileProcessor {
public:
    void process_file(const std::string& filename) {
        if (filename.empty()) {
            throw std::invalid_argument("Filename cannot be empty");
        }
        
        if (filename == "nonexistent.txt") {
            throw std::runtime_error("File not found: " + filename);
        }
        
        if (filename == "corrupt.txt") {
            throw CustomException("File is corrupted: " + filename);
        }
        
        std::cout << "Processing file: " << filename << std::endl;
    }
    
    void risky_operation() {
        std::vector<int> vec(10);
        
        // This will throw std::out_of_range
        try {
            int value = vec.at(20);  // Out of bounds access
            std::cout << "Value: " << value << std::endl;
        } catch (const std::out_of_range& e) {
            std::cout << "Caught out_of_range: " << e.what() << std::endl;
            throw;  // Re-throw the exception
        }
    }
};

int main() {
    FileProcessor processor;
    
    try {
        // Various exception scenarios
        processor.process_file("");                    // std::invalid_argument
        processor.process_file("nonexistent.txt");    // std::runtime_error
        processor.process_file("corrupt.txt");        // CustomException
        processor.risky_operation();                  // std::out_of_range
        
    } catch (const CustomException& e) {
        std::cout << "Custom exception: " << e.what() << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "Runtime error: " << e.what() << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "Invalid argument: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Standard exception: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
    }
    
    return 0;
}
```

**Exception Debugging Commands:**
```bash
# Compile with exception handling debug info
g++ -g -O0 -std=c++17 -o exception_debug exception_debug.cpp

# Enable exception catching in GDB
(gdb) catch throw
(gdb) catch catch
(gdb) catch rethrow

# Catch specific exception types
(gdb) catch throw std::runtime_error
(gdb) catch throw std::invalid_argument
(gdb) catch throw CustomException

# Run and catch exceptions
(gdb) run

# When exception is caught, examine the exception
(gdb) print $_exception
(gdb) call $_exception->what()

# Print stack trace when exception is thrown
(gdb) bt
(gdb) info registers

# Continue to catch handler
(gdb) continue

# Set breakpoints in catch blocks
(gdb) break main.cpp:65  # catch (const CustomException& e)
(gdb) break main.cpp:67  # catch (const std::runtime_error& e)

# Examine exception object in catch block
(gdb) print e
(gdb) print e.what()
(gdb) ptype e

# Debug exception propagation
(gdb) set print unwind-info on
(gdb) info unwinder
```

### Advanced Exception Debugging

**Example: Exception Safety and RAII**
```cpp
#include <iostream>
#include <memory>
#include <stdexcept>
#include <fstream>

class Resource {
private:
    std::string name;
    
public:
    Resource(const std::string& n) : name(n) {
        std::cout << "Resource " << name << " acquired" << std::endl;
    }
    
    ~Resource() {
        std::cout << "Resource " << name << " released" << std::endl;
    }
    
    void use() {
        std::cout << "Using resource " << name << std::endl;
        if (name == "bad_resource") {
            throw std::runtime_error("Resource " + name + " failed");
        }
    }
};

class ExceptionSafeClass {
private:
    std::unique_ptr<Resource> resource1;
    std::unique_ptr<Resource> resource2;
    
public:
    ExceptionSafeClass() {
        try {
            resource1 = std::make_unique<Resource>("resource1");
            resource2 = std::make_unique<Resource>("bad_resource");  // This might throw
            
            resource1->use();
            resource2->use();  // This will throw
            
        } catch (...) {
            std::cout << "Exception in constructor, resources will be cleaned up" << std::endl;
            throw;  // Re-throw to let caller handle
        }
    }
    
    void operate() {
        if (resource1) resource1->use();
        if (resource2) resource2->use();
    }
};

int main() {
    try {
        std::cout << "=== Creating ExceptionSafeClass ===" << std::endl;
        ExceptionSafeClass obj;  // Constructor will throw
        obj.operate();
        
    } catch (const std::exception& e) {
        std::cout << "Caught in main: " << e.what() << std::endl;
    }
    
    std::cout << "=== End of main ===" << std::endl;
    return 0;
}
```

**RAII and Exception Safety Debugging:**
```bash
# Debug constructor/destructor calls during exceptions
(gdb) break Resource::Resource
(gdb) break Resource::~Resource
(gdb) break ExceptionSafeClass::ExceptionSafeClass

# Enable destructors debugging
(gdb) set debug-unwinder on

# Run and observe RAII in action
(gdb) run

# When in constructor
(gdb) print this
(gdb) print name

# Continue to see destruction order
(gdb) continue

# Check smart pointer states
(gdb) print resource1
(gdb) print resource1.get()
(gdb) print resource2
(gdb) print resource2.get()
```

## Virtual Function and Inheritance Debugging

### Understanding Virtual Function Calls

**Example: Polymorphism and Virtual Functions**
```cpp
#include <iostream>
#include <vector>
#include <memory>

class Shape {
protected:
    std::string name;
    
public:
    Shape(const std::string& n) : name(n) {}
    virtual ~Shape() = default;
    
    // Pure virtual function
    virtual double area() const = 0;
    virtual void draw() const = 0;
    
    // Virtual function with implementation
    virtual void print_info() const {
        std::cout << "Shape: " << name << ", Area: " << area() << std::endl;
    }
    
    // Non-virtual function
    std::string get_name() const { return name; }
};

class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(const std::string& name, double w, double h) 
        : Shape(name), width(w), height(h) {}
    
    double area() const override {
        return width * height;
    }
    
    void draw() const override {
        std::cout << "Drawing rectangle: " << name << " (" << width << "x" << height << ")" << std::endl;
    }
    
    // Override virtual function
    void print_info() const override {
        std::cout << "Rectangle: " << name << ", Width: " << width 
                  << ", Height: " << height << ", Area: " << area() << std::endl;
    }
};

class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(const std::string& name, double r) 
        : Shape(name), radius(r) {}
    
    double area() const override {
        return 3.14159 * radius * radius;
    }
    
    void draw() const override {
        std::cout << "Drawing circle: " << name << " (radius: " << radius << ")" << std::endl;
    }
};

class Triangle : public Shape {
private:
    double base, height;
    
public:
    Triangle(const std::string& name, double b, double h) 
        : Shape(name), base(b), height(h) {}
    
    double area() const override {
        return 0.5 * base * height;
    }
    
    void draw() const override {
        std::cout << "Drawing triangle: " << name << " (base: " << base << ", height: " << height << ")" << std::endl;
    }
};

int main() {
    // Create objects
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Rectangle>("Rect1", 5.0, 3.0));
    shapes.push_back(std::make_unique<Circle>("Circle1", 2.5));
    shapes.push_back(std::make_unique<Triangle>("Triangle1", 4.0, 6.0));
    
    // Polymorphic calls
    for (const auto& shape : shapes) {
        shape->draw();        // Virtual function call
        shape->print_info();  // Virtual function call
        std::cout << "Name: " << shape->get_name() << std::endl;  // Non-virtual call
        std::cout << "---" << std::endl;
    }
    
    return 0;
}
```

**Virtual Function Debugging Commands:**
```bash
# Compile with debug info
g++ -g -O0 -std=c++17 -o virtual_debug virtual_debug.cpp

# Debug virtual function calls
(gdb) break main
(gdb) run

# Examine vtable
(gdb) print shapes[0]
(gdb) print *(shapes[0].get())
(gdb) info vtbl *(shapes[0].get())  # Show vtable contents

# Print object type information
(gdb) print typeid(*(shapes[0].get())).name()
(gdb) whatis *(shapes[0].get())
(gdb) ptype *(shapes[0].get())

# Set breakpoints on virtual functions
(gdb) break Shape::draw
(gdb) break Rectangle::draw
(gdb) break Circle::draw
(gdb) break Triangle::draw

# Step through virtual function calls
(gdb) break main.cpp:95  # The draw() call in loop
(gdb) continue
(gdb) step  # This will step into the actual implementation

# Examine which function is called
(gdb) print shape.get()
(gdb) print *(shape.get())
(gdb) info symbol $pc  # Shows current function

# Debug virtual function resolution
(gdb) disassemble  # Show assembly for virtual call
```

### Multiple Inheritance Debugging

**Example: Multiple Inheritance and Virtual Base Classes**
```cpp
#include <iostream>

class Base1 {
public:
    int value1;
    Base1(int v) : value1(v) {
        std::cout << "Base1 constructor: " << value1 << std::endl;
    }
    virtual ~Base1() = default;
    virtual void func1() { std::cout << "Base1::func1" << std::endl; }
};

class Base2 {
public:
    int value2;
    Base2(int v) : value2(v) {
        std::cout << "Base2 constructor: " << value2 << std::endl;
    }
    virtual ~Base2() = default;
    virtual void func2() { std::cout << "Base2::func2" << std::endl; }
};

class VirtualBase {
public:
    int virtual_value;
    VirtualBase(int v) : virtual_value(v) {
        std::cout << "VirtualBase constructor: " << virtual_value << std::endl;
    }
    virtual ~VirtualBase() = default;
    virtual void virtual_func() { std::cout << "VirtualBase::virtual_func" << std::endl; }
};

class Derived1 : public virtual VirtualBase, public Base1 {
public:
    Derived1(int vb, int b1) : VirtualBase(vb), Base1(b1) {
        std::cout << "Derived1 constructor" << std::endl;
    }
};

class Derived2 : public virtual VirtualBase, public Base2 {
public:
    Derived2(int vb, int b2) : VirtualBase(vb), Base2(b2) {
        std::cout << "Derived2 constructor" << std::endl;
    }
};

class MultiDerived : public Derived1, public Derived2 {
public:
    MultiDerived(int vb, int b1, int b2) 
        : VirtualBase(vb), Derived1(vb, b1), Derived2(vb, b2) {
        std::cout << "MultiDerived constructor" << std::endl;
    }
    
    void func1() override { std::cout << "MultiDerived::func1" << std::endl; }
    void func2() override { std::cout << "MultiDerived::func2" << std::endl; }
    void virtual_func() override { std::cout << "MultiDerived::virtual_func" << std::endl; }
};

int main() {
    MultiDerived obj(100, 200, 300);
    
    obj.func1();
    obj.func2();
    obj.virtual_func();
    
    std::cout << "VirtualBase value: " << obj.virtual_value << std::endl;
    std::cout << "Base1 value: " << obj.value1 << std::endl;
    std::cout << "Base2 value: " << obj.value2 << std::endl;
    
    return 0;
}
```

**Multiple Inheritance Debugging:**
```bash
# Examine object layout
(gdb) print obj
(gdb) print sizeof(obj)
(gdb) print &obj

# Print base class subobjects
(gdb) print (Base1&)obj
(gdb) print (Base2&)obj
(gdb) print (VirtualBase&)obj
(gdb) print (Derived1&)obj
(gdb) print (Derived2&)obj

# Show object memory layout
(gdb) x/20w &obj  # Examine memory as words

# Print vtable for each base
(gdb) info vtbl (Base1&)obj
(gdb) info vtbl (Base2&)obj
(gdb) info vtbl (VirtualBase&)obj

# Check virtual base offset
(gdb) print &obj.virtual_value
(gdb) print &((VirtualBase&)obj)
```

## Memory Management Debugging

### Smart Pointers Debugging

**Example: Smart Pointer Usage**
```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>

class Resource {
private:
    std::string name;
    int* data;
    
public:
    Resource(const std::string& n, int size) : name(n) {
        data = new int[size];
        for (int i = 0; i < size; ++i) {
            data[i] = i * 10;
        }
        std::cout << "Resource " << name << " created with " << size << " elements" << std::endl;
    }
    
    ~Resource() {
        delete[] data;
        std::cout << "Resource " << name << " destroyed" << std::endl;
    }
    
    void print_data(int count) {
        std::cout << name << " data: ";
        for (int i = 0; i < count; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    
    const std::string& get_name() const { return name; }
};

void test_unique_ptr() {
    std::cout << "=== Testing unique_ptr ===" << std::endl;
    
    auto ptr1 = std::make_unique<Resource>("UniqueRes1", 5);
    ptr1->print_data(5);
    
    // Transfer ownership
    auto ptr2 = std::move(ptr1);
    
    std::cout << "After move:" << std::endl;
    std::cout << "ptr1 is " << (ptr1 ? "valid" : "null") << std::endl;
    std::cout << "ptr2 is " << (ptr2 ? "valid" : "null") << std::endl;
    
    if (ptr2) {
        ptr2->print_data(3);
    }
}

void test_shared_ptr() {
    std::cout << "=== Testing shared_ptr ===" << std::endl;
    
    std::shared_ptr<Resource> shared1 = std::make_shared<Resource>("SharedRes1", 4);
    std::cout << "shared1 use_count: " << shared1.use_count() << std::endl;
    
    {
        std::shared_ptr<Resource> shared2 = shared1;
        std::cout << "After creating shared2, use_count: " << shared1.use_count() << std::endl;
        
        std::weak_ptr<Resource> weak1 = shared1;
        std::cout << "weak_ptr expired: " << weak1.expired() << std::endl;
        
        if (auto locked = weak1.lock()) {
            locked->print_data(2);
            std::cout << "Locked use_count: " << locked.use_count() << std::endl;
        }
    }
    
    std::cout << "After scope exit, use_count: " << shared1.use_count() << std::endl;
}

class Node {
public:
    int value;
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> parent;
    
    Node(int v) : value(v) {
        std::cout << "Node " << value << " created" << std::endl;
    }
    
    ~Node() {
        std::cout << "Node " << value << " destroyed" << std::endl;
    }
};

void test_circular_reference() {
    std::cout << "=== Testing circular reference ===" << std::endl;
    
    auto node1 = std::make_shared<Node>(1);
    auto node2 = std::make_shared<Node>(2);
    auto node3 = std::make_shared<Node>(3);
    
    // Create chain: node1 -> node2 -> node3
    node1->next = node2;
    node2->next = node3;
    
    // Set parent pointers (using weak_ptr to avoid cycles)
    node2->parent = node1;
    node3->parent = node2;
    
    std::cout << "node1 use_count: " << node1.use_count() << std::endl;
    std::cout << "node2 use_count: " << node2.use_count() << std::endl;
    std::cout << "node3 use_count: " << node3.use_count() << std::endl;
}

int main() {
    test_unique_ptr();
    test_shared_ptr();
    test_circular_reference();
    
    return 0;
}
```

**Smart Pointer Debugging Commands:**
```bash
# Compile with debug info
g++ -g -O0 -std=c++17 -o smart_ptr_debug smart_ptr_debug.cpp

# Debug unique_ptr
(gdb) break test_unique_ptr
(gdb) run
(gdb) next 3

# Examine unique_ptr
(gdb) print ptr1
(gdb) print ptr1.get()
(gdb) print *ptr1
(gdb) ptype ptr1

# After move operation
(gdb) next 3
(gdb) print ptr1
(gdb) print ptr1.get()  # Should be nullptr
(gdb) print ptr2
(gdb) print ptr2.get()

# Debug shared_ptr
(gdb) break test_shared_ptr
(gdb) continue
(gdb) next 3

# Examine shared_ptr reference counting
(gdb) print shared1
(gdb) print shared1.get()
(gdb) print shared1.use_count()
(gdb) print shared1._M_refcount  # Internal reference count object

# Step into scope with shared2
(gdb) next 3
(gdb) print shared1.use_count()
(gdb) print shared2.use_count()

# Debug weak_ptr
(gdb) next 3
(gdb) print weak1
(gdb) print weak1.expired()
(gdb) print weak1.use_count()

# Debug circular references
(gdb) break test_circular_reference
(gdb) continue

# Examine node relationships
(gdb) print node1.use_count()
(gdb) print node1->next.use_count()
(gdb) print node2->parent.expired()
```

### RAII and Custom Deleters

**Example: Custom Memory Management**
```cpp
#include <iostream>
#include <memory>
#include <cstdlib>

// Custom deleter for malloc'd memory
struct MallocDeleter {
    void operator()(void* ptr) {
        std::cout << "Custom deleter: freeing malloc'd memory at " << ptr << std::endl;
        std::free(ptr);
    }
};

// Custom deleter for arrays
template<typename T>
struct ArrayDeleter {
    void operator()(T* ptr) {
        std::cout << "Custom deleter: deleting array at " << ptr << std::endl;
        delete[] ptr;
    }
};

// RAII wrapper for C-style resources
class FileWrapper {
private:
    FILE* file;
    std::string filename;
    
public:
    FileWrapper(const std::string& name, const char* mode) : filename(name) {
        file = fopen(name.c_str(), mode);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + name);
        }
        std::cout << "File " << filename << " opened" << std::endl;
    }
    
    ~FileWrapper() {
        if (file) {
            fclose(file);
            std::cout << "File " << filename << " closed" << std::endl;
        }
    }
    
    // Disable copy
    FileWrapper(const FileWrapper&) = delete;
    FileWrapper& operator=(const FileWrapper&) = delete;
    
    // Enable move
    FileWrapper(FileWrapper&& other) noexcept : file(other.file), filename(std::move(other.filename)) {
        other.file = nullptr;
    }
    
    FileWrapper& operator=(FileWrapper&& other) noexcept {
        if (this != &other) {
            if (file) fclose(file);
            file = other.file;
            filename = std::move(other.filename);
            other.file = nullptr;
        }
        return *this;
    }
    
    FILE* get() { return file; }
    bool is_open() const { return file != nullptr; }
};

int main() {
    // Custom deleter with unique_ptr
    {
        std::unique_ptr<int, MallocDeleter> malloc_ptr(static_cast<int*>(std::malloc(sizeof(int))));
        *malloc_ptr = 42;
        std::cout << "Malloc'd value: " << *malloc_ptr << std::endl;
    }  // Custom deleter called here
    
    // Array deleter
    {
        std::unique_ptr<int[], ArrayDeleter<int>> array_ptr(new int[5]);
        for (int i = 0; i < 5; ++i) {
            array_ptr[i] = i * 10;
        }
    }  // Array deleter called here
    
    // RAII file wrapper
    try {
        FileWrapper file("test.txt", "w");
        if (file.is_open()) {
            fprintf(file.get(), "Hello, RAII!\n");
        }
    }  // File automatically closed here
    catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
```

**Custom Memory Management Debugging:**
```bash
# Debug custom deleters
(gdb) break MallocDeleter::operator()
(gdb) break ArrayDeleter<int>::operator()

# Debug RAII object lifecycle
(gdb) break FileWrapper::FileWrapper
(gdb) break FileWrapper::~FileWrapper

# Run and observe custom deleters
(gdb) run

# When in custom deleter
(gdb) print ptr
(gdb) bt  # Show call stack

# Debug move semantics
(gdb) break FileWrapper::FileWrapper(FileWrapper&&)
(gdb) print other.file
(gdb) print this->file
```

## Name Mangling and Symbol Debugging

### Understanding C++ Name Mangling

**Example: Name Mangling Investigation**
```cpp
#include <iostream>
#include <typeinfo>

namespace MyNamespace {
    class MyClass {
    public:
        void simple_function() {}
        void overloaded_function(int x) {}
        void overloaded_function(double x) {}
        void overloaded_function(int x, double y) {}
        
        template<typename T>
        void template_function(T value) {}
        
        static void static_function() {}
    };
    
    template<typename T, int N>
    class TemplateClass {
    public:
        void member_function() {}
    };
}

extern "C" void c_function() {
    std::cout << "C function (no mangling)" << std::endl;
}

int main() {
    MyNamespace::MyClass obj;
    obj.simple_function();
    obj.overloaded_function(42);
    obj.overloaded_function(3.14);
    obj.overloaded_function(10, 2.5);
    obj.template_function(100);
    obj.template_function(std::string("hello"));
    
    MyNamespace::TemplateClass<int, 5> template_obj;
    template_obj.member_function();
    
    c_function();
    
    return 0;
}
```

**Name Mangling Debugging Commands:**
```bash
# View mangled symbols
objdump -t name_mangle_debug | grep MyNamespace
nm name_mangle_debug | grep MyNamespace

# In GDB, view mangled names
(gdb) info functions MyNamespace
(gdb) maint print symbols
(gdb) maint print msymbols

# Demangle names manually
(gdb) set demangle-style none  # Show mangled names
(gdb) info functions MyClass
(gdb) set demangle-style gnu-v3  # Show demangled names
(gdb) info functions MyClass

# Use c++filt to demangle externally
# c++filt _ZN11MyNamespace7MyClass15simple_functionEv

# Debug template instantiations
(gdb) info functions template_function
(gdb) break MyNamespace::MyClass::template_function<int>
(gdb) break MyNamespace::MyClass::template_function<std::string>

# Set breakpoints using mangled names
(gdb) break _ZN11MyNamespace7MyClass15simple_functionEv
```

## Advanced C++ Debugging Techniques

### Lambda Function Debugging

**Example: Lambda Debugging**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3};
    
    // Simple lambda
    auto print_lambda = [](int x) {
        std::cout << x << " ";
    };
    
    // Lambda with capture
    int multiplier = 10;
    auto multiply_lambda = [multiplier](int x) -> int {
        return x * multiplier;
    };
    
    // Lambda with mutable capture
    int counter = 0;
    auto counting_lambda = [counter](int x) mutable -> int {
        return x + (++counter);
    };
    
    // Generic lambda (C++14)
    auto generic_lambda = [](auto x, auto y) {
        return x + y;
    };
    
    std::cout << "Original numbers: ";
    std::for_each(numbers.begin(), numbers.end(), print_lambda);
    std::cout << std::endl;
    
    // Transform with lambda
    std::vector<int> multiplied;
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(multiplied), multiply_lambda);
    
    std::cout << "Multiplied numbers: ";
    std::for_each(multiplied.begin(), multiplied.end(), print_lambda);
    std::cout << std::endl;
    
    // Use counting lambda
    for (int i : {1, 2, 3, 4, 5}) {
        std::cout << "Counting result: " << counting_lambda(i) << std::endl;
    }
    
    // Generic lambda usage
    std::cout << "Generic lambda: " << generic_lambda(5, 3) << std::endl;
    std::cout << "Generic lambda: " << generic_lambda(2.5, 1.5) << std::endl;
    
    return 0;
}
```

**Lambda Debugging Commands:**
```bash
# Debug lambda captures
(gdb) break main
(gdb) run
(gdb) next 10  # After lambda definitions

# Print lambda objects
(gdb) print print_lambda
(gdb) ptype print_lambda
(gdb) print multiply_lambda
(gdb) print multiply_lambda.__multiplier  # Captured variable

# Set breakpoints in lambdas (GDB may show them as operator())
(gdb) info functions operator()
(gdb) break lambda  # Set breakpoint in lambda body

# Step into lambda execution
(gdb) step  # When calling std::for_each
```

### Modern C++ Features Debugging

**Example: C++11/14/17 Features**
```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <tuple>
#include <optional>
#include <variant>
#include <any>

// Structured bindings (C++17)
std::tuple<int, std::string, double> get_data() {
    return std::make_tuple(42, "hello", 3.14);
}

int main() {
    // Auto type deduction
    auto numbers = std::vector<int>{1, 2, 3, 4, 5};
    auto name = std::string("World");
    
    // Range-based for loop
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // Structured bindings (C++17)
    auto [id, message, value] = get_data();
    std::cout << "ID: " << id << ", Message: " << message << ", Value: " << value << std::endl;
    
    // Optional (C++17)
    std::optional<int> maybe_value = 100;
    if (maybe_value) {
        std::cout << "Optional value: " << *maybe_value << std::endl;
    }
    
    // Variant (C++17)
    std::variant<int, std::string, double> var_value = 42;
    std::cout << "Variant holds int: " << std::get<int>(var_value) << std::endl;
    
    var_value = std::string("Hello");
    std::cout << "Variant holds string: " << std::get<std::string>(var_value) << std::endl;
    
    // Any (C++17)
    std::any any_value = 123;
    std::cout << "Any value: " << std::any_cast<int>(any_value) << std::endl;
    
    return 0;
}
```

**Modern C++ Debugging:**
```bash
# Debug auto type deduction
(gdb) whatis numbers
(gdb) ptype numbers
(gdb) whatis name
(gdb) ptype name

# Debug structured bindings
(gdb) print id
(gdb) print message
(gdb) print value
(gdb) whatis id
(gdb) ptype message

# Debug std::optional
(gdb) print maybe_value
(gdb) print maybe_value.has_value()
(gdb) print maybe_value.value()

# Debug std::variant
(gdb) print var_value
(gdb) print var_value.index()
(gdb) call std::holds_alternative<int>(var_value)
(gdb) call std::holds_alternative<std::string>(var_value)

# Debug std::any
(gdb) print any_value
(gdb) print any_value.type().name()
```

## Best Practices and Tips

### Compilation Flags for Better Debugging

```bash
# Essential debugging flags
g++ -g -O0 -std=c++17 -Wall -Wextra -Wpedantic \
    -fno-omit-frame-pointer \
    -fno-optimize-sibling-calls \
    -ftemplate-backtrace-limit=0 \
    -o debug_program program.cpp

# For memory debugging
g++ -g -O0 -std=c++17 -fsanitize=address -fsanitize=leak \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -o debug_program program.cpp

# For thread debugging
g++ -g -O0 -std=c++17 -fsanitize=thread \
    -o debug_program program.cpp
```

### GDB Configuration for C++

Create `.gdbinit` file:
```bash
# .gdbinit configuration for C++ debugging
set print pretty on
set print object on
set print static-members on
set print vtbl on
set print demangle on
set demangle-style gnu-v3
set print sevenbit-strings off

# Python pretty printers for STL
python
import sys
sys.path.insert(0, '/usr/share/gdb/auto-load/usr/lib/x86_64-linux-gnu')
from libstdcxx.v6.printers import register_libstdcxx_printers
register_libstdcxx_printers (None)
end

# Useful aliases
define pv
    print $arg0
    print *(&$arg0)
end

define pvector
    print $arg0._M_impl._M_start@$arg0.size()
end

define pmap
    print $arg0._M_t._M_impl._M_header._M_parent
end
```

### Common C++ Debugging Pitfalls

1. **Optimized Code Issues**
   - Variables optimized away
   - Inlined functions
   - Reordered instructions

2. **Template Debugging Challenges**
   - Complex mangled names
   - Multiple instantiations
   - SFINAE failures

3. **STL Container Issues**
   - Iterator invalidation
   - Internal implementation details
   - Allocator complications

4. **Exception Handling Problems**
   - Stack unwinding
   - Exception specifications
   - RAII destruction order
