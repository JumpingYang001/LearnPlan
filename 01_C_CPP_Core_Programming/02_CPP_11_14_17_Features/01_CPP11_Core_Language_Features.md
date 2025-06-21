# C++11 Core Language Features

## Overview

C++11 introduced fundamental language improvements that modernized C++ and made it more expressive and easier to use. These features form the foundation of modern C++ programming.

## Key Features

### 1. Auto Type Deduction

The `auto` keyword allows the compiler to automatically deduce the type of a variable from its initializer.

#### Basic Usage

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <string>

int main() {
    // Basic auto usage
    auto x = 42;           // int
    auto y = 3.14;         // double
    auto z = "Hello";      // const char*
    auto str = std::string("Modern C++"); // std::string
    
    // Complex types
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin(); // std::vector<int>::iterator
    
    // With containers
    std::map<std::string, int> scores;
    scores["Alice"] = 95;
    scores["Bob"] = 87;
    
    // Instead of: std::map<std::string, int>::iterator
    for (auto it = scores.begin(); it != scores.end(); ++it) {
        std::cout << it->first << ": " << it->second << std::endl;
    }
    
    return 0;
}
```

#### Best Practices

```cpp
#include <vector>
#include <memory>

// Good: Clear intent, avoid repetition
auto numbers = std::vector<int>{1, 2, 3, 4, 5};
auto ptr = std::make_unique<int>(42);

// Good: Complex template types
std::vector<std::pair<std::string, std::vector<int>>> data;
auto it = data.begin(); // Much cleaner than explicit type

// Be careful: Don't use auto when type isn't obvious
auto flag = true;  // OK, bool is clear
auto result = calculateSomething(); // Not clear without seeing function

// Prefer explicit types for function parameters
void processData(const std::vector<int>& vec) { // Not auto
    auto size = vec.size(); // OK, size_t is clear from context
}
```

### 2. Range-Based For Loops

Simplified iteration over containers and arrays.

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <array>

int main() {
    // Traditional array
    int arr[] = {1, 2, 3, 4, 5};
    for (int value : arr) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    // STL containers
    std::vector<std::string> names = {"Alice", "Bob", "Charlie"};
    for (const auto& name : names) {
        std::cout << "Hello, " << name << std::endl;
    }
    
    // Modifying elements
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    for (auto& num : numbers) {
        num *= 2;  // Double each number
    }
    
    // Map iteration
    std::map<std::string, int> ages = {{"Alice", 25}, {"Bob", 30}};
    for (const auto& [name, age] : ages) { // C++17 structured binding
        std::cout << name << " is " << age << " years old" << std::endl;
    }
    
    return 0;
}
```

### 3. Lambda Expressions

Anonymous functions that can capture variables from their surrounding scope.

#### Basic Lambda Syntax

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    // Basic lambda
    auto simple_lambda = []() {
        std::cout << "Hello from lambda!" << std::endl;
    };
    simple_lambda();
    
    // Lambda with parameters
    auto add = [](int a, int b) {
        return a + b;
    };
    std::cout << "5 + 3 = " << add(5, 3) << std::endl;
    
    // Lambda with explicit return type
    auto divide = [](double a, double b) -> double {
        return a / b;
    };
    
    return 0;
}
```

#### Capture Modes

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    int x = 10;
    int y = 20;
    
    // Capture by value
    auto lambda1 = [x](int param) {
        return x + param; // x is copied
    };
    
    // Capture by reference
    auto lambda2 = [&y](int param) {
        y += param; // y is modified
        return y;
    };
    
    // Capture all by value
    auto lambda3 = [=](int param) {
        return x + y + param; // All variables copied
    };
    
    // Capture all by reference
    auto lambda4 = [&](int param) {
        x += param;
        y += param;
        return x + y;
    };
    
    // Mixed capture
    auto lambda5 = [x, &y](int param) {
        y += param;  // y by reference
        return x + y; // x by value
    };
    
    // Using lambdas with algorithms
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Find first even number
    auto it = std::find_if(numbers.begin(), numbers.end(), 
                          [](int n) { return n % 2 == 0; });
    
    if (it != numbers.end()) {
        std::cout << "First even number: " << *it << std::endl;
    }
    
    // Transform elements
    std::vector<int> squared;
    std::transform(numbers.begin(), numbers.end(), 
                   std::back_inserter(squared),
                   [](int n) { return n * n; });
    
    return 0;
}
```

### 4. nullptr

Type-safe null pointer constant.

```cpp
#include <iostream>
#include <memory>

void process(int* ptr) {
    if (ptr != nullptr) {
        std::cout << "Processing integer: " << *ptr << std::endl;
    } else {
        std::cout << "Null pointer received" << std::endl;
    }
}

void process(int value) {
    std::cout << "Processing value: " << value << std::endl;
}

int main() {
    // Old way - problematic
    // int* ptr1 = NULL;  // NULL is typically defined as 0
    // process(NULL);     // Ambiguous! Could call process(int) or process(int*)
    
    // New way - type safe
    int* ptr2 = nullptr;
    process(nullptr);     // Unambiguous - calls process(int*)
    
    // nullptr vs NULL
    auto p1 = nullptr;    // p1 is std::nullptr_t
    // auto p2 = NULL;    // p2 is int (usually)
    
    // Template disambiguation
    template<typename T>
    void template_func(T* ptr) {
        if (ptr == nullptr) {
            std::cout << "Template received null pointer" << std::endl;
        }
    }
    
    template_func<int>(nullptr); // Clear and unambiguous
    
    return 0;
}
```

### 5. Strongly-Typed Enumerations (enum class)

Scoped enumerations that don't implicitly convert to integers.

```cpp
#include <iostream>

// Old style enum - problematic
enum Color { RED, GREEN, BLUE };
enum Traffic { RED_LIGHT, YELLOW_LIGHT, GREEN_LIGHT }; // Name collision!

// New style enum class
enum class Status { PENDING, APPROVED, REJECTED };
enum class Priority { LOW, MEDIUM, HIGH };

// Enum class with specific underlying type
enum class ErrorCode : int { SUCCESS = 0, INVALID_INPUT = 1, NETWORK_ERROR = 2 };

int main() {
    // Old enum issues
    // int x = RED;  // Implicit conversion
    // if (RED == RED_LIGHT) { } // Compilation error due to redefinition
    
    // New enum class - type safe
    Status status = Status::PENDING;
    Priority priority = Priority::HIGH;
    
    // No implicit conversion
    // int y = Status::PENDING;  // Compilation error
    int y = static_cast<int>(Status::PENDING); // Explicit conversion required
    
    // Comparison
    if (status == Status::PENDING) {
        std::cout << "Status is pending" << std::endl;
    }
    
    // Switch statement
    switch (priority) {
        case Priority::LOW:
            std::cout << "Low priority task" << std::endl;
            break;
        case Priority::MEDIUM:
            std::cout << "Medium priority task" << std::endl;
            break;
        case Priority::HIGH:
            std::cout << "High priority task" << std::endl;
            break;
    }
    
    // Underlying type access
    ErrorCode error = ErrorCode::SUCCESS;
    int error_value = static_cast<int>(error);
    std::cout << "Error code: " << error_value << std::endl;
    
    return 0;
}
```

### 6. Static Assertions

Compile-time assertions for template metaprogramming and validation.

```cpp
#include <iostream>
#include <type_traits>
#include <vector>

template<typename T>
class SafeArray {
    static_assert(std::is_arithmetic<T>::value, 
                  "SafeArray only supports arithmetic types");
    
    static_assert(sizeof(T) <= 8, 
                  "SafeArray doesn't support types larger than 8 bytes");
    
private:
    std::vector<T> data;
    
public:
    void add(T value) {
        data.push_back(value);
    }
    
    T get(size_t index) const {
        if (index >= data.size()) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }
};

// Compile-time size checking
template<size_t N>
void process_array() {
    static_assert(N > 0, "Array size must be positive");
    static_assert(N <= 1000, "Array size too large");
    
    std::cout << "Processing array of size " << N << std::endl;
}

int main() {
    // These will compile
    SafeArray<int> int_array;
    SafeArray<double> double_array;
    
    // These would cause compilation errors:
    // SafeArray<std::string> string_array;  // Not arithmetic
    // SafeArray<long double> ld_array;      // Too large (typically 16 bytes)
    
    process_array<10>();   // OK
    process_array<500>();  // OK
    // process_array<0>();    // Compilation error
    // process_array<2000>(); // Compilation error
    
    return 0;
}
```

### 7. Delegating Constructors

Constructors that call other constructors in the same class.

```cpp
#include <iostream>
#include <string>
#include <vector>

class Person {
private:
    std::string name;
    int age;
    std::string email;
    
public:
    // Primary constructor
    Person(const std::string& name, int age, const std::string& email)
        : name(name), age(age), email(email) {
        std::cout << "Full constructor called" << std::endl;
        validate();
    }
    
    // Delegating constructors
    Person(const std::string& name, int age) 
        : Person(name, age, "") {  // Delegate to primary constructor
        std::cout << "Name+Age constructor called" << std::endl;
    }
    
    Person(const std::string& name) 
        : Person(name, 0, "") {    // Delegate to primary constructor
        std::cout << "Name-only constructor called" << std::endl;
    }
    
    Person() 
        : Person("Unknown", 0, "") {
        std::cout << "Default constructor called" << std::endl;
    }
    
    void validate() {
        if (age < 0) {
            throw std::invalid_argument("Age cannot be negative");
        }
    }
    
    void display() const {
        std::cout << "Name: " << name << ", Age: " << age 
                  << ", Email: " << email << std::endl;
    }
};

class Buffer {
private:
    std::vector<char> data;
    size_t capacity;
    
public:
    // Primary constructor
    Buffer(size_t size, char fill_value) 
        : data(size, fill_value), capacity(size) {
        std::cout << "Buffer created with size " << size 
                  << " and fill value '" << fill_value << "'" << std::endl;
    }
    
    // Delegating constructors
    Buffer(size_t size) : Buffer(size, '\0') {
        std::cout << "Buffer created with default fill" << std::endl;
    }
    
    Buffer() : Buffer(1024) {
        std::cout << "Default buffer created" << std::endl;
    }
    
    size_t size() const { return data.size(); }
    char* data_ptr() { return data.data(); }
};

int main() {
    std::cout << "Creating persons:" << std::endl;
    Person p1("Alice", 25, "alice@example.com");
    Person p2("Bob", 30);
    Person p3("Charlie");
    Person p4;
    
    std::cout << "\nPersons created:" << std::endl;
    p1.display();
    p2.display();
    p3.display();
    p4.display();
    
    std::cout << "\nCreating buffers:" << std::endl;
    Buffer b1(512, 'X');
    Buffer b2(256);
    Buffer b3;
    
    return 0;
}
```

### 8. Initializer Lists

Uniform initialization syntax for containers and objects.

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <initializer_list>

class Point {
private:
    double x, y;
    
public:
    Point(double x, double y) : x(x), y(y) {}
    
    // Constructor taking initializer list
    Point(std::initializer_list<double> coords) {
        auto it = coords.begin();
        x = (it != coords.end()) ? *it++ : 0.0;
        y = (it != coords.end()) ? *it : 0.0;
    }
    
    void display() const {
        std::cout << "(" << x << ", " << y << ")" << std::endl;
    }
};

class Matrix {
private:
    std::vector<std::vector<double>> data;
    
public:
    Matrix(std::initializer_list<std::initializer_list<double>> rows) {
        for (const auto& row : rows) {
            data.emplace_back(row);
        }
    }
    
    void display() const {
        for (const auto& row : data) {
            for (double val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
};

int main() {
    // Container initialization
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::vector<std::string> names = {"Alice", "Bob", "Charlie"};
    
    // Map initialization
    std::map<std::string, int> ages = {
        {"Alice", 25},
        {"Bob", 30},
        {"Charlie", 35}
    };
    
    // Set initialization
    std::set<int> unique_numbers = {3, 1, 4, 1, 5, 9, 2, 6, 5};
    
    // Custom class initialization
    Point p1(3.0, 4.0);          // Regular constructor
    Point p2{5.0, 6.0};          // Uniform initialization
    Point p3 = {7.0, 8.0};       // Copy-list-initialization
    Point p4{1.0};               // Initializer list constructor
    
    std::cout << "Points:" << std::endl;
    p1.display();
    p2.display();
    p3.display();
    p4.display();
    
    // Matrix initialization
    Matrix m = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    
    std::cout << "\nMatrix:" << std::endl;
    m.display();
    
    // Function with initializer list
    auto print_list = [](std::initializer_list<int> list) {
        for (int value : list) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    };
    
    print_list({10, 20, 30, 40, 50});
    
    return 0;
}
```

### 9. Uniform Initialization

Consistent initialization syntax across different contexts.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <complex>

struct Config {
    std::string name;
    int version;
    bool enabled;
    
    Config(const std::string& n, int v, bool e) 
        : name(n), version(v), enabled(e) {}
};

int main() {
    // Uniform initialization examples
    
    // Built-in types
    int a{42};
    double b{3.14};
    char c{'A'};
    
    // Arrays
    int arr[]{1, 2, 3, 4, 5};
    
    // STL containers
    std::vector<int> vec{1, 2, 3, 4, 5};
    std::string str{"Hello, World!"};
    
    // Complex types
    std::complex<double> comp{1.0, 2.0};
    
    // Custom types
    Config config{"MyApp", 1, true};
    
    // Preventing narrowing conversions
    int x = 42;
    // int y{3.14};  // Compilation error - narrowing conversion
    int y{static_cast<int>(3.14)}; // Explicit conversion
    
    // Most vexing parse resolution
    // Widget w();    // Function declaration (most vexing parse)
    // Widget w{};    // Object creation
    
    // Uniform initialization in member initialization lists
    struct Person {
        std::string name;
        std::vector<int> scores;
        
        Person(const std::string& n) : name{n}, scores{95, 87, 92} {}
    };
    
    Person person{"Alice"};
    
    std::cout << "Uniform initialization examples completed" << std::endl;
    
    return 0;
}
```

## Exercises

### Exercise 1: Auto Usage Practice
Write a function that processes a map of strings to vectors of integers, using `auto` appropriately throughout.

### Exercise 2: Lambda Calculator
Create a calculator using lambdas for different operations (add, subtract, multiply, divide).

### Exercise 3: Safe Enum System
Design a traffic light system using enum classes that prevents invalid state transitions.

### Exercise 4: Resource Manager
Create a resource manager class using delegating constructors and uniform initialization.

## Common Pitfalls

1. **Overusing auto**: Don't use auto when the type isn't obvious
2. **Lambda captures**: Be careful with capture by reference and object lifetime
3. **Enum scope**: Remember that enum class requires scope resolution
4. **Initializer list ambiguity**: Be aware of potential ambiguity with constructors

## Performance Considerations

- Lambda expressions have zero overhead when they don't capture anything
- Range-based for loops are as efficient as traditional loops
- Uniform initialization can help avoid unnecessary temporaries
- Auto deduction happens at compile time with no runtime cost

## Best Practices

1. Use `auto` for complex types and when the type is obvious from context
2. Prefer range-based for loops when you don't need indices
3. Use lambda expressions for short, local functions
4. Always use `nullptr` instead of `NULL` or `0` for pointers
5. Use enum classes for better type safety
6. Leverage uniform initialization for consistency and safety

## Summary

C++11 core language features modernized C++ significantly, making code more readable, safer, and more expressive. These features form the foundation for modern C++ programming and should be mastered before moving on to more advanced topics.
