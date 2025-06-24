# C++11 Core Language Features

*Duration: 2 weeks*

## Overview

C++11 represents one of the most significant updates to the C++ language since its inception. Released in 2011, it introduced fundamental language improvements that modernized C++ and made it more expressive, safer, and easier to use. These features form the foundation of modern C++ programming and are essential for any C++ developer.

### Why C++11 Matters

Before C++11, C++ developers often struggled with:
- Verbose and error-prone code
- Lack of type safety in certain areas
- Cumbersome iteration patterns
- Inconsistent initialization syntax
- Difficulty with generic programming

C++11 addressed these issues and introduced features that make C++ competitive with modern languages while maintaining its performance characteristics.

### Key Improvements Overview

| Feature | Problem Solved | Benefit |
|---------|----------------|---------|
| `auto` | Verbose type declarations | Cleaner, more maintainable code |
| Range-based for | Complex iterator syntax | Simpler iteration patterns |
| Lambdas | Lack of local functions | Functional programming support |
| `nullptr` | Type-unsafe null pointers | Better type safety |
| `enum class` | Global enum pollution | Scoped, type-safe enumerations |
| Uniform initialization | Inconsistent syntax | Consistent initialization patterns |

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
#include <algorithm>
#include <map>

// Good: Clear intent, avoid repetition
auto numbers = std::vector<int>{1, 2, 3, 4, 5};
auto ptr = std::make_unique<int>(42);

// Good: Complex template types
std::vector<std::pair<std::string, std::vector<int>>> data;
auto it = data.begin(); // Much cleaner than explicit type

// Good: Function return types that are obvious from context
auto calculate_average(const std::vector<double>& values) -> double {
    auto sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

// Be careful: Don't use auto when type isn't obvious
auto flag = true;  // OK, bool is clear
auto result = calculateSomething(); // Not clear without seeing function

// Bad: Losing important type information
auto timeout = 1000;  // Is this milliseconds? seconds? What type?
std::chrono::milliseconds timeout{1000}; // Much better

// Prefer explicit types for function parameters
void processData(const std::vector<int>& vec) { // Not auto
    auto size = vec.size(); // OK, size_t is clear from context
}

// Advanced: Using auto with trailing return types
template<typename T, typename U>
auto multiply(T t, U u) -> decltype(t * u) {
    return t * u;
}

// Advanced: Auto in templates
template<typename Container>
void print_container(const Container& c) {
    for (const auto& element : c) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}
```

#### Performance Considerations

`auto` is a compile-time feature with zero runtime overhead:

```cpp
#include <chrono>
#include <iostream>

void performance_comparison() {
    const size_t ITERATIONS = 1000000;
    std::vector<int> large_vector(ITERATIONS, 42);
    
    // Both versions compile to identical assembly code
    auto start1 = std::chrono::high_resolution_clock::now();
    for (std::vector<int>::iterator it = large_vector.begin(); 
         it != large_vector.end(); ++it) {
        // Process element
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    
    auto start2 = std::chrono::high_resolution_clock::now();
    for (auto it = large_vector.begin(); it != large_vector.end(); ++it) {
        // Process element
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    
    // Times will be virtually identical
    std::cout << "Explicit type: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() 
              << " microseconds\n";
    std::cout << "Auto type: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() 
              << " microseconds\n";
}

### 2. Range-Based For Loops

Simplified iteration over containers and arrays, eliminating common iterator-related bugs and making code more readable.

#### Why Range-Based For Loops?

**Before C++11 (Problematic):**
```cpp
// Error-prone: easy to make mistakes with iterators
for (std::vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {
    std::cout << *it << " ";
}

// Array iteration was inconsistent
int arr[] = {1, 2, 3, 4, 5};
for (int i = 0; i < sizeof(arr)/sizeof(arr[0]); ++i) {
    std::cout << arr[i] << " ";
}
```

**C++11 Solution:**
```cpp
#include <iostream>
#include <vector>
#include <map>
#include <array>
#include <list>

int main() {
    // Traditional array
    int arr[] = {1, 2, 3, 4, 5};
    for (int value : arr) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    // STL containers
    std::vector<std::string> names = {"Alice", "Bob", "Charlie"};
    for (const auto& name : names) {  // const auto& avoids copying
        std::cout << "Hello, " << name << std::endl;
    }
    
    // Modifying elements
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    for (auto& num : numbers) {  // auto& allows modification
        num *= 2;  // Double each number
    }
    
    // Display modified numbers
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### Advanced Usage Patterns

```cpp
#include <map>
#include <set>
#include <string>
#include <iostream>

void advanced_range_based_loops() {
    // Map iteration (C++11 way)
    std::map<std::string, int> ages = {{"Alice", 25}, {"Bob", 30}, {"Charlie", 35}};
    
    for (const auto& pair : ages) {
        std::cout << pair.first << " is " << pair.second << " years old\n";
    }
    
    // Set iteration
    std::set<std::string> unique_names = {"Alice", "Bob", "Alice", "Charlie"};
    for (const auto& name : unique_names) {
        std::cout << "Unique name: " << name << std::endl;
    }
    
    // Nested containers
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
    
    // Working with indices when needed
    std::vector<std::string> items = {"apple", "banana", "cherry"};
    size_t index = 0;
    for (const auto& item : items) {
        std::cout << "Item " << index++ << ": " << item << std::endl;
    }
}
```

#### Performance Comparison

```cpp
#include <chrono>
#include <vector>
#include <numeric>

void performance_test() {
    const size_t SIZE = 10000000;
    std::vector<int> data(SIZE);
    std::iota(data.begin(), data.end(), 1); // Fill with 1, 2, 3, ...
    
    // Traditional iterator approach
    auto start1 = std::chrono::high_resolution_clock::now();
    long long sum1 = 0;
    for (auto it = data.begin(); it != data.end(); ++it) {
        sum1 += *it;
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    
    // Range-based for loop
    auto start2 = std::chrono::high_resolution_clock::now();
    long long sum2 = 0;
    for (const auto& value : data) {
        sum2 += value;
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    
    // Results should be identical, performance should be nearly identical
    std::cout << "Iterator sum: " << sum1 << ", Time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() 
              << "ms\n";
    std::cout << "Range-based sum: " << sum2 << ", Time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() 
              << "ms\n";
}
```

#### Custom Types with Range-Based For

```cpp
#include <iostream>

// Making your own types work with range-based for
class NumberSequence {
private:
    int start, end;
    
public:
    NumberSequence(int s, int e) : start(s), end(e) {}
    
    // Iterator class
    class iterator {
    private:
        int current;
    public:
        iterator(int val) : current(val) {}
        
        int operator*() const { return current; }
        iterator& operator++() { ++current; return *this; }
        bool operator!=(const iterator& other) const { 
            return current != other.current; 
        }
    };
    
    iterator begin() const { return iterator(start); }
    iterator end() const { return iterator(end); }
};

int main() {
    NumberSequence seq(1, 6);
    
    // Now this works!
    for (int num : seq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```    std::map<std::string, int> ages = {{"Alice", 25}, {"Bob", 30}};
    for (const auto& pair : ages) {
        std::cout << pair.first << " is " << pair.second << " years old" << std::endl;
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
#include <functional>

int main() {
    int x = 10;
    int y = 20;
    
    // Capture by value [=] - creates copies
    auto lambda1 = [x](int param) {
        // x is a copy - original x is unchanged
        return x + param; 
    };
    
    // Capture by reference [&] - references original variables
    auto lambda2 = [&y](int param) {
        y += param; // y is modified in the outer scope
        return y;
    };
    
    // Capture all by value [=]
    auto lambda3 = [=](int param) {
        // All variables in scope are copied
        return x + y + param; 
    };
    
    // Capture all by reference [&]
    auto lambda4 = [&](int param) {
        x += param;  // Modifies original x
        y += param;  // Modifies original y
        return x + y;
    };
    
    // Mixed capture - specify individual variables
    auto lambda5 = [x, &y](int param) {
        y += param;   // y by reference (modified)
        return x + y; // x by value (copy)
    };
    
    // Mutable lambdas - modify captured values
    auto lambda6 = [x](int param) mutable {
        x += param;   // Can modify the copy
        return x;     // Original x unchanged
    };
    
    std::cout << "Original x: " << x << ", y: " << y << std::endl;
    
    std::cout << "Lambda1 result: " << lambda1(5) << std::endl;
    std::cout << "After lambda1 - x: " << x << ", y: " << y << std::endl;
    
    std::cout << "Lambda2 result: " << lambda2(3) << std::endl;
    std::cout << "After lambda2 - x: " << x << ", y: " << y << std::endl;
    
    return 0;
}
```

#### Practical Lambda Applications

```cpp
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

void practical_lambda_examples() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 1. Filtering with std::copy_if
    std::vector<int> evens;
    std::copy_if(numbers.begin(), numbers.end(), std::back_inserter(evens),
                 [](int n) { return n % 2 == 0; });
    
    // 2. Transformation with std::transform
    std::vector<int> squares;
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(squares),
                   [](int n) { return n * n; });
    
    // 3. Custom comparator for sorting
    std::vector<std::string> words = {"apple", "pie", "cherry", "a"};
    std::sort(words.begin(), words.end(), 
              [](const std::string& a, const std::string& b) {
                  return a.length() < b.length(); // Sort by length
              });
    
    // 4. Accumulation with custom operation
    int sum_of_squares = std::accumulate(numbers.begin(), numbers.end(), 0,
                                        [](int acc, int n) { 
                                            return acc + n * n; 
                                        });
    
    // 5. Complex capture example - closure
    int threshold = 5;
    auto create_filter = [threshold](const std::string& prefix) {
        return [threshold, prefix](int value) {
            std::cout << prefix << ": ";
            return value > threshold;
        };
    };
    
    auto filter_func = create_filter("Checking");
    auto it = std::find_if(numbers.begin(), numbers.end(), filter_func);
    
    std::cout << "First number > " << threshold << ": " << *it << std::endl;
}
```

#### Lambda as Function Objects

```cpp
#include <functional>
#include <map>
#include <iostream>

class Calculator {
private:
    std::map<std::string, std::function<double(double, double)>> operations;
    
public:
    Calculator() {
        // Store lambdas as function objects
        operations["add"] = [](double a, double b) { return a + b; };
        operations["subtract"] = [](double a, double b) { return a - b; };
        operations["multiply"] = [](double a, double b) { return a * b; };
        operations["divide"] = [](double a, double b) -> double {
            if (b == 0) throw std::runtime_error("Division by zero");
            return a / b;
        };
        
        // More complex lambda with capture
        operations["power"] = [](double base, double exp) {
            return std::pow(base, exp);
        };
    }
    
    double calculate(const std::string& op, double a, double b) {
        auto it = operations.find(op);
        if (it != operations.end()) {
            return it->second(a, b);
        }
        throw std::runtime_error("Unknown operation: " + op);
    }
    
    void add_operation(const std::string& name, 
                      std::function<double(double, double)> func) {
        operations[name] = func;
    }
};

void lambda_function_objects_demo() {
    Calculator calc;
    
    std::cout << "5 + 3 = " << calc.calculate("add", 5, 3) << std::endl;
    std::cout << "10 / 2 = " << calc.calculate("divide", 10, 2) << std::endl;
    
    // Add custom operation using lambda
    calc.add_operation("modulo", [](double a, double b) { 
        return std::fmod(a, b); 
    });
    
    std::cout << "10 % 3 = " << calc.calculate("modulo", 10, 3) << std::endl;
}
```

#### Performance Considerations

```cpp
#include <chrono>
#include <vector>
#include <algorithm>

// Traditional function
bool is_even_function(int n) {
    return n % 2 == 0;
}

void lambda_performance_test() {
    const size_t SIZE = 10000000;
    std::vector<int> data(SIZE);
    std::iota(data.begin(), data.end(), 1);
    
    // Function pointer
    auto start1 = std::chrono::high_resolution_clock::now();
    auto count1 = std::count_if(data.begin(), data.end(), is_even_function);
    auto end1 = std::chrono::high_resolution_clock::now();
    
    // Lambda (can be inlined by compiler)
    auto start2 = std::chrono::high_resolution_clock::now();
    auto count2 = std::count_if(data.begin(), data.end(), 
                               [](int n) { return n % 2 == 0; });
    auto end2 = std::chrono::high_resolution_clock::now();
    
    // Lambda with capture (slight overhead)
    int divisor = 2;
    auto start3 = std::chrono::high_resolution_clock::now();
    auto count3 = std::count_if(data.begin(), data.end(), 
                               [divisor](int n) { return n % divisor == 0; });
    auto end3 = std::chrono::high_resolution_clock::now();
    
    std::cout << "Function pointer: " << count1 << " evens, Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() 
              << "ms\n";
    std::cout << "Lambda (no capture): " << count2 << " evens, Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() 
              << "ms\n";
    std::cout << "Lambda (with capture): " << count3 << " evens, Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count() 
              << "ms\n";
}
    
    return 0;
}
```

### 4. nullptr

Type-safe null pointer constant that eliminates ambiguity and improves code safety.

#### The Problem with NULL

Before C++11, using `NULL` or `0` for null pointers created several issues:

```cpp
#include <iostream>

// Function overloads that create ambiguity
void process(int value) {
    std::cout << "Processing integer: " << value << std::endl;
}

void process(int* ptr) {
    if (ptr != nullptr) {
        std::cout << "Processing pointer to integer: " << *ptr << std::endl;
    } else {
        std::cout << "Null pointer received" << std::endl;
    }
}

void process(char* ptr) {
    if (ptr != nullptr) {
        std::cout << "Processing string: " << ptr << std::endl;
    } else {
        std::cout << "Null string pointer received" << std::endl;
    }
}

void demonstrate_null_problems() {
    // Problem 1: NULL is typically defined as 0 or ((void*)0)
    // This creates ambiguity in function calls
    
    // process(NULL);     // ERROR: Ambiguous call - could match any overload
    // process(0);        // ERROR: Calls process(int), not process(int*)
    
    // Problem 2: Template issues
    template<typename T>
    void template_func(T* ptr) {
        if (ptr == nullptr) {  // This works
            std::cout << "Template received null pointer" << std::endl;
        }
        // if (ptr == NULL) { } // This might not work in all template contexts
    }
}
```

#### The nullptr Solution

```cpp
#include <iostream>
#include <memory>
#include <type_traits>

int main() {
    // nullptr is type-safe and unambiguous
    int* ptr1 = nullptr;      // Clear and unambiguous
    char* ptr2 = nullptr;     // Works with any pointer type
    double* ptr3 = nullptr;   // No casting needed
    
    // Function calls are now unambiguous
    process(nullptr);         // Clearly calls process(int*)
    process(42);             // Clearly calls process(int)
    
    // Template usage is clean
    template_func<int>(nullptr);     // Clear and works
    template_func<char>(nullptr);    // Works with any type
    
    // Type information
    auto null_ptr = nullptr;         // Type is std::nullptr_t
    std::cout << "nullptr type: " << typeid(null_ptr).name() << std::endl;
    
    // Comparison operations
    if (ptr1 == nullptr) {
        std::cout << "ptr1 is null" << std::endl;
    }
    
    if (ptr1 != nullptr) {
        std::cout << "ptr1 is not null" << std::endl;
    }
    
    return 0;
}
```

#### Advanced nullptr Usage

```cpp
#include <memory>
#include <functional>
#include <iostream>

// Smart pointer usage
void smart_pointer_with_nullptr() {
    // unique_ptr with nullptr
    std::unique_ptr<int> ptr1 = nullptr;
    std::unique_ptr<int> ptr2(nullptr);
    
    if (!ptr1) {  // Preferred way to check
        std::cout << "ptr1 is empty" << std::endl;
    }
    
    if (ptr2 == nullptr) {  // Also valid
        std::cout << "ptr2 is empty" << std::endl;
    }
    
    // shared_ptr with nullptr
    std::shared_ptr<std::string> str_ptr = nullptr;
    str_ptr.reset();  // Equivalent to setting to nullptr
    
    // Function pointers
    std::function<void()> func = nullptr;
    if (func == nullptr) {
        std::cout << "Function is not set" << std::endl;
    }
}

// Template metaprogramming with nullptr
template<typename T>
class SafePointer {
private:
    T* ptr;
    
public:
    SafePointer() : ptr(nullptr) {}
    SafePointer(T* p) : ptr(p) {}
    SafePointer(std::nullptr_t) : ptr(nullptr) {}  // Explicit nullptr constructor
    
    // Assignment operators
    SafePointer& operator=(T* p) { ptr = p; return *this; }
    SafePointer& operator=(std::nullptr_t) { ptr = nullptr; return *this; }
    
    // Comparison operators
    bool operator==(std::nullptr_t) const { return ptr == nullptr; }
    bool operator!=(std::nullptr_t) const { return ptr != nullptr; }
    
    // Conversion operators
    explicit operator bool() const { return ptr != nullptr; }
    
    T& operator*() const {
        if (ptr == nullptr) {
            throw std::runtime_error("Dereferencing null pointer");
        }
        return *ptr;
    }
    
    T* operator->() const {
        if (ptr == nullptr) {
            throw std::runtime_error("Dereferencing null pointer");
        }
        return ptr;
    }
    
    T* get() const { return ptr; }
    
    void reset(T* new_ptr = nullptr) {
        delete ptr;
        ptr = new_ptr;
    }
    
    ~SafePointer() { delete ptr; }
};

void safe_pointer_demo() {
    SafePointer<int> safe_ptr = nullptr;
    
    if (safe_ptr == nullptr) {
        std::cout << "SafePointer is null" << std::endl;
    }
    
    safe_ptr = new int(42);
    
    if (safe_ptr != nullptr) {
        std::cout << "SafePointer value: " << *safe_ptr << std::endl;
    }
    
    safe_ptr = nullptr;  // Automatic cleanup
}
```

#### Performance and Memory Impact

```cpp
#include <chrono>
#include <vector>

void nullptr_performance_test() {
    const size_t SIZE = 10000000;
    std::vector<int*> pointers(SIZE);
    
    // Initialize with nullptr (same performance as NULL)
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < SIZE; ++i) {
        pointers[i] = nullptr;
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "nullptr initialization time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " microseconds" << std::endl;
    
    // Check for null (same performance as NULL check)
    start = std::chrono::high_resolution_clock::now();
    size_t null_count = 0;
    for (const auto& ptr : pointers) {
        if (ptr == nullptr) {
            ++null_count;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    
    std::cout << "nullptr checking time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " microseconds" << std::endl;
    std::cout << "Null pointers found: " << null_count << std::endl;
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
Create a comprehensive data processing system that demonstrates proper `auto` usage:

```cpp
// TODO: Complete this data processor
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

class DataProcessor {
public:
    // Use auto appropriately to process map of vectors
    void process_data(const std::map<std::string, std::vector<int>>& data) {
        // TODO: Use auto to iterate through the map
        // TODO: Calculate statistics for each vector
        // TODO: Use auto with algorithm calls
    }
    
    // TODO: Create a function that returns a complex type using auto
    auto create_lookup_table() {
        // Return type should be deduced
    }
};
```

### Exercise 2: Lambda Calculator
Build a flexible calculator using lambdas and function objects:

```cpp
// TODO: Complete this lambda-based calculator
class AdvancedCalculator {
private:
    std::map<std::string, std::function<double(double, double)>> binary_ops;
    std::map<std::string, std::function<double(double)>> unary_ops;
    
public:
    AdvancedCalculator() {
        // TODO: Initialize with basic operations using lambdas
        // TODO: Add operations like sin, cos, log using lambdas
    }
    
    // TODO: Add method to register custom operations
    void add_binary_operation(const std::string& name, /* lambda parameter */) {
        // TODO: Store lambda in binary_ops
    }
    
    // TODO: Create calculator chain using lambda captures
    auto create_calculation_chain(double initial_value) {
        // TODO: Return a lambda that can chain operations
    }
};
```

### Exercise 3: Safe Enum System
Design a comprehensive state machine using enum classes:

```cpp
// TODO: Create a traffic light controller with safe state transitions
enum class TrafficLight { /* TODO */ };
enum class PedestrianSignal { /* TODO */ };
enum class EmergencyMode { /* TODO */ };

class TrafficController {
private:
    TrafficLight current_light;
    PedestrianSignal pedestrian_signal;
    EmergencyMode emergency_mode;
    
public:
    // TODO: Implement safe state transitions
    bool transition_to(TrafficLight new_state);
    
    // TODO: Handle emergency mode transitions
    void activate_emergency_mode(EmergencyMode mode);
    
    // TODO: Use static_assert to validate enum sizes
};
```

### Exercise 4: Resource Manager with Modern C++11
Create a comprehensive resource management system:

```cpp
// TODO: Implement using delegating constructors, uniform initialization, nullptr
class ResourceManager {
private:
    std::vector<std::unique_ptr<Resource>> resources;
    std::map<std::string, size_t> resource_index;
    
public:
    // TODO: Multiple constructors using delegation
    ResourceManager(/* various parameter combinations */);
    
    // TODO: Use uniform initialization for resource creation
    void create_resource(std::initializer_list</* TODO */> params);
    
    // TODO: Safe resource access using nullptr checks
    Resource* get_resource(const std::string& name);
    
    // TODO: Range-based for loop for resource iteration
    template<typename Func>
    void for_each_resource(Func func) {
        // TODO: Apply func to each resource
    }
};
```

### Exercise 5: Performance Benchmark Suite
Create benchmarks comparing C++11 features with older approaches:

```cpp
// TODO: Implement performance comparisons
class PerformanceBenchmark {
public:
    // TODO: Compare auto vs explicit types (compile time)
    void benchmark_auto_performance();
    
    // TODO: Compare range-based for vs traditional loops
    void benchmark_iteration_methods();
    
    // TODO: Compare lambda vs function pointers
    void benchmark_function_objects();
    
    // TODO: Compare nullptr vs NULL performance
    void benchmark_null_checks();
    
    // TODO: Create comprehensive report
    void generate_report();
};
```

## Common Pitfalls and How to Avoid Them

### 1. Overusing `auto`

**❌ Bad Examples:**
```cpp
auto timeout = 1000;        // What unit? int? milliseconds?
auto result = getValue();   // What type is returned?
auto data = getImportantData(); // Unclear without reading function
```

**✅ Good Examples:**
```cpp
std::chrono::milliseconds timeout{1000};  // Clear unit and type
auto result = static_cast<int>(getValue()); // Clear intent
auto complex_iter = container.begin();     // Obviously an iterator
```

### 2. Lambda Capture Pitfalls

**❌ Dangerous Reference Captures:**
```cpp
std::function<int()> create_problematic_lambda() {
    int local_var = 42;
    
    // DANGER: Capturing local variable by reference
    return [&local_var]() { 
        return local_var; // local_var is destroyed when function returns!
    };
}

// DANGER: Capturing 'this' when object might be destroyed
class MyClass {
    int value = 100;
public:
    std::function<int()> get_lambda() {
        return [this]() { return value; }; // 'this' might become invalid
    }
};
```

**✅ Safe Capture Patterns:**
```cpp
std::function<int()> create_safe_lambda() {
    int local_var = 42;
    
    // SAFE: Capture by value
    return [local_var]() { 
        return local_var; // Copy is safe
    };
}

class MyClass {
    int value = 100;
public:
    std::function<int()> get_safe_lambda() {
        return [value = this->value]() { return value; }; // C++14 init capture
        // Or for C++11:
        // int captured_value = value;
        // return [captured_value]() { return captured_value; };
    }
};
```

### 3. Enum Class Scope Confusion

**❌ Common Mistakes:**
```cpp
enum class Color { RED, GREEN, BLUE };

int main() {
    // Color color = RED;           // ERROR: RED not in scope
    // if (color == 0) { }          // ERROR: No implicit conversion
    // Color mix = RED + GREEN;     // ERROR: No arithmetic operations
    
    Color color = Color::RED;       // Correct
    if (color == Color::RED) { }    // Correct
}
```

**✅ Best Practices:**
```cpp
enum class Status : uint8_t { PENDING = 1, APPROVED = 2, REJECTED = 3 };

// Use switch for comprehensive handling
std::string status_to_string(Status s) {
    switch (s) {
        case Status::PENDING:  return "Pending";
        case Status::APPROVED: return "Approved";
        case Status::REJECTED: return "Rejected";
        // No default: compiler will warn if cases are missing
    }
    return "Unknown"; // Unreachable but satisfies compiler
}
```

### 4. Initializer List Ambiguity

**❌ Ambiguous Constructor Calls:**
```cpp
class Container {
public:
    Container(int size) { /* allocate size elements */ }
    Container(std::initializer_list<int> values) { /* initialize with values */ }
};

int main() {
    Container c1{10};    // Ambiguous! Which constructor?
    Container c2(10);    // Clear: calls Container(int)
    Container c3{{10}};  // Clear: calls initializer_list constructor
}
```

**✅ Clear Constructor Design:**
```cpp
class Container {
public:
    explicit Container(size_t size) { /* allocate size elements */ }
    Container(std::initializer_list<int> values) { /* initialize with values */ }
    
    // Factory methods for clarity
    static Container with_size(size_t size) { return Container(size); }
    static Container with_values(std::initializer_list<int> values) { 
        return Container(values); 
    }
};
```

### 5. nullptr Misconceptions

**❌ Mixing nullptr with other types:**
```cpp
void process(int* ptr) { /* ... */ }
void process(int value) { /* ... */ }

int main() {
    process(0);         // Calls process(int), not process(int*)
    process(NULL);      // Might be ambiguous
    process(nullptr);   // Clear: calls process(int*)
}
```

### 6. Static Assert Misuse

**❌ Runtime Checks in static_assert:**
```cpp
void bad_static_assert(int size) {
    static_assert(size > 0, "Size must be positive"); // ERROR: size not compile-time constant
}
```

**✅ Proper Static Assert Usage:**
```cpp
template<int Size>
void good_static_assert() {
    static_assert(Size > 0, "Size must be positive"); // OK: Size is compile-time constant
}

constexpr int calculate_size() { return 42; }
static_assert(calculate_size() > 0, "Calculated size must be positive"); // OK: constexpr
```

### 7. Range-Based For Loop Gotchas

**❌ Dangerous Patterns:**
```cpp
// Temporary object destruction
for (const auto& item : get_vector()) { // get_vector() returns by value
    // item refers to destroyed temporary!
}

// Modifying container during iteration
std::vector<int> vec = {1, 2, 3, 4, 5};
for (const auto& item : vec) {
    if (item == 3) {
        vec.push_back(6); // DANGER: Undefined behavior
    }
}
```

**✅ Safe Patterns:**
```cpp
// Store temporary in variable
auto temp_vec = get_vector();
for (const auto& item : temp_vec) {
    // Safe: temp_vec lives for the duration of the loop
}

// Collect modifications, apply later
std::vector<int> vec = {1, 2, 3, 4, 5};
std::vector<int> to_add;
for (const auto& item : vec) {
    if (item == 3) {
        to_add.push_back(6);
    }
}
vec.insert(vec.end(), to_add.begin(), to_add.end());
```

## Performance Considerations

### Compile-Time vs Runtime Impact

C++11 features are designed to have minimal or zero runtime overhead:

```cpp
#include <chrono>
#include <vector>
#include <algorithm>
#include <functional>

class PerformanceAnalyzer {
private:
    static constexpr size_t DATA_SIZE = 10000000;
    
public:
    // 1. Auto has ZERO runtime cost - purely compile-time
    void auto_performance_test() {
        std::vector<int> data(DATA_SIZE);
        std::iota(data.begin(), data.end(), 1);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Both approaches compile to identical assembly
        for (std::vector<int>::iterator it = data.begin(); it != data.end(); ++it) {
            *it *= 2;
        }
        
        auto mid = std::chrono::high_resolution_clock::now();
        
        for (auto it = data.begin(); it != data.end(); ++it) {
            *it *= 2;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        std::cout << "Explicit type: " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count() 
                  << "μs\n";
        std::cout << "Auto type: " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count() 
                  << "μs\n";
    }
    
    // 2. Lambda performance - can be better than function pointers
    void lambda_vs_function_performance() {
        std::vector<int> data(DATA_SIZE);
        std::iota(data.begin(), data.end(), 1);
        
        // Function pointer (prevents inlining)
        auto start1 = std::chrono::high_resolution_clock::now();
        std::transform(data.begin(), data.end(), data.begin(), 
                      [](int x) { return x * x; }); // Can be inlined
        auto end1 = std::chrono::high_resolution_clock::now();
        
        // Reset data
        std::iota(data.begin(), data.end(), 1);
        
        // Lambda with capture (slight overhead)
        int multiplier = 2;
        auto start2 = std::chrono::high_resolution_clock::now();
        std::transform(data.begin(), data.end(), data.begin(), 
                      [multiplier](int x) { return x * multiplier; });
        auto end2 = std::chrono::high_resolution_clock::now();
        
        std::cout << "Lambda (no capture): " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() 
                  << "μs\n";
        std::cout << "Lambda (with capture): " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() 
                  << "μs\n";
    }
    
    // 3. Range-based for vs traditional loops
    void range_based_for_performance() {
        std::vector<long long> data(DATA_SIZE);
        std::iota(data.begin(), data.end(), 1);
        
        // Traditional indexed loop
        auto start1 = std::chrono::high_resolution_clock::now();
        long long sum1 = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            sum1 += data[i];
        }
        auto end1 = std::chrono::high_resolution_clock::now();
        
        // Iterator loop
        auto start2 = std::chrono::high_resolution_clock::now();
        long long sum2 = 0;
        for (auto it = data.begin(); it != data.end(); ++it) {
            sum2 += *it;
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        
        // Range-based for loop
        auto start3 = std::chrono::high_resolution_clock::now();
        long long sum3 = 0;
        for (const auto& value : data) {
            sum3 += value;
        }
        auto end3 = std::chrono::high_resolution_clock::now();
        
        std::cout << "Indexed loop: sum=" << sum1 << ", time=" 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() 
                  << "μs\n";
        std::cout << "Iterator loop: sum=" << sum2 << ", time=" 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() 
                  << "μs\n";
        std::cout << "Range-based loop: sum=" << sum3 << ", time=" 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count() 
                  << "μs\n";
    }
    
    // 4. Uniform initialization performance
    void initialization_performance() {
        const size_t ITERATIONS = 1000000;
        
        // Traditional initialization
        auto start1 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ITERATIONS; ++i) {
            std::vector<int> vec(5, 42);  // 5 elements, all 42
        }
        auto end1 = std::chrono::high_resolution_clock::now();
        
        // Uniform initialization
        auto start2 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ITERATIONS; ++i) {
            std::vector<int> vec{42, 42, 42, 42, 42};  // Initializer list
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        
        std::cout << "Traditional initialization: " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() 
                  << "μs\n";
        std::cout << "Uniform initialization: " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() 
                  << "μs\n";
    }
};
```

### Memory Usage Optimization

```cpp
// Demonstrating memory-efficient C++11 patterns
class MemoryOptimizedExample {
public:
    // 1. enum class uses minimum required storage
    enum class SmallEnum : uint8_t { A, B, C };     // 1 byte
    enum class MediumEnum : uint16_t { X = 1000 };  // 2 bytes
    
    // 2. Lambda captures only what's needed
    auto create_efficient_lambda(int important_value) {
        int unused_large_array[10000] = {0};
        
        // Only captures important_value, not the large array
        return [important_value](int param) {
            return important_value + param;
        };
    }
    
    // 3. Move semantics with uniform initialization
    std::vector<std::string> create_optimized_vector() {
        return {"string1", "string2", "string3"}; // Move construction
    }
    
    // 4. Static assertions prevent memory waste
    template<typename T>
    class CompactContainer {
        static_assert(sizeof(T) <= 64, "Type too large for compact storage");
        static_assert(std::is_trivially_copyable<T>::value, 
                     "Type must be trivially copyable for performance");
        
        T data[1000];  // Fixed-size array for predictable memory usage
    };
};
```

### Compilation Time Impact

```cpp
// These features can affect compilation time:

// 1. Heavy template instantiation with auto
template<typename T>
auto heavy_template_function(T&& t) -> decltype(std::forward<T>(t).some_method()) {
    return std::forward<T>(t).some_method();
}

// 2. Complex lambda captures
auto create_complex_lambda() {
    std::map<std::string, std::vector<int>> complex_data;
    // ... populate data ...
    
    return [complex_data](const std::string& key) {
        auto it = complex_data.find(key);
        return it != complex_data.end() ? it->second : std::vector<int>{};
    };
}

// 3. static_assert with complex type traits
template<typename T>
class TypeValidator {
    static_assert(std::is_move_constructible<T>::value, "T must be move constructible");
    static_assert(std::is_move_assignable<T>::value, "T must be move assignable");
    static_assert(!std::is_reference<T>::value, "T cannot be a reference");
    static_assert(std::is_destructible<T>::value, "T must be destructible");
    // Multiple static_assert calls can slow compilation
};
```

### Optimization Guidelines

1. **Use `auto` judiciously**: Don't let it hide important type information
2. **Prefer capture by value for small objects**: Avoid reference capture pitfalls
3. **Use `constexpr` where possible**: Move computation to compile-time
4. **Profile before optimizing**: Measure actual performance impact
5. **Consider compilation time**: Heavy template usage can slow builds

## Best Practices

### 1. Auto Usage Guidelines

**✅ Use `auto` when:**
- The type is obvious from the initializer
- Working with complex template types
- Avoiding code duplication
- The exact type is not important for understanding

```cpp
// Good examples
auto numbers = std::vector<int>{1, 2, 3};           // Type is obvious
auto it = complicated_container.begin();            // Iterator type is complex
auto result = static_cast<double>(int_value);       // Cast result is clear
auto lambda = [](int x) { return x * 2; };         // Lambda type is complex
```

**❌ Avoid `auto` when:**
- The type conveys important information
- Initialization might be ambiguous
- You need specific numeric types

```cpp
// Questionable examples
auto timeout = 1000;                    // What unit? int? long?
auto threshold = 0.5;                   // float? double?
auto result = getValue();               // What does getValue() return?
```

### 2. Lambda Best Practices

**Capture Guidelines:**
```cpp
// ✅ Prefer capture by value for small objects
auto safe_lambda = [small_value](int param) {
    return small_value + param;  // Copy is safe and fast
};

// ✅ Use explicit capture for clarity
auto explicit_lambda = [x, &y, z](int param) {
    // Clear what's captured and how
    return x + y + z + param;
};

// ✅ Use mutable when you need to modify captured values
auto mutable_lambda = [counter = 0](int increment) mutable {
    counter += increment;
    return counter;
};

// ❌ Avoid capturing everything by reference
auto dangerous_lambda = [&]() {  // Dangerous - what if referenced objects are destroyed?
    // ... use variables that might not exist
};
```

**Performance Tips:**
```cpp
// ✅ Use lambdas for local, short-lived operations
std::sort(vec.begin(), vec.end(), [](const Item& a, const Item& b) {
    return a.priority > b.priority;
});

// ✅ Store lambdas in std::function when type erasure is needed
std::function<int(int)> operation = [multiplier](int x) {
    return x * multiplier;
};

// ✅ Use generic lambdas for flexibility (C++14, but good practice)
auto generic_comparator = [](const auto& a, const auto& b) {
    return a < b;
};
```

### 3. Enum Class Guidelines

**Design Principles:**
```cpp
// ✅ Use descriptive names and explicit underlying types
enum class HttpStatus : int {
    OK = 200,
    NOT_FOUND = 404,
    INTERNAL_ERROR = 500
};

// ✅ Group related enums in namespaces
namespace Graphics {
    enum class Color : uint8_t { RED, GREEN, BLUE, ALPHA };
    enum class BlendMode : uint8_t { NORMAL, MULTIPLY, SCREEN };
}

// ✅ Provide utility functions
std::string to_string(HttpStatus status) {
    switch (status) {
        case HttpStatus::OK: return "OK";
        case HttpStatus::NOT_FOUND: return "Not Found";
        case HttpStatus::INTERNAL_ERROR: return "Internal Error";
    }
    return "Unknown";
}

bool is_error(HttpStatus status) {
    return static_cast<int>(status) >= 400;
}
```

### 4. Initialization Best Practices

**Uniform Initialization Guidelines:**
```cpp
// ✅ Use uniform initialization for consistency
class MyClass {
    int value{42};                    // Member initialization
    std::vector<int> data{1, 2, 3};   // Container initialization
    
public:
    MyClass() : value{0}, data{} {}   // Constructor initialization
    
    void method() {
        auto temp = MyClass{};        // Object initialization
        std::vector<int> local{1, 2, 3, 4, 5}; // Local initialization
    }
};

// ✅ Be explicit about constructor calls when ambiguous
class Container {
public:
    explicit Container(size_t size);
    Container(std::initializer_list<int> values);
};

Container c1(10);        // Size constructor
Container c2{1, 2, 3};   // Initializer list constructor
Container c3{{10}};      // Initializer list with single element
```

### 5. nullptr Best Practices

```cpp
// ✅ Always use nullptr for null pointers
int* ptr = nullptr;
if (ptr == nullptr) { /* ... */ }

// ✅ Use nullptr in templates
template<typename T>
void process(T* ptr) {
    if (ptr != nullptr) {
        // Process
    }
}

// ✅ Prefer smart pointers with nullptr
std::unique_ptr<int> smart_ptr = nullptr;
std::shared_ptr<Object> shared_obj = nullptr;

// ✅ Use nullptr in function parameters
void function(int* optional_param = nullptr) {
    if (optional_param != nullptr) {
        // Use parameter
    }
}
```

### 6. Static Assertion Strategy

```cpp
// ✅ Use static_assert for compile-time validation
template<typename T, size_t N>
class FixedArray {
    static_assert(N > 0, "Array size must be positive");
    static_assert(N <= 10000, "Array size too large");
    static_assert(std::is_default_constructible<T>::value, 
                  "Element type must be default constructible");
    
    T data[N];
};

// ✅ Document assumptions with static_assert
class NetworkBuffer {
    static_assert(sizeof(int) == 4, "Code assumes 32-bit integers");
    static_assert(CHAR_BIT == 8, "Code assumes 8-bit bytes");
    
    // Implementation that depends on these assumptions
};
```

### 7. Range-Based For Loop Guidelines

```cpp
// ✅ Use const auto& for read-only access
for (const auto& item : container) {
    process(item);  // No copying, no modification
}

// ✅ Use auto& for modification
for (auto& item : container) {
    item.update();  // Modify in place
}

// ✅ Use auto (copy) for small types or when you need a copy
for (auto value : numeric_container) {
    value *= 2;  // Modify the copy, original unchanged
}

// ✅ Store temporary objects to avoid dangling references
auto temp_container = get_temporary_container();
for (const auto& item : temp_container) {
    // Safe: temp_container lives throughout the loop
}
```

### 8. Delegating Constructor Patterns

```cpp
class ConfigurableClass {
private:
    std::string name;
    int value;
    bool enabled;
    
    void validate() {
        if (value < 0) throw std::invalid_argument("Value cannot be negative");
    }
    
public:
    // Primary constructor with full validation
    ConfigurableClass(const std::string& n, int v, bool e) 
        : name(n), value(v), enabled(e) {
        validate();
    }
    
    // Delegating constructors for convenience
    ConfigurableClass(const std::string& n, int v) 
        : ConfigurableClass(n, v, true) {}  // Default enabled
    
    ConfigurableClass(const std::string& n) 
        : ConfigurableClass(n, 0, true) {}  // Default value and enabled
    
    ConfigurableClass() 
        : ConfigurableClass("Default", 0, true) {}  // All defaults
};
```

### 9. General Modern C++11 Principles

1. **Prefer RAII**: Use constructors and destructors for resource management
2. **Make interfaces hard to use incorrectly**: Use type system to prevent errors
3. **Be explicit**: Use `explicit` constructors, clear variable names
4. **Fail fast**: Use static_assert and constructor validation
5. **Optimize for readability**: Code is read more often than written
6. **Use the type system**: Let the compiler catch errors for you
7. **Be consistent**: Establish coding standards and follow them
8. **Profile don't guess**: Measure performance impact of features

## Summary

C++11 represents a watershed moment in C++ evolution, introducing features that fundamentally changed how we write modern C++ code. The core language features covered in this section form the foundation for all modern C++ development.

### Key Takeaways

| Feature | Primary Benefit | Use When |
|---------|----------------|----------|
| **`auto`** | Reduced verbosity, better maintainability | Type is obvious or complex |
| **Range-based for** | Simpler, safer iteration | Iterating over containers |
| **Lambda expressions** | Local functions, functional programming | Short operations, algorithm parameters |
| **`nullptr`** | Type safety, no ambiguity | Any null pointer scenario |
| **`enum class`** | Scoped enums, type safety | Better than traditional enums |
| **Static assertions** | Compile-time validation | Template constraints, assumptions |
| **Delegating constructors** | Code reuse, consistent initialization | Multiple constructor overloads |
| **Initializer lists** | Uniform syntax, container initialization | Object and container creation |
| **Uniform initialization** | Consistent syntax, prevents narrowing | Any initialization scenario |

### Migration Strategy

**From C++03 to C++11:**

1. **Start with `auto`**: Replace verbose type declarations
   ```cpp
   // Old
   std::map<std::string, std::vector<int>>::iterator it = map.begin();
   // New
   auto it = map.begin();
   ```

2. **Adopt range-based for**: Simplify iteration
   ```cpp
   // Old
   for (std::vector<int>::const_iterator it = vec.begin(); it != vec.end(); ++it) {
       process(*it);
   }
   // New
   for (const auto& item : vec) {
       process(item);
   }
   ```

3. **Replace NULL with nullptr**: Improve type safety
   ```cpp
   // Old
   int* ptr = NULL;
   // New
   int* ptr = nullptr;
   ```

4. **Use enum class**: Replace traditional enums
   ```cpp
   // Old
   enum Color { RED, GREEN, BLUE };
   // New
   enum class Color { RED, GREEN, BLUE };
   ```

5. **Leverage lambdas**: Replace function objects and simple functions
   ```cpp
   // Old
   struct Comparator {
       bool operator()(int a, int b) const { return a < b; }
   };
   std::sort(vec.begin(), vec.end(), Comparator());
   // New
   std::sort(vec.begin(), vec.end(), [](int a, int b) { return a < b; });
   ```

### Impact on Code Quality

**Before C++11:**
```cpp
// Verbose, error-prone, inconsistent
std::map<std::string, std::vector<int>> data;
for (std::map<std::string, std::vector<int>>::iterator it = data.begin(); 
     it != data.end(); ++it) {
    if (it->second.size() > 0) {
        for (std::vector<int>::iterator vec_it = it->second.begin();
             vec_it != it->second.end(); ++vec_it) {
            if (*vec_it > 0) {
                *vec_it *= 2;
            }
        }
    }
}
```

**With C++11:**
```cpp
// Clean, readable, safe
std::map<std::string, std::vector<int>> data;
for (auto& [key, values] : data) {  // C++17 structured binding shown for comparison
    for (auto& value : values) {
        if (value > 0) {
            value *= 2;
        }
    }
}
```

### Next Steps

Having mastered C++11 core language features, you're ready to explore:
1. **C++11 Standard Library features** (smart pointers, threading, etc.)
2. **C++14 enhancements** (generic lambdas, auto return types)
3. **C++17 features** (structured bindings, if constexpr)
4. **Modern C++ design patterns** using these features

### Resources for Continued Learning

**Books:**
- "Effective Modern C++" by Scott Meyers
- "C++ Primer" (5th Edition) by Stanley Lippman
- "Professional C++" by Gregoire, Solter, and Kleper

**Online Resources:**
- [cppreference.com](https://cppreference.com) - Comprehensive reference
- [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines) - Best practices
- [Compiler Explorer](https://godbolt.org) - See generated assembly code

**Practice Platforms:**
- [HackerRank C++](https://www.hackerrank.com/domains/cpp)
- [LeetCode](https://leetcode.com) - Algorithm practice with modern C++
- [Exercism C++](https://exercism.org/tracks/cpp) - Guided practice

The journey to mastering modern C++ begins with these fundamental features. Practice them extensively, understand their implications, and you'll be well-equipped to write efficient, maintainable, and expressive C++ code.
