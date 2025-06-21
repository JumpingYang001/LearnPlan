# C++14 Improvements

## Overview

C++14 was a minor release that refined and extended C++11 features. It introduced several quality-of-life improvements, making C++ more convenient and expressive while maintaining backward compatibility.

## Key Features

### 1. Generic Lambdas

C++14 allows lambda parameters to be declared with `auto`, enabling generic lambdas.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>

void demonstrate_generic_lambdas() {
    std::cout << "\n=== Generic Lambdas ===" << std::endl;
    
    // Generic lambda with auto parameters
    auto generic_print = [](const auto& value) {
        std::cout << "Value: " << value << std::endl;
    };
    
    // Works with different types
    generic_print(42);
    generic_print(3.14);
    generic_print(std::string("Hello"));
    generic_print('A');
    
    // Generic lambda for comparison
    auto generic_less = [](const auto& a, const auto& b) {
        return a < b;
    };
    
    std::cout << "5 < 10: " << generic_less(5, 10) << std::endl;
    std::cout << "3.14 < 2.71: " << generic_less(3.14, 2.71) << std::endl;
    std::cout << "\"apple\" < \"banana\": " << generic_less(std::string("apple"), std::string("banana")) << std::endl;
    
    // Using generic lambdas with algorithms
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3};
    std::vector<std::string> words = {"banana", "apple", "cherry", "date"};
    
    // Sort with generic lambda
    std::sort(numbers.begin(), numbers.end(), generic_less);
    std::sort(words.begin(), words.end(), generic_less);
    
    std::cout << "Sorted numbers: ";
    for (int n : numbers) std::cout << n << " ";
    std::cout << std::endl;
    
    std::cout << "Sorted words: ";
    for (const auto& word : words) std::cout << word << " ";
    std::cout << std::endl;
}

// Advanced generic lambda usage
void demonstrate_advanced_generic_lambdas() {
    std::cout << "\n=== Advanced Generic Lambdas ===" << std::endl;
    
    // Lambda with multiple auto parameters
    auto generic_operation = [](auto op, const auto& a, const auto& b) {
        return op(a, b);
    };
    
    auto add = [](const auto& x, const auto& y) { return x + y; };
    auto multiply = [](const auto& x, const auto& y) { return x * y; };
    
    std::cout << "5 + 3 = " << generic_operation(add, 5, 3) << std::endl;
    std::cout << "2.5 * 4.0 = " << generic_operation(multiply, 2.5, 4.0) << std::endl;
    std::cout << "\"Hello\" + \" World\" = " << generic_operation(add, std::string("Hello"), std::string(" World")) << std::endl;
    
    // Generic lambda with perfect forwarding
    auto perfect_forwarder = [](auto&& func, auto&&... args) {
        return func(std::forward<decltype(args)>(args)...);
    };
    
    auto print_sum = [](int a, int b) {
        std::cout << "Sum: " << (a + b) << std::endl;
        return a + b;
    };
    
    perfect_forwarder(print_sum, 10, 20);
    
    // Generic lambda with containers
    auto process_container = [](const auto& container, auto processor) {
        for (const auto& item : container) {
            processor(item);
        }
    };
    
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::map<std::string, int> map = {{"one", 1}, {"two", 2}, {"three", 3}};
    
    process_container(vec, [](int value) {
        std::cout << "Vector item: " << value << std::endl;
    });
    
    process_container(map, [](const auto& pair) {
        std::cout << "Map item: " << pair.first << " -> " << pair.second << std::endl;
    });
}
```

### 2. Return Type Deduction for Functions

C++14 allows functions to use `auto` as the return type with automatic deduction.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

// Simple return type deduction
auto add_numbers(int a, int b) {
    return a + b;  // Returns int
}

auto multiply_values(double x, double y) {
    return x * y;  // Returns double
}

// Complex return type deduction
auto create_vector(int size) {
    std::vector<int> result;
    result.reserve(size);
    for (int i = 0; i < size; ++i) {
        result.push_back(i * i);
    }
    return result;  // Returns std::vector<int>
}

// Conditional return type deduction
auto get_value(bool use_int) {
    if (use_int) {
        return 42;      // int
    } else {
        return 3.14;    // double - ERROR! Inconsistent return types
    }
}

// Fixed version with consistent return type
auto get_value_fixed(bool use_int) -> double {
    if (use_int) {
        return 42.0;    // double
    } else {
        return 3.14;    // double
    }
}

// Template function with auto return type
template<typename T, typename U>
auto generic_add(T t, U u) {
    return t + u;  // Return type deduced from T + U
}

void demonstrate_return_type_deduction() {
    std::cout << "\n=== Return Type Deduction ===" << std::endl;
    
    // Basic usage
    auto sum = add_numbers(5, 10);
    auto product = multiply_values(2.5, 4.0);
    
    std::cout << "Sum: " << sum << " (type size: " << sizeof(sum) << ")" << std::endl;
    std::cout << "Product: " << product << " (type size: " << sizeof(product) << ")" << std::endl;
    
    // Complex type deduction
    auto vec = create_vector(5);
    std::cout << "Vector size: " << vec.size() << std::endl;
    std::cout << "Vector contents: ";
    for (auto value : vec) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    // Template function with auto
    auto result1 = generic_add(5, 10);        // int + int = int
    auto result2 = generic_add(2.5, 3);       // double + int = double
    auto result3 = generic_add(std::string("Hello"), std::string(" World")); // string + string = string
    
    std::cout << "Generic add results:" << std::endl;
    std::cout << "  5 + 10 = " << result1 << std::endl;
    std::cout << "  2.5 + 3 = " << result2 << std::endl;
    std::cout << "  \"Hello\" + \" World\" = " << result3 << std::endl;
    
    // Fixed conditional return
    auto fixed_result = get_value_fixed(true);
    std::cout << "Fixed conditional result: " << fixed_result << std::endl;
}

// Recursive function with auto return type
auto fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Member function with auto return type
class Calculator {
public:
    auto calculate(double x, double y) {
        return x * y + 10.0;
    }
    
    template<typename T>
    auto process(T value) {
        return value * 2;
    }
    
    // Auto return type with trailing return type for clarity
    auto get_result() -> std::vector<double> {
        return {1.0, 2.0, 3.0};
    }
};

void demonstrate_advanced_return_deduction() {
    std::cout << "\n=== Advanced Return Type Deduction ===" << std::endl;
    
    // Recursive function
    auto fib_result = fibonacci(10);
    std::cout << "Fibonacci(10) = " << fib_result << std::endl;
    
    // Member functions
    Calculator calc;
    auto calc_result = calc.calculate(5.0, 3.0);
    auto processed_int = calc.process(42);
    auto processed_double = calc.process(3.14);
    auto result_vector = calc.get_result();
    
    std::cout << "Calculator result: " << calc_result << std::endl;
    std::cout << "Processed int: " << processed_int << std::endl;
    std::cout << "Processed double: " << processed_double << std::endl;
    std::cout << "Result vector size: " << result_vector.size() << std::endl;
}
```

### 3. Variable Templates

C++14 introduces templates for variables, not just functions and classes.

```cpp
#include <iostream>
#include <type_traits>
#include <cmath>

// Basic variable template
template<typename T>
constexpr T pi = T(3.14159265358979323846);

// Variable template with specialization
template<typename T>
constexpr T max_value = std::numeric_limits<T>::max();

// Specialized for floating point
template<>
constexpr double max_value<double> = 1e308;

// Variable template for type traits (C++14 style)
template<typename T>
constexpr bool is_integral_v = std::is_integral<T>::value;

template<typename T>
constexpr bool is_floating_point_v = std::is_floating_point<T>::value;

template<typename T>
constexpr bool is_arithmetic_v = std::is_arithmetic<T>::value;

// Variable template with non-type parameters
template<int N>
constexpr int factorial = N * factorial<N - 1>;

template<>
constexpr int factorial<0> = 1;

template<>
constexpr int factorial<1> = 1;

void demonstrate_variable_templates() {
    std::cout << "\n=== Variable Templates ===" << std::endl;
    
    // Using pi with different types
    std::cout << "pi<float>: " << pi<float> << std::endl;
    std::cout << "pi<double>: " << pi<double> << std::endl;
    std::cout << "pi<long double>: " << pi<long double> << std::endl;
    
    // Circle area calculation
    double radius = 5.0;
    auto area = pi<double> * radius * radius;
    std::cout << "Circle area (radius " << radius << "): " << area << std::endl;
    
    // Max values
    std::cout << "max_value<int>: " << max_value<int> << std::endl;
    std::cout << "max_value<float>: " << max_value<float> << std::endl;
    std::cout << "max_value<double>: " << max_value<double> << std::endl;
    
    // Type traits shortcuts
    std::cout << "int is integral: " << is_integral_v<int> << std::endl;
    std::cout << "double is floating point: " << is_floating_point_v<double> << std::endl;
    std::cout << "char is arithmetic: " << is_arithmetic_v<char> << std::endl;
    
    // Factorial computation
    std::cout << "5! = " << factorial<5> << std::endl;
    std::cout << "7! = " << factorial<7> << std::endl;
}

// Advanced variable template usage
template<typename T, T... values>
constexpr T sum_values = (values + ...);  // This is C++17 fold expression

// C++14 compatible version
template<typename T, T value>
constexpr T sum_values_single = value;

template<typename T, T first, T... rest>
constexpr T sum_values_recursive = first + sum_values_recursive<T, rest...>;

// Variable template for array sizes
template<typename T, size_t N>
constexpr size_t array_size(T(&)[N]) {
    return N;
}

template<typename T>
constexpr size_t type_size_v = sizeof(T);

// Mathematical constants as variable templates
template<typename T>
constexpr T e = T(2.71828182845904523536);

template<typename T>
constexpr T sqrt2 = T(1.41421356237309504880);

template<typename T>
constexpr T golden_ratio = T(1.61803398874989484820);

void demonstrate_advanced_variable_templates() {
    std::cout << "\n=== Advanced Variable Templates ===" << std::endl;
    
    // Mathematical constants
    std::cout << "e<double>: " << e<double> << std::endl;
    std::cout << "sqrt(2)<float>: " << sqrt2<float> << std::endl;
    std::cout << "Golden ratio<double>: " << golden_ratio<double> << std::endl;
    
    // Type sizes
    std::cout << "sizeof(int): " << type_size_v<int> << std::endl;
    std::cout << "sizeof(double): " << type_size_v<double> << std::endl;
    std::cout << "sizeof(std::string): " << type_size_v<std::string> << std::endl;
    
    // Sum values (C++14 recursive version)
    constexpr int sum1 = sum_values_recursive<int, 1, 2, 3, 4, 5>;
    std::cout << "Sum of 1,2,3,4,5: " << sum1 << std::endl;
    
    // Array size detection
    int arr[] = {1, 2, 3, 4, 5, 6};
    std::cout << "Array size: " << array_size(arr) << std::endl;
}
```

### 4. Extended constexpr

C++14 relaxes restrictions on `constexpr` functions, allowing more complex compile-time computations.

```cpp
#include <iostream>
#include <array>
#include <algorithm>

// C++11 constexpr was very limited - only single return statement
constexpr int old_factorial(int n) {
    return (n <= 1) ? 1 : n * old_factorial(n - 1);
}

// C++14 constexpr allows loops, multiple statements, etc.
constexpr int new_factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Constexpr with complex logic
constexpr int fibonacci_iterative(int n) {
    if (n <= 1) return n;
    
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

// Constexpr function that modifies local variables
constexpr bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

// Constexpr function that works with arrays
constexpr int sum_array(const int* arr, size_t size) {
    int sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum;
}

void demonstrate_extended_constexpr() {
    std::cout << "\n=== Extended constexpr ===" << std::endl;
    
    // Compile-time computation
    constexpr int fact5 = new_factorial(5);
    constexpr int fib10 = fibonacci_iterative(10);
    constexpr bool prime17 = is_prime(17);
    constexpr bool prime18 = is_prime(18);
    
    std::cout << "5! = " << fact5 << " (computed at compile time)" << std::endl;
    std::cout << "fib(10) = " << fib10 << " (computed at compile time)" << std::endl;
    std::cout << "17 is prime: " << prime17 << std::endl;
    std::cout << "18 is prime: " << prime18 << std::endl;
    
    // Array operations
    constexpr int arr[] = {1, 2, 3, 4, 5};
    constexpr int arr_sum = sum_array(arr, 5);
    std::cout << "Array sum: " << arr_sum << " (computed at compile time)" << std::endl;
    
    // Can also be computed at runtime
    int runtime_n = 6;
    int runtime_fact = new_factorial(runtime_n);
    std::cout << runtime_n << "! = " << runtime_fact << " (computed at runtime)" << std::endl;
}

// Constexpr class with extended capabilities
class ConstexprCalculator {
private:
    int value;
    
public:
    constexpr ConstexprCalculator(int v) : value(v) {}
    
    constexpr int get_value() const { return value; }
    
    constexpr void set_value(int v) { value = v; }  // C++14: can modify member variables
    
    constexpr int add(int x) {
        value += x;
        return value;
    }
    
    constexpr int power(int exp) {
        int result = 1;
        int base = value;
        for (int i = 0; i < exp; ++i) {
            result *= base;
        }
        return result;
    }
    
    constexpr bool is_even() const {
        return value % 2 == 0;
    }
};

// Constexpr function that creates and modifies objects
constexpr int complex_calculation() {
    ConstexprCalculator calc(5);
    calc.add(10);        // calc.value is now 15
    calc.set_value(3);   // calc.value is now 3
    return calc.power(4); // Returns 3^4 = 81
}

void demonstrate_constexpr_classes() {
    std::cout << "\n=== Constexpr Classes ===" << std::endl;
    
    constexpr ConstexprCalculator calc(10);
    constexpr bool is_even = calc.is_even();
    constexpr int power_result = calc.power(3);
    
    std::cout << "Calculator with value 10:" << std::endl;
    std::cout << "  Is even: " << is_even << std::endl;
    std::cout << "  10^3 = " << power_result << std::endl;
    
    constexpr int complex_result = complex_calculation();
    std::cout << "Complex calculation result: " << complex_result << std::endl;
    
    // Runtime usage is also possible
    ConstexprCalculator runtime_calc(7);
    runtime_calc.add(3);
    std::cout << "Runtime calculation: " << runtime_calc.get_value() << std::endl;
}
```

### 5. std::make_unique

C++14 adds the missing `std::make_unique` function template.

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>

class Resource {
private:
    std::string name;
    int id;
    std::vector<int> data;
    
public:
    Resource(const std::string& n, int i) : name(n), id(i) {
        std::cout << "Resource created: " << name << " (ID: " << id << ")" << std::endl;
    }
    
    Resource(const std::string& n, int i, std::initializer_list<int> init_data) 
        : name(n), id(i), data(init_data) {
        std::cout << "Resource created with data: " << name << " (ID: " << id 
                  << ", data size: " << data.size() << ")" << std::endl;
    }
    
    ~Resource() {
        std::cout << "Resource destroyed: " << name << " (ID: " << id << ")" << std::endl;
    }
    
    void display() const {
        std::cout << "Resource: " << name << " (ID: " << id << ")";
        if (!data.empty()) {
            std::cout << " Data: ";
            for (int val : data) {
                std::cout << val << " ";
            }
        }
        std::cout << std::endl;
    }
    
    const std::string& get_name() const { return name; }
    int get_id() const { return id; }
};

void demonstrate_make_unique() {
    std::cout << "\n=== std::make_unique ===" << std::endl;
    
    // Basic usage
    auto resource1 = std::make_unique<Resource>("Resource1", 1);
    resource1->display();
    
    // With multiple parameters
    auto resource2 = std::make_unique<Resource>("Resource2", 2, std::initializer_list<int>{10, 20, 30});
    resource2->display();
    
    // Array allocation
    auto int_array = std::make_unique<int[]>(10);
    for (int i = 0; i < 10; ++i) {
        int_array[i] = i * i;
    }
    
    std::cout << "Array contents: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << int_array[i] << " ";
    }
    std::cout << std::endl;
    
    // Container of unique_ptrs
    std::vector<std::unique_ptr<Resource>> resources;
    resources.push_back(std::make_unique<Resource>("VectorResource1", 10));
    resources.push_back(std::make_unique<Resource>("VectorResource2", 11));
    resources.push_back(std::make_unique<Resource>("VectorResource3", 12));
    
    std::cout << "Resources in vector:" << std::endl;
    for (const auto& res : resources) {
        res->display();
    }
}

// Exception safety comparison
void demonstrate_exception_safety() {
    std::cout << "\n=== Exception Safety with make_unique ===" << std::endl;
    
    // Function that might throw
    auto risky_function = [](std::unique_ptr<Resource> r1, std::unique_ptr<Resource> r2) {
        std::cout << "Processing resources: " << r1->get_name() << " and " << r2->get_name() << std::endl;
        return r1->get_id() + r2->get_id();
    };
    
    try {
        // Exception-safe way with make_unique
        auto result = risky_function(
            std::make_unique<Resource>("Safe1", 1),
            std::make_unique<Resource>("Safe2", 2)
        );
        std::cout << "Result: " << result << std::endl;
        
        // Less safe way (though still works in this simple case)
        // auto result2 = risky_function(
        //     std::unique_ptr<Resource>(new Resource("Less_Safe1", 3)),
        //     std::unique_ptr<Resource>(new Resource("Less_Safe2", 4))
        // );
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }
}

// Perfect forwarding with make_unique
template<typename T, typename... Args>
std::unique_ptr<T> create_resource(Args&&... args) {
    std::cout << "Creating resource with perfect forwarding..." << std::endl;
    return std::make_unique<T>(std::forward<Args>(args)...);
}

void demonstrate_perfect_forwarding() {
    std::cout << "\n=== Perfect Forwarding with make_unique ===" << std::endl;
    
    auto res1 = create_resource<Resource>("Forwarded1", 100);
    auto res2 = create_resource<Resource>("Forwarded2", 101, std::initializer_list<int>{1, 2, 3, 4});
    
    res1->display();
    res2->display();
    
    // Move semantics
    std::string name = "MoveResource";
    auto res3 = std::make_unique<Resource>(std::move(name), 200);
    res3->display();
    
    std::cout << "Original name after move: '" << name << "'" << std::endl;
}
```

### 6. Shared Locking (shared_mutex)

C++14 introduces shared mutexes for reader-writer scenarios.

```cpp
#include <iostream>
#include <shared_mutex>
#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <random>

class ThreadSafeDatabase {
private:
    mutable std::shared_mutex mutex_;
    std::vector<std::string> data_;
    
public:
    // Multiple readers can access simultaneously
    std::string read(size_t index) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        
        if (index >= data_.size()) {
            return "Index out of range";
        }
        
        // Simulate some read work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        return data_[index];
    }
    
    // Only one writer can access at a time
    void write(size_t index, const std::string& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        if (index >= data_.size()) {
            data_.resize(index + 1);
        }
        
        // Simulate some write work
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        data_[index] = value;
        
        std::cout << "Written '" << value << "' to index " << index << std::endl;
    }
    
    // Add new data (exclusive write)
    void append(const std::string& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        data_.push_back(value);
        
        std::cout << "Appended '" << value << "' at index " << (data_.size() - 1) << std::endl;
    }
    
    // Get size (shared read)
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return data_.size();
    }
    
    // Get all data (shared read)
    std::vector<std::string> get_all() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        
        // Simulate reading all data
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        
        return data_;
    }
};

void demonstrate_shared_locking() {
    std::cout << "\n=== Shared Locking ===" << std::endl;
    
    ThreadSafeDatabase database;
    
    // Initialize with some data
    database.append("Initial Data 0");
    database.append("Initial Data 1");
    database.append("Initial Data 2");
    
    std::vector<std::thread> threads;
    
    // Reader threads
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&database, i]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 2);
            
            for (int j = 0; j < 3; ++j) {
                size_t index = dis(gen);
                auto data = database.read(index);
                std::cout << "Reader " << i << " read from index " << index 
                          << ": '" << data << "'" << std::endl;
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    
    // Writer threads
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&database, i]() {
            for (int j = 0; j < 2; ++j) {
                std::string value = "Writer" + std::to_string(i) + "_" + std::to_string(j);
                database.append(value);
                
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    }
    
    // Mixed reader/writer thread
    threads.emplace_back([&database]() {
        for (int i = 0; i < 3; ++i) {
            // Read all data
            auto all_data = database.get_all();
            std::cout << "Mixed thread read " << all_data.size() << " items" << std::endl;
            
            // Write to existing index
            if (!all_data.empty()) {
                database.write(0, "Modified by mixed thread " + std::to_string(i));
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
    });
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    // Final state
    std::cout << "\nFinal database state:" << std::endl;
    auto final_data = database.get_all();
    for (size_t i = 0; i < final_data.size(); ++i) {
        std::cout << "  [" << i << "] = '" << final_data[i] << "'" << std::endl;
    }
}
```

### 7. Heterogeneous Lookup in Associative Containers

C++14 allows lookup in associative containers with types different from the key type.

```cpp
#include <iostream>
#include <set>
#include <map>
#include <string>
#include <string_view>  // C++17, but showing the concept

// Custom comparator that enables heterogeneous lookup
struct string_less {
    using is_transparent = void;  // This enables heterogeneous lookup
    
    bool operator()(const std::string& lhs, const std::string& rhs) const {
        return lhs < rhs;
    }
    
    bool operator()(const std::string& lhs, const char* rhs) const {
        return lhs < rhs;
    }
    
    bool operator()(const char* lhs, const std::string& rhs) const {
        return lhs < rhs;
    }
    
    bool operator()(const char* lhs, const char* rhs) const {
        return std::string(lhs) < std::string(rhs);
    }
};

void demonstrate_heterogeneous_lookup() {
    std::cout << "\n=== Heterogeneous Lookup ===" << std::endl;
    
    // Set with heterogeneous lookup
    std::set<std::string, string_less> string_set = {
        "apple", "banana", "cherry", "date", "elderberry"
    };
    
    // Traditional lookup (creates temporary string)
    std::string key1 = "banana";
    auto it1 = string_set.find(key1);
    if (it1 != string_set.end()) {
        std::cout << "Found with string key: " << *it1 << std::endl;
    }
    
    // Heterogeneous lookup (no temporary string created)
    const char* key2 = "cherry";
    auto it2 = string_set.find(key2);  // Direct lookup with const char*
    if (it2 != string_set.end()) {
        std::cout << "Found with const char* key: " << *it2 << std::endl;
    }
    
    // Count with heterogeneous lookup
    std::cout << "Count of 'date': " << string_set.count("date") << std::endl;
    
    // Lower bound with heterogeneous lookup
    auto lower = string_set.lower_bound("c");
    if (lower != string_set.end()) {
        std::cout << "First element >= 'c': " << *lower << std::endl;
    }
    
    // Map with heterogeneous lookup
    std::map<std::string, int, string_less> string_map = {
        {"apple", 1}, {"banana", 2}, {"cherry", 3}, {"date", 4}
    };
    
    // Heterogeneous lookup in map
    auto map_it = string_map.find("banana");  // No temporary string
    if (map_it != string_map.end()) {
        std::cout << "Map value for 'banana': " << map_it->second << std::endl;
    }
    
    // Performance comparison (conceptual)
    std::cout << "\nPerformance benefit: heterogeneous lookup avoids creating temporary strings" << std::endl;
}
```

## Summary

C++14 improvements focus on convenience and usability:

- **Generic lambdas**: Enable more flexible lambda expressions with `auto` parameters
- **Return type deduction**: Allow functions to use `auto` return type with automatic deduction  
- **Variable templates**: Templates for variables, not just functions and classes
- **Extended constexpr**: More complex compile-time computations allowed
- **std::make_unique**: Complete the smart pointer factory function family
- **Shared locking**: Reader-writer synchronization with shared_mutex
- **Heterogeneous lookup**: Efficient lookup in associative containers without temporary objects

Key benefits:
- Reduced boilerplate code
- Better compile-time programming capabilities
- Improved performance through heterogeneous lookup
- Enhanced concurrency support with shared locking
- More expressive and generic lambda expressions

C++14 maintains full backward compatibility with C++11 while making the language more convenient and expressive for everyday programming tasks.
