# C++17 Core Language Features

## Overview

C++17 introduced significant language improvements that further modernized C++ programming. These features enhance expressiveness, safety, and performance while maintaining backward compatibility.

## Key Features

### 1. Structured Bindings

Structured bindings allow unpacking of tuples, pairs, arrays, and custom types into individual variables.

```cpp
#include <iostream>
#include <tuple>
#include <map>
#include <string>
#include <array>
#include <utility>

// Function returning multiple values
std::tuple<std::string, int, double> get_person_info() {
    return {"Alice", 25, 65000.50};
}

// Function returning pair
std::pair<bool, std::string> validate_input(const std::string& input) {
    if (input.empty()) {
        return {false, "Input cannot be empty"};
    }
    if (input.length() > 100) {
        return {false, "Input too long"};
    }
    return {true, "Valid input"};
}

void demonstrate_structured_bindings_basic() {
    std::cout << "\n=== Basic Structured Bindings ===" << std::endl;
    
    // Tuple unpacking
    auto [name, age, salary] = get_person_info();
    std::cout << "Person: " << name << ", Age: " << age << ", Salary: $" << salary << std::endl;
    
    // Pair unpacking
    auto [valid, message] = validate_input("Hello World");
    std::cout << "Validation result: " << (valid ? "Success" : "Error") << " - " << message << std::endl;
    
    // Array unpacking
    int coordinates[3] = {10, 20, 30};
    auto [x, y, z] = coordinates;
    std::cout << "Coordinates: (" << x << ", " << y << ", " << z << ")" << std::endl;
    
    // std::array unpacking
    std::array<double, 2> point = {3.14, 2.71};
    auto [px, py] = point;
    std::cout << "Point: (" << px << ", " << py << ")" << std::endl;
    
    // Map iteration with structured bindings
    std::map<std::string, int> grades = {
        {"Math", 95}, {"Science", 87}, {"History", 92}
    };
    
    std::cout << "Student grades:" << std::endl;
    for (const auto& [subject, grade] : grades) {
        std::cout << "  " << subject << ": " << grade << std::endl;
    }
}

// Custom class supporting structured bindings
class Point3D {
private:
    double x_, y_, z_;
    
public:
    Point3D(double x, double y, double z) : x_(x), y_(y), z_(z) {}
    
    double x() const { return x_; }
    double y() const { return y_; }
    double z() const { return z_; }
    
    void display() const {
        std::cout << "(" << x_ << ", " << y_ << ", " << z_ << ")" << std::endl;
    }
};

// Specializations to enable structured bindings for Point3D
namespace std {
    template<>
    struct tuple_size<Point3D> : std::integral_constant<size_t, 3> {};
    
    template<>
    struct tuple_element<0, Point3D> { using type = double; };
    
    template<>
    struct tuple_element<1, Point3D> { using type = double; };
    
    template<>
    struct tuple_element<2, Point3D> { using type = double; };
}

// get function for Point3D
template<size_t I>
double get(const Point3D& p) {
    if constexpr (I == 0) return p.x();
    else if constexpr (I == 1) return p.y();
    else if constexpr (I == 2) return p.z();
}

void demonstrate_custom_structured_bindings() {
    std::cout << "\n=== Custom Structured Bindings ===" << std::endl;
    
    Point3D point(1.5, 2.5, 3.5);
    auto [x, y, z] = point;
    
    std::cout << "Original point: ";
    point.display();
    std::cout << "Unpacked coordinates: x=" << x << ", y=" << y << ", z=" << z << std::endl;
    
    // Using in algorithms
    std::vector<Point3D> points = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    std::cout << "Processing points:" << std::endl;
    for (const auto& [px, py, pz] : points) {
        double magnitude = std::sqrt(px*px + py*py + pz*pz);
        std::cout << "  Point(" << px << ", " << py << ", " << pz 
                  << ") magnitude: " << magnitude << std::endl;
    }
}

// References and const with structured bindings
void demonstrate_structured_binding_modifiers() {
    std::cout << "\n=== Structured Binding Modifiers ===" << std::endl;
    
    std::tuple<int, std::string, double> data = {42, "Hello", 3.14};
    
    // By value (copies)
    auto [val1, str1, dbl1] = data;
    val1 = 100;  // Doesn't affect original
    
    // By reference (can modify original)
    auto& [val2, str2, dbl2] = data;
    val2 = 200;  // Modifies original
    
    // Const reference (read-only)
    const auto& [val3, str3, dbl3] = data;
    // val3 = 300;  // Error: const
    
    std::cout << "Original tuple after modifications: " 
              << std::get<0>(data) << ", " << std::get<1>(data) << ", " << std::get<2>(data) << std::endl;
    
    // With auto*
    std::array<int, 3> arr = {1, 2, 3};
    auto& [a, b, c] = arr;
    a = 10;
    b = 20;
    c = 30;
    
    std::cout << "Modified array: ";
    for (int val : arr) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
```

### 2. if constexpr

Compile-time conditional compilation for templates.

```cpp
#include <iostream>
#include <type_traits>
#include <string>
#include <vector>

// Template function with compile-time branching
template<typename T>
void process_type(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Processing integer: " << value << " (doubled: " << value * 2 << ")" << std::endl;
    }
    else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Processing floating point: " << value << " (squared: " << value * value << ")" << std::endl;
    }
    else if constexpr (std::is_same_v<T, std::string>) {
        std::cout << "Processing string: \"" << value << "\" (length: " << value.length() << ")" << std::endl;
    }
    else {
        std::cout << "Processing unknown type" << std::endl;
    }
}

// Generic container processor
template<typename Container>
void process_container(const Container& container) {
    std::cout << "Container processing:" << std::endl;
    
    if constexpr (std::is_same_v<Container, std::string>) {
        std::cout << "  String with " << container.length() << " characters" << std::endl;
        for (char c : container) {
            std::cout << "    '" << c << "'" << std::endl;
        }
    }
    else {
        std::cout << "  Container with " << container.size() << " elements" << std::endl;
        for (const auto& element : container) {
            std::cout << "    " << element << std::endl;
        }
    }
}

void demonstrate_if_constexpr_basic() {
    std::cout << "\n=== Basic if constexpr ===" << std::endl;
    
    process_type(42);
    process_type(3.14);
    process_type(std::string("Hello"));
    process_type('A');
    
    std::cout << std::endl;
    
    process_container(std::vector<int>{1, 2, 3, 4});
    process_container(std::string("Hello"));
}

// Advanced usage: recursive template with if constexpr
template<typename T, typename... Args>
void print_all(T&& first, Args&&... args) {
    std::cout << first;
    
    if constexpr (sizeof...(args) > 0) {
        std::cout << ", ";
        print_all(args...);  // Recursive call only if there are more arguments
    }
}

// Template with different behavior based on type traits
template<typename Iterator>
auto advance_iterator(Iterator it, int n) {
    if constexpr (std::is_same_v<typename std::iterator_traits<Iterator>::iterator_category,
                                std::random_access_iterator_tag>) {
        // O(1) for random access iterators
        return it + n;
    }
    else {
        // O(n) for other iterator types
        for (int i = 0; i < n; ++i) {
            ++it;
        }
        return it;
    }
}

// Type-dependent member access
template<typename T>
void access_members(T obj) {
    if constexpr (std::is_class_v<T>) {
        // Only compile this branch if T is a class
        std::cout << "Object size: " << sizeof(obj) << " bytes" << std::endl;
        
        // Additional class-specific operations could go here
        if constexpr (requires { obj.size(); }) {  // C++20 concepts syntax
            // This would be C++20, showing the progression
            // std::cout << "Container size: " << obj.size() << std::endl;
        }
    }
    else {
        std::cout << "Non-class type, value: " << obj << std::endl;
    }
}

void demonstrate_advanced_if_constexpr() {
    std::cout << "\n=== Advanced if constexpr ===" << std::endl;
    
    // Variadic template with conditional recursion
    std::cout << "Printing arguments: ";
    print_all(1, 2.5, "hello", 'c', true);
    std::cout << std::endl;
    
    // Iterator advancement
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto it1 = advance_iterator(vec.begin(), 3);
    std::cout << "Advanced vector iterator points to: " << *it1 << std::endl;
    
    std::list<int> lst = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto it2 = advance_iterator(lst.begin(), 3);
    std::cout << "Advanced list iterator points to: " << *it2 << std::endl;
    
    // Type-dependent member access
    access_members(42);
    access_members(std::vector<int>{1, 2, 3});
}

// Generic serialize function
template<typename T>
std::string serialize(const T& value) {
    if constexpr (std::is_arithmetic_v<T>) {
        return std::to_string(value);
    }
    else if constexpr (std::is_same_v<T, std::string>) {
        return "\"" + value + "\"";
    }
    else if constexpr (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    }
    else {
        return "unsupported_type";
    }
}

// SFINAE vs if constexpr comparison
// SFINAE version (C++11/14 style)
template<typename T>
typename std::enable_if_t<std::is_integral_v<T>, T>
increment_sfinae(T value) {
    return value + 1;
}

template<typename T>
typename std::enable_if_t<std::is_floating_point_v<T>, T>
increment_sfinae(T value) {
    return value + 0.1;
}

// if constexpr version (C++17 style)
template<typename T>
T increment_constexpr(T value) {
    if constexpr (std::is_integral_v<T>) {
        return value + 1;
    }
    else if constexpr (std::is_floating_point_v<T>) {
        return value + 0.1;
    }
    else {
        static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");
    }
}

void demonstrate_constexpr_vs_sfinae() {
    std::cout << "\n=== if constexpr vs SFINAE ===" << std::endl;
    
    // Serialization
    std::cout << "Serialization examples:" << std::endl;
    std::cout << "  int: " << serialize(42) << std::endl;
    std::cout << "  double: " << serialize(3.14) << std::endl;
    std::cout << "  string: " << serialize(std::string("hello")) << std::endl;
    std::cout << "  bool: " << serialize(true) << std::endl;
    
    // SFINAE vs if constexpr
    std::cout << "\nIncrement comparison:" << std::endl;
    std::cout << "  SFINAE int: " << increment_sfinae(10) << std::endl;
    std::cout << "  SFINAE double: " << increment_sfinae(3.14) << std::endl;
    std::cout << "  constexpr int: " << increment_constexpr(10) << std::endl;
    std::cout << "  constexpr double: " << increment_constexpr(3.14) << std::endl;
}
```

### 3. Inline Variables

C++17 allows variables to be declared inline, enabling header-only libraries with global variables.

```cpp
#include <iostream>
#include <string>
#include <atomic>

// Header file simulation - these would typically be in a .h file

// Inline variable - can be defined in header without ODR violations
inline int global_counter = 0;

// Inline constant
inline const std::string application_name = "MyApp";

// Inline static data member (alternative to defining in .cpp file)
class Configuration {
public:
    inline static std::string default_path = "/etc/myapp/";
    inline static int max_connections = 100;
    inline static bool debug_mode = false;
    
    // Inline static with complex initialization
    inline static std::atomic<int> instance_count{0};
    
    Configuration() {
        instance_count.fetch_add(1);
    }
    
    ~Configuration() {
        instance_count.fetch_sub(1);
    }
    
    static void print_config() {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Default path: " << default_path << std::endl;
        std::cout << "  Max connections: " << max_connections << std::endl;
        std::cout << "  Debug mode: " << std::boolalpha << debug_mode << std::endl;
        std::cout << "  Active instances: " << instance_count.load() << std::endl;
    }
};

// Template inline variables
template<typename T>
inline constexpr T pi = T(3.14159265358979323846);

template<typename T>
inline T default_value = T{};

// Specialized inline variables
template<>
inline std::string default_value<std::string> = "default";

template<>
inline int default_value<int> = -1;

void demonstrate_inline_variables() {
    std::cout << "\n=== Inline Variables ===" << std::endl;
    
    // Global inline variables
    std::cout << "Application name: " << application_name << std::endl;
    std::cout << "Global counter: " << global_counter << std::endl;
    
    global_counter = 42;
    std::cout << "Modified global counter: " << global_counter << std::endl;
    
    // Inline static members
    Configuration::print_config();
    
    {
        Configuration config1;
        Configuration config2;
        Configuration::print_config();
    }
    
    Configuration::print_config();
    
    // Template inline variables
    std::cout << "\nTemplate inline variables:" << std::endl;
    std::cout << "pi<float>: " << pi<float> << std::endl;
    std::cout << "pi<double>: " << pi<double> << std::endl;
    
    std::cout << "default_value<int>: " << default_value<int> << std::endl;
    std::cout << "default_value<double>: " << default_value<double> << std::endl;
    std::cout << "default_value<string>: \"" << default_value<std::string> << "\"" << std::endl;
}

// Practical example: Header-only logging library
namespace logging {
    // Inline variables for logging configuration
    inline bool enabled = true;
    inline std::string log_level = "INFO";
    inline std::string log_format = "[{level}] {message}";
    
    // Inline function using inline variables
    inline void log(const std::string& level, const std::string& message) {
        if (!enabled) return;
        
        std::string formatted = log_format;
        size_t pos = formatted.find("{level}");
        if (pos != std::string::npos) {
            formatted.replace(pos, 7, level);
        }
        pos = formatted.find("{message}");
        if (pos != std::string::npos) {
            formatted.replace(pos, 9, message);
        }
        
        std::cout << formatted << std::endl;
    }
    
    // Convenience functions
    inline void info(const std::string& message) { log("INFO", message); }
    inline void warning(const std::string& message) { log("WARNING", message); }
    inline void error(const std::string& message) { log("ERROR", message); }
}

void demonstrate_header_only_library() {
    std::cout << "\n=== Header-Only Library with Inline Variables ===" << std::endl;
    
    logging::info("Application started");
    logging::warning("This is a warning message");
    logging::error("This is an error message");
    
    // Change configuration
    logging::log_format = "{level}: {message}";
    logging::info("Changed log format");
    
    // Disable logging
    logging::enabled = false;
    logging::info("This message won't appear");
    
    // Re-enable logging
    logging::enabled = true;
    logging::info("Logging re-enabled");
}
```

### 4. Fold Expressions

C++17 introduces fold expressions for variadic templates, simplifying common patterns.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

// Basic fold expressions
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);  // Unary right fold: (arg1 + (arg2 + (arg3 + ...)))
}

template<typename... Args>
auto sum_left(Args... args) {
    return (... + args);  // Unary left fold: (((... + arg1) + arg2) + arg3)
}

template<typename... Args>
auto multiply(Args... args) {
    return (args * ...);
}

template<typename... Args>
bool all_true(Args... args) {
    return (args && ...);  // Logical AND fold
}

template<typename... Args>
bool any_true(Args... args) {
    return (args || ...);  // Logical OR fold
}

void demonstrate_basic_fold_expressions() {
    std::cout << "\n=== Basic Fold Expressions ===" << std::endl;
    
    // Arithmetic operations
    auto sum_result = sum(1, 2, 3, 4, 5);
    auto multiply_result = multiply(2, 3, 4);
    
    std::cout << "Sum of 1,2,3,4,5: " << sum_result << std::endl;
    std::cout << "Product of 2,3,4: " << multiply_result << std::endl;
    
    // String concatenation
    auto concat_result = sum(std::string("Hello"), std::string(" "), std::string("World"));
    std::cout << "String concatenation: " << concat_result << std::endl;
    
    // Logical operations
    std::cout << "All true (true, true, true): " << all_true(true, true, true) << std::endl;
    std::cout << "All true (true, false, true): " << all_true(true, false, true) << std::endl;
    std::cout << "Any true (false, false, true): " << any_true(false, false, true) << std::endl;
    std::cout << "Any true (false, false, false): " << any_true(false, false, false) << std::endl;
}

// Binary fold expressions with initial value
template<typename... Args>
auto sum_with_init(Args... args) {
    return (0 + ... + args);  // Binary left fold with initial value
}

template<typename... Args>
auto multiply_with_init(Args... args) {
    return (1 * ... * args);  // Binary left fold with initial value
}

// More complex fold expressions
template<typename... Args>
void print_all(Args... args) {
    ((std::cout << args << " "), ...);  // Comma operator fold
    std::cout << std::endl;
}

template<typename... Args>
void print_with_separator(Args... args) {
    std::size_t index = 0;
    ((std::cout << args << (++index != sizeof...(args) ? ", " : "")), ...);
    std::cout << std::endl;
}

// Fold with function calls
template<typename... Funcs>
void call_all(Funcs... funcs) {
    (funcs(), ...);  // Call all functions
}

// Container operations with fold
template<typename Container, typename... Args>
void push_back_all(Container& container, Args... args) {
    (container.push_back(args), ...);
}

template<typename... Containers>
auto total_size(const Containers&... containers) {
    return (containers.size() + ...);
}

void demonstrate_advanced_fold_expressions() {
    std::cout << "\n=== Advanced Fold Expressions ===" << std::endl;
    
    // Binary fold with initial value
    std::cout << "Sum with init (empty): " << sum_with_init() << std::endl;
    std::cout << "Sum with init (1,2,3): " << sum_with_init(1, 2, 3) << std::endl;
    std::cout << "Multiply with init (empty): " << multiply_with_init() << std::endl;
    std::cout << "Multiply with init (2,3,4): " << multiply_with_init(2, 3, 4) << std::endl;
    
    // Printing with fold expressions
    std::cout << "Print all: ";
    print_all(1, 2.5, "hello", 'c');
    
    std::cout << "Print with separator: ";
    print_with_separator(1, 2.5, "hello", 'c');
    
    // Function calls
    auto func1 = []() { std::cout << "Function 1 called "; };
    auto func2 = []() { std::cout << "Function 2 called "; };
    auto func3 = []() { std::cout << "Function 3 called "; };
    
    std::cout << "Calling all functions: ";
    call_all(func1, func2, func3);
    std::cout << std::endl;
    
    // Container operations
    std::vector<int> vec;
    push_back_all(vec, 1, 2, 3, 4, 5);
    std::cout << "Vector after push_back_all: ";
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Multiple container sizes
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<std::string> vec2 = {"a", "b"};
    std::vector<double> vec3 = {1.1, 2.2, 3.3, 4.4};
    
    std::cout << "Total size of all containers: " << total_size(vec1, vec2, vec3) << std::endl;
}

// Practical example: Variadic min/max
template<typename T, typename... Args>
constexpr T min_value(T first, Args... args) {
    if constexpr (sizeof...(args) == 0) {
        return first;
    } else {
        return std::min(first, min_value(args...));
    }
}

// Using fold expression for min (C++17)
template<typename T, typename... Args>
constexpr T min_fold(T first, Args... args) {
    return (first < ... < args) ? first : min_fold(args...);  // This is conceptual
}

// Better version using fold
template<typename... Args>
constexpr auto min_all(Args... args) {
    return std::min({args...});  // Uses initializer_list
}

// Advanced: Type checking with fold
template<typename T, typename... Args>
constexpr bool all_same_type_v = (std::is_same_v<T, Args> && ...);

template<typename... Args>
constexpr bool all_arithmetic_v = (std::is_arithmetic_v<Args> && ...);

// Conditional execution with fold
template<typename... Predicates>
bool check_all_conditions(Predicates... preds) {
    return (preds() && ...);
}

void demonstrate_practical_fold_examples() {
    std::cout << "\n=== Practical Fold Examples ===" << std::endl;
    
    // Min/max operations
    std::cout << "Min of (5, 2, 8, 1, 9): " << min_all(5, 2, 8, 1, 9) << std::endl;
    
    // Type checking
    std::cout << "All same type (int, int, int): " << all_same_type_v<int, int, int> << std::endl;
    std::cout << "All same type (int, double, int): " << all_same_type_v<int, double, int> << std::endl;
    std::cout << "All arithmetic (int, double, float): " << all_arithmetic_v<int, double, float> << std::endl;
    std::cout << "All arithmetic (int, string, float): " << all_arithmetic_v<int, std::string, float> << std::endl;
    
    // Conditional execution
    auto condition1 = []() { std::cout << "Checking condition 1... "; return true; };
    auto condition2 = []() { std::cout << "Checking condition 2... "; return true; };
    auto condition3 = []() { std::cout << "Checking condition 3... "; return false; };
    
    std::cout << "All conditions passed: " << check_all_conditions(condition1, condition2) << std::endl;
    std::cout << "All conditions passed: " << check_all_conditions(condition1, condition2, condition3) << std::endl;
}
```

### 5. Class Template Argument Deduction (CTAD)

C++17 allows automatic deduction of template arguments for class templates.

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <utility>

// Custom class template with deduction guides
template<typename T>
class SimpleContainer {
private:
    std::vector<T> data;
    
public:
    SimpleContainer() = default;
    
    SimpleContainer(std::initializer_list<T> init) : data(init) {}
    
    template<typename Iterator>
    SimpleContainer(Iterator first, Iterator last) : data(first, last) {}
    
    void push_back(const T& value) {
        data.push_back(value);
    }
    
    size_t size() const { return data.size(); }
    
    void print() const {
        std::cout << "Container contents: ";
        for (const auto& item : data) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
    
    typename std::vector<T>::iterator begin() { return data.begin(); }
    typename std::vector<T>::iterator end() { return data.end(); }
    typename std::vector<T>::const_iterator begin() const { return data.begin(); }
    typename std::vector<T>::const_iterator end() const { return data.end(); }
};

// Deduction guides for SimpleContainer
SimpleContainer() -> SimpleContainer<int>;  // Default to int

template<typename T>
SimpleContainer(std::initializer_list<T>) -> SimpleContainer<T>;

template<typename Iterator>
SimpleContainer(Iterator, Iterator) -> SimpleContainer<typename std::iterator_traits<Iterator>::value_type>;

void demonstrate_basic_ctad() {
    std::cout << "\n=== Basic Class Template Argument Deduction ===" << std::endl;
    
    // Standard library containers with CTAD
    std::vector vec = {1, 2, 3, 4, 5};  // Deduced as std::vector<int>
    std::map map = {std::pair{1, "one"}, std::pair{2, "two"}};  // std::map<int, const char*>
    
    std::cout << "Vector size: " << vec.size() << std::endl;
    std::cout << "Map size: " << map.size() << std::endl;
    
    // Pair deduction
    std::pair p1{42, 3.14};  // std::pair<int, double>
    auto p2 = std::make_pair(42, 3.14);  // Equivalent
    
    std::cout << "Pair: (" << p1.first << ", " << p1.second << ")" << std::endl;
    
    // Custom container with CTAD
    SimpleContainer container1{1, 2, 3, 4};  // Deduced as SimpleContainer<int>
    container1.print();
    
    SimpleContainer container2{"hello", "world"};  // Deduced as SimpleContainer<const char*>
    container2.print();
    
    // Iterator-based construction
    std::vector<double> source = {1.1, 2.2, 3.3};
    SimpleContainer container3(source.begin(), source.end());  // SimpleContainer<double>
    container3.print();
}

// More complex deduction guide examples
template<typename T, typename Allocator = std::allocator<T>>
class CustomVector {
private:
    std::vector<T, Allocator> data;
    
public:
    CustomVector() = default;
    
    CustomVector(size_t size, const T& value) : data(size, value) {}
    
    template<typename Iterator>
    CustomVector(Iterator first, Iterator last) : data(first, last) {}
    
    void push_back(const T& value) { data.push_back(value); }
    size_t size() const { return data.size(); }
    
    void print() const {
        std::cout << "CustomVector: ";
        for (const auto& item : data) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
};

// Deduction guides for CustomVector
template<typename T>
CustomVector(size_t, T) -> CustomVector<T>;

template<typename Iterator>
CustomVector(Iterator, Iterator) -> CustomVector<typename std::iterator_traits<Iterator>::value_type>;

// Function template with CTAD helper
template<typename T>
auto make_unique_vector(std::initializer_list<T> init) {
    return std::make_unique<std::vector<T>>(init);
}

void demonstrate_advanced_ctad() {
    std::cout << "\n=== Advanced CTAD Examples ===" << std::endl;
    
    // Custom vector with deduction
    CustomVector vec1(5, 42);  // CustomVector<int>
    vec1.print();
    
    std::string arr[] = {"apple", "banana", "cherry"};
    CustomVector vec2(std::begin(arr), std::end(arr));  // CustomVector<std::string>
    vec2.print();
    
    // Smart pointers with CTAD
    auto ptr1 = std::make_unique<std::vector<int>>(std::initializer_list<int>{1, 2, 3});
    auto ptr2 = make_unique_vector({1.1, 2.2, 3.3});
    
    std::cout << "Smart pointer vector sizes: " << ptr1->size() << ", " << ptr2->size() << std::endl;
}

// CTAD with function objects and lambdas
template<typename Func>
class FunctionWrapper {
private:
    Func func;
    
public:
    FunctionWrapper(Func f) : func(f) {}
    
    template<typename... Args>
    auto operator()(Args&&... args) -> decltype(func(std::forward<Args>(args)...)) {
        std::cout << "Calling wrapped function..." << std::endl;
        return func(std::forward<Args>(args)...);
    }
};

// Deduction guide for function wrapper
template<typename Func>
FunctionWrapper(Func) -> FunctionWrapper<Func>;

// Factory function using CTAD
template<typename... Args>
auto make_tuple_auto(Args... args) {
    return std::tuple{args...};  // CTAD for std::tuple
}

void demonstrate_ctad_with_functions() {
    std::cout << "\n=== CTAD with Functions ===" << std::endl;
    
    // Function wrapper with lambda
    auto lambda = [](int x, int y) { return x + y; };
    FunctionWrapper wrapper{lambda};  // Deduced type
    
    auto result = wrapper(5, 10);
    std::cout << "Wrapped function result: " << result << std::endl;
    
    // Tuple creation with CTAD
    auto tuple1 = std::tuple{1, 2.5, "hello"};  // std::tuple<int, double, const char*>
    auto tuple2 = make_tuple_auto(42, 3.14, std::string("world"));
    
    std::cout << "Tuple sizes: " << std::tuple_size_v<decltype(tuple1)> 
              << ", " << std::tuple_size_v<decltype(tuple2)> << std::endl;
    
    // Array deduction
    int arr[] = {1, 2, 3, 4, 5};
    std::array array{1, 2, 3, 4, 5};  // std::array<int, 5>
    
    std::cout << "Array size: " << array.size() << std::endl;
}

// Practical example: Configuration system with CTAD
template<typename T>
class ConfigValue {
private:
    T value;
    std::string description;
    
public:
    ConfigValue(T val, std::string desc) : value(val), description(desc) {}
    
    const T& get() const { return value; }
    void set(const T& val) { value = val; }
    const std::string& get_description() const { return description; }
    
    void print() const {
        std::cout << description << ": " << value << std::endl;
    }
};

// Deduction guide
template<typename T>
ConfigValue(T, std::string) -> ConfigValue<T>;

void demonstrate_practical_ctad() {
    std::cout << "\n=== Practical CTAD Example ===" << std::endl;
    
    // Configuration values with automatic type deduction
    ConfigValue port{8080, "Server port"};           // ConfigValue<int>
    ConfigValue timeout{30.5, "Connection timeout"}; // ConfigValue<double>
    ConfigValue name{"MyServer", "Server name"};     // ConfigValue<const char*>
    ConfigValue debug{true, "Debug mode"};           // ConfigValue<bool>
    
    port.print();
    timeout.print();
    name.print();
    debug.print();
    
    // Vector of mixed config values (would need std::variant in practice)
    std::vector configs = {
        std::make_pair("port", "8080"),
        std::make_pair("timeout", "30.5"),
        std::make_pair("debug", "true")
    };
    
    std::cout << "Configuration items: " << configs.size() << std::endl;
}
```

### 6. Guaranteed Copy Elision

C++17 guarantees copy elision in certain scenarios, eliminating temporary objects.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>

class ExpensiveObject {
private:
    std::string name;
    std::vector<int> data;
    
public:
    // Constructor
    ExpensiveObject(const std::string& n, size_t size) : name(n), data(size, 42) {
        std::cout << "ExpensiveObject constructor: " << name << std::endl;
    }
    
    // Copy constructor
    ExpensiveObject(const ExpensiveObject& other) : name(other.name + "_copy"), data(other.data) {
        std::cout << "ExpensiveObject copy constructor: " << name << std::endl;
    }
    
    // Move constructor
    ExpensiveObject(ExpensiveObject&& other) noexcept 
        : name(std::move(other.name)), data(std::move(other.data)) {
        std::cout << "ExpensiveObject move constructor: " << name << std::endl;
    }
    
    // Copy assignment
    ExpensiveObject& operator=(const ExpensiveObject& other) {
        std::cout << "ExpensiveObject copy assignment: " << name << " = " << other.name << std::endl;
        if (this != &other) {
            name = other.name + "_assigned";
            data = other.data;
        }
        return *this;
    }
    
    // Move assignment
    ExpensiveObject& operator=(ExpensiveObject&& other) noexcept {
        std::cout << "ExpensiveObject move assignment: " << name << " = " << other.name << std::endl;
        if (this != &other) {
            name = std::move(other.name);
            data = std::move(other.data);
        }
        return *this;
    }
    
    // Destructor
    ~ExpensiveObject() {
        std::cout << "ExpensiveObject destructor: " << name << std::endl;
    }
    
    const std::string& get_name() const { return name; }
    size_t get_data_size() const { return data.size(); }
    
    void display() const {
        std::cout << "Object: " << name << " (data size: " << data.size() << ")" << std::endl;
    }
};

// Factory functions demonstrating guaranteed copy elision
ExpensiveObject create_object(const std::string& name) {
    return ExpensiveObject(name, 1000);  // Guaranteed copy elision
}

ExpensiveObject create_conditional(bool condition) {
    if (condition) {
        return ExpensiveObject("Conditional_True", 500);   // Guaranteed copy elision
    } else {
        return ExpensiveObject("Conditional_False", 750);  // Guaranteed copy elision
    }
}

// Function returning different objects (no guaranteed elision)
ExpensiveObject create_variable(bool condition) {
    ExpensiveObject obj1("Variable_True", 300);
    ExpensiveObject obj2("Variable_False", 400);
    
    return condition ? obj1 : obj2;  // Move constructor will be called
}

void demonstrate_guaranteed_copy_elision() {
    std::cout << "\n=== Guaranteed Copy Elision ===" << std::endl;
    
    // Direct initialization - guaranteed copy elision
    std::cout << "\n1. Direct initialization:" << std::endl;
    ExpensiveObject obj1 = ExpensiveObject("Direct", 100);
    
    // Factory function - guaranteed copy elision
    std::cout << "\n2. Factory function:" << std::endl;
    ExpensiveObject obj2 = create_object("Factory");
    
    // Conditional return - guaranteed copy elision for each path
    std::cout << "\n3. Conditional return (true):" << std::endl;
    ExpensiveObject obj3 = create_conditional(true);
    
    std::cout << "\n4. Conditional return (false):" << std::endl;
    ExpensiveObject obj4 = create_conditional(false);
    
    // Variable return - move semantics (not guaranteed elision)
    std::cout << "\n5. Variable return (move semantics):" << std::endl;
    ExpensiveObject obj5 = create_variable(true);
    
    std::cout << "\n6. Displaying all objects:" << std::endl;
    obj1.display();
    obj2.display();
    obj3.display();
    obj4.display();
    obj5.display();
}

// Practical example: Builder pattern with guaranteed copy elision
class ConfigBuilder {
private:
    std::string name_;
    int port_ = 8080;
    bool debug_ = false;
    std::vector<std::string> modules_;
    
public:
    ConfigBuilder(const std::string& name) : name_(name) {
        std::cout << "ConfigBuilder created for: " << name_ << std::endl;
    }
    
    ConfigBuilder& port(int p) {
        port_ = p;
        return *this;
    }
    
    ConfigBuilder& debug(bool d) {
        debug_ = d;
        return *this;
    }
    
    ConfigBuilder& add_module(const std::string& module) {
        modules_.push_back(module);
        return *this;
    }
    
    // Build method that returns by value
    class Config {
    private:
        std::string name;
        int port;
        bool debug;
        std::vector<std::string> modules;
        
    public:
        Config(const std::string& n, int p, bool d, std::vector<std::string> m)
            : name(n), port(p), debug(d), modules(std::move(m)) {
            std::cout << "Config created: " << name << std::endl;
        }
        
        Config(const Config& other) : name(other.name + "_copy"), port(other.port), 
                                     debug(other.debug), modules(other.modules) {
            std::cout << "Config copy constructor: " << name << std::endl;
        }
        
        Config(Config&& other) noexcept : name(std::move(other.name)), port(other.port),
                                         debug(other.debug), modules(std::move(other.modules)) {
            std::cout << "Config move constructor: " << name << std::endl;
        }
        
        ~Config() {
            std::cout << "Config destructor: " << name << std::endl;
        }
        
        void print() const {
            std::cout << "Configuration '" << name << "':" << std::endl;
            std::cout << "  Port: " << port << std::endl;
            std::cout << "  Debug: " << std::boolalpha << debug << std::endl;
            std::cout << "  Modules: ";
            for (const auto& module : modules) {
                std::cout << module << " ";
            }
            std::cout << std::endl;
        }
    };
    
    Config build() {
        return Config(name_, port_, debug_, modules_);  // Guaranteed copy elision
    }
};

void demonstrate_builder_pattern() {
    std::cout << "\n=== Builder Pattern with Copy Elision ===" << std::endl;
    
    // Builder pattern - guaranteed copy elision in build()
    auto config = ConfigBuilder("WebServer")
        .port(9090)
        .debug(true)
        .add_module("authentication")
        .add_module("logging")
        .add_module("database")
        .build();  // No copy/move of Config object
    
    config.print();
}

// Performance demonstration
void demonstrate_performance_impact() {
    std::cout << "\n=== Performance Impact of Copy Elision ===" << std::endl;
    
    auto create_many_objects = [](int count) {
        std::vector<ExpensiveObject> objects;
        objects.reserve(count);
        
        for (int i = 0; i < count; ++i) {
            // In C++17, this is guaranteed copy elision
            objects.emplace_back("Object_" + std::to_string(i), 100);
        }
        
        return objects;  // RVO (Return Value Optimization)
    };
    
    std::cout << "Creating vector of objects..." << std::endl;
    auto objects = create_many_objects(3);
    
    std::cout << "Created " << objects.size() << " objects" << std::endl;
}

int main() {
    demonstrate_structured_bindings_basic();
    demonstrate_custom_structured_bindings();
    demonstrate_structured_binding_modifiers();
    
    demonstrate_if_constexpr_basic();
    demonstrate_advanced_if_constexpr();
    demonstrate_constexpr_vs_sfinae();
    
    demonstrate_inline_variables();
    demonstrate_header_only_library();
    
    demonstrate_basic_fold_expressions();
    demonstrate_advanced_fold_expressions();
    demonstrate_practical_fold_examples();
    
    demonstrate_basic_ctad();
    demonstrate_advanced_ctad();
    demonstrate_ctad_with_functions();
    demonstrate_practical_ctad();
    
    demonstrate_guaranteed_copy_elision();
    demonstrate_builder_pattern();
    demonstrate_performance_impact();
    
    return 0;
}
```

## Summary

C++17 core language features significantly enhance the language's expressiveness and safety:

- **Structured bindings**: Unpack tuples, pairs, and custom types into individual variables
- **if constexpr**: Compile-time conditional compilation for templates
- **Inline variables**: Enable header-only libraries with global variables
- **Fold expressions**: Simplify variadic template patterns with concise syntax
- **Class template argument deduction**: Automatic deduction of template arguments
- **Guaranteed copy elision**: Eliminate temporary objects in specific scenarios

Key benefits:
- More readable and maintainable code
- Better template metaprogramming capabilities
- Improved performance through guaranteed optimizations
- Reduced boilerplate in template programming
- Enhanced support for generic programming patterns

These features work together to make C++ more expressive while maintaining its performance characteristics and backward compatibility.
