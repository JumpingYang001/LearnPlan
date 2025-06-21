# C++11 Type Traits and Metaprogramming

## Overview

C++11 enhanced template metaprogramming capabilities significantly with type traits, SFINAE improvements, variadic templates, and better template argument deduction. These features enable powerful compile-time programming and generic library development.

## Key Concepts

### 1. Type Traits Library

The `<type_traits>` header provides utilities for compile-time type information and manipulation.

#### Basic Type Traits

```cpp
#include <iostream>
#include <type_traits>
#include <string>
#include <vector>

void demonstrate_basic_type_traits() {
    std::cout << "\n=== Basic Type Traits ===" << std::endl;
    
    // Primary type categories
    std::cout << "int is integral: " << std::is_integral<int>::value << std::endl;
    std::cout << "float is floating point: " << std::is_floating_point<float>::value << std::endl;
    std::cout << "int* is pointer: " << std::is_pointer<int*>::value << std::endl;
    std::cout << "int& is reference: " << std::is_reference<int&>::value << std::endl;
    std::cout << "std::string is class: " << std::is_class<std::string>::value << std::endl;
    
    // Composite type categories
    std::cout << "int is arithmetic: " << std::is_arithmetic<int>::value << std::endl;
    std::cout << "int* is scalar: " << std::is_scalar<int*>::value << std::endl;
    std::cout << "std::vector<int> is compound: " << std::is_compound<std::vector<int>>::value << std::endl;
    
    // Type properties
    std::cout << "const int is const: " << std::is_const<const int>::value << std::endl;
    std::cout << "volatile int is volatile: " << std::is_volatile<volatile int>::value << std::endl;
    std::cout << "int is signed: " << std::is_signed<int>::value << std::endl;
    std::cout << "unsigned int is unsigned: " << std::is_unsigned<unsigned int>::value << std::endl;
    
    // Type relationships
    std::cout << "int and int are same: " << std::is_same<int, int>::value << std::endl;
    std::cout << "int and const int are same: " << std::is_same<int, const int>::value << std::endl;
    std::cout << "Derived* convertible to Base*: " 
              << std::is_convertible<int*, void*>::value << std::endl;
}

// Custom type traits
template<typename T>
struct is_string : std::false_type {};

template<>
struct is_string<std::string> : std::true_type {};

template<>
struct is_string<const char*> : std::true_type {};

template<>
struct is_string<char*> : std::true_type {};

// Helper variable template (C++14 style, but can be simulated in C++11)
template<typename T>
constexpr bool is_string_v = is_string<T>::value;

void demonstrate_custom_type_traits() {
    std::cout << "\n=== Custom Type Traits ===" << std::endl;
    
    std::cout << "std::string is string: " << is_string<std::string>::value << std::endl;
    std::cout << "const char* is string: " << is_string<const char*>::value << std::endl;
    std::cout << "int is string: " << is_string<int>::value << std::endl;
}
```

#### Type Transformations

```cpp
#include <iostream>
#include <type_traits>

template<typename T>
void print_type_info() {
    std::cout << "Original type info:" << std::endl;
    std::cout << "  Size: " << sizeof(T) << " bytes" << std::endl;
    std::cout << "  Is const: " << std::is_const<T>::value << std::endl;
    std::cout << "  Is pointer: " << std::is_pointer<T>::value << std::endl;
    std::cout << "  Is reference: " << std::is_reference<T>::value << std::endl;
    
    // Type modifications
    using no_const = typename std::remove_const<T>::type;
    using no_ref = typename std::remove_reference<T>::type;
    using no_ptr = typename std::remove_pointer<T>::type;
    using no_cv = typename std::remove_cv<T>::type;
    
    std::cout << "After remove_const, is const: " 
              << std::is_const<no_const>::value << std::endl;
    std::cout << "After remove_reference, is reference: " 
              << std::is_reference<no_ref>::value << std::endl;
    std::cout << "After remove_pointer, is pointer: " 
              << std::is_pointer<no_ptr>::value << std::endl;
    
    // Add modifications
    using add_const = typename std::add_const<T>::type;
    using add_ptr = typename std::add_pointer<T>::type;
    using add_lref = typename std::add_lvalue_reference<T>::type;
    using add_rref = typename std::add_rvalue_reference<T>::type;
    
    std::cout << "After add_const, is const: " 
              << std::is_const<add_const>::value << std::endl;
    std::cout << "After add_pointer, is pointer: " 
              << std::is_pointer<add_ptr>::value << std::endl;
    std::cout << "After add_lvalue_reference, is reference: " 
              << std::is_reference<add_lref>::value << std::endl;
}

void demonstrate_type_transformations() {
    std::cout << "\n=== Type Transformations ===" << std::endl;
    
    std::cout << "\n--- int ---" << std::endl;
    print_type_info<int>();
    
    std::cout << "\n--- const int& ---" << std::endl;
    print_type_info<const int&>();
    
    std::cout << "\n--- int* ---" << std::endl;
    print_type_info<int*>();
}

// Advanced type manipulation
template<typename T>
struct decay_equivalent {
    using U = typename std::remove_reference<T>::type;
    using type = typename std::conditional<
        std::is_array<U>::value,
        typename std::remove_extent<U>::type*,
        typename std::conditional<
            std::is_function<U>::value,
            typename std::add_pointer<U>::type,
            typename std::remove_cv<U>::type
        >::type
    >::type;
};

void demonstrate_advanced_transformations() {
    std::cout << "\n=== Advanced Type Transformations ===" << std::endl;
    
    // std::decay removes references, cv-qualifiers, and converts arrays/functions to pointers
    std::cout << "std::decay examples:" << std::endl;
    std::cout << "const int& -> int: " 
              << std::is_same<std::decay<const int&>::type, int>::value << std::endl;
    std::cout << "int[10] -> int*: " 
              << std::is_same<std::decay<int[10]>::type, int*>::value << std::endl;
    std::cout << "int() -> int(*)(): " 
              << std::is_same<std::decay<int()>::type, int(*)()>::value << std::endl;
    
    // std::conditional - compile-time if
    using result_type = std::conditional<true, int, double>::type;  // int
    using result_type2 = std::conditional<false, int, double>::type; // double
    
    std::cout << "conditional<true, int, double> is int: " 
              << std::is_same<result_type, int>::value << std::endl;
    std::cout << "conditional<false, int, double> is double: " 
              << std::is_same<result_type2, double>::value << std::endl;
}
```

### 2. SFINAE (Substitution Failure Is Not An Error)

SFINAE allows template specialization based on whether certain expressions are valid.

```cpp
#include <iostream>
#include <type_traits>
#include <vector>
#include <string>

// Basic SFINAE example
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
process_integral(T value) {
    std::cout << "Processing integral: " << value << std::endl;
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
process_integral(T value) {
    std::cout << "Processing floating point: " << value << std::endl;
}

// SFINAE with member detection
template<typename T>
class has_size_method {
private:
    template<typename U>
    static auto test(int) -> decltype(std::declval<U>().size(), std::true_type{});
    
    template<typename>
    static std::false_type test(...);
    
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename T>
class has_push_back_method {
private:
    template<typename U>
    static auto test(int) -> decltype(
        std::declval<U>().push_back(std::declval<typename U::value_type>()),
        std::true_type{}
    );
    
    template<typename>
    static std::false_type test(...);
    
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename Container>
typename std::enable_if<has_size_method<Container>::value, size_t>::type
get_container_size(const Container& c) {
    return c.size();
}

template<typename Container>
typename std::enable_if<!has_size_method<Container>::value, size_t>::type
get_container_size(const Container& c) {
    return std::distance(std::begin(c), std::end(c));
}

void demonstrate_sfinae() {
    std::cout << "\n=== SFINAE Examples ===" << std::endl;
    
    // Basic SFINAE
    process_integral(42);        // Calls integral version
    process_integral(3.14);      // Calls floating point version
    
    // Member detection
    std::cout << "std::vector has size method: " 
              << has_size_method<std::vector<int>>::value << std::endl;
    std::cout << "int has size method: " 
              << has_size_method<int>::value << std::endl;
    std::cout << "std::vector has push_back method: " 
              << has_push_back_method<std::vector<int>>::value << std::endl;
    std::cout << "std::string has push_back method: " 
              << has_push_back_method<std::string>::value << std::endl;
    
    // Container size detection
    std::vector<int> vec = {1, 2, 3, 4, 5};
    int arr[] = {1, 2, 3};
    
    std::cout << "Vector size: " << get_container_size(vec) << std::endl;
    std::cout << "Array size: " << get_container_size(arr) << std::endl;
}

// Expression SFINAE (C++11)
template<typename T, typename U>
auto add(T t, U u) -> decltype(t + u) {
    return t + u;
}

// This version won't be considered if T + U is invalid
template<typename T, typename U>
void add(...) {
    std::cout << "Addition not supported for these types" << std::endl;
}

void demonstrate_expression_sfinae() {
    std::cout << "\n=== Expression SFINAE ===" << std::endl;
    
    auto result1 = add(5, 3);        // int + int
    auto result2 = add(2.5, 1.5);    // double + double
    auto result3 = add(std::string("Hello"), std::string(" World")); // string + string
    
    std::cout << "5 + 3 = " << result1 << std::endl;
    std::cout << "2.5 + 1.5 = " << result2 << std::endl;
    std::cout << "\"Hello\" + \" World\" = " << result3 << std::endl;
}
```

### 3. Variadic Templates

Templates that accept a variable number of template parameters.

```cpp
#include <iostream>
#include <tuple>
#include <string>
#include <sstream>

// Basic variadic template
template<typename... Args>
void print_args(Args... args) {
    // C++11 way using initializer list trick
    auto dummy = {(std::cout << args << " ", 0)...};
    (void)dummy; // Suppress unused variable warning
    std::cout << std::endl;
}

// Recursive variadic template
template<typename T>
void print_recursive(T&& t) {
    std::cout << t << std::endl;
}

template<typename T, typename... Args>
void print_recursive(T&& t, Args&&... args) {
    std::cout << t << " ";
    print_recursive(args...);
}

// Parameter pack operations
template<typename... Args>
void print_info(Args... args) {
    std::cout << "Number of arguments: " << sizeof...(args) << std::endl;
    std::cout << "Number of types: " << sizeof...(Args) << std::endl;
    
    print_recursive(args...);
}

void demonstrate_basic_variadic() {
    std::cout << "\n=== Basic Variadic Templates ===" << std::endl;
    
    print_args(1, 2.5, "hello", 'c');
    
    print_info(42, 3.14, std::string("world"), true);
}

// Advanced variadic: perfect forwarding
template<typename... Args>
auto make_tuple_perfect(Args&&... args) -> std::tuple<Args...> {
    return std::tuple<Args...>(std::forward<Args>(args)...);
}

// Variadic class template
template<typename... Types>
class variant_printer;

template<>
class variant_printer<> {
public:
    static void print() {
        std::cout << "No more types" << std::endl;
    }
};

template<typename T, typename... Rest>
class variant_printer<T, Rest...> {
public:
    static void print() {
        std::cout << "Type: " << typeid(T).name() << std::endl;
        variant_printer<Rest...>::print();
    }
};

// Fold expressions simulation (C++11 way)
template<typename... Args>
auto sum_all(Args... args) -> decltype((args + ...)) {  // This is C++17 syntax
    // C++11 way to sum:
    return sum_impl(args...);
}

template<typename T>
T sum_impl(T t) {
    return t;
}

template<typename T, typename... Args>
T sum_impl(T first, Args... rest) {
    return first + sum_impl(rest...);
}

// Variadic function template with perfect forwarding
template<typename F, typename... Args>
auto call_with_timing(F&& func, Args&&... args) 
    -> decltype(func(std::forward<Args>(args)...)) {
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;
    
    return result;
}

int expensive_calculation(int a, int b, int c) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return a * b + c;
}

void demonstrate_advanced_variadic() {
    std::cout << "\n=== Advanced Variadic Templates ===" << std::endl;
    
    // Perfect forwarding
    auto t1 = make_tuple_perfect(1, 2.5, std::string("hello"));
    std::cout << "Tuple size: " << std::tuple_size<decltype(t1)>::value << std::endl;
    
    // Variadic class template
    variant_printer<int, double, std::string, char>::print();
    
    // Sum with variadic
    auto result = sum_impl(1, 2, 3, 4, 5);
    std::cout << "Sum: " << result << std::endl;
    
    // Function timing
    auto calc_result = call_with_timing(expensive_calculation, 10, 20, 30);
    std::cout << "Calculation result: " << calc_result << std::endl;
}

// Type list operations
template<typename... Types>
struct type_list {};

template<typename List>
struct list_size;

template<typename... Types>
struct list_size<type_list<Types...>> {
    static constexpr size_t value = sizeof...(Types);
};

template<size_t N, typename List>
struct nth_type;

template<size_t N, typename T, typename... Rest>
struct nth_type<N, type_list<T, Rest...>> {
    using type = typename nth_type<N-1, type_list<Rest...>>::type;
};

template<typename T, typename... Rest>
struct nth_type<0, type_list<T, Rest...>> {
    using type = T;
};

// Index sequence (C++11 implementation of C++14 feature)
template<size_t... Indices>
struct index_sequence {};

template<size_t N, size_t... Indices>
struct make_index_sequence_impl {
    using type = typename make_index_sequence_impl<N-1, N-1, Indices...>::type;
};

template<size_t... Indices>
struct make_index_sequence_impl<0, Indices...> {
    using type = index_sequence<Indices...>;
};

template<size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;

// Apply function to tuple elements
template<typename F, typename Tuple, size_t... Indices>
auto apply_impl(F&& f, Tuple&& t, index_sequence<Indices...>)
    -> decltype(f(std::get<Indices>(t)...)) {
    return f(std::get<Indices>(t)...);
}

template<typename F, typename Tuple>
auto apply(F&& f, Tuple&& t)
    -> decltype(apply_impl(std::forward<F>(f), 
                          std::forward<Tuple>(t),
                          make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{})) {
    return apply_impl(std::forward<F>(f), 
                     std::forward<Tuple>(t),
                     make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
}

void demonstrate_type_list_operations() {
    std::cout << "\n=== Type List Operations ===" << std::endl;
    
    using my_types = type_list<int, double, std::string, char>;
    
    std::cout << "Type list size: " << list_size<my_types>::value << std::endl;
    
    using second_type = nth_type<1, my_types>::type;  // double
    std::cout << "Second type is double: " 
              << std::is_same<second_type, double>::value << std::endl;
    
    // Apply function to tuple
    auto my_tuple = std::make_tuple(1, 2.5, std::string("hello"));
    
    auto print_tuple = [](auto&&... args) {
        auto dummy = {(std::cout << args << " ", 0)...};
        (void)dummy;
        std::cout << std::endl;
    };
    
    std::cout << "Applying function to tuple: ";
    apply(print_tuple, my_tuple);
}
```

### 4. Template Argument Deduction and Auto

Enhanced template argument deduction and the `auto` keyword for type deduction.

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <functional>

// Function template argument deduction
template<typename T, typename U>
auto multiply(T t, U u) -> decltype(t * u) {
    return t * u;
}

// Class template argument deduction (simulated for C++11)
template<typename T>
class SimpleContainer {
private:
    std::vector<T> data;
    
public:
    SimpleContainer(std::initializer_list<T> init) : data(init) {}
    
    void print() const {
        for (const auto& item : data) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
    
    size_t size() const { return data.size(); }
};

// Factory function for type deduction
template<typename T>
SimpleContainer<T> make_container(std::initializer_list<T> init) {
    return SimpleContainer<T>(init);
}

// Auto with complex types
void demonstrate_auto_deduction() {
    std::cout << "\n=== Auto Type Deduction ===" << std::endl;
    
    // Basic auto usage
    auto x = 42;           // int
    auto y = 3.14;         // double
    auto z = "hello";      // const char*
    
    std::cout << "x type size: " << sizeof(x) << std::endl;
    std::cout << "y type size: " << sizeof(y) << std::endl;
    
    // Auto with containers
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin();  // std::vector<int>::iterator
    
    // Auto with function pointers
    auto func_ptr = multiply<int, double>;
    auto result = func_ptr(5, 2.5);
    std::cout << "Function result: " << result << std::endl;
    
    // Auto with lambdas
    auto lambda = [](int a, int b) { return a + b; };
    std::cout << "Lambda result: " << lambda(10, 20) << std::endl;
    
    // Auto with complex expressions
    std::map<std::string, std::vector<int>> complex_map;
    complex_map["test"] = {1, 2, 3};
    
    auto map_it = complex_map.find("test");
    if (map_it != complex_map.end()) {
        std::cout << "Found key in map" << std::endl;
    }
}

// Trailing return type syntax
template<typename Container>
auto get_first_element(Container&& c) -> decltype(*std::begin(c)) {
    return *std::begin(c);
}

template<typename T, typename U>
auto safe_divide(T t, U u) -> decltype(t / u) {
    static_assert(std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
                  "Both arguments must be arithmetic types");
    
    if (u == 0) {
        throw std::invalid_argument("Division by zero");
    }
    
    return t / u;
}

void demonstrate_trailing_return_types() {
    std::cout << "\n=== Trailing Return Types ===" << std::endl;
    
    std::vector<int> vec = {10, 20, 30};
    auto first = get_first_element(vec);
    std::cout << "First element: " << first << std::endl;
    
    try {
        auto division_result = safe_divide(10.0, 3.0);
        std::cout << "Division result: " << division_result << std::endl;
        
        // This will throw
        auto error_result = safe_divide(10, 0);
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }
    
    // Factory function usage
    auto container = make_container({1, 2, 3, 4, 5});
    std::cout << "Container size: " << container.size() << std::endl;
    container.print();
}

// decltype usage
template<typename T>
class advanced_container {
private:
    T data;
    
public:
    advanced_container(T d) : data(std::move(d)) {}
    
    // Return type depends on what data.begin() returns
    auto begin() -> decltype(data.begin()) {
        return data.begin();
    }
    
    auto end() -> decltype(data.end()) {
        return data.end();
    }
    
    // Const versions
    auto begin() const -> decltype(data.begin()) {
        return data.begin();
    }
    
    auto end() const -> decltype(data.end()) {
        return data.end();
    }
    
    // Size member that works for any container with size()
    auto size() const -> decltype(data.size()) {
        return data.size();
    }
};

void demonstrate_decltype() {
    std::cout << "\n=== decltype Usage ===" << std::endl;
    
    // Basic decltype
    int x = 42;
    decltype(x) y = x;  // y is int
    
    std::cout << "x and y are same type: " 
              << std::is_same<decltype(x), decltype(y)>::value << std::endl;
    
    // decltype with expressions
    int a = 5, b = 10;
    decltype(a + b) sum = a + b;  // int
    decltype(a * 2.5) product = a * 2.5;  // double
    
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Product: " << product << std::endl;
    
    // Advanced container usage
    advanced_container<std::vector<std::string>> str_container({{"hello", "world", "c++"}});
    
    std::cout << "String container contents: ";
    for (const auto& str : str_container) {
        std::cout << str << " ";
    }
    std::cout << std::endl;
    std::cout << "Container size: " << str_container.size() << std::endl;
}
```

## Practical Applications

### Generic Algorithm Implementation

```cpp
#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <iterator>
#include <type_traits>

// Generic find_if implementation
template<typename Iterator, typename Predicate>
Iterator my_find_if(Iterator first, Iterator last, Predicate pred) {
    while (first != last) {
        if (pred(*first)) {
            return first;
        }
        ++first;
    }
    return last;
}

// Generic transform implementation with type deduction
template<typename InputIt, typename OutputIt, typename UnaryOp>
OutputIt my_transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp op) {
    while (first != last) {
        *d_first++ = op(*first++);
    }
    return d_first;
}

// Iterator category detection and optimization
template<typename Iterator>
struct iterator_traits_helper {
    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using reference = typename std::iterator_traits<Iterator>::reference;
};

// Optimized distance calculation based on iterator category
template<typename Iterator>
typename std::iterator_traits<Iterator>::difference_type
my_distance_impl(Iterator first, Iterator last, std::input_iterator_tag) {
    typename std::iterator_traits<Iterator>::difference_type count = 0;
    while (first != last) {
        ++first;
        ++count;
    }
    return count;
}

template<typename Iterator>
typename std::iterator_traits<Iterator>::difference_type
my_distance_impl(Iterator first, Iterator last, std::random_access_iterator_tag) {
    return last - first;  // O(1) for random access iterators
}

template<typename Iterator>
typename std::iterator_traits<Iterator>::difference_type
my_distance(Iterator first, Iterator last) {
    using category = typename std::iterator_traits<Iterator>::iterator_category;
    return my_distance_impl(first, last, category{});
}

void demonstrate_generic_algorithms() {
    std::cout << "\n=== Generic Algorithm Implementation ===" << std::endl;
    
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::list<std::string> str_list = {"apple", "banana", "cherry", "date"};
    
    // Generic find_if
    auto it = my_find_if(vec.begin(), vec.end(), [](int x) { return x > 5; });
    if (it != vec.end()) {
        std::cout << "First element > 5: " << *it << std::endl;
    }
    
    // Generic transform
    std::vector<int> squared;
    squared.resize(vec.size());
    my_transform(vec.begin(), vec.end(), squared.begin(), [](int x) { return x * x; });
    
    std::cout << "Squared values: ";
    for (int val : squared) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Distance calculation with different iterator types
    std::cout << "Vector distance (random access): " 
              << my_distance(vec.begin(), vec.end()) << std::endl;
    std::cout << "List distance (bidirectional): " 
              << my_distance(str_list.begin(), str_list.end()) << std::endl;
}
```

### Type-Safe Configuration System

```cpp
#include <iostream>
#include <string>
#include <unordered_map>
#include <type_traits>
#include <stdexcept>

// Type-safe configuration value
class ConfigValue {
private:
    enum class Type { INT, DOUBLE, STRING, BOOL };
    
    union {
        int int_val;
        double double_val;
        bool bool_val;
    };
    std::string string_val;
    Type type;
    
public:
    ConfigValue(int val) : int_val(val), type(Type::INT) {}
    ConfigValue(double val) : double_val(val), type(Type::DOUBLE) {}
    ConfigValue(const std::string& val) : string_val(val), type(Type::STRING) {}
    ConfigValue(bool val) : bool_val(val), type(Type::BOOL) {}
    
    template<typename T>
    T get() const {
        static_assert(std::is_same<T, int>::value || 
                     std::is_same<T, double>::value ||
                     std::is_same<T, std::string>::value ||
                     std::is_same<T, bool>::value,
                     "Unsupported type for ConfigValue::get()");
        
        if constexpr (std::is_same<T, int>::value) {
            if (type != Type::INT) throw std::runtime_error("Type mismatch: expected int");
            return int_val;
        } else if constexpr (std::is_same<T, double>::value) {
            if (type != Type::DOUBLE) throw std::runtime_error("Type mismatch: expected double");
            return double_val;
        } else if constexpr (std::is_same<T, std::string>::value) {
            if (type != Type::STRING) throw std::runtime_error("Type mismatch: expected string");
            return string_val;
        } else if constexpr (std::is_same<T, bool>::value) {
            if (type != Type::BOOL) throw std::runtime_error("Type mismatch: expected bool");
            return bool_val;
        }
    }
    
    // C++11 compatible version using SFINAE
    template<typename T>
    typename std::enable_if<std::is_same<T, int>::value, T>::type
    get_sfinae() const {
        if (type != Type::INT) throw std::runtime_error("Type mismatch: expected int");
        return int_val;
    }
    
    template<typename T>
    typename std::enable_if<std::is_same<T, double>::value, T>::type
    get_sfinae() const {
        if (type != Type::DOUBLE) throw std::runtime_error("Type mismatch: expected double");
        return double_val;
    }
    
    template<typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, T>::type
    get_sfinae() const {
        if (type != Type::STRING) throw std::runtime_error("Type mismatch: expected string");
        return string_val;
    }
    
    template<typename T>
    typename std::enable_if<std::is_same<T, bool>::value, T>::type
    get_sfinae() const {
        if (type != Type::BOOL) throw std::runtime_error("Type mismatch: expected bool");
        return bool_val;
    }
};

class Configuration {
private:
    std::unordered_map<std::string, ConfigValue> values;
    
public:
    template<typename T>
    void set(const std::string& key, T value) {
        values.emplace(key, ConfigValue(value));
    }
    
    template<typename T>
    T get(const std::string& key) const {
        auto it = values.find(key);
        if (it == values.end()) {
            throw std::runtime_error("Key not found: " + key);
        }
        return it->second.get_sfinae<T>();
    }
    
    bool has_key(const std::string& key) const {
        return values.find(key) != values.end();
    }
};

void demonstrate_type_safe_config() {
    std::cout << "\n=== Type-Safe Configuration System ===" << std::endl;
    
    Configuration config;
    
    // Set various types
    config.set("port", 8080);
    config.set("timeout", 30.5);
    config.set("server_name", std::string("MyServer"));
    config.set("debug_mode", true);
    
    try {
        // Get values with correct types
        int port = config.get<int>("port");
        double timeout = config.get<double>("timeout");
        std::string server_name = config.get<std::string>("server_name");
        bool debug = config.get<bool>("debug_mode");
        
        std::cout << "Port: " << port << std::endl;
        std::cout << "Timeout: " << timeout << std::endl;
        std::cout << "Server name: " << server_name << std::endl;
        std::cout << "Debug mode: " << std::boolalpha << debug << std::endl;
        
        // This will throw a type mismatch error
        std::string wrong_type = config.get<std::string>("port");
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}
```

## Exercises

### Exercise 1: Custom Type Traits
Create type traits to detect if a type has specific member functions (e.g., `begin()`, `end()`, `size()`).

### Exercise 2: Variadic Logger
Implement a variadic template logger that can format and log messages with different types of arguments.

### Exercise 3: Template Factory
Design a factory class template that uses perfect forwarding and type deduction to create objects.

### Exercise 4: Compile-time String Processing
Create template metafunctions to process string literals at compile time.

## Summary

C++11 metaprogramming features provide powerful tools for generic programming:

- **Type traits**: Compile-time type information and manipulation
- **SFINAE**: Template specialization based on expression validity
- **Variadic templates**: Variable number of template parameters
- **Auto and decltype**: Enhanced type deduction
- **Perfect forwarding**: Preserving value categories in generic code

These features enable:
- More generic and reusable code
- Better compile-time error messages
- Performance optimizations through compile-time computation
- Type-safe interfaces with minimal runtime overhead
- Powerful library design patterns

Best practices:
- Use type traits for template constraints and optimizations
- Employ SFINAE for graceful template specialization
- Leverage variadic templates for flexible interfaces
- Use perfect forwarding in generic wrapper functions
- Combine features for powerful metaprogramming solutions
