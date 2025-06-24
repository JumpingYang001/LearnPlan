# Generic Programming Utilities

*Duration: 1 week*

# Generic Programming Utilities

*Duration: 1 week*

## Overview

Generic programming and metaprogramming are powerful C++ techniques that allow you to write code that operates on types rather than values, enabling highly flexible and efficient libraries. Boost provides three fundamental libraries that make advanced generic programming accessible and practical:

- **Boost.TypeTraits**: Advanced type inspection and manipulation
- **Boost.MPL**: Compile-time algorithms and data structures  
- **Boost.Fusion**: Bridge between compile-time and runtime programming

This section will take you from basic concepts to advanced metaprogramming techniques, showing you how to leverage these tools to write more generic, efficient, and maintainable code.

### Why Generic Programming Matters

**Problem Scenario**: Imagine you need to create a serialization library that can handle any data type, optimize performance based on type characteristics, and provide compile-time safety guarantees.

```cpp
// Without metaprogramming - limited and repetitive
void serialize_int(int value, std::ostream& out);
void serialize_double(double value, std::ostream& out);
void serialize_string(const std::string& value, std::ostream& out);
// ... hundreds of functions for different types

// With metaprogramming - generic and extensible
template<typename T>
void serialize(const T& value, std::ostream& out) {
    // Automatically adapts behavior based on T's characteristics
    // Compile-time optimizations based on type traits
    // Single implementation handles infinite types
}
```

### Learning Path Overview

1. **Foundation**: Understanding type traits and compile-time programming
2. **Core Skills**: Using MPL for compile-time algorithms
3. **Integration**: Bridging compile-time and runtime with Fusion
4. **Advanced**: Building practical metaprogramming solutions
5. **Mastery**: Performance optimization and best practices

## Learning Topics

### Boost.TypeTraits - Type Inspection and Manipulation

#### What are Type Traits?
Type traits are compile-time utilities that provide information about types and enable type-based template specialization. They're the foundation of modern generic programming.

**Key Concepts:**
- **Type inspection**: Query properties of types at compile time
- **Type transformation**: Create new types based on existing ones
- **SFINAE**: "Substitution Failure Is Not An Error" - enable/disable templates
- **Template specialization**: Customize behavior for specific types

#### Why Use Type Traits?
```cpp
// Problem: Different types need different handling
template<typename T>
void process(T value) {
    // How do we know if T is:
    // - A pointer that needs null checking?
    // - A large object that should be passed by reference?
    // - An arithmetic type that supports mathematical operations?
    // - A container that has iterators?
}

// Solution: Type traits provide compile-time answers
template<typename T>
void process(T value) {
    if constexpr (boost::is_pointer<T>::value) {
        // Handle pointer types
    } else if constexpr (boost::is_arithmetic<T>::value) {
        // Handle numeric types
    } else if constexpr (boost::is_class<T>::value) {
        // Handle class types
    }
}
```

#### Advanced Type Traits Beyond std::type_traits

Boost.TypeTraits provides many utilities not available in the standard library:

```cpp
#include <boost/type_traits.hpp>

// Function type analysis
template<typename F>
void analyze_function() {
    std::cout << "Is function: " << boost::is_function<F>::value << "\n";
    std::cout << "Function arity: " << boost::function_traits<F>::arity << "\n";
    
    if constexpr (boost::is_function<F>::value) {
        using return_type = typename boost::function_traits<F>::result_type;
        using first_arg = typename boost::function_traits<F>::arg1_type;
        std::cout << "Return type size: " << sizeof(return_type) << "\n";
        std::cout << "First arg size: " << sizeof(first_arg) << "\n";
    }
}

// Usage
int add(int a, int b) { return a + b; }
analyze_function<decltype(add)>();
```

#### Type Information and Manipulation at Compile Time

**Real-world Example: Smart Parameter Passing**
```cpp
// Automatically choose optimal parameter passing strategy
template<typename T>
struct parameter_type {
    static constexpr bool is_small = sizeof(T) <= sizeof(void*);
    static constexpr bool is_trivial = boost::is_trivially_copyable<T>::value;
    
    using type = typename boost::conditional<
        is_small && is_trivial,
        T,                    // Pass small trivial types by value
        const T&              // Pass others by const reference
    >::type;
};

// Automatic optimization based on type characteristics
template<typename T>
void optimized_function(typename parameter_type<T>::type param) {
    // Compiler automatically chooses best passing method
    std::cout << "Processing: " << param << "\n";
}

// Usage - same interface, different optimizations
optimized_function<int>(42);           // Passed by value
optimized_function<std::string>("hi"); // Passed by const&
optimized_function<std::vector<int>>(vec); // Passed by const&
```

#### SFINAE and Template Metaprogramming Support

**SFINAE** enables conditional template instantiation:

```cpp
// Enable function only for arithmetic types
template<typename T>
typename boost::enable_if<boost::is_arithmetic<T>, T>::type
safe_divide(T numerator, T denominator) {
    if (denominator == T(0)) {
        throw std::runtime_error("Division by zero");
    }
    return numerator / denominator;
}

// Different overload for non-arithmetic types
template<typename T>
typename boost::disable_if<boost::is_arithmetic<T>, void>::type
safe_divide(const T&, const T&) {
    static_assert(boost::is_arithmetic<T>::value, 
                  "Division only supported for arithmetic types");
}

// Modern C++ equivalent (for comparison)
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, T>
modern_safe_divide(T numerator, T denominator) {
    // Same logic...
}
```

#### Integration with Template Specialization

**Advanced Example: Automatic Serialization Framework**
```cpp
#include <boost/type_traits.hpp>
#include <sstream>

// Base serialization interface
template<typename T>
struct serializer {
    static std::string serialize(const T& value);
};

// Specialization for arithmetic types
template<typename T>
struct serializer<T, typename boost::enable_if<boost::is_arithmetic<T>>::type> {
    static std::string serialize(const T& value) {
        return std::to_string(value);
    }
};

// Specialization for string types
template<>
struct serializer<std::string> {
    static std::string serialize(const std::string& value) {
        return "\"" + value + "\"";
    }
};

// Specialization for container types
template<typename T>
struct serializer<T, typename boost::enable_if<
    boost::is_same<T, std::vector<typename T::value_type>>::value
>::type> {
    static std::string serialize(const T& container) {
        std::ostringstream oss;
        oss << "[";
        for (auto it = container.begin(); it != container.end(); ++it) {
            if (it != container.begin()) oss << ", ";
            oss << serializer<typename T::value_type>::serialize(*it);
        }
        oss << "]";
        return oss.str();
    }
};

// Generic serialize function
template<typename T>
std::string serialize(const T& value) {
    return serializer<T>::serialize(value);
}
```

### Boost.MPL - Metaprogramming Library

#### Understanding Compile-time Programming

**What is Metaprogramming?**
Metaprogramming is programming with programs - writing code that generates or manipulates other code. In C++, template metaprogramming happens at compile time.

```cpp
// Regular programming (runtime)
std::vector<int> numbers = {1, 2, 3, 4, 5};
int sum = 0;
for (int n : numbers) {
    sum += n;  // Runtime computation
}

// Metaprogramming (compile time)
template<int... Numbers>
struct sum {
    static constexpr int value = (Numbers + ...); // C++17 fold expression
};

// Boost.MPL approach
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/int.hpp>

typedef boost::mpl::vector_c<int, 1, 2, 3, 4, 5> numbers;
typedef boost::mpl::fold<
    numbers,
    boost::mpl::int_<0>,
    boost::mpl::plus<boost::mpl::_1, boost::mpl::_2>
>::type compile_time_sum;

// Result available at compile time
static_assert(compile_time_sum::value == 15);
```

#### Compile-time Data Structures

**Sequences, Maps, Sets:**
```cpp
#include <boost/mpl/vector.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/pair.hpp>

// Type sequence
typedef boost::mpl::vector<int, double, std::string> type_list;

// Type map (compile-time key-value store)
typedef boost::mpl::map<
    boost::mpl::pair<int, std::string>,           // int -> string
    boost::mpl::pair<double, std::vector<int>>,   // double -> vector<int>
    boost::mpl::pair<char, bool>                  // char -> bool
> type_map;

// Type set (unique types only)
typedef boost::mpl::set<int, double, char, int> type_set; // int appears twice but stored once
```

#### Metafunctions and Higher-order Metaprogramming

**Metafunctions** are templates that take types as input and produce types as output:

```cpp
// Simple metafunction
template<typename T>
struct add_pointer {
    typedef T* type;
};

// Higher-order metafunction (takes other metafunctions as parameters)
template<template<typename> class MetaFunction, typename TypeList>
struct transform_all;

// Specialization for vector
template<template<typename> class MetaFunction, typename... Types>
struct transform_all<MetaFunction, boost::mpl::vector<Types...>> {
    typedef boost::mpl::vector<typename MetaFunction<Types>::type...> type;
};

// Usage
typedef boost::mpl::vector<int, double, char> original;
typedef transform_all<add_pointer, original>::type pointer_types;
// Result: vector<int*, double*, char*>
```

#### Real-world MPL Example: Type-safe State Machine

```cpp
#include <boost/mpl/vector.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/has_key.hpp>

// States
struct Idle {};
struct Working {};
struct Completed {};
struct Error {};

// Events
struct StartWork {};
struct FinishWork {};
struct ErrorOccurred {};
struct Reset {};

// State transition table
typedef boost::mpl::map<
    // From Idle
    boost::mpl::pair<boost::mpl::pair<Idle, StartWork>, Working>,
    
    // From Working
    boost::mpl::pair<boost::mpl::pair<Working, FinishWork>, Completed>,
    boost::mpl::pair<boost::mpl::pair<Working, ErrorOccurred>, Error>,
    
    // From Completed
    boost::mpl::pair<boost::mpl::pair<Completed, Reset>, Idle>,
    
    // From Error
    boost::mpl::pair<boost::mpl::pair<Error, Reset>, Idle>
> transition_table;

// Compile-time transition validation
template<typename CurrentState, typename Event>
struct is_valid_transition {
    typedef boost::mpl::pair<CurrentState, Event> transition_key;
    static constexpr bool value = boost::mpl::has_key<transition_table, transition_key>::value;
};

// Get next state
template<typename CurrentState, typename Event>
struct next_state {
    static_assert(is_valid_transition<CurrentState, Event>::value, 
                  "Invalid state transition");
    
    typedef boost::mpl::pair<CurrentState, Event> transition_key;
    typedef typename boost::mpl::at<transition_table, transition_key>::type type;
};

// Usage
static_assert(is_valid_transition<Idle, StartWork>::value);
static_assert(!is_valid_transition<Idle, FinishWork>::value);

typedef next_state<Idle, StartWork>::type after_start; // Working
typedef next_state<Working, FinishWork>::type after_finish; // Completed
```

### Boost.Fusion - Bridging Compile-time and Runtime

#### What is Fusion?

Fusion provides **heterogeneous containers** that can hold different types in a single container, bridging the gap between compile-time type manipulation and runtime data processing.

**Key Concepts:**
- **Heterogeneous sequences**: Containers holding different types
- **Compile-time algorithms**: Operations that work on type information
- **Runtime algorithms**: Operations that work on actual data
- **Introspection**: Querying container properties at both compile-time and runtime

#### Heterogeneous Containers and Algorithms

**Comparison with Standard Containers:**
```cpp
// Standard homogeneous container
std::vector<int> numbers = {1, 2, 3, 4, 5};
// All elements must be the same type

// Fusion heterogeneous container
#include <boost/fusion/container/vector.hpp>
auto mixed_data = boost::fusion::make_vector(
    42,                    // int
    "hello",              // const char*
    3.14,                 // double
    true,                 // bool
    std::vector<int>{1,2,3} // std::vector<int>
);
// Each element can be a different type
```

#### Compile-time and Runtime Fusion

**Compile-time Operations:**
```cpp
#include <boost/fusion/sequence/intrinsic.hpp>

// Size known at compile time
constexpr auto size = boost::fusion::size(mixed_data);
static_assert(size == 5);

// Types known at compile time
using first_type = typename boost::fusion::result_of::at_c<decltype(mixed_data), 0>::type;
static_assert(std::is_same_v<first_type, int>);
```

**Runtime Operations:**
```cpp
#include <boost/fusion/algorithm.hpp>

// Access elements at runtime
auto first_element = boost::fusion::at_c<0>(mixed_data);  // 42
auto third_element = boost::fusion::at_c<2>(mixed_data); // 3.14

// Iterate over all elements
boost::fusion::for_each(mixed_data, [](const auto& element) {
    std::cout << element << " ";
});
```

#### Integration Between Compile-time and Runtime Worlds

**Real-world Example: Automatic Struct Serialization**
```cpp
#include <boost/fusion/adapted/struct.hpp>
#include <boost/fusion/algorithm.hpp>
#include <sstream>

// Regular struct
struct Person {
    std::string name;
    int age;
    double salary;
    bool is_employed;
};

// Make it Fusion-compatible
BOOST_FUSION_ADAPT_STRUCT(
    Person,
    (std::string, name)
    (int, age)
    (double, salary)
    (bool, is_employed)
)

// Automatic serialization visitor
struct json_serializer {
    std::ostringstream& output;
    bool first = true;
    
    template<typename T>
    void operator()(const T& value) {
        if (!first) output << ", ";
        first = false;
        
        if constexpr (std::is_same_v<T, std::string>) {
            output << "\"" << value << "\"";
        } else if constexpr (std::is_same_v<T, bool>) {
            output << (value ? "true" : "false");
        } else {
            output << value;
        }
    }
};

// Generic serialization function
template<typename T>
std::string to_json(const T& object) {
    std::ostringstream oss;
    oss << "{";
    boost::fusion::for_each(object, json_serializer{oss});
    oss << "}";
    return oss.str();
}

// Usage
Person person{"John Doe", 30, 75000.0, true};
std::string json = to_json(person);
// Result: {"John Doe", 30, 75000, true}
```

#### Sequence Manipulation and Transformation

**Advanced Example: Database Query Builder**
```cpp
#include <boost/fusion/container/map.hpp>
#include <boost/fusion/support/pair.hpp>

// Define field types
struct name_field {};
struct age_field {};
struct salary_field {};

// Create a query
auto query_fields = boost::fusion::make_map<name_field, age_field, salary_field>(
    std::string("John%"),     // LIKE pattern
    std::make_pair(25, 65),   // age range
    std::make_pair(50000.0, 100000.0) // salary range
);

// Generate SQL at runtime using compile-time information
template<typename QueryMap>
std::string generate_sql(const QueryMap& fields) {
    std::string sql = "SELECT * FROM employees WHERE ";
    
    bool first = true;
    boost::fusion::for_each(fields, [&](const auto& field_pair) {
        if (!first) sql += " AND ";
        first = false;
        
        // Type-based query generation
        using field_type = typename std::decay_t<decltype(field_pair)>::first_type;
        const auto& value = field_pair.second;
        
        if constexpr (std::is_same_v<field_type, name_field>) {
            sql += "name LIKE '" + value + "'";
        } else if constexpr (std::is_same_v<field_type, age_field>) {
            sql += "age BETWEEN " + std::to_string(value.first) + 
                   " AND " + std::to_string(value.second);
        } else if constexpr (std::is_same_v<field_type, salary_field>) {
            sql += "salary BETWEEN " + std::to_string(value.first) + 
                   " AND " + std::to_string(value.second);
        }
    });
    
    return sql;
}

// Result: "SELECT * FROM employees WHERE name LIKE 'John%' AND age BETWEEN 25 AND 65 AND salary BETWEEN 50000 AND 100000"
```

## Code Examples

### Boost.TypeTraits - Basic Usage
```cpp
#include <boost/type_traits.hpp>
#include <iostream>
#include <vector>
#include <string>

template<typename T>
void analyze_type() {
    std::cout << "Type analysis for: " << typeid(T).name() << "\n";
    std::cout << "  Is fundamental: " << boost::is_fundamental<T>::value << "\n";
    std::cout << "  Is arithmetic: " << boost::is_arithmetic<T>::value << "\n";
    std::cout << "  Is integral: " << boost::is_integral<T>::value << "\n";
    std::cout << "  Is floating point: " << boost::is_floating_point<T>::value << "\n";
    std::cout << "  Is pointer: " << boost::is_pointer<T>::value << "\n";
    std::cout << "  Is reference: " << boost::is_reference<T>::value << "\n";
    std::cout << "  Is const: " << boost::is_const<T>::value << "\n";
    std::cout << "  Is class: " << boost::is_class<T>::value << "\n";
    std::cout << "  Is enum: " << boost::is_enum<T>::value << "\n";
    std::cout << "  Has trivial constructor: " << boost::has_trivial_constructor<T>::value << "\n";
    std::cout << "  Has trivial destructor: " << boost::has_trivial_destructor<T>::value << "\n";
    std::cout << "\n";
}

enum Color { RED, GREEN, BLUE };

class SimpleClass {
public:
    int value;
    SimpleClass() : value(0) {}
};

class ComplexClass {
public:
    std::vector<int> data;
    std::string name;
    ComplexClass(const std::string& n) : name(n) {}
    ~ComplexClass() = default;
};

void demonstrate_type_traits() {
    std::cout << "=== Type Traits Analysis ===\n";
    
    analyze_type<int>();
    analyze_type<double>();
    analyze_type<int*>();
    analyze_type<const int&>();
    analyze_type<std::string>();
    analyze_type<Color>();
    analyze_type<SimpleClass>();
    analyze_type<ComplexClass>();
}
```

### Type Trait Applications
```cpp
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <iostream>
#include <vector>

// SFINAE example using boost::enable_if
template<typename T>
typename boost::enable_if<boost::is_arithmetic<T>, T>::type
safe_divide(T numerator, T denominator) {
    if (denominator == T(0)) {
        throw std::runtime_error("Division by zero");
    }
    return numerator / denominator;
}

// Overload for non-arithmetic types
template<typename T>
typename boost::disable_if<boost::is_arithmetic<T>, void>::type
safe_divide(T, T) {
    std::cout << "Division not supported for this type\n";
}

// Template specialization helper
template<typename T>
struct is_container : boost::false_type {};

template<typename T, typename Alloc>
struct is_container<std::vector<T, Alloc>> : boost::true_type {};

// Function template with container detection
template<typename T>
void process_data(const T& data) {
    if (is_container<T>::value) {
        std::cout << "Processing container with " << data.size() << " elements\n";
    } else {
        std::cout << "Processing single value: " << data << "\n";
    }
}

// Conditional type selection
template<typename T>
struct optimal_parameter_type {
    typedef typename boost::conditional<
        boost::is_fundamental<T>::value || sizeof(T) <= sizeof(void*),
        T,  // Pass by value for small/fundamental types
        const T&  // Pass by reference for larger types
    >::type type;
};

template<typename T>
void optimized_function(typename optimal_parameter_type<T>::type param) {
    std::cout << "Received parameter of size: " << sizeof(T) << " bytes\n";
    // Process param...
}

void demonstrate_type_traits_applications() {
    std::cout << "=== Type Traits Applications ===\n";
    
    // SFINAE examples
    try {
        std::cout << "safe_divide(10, 2) = " << safe_divide(10, 2) << "\n";
        std::cout << "safe_divide(7.5, 2.5) = " << safe_divide(7.5, 2.5) << "\n";
        safe_divide(std::string("hello"), std::string("world"));
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << "\n";
    }
    
    // Container detection
    std::vector<int> vec = {1, 2, 3, 4, 5};
    int single_value = 42;
    
    process_data(vec);
    process_data(single_value);
    
    // Optimal parameter passing
    optimized_function<int>(10);
    optimized_function<std::string>(std::string("large object"));
}
```

### Boost.MPL - Compile-time Programming
```cpp
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/type_traits.hpp>
#include <iostream>

namespace mpl = boost::mpl;

// Define a type sequence
typedef mpl::vector<int, double, char, std::string> type_sequence;

// Metafunction to add pointer
template<typename T>
struct add_pointer {
    typedef T* type;
};

// Metafunction to get size
template<typename T>
struct size_of {
    static const std::size_t value = sizeof(T);
};

// Metafunction for maximum size
template<typename State, typename T>
struct max_size {
    static const std::size_t current_size = sizeof(T);
    static const std::size_t state_size = State::value;
    static const std::size_t value = (current_size > state_size) ? current_size : state_size;
};

void demonstrate_mpl_basics() {
    std::cout << "=== MPL Basics ===\n";
    
    // Sequence operations
    std::cout << "Sequence size: " << mpl::size<type_sequence>::value << "\n";
    
    // Access elements
    typedef mpl::at_c<type_sequence, 0>::type first_type;  // int
    typedef mpl::at_c<type_sequence, 1>::type second_type; // double
    
    std::cout << "First type size: " << sizeof(first_type) << "\n";
    std::cout << "Second type size: " << sizeof(second_type) << "\n";
    
    // Transform sequence
    typedef mpl::transform<type_sequence, add_pointer<mpl::_1>>::type pointer_sequence;
    typedef mpl::at_c<pointer_sequence, 0>::type first_pointer_type; // int*
    
    std::cout << "First pointer type size: " << sizeof(first_pointer_type) << "\n";
    
    // Find type in sequence
    typedef mpl::find_if<type_sequence, boost::is_floating_point<mpl::_1>>::type float_iter;
    typedef mpl::deref<float_iter>::type found_float_type; // double
    
    std::cout << "Found floating point type size: " << sizeof(found_float_type) << "\n";
    
    // Fold (accumulate) - find maximum size
    typedef mpl::fold<
        type_sequence,
        mpl::size_t<0>,
        max_size<mpl::_1, mpl::_2>
    >::type max_size_result;
    
    std::cout << "Maximum type size in sequence: " << max_size_result::value << "\n";
}
```

### Advanced MPL - Compile-time Algorithms
```cpp
#include <boost/mpl/vector.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/type_traits.hpp>
#include <iostream>
#include <string>

namespace mpl = boost::mpl;

// Type map for compile-time lookups
typedef mpl::map<
    mpl::pair<int, std::string>,
    mpl::pair<double, std::string>,
    mpl::pair<char, std::string>,
    mpl::pair<bool, std::string>
> printable_types;

// Check if type is printable
template<typename T>
struct is_printable : mpl::has_key<printable_types, T> {};

// Filter predicate
template<typename T>
struct is_small_type {
    static const bool value = sizeof(T) <= 4;
};

// Complex type sequence
typedef mpl::vector<
    char, short, int, long, long long,
    float, double, long double,
    void*, std::string
> all_types;

void demonstrate_mpl_advanced() {
    std::cout << "=== Advanced MPL ===\n";
    
    // Map operations
    std::cout << "int is printable: " << is_printable<int>::value << "\n";
    std::cout << "void* is printable: " << is_printable<void*>::value << "\n";
    
    // Filter types by size
    typedef mpl::copy_if<
        all_types,
        is_small_type<mpl::_1>,
        mpl::back_inserter<mpl::vector<>>
    >::type small_types;
    
    std::cout << "Original sequence size: " << mpl::size<all_types>::value << "\n";
    std::cout << "Small types count: " << mpl::size<small_types>::value << "\n";
    
    // Print information about small types
    typedef mpl::at_c<small_types, 0>::type first_small;
    typedef mpl::at_c<small_types, 1>::type second_small;
    
    std::cout << "First small type size: " << sizeof(first_small) << "\n";
    std::cout << "Second small type size: " << sizeof(second_small) << "\n";
    
    // Filter view (lazy evaluation)
    typedef mpl::filter_view<all_types, boost::is_integral<mpl::_1>> integral_view;
    std::cout << "Integral types count: " << mpl::size<integral_view>::value << "\n";
}
```

### Boost.Fusion - Heterogeneous Containers
```cpp
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/container/map.hpp>
#include <boost/fusion/sequence/intrinsic.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/support/pair.hpp>
#include <iostream>
#include <string>

namespace fusion = boost::fusion;

// Define keys for map
struct name_key;
struct age_key;
struct salary_key;

// Visitor for printing
struct print_visitor {
    template<typename T>
    void operator()(const T& value) const {
        std::cout << value << " ";
    }
};

// Transform functor
struct double_numeric {
    template<typename T>
    typename boost::enable_if<boost::is_arithmetic<T>, T>::type
    operator()(const T& value) const {
        return value * 2;
    }
    
    template<typename T>
    typename boost::disable_if<boost::is_arithmetic<T>, T>::type
    operator()(const T& value) const {
        return value; // Return unchanged for non-numeric types
    }
};

void demonstrate_fusion_vector() {
    std::cout << "=== Fusion Vector ===\n";
    
    // Create heterogeneous vector
    fusion::vector<int, std::string, double, bool> data(
        42, "hello", 3.14, true
    );
    
    std::cout << "Vector contents: ";
    fusion::for_each(data, print_visitor());
    std::cout << "\n";
    
    // Access elements
    std::cout << "First element: " << fusion::at_c<0>(data) << "\n";
    std::cout << "Second element: " << fusion::at_c<1>(data) << "\n";
    
    // Modify elements
    fusion::at_c<0>(data) = 100;
    fusion::at_c<2>(data) = 2.718;
    
    std::cout << "After modification: ";
    fusion::for_each(data, print_visitor());
    std::cout << "\n";
    
    // Transform
    auto transformed = fusion::transform(data, double_numeric());
    std::cout << "Transformed: ";
    fusion::for_each(transformed, print_visitor());
    std::cout << "\n";
    
    // Size and properties
    std::cout << "Vector size: " << fusion::size(data) << "\n";
    std::cout << "Is empty: " << std::boolalpha << fusion::empty(data) << "\n";
}

void demonstrate_fusion_map() {
    std::cout << "\n=== Fusion Map ===\n";
    
    // Create map with named fields
    auto person = fusion::make_map<name_key, age_key, salary_key>(
        std::string("John Doe"),
        30,
        75000.0
    );
    
    // Access by key
    std::cout << "Name: " << fusion::at_key<name_key>(person) << "\n";
    std::cout << "Age: " << fusion::at_key<age_key>(person) << "\n";
    std::cout << "Salary: " << fusion::at_key<salary_key>(person) << "\n";
    
    // Modify values
    fusion::at_key<age_key>(person) = 31;
    fusion::at_key<salary_key>(person) = 80000.0;
    
    std::cout << "After promotion:\n";
    std::cout << "Age: " << fusion::at_key<age_key>(person) << "\n";
    std::cout << "Salary: " << fusion::at_key<salary_key>(person) << "\n";
    
    // Check if key exists
    std::cout << "Has name key: " << std::boolalpha 
              << fusion::has_key<name_key>(person) << "\n";
}
```

### Compile-time String Processing
```cpp
#include <boost/mpl/string.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/size.hpp>
#include <iostream>

namespace mpl = boost::mpl;

// Compile-time string
typedef mpl::string<'Hell','o Wo','rld!'> hello_string;

// Character operations
template<int C>
struct to_upper {
    static const int value = (C >= 'a' && C <= 'z') ? C - 'a' + 'A' : C;
};

template<int C>
struct is_vowel {
    static const bool value = (C == 'a' || C == 'e' || C == 'i' || C == 'o' || C == 'u' ||
                              C == 'A' || C == 'E' || C == 'I' || C == 'O' || C == 'U');
};

// Count vowels
template<typename State, int C>
struct vowel_counter {
    static const int value = State::value + (is_vowel<C>::value ? 1 : 0);
};

void demonstrate_compile_time_strings() {
    std::cout << "=== Compile-time String Processing ===\n";
    
    // String properties
    std::cout << "String length: " << mpl::size<hello_string>::value << "\n";
    
    // Extract characters
    typedef mpl::at_c<hello_string, 0>::type first_char;
    typedef mpl::at_c<hello_string, 1>::type second_char;
    
    std::cout << "First character code: " << first_char::value << " ('" 
              << static_cast<char>(first_char::value) << "')\n";
    std::cout << "Second character code: " << second_char::value << " ('" 
              << static_cast<char>(second_char::value) << "')\n";
    
    // Transform to uppercase
    typedef mpl::transform<hello_string, to_upper<mpl::_1>>::type upper_string;
    
    // Count vowels at compile time
    typedef mpl::fold<
        hello_string,
        mpl::int_<0>,
        vowel_counter<mpl::_1, mpl::_2>
    >::type vowel_count;
    
    std::cout << "Vowel count: " << vowel_count::value << "\n";
    
    // Runtime string extraction (for printing)
    std::string runtime_string;
    mpl::for_each<hello_string>([&](auto c) {
        runtime_string += static_cast<char>(c.value);
    });
    
    std::cout << "Runtime string: " << runtime_string << "\n";
}
```

### Template Metaprogramming Utilities
```cpp
#include <boost/mpl/if.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/type_traits.hpp>
#include <iostream>
#include <vector>
#include <list>

namespace mpl = boost::mpl;

// Smart pointer selection based on type properties
template<typename T>
struct smart_pointer_selector {
    typedef typename mpl::if_<
        boost::is_polymorphic<T>,
        std::shared_ptr<T>,
        std::unique_ptr<T>
    >::type type;
};

// Container optimization based on usage pattern
template<typename T, bool RandomAccess>
struct container_selector {
    typedef typename mpl::if_c<
        RandomAccess,
        std::vector<T>,
        std::list<T>
    >::type type;
};

// Lazy evaluation example
template<typename T>
struct expensive_computation {
    // This would be expensive to compute
    typedef std::vector<std::vector<T>> type;
};

template<typename T>
struct maybe_compute {
    typedef typename mpl::eval_if<
        boost::is_arithmetic<T>,
        expensive_computation<T>,
        mpl::identity<T>
    >::type type;
};

// Higher-order metafunction example
template<template<typename> class F, typename T>
struct apply_metafunction {
    typedef typename F<T>::type type;
};

template<typename T>
struct add_const_pointer {
    typedef const T* type;
};

void demonstrate_metaprogramming_utilities() {
    std::cout << "=== Metaprogramming Utilities ===\n";
    
    // Smart pointer selection
    class Base { virtual ~Base() = default; };
    class Derived : public Base {};
    
    std::cout << "Base is polymorphic: " 
              << std::boolalpha << boost::is_polymorphic<Base>::value << "\n";
    std::cout << "int is polymorphic: " 
              << std::boolalpha << boost::is_polymorphic<int>::value << "\n";
    
    // Container selection
    typedef container_selector<int, true>::type fast_access_container;
    typedef container_selector<int, false>::type sequential_container;
    
    std::cout << "Fast access container is vector: " 
              << std::is_same<fast_access_container, std::vector<int>>::value << "\n";
    std::cout << "Sequential container is list: " 
              << std::is_same<sequential_container, std::list<int>>::value << "\n";
    
    // Lazy evaluation
    typedef maybe_compute<int>::type computed_for_int;
    typedef maybe_compute<std::string>::type computed_for_string;
    
    std::cout << "Computed type for int is complex: " 
              << !std::is_same<computed_for_int, int>::value << "\n";
    std::cout << "Computed type for string is string: " 
              << std::is_same<computed_for_string, std::string>::value << "\n";
    
    // Higher-order metafunctions
    typedef apply_metafunction<add_const_pointer, int>::type const_int_ptr;
    std::cout << "Applied metafunction result is const int*: " 
              << std::is_same<const_int_ptr, const int*>::value << "\n";
}
```

## Practical Exercises

### Exercise 1: Type-Safe Configuration System ⭐⭐
**Objective**: Build a compile-time validated configuration system using type traits.

```cpp
// TODO: Complete this configuration system
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

template<typename T>
class config_value {
private:
    T value_;
    
public:
    // TODO: Only allow construction for supported types
    template<typename U>
    config_value(U&& val, 
        typename boost::enable_if</* your condition here */>::type* = 0)
        : value_(std::forward<U>(val)) {}
    
    // TODO: Provide different getter behavior based on type
    T get() const {
        // Hint: Use type traits to customize behavior
        // - For arithmetic types: add bounds checking
        // - For strings: add validation
        // - For containers: add size limits
    }
};

// Test cases
config_value<int> port(8080);
config_value<std::string> hostname("localhost");
config_value<bool> debug_mode(true);
// config_value<void*> invalid(nullptr); // Should not compile
```

### Exercise 2: Compile-time Expression Evaluator ⭐⭐⭐
**Objective**: Build a mathematical expression evaluator using MPL.

```cpp
// TODO: Implement compile-time arithmetic expressions
#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>

// Represent: 2 + 3 * 4 - 1 = 13
// As: vector<int_<2>, plus, int_<3>, multiply, int_<4>, minus, int_<1>>

template<typename Expression>
struct evaluate {
    // TODO: Implement expression evaluation
    // Hint: Use fold with custom accumulator that handles operators
    static constexpr int value = /* your implementation */;
};

// Test
typedef boost::mpl::vector<
    boost::mpl::int_<2>, plus_op,
    boost::mpl::int_<3>, multiply_op,
    boost::mpl::int_<4>, minus_op,
    boost::mpl::int_<1>
> expression;

static_assert(evaluate<expression>::value == 13);
```

### Exercise 3: Generic Serialization Framework ⭐⭐⭐⭐
**Objective**: Create automatic serialization using Fusion and type traits.

```cpp
// TODO: Build a complete serialization framework
#include <boost/fusion/adapted/struct.hpp>
#include <boost/type_traits.hpp>

// Support multiple formats
enum class format { JSON, XML, BINARY };

template<format F>
struct serializer {
    template<typename T>
    static std::string serialize(const T& obj) {
        // TODO: Implement format-specific serialization
        // Use type traits to handle different types appropriately
        // Use Fusion to iterate over struct members
    }
};

// Example struct
struct Product {
    std::string name;
    double price;
    int quantity;
    std::vector<std::string> tags;
};

// Make it serializable
BOOST_FUSION_ADAPT_STRUCT(Product,
    (std::string, name)
    (double, price)
    (int, quantity)
    (std::vector<std::string>, tags)
)

// Usage should work like this:
Product product{"Laptop", 999.99, 5, {"electronics", "computers"}};
std::string json = serializer<format::JSON>::serialize(product);
std::string xml = serializer<format::XML>::serialize(product);
```

### Exercise 4: Template-based State Machine ⭐⭐⭐⭐⭐
**Objective**: Design a compile-time validated state machine using MPL.

```cpp
// TODO: Complete this state machine framework
#include <boost/mpl/map.hpp>
#include <boost/mpl/vector.hpp>

template<typename States, typename Events, typename TransitionTable>
class state_machine {
    // TODO: Implement state machine with compile-time validation
    // Features to implement:
    // 1. Current state tracking
    // 2. Event processing with validation
    // 3. State entry/exit actions
    // 4. Compile-time transition table validation
    // 5. Runtime state queries
    
public:
    template<typename Event>
    bool process_event(const Event& event) {
        // TODO: Process event and transition state
        // Should only compile for valid transitions
    }
    
    template<typename State>
    bool is_current_state() const {
        // TODO: Check if current state matches State
    }
};

// Example usage:
struct traffic_light_states {
    struct Red {};
    struct Yellow {};
    struct Green {};
};

struct traffic_light_events {
    struct Timer {};
    struct Emergency {};
};

// TODO: Define transition table
// Red + Timer -> Green
// Green + Timer -> Yellow  
// Yellow + Timer -> Red
// Any + Emergency -> Red
```

### Exercise 5: Optimizing Compiler Plugin ⭐⭐⭐⭐⭐
**Objective**: Create compile-time optimizations using advanced metaprogramming.

```cpp
// TODO: Build a system that optimizes operations at compile time
template<typename Operations>
struct optimizer {
    // TODO: Implement optimizations such as:
    // 1. Constant folding: add<2, 3> -> 5
    // 2. Dead code elimination: if<true, A, B> -> A
    // 3. Algebraic simplification: multiply<X, 1> -> X
    // 4. Loop unrolling for small fixed iterations
    
    typedef /* optimized result */ type;
};

// Example optimizations to implement:
// add<int_<2>, int_<3>> -> int_<5>
// multiply<X, int_<1>> -> X
// if_<true_, A, B> -> A
// for_each<range<0, 3>, F> -> F(0); F(1); F(2);
```

## Learning Checkpoints

### Checkpoint 1: Type Traits Mastery 📚
Before proceeding, ensure you can:

□ **Explain** the difference between type traits and regular functions  
□ **Use** SFINAE to enable/disable template instantiations  
□ **Create** custom type traits for domain-specific needs  
□ **Apply** type traits for automatic optimization decisions  
□ **Debug** template metaprogramming compilation errors  

**Validation Exercise**: 
```cpp
// Create a type trait that detects if a type has a specific member function
template<typename T>
struct has_serialize_method {
    // TODO: Implement detection for T::serialize() method
    static constexpr bool value = /* your implementation */;
};
```

### Checkpoint 2: MPL Proficiency 🏗️
Before proceeding, ensure you can:

□ **Create** and manipulate compile-time sequences  
□ **Implement** metafunctions and higher-order metaprogramming  
□ **Use** MPL algorithms (fold, transform, find_if, etc.)  
□ **Design** compile-time data structures  
□ **Optimize** compile-time computations for performance  

**Validation Exercise**:
```cpp
// Implement a compile-time sorting algorithm for type sequences
template<typename Sequence, template<typename, typename> class Comparator>
struct sort {
    // TODO: Sort types in sequence based on Comparator
    typedef /* sorted sequence */ type;
};
```

### Checkpoint 3: Fusion Integration 🌉
Before proceeding, ensure you can:

□ **Bridge** compile-time and runtime programming effectively  
□ **Use** heterogeneous containers for real applications  
□ **Adapt** existing structs for Fusion compatibility  
□ **Implement** generic algorithms over heterogeneous data  
□ **Optimize** runtime performance with compile-time information  

**Validation Exercise**:
```cpp
// Create a generic visitor system for any Fusion-adapted struct
template<typename Visitor>
struct apply_visitor {
    template<typename Struct>
    static auto visit(Struct& s, Visitor&& visitor) {
        // TODO: Apply visitor to each member of the struct
        // Return aggregated results
    }
};
```

## Real-World Applications

### 1. High-Performance Computing Library 🚀
```cpp
// Matrix operations optimized at compile time
template<typename T, size_t Rows, size_t Cols>
class matrix {
    // Use type traits to choose optimal storage
    using storage_type = typename boost::conditional<
        sizeof(T) * Rows * Cols <= 64,  // Small matrices
        std::array<T, Rows * Cols>,     // Stack storage
        std::vector<T>                  // Heap storage
    >::type;
    
    // Compile-time loop unrolling for small matrices
    template<size_t N>
    typename boost::enable_if_c<(N <= 4), void>::type
    multiply_small(const matrix& other, matrix& result) {
        // Unrolled loops for maximum performance
    }
    
    template<size_t N>
    typename boost::disable_if_c<(N <= 4), void>::type
    multiply_large(const matrix& other, matrix& result) {
        // General algorithm with SIMD optimization
    }
};
```

### 2. Type-Safe Database ORM 💾
```cpp
// Compile-time SQL generation with type safety
template<typename Table>
class query_builder {
    // Use MPL to validate column names at compile time
    template<typename Column>
    static constexpr bool is_valid_column() {
        return boost::mpl::contains<
            typename Table::columns,
            Column
        >::value;
    }
    
public:
    template<typename Column>
    auto select() -> typename boost::enable_if<
        is_valid_column<Column>(),
        query_builder&
    >::type {
        // Add column to SELECT clause
        return *this;
    }
};
```

### 3. Game Engine Component System 🎮
```cpp
// Entity-Component-System with compile-time validation
template<typename... Components>
class entity {
    boost::fusion::vector<Components...> components_;
    
public:
    template<typename Component>
    auto get_component() -> typename boost::enable_if<
        boost::mpl::contains<
            boost::mpl::vector<Components...>,
            Component
        >::value,
        Component&
    >::type {
        return boost::fusion::at_key<Component>(components_);
    }
    
    template<typename System>
    void update(System& system) {
        // Only call system if entity has required components
        if constexpr (System::template can_process<entity>::value) {
            system.process(*this);
        }
    }
};
```

## Performance Considerations

### Compilation Time Impact 🕐

**Understanding the Trade-offs:**
```cpp
// Fast compilation - simple templates
template<typename T>
void simple_function(T value) {
    std::cout << value << std::endl;
}

// Slower compilation - heavy metaprogramming
template<typename T>
auto complex_function(T value) -> typename boost::enable_if<
    boost::mpl::and_<
        boost::is_arithmetic<T>,
        boost::mpl::not_<boost::is_same<T, bool>>,
        boost::mpl::or_<
            boost::is_integral<T>,
            boost::is_floating_point<T>
        >
    >::value,
    decltype(value * 2)
>::type {
    return value * 2;
}
```

**Compilation Time Optimization Strategies:**
```cpp
// 1. Template instantiation depth control
// Avoid deeply recursive metafunctions
template<int N>
struct factorial {
    static constexpr int value = N * factorial<N-1>::value; // Deep recursion
};

// Better: Use iteration or library functions
template<int N>
struct factorial_optimized {
    static constexpr int value = boost::mpl::fold<
        boost::mpl::range_c<int, 1, N+1>,
        boost::mpl::int_<1>,
        boost::mpl::times<boost::mpl::_1, boost::mpl::_2>
    >::type::value;
};

// 2. Lazy evaluation to avoid unnecessary instantiations
template<typename T>
struct expensive_computation {
    // Only computed if actually used
    typedef typename boost::mpl::eval_if<
        boost::is_arithmetic<T>,
        boost::mpl::identity<std::vector<T>>,
        boost::mpl::identity<T>
    >::type type;
};

// 3. Precompiled template libraries
// Use extern template declarations to reduce compilation time
extern template class boost::fusion::vector<int, double, std::string>;
```

**Measuring Compilation Performance:**
```bash
# Measure compilation time
time g++ -std=c++17 -I/path/to/boost program.cpp

# Profile template instantiations
g++ -std=c++17 -ftime-report -ftemplate-backtrace-limit=0 program.cpp

# Analyze template instantiation depth
g++ -std=c++17 -ftemplate-depth=500 program.cpp  # Increase if needed
```

### Runtime Performance Optimization 🚀

**Zero Runtime Cost Principle:**
```cpp
// Compile-time computations have zero runtime cost
constexpr int fibonacci(int n) {
    return (n <= 1) ? n : fibonacci(n-1) + fibonacci(n-2);
}

// MPL equivalent - computed at compile time
template<int N>
struct fibonacci_mpl {
    static constexpr int value = fibonacci_mpl<N-1>::value + fibonacci_mpl<N-2>::value;
};

template<> struct fibonacci_mpl<0> { static constexpr int value = 0; };
template<> struct fibonacci_mpl<1> { static constexpr int value = 1; };

// Usage - no runtime computation
constexpr int result = fibonacci_mpl<10>::value;  // Computed at compile time
```

**Template Specialization for Optimization:**
```cpp
// Generic algorithm
template<typename Container>
void sort_container(Container& container) {
    std::sort(container.begin(), container.end());
}

// Optimized specialization for specific types
template<>
void sort_container<std::vector<int>>(std::vector<int>& container) {
    // Use radix sort for integers
    radix_sort(container.begin(), container.end());
}

// Conditional optimization based on traits
template<typename Container>
void smart_sort(Container& container) {
    if constexpr (boost::is_same<Container, std::vector<int>>::value) {
        radix_sort(container);  // Optimal for integers
    } else if constexpr (boost::is_arithmetic<typename Container::value_type>::value) {
        std::sort(container.begin(), container.end());  // Good for other arithmetic types
    } else {
        stable_sort(container.begin(), container.end());  // Safe for complex types
    }
}
```

**Memory Usage Optimization:**
```cpp
// Minimize template instantiations
// Bad: Each call creates new template instantiation
template<typename T>
void process_data(const T& data) {
    // Complex processing logic
}

// Good: Use type erasure to reduce instantiations
class data_processor {
public:
    template<typename T>
    void process(const T& data) {
        process_impl(data, typename processor_traits<T>::category{});
    }
    
private:
    struct arithmetic_tag {};
    struct string_tag {};
    struct container_tag {};
    
    template<typename T>
    struct processor_traits {
        using category = typename boost::conditional<
            boost::is_arithmetic<T>::value, arithmetic_tag,
            typename boost::conditional<
                boost::is_same<T, std::string>::value, string_tag,
                container_tag
            >::type
        >::type;
    };
    
    template<typename T>
    void process_impl(const T& data, arithmetic_tag) {
        // Shared implementation for all arithmetic types
    }
    
    template<typename T>
    void process_impl(const T& data, string_tag) {
        // Shared implementation for string types
    }
    
    template<typename T>
    void process_impl(const T& data, container_tag) {
        // Shared implementation for container types
    }
};
```

### Debug Symbol Size Management 🔍

**Problem**: Heavy metaprogramming can create enormous debug symbols.

**Solutions:**
```cpp
// 1. Use type aliases to simplify debug info
template<typename T>
using optimized_vector = typename boost::conditional<
    sizeof(T) <= 4,
    std::vector<T, stack_allocator<T>>,
    std::vector<T>
>::type;

// Instead of exposing the full conditional type
typedef optimized_vector<int> int_vector;  // Cleaner debug info

// 2. Separate template heavy code into different translation units
// heavy_templates.hpp
template<typename T>
struct complex_metafunction { /* ... */ };

// heavy_templates.cpp
template struct complex_metafunction<int>;
template struct complex_metafunction<double>;
// Explicit instantiation reduces debug symbol duplication

// 3. Use template aliases for commonly used combinations
template<typename T>
using json_serializer = boost::fusion::transform<
    T,
    json_converter<boost::mpl::_1>
>;

// Usage
typedef json_serializer<Person> person_serializer;  // Cleaner than full type
```

### Executable Size Considerations 📦

**Code Bloat Prevention:**
```cpp
// Problem: Template instantiation explosion
template<typename T, size_t N>
class fixed_array {
    T data[N];
    // ... methods ...
};

// Each (T, N) combination creates a new class
fixed_array<int, 10> arr1;
fixed_array<int, 20> arr2;  // Different class!
fixed_array<double, 10> arr3;  // Different class!

// Solution: Template base class with shared implementation
class fixed_array_base {
protected:
    void* data_;
    size_t size_;
    size_t element_size_;
    
    void copy_elements(const void* src, void* dst, size_t count, size_t element_size);
    // ... shared implementations ...
};

template<typename T, size_t N>
class fixed_array : private fixed_array_base {
    T data_[N];
    
public:
    fixed_array() : fixed_array_base(&data_[0], N, sizeof(T)) {}
    
    T& operator[](size_t index) {
        // Type-safe wrapper around base implementation
        return static_cast<T*>(data_)[index];
    }
};
```

**Link Time Optimization:**
```bash
# Enable LTO to eliminate unused template instantiations
g++ -std=c++17 -flto -O2 program.cpp

# Use gold linker for better template handling
g++ -std=c++17 -fuse-ld=gold program.cpp

# Generate size report
g++ -std=c++17 -Wl,--print-map program.cpp > size_report.txt
```

## Best Practices

### 1. Metaprogramming Design Principles 🎯

#### Keep Metafunctions Simple and Focused
```cpp
// Bad: Monolithic metafunction doing too much
template<typename T>
struct complex_type_analyzer {
    static constexpr bool is_numeric = boost::is_arithmetic<T>::value;
    static constexpr bool is_large = sizeof(T) > 8;
    static constexpr bool is_copyable = boost::is_copy_constructible<T>::value;
    static constexpr bool is_optimal = is_numeric && !is_large && is_copyable;
    
    typedef typename boost::conditional<
        is_optimal,
        T,
        const T&
    >::type parameter_type;
    
    typedef typename boost::conditional<
        is_numeric,
        std::vector<T>,
        std::list<T>
    >::type container_type;
    
    // ... more complex logic
};

// Good: Separate, focused metafunctions
template<typename T>
struct is_small_trivial {
    static constexpr bool value = 
        sizeof(T) <= sizeof(void*) && boost::is_trivially_copyable<T>::value;
};

template<typename T>
struct optimal_parameter_type {
    typedef typename boost::conditional<
        is_small_trivial<T>::value,
        T,
        const T&
    >::type type;
};

template<typename T>
struct optimal_container_type {
    typedef typename boost::conditional<
        boost::is_arithmetic<T>::value,
        std::vector<T>,
        std::list<T>
    >::type type;
};
```

#### Use Descriptive Names for Metafunction Parameters
```cpp
// Bad: Cryptic parameter names
template<typename _Tp, bool _Cond, typename _U>
struct mysterious_selector {
    typedef typename boost::conditional<_Cond, _Tp, _U>::type type;
};

// Good: Clear, descriptive names
template<typename ValueType, bool UseOptimized, typename FallbackType>
struct storage_selector {
    typedef typename boost::conditional<
        UseOptimized,
        ValueType,
        FallbackType
    >::type type;
};

// Even better: Document the metafunction's purpose
/**
 * Selects optimal storage type based on usage pattern.
 * @tparam ValueType The type to be stored
 * @tparam UseOptimized Whether to use optimized storage
 * @tparam FallbackType Alternative storage type
 */
template<typename ValueType, bool UseOptimized, typename FallbackType>
struct storage_selector {
    static_assert(std::is_same_v<ValueType, FallbackType> || UseOptimized,
                  "FallbackType must match ValueType when not optimized");
    
    typedef typename boost::conditional<
        UseOptimized && sizeof(ValueType) <= 64,
        ValueType,
        FallbackType
    >::type type;
};
```

#### Document Complex Metaprogramming Logic
```cpp
/**
 * Automatic function parameter optimization metafunction.
 * 
 * Decision tree:
 * 1. For fundamental types <= pointer size: pass by value
 * 2. For trivially copyable types <= 64 bytes: pass by value  
 * 3. For move-only types: pass by universal reference
 * 4. Otherwise: pass by const reference
 * 
 * @tparam T The parameter type to optimize
 */
template<typename T>
struct function_parameter_traits {
private:
    static constexpr bool is_small_fundamental = 
        boost::is_fundamental<T>::value && sizeof(T) <= sizeof(void*);
    
    static constexpr bool is_small_trivial =
        boost::is_trivially_copyable<T>::value && sizeof(T) <= 64;
    
    static constexpr bool is_move_only =
        std::is_move_constructible_v<T> && !std::is_copy_constructible_v<T>;
    
public:
    typedef typename boost::conditional<
        is_small_fundamental || is_small_trivial,
        T,                              // Pass by value
        typename boost::conditional<
            is_move_only,
            T&&,                        // Universal reference
            const T&                    // Const reference
        >::type
    >::type optimal_type;
    
    static constexpr bool pass_by_value = is_small_fundamental || is_small_trivial;
    static constexpr bool pass_by_reference = !pass_by_value && !is_move_only;
    static constexpr bool pass_by_move = is_move_only;
};
```

#### Test Metafunctions Thoroughly
```cpp
// Comprehensive metafunction testing
namespace test {
    // Test basic types
    static_assert(function_parameter_traits<int>::pass_by_value);
    static_assert(function_parameter_traits<double>::pass_by_value);
    static_assert(!function_parameter_traits<std::string>::pass_by_value);
    
    // Test large trivial types
    struct large_trivial { char data[100]; };
    static_assert(!function_parameter_traits<large_trivial>::pass_by_value);
    
    // Test move-only types
    static_assert(function_parameter_traits<std::unique_ptr<int>>::pass_by_move);
    
    // Test complex types
    static_assert(function_parameter_traits<std::vector<int>>::pass_by_reference);
    
    // Test const correctness
    static_assert(std::is_same_v<
        function_parameter_traits<std::string>::optimal_type,
        const std::string&
    >);
}

// Integration tests
template<typename T>
void test_function(typename function_parameter_traits<T>::optimal_type param) {
    // Function automatically uses optimal parameter passing
    std::cout << "Received parameter\n";
}

void run_integration_tests() {
    test_function<int>(42);                          // Pass by value
    test_function<std::string>(std::string("hi"));   // Pass by const&
    test_function<std::unique_ptr<int>>(std::make_unique<int>(42)); // Pass by &&
}
```

### 2. Type Trait Usage Guidelines 🛠️

#### Prefer Standard Library Traits When Available
```cpp
// Bad: Using Boost traits when standard equivalents exist
boost::is_integral<T>::value       // C++11+
boost::is_same<T, U>::value        // C++11+
boost::remove_const<T>::type       // C++11+

// Good: Use standard library (shorter, more portable)
std::is_integral_v<T>              // C++17
std::is_same_v<T, U>               // C++17  
std::remove_const_t<T>             // C++14

// Use Boost traits only when standard library lacks them
boost::is_detected<T, Op>::value   // Not in standard library
boost::function_traits<F>::arity   // Not in standard library
```

#### Create Custom Traits for Domain-Specific Needs
```cpp
// Domain-specific trait: Is this type serializable?
template<typename T, typename = void>
struct is_serializable : std::false_type {};

template<typename T>
struct is_serializable<T, std::void_t<
    decltype(std::declval<T>().serialize(std::declval<std::ostream&>()))
>> : std::true_type {};

// Usage
template<typename T>
std::enable_if_t<is_serializable_v<T>, void>
save_to_file(const T& obj, const std::string& filename) {
    std::ofstream file(filename);
    obj.serialize(file);
}

// Domain-specific trait: Container detection
template<typename T, typename = void>
struct is_container : std::false_type {};

template<typename T>
struct is_container<T, std::void_t<
    typename T::value_type,
    typename T::iterator,
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end()),
    decltype(std::declval<T>().size())
>> : std::true_type {};
```

#### Use SFINAE Judiciously to Avoid Overcomplication
```cpp
// Bad: Overly complex SFINAE condition
template<typename T>
std::enable_if_t<
    std::is_arithmetic_v<T> && 
    !std::is_same_v<T, bool> && 
    !std::is_same_v<T, char> &&
    sizeof(T) >= 4 &&
    std::is_signed_v<T>,
    T
>
complex_numeric_function(T value) {
    return value * 2;
}

// Good: Extract logic into a trait
template<typename T>
struct is_suitable_numeric {
    static constexpr bool value = 
        std::is_arithmetic_v<T> && 
        !std::is_same_v<T, bool> && 
        !std::is_same_v<T, char> &&
        sizeof(T) >= 4 &&
        std::is_signed_v<T>;
};

template<typename T>
std::enable_if_t<is_suitable_numeric<T>::value, T>
simple_numeric_function(T value) {
    return value * 2;
}

// Even better: Use concepts in C++20
template<typename T>
concept SuitableNumeric = 
    std::is_arithmetic_v<T> && 
    !std::is_same_v<T, bool> && 
    !std::is_same_v<T, char> &&
    sizeof(T) >= 4 &&
    std::is_signed_v<T>;

template<SuitableNumeric T>
T modern_numeric_function(T value) {
    return value * 2;
}
```

### 3. Fusion Applications Best Practices 🌉

#### Use for Bridging Compile-time and Runtime Worlds
```cpp
// Excellent use case: Reflection-like operations
template<typename Struct>
void print_all_members(const Struct& obj) {
    std::cout << "Object contents:\n";
    boost::fusion::for_each(obj, [](const auto& member) {
        std::cout << "  " << member << "\n";
    });
}

// Another good use case: Generic validation
template<typename Struct>
bool validate_all_members(const Struct& obj) {
    bool all_valid = true;
    boost::fusion::for_each(obj, [&](const auto& member) {
        if constexpr (has_validate_method_v<decltype(member)>) {
            if (!member.validate()) {
                all_valid = false;
            }
        }
    });
    return all_valid;
}
```

#### Prefer When Type Safety is Crucial
```cpp
// Type-safe configuration using Fusion map
struct config_keys {
    struct database_url {};
    struct port {};
    struct debug_mode {};
    struct timeout {};
};

using config_map = boost::fusion::map<
    boost::fusion::pair<config_keys::database_url, std::string>,
    boost::fusion::pair<config_keys::port, int>,
    boost::fusion::pair<config_keys::debug_mode, bool>,
    boost::fusion::pair<config_keys::timeout, std::chrono::seconds>
>;

class application_config {
    config_map config_;
    
public:
    template<typename Key>
    auto get() const -> const decltype(boost::fusion::at_key<Key>(config_))& {
        return boost::fusion::at_key<Key>(config_);
    }
    
    template<typename Key, typename Value>
    void set(Value&& value) {
        boost::fusion::at_key<Key>(config_) = std::forward<Value>(value);
    }
};

// Usage - compile-time type safety
application_config config;
config.set<config_keys::port>(8080);                    // OK
config.set<config_keys::debug_mode>(true);              // OK
// config.set<config_keys::port>("invalid");            // Compile error!
```

#### Consider Performance Implications of Heterogeneous Containers
```cpp
// Performance consideration: Fusion vector vs tuple
struct performance_test {
    // Fusion vector - good for algorithms, reflection
    boost::fusion::vector<int, double, std::string> fusion_data;
    
    // std::tuple - better performance for simple access
    std::tuple<int, double, std::string> tuple_data;
    
    void benchmark_access() {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Fusion access
        for (int i = 0; i < 1000000; ++i) {
            auto val = boost::fusion::at_c<0>(fusion_data);
            (void)val;
        }
        
        auto fusion_time = std::chrono::high_resolution_clock::now() - start;
        
        start = std::chrono::high_resolution_clock::now();
        
        // Tuple access  
        for (int i = 0; i < 1000000; ++i) {
            auto val = std::get<0>(tuple_data);
            (void)val;
        }
        
        auto tuple_time = std::chrono::high_resolution_clock::now() - start;
        
        std::cout << "Fusion: " << fusion_time.count() << "ns\n";
        std::cout << "Tuple: " << tuple_time.count() << "ns\n";
    }
};

// Guideline: Use Fusion when you need algorithms/reflection
// Use std::tuple for simple heterogeneous storage
```

### 4. Error Handling and Debugging 🐛

#### Provide Clear Error Messages
```cpp
// Bad: Cryptic template error
template<typename T>
void mysterious_function(T value) {
    // Will cause incomprehensible error if T doesn't support operator+
    auto result = value + value;
}

// Good: Clear static assertions
template<typename T>
void clear_function(T value) {
    static_assert(std::is_arithmetic_v<T> || has_plus_operator_v<T>,
                  "Type T must be arithmetic or support operator+");
    
    static_assert(!std::is_pointer_v<T>,
                  "Pointer types are not supported for this operation");
    
    auto result = value + value;
}

// Even better: Concept-based constraints (C++20)
template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

template<Addable T>
void concept_function(T value) {
    auto result = value + value;  // Clear error if T is not Addable
}
```

#### Use Debugging-Friendly Type Names
```cpp
// Bad: Expose complex internal types
template<typename T>
using complex_type = typename boost::mpl::transform<
    typename boost::mpl::if_<
        boost::is_arithmetic<T>,
        boost::mpl::vector<T, T*, const T&>,
        boost::mpl::vector<T>
    >::type,
    boost::add_pointer<boost::mpl::_1>
>::type;

// Good: Provide meaningful aliases
template<typename T>
struct type_variants {
    using arithmetic_variants = boost::mpl::vector<T, T*, const T&>;
    using non_arithmetic_variants = boost::mpl::vector<T>;
    
    using type = typename boost::mpl::transform<
        typename boost::mpl::if_<
            boost::is_arithmetic<T>,
            arithmetic_variants,
            non_arithmetic_variants
        >::type,
        boost::add_pointer<boost::mpl::_1>
    >::type;
};

template<typename T>
using type_variants_t = typename type_variants<T>::type;
```

## Migration to Modern C++

### C++11 and Later Features Evolution 📈

#### From Boost.TypeTraits to std::type_traits
```cpp
// Migration guide with performance comparisons

// C++98/03 + Boost (Legacy)
template<typename T>
typename boost::enable_if<boost::is_integral<T>, T>::type
legacy_function(T value) {
    return value * 2;
}

// C++11 Migration
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
cpp11_function(T value) {
    return value * 2;
}

// C++14 Improvements 
template<typename T>
std::enable_if_t<std::is_integral<T>::value, T>
cpp14_function(T value) {
    return value * 2;
}

// C++17 Modernization
template<typename T>
std::enable_if_t<std::is_integral_v<T>, T>
cpp17_function(T value) {
    return value * 2;
}

// C++20 Revolution
template<typename T>
requires std::integral<T>
T cpp20_function(T value) {
    return value * 2;
}

// Performance comparison
void benchmark_type_traits() {
    // All versions generate identical optimized code
    // but compilation time decreases with newer standards
    static_assert(legacy_function(5) == 10);
    static_assert(cpp11_function(5) == 10);
    static_assert(cpp14_function(5) == 10);
    static_assert(cpp17_function(5) == 10);
    static_assert(cpp20_function(5) == 10);
}
```

#### Template Metaprogramming Evolution
```cpp
// Boost.MPL vs Modern C++ metaprogramming

// Boost.MPL approach
#include <boost/mpl/vector.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/at.hpp>

template<typename T>
struct add_pointer { typedef T* type; };

typedef boost::mpl::vector<int, double, char> types;
typedef boost::mpl::transform<types, add_pointer<boost::mpl::_1>>::type pointer_types;

// C++11 variadic templates
template<typename... Types>
struct type_list {};

template<template<typename> class F, typename... Types>
struct transform {
    using type = type_list<typename F<Types>::type...>;
};

using modern_pointer_types = transform<add_pointer, int, double, char>::type;

// C++14 variable templates + alias templates
template<typename T>
using add_pointer_t = T*;

template<typename... Types>
using add_pointers = type_list<add_pointer_t<Types>...>;

using cpp14_pointer_types = add_pointers<int, double, char>;

// C++17 fold expressions
template<typename... Types>
constexpr size_t total_size() {
    return (sizeof(Types) + ...);
}

static_assert(total_size<int, double, char>() == sizeof(int) + sizeof(double) + sizeof(char));

// C++20 concepts + template lambdas  
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric... Types>
constexpr auto sum_all(Types... values) {
    return (values + ...);
}
```

#### constexpr Functions vs Metaprogramming
```cpp
// Boost.MPL recursive metafunction
template<int N>
struct factorial_mpl {
    static constexpr int value = N * factorial_mpl<N-1>::value;
};
template<> struct factorial_mpl<0> { static constexpr int value = 1; };

// C++11 constexpr function
constexpr int factorial_constexpr(int n) {
    return (n == 0) ? 1 : n * factorial_constexpr(n - 1);
}

// C++14 relaxed constexpr
constexpr int factorial_cpp14(int n) {
    int result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Performance and usability comparison
void compare_approaches() {
    // All computed at compile time
    constexpr int mpl_result = factorial_mpl<5>::value;
    constexpr int cpp11_result = factorial_constexpr(5);
    constexpr int cpp14_result = factorial_cpp14(5);
    
    static_assert(mpl_result == cpp11_result);
    static_assert(cpp11_result == cpp14_result);
    
    // But constexpr functions are more flexible
    std::array<int, factorial_cpp14(5)> compile_time_array;  // OK
    // std::array<int, factorial_mpl<5>::value> mpl_array;  // More verbose
}
```

### Modern Alternatives Comparison 🆚

#### std::variant vs Boost.Variant
```cpp
#include <boost/variant.hpp>
#include <variant>

// Boost.Variant (C++03 era)
boost::variant<int, std::string, double> boost_var = 42;

struct boost_visitor : boost::static_visitor<void> {
    void operator()(int i) const { std::cout << "int: " << i << "\n"; }
    void operator()(const std::string& s) const { std::cout << "string: " << s << "\n"; }
    void operator()(double d) const { std::cout << "double: " << d << "\n"; }
};

boost::apply_visitor(boost_visitor{}, boost_var);

// std::variant (C++17)
std::variant<int, std::string, double> std_var = 42;

std::visit([](const auto& value) {
    std::cout << "value: " << value << "\n";
}, std_var);

// Performance comparison
void benchmark_variants() {
    // std::variant is typically faster due to:
    // 1. Better compiler optimization
    // 2. No virtual function overhead  
    // 3. More efficient storage layout
    
    // Memory usage
    static_assert(sizeof(std_var) <= sizeof(boost_var)); // Usually true
}
```

#### std::tuple vs Fusion Sequences
```cpp
#include <boost/fusion/container/vector.hpp>
#include <tuple>

// Fusion vector
boost::fusion::vector<int, std::string, double> fusion_vec(42, "hello", 3.14);

// Access
auto fusion_first = boost::fusion::at_c<0>(fusion_vec);
boost::fusion::for_each(fusion_vec, [](const auto& x) { std::cout << x << " "; });

// std::tuple
std::tuple<int, std::string, double> std_tup(42, "hello", 3.14);

// Access (C++14+)
auto tuple_first = std::get<0>(std_tup);
std::apply([](const auto&... args) { ((std::cout << args << " "), ...); }, std_tup);

// When to use which:
// - Use std::tuple for simple heterogeneous storage
// - Use Fusion when you need rich metaprogramming algorithms
// - Use Fusion for reflection-like operations
// - std::tuple has better compile times and runtime performance

// Example: Reflection comparison
struct Person { std::string name; int age; double salary; };

// Fusion approach - automatic adaptation
BOOST_FUSION_ADAPT_STRUCT(Person, (std::string, name)(int, age)(double, salary))

template<typename T>
void print_fusion_struct(const T& obj) {
    boost::fusion::for_each(obj, [](const auto& field) {
        std::cout << field << " ";
    });
}

// std::tuple approach - manual conversion
template<typename T>
auto to_tuple(const T& obj) {
    if constexpr (std::is_same_v<T, Person>) {
        return std::make_tuple(obj.name, obj.age, obj.salary);
    }
}

template<typename T>
void print_tuple_struct(const T& obj) {
    auto tup = to_tuple(obj);
    std::apply([](const auto&... fields) {
        ((std::cout << fields << " "), ...);
    }, tup);
}
```

#### Concepts (C++20) vs SFINAE Patterns
```cpp
// SFINAE pattern (Boost era)
template<typename T>
typename std::enable_if_t<
    std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
    T
>
sfinae_function(T value) {
    return value * 2;
}

// Concepts (C++20)
template<typename T>
concept ArithmeticNotBool = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<ArithmeticNotBool T>
T concepts_function(T value) {
    return value * 2;
}

// Benefits of concepts:
// 1. Much clearer error messages
// 2. Better IDE support
// 3. Can be used in more contexts
// 4. Composable and reusable

// Advanced concept composition
template<typename T>
concept Serializable = requires(T obj, std::ostream& os) {
    { obj.serialize(os) } -> std::same_as<void>;
};

template<typename T>
concept PrintableNumeric = ArithmeticNotBool<T> && requires(T value) {
    std::cout << value;
};

template<PrintableNumeric T>
void modern_function(T value) {
    std::cout << "Processing: " << value * 2 << "\n";
}
```

### Migration Strategy 🚀

#### Gradual Migration Approach
```cpp
// Phase 1: Dual compatibility
#if __cplusplus >= 201703L
    #include <type_traits>
    #include <variant>
    #include <tuple>
    namespace traits = std;
#else
    #include <boost/type_traits.hpp>
    #include <boost/variant.hpp>
    #include <boost/fusion/container/vector.hpp>
    namespace traits = boost;
#endif

template<typename T>
using enable_if_integral_t = typename traits::enable_if<
    traits::is_integral<T>::value, T
>::type;

// Phase 2: Feature detection macros
#ifdef __cpp_concepts
    #define REQUIRES(x) requires x
    #define CONCEPT_FUNCTION(concept_name, type_name) \
        template<type_name T> requires concept_name<T>
#else
    #define REQUIRES(x)
    #define CONCEPT_FUNCTION(concept_name, type_name) \
        template<typename T, typename = std::enable_if_t<concept_name<T>>>
#endif

// Phase 3: Wrapper utilities
template<typename T>
constexpr bool is_integral_v = 
#if __cplusplus >= 201703L
    std::is_integral_v<T>;
#else
    boost::is_integral<T>::value;
#endif

// Phase 4: Modern interface with legacy implementation
template<typename... Types>
class modern_variant {
#if __cplusplus >= 201703L
    std::variant<Types...> impl_;
#else
    boost::variant<Types...> impl_;
#endif

public:
    template<typename T>
    modern_variant(T&& value) : impl_(std::forward<T>(value)) {}
    
    template<typename Visitor>
    auto visit(Visitor&& visitor) {
#if __cplusplus >= 201703L
        return std::visit(std::forward<Visitor>(visitor), impl_);
#else
        return boost::apply_visitor(std::forward<Visitor>(visitor), impl_);
#endif
    }
};
```

#### Performance Benchmarking During Migration
```cpp
// Benchmark template to measure migration impact
template<typename Implementation>
class migration_benchmark {
public:
    static void measure_compilation_time() {
        // Use build system timing
        // cmake --build . --target benchmark -- -j1 -v
    }
    
    static void measure_runtime_performance() {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run implementation-specific operations
        Implementation::run_test();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Runtime: " << duration.count() << " μs\n";
    }
    
    static void measure_memory_usage() {
        // Platform-specific memory measurement
        size_t before = get_memory_usage();
        Implementation::allocate_test_data();
        size_t after = get_memory_usage();
        
        std::cout << "Memory used: " << (after - before) << " bytes\n";
    }
};

// Usage
migration_benchmark<boost_implementation>::measure_runtime_performance();
migration_benchmark<modern_implementation>::measure_runtime_performance();
```

## Assessment & Mastery Evaluation 🎓

### Knowledge Check Quiz

**Conceptual Understanding (Score: /40)**

1. **Type Traits Fundamentals (10 points)**
   - Explain the difference between `boost::is_arithmetic<T>::value` and `std::is_arithmetic_v<T>` (2 pts)
   - When would you use `boost::enable_if` vs `std::enable_if_t`? (2 pts)
   - What is SFINAE and why is it important in template metaprogramming? (3 pts)
   - Give an example of when you'd create a custom type trait (3 pts)

2. **MPL Mastery (15 points)**
   - What's the difference between compile-time and runtime algorithms? (3 pts)
   - Explain how `boost::mpl::fold` works with an example (4 pts)
   - When would you use `boost::mpl::vector` vs `std::tuple`? (3 pts)
   - Design a metafunction that counts arithmetic types in a type sequence (5 pts)

3. **Fusion Integration (15 points)**
   - What problem does Fusion solve that standard containers cannot? (3 pts)
   - Explain the difference between `boost::fusion::vector` and `std::vector` (4 pts)
   - How does `BOOST_FUSION_ADAPT_STRUCT` work internally? (4 pts)
   - When would you choose Fusion over `std::tuple`? (4 pts)

### Practical Coding Assessment

**Implementation Tasks (Score: /60)**

#### Task 1: Advanced Type Trait System (20 points)
```cpp
// Implement a comprehensive type analysis system
template<typename T>
class type_analyzer {
public:
    // TODO: Implement these static constexpr members (5 pts each)
    static constexpr bool is_container = /* detect containers */;
    static constexpr bool is_smart_pointer = /* detect smart pointers */;
    static constexpr bool is_callable = /* detect function objects */;
    static constexpr size_t optimal_alignment = /* compute optimal alignment */;
    
    // TODO: Implement optimal storage type selection (5 pts)
    using storage_type = /* choose best storage based on T characteristics */;
};

// Test cases must pass:
static_assert(type_analyzer<std::vector<int>>::is_container);
static_assert(type_analyzer<std::unique_ptr<int>>::is_smart_pointer);
static_assert(type_analyzer<std::function<int()>>::is_callable);
static_assert(type_analyzer<int>::optimal_alignment == alignof(int));
```

#### Task 2: Compile-time Algorithm Implementation (20 points)
```cpp
// Implement a compile-time sorting algorithm using MPL
template<typename Sequence, template<typename, typename> class Comparator>
struct mpl_sort {
    // TODO: Sort the sequence using compile-time comparisons (15 pts)
    using type = /* your implementation */;
};

// Size-based comparator
template<typename T, typename U>
struct size_comparator {
    static constexpr bool value = sizeof(T) < sizeof(U);
};

// Test case must pass:
using unsorted = boost::mpl::vector<double, char, int, long long>;
using sorted = mpl_sort<unsorted, size_comparator>::type;
static_assert(std::is_same_v<
    sorted,
    boost::mpl::vector<char, int, double, long long>
>);

// TODO: Implement compile-time unique operation (5 pts)
template<typename Sequence>
struct mpl_unique {
    using type = /* remove duplicate types */;
};
```

#### Task 3: Generic Serialization Framework (20 points)
```cpp
// Build a complete serialization system using Fusion
template<typename T>
class universal_serializer {
public:
    // TODO: Implement format-agnostic serialization (10 pts)
    template<typename Format>
    static std::string serialize(const T& obj) {
        // Must work with any Fusion-adapted struct
        // Support JSON, XML, Binary formats
    }
    
    // TODO: Implement type-safe deserialization (10 pts)
    template<typename Format>
    static T deserialize(const std::string& data) {
        // Parse format and reconstruct object
    }
};

// Test with complex nested structure
struct Address {
    std::string street;
    std::string city;
    int zip_code;
};

struct Person {
    std::string name;
    int age;
    Address address;
    std::vector<std::string> hobbies;
};

BOOST_FUSION_ADAPT_STRUCT(Address,
    (std::string, street)(std::string, city)(int, zip_code))
BOOST_FUSION_ADAPT_STRUCT(Person,
    (std::string, name)(int, age)(Address, address)(std::vector<std::string>, hobbies))

// Must work:
Person original{/* ... */};
auto json = universal_serializer<Person>::serialize<JSON>(original);
auto restored = universal_serializer<Person>::deserialize<JSON>(json);
assert(original == restored);
```

### Performance & Design Assessment

**Architecture Evaluation (Score: /20)**

#### Task 4: Metaprogramming Performance Analysis (10 points)
```cpp
// TODO: Analyze and optimize this metaprogramming code
template<int N>
struct slow_fibonacci {
    static constexpr int value = slow_fibonacci<N-1>::value + slow_fibonacci<N-2>::value;
};
template<> struct slow_fibonacci<0> { static constexpr int value = 0; };
template<> struct slow_fibonacci<1> { static constexpr int value = 1; };

// Problems to identify and fix:
// 1. Exponential compilation time growth (3 pts)
// 2. Template instantiation depth issues (2 pts)  
// 3. Memory usage during compilation (2 pts)
// 4. Propose and implement optimized version (3 pts)
```

#### Task 5: Design Pattern Implementation (10 points)
```cpp
// TODO: Implement a type-safe visitor pattern using metaprogramming
template<typename VariantType, typename... Visitors>
class metaprogramming_visitor {
    // Requirements:
    // 1. Compile-time visitor validation (3 pts)
    // 2. Automatic return type deduction (2 pts)
    // 3. Overload resolution for best match (3 pts) 
    // 4. Error messages for missing visitors (2 pts)
};

// Usage should work like this:
using number_variant = std::variant<int, double, std::string>;
auto visitor = metaprogramming_visitor<number_variant>{
    [](int i) { return i * 2; },
    [](double d) { return d * 3.14; },
    [](const std::string& s) { return s + "_processed"; }
};
```

### Mastery Levels

**🥉 Bronze Level (60-69 points)**
- Basic understanding of type traits and SFINAE
- Can use existing MPL algorithms
- Understands Fusion containers fundamentally

**🥈 Silver Level (70-84 points)**
- Creates custom type traits for specific needs
- Implements simple metafunctions and algorithms
- Effectively bridges compile-time and runtime with Fusion
- Recognizes performance implications

**🥇 Gold Level (85-94 points)**
- Designs complex metaprogramming systems
- Optimizes for both compile-time and runtime performance
- Creates reusable metaprogramming libraries
- Debugs template metaprogramming issues effectively

**💎 Diamond Level (95-100 points)**
- Masters advanced metaprogramming patterns
- Contributes to metaprogramming library design
- Teaches metaprogramming concepts to others
- Innovates new approaches to generic programming

### Self-Assessment Checklist

**Before Claiming Mastery, Can You:**

□ **Design** a type trait system for a specific domain problem?  
□ **Implement** compile-time algorithms that are more efficient than runtime equivalents?  
□ **Bridge** seamlessly between compile-time and runtime programming?  
□ **Debug** complex template metaprogramming compilation errors?  
□ **Optimize** metaprogramming code for compilation performance?  
□ **Explain** when metaprogramming adds value vs complexity?  
□ **Migrate** legacy Boost metaprogramming to modern C++ standards?  
□ **Teach** these concepts to junior developers?  

### Final Capstone Project

**Build a Complete Generic Programming Framework (Optional - 100 points)**

Design and implement a comprehensive framework that demonstrates mastery of all three libraries:

1. **Type System** (25 pts): Advanced type traits for automatic optimization
2. **Compile-time Algorithms** (25 pts): MPL-based algorithms for complex transformations  
3. **Runtime Integration** (25 pts): Fusion-based reflection and serialization
4. **Performance** (15 pts): Benchmarked optimizations and minimal overhead
5. **Documentation** (10 pts): Clear examples and migration guides

**Example Framework Ideas:**
- Advanced ORM with compile-time SQL generation
- High-performance mathematical computing library
- Generic game engine component system
- Automatic API binding generator
- Type-safe configuration management system

### Certification Path

**To be considered expert-level in Generic Programming Utilities:**

1. ✅ Score 85+ on knowledge and practical assessments
2. ✅ Complete capstone project with 80+ score  
3. ✅ Demonstrate teaching capability (explain concepts to others)
4. ✅ Contribute to open-source metaprogramming project
5. ✅ Write technical blog post or documentation on advanced topic

**Recommended Timeline:**
- Weeks 1-2: Master concepts and basic implementations
- Week 3: Complete practical assessments
- Week 4: Build capstone project
- Week 5: Teaching and contribution activities

## Next Steps

Move on to [Advanced Boost Libraries](10_Advanced_Boost_Libraries.md) to explore specialized Boost libraries for graphs, geometry, and interprocess communication.
