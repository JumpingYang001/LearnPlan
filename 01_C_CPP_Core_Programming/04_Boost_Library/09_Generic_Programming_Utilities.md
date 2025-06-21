# Generic Programming Utilities

*Duration: 1 week*

## Overview

This section covers Boost's metaprogramming and generic programming utilities, including type traits, compile-time algorithms, and heterogeneous containers.

## Learning Topics

### Boost.TypeTraits
- Advanced type traits beyond std::type_traits
- Type information and manipulation at compile time
- SFINAE and template metaprogramming support
- Integration with template specialization

### Boost.MPL
- Metaprogramming library for compile-time algorithms
- Compile-time data structures (sequences, maps, sets)
- Metafunctions and higher-order metaprogramming
- Algorithm implementation at compile time

### Boost.Fusion
- Heterogeneous containers and algorithms
- Compile-time and runtime fusion of data
- Integration between compile-time and runtime worlds
- Sequence manipulation and transformation

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

1. **Type-Safe Variant Implementation**
   - Create a type-safe union using MPL and type traits
   - Implement visitor pattern for type-safe access
   - Add compile-time type checking and conversions

2. **Compile-time Expression Parser**
   - Build a mathematical expression evaluator at compile time
   - Use MPL sequences to represent expressions
   - Implement optimization passes as metafunctions

3. **Generic Serialization Framework**
   - Create automatic serialization using Fusion and type traits
   - Support different output formats (binary, JSON, XML)
   - Handle nested structures and collections

4. **Template-based State Machine**
   - Design a compile-time state machine using MPL
   - Implement state transitions and event handling
   - Generate runtime dispatch tables at compile time

## Performance Considerations

### Compilation Time
- Template instantiation depth and complexity
- Recursive metafunction limitations
- Code bloat from excessive template specialization

### Runtime Performance
- Zero runtime cost for compile-time computations
- Template specialization for optimization
- Avoiding unnecessary type conversions

### Memory Usage
- Template instantiation memory requirements
- Debug symbol size with heavy metaprogramming
- Executable size considerations

## Best Practices

1. **Metaprogramming Design**
   - Keep metafunctions simple and focused
   - Use descriptive names for metafunction parameters
   - Document complex metaprogramming logic
   - Test metafunctions thoroughly

2. **Type Trait Usage**
   - Prefer standard library traits when available
   - Create custom traits for domain-specific needs
   - Use SFINAE judiciously to avoid overcomplication

3. **Fusion Applications**
   - Use for bridging compile-time and runtime worlds
   - Prefer when type safety is crucial
   - Consider performance implications of heterogeneous containers

## Migration to Modern C++

### C++11 and Later Features
```cpp
// Boost.TypeTraits -> std::type_traits
boost::is_integral<T>::value -> std::is_integral_v<T>
boost::enable_if<C, T>::type -> std::enable_if_t<C, T>

// Template metaprogramming improvements
constexpr functions, variable templates, if constexpr
```

### Modern Alternatives
- `std::variant` vs Boost.Variant
- `std::tuple` vs Fusion sequences
- Concepts (C++20) vs SFINAE patterns

## Assessment

- Can design compile-time algorithms using MPL
- Understands type trait applications and SFINAE
- Can implement heterogeneous containers with Fusion
- Knows when metaprogramming provides value vs complexity

## Next Steps

Move on to [Advanced Boost Libraries](10_Advanced_Boost_Libraries.md) to explore specialized Boost libraries for graphs, geometry, and interprocess communication.
