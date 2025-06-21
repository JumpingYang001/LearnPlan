# Template Metaprogramming Library Project

## Project Overview

Build an advanced template metaprogramming library showcasing C++11/14/17 metaprogramming capabilities including type traits, SFINAE, constexpr functions, variable templates, and compile-time computations. This project demonstrates modern C++ template techniques for generic programming and compile-time optimization.

## Learning Objectives

- Master advanced template metaprogramming techniques
- Understand SFINAE (Substitution Failure Is Not An Error)
- Implement custom type traits and concepts
- Use constexpr for compile-time computations
- Create template specializations and partial specializations
- Build compile-time algorithms and data structures
- Understand template argument deduction and perfect forwarding

## Project Structure

```
template_metaprogramming_library/
├── include/
│   ├── tmp/
│   │   ├── core/
│   │   │   ├── type_traits.hpp
│   │   │   ├── concepts.hpp
│   │   │   ├── sfinae_helpers.hpp
│   │   │   └── meta_utilities.hpp
│   │   ├── algorithms/
│   │   │   ├── compile_time_sort.hpp
│   │   │   ├── type_list_algorithms.hpp
│   │   │   ├── string_algorithms.hpp
│   │   │   └── math_algorithms.hpp
│   │   ├── containers/
│   │   │   ├── compile_time_map.hpp
│   │   │   ├── type_list.hpp
│   │   │   ├── static_vector.hpp
│   │   │   └── tuple_extensions.hpp
│   │   ├── functional/
│   │   │   ├── function_traits.hpp
│   │   │   ├── curry.hpp
│   │   │   ├── compose.hpp
│   │   │   └── lazy_evaluation.hpp
│   │   └── tmp.hpp (main header)
│   └── examples/
│       ├── basic_metaprogramming.hpp
│       ├── advanced_type_traits.hpp
│       ├── compile_time_algorithms.hpp
│       └── functional_metaprogramming.hpp
├── src/
│   ├── examples/
│   │   ├── basic_usage.cpp
│   │   ├── type_manipulation.cpp
│   │   ├── compile_time_computations.cpp
│   │   ├── sfinae_examples.cpp
│   │   └── advanced_templates.cpp
├── tests/
│   ├── core_tests.cpp
│   ├── algorithm_tests.cpp
│   ├── container_tests.cpp
│   └── functional_tests.cpp
├── benchmarks/
│   ├── compile_time_vs_runtime.cpp
│   └── template_instantiation_cost.cpp
└── CMakeLists.txt
```

## Core Components

### 1. Core Type Traits and Utilities

```cpp
// include/tmp/core/type_traits.hpp
#pragma once
#include <type_traits>
#include <utility>
#include <functional>

namespace tmp {

// Enhanced type traits
template<typename T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template<typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

// Void_t implementation for SFINAE
template<typename...>
using void_t = void;

// Detection idiom implementation
namespace detail {
    template<template<typename...> class Op, typename... Args>
    using is_detected_impl = std::is_same<void_t<Op<Args...>>, void>;
    
    template<typename Default, template<typename...> class Op, typename... Args>
    struct detected_or_impl {
        using value_t = std::false_type;
        using type = Default;
    };
    
    template<typename Default, template<typename...> class Op, typename... Args>
    struct detected_or_impl<Default, Op, Args...> {
        using value_t = std::true_type;
        using type = Op<Args...>;
    };
}

template<template<typename...> class Op, typename... Args>
using is_detected = detail::is_detected_impl<Op, Args...>;

template<template<typename...> class Op, typename... Args>
constexpr bool is_detected_v = is_detected<Op, Args...>::value;

template<typename Default, template<typename...> class Op, typename... Args>
using detected_or = detail::detected_or_impl<Default, Op, Args...>;

template<typename Default, template<typename...> class Op, typename... Args>
using detected_or_t = typename detected_or<Default, Op, Args...>::type;

// Advanced type checking
template<typename T>
using has_value_type = typename T::value_type;

template<typename T>
using has_iterator = typename T::iterator;

template<typename T>
using has_size_method = decltype(std::declval<T>().size());

template<typename T>
using has_begin_end = decltype(std::begin(std::declval<T>()), std::end(std::declval<T>()));

// Container trait checks
template<typename T>
constexpr bool is_container_v = is_detected_v<has_value_type, T> && 
                               is_detected_v<has_iterator, T> &&
                               is_detected_v<has_begin_end, T>;

template<typename T>
constexpr bool has_size_v = is_detected_v<has_size_method, T>;

// Function signature analysis
template<typename T>
struct function_signature;

template<typename R, typename... Args>
struct function_signature<R(Args...)> {
    using return_type = R;
    using argument_types = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
    
    template<size_t N>
    using argument_type = std::tuple_element_t<N, argument_types>;
};

template<typename R, typename... Args>
struct function_signature<R(*)(Args...)> : function_signature<R(Args...)> {};

template<typename R, typename C, typename... Args>
struct function_signature<R(C::*)(Args...)> : function_signature<R(Args...)> {
    using class_type = C;
};

template<typename R, typename C, typename... Args>
struct function_signature<R(C::*)(Args...) const> : function_signature<R(Args...)> {
    using class_type = C;
};

// Lambda and callable analysis
template<typename T>
struct callable_traits : callable_traits<decltype(&T::operator())> {};

template<typename R, typename C, typename... Args>
struct callable_traits<R(C::*)(Args...)> : function_signature<R(Args...)> {};

template<typename R, typename C, typename... Args>
struct callable_traits<R(C::*)(Args...) const> : function_signature<R(Args...)> {};

// Type list operations
template<typename...>
struct type_list {};

template<typename List>
struct type_list_size;

template<typename... Types>
struct type_list_size<type_list<Types...>> {
    static constexpr size_t value = sizeof...(Types);
};

template<typename List>
constexpr size_t type_list_size_v = type_list_size<List>::value;

template<size_t Index, typename List>
struct type_list_element;

template<size_t Index, typename Head, typename... Tail>
struct type_list_element<Index, type_list<Head, Tail...>> {
    using type = typename type_list_element<Index - 1, type_list<Tail...>>::type;
};

template<typename Head, typename... Tail>
struct type_list_element<0, type_list<Head, Tail...>> {
    using type = Head;
};

template<size_t Index, typename List>
using type_list_element_t = typename type_list_element<Index, List>::type;

// Type list concatenation
template<typename... Lists>
struct type_list_concat;

template<typename... Types1, typename... Types2>
struct type_list_concat<type_list<Types1...>, type_list<Types2...>> {
    using type = type_list<Types1..., Types2...>;
};

template<typename List1, typename List2, typename... Rest>
struct type_list_concat<List1, List2, Rest...> {
    using type = typename type_list_concat<
        typename type_list_concat<List1, List2>::type,
        Rest...
    >::type;
};

template<typename... Lists>
using type_list_concat_t = typename type_list_concat<Lists...>::type;

// Type checking utilities
template<typename T, typename... Types>
struct is_one_of : std::disjunction<std::is_same<T, Types>...> {};

template<typename T, typename... Types>
constexpr bool is_one_of_v = is_one_of<T, Types...>::value;

template<typename T>
struct is_specialization_of {
    template<template<typename...> class Template>
    struct type : std::false_type {};
};

template<template<typename...> class Template, typename... Args>
struct is_specialization_of<Template<Args...>> {
    template<template<typename...> class T>
    struct type : std::is_same<Template<Args...>, T<Args...>> {};
};

// Compile-time string utilities
template<size_t N>
struct compile_time_string {
    constexpr compile_time_string(const char (&str)[N]) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = str[i];
        }
    }
    
    constexpr char operator[](size_t index) const {
        return data[index];
    }
    
    constexpr size_t size() const {
        return N - 1; // Exclude null terminator
    }
    
    constexpr const char* c_str() const {
        return data;
    }
    
    char data[N];
};

// Deduction guide for compile_time_string
template<size_t N>
compile_time_string(const char (&)[N]) -> compile_time_string<N>;

// Compile-time string operations
template<compile_time_string str>
constexpr auto string_length() {
    return str.size();
}

template<compile_time_string str, size_t pos>
constexpr char string_at() {
    static_assert(pos < str.size(), "Index out of bounds");
    return str[pos];
}

// Template parameter pack utilities
template<typename... Types>
constexpr size_t count_types() {
    return sizeof...(Types);
}

template<typename T, typename... Types>
constexpr size_t count_type_occurrences() {
    return (std::is_same_v<T, Types> + ...);
}

template<template<typename> class Predicate, typename... Types>
constexpr size_t count_if_types() {
    return (Predicate<Types>::value + ...);
}

// All/any/none predicates for type packs
template<template<typename> class Predicate, typename... Types>
constexpr bool all_types() {
    return (Predicate<Types>::value && ...);
}

template<template<typename> class Predicate, typename... Types>
constexpr bool any_types() {
    return (Predicate<Types>::value || ...);
}

template<template<typename> class Predicate, typename... Types>
constexpr bool none_types() {
    return (!Predicate<Types>::value && ...);
}

} // namespace tmp
```

### 2. SFINAE Helpers and Concepts

```cpp
// include/tmp/core/concepts.hpp
#pragma once
#include "type_traits.hpp"
#include <iterator>
#include <iostream>

namespace tmp {

// C++17 concepts emulation using SFINAE
#define REQUIRES(...) std::enable_if_t<(__VA_ARGS__), int> = 0

// Basic concept emulation
template<typename T>
using Integral = std::enable_if_t<std::is_integral_v<T>>;

template<typename T>
using FloatingPoint = std::enable_if_t<std::is_floating_point_v<T>>;

template<typename T>
using Arithmetic = std::enable_if_t<std::is_arithmetic_v<T>>;

// Iterator concepts
template<typename T>
using InputIterator = std::enable_if_t<
    std::is_same_v<typename std::iterator_traits<T>::iterator_category, std::input_iterator_tag> ||
    std::is_base_of_v<std::input_iterator_tag, typename std::iterator_traits<T>::iterator_category>
>;

template<typename T>
using ForwardIterator = std::enable_if_t<
    std::is_base_of_v<std::forward_iterator_tag, typename std::iterator_traits<T>::iterator_category>
>;

template<typename T>
using BidirectionalIterator = std::enable_if_t<
    std::is_base_of_v<std::bidirectional_iterator_tag, typename std::iterator_traits<T>::iterator_category>
>;

template<typename T>
using RandomAccessIterator = std::enable_if_t<
    std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<T>::iterator_category>
>;

// Container concepts
template<typename T>
using Container = std::enable_if_t<is_container_v<T>>;

template<typename T>
using Sequence = std::enable_if_t<is_container_v<T> && has_size_v<T>>;

// Callable concepts
template<typename F, typename... Args>
using Callable = std::enable_if_t<std::is_invocable_v<F, Args...>>;

template<typename F, typename R, typename... Args>
using CallableReturning = std::enable_if_t<std::is_invocable_r_v<R, F, Args...>>;

// Comparison concepts
template<typename T>
using EqualityComparable = std::enable_if_t<
    is_detected_v<decltype, std::declval<T>() == std::declval<T>()>
>;

template<typename T>
using LessThanComparable = std::enable_if_t<
    is_detected_v<decltype, std::declval<T>() < std::declval<T>()>
>;

// Streamable concepts
template<typename T>
using OutputStreamable = std::enable_if_t<
    is_detected_v<decltype, std::declval<std::ostream&>() << std::declval<T>()>
>;

template<typename T>
using InputStreamable = std::enable_if_t<
    is_detected_v<decltype, std::declval<std::istream&>() >> std::declval<T&>()>
>;

// Advanced SFINAE patterns
namespace sfinae {
    // Yes/No types for SFINAE testing
    using yes = char;
    using no = char[2];
    
    // SFINAE test macros
    #define DEFINE_HAS_MEMBER(member_name) \
        template<typename T> \
        struct has_##member_name { \
        private: \
            template<typename U> \
            static auto test(int) -> decltype(std::declval<U>().member_name, yes{}); \
            template<typename> \
            static no& test(...); \
        public: \
            static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes); \
        }; \
        template<typename T> \
        constexpr bool has_##member_name##_v = has_##member_name<T>::value;
    
    #define DEFINE_HAS_METHOD(method_name) \
        template<typename T, typename... Args> \
        struct has_##method_name { \
        private: \
            template<typename U> \
            static auto test(int) -> decltype(std::declval<U>().method_name(std::declval<Args>()...), yes{}); \
            template<typename> \
            static no& test(...); \
        public: \
            static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes); \
        }; \
        template<typename T, typename... Args> \
        constexpr bool has_##method_name##_v = has_##method_name<T, Args...>::value;
    
    #define DEFINE_HAS_TYPE(type_name) \
        template<typename T> \
        struct has_##type_name { \
        private: \
            template<typename U> \
            static yes test(typename U::type_name*); \
            template<typename> \
            static no& test(...); \
        public: \
            static constexpr bool value = sizeof(test<T>(nullptr)) == sizeof(yes); \
        }; \
        template<typename T> \
        constexpr bool has_##type_name##_v = has_##type_name<T>::value;
    
    // Common member/method detectors
    DEFINE_HAS_METHOD(size)
    DEFINE_HAS_METHOD(empty)
    DEFINE_HAS_METHOD(clear)
    DEFINE_HAS_METHOD(push_back)
    DEFINE_HAS_METHOD(insert)
    DEFINE_HAS_METHOD(find)
    DEFINE_HAS_METHOD(begin)
    DEFINE_HAS_METHOD(end)
    
    DEFINE_HAS_TYPE(value_type)
    DEFINE_HAS_TYPE(iterator)
    DEFINE_HAS_TYPE(const_iterator)
    DEFINE_HAS_TYPE(reference)
    DEFINE_HAS_TYPE(const_reference)
    
    // Expression SFINAE
    template<typename T, typename U>
    using addition_result_t = decltype(std::declval<T>() + std::declval<U>());
    
    template<typename T, typename U>
    using subtraction_result_t = decltype(std::declval<T>() - std::declval<U>());
    
    template<typename T, typename U>
    using multiplication_result_t = decltype(std::declval<T>() * std::declval<U>());
    
    template<typename T, typename U>
    using division_result_t = decltype(std::declval<T>() / std::declval<U>());
    
    template<typename T, typename U>
    constexpr bool is_addable_v = is_detected_v<addition_result_t, T, U>;
    
    template<typename T, typename U>
    constexpr bool is_subtractable_v = is_detected_v<subtraction_result_t, T, U>;
    
    template<typename T, typename U>
    constexpr bool is_multipliable_v = is_detected_v<multiplication_result_t, T, U>;
    
    template<typename T, typename U>
    constexpr bool is_dividable_v = is_detected_v<division_result_t, T, U>;
    
    // Arithmetic concept
    template<typename T>
    constexpr bool is_arithmetic_type_v = 
        is_addable_v<T, T> && 
        is_subtractable_v<T, T> && 
        is_multipliable_v<T, T> && 
        is_dividable_v<T, T>;
}

// Conditional type selection
template<bool Condition, typename TrueType, typename FalseType>
using conditional_t = std::conditional_t<Condition, TrueType, FalseType>;

// Enable if shortcuts
template<bool Condition, typename T = void>
using enable_if_t = std::enable_if_t<Condition, T>;

template<typename Condition, typename T = void>
using enable_if_type_t = std::enable_if_t<Condition::value, T>;

// Tag dispatching utilities
struct compile_time_tag {};
struct runtime_tag {};

template<bool CompileTime>
using execution_tag = conditional_t<CompileTime, compile_time_tag, runtime_tag>;

// Overload resolution helpers
template<typename... Ts>
struct overload : Ts... {
    using Ts::operator()...;
};

template<typename... Ts>
overload(Ts...) -> overload<Ts...>;

// Perfect forwarding utilities
template<typename T>
constexpr decltype(auto) perfect_forward(std::remove_reference_t<T>& t) noexcept {
    return static_cast<T&&>(t);
}

template<typename T>
constexpr decltype(auto) perfect_forward(std::remove_reference_t<T>&& t) noexcept {
    static_assert(!std::is_lvalue_reference_v<T>, "Cannot forward rvalue as lvalue");
    return static_cast<T&&>(t);
}

} // namespace tmp
```

### 3. Compile-Time Algorithms

```cpp
// include/tmp/algorithms/compile_time_sort.hpp
#pragma once
#include "../core/type_traits.hpp"
#include <array>

namespace tmp {

// Compile-time quicksort implementation
namespace detail {
    template<typename T, size_t N>
    constexpr void swap_elements(std::array<T, N>& arr, size_t i, size_t j) {
        T temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
    
    template<typename T, size_t N, typename Compare>
    constexpr size_t partition(std::array<T, N>& arr, size_t low, size_t high, Compare comp) {
        T pivot = arr[high];
        size_t i = low;
        
        for (size_t j = low; j < high; ++j) {
            if (comp(arr[j], pivot)) {
                swap_elements(arr, i, j);
                ++i;
            }
        }
        
        swap_elements(arr, i, high);
        return i;
    }
    
    template<typename T, size_t N, typename Compare>
    constexpr void quicksort_impl(std::array<T, N>& arr, size_t low, size_t high, Compare comp) {
        if (low < high) {
            size_t pi = partition(arr, low, high, comp);
            
            if (pi > 0) {
                quicksort_impl(arr, low, pi - 1, comp);
            }
            quicksort_impl(arr, pi + 1, high, comp);
        }
    }
}

// Compile-time sort function
template<typename T, size_t N, typename Compare = std::less<T>>
constexpr std::array<T, N> compile_time_sort(std::array<T, N> arr, Compare comp = {}) {
    if constexpr (N > 1) {
        detail::quicksort_impl(arr, 0, N - 1, comp);
    }
    return arr;
}

// Compile-time binary search
template<typename T, size_t N, typename Compare = std::less<T>>
constexpr bool compile_time_binary_search(const std::array<T, N>& sorted_arr, const T& value, Compare comp = {}) {
    size_t left = 0;
    size_t right = N;
    
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        
        if (comp(sorted_arr[mid], value)) {
            left = mid + 1;
        } else if (comp(value, sorted_arr[mid])) {
            right = mid;
        } else {
            return true;
        }
    }
    
    return false;
}

// Compile-time find algorithm
template<typename T, size_t N>
constexpr size_t compile_time_find(const std::array<T, N>& arr, const T& value) {
    for (size_t i = 0; i < N; ++i) {
        if (arr[i] == value) {
            return i;
        }
    }
    return N; // Not found
}

// Compile-time unique algorithm
template<typename T, size_t N>
constexpr auto compile_time_unique(const std::array<T, N>& arr) {
    std::array<T, N> result{};
    size_t result_size = 0;
    
    for (size_t i = 0; i < N; ++i) {
        bool found = false;
        for (size_t j = 0; j < result_size; ++j) {
            if (result[j] == arr[i]) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            result[result_size++] = arr[i];
        }
    }
    
    return std::make_pair(result, result_size);
}

// Compile-time mathematical algorithms
namespace math {
    // Compile-time factorial
    constexpr uintmax_t factorial(unsigned n) {
        return n == 0 ? 1 : n * factorial(n - 1);
    }
    
    // Compile-time fibonacci
    constexpr uintmax_t fibonacci(unsigned n) {
        return n <= 1 ? n : fibonacci(n - 1) + fibonacci(n - 2);
    }
    
    // Compile-time power
    constexpr uintmax_t power(uintmax_t base, unsigned exp) {
        return exp == 0 ? 1 : base * power(base, exp - 1);
    }
    
    // Compile-time GCD
    constexpr uintmax_t gcd(uintmax_t a, uintmax_t b) {
        return b == 0 ? a : gcd(b, a % b);
    }
    
    // Compile-time LCM
    constexpr uintmax_t lcm(uintmax_t a, uintmax_t b) {
        return (a * b) / gcd(a, b);
    }
    
    // Compile-time prime checking
    constexpr bool is_prime(uintmax_t n) {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        
        for (uintmax_t i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        
        return true;
    }
    
    // Compile-time square root (integer)
    constexpr uintmax_t sqrt_int(uintmax_t n) {
        if (n == 0) return 0;
        
        uintmax_t left = 1, right = n;
        uintmax_t result = 0;
        
        while (left <= right) {
            uintmax_t mid = left + (right - left) / 2;
            
            if (mid <= n / mid) {
                result = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
}

// String algorithms (compile-time)
namespace string {
    // Compile-time string length
    constexpr size_t strlen_ct(const char* str) {
        size_t len = 0;
        while (str[len] != '\0') {
            ++len;
        }
        return len;
    }
    
    // Compile-time string comparison
    constexpr int strcmp_ct(const char* s1, const char* s2) {
        while (*s1 && (*s1 == *s2)) {
            ++s1;
            ++s2;
        }
        return *s1 - *s2;
    }
    
    // Compile-time string concatenation
    template<size_t N1, size_t N2>
    constexpr auto strcat_ct(const char (&s1)[N1], const char (&s2)[N2]) {
        std::array<char, N1 + N2 - 1> result{};
        size_t pos = 0;
        
        // Copy first string (excluding null terminator)
        for (size_t i = 0; i < N1 - 1; ++i) {
            result[pos++] = s1[i];
        }
        
        // Copy second string (including null terminator)
        for (size_t i = 0; i < N2; ++i) {
            result[pos++] = s2[i];
        }
        
        return result;
    }
    
    // Compile-time character counting
    constexpr size_t count_char(const char* str, char c) {
        size_t count = 0;
        while (*str) {
            if (*str == c) {
                ++count;
            }
            ++str;
        }
        return count;
    }
    
    // Compile-time string reversal
    template<size_t N>
    constexpr std::array<char, N> reverse_string(const char (&str)[N]) {
        std::array<char, N> result{};
        
        for (size_t i = 0; i < N - 1; ++i) {
            result[i] = str[N - 2 - i];
        }
        result[N - 1] = '\0';
        
        return result;
    }
}

} // namespace tmp
```

### 4. Functional Metaprogramming

```cpp
// include/tmp/functional/function_traits.hpp
#pragma once
#include "../core/type_traits.hpp"
#include <tuple>
#include <functional>

namespace tmp {

// Enhanced function traits
template<typename F>
struct function_traits : function_traits<decltype(&F::operator())> {};

// Function pointer specialization
template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> {
    using return_type = R;
    using argument_types = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
    
    template<size_t N>
    using argument_type = std::tuple_element_t<N, argument_types>;
    
    using function_type = R(Args...);
    using function_pointer_type = R(*)(Args...);
};

// Member function pointer specialization
template<typename R, typename C, typename... Args>
struct function_traits<R(C::*)(Args...)> {
    using return_type = R;
    using argument_types = std::tuple<Args...>;
    using class_type = C;
    static constexpr size_t arity = sizeof...(Args);
    
    template<size_t N>
    using argument_type = std::tuple_element_t<N, argument_types>;
    
    using function_type = R(Args...);
    using member_function_pointer_type = R(C::*)(Args...);
};

// Const member function pointer specialization
template<typename R, typename C, typename... Args>
struct function_traits<R(C::*)(Args...) const> {
    using return_type = R;
    using argument_types = std::tuple<Args...>;
    using class_type = C;
    static constexpr size_t arity = sizeof...(Args);
    static constexpr bool is_const = true;
    
    template<size_t N>
    using argument_type = std::tuple_element_t<N, argument_types>;
    
    using function_type = R(Args...);
    using const_member_function_pointer_type = R(C::*)(Args...) const;
};

// Lambda and callable object support
template<typename R, typename C, typename... Args>
struct function_traits<R(C::*)(Args...) const> {
    using return_type = R;
    using argument_types = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
    
    template<size_t N>
    using argument_type = std::tuple_element_t<N, argument_types>;
};

// Function composition
template<typename F, typename G>
class composed_function {
private:
    F f_;
    G g_;
    
public:
    composed_function(F f, G g) : f_(std::move(f)), g_(std::move(g)) {}
    
    template<typename... Args>
    auto operator()(Args&&... args) -> decltype(f_(g_(std::forward<Args>(args)...))) {
        return f_(g_(std::forward<Args>(args)...));
    }
};

template<typename F, typename G>
auto compose(F&& f, G&& g) {
    return composed_function<std::decay_t<F>, std::decay_t<G>>{
        std::forward<F>(f), std::forward<G>(g)
    };
}

// Variadic function composition
template<typename F>
auto compose_all(F&& f) {
    return std::forward<F>(f);
}

template<typename F, typename G, typename... Rest>
auto compose_all(F&& f, G&& g, Rest&&... rest) {
    return compose_all(compose(std::forward<F>(f), std::forward<G>(g)), std::forward<Rest>(rest)...);
}

// Currying implementation
template<typename F, typename... CapturedArgs>
class curried_function {
private:
    F f_;
    std::tuple<CapturedArgs...> captured_args_;
    
    template<typename... Args>
    static constexpr bool has_enough_args() {
        using traits = function_traits<F>;
        return sizeof...(CapturedArgs) + sizeof...(Args) >= traits::arity;
    }
    
public:
    curried_function(F f, CapturedArgs... args) 
        : f_(std::move(f)), captured_args_(std::move(args)...) {}
    
    template<typename... Args>
    auto operator()(Args&&... args) {
        if constexpr (has_enough_args<Args...>()) {
            // We have enough arguments, call the function
            return std::apply([&](auto&&... captured) {
                return f_(std::forward<decltype(captured)>(captured)..., std::forward<Args>(args)...);
            }, captured_args_);
        } else {
            // Not enough arguments yet, return a new curried function
            return curried_function<F, CapturedArgs..., std::decay_t<Args>...>{
                f_, 
                std::get<CapturedArgs>(captured_args_)..., 
                std::forward<Args>(args)...
            };
        }
    }
};

template<typename F>
auto curry(F&& f) {
    return curried_function<std::decay_t<F>>{std::forward<F>(f)};
}

// Partial application
template<typename F, typename... Args>
class partial_application {
private:
    F f_;
    std::tuple<Args...> args_;
    
public:
    partial_application(F f, Args... args) 
        : f_(std::move(f)), args_(std::move(args)...) {}
    
    template<typename... RestArgs>
    auto operator()(RestArgs&&... rest_args) {
        return std::apply([&](auto&&... captured) {
            return f_(std::forward<decltype(captured)>(captured)..., std::forward<RestArgs>(rest_args)...);
        }, args_);
    }
};

template<typename F, typename... Args>
auto partial(F&& f, Args&&... args) {
    return partial_application<std::decay_t<F>, std::decay_t<Args>...>{
        std::forward<F>(f), std::forward<Args>(args)...
    };
}

// Function binding with placeholders
namespace placeholders {
    struct placeholder_1 {};
    struct placeholder_2 {};
    struct placeholder_3 {};
    
    constexpr placeholder_1 _1{};
    constexpr placeholder_2 _2{};
    constexpr placeholder_3 _3{};
    
    template<int N>
    struct placeholder {};
    
    template<int N>
    constexpr placeholder<N> _n{};
}

template<typename T>
struct is_placeholder : std::false_type {};

template<int N>
struct is_placeholder<placeholders::placeholder<N>> : std::true_type {};

template<>
struct is_placeholder<placeholders::placeholder_1> : std::true_type {};

template<>
struct is_placeholder<placeholders::placeholder_2> : std::true_type {};

template<>
struct is_placeholder<placeholders::placeholder_3> : std::true_type {};

// Lazy evaluation wrapper
template<typename F, typename... Args>
class lazy_evaluation {
private:
    F f_;
    std::tuple<Args...> args_;
    mutable std::optional<std::invoke_result_t<F, Args...>> cached_result_;
    
public:
    lazy_evaluation(F f, Args... args) 
        : f_(std::move(f)), args_(std::move(args)...) {}
    
    auto operator()() const -> std::invoke_result_t<F, Args...> {
        if (!cached_result_) {
            cached_result_ = std::apply(f_, args_);
        }
        return *cached_result_;
    }
    
    auto get() const -> std::invoke_result_t<F, Args...> {
        return operator()();
    }
    
    void reset() {
        cached_result_.reset();
    }
};

template<typename F, typename... Args>
auto lazy(F&& f, Args&&... args) {
    return lazy_evaluation<std::decay_t<F>, std::decay_t<Args>...>{
        std::forward<F>(f), std::forward<Args>(args)...
    };
}

// Higher-order function utilities
template<typename F, typename Container>
auto map(F&& f, const Container& container) {
    using value_type = std::invoke_result_t<F, typename Container::value_type>;
    std::vector<value_type> result;
    result.reserve(container.size());
    
    for (const auto& item : container) {
        result.push_back(f(item));
    }
    
    return result;
}

template<typename F, typename Container>
auto filter(F&& predicate, const Container& container) {
    Container result;
    
    for (const auto& item : container) {
        if (predicate(item)) {
            if constexpr (sfinae::has_push_back_v<Container, typename Container::value_type>) {
                result.push_back(item);
            } else {
                result.insert(result.end(), item);
            }
        }
    }
    
    return result;
}

template<typename F, typename T, typename Container>
auto fold_left(F&& f, T&& initial, const Container& container) {
    auto result = std::forward<T>(initial);
    
    for (const auto& item : container) {
        result = f(std::move(result), item);
    }
    
    return result;
}

template<typename F, typename T, typename Container>
auto fold_right(F&& f, T&& initial, const Container& container) {
    auto result = std::forward<T>(initial);
    
    for (auto it = container.rbegin(); it != container.rend(); ++it) {
        result = f(*it, std::move(result));
    }
    
    return result;
}

// Function memoization
template<typename F>
class memoized_function {
private:
    F f_;
    mutable std::map<std::tuple<typename function_traits<F>::argument_types>, 
                     typename function_traits<F>::return_type> cache_;
    
public:
    memoized_function(F f) : f_(std::move(f)) {}
    
    template<typename... Args>
    auto operator()(Args&&... args) const -> std::invoke_result_t<F, Args...> {
        auto key = std::make_tuple(args...);
        
        if (auto it = cache_.find(key); it != cache_.end()) {
            return it->second;
        }
        
        auto result = f_(std::forward<Args>(args)...);
        cache_[key] = result;
        return result;
    }
    
    void clear_cache() {
        cache_.clear();
    }
    
    size_t cache_size() const {
        return cache_.size();
    }
};

template<typename F>
auto memoize(F&& f) {
    return memoized_function<std::decay_t<F>>{std::forward<F>(f)};
}

} // namespace tmp
```

### 5. Usage Examples and Tests

```cpp
// src/examples/basic_usage.cpp
#include <iostream>
#include <vector>
#include <string>
#include "tmp/tmp.hpp"

using namespace tmp;

// Demonstrate basic type traits
void demonstrate_type_traits() {
    std::cout << "\n=== Type Traits Demonstration ===" << std::endl;
    
    // Basic type checking
    std::cout << "std::vector<int> is container: " << is_container_v<std::vector<int>> << std::endl;
    std::cout << "int is container: " << is_container_v<int> << std::endl;
    
    // Function analysis
    auto lambda = [](int x, double y) -> std::string { 
        return std::to_string(x + y); 
    };
    
    using lambda_traits = callable_traits<decltype(lambda)>;
    std::cout << "Lambda arity: " << lambda_traits::arity << std::endl;
    std::cout << "Lambda return type is string: " << 
        std::is_same_v<lambda_traits::return_type, std::string> << std::endl;
    
    // Type list operations
    using types = type_list<int, double, std::string, char>;
    std::cout << "Type list size: " << type_list_size_v<types> << std::endl;
    std::cout << "Second type is double: " << 
        std::is_same_v<type_list_element_t<1, types>, double> << std::endl;
    
    // SFINAE detection
    std::cout << "std::vector<int> has size method: " << 
        sfinae::has_size_v<std::vector<int>> << std::endl;
    std::cout << "int has size method: " << 
        sfinae::has_size_v<int> << std::endl;
}

// Demonstrate compile-time computations
void demonstrate_compile_time_computation() {
    std::cout << "\n=== Compile-Time Computation Demonstration ===" << std::endl;
    
    // Compile-time math
    constexpr auto fact_10 = math::factorial(10);
    constexpr auto fib_20 = math::fibonacci(20);
    constexpr auto pow_2_16 = math::power(2, 16);
    constexpr auto gcd_48_18 = math::gcd(48, 18);
    
    std::cout << "10! = " << fact_10 << std::endl;
    std::cout << "fibonacci(20) = " << fib_20 << std::endl;
    std::cout << "2^16 = " << pow_2_16 << std::endl;
    std::cout << "gcd(48, 18) = " << gcd_48_18 << std::endl;
    
    // Compile-time array sorting
    constexpr std::array<int, 10> unsorted = {64, 34, 25, 12, 22, 11, 90, 88, 76, 50};
    constexpr auto sorted = compile_time_sort(unsorted);
    
    std::cout << "Original array: ";
    for (const auto& val : unsorted) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Sorted array: ";
    for (const auto& val : sorted) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Compile-time binary search
    constexpr bool found_25 = compile_time_binary_search(sorted, 25);
    constexpr bool found_99 = compile_time_binary_search(sorted, 99);
    
    std::cout << "25 found in sorted array: " << found_25 << std::endl;
    std::cout << "99 found in sorted array: " << found_99 << std::endl;
    
    // Compile-time string operations
    constexpr const char* hello = "Hello";
    constexpr const char* world = "World";
    constexpr auto concatenated = string::strcat_ct(hello, world);
    constexpr auto hello_length = string::strlen_ct(hello);
    constexpr auto char_count = string::count_char("Hello World", 'l');
    
    std::cout << "String concatenation: ";
    for (char c : concatenated) {
        if (c == '\0') break;
        std::cout << c;
    }
    std::cout << std::endl;
    
    std::cout << "Length of 'Hello': " << hello_length << std::endl;
    std::cout << "Count of 'l' in 'Hello World': " << char_count << std::endl;
}

// Demonstrate functional programming features
void demonstrate_functional_programming() {
    std::cout << "\n=== Functional Programming Demonstration ===" << std::endl;
    
    // Function composition
    auto add_one = [](int x) { return x + 1; };
    auto multiply_two = [](int x) { return x * 2; };
    auto square = [](int x) { return x * x; };
    
    auto composed = compose_all(square, multiply_two, add_one);
    auto result = composed(5); // ((5 + 1) * 2)^2 = 144
    
    std::cout << "Composed function result for input 5: " << result << std::endl;
    
    // Currying
    auto add_three_numbers = [](int a, int b, int c) { return a + b + c; };
    auto curried_add = curry(add_three_numbers);
    
    auto add_5_and = curried_add(5);
    auto add_5_and_10 = add_5_and(10);
    auto final_result = add_5_and_10(7); // 5 + 10 + 7 = 22
    
    std::cout << "Curried function result (5 + 10 + 7): " << final_result << std::endl;
    
    // Partial application
    auto multiply = [](int a, int b, int c) { return a * b * c; };
    auto multiply_by_2_and_3 = partial(multiply, 2, 3);
    auto partial_result = multiply_by_2_and_3(4); // 2 * 3 * 4 = 24
    
    std::cout << "Partial application result (2 * 3 * 4): " << partial_result << std::endl;
    
    // Higher-order functions
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    auto squared_numbers = map([](int x) { return x * x; }, numbers);
    auto even_numbers = filter([](int x) { return x % 2 == 0; }, numbers);
    auto sum = fold_left([](int acc, int x) { return acc + x; }, 0, numbers);
    
    std::cout << "Squared numbers: ";
    for (int n : squared_numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Even numbers: ";
    for (int n : even_numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Sum of all numbers: " << sum << std::endl;
    
    // Lazy evaluation
    auto expensive_computation = lazy([](int n) {
        std::cout << "Computing factorial of " << n << "..." << std::endl;
        return math::factorial(n);
    }, 10);
    
    std::cout << "Lazy computation created (not executed yet)" << std::endl;
    std::cout << "First call result: " << expensive_computation() << std::endl;
    std::cout << "Second call result (cached): " << expensive_computation() << std::endl;
    
    // Memoization
    auto fib_memo = memoize([](int n) -> int {
        if (n <= 1) return n;
        std::cout << "Computing fibonacci(" << n << ")" << std::endl;
        return math::fibonacci(n); // This would be recursive in real implementation
    });
    
    std::cout << "Memoized fibonacci(10): " << fib_memo(10) << std::endl;
    std::cout << "Memoized fibonacci(10) again (cached): " << fib_memo(10) << std::endl;
    std::cout << "Cache size: " << fib_memo.cache_size() << std::endl;
}

// Demonstrate template specialization and SFINAE
template<typename T, REQUIRES(std::is_arithmetic_v<T>)>
void print_arithmetic_info(const T& value) {
    std::cout << "Arithmetic value: " << value << std::endl;
    
    if constexpr (std::is_integral_v<T>) {
        std::cout << "  Type: integral" << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "  Type: floating point" << std::endl;
    }
    
    std::cout << "  Size: " << sizeof(T) << " bytes" << std::endl;
    std::cout << "  Min: " << std::numeric_limits<T>::min() << std::endl;
    std::cout << "  Max: " << std::numeric_limits<T>::max() << std::endl;
}

template<typename Container, REQUIRES(is_container_v<Container>)>
void print_container_info(const Container& container) {
    std::cout << "Container information:" << std::endl;
    std::cout << "  Size: " << container.size() << std::endl;
    std::cout << "  Empty: " << container.empty() << std::endl;
    
    using value_type = typename Container::value_type;
    std::cout << "  Element type size: " << sizeof(value_type) << " bytes" << std::endl;
    
    if constexpr (sfinae::has_find_v<Container, value_type>) {
        std::cout << "  Has find method: yes" << std::endl;
    } else {
        std::cout << "  Has find method: no" << std::endl;
    }
}

void demonstrate_sfinae_and_specialization() {
    std::cout << "\n=== SFINAE and Specialization Demonstration ===" << std::endl;
    
    // Arithmetic types
    print_arithmetic_info(42);
    print_arithmetic_info(3.14);
    print_arithmetic_info(100L);
    
    // Container types
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::string str = "Hello, World!";
    
    print_container_info(vec);
    print_container_info(str);
}

int main() {
    try {
        demonstrate_type_traits();
        demonstrate_compile_time_computation();
        demonstrate_functional_programming();
        demonstrate_sfinae_and_specialization();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Build Configuration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(TemplateMetaprogrammingLibrary)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only library
add_library(tmp_lib INTERFACE)
target_include_directories(tmp_lib INTERFACE include)

# Examples
add_executable(basic_usage src/examples/basic_usage.cpp)
target_link_libraries(basic_usage tmp_lib)

add_executable(type_manipulation src/examples/type_manipulation.cpp)
target_link_libraries(type_manipulation tmp_lib)

add_executable(compile_time_computations src/examples/compile_time_computations.cpp)
target_link_libraries(compile_time_computations tmp_lib)

add_executable(sfinae_examples src/examples/sfinae_examples.cpp)
target_link_libraries(sfinae_examples tmp_lib)

add_executable(advanced_templates src/examples/advanced_templates.cpp)
target_link_libraries(advanced_templates tmp_lib)

# Tests
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(tmp_tests
        tests/core_tests.cpp
        tests/algorithm_tests.cpp
        tests/container_tests.cpp
        tests/functional_tests.cpp
    )
    target_link_libraries(tmp_tests GTest::gtest_main tmp_lib)
    
    enable_testing()
    add_test(NAME TMP_Tests COMMAND tmp_tests)
endif()

# Benchmarks
add_executable(compile_time_vs_runtime benchmarks/compile_time_vs_runtime.cpp)
target_link_libraries(compile_time_vs_runtime tmp_lib)

add_executable(template_instantiation_cost benchmarks/template_instantiation_cost.cpp)
target_link_libraries(template_instantiation_cost tmp_lib)

# Compiler-specific options
if(MSVC)
    target_compile_options(tmp_lib INTERFACE /W4 /permissive-)
    # Enable additional template debugging for MSVC
    target_compile_options(tmp_lib INTERFACE /diagnostics:caret)
else()
    target_compile_options(tmp_lib INTERFACE -Wall -Wextra -Wpedantic)
    
    # Enable template backtrace for GCC
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(tmp_lib INTERFACE -ftemplate-backtrace-limit=0)
    endif()
    
    # Enable additional warnings for template code
    target_compile_options(tmp_lib INTERFACE -Wtemplate-conversions)
endif()

# Debug/Release specific options
target_compile_definitions(tmp_lib INTERFACE
    $<$<CONFIG:Debug>:TMP_DEBUG_ENABLED>
    $<$<CONFIG:Release>:TMP_OPTIMIZED_BUILD>
)

# Install targets (for header-only library)
install(DIRECTORY include/tmp DESTINATION include)
install(TARGETS tmp_lib EXPORT tmp_libTargets)
install(EXPORT tmp_libTargets
    FILE tmp_libTargets.cmake
    DESTINATION lib/cmake/tmp_lib
)
```

## Learning Exercises

### Exercise 1: Custom Type Traits
Create type traits to detect if a type has specific operators or methods:
- `has_plus_operator_v<T, U>`
- `has_stream_operator_v<T>`
- `has_hash_specialization_v<T>`

### Exercise 2: Compile-Time String Processing
Implement compile-time string processing functions:
- String hashing algorithm
- Pattern matching
- Regular expression subset

### Exercise 3: Template Recursion Control
Implement safe recursive template patterns:
- Tail recursion optimization
- Stack depth limiting
- Compile-time loop unrolling

### Exercise 4: Advanced SFINAE Patterns
Create sophisticated SFINAE-based function overloading:
- Method resolution based on capabilities
- Concept-like constraints
- Priority-based overload selection

## Expected Learning Outcomes

After completing this project, you should master:

1. **Advanced Template Techniques**
   - Template specialization and partial specialization
   - Variadic templates and parameter pack manipulation
   - Template template parameters
   - CRTP (Curiously Recurring Template Pattern)

2. **SFINAE and Type Traits**
   - Complex SFINAE patterns and detection idioms
   - Custom type trait implementation
   - Enable_if patterns and conditional compilation

3. **Compile-Time Programming**
   - Constexpr functions and algorithms
   - Compile-time data structures
   - Template metaprogramming for performance

4. **Functional Programming Concepts**
   - Higher-order functions and combinators
   - Function composition and currying
   - Lazy evaluation and memoization

This comprehensive template metaprogramming library showcases the full power of modern C++ template features and prepares you for advanced generic programming challenges.
