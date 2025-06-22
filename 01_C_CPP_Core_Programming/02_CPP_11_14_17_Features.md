# C++ 11/14/17 Features

*Last Updated: May 25, 2025*

## Overview

Modern C++ (C++11, C++14, and C++17) introduced significant improvements to the language that make code more expressive, efficient, and safer. This learning track covers the key features of these standards that every C++ developer should know.

## Learning Path

### 1. C++11 Core Language Features (2 weeks)
[See details in 01_CPP11_Core_Language_Features.md](02_CPP_11_14_17_Features/01_CPP11_Core_Language_Features.md)
- Auto type deduction
- Range-based for loops
- Lambda expressions
- nullptr
- Strongly-typed enumerations (enum class)
- Static assertions
- Delegating constructors
- Initializer lists
- Uniform initialization

### 2. C++11 Move Semantics (2 weeks)
[See details in 02_CPP11_Move_Semantics.md](02_CPP_11_14_17_Features/02_CPP11_Move_Semantics.md)
- Rvalue references
- Move constructors and move assignment operators
- Perfect forwarding
- std::move and std::forward
- Return value optimization

### 3. C++11 Smart Pointers (1 week)
[See details in 03_CPP11_Smart_Pointers.md](02_CPP_11_14_17_Features/03_CPP11_Smart_Pointers.md)
- std::unique_ptr
- std::shared_ptr
- std::weak_ptr
- Custom deleters
- Best practices for memory management

### 4. C++11 Concurrency Support (2 weeks)
[See details in 04_CPP11_Concurrency_Support.md](02_CPP_11_14_17_Features/04_CPP11_Concurrency_Support.md)
- std::thread
- Futures and promises (std::future, std::promise)
- std::async
- Atomic operations
- Memory model and memory ordering
- Mutexes and locks

### 5. C++11 Type Traits and Metaprogramming (1 week)
[See details in 05_CPP11_Type_Traits_Metaprogramming.md](02_CPP_11_14_17_Features/05_CPP11_Type_Traits_Metaprogramming.md)
- SFINAE (Substitution Failure Is Not An Error)
- std::enable_if
- Type traits library
- Variadic templates
- Template metaprogramming techniques

### 6. C++14 Improvements (1 week)
[See details in 06_CPP14_Improvements.md](02_CPP_11_14_17_Features/06_CPP14_Improvements.md)
- Generic lambdas
- Return type deduction for functions
- Variable templates
- Extended constexpr
- std::make_unique
- Shared locking
- Heterogeneous lookup in associative containers

- Structured bindings
- if constexpr
- Inline variables
- fold expressions
- constexpr if
- Class template argument deduction
- Guaranteed copy elision
### 7. C++17 Core Language Features (2 weeks)
[See details in 07_CPP17_Core_Language_Features.md](02_CPP_11_14_17_Features/07_CPP17_Core_Language_Features.md)
- Structured bindings
- if constexpr
- Inline variables
- fold expressions
- constexpr if
- Class template argument deduction
- Guaranteed copy elision
- Structured bindings
- if constexpr
- Inline variables
- fold expressions
- constexpr if
- Class template argument deduction
- Guaranteed copy elision

### 8. C++17 Library Additions (2 weeks)
[See details in 08_CPP17_Library_Additions.md](02_CPP_11_14_17_Features/08_CPP17_Library_Additions.md)
- std::optional
- std::variant
- std::any
- std::string_view
- std::filesystem
- Parallel algorithms
- std::invoke and other utilities

## Projects

1. **Move Semantics Implementation**  
   [See project details](02_CPP_11_14_17_Features/Projects/01_Move_Semantics_Implementation.md)
   - Create a resource-owning class with proper move semantics
   - Demonstrate performance improvements

2. **Thread Pool with Modern C++**  
   [See project details](02_CPP_11_14_17_Features/Projects/02_Thread_Pool_Modern_CPP.md)
   - Implement a thread pool using std::thread
   - Use futures and promises for task results

3. **Type-Safe Configuration System**  
   [See project details](02_CPP_11_14_17_Features/Projects/03_Type_Safe_Configuration_System.md)
   - Use variant and optional for type-safe configuration
   - Implement visitors for configuration processing

4. **File System Explorer**  
   [See project details](02_CPP_11_14_17_Features/Projects/04_File_System_Explorer.md)
   - Use std::filesystem to create a simple file explorer
   - Implement recursive directory traversal

5. **Template Metaprogramming Library**  
   [See project details](02_CPP_11_14_17_Features/Projects/05_Template_Metaprogramming_Library.md)
   - Create compile-time utilities using C++11/14/17 features
   - Implement type-safe containers or algorithms

## Resources

### Books
- "Effective Modern C++" by Scott Meyers
- "C++ Templates: The Complete Guide" by David Vandevoorde and Nicolai M. Josuttis
- "C++17 - The Complete Guide" by Nicolai M. Josuttis
- "Concurrency with Modern C++" by Rainer Grimm

### Online Resources
- [C++ Reference](https://en.cppreference.com/w/)
- [CppCon Conference Videos](https://www.youtube.com/user/CppCon)
- [ISO C++ Website](https://isocpp.org/)
- [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines)

### Video Courses
- "Learn Advanced Modern C++" on Udemy
- "C++11/14/17 Features" on Pluralsight

## Assessment Criteria

You should be able to:
- Implement efficient and safe memory management using smart pointers
- Use move semantics appropriately for performance-critical code
- Write thread-safe concurrent code using modern C++ primitives
- Leverage type traits and template metaprogramming for generic code
- Apply C++17 features to simplify common programming tasks

## Next Steps

After mastering modern C++ features, consider exploring:
- C++20 features (concepts, ranges, coroutines)
- Template metaprogramming libraries (Boost.Hana, Boost.MP11)
- Lock-free programming
- Modern C++ design patterns
- Reflection techniques in C++
