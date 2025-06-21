# C++11/14/17 Features Learning Guide

This directory contains a comprehensive breakdown of C++11, C++14, and C++17 features organized into modular study materials and practical projects.

## 📚 Study Materials

### Core Language Features
1. **[C++11 Core Language Features](01_CPP11_Core_Language_Features.md)**
   - Auto type deduction and decltype
   - Range-based for loops
   - Lambda expressions
   - Rvalue references and move semantics basics
   - Uniform initialization
   - Nullptr and strongly typed enums

2. **[C++11 Move Semantics](02_CPP11_Move_Semantics.md)**
   - Deep dive into rvalue references
   - Move constructors and assignment operators
   - Perfect forwarding with std::forward
   - Universal references
   - Move-only types and std::unique_ptr

3. **[C++11 Smart Pointers](03_CPP11_Smart_Pointers.md)**
   - std::unique_ptr comprehensive guide
   - std::shared_ptr and reference counting
   - std::weak_ptr for breaking cycles
   - Custom deleters and memory management
   - RAII patterns and best practices

4. **[C++11 Concurrency Support](04_CPP11_Concurrency_Support.md)**
   - std::thread and thread management
   - std::mutex, std::lock_guard, std::unique_lock
   - std::condition_variable for synchronization
   - std::atomic operations and memory ordering
   - std::future, std::promise, and std::async

5. **[C++11 Type Traits and Metaprogramming](05_CPP11_Type_Traits_Metaprogramming.md)**
   - Type traits library (<type_traits>)
   - SFINAE patterns and enable_if
   - Template metaprogramming techniques
   - Variadic templates and parameter packs
   - Template aliases and using declarations

### Evolution Features
6. **[C++14 Improvements](06_CPP14_Improvements.md)**
   - Generic lambdas and lambda captures
   - Return type deduction for functions
   - Variable templates
   - Binary literals and digit separators
   - std::make_unique and library enhancements

7. **[C++17 Core Language Features](07_CPP17_Core_Language_Features.md)**
   - Structured bindings
   - if constexpr for template branching
   - Fold expressions for variadic templates
   - Class template argument deduction (CTAD)
   - Inline variables

8. **[C++17 Library Additions](08_CPP17_Library_Additions.md)**
   - std::optional for nullable values
   - std::variant as type-safe unions
   - std::any for type erasure
   - std::filesystem for portable file operations
   - std::string_view for efficient string handling
   - Parallel algorithms with execution policies

## 🚀 Practical Projects

The `Projects/` directory contains comprehensive projects that combine multiple C++ features in real-world scenarios:

### 1. [Move Semantics Implementation](Projects/01_Move_Semantics_Implementation.md)
**Focus**: Deep understanding of move semantics and perfect forwarding
- Custom container with optimal move operations
- Memory pool allocator with move support
- Performance benchmarking framework
- **Key Features**: Rvalue references, std::forward, move constructors

### 2. [Thread Pool with Modern C++](Projects/02_Thread_Pool_Modern_CPP.md)
**Focus**: Advanced concurrency and template programming
- Lock-free work stealing queue
- Templated task submission system
- Future-based result handling
- **Key Features**: std::thread, std::future, variadic templates, perfect forwarding

### 3. [Type-Safe Configuration System](Projects/03_Type_Safe_Configuration_System.md)
**Focus**: Template metaprogramming and type safety
- Compile-time configuration validation
- JSON/XML parsing with type safety
- Flexible configuration hierarchy
- **Key Features**: std::variant, template specialization, SFINAE

### 4. [File System Explorer](Projects/04_File_System_Explorer.md)
**Focus**: C++17 filesystem and structured bindings
- Comprehensive file operations wrapper
- Parallel file processing algorithms
- Advanced search and filtering system
- **Key Features**: std::filesystem, structured bindings, std::optional, parallel algorithms

### 5. [Template Metaprogramming Library](Projects/05_Template_Metaprogramming_Library.md)
**Focus**: Advanced template techniques and compile-time programming
- Compile-time algorithms and data structures
- SFINAE patterns and type traits
- Functional programming utilities
- **Key Features**: constexpr, template specialization, variadic templates, type traits

## 📖 Recommended Study Path

### Beginner Path (New to Modern C++)
1. Start with **C++11 Core Language Features** - fundamental concepts
2. Study **Smart Pointers** - essential for memory management
3. Practice with **Move Semantics Implementation** project
4. Learn **C++14 Improvements** - natural evolution
5. Explore **C++17 Core Language Features** - latest conveniences

### Intermediate Path (Some Modern C++ Experience)
1. Review **Move Semantics** in detail - master perfect forwarding
2. Deep dive into **Concurrency Support** - threading fundamentals
3. Tackle **Thread Pool Modern C++** project
4. Study **Type Traits and Metaprogramming** - advanced templates
5. Build **Type-Safe Configuration System** project

### Advanced Path (Experienced with Modern C++)
1. Master **Template Metaprogramming** concepts
2. Explore **C++17 Library Additions** - comprehensive features
3. Complete **File System Explorer** project
4. Build **Template Metaprogramming Library** project
5. Contribute to or review all projects for deeper understanding

## 🔧 Setup and Build Instructions

### Prerequisites
- C++17 compliant compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15 or higher
- Optional: Google Test for running unit tests

### Building Projects
Each project includes its own `CMakeLists.txt`. General build process:

```bash
# Navigate to project directory
cd Projects/01_Move_Semantics_Implementation

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build .

# Run examples (if available)
./basic_usage
./performance_tests
```

### IDE Setup
- **Visual Studio Code**: Install C++ extensions and CMake tools
- **CLion**: Import CMake projects directly
- **Visual Studio**: Use "Open Folder" with CMake projects
- **Qt Creator**: Import CMakeLists.txt files

## 📝 Learning Tips

### Study Techniques
1. **Read First, Code Second**: Understand concepts before implementation
2. **Incremental Learning**: Master one feature before moving to the next
3. **Practice Variations**: Modify examples to test understanding
4. **Performance Testing**: Use benchmarks to understand impact
5. **Code Review**: Compare your solutions with provided examples

### Common Pitfalls to Avoid
- **Move Semantics**: Don't assume moving is always faster
- **Smart Pointers**: Avoid circular references with shared_ptr
- **Templates**: Watch out for compilation time and error message clarity
- **Concurrency**: Always consider data races and deadlocks
- **Auto**: Be careful with proxy types and reference lifetimes

### Best Practices
- Use RAII consistently for resource management
- Prefer const correctness and immutability when possible
- Apply the Rule of Zero/Three/Five appropriately
- Choose the right smart pointer for the job
- Use structured bindings for cleaner code
- Leverage constexpr for compile-time optimization

## 🔗 Additional Resources

### Books
- "Effective Modern C++" by Scott Meyers
- "C++ Concurrency in Action" by Anthony Williams
- "Template Metaprogramming with C++" by Marius Bancila

### Online Resources
- [C++ Reference](https://en.cppreference.com/)
- [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines)
- [Compiler Explorer](https://godbolt.org/) for assembly analysis

### Standards Documentation
- [C++11 Standard](https://www.iso.org/standard/50372.html)
- [C++14 Standard](https://www.iso.org/standard/64029.html)
- [C++17 Standard](https://www.iso.org/standard/68564.html)

## 🎯 Learning Objectives Checklist

After completing this learning path, you should be able to:

### C++11 Mastery
- [ ] Write efficient move constructors and assignment operators
- [ ] Implement custom smart pointer types
- [ ] Create thread-safe concurrent programs
- [ ] Use lambda expressions effectively
- [ ] Apply template metaprogramming techniques

### C++14 Competency
- [ ] Utilize generic lambdas for flexible code
- [ ] Implement variable templates
- [ ] Use return type deduction appropriately

### C++17 Proficiency
- [ ] Apply structured bindings for cleaner code
- [ ] Use constexpr if for conditional compilation
- [ ] Implement variant-based state machines
- [ ] Work with filesystem operations
- [ ] Optimize with string_view and optional

### Advanced Skills
- [ ] Design template libraries with SFINAE
- [ ] Implement lock-free data structures
- [ ] Create compile-time algorithms
- [ ] Build type-safe domain-specific languages
- [ ] Optimize performance with modern C++ features

## 📞 Support and Contributions

If you find errors, have questions, or want to contribute improvements:
1. Review the existing materials thoroughly
2. Check if your question is addressed in the learning tips
3. Experiment with code modifications to understand better
4. Consider contributing additional examples or clarifications

Happy learning! Modern C++ offers powerful tools for writing efficient, safe, and maintainable code. Take your time to master each concept and build strong foundations for advanced C++ development.
