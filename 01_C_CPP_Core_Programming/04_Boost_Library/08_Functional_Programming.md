# Functional Programming

*Duration: 1 week*

## Overview

This section covers Boost's functional programming utilities, including function wrappers, binding, lambda expressions, and advanced functional programming constructs.

## Learning Topics

### Boost.Function
- Function wrappers and type erasure
- Comparison with std::function
- Performance considerations
- Integration with other Boost libraries

### Boost.Bind
- Function composition and partial application
- Placeholder arguments and binding strategies
- Comparison with std::bind and lambdas

### Boost.Lambda
- Lambda expressions (pre-C++11)
- Expression templates and lazy evaluation
- Integration with STL algorithms

### Boost.Phoenix
- Advanced functional programming
- Actor framework and expression composition
- Custom operators and lazy evaluation

## Code Examples

### Boost.Function - Basic Usage
```cpp
#include <boost/function.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// Function types for different operations
typedef boost::function<int(int, int)> BinaryIntOp;
typedef boost::function<bool(int)> IntPredicate;
typedef boost::function<void(const std::string&)> StringProcessor;

// Regular functions
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
bool is_even(int n) { return n % 2 == 0; }
bool is_positive(int n) { return n > 0; }

// Function object
class Accumulator {
public:
    Accumulator(int initial = 0) : sum_(initial) {}
    
    int operator()(int value) {
        sum_ += value;
        return sum_;
    }
    
    int get_sum() const { return sum_; }
    
private:
    int sum_;
};

void demonstrate_function_wrappers() {
    // Store different types of callables
    std::vector<BinaryIntOp> operations;
    operations.push_back(add);
    operations.push_back(multiply);
    operations.push_back([](int a, int b) { return a - b; }); // Lambda
    
    // Use the operations
    int x = 10, y = 5;
    for (size_t i = 0; i < operations.size(); ++i) {
        int result = operations[i](x, y);
        std::cout << "Operation " << i << ": " << x << " op " << y 
                  << " = " << result << "\n";
    }
    
    // Predicates
    std::vector<IntPredicate> predicates = { is_even, is_positive };
    std::vector<int> numbers = { -4, -1, 0, 3, 8 };
    
    for (size_t i = 0; i < predicates.size(); ++i) {
        std::cout << "Predicate " << i << " results: ";
        for (int num : numbers) {
            std::cout << num << ":" << (predicates[i](num) ? "T" : "F") << " ";
        }
        std::cout << "\n";
    }
    
    // Function object
    Accumulator acc;
    boost::function<int(int)> accumulate_func = acc;
    
    std::cout << "Accumulating: ";
    for (int i = 1; i <= 5; ++i) {
        int result = accumulate_func(i);
        std::cout << result << " ";
    }
    std::cout << "\n";
}
```

### Boost.Function - Advanced Features
```cpp
#include <boost/function.hpp>
#include <iostream>
#include <memory>

class EventHandler {
public:
    typedef boost::function<void(const std::string&)> EventCallback;
    
    void set_callback(EventCallback callback) {
        callback_ = callback;
    }
    
    void trigger_event(const std::string& event_data) {
        if (callback_) {
            callback_(event_data);
        } else {
            std::cout << "No callback registered\n";
        }
    }
    
    bool has_callback() const {
        return static_cast<bool>(callback_);
    }
    
private:
    EventCallback callback_;
};

class Logger {
public:
    void log_info(const std::string& message) {
        std::cout << "[INFO] " << message << "\n";
    }
    
    void log_error(const std::string& message) {
        std::cout << "[ERROR] " << message << "\n";
    }
};

void demonstrate_function_advanced() {
    EventHandler handler;
    Logger logger;
    
    // Test empty function
    std::cout << "Has callback: " << handler.has_callback() << "\n";
    handler.trigger_event("Test event 1");
    
    // Set member function as callback
    handler.set_callback([&logger](const std::string& msg) {
        logger.log_info(msg);
    });
    
    std::cout << "Has callback: " << handler.has_callback() << "\n";
    handler.trigger_event("Test event 2");
    
    // Change callback
    handler.set_callback([&logger](const std::string& msg) {
        logger.log_error(msg);
    });
    
    handler.trigger_event("Test event 3");
    
    // Clear callback
    handler.set_callback(EventHandler::EventCallback());
    std::cout << "Has callback: " << handler.has_callback() << "\n";
    handler.trigger_event("Test event 4");
}
```

### Boost.Bind - Function Composition
```cpp
#include <boost/bind/bind.hpp>
#include <boost/function.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace boost::placeholders;

class Calculator {
public:
    int add(int a, int b) const { return a + b; }
    int multiply(int a, int b) const { return a * b; }
    void set_value(int value) { value_ = value; }
    int get_value() const { return value_; }
    
private:
    int value_ = 0;
};

// Free functions for demonstration
bool is_greater_than(int value, int threshold) {
    return value > threshold;
}

void print_with_prefix(const std::string& prefix, int value) {
    std::cout << prefix << value << "\n";
}

void demonstrate_bind_basics() {
    Calculator calc;
    
    // Bind member functions
    auto add_5 = boost::bind(&Calculator::add, &calc, _1, 5);
    auto multiply_by_3 = boost::bind(&Calculator::multiply, &calc, _1, 3);
    
    std::cout << "add_5(10) = " << add_5(10) << "\n";
    std::cout << "multiply_by_3(7) = " << multiply_by_3(7) << "\n";
    
    // Bind with specific values
    auto add_10_20 = boost::bind(&Calculator::add, &calc, 10, 20);
    std::cout << "add_10_20() = " << add_10_20() << "\n";
    
    // Bind free functions
    auto is_greater_than_5 = boost::bind(is_greater_than, _1, 5);
    auto print_number = boost::bind(print_with_prefix, "Number: ", _1);
    
    std::vector<int> numbers = { 1, 3, 6, 8, 4, 9 };
    
    std::cout << "Numbers greater than 5:\n";
    for (int num : numbers) {
        if (is_greater_than_5(num)) {
            print_number(num);
        }
    }
    
    // Use with STL algorithms
    std::cout << "Count of numbers > 5: " 
              << std::count_if(numbers.begin(), numbers.end(), is_greater_than_5) 
              << "\n";
}

void demonstrate_bind_advanced() {
    Calculator calc;
    
    // Bind with argument reordering
    auto subtract = [](int a, int b) { return a - b; };
    auto subtract_from_100 = boost::bind(subtract, 100, _1);
    auto reverse_subtract = boost::bind(subtract, _2, _1);
    
    std::cout << "subtract_from_100(30) = " << subtract_from_100(30) << "\n";
    std::cout << "reverse_subtract(5, 15) = " << reverse_subtract(5, 15) << "\n";
    
    // Nested binding
    auto add_then_multiply = boost::bind(
        &Calculator::multiply, &calc,
        boost::bind(&Calculator::add, &calc, _1, _2),
        _3
    );
    
    std::cout << "add_then_multiply(2, 3, 4) = " 
              << add_then_multiply(2, 3, 4) << "\n"; // (2+3)*4 = 20
    
    // Binding with member variables (through getters/setters)
    calc.set_value(42);
    auto get_calc_value = boost::bind(&Calculator::get_value, &calc);
    auto set_calc_value = boost::bind(&Calculator::set_value, &calc, _1);
    
    std::cout << "Current calculator value: " << get_calc_value() << "\n";
    set_calc_value(100);
    std::cout << "New calculator value: " << get_calc_value() << "\n";
}
```

### Boost.Lambda - Lazy Evaluation
```cpp
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/if.hpp>
#include <boost/lambda/algorithm.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

namespace bl = boost::lambda;

void demonstrate_lambda_basics() {
    std::vector<int> numbers = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    
    // Basic lambda expressions
    std::cout << "Original numbers: ";
    std::for_each(numbers.begin(), numbers.end(), 
                  std::cout << bl::_1 << " ");
    std::cout << "\n";
    
    // Transform with lambda
    std::vector<int> squared(numbers.size());
    std::transform(numbers.begin(), numbers.end(), squared.begin(),
                   bl::_1 * bl::_1);
    
    std::cout << "Squared numbers: ";
    std::for_each(squared.begin(), squared.end(),
                  std::cout << bl::_1 << " ");
    std::cout << "\n";
    
    // Count with predicate
    int even_count = std::count_if(numbers.begin(), numbers.end(),
                                   bl::_1 % 2 == 0);
    std::cout << "Even numbers count: " << even_count << "\n";
    
    // Find with complex predicate
    auto it = std::find_if(numbers.begin(), numbers.end(),
                          bl::_1 > 5 && bl::_1 < 8);
    if (it != numbers.end()) {
        std::cout << "Found number between 5 and 8: " << *it << "\n";
    }
}

void demonstrate_lambda_advanced() {
    std::vector<int> numbers = { -3, -1, 0, 2, 5, -7, 8, -4 };
    
    // Conditional operations
    std::cout << "Conditional output:\n";
    std::for_each(numbers.begin(), numbers.end(),
        bl::if_then_else(
            bl::_1 >= 0,
            std::cout << "Positive: " << bl::_1 << "\\n",
            std::cout << "Negative: " << bl::_1 << "\\n"
        )
    );
    
    // Complex transformations
    std::vector<int> processed(numbers.size());
    std::transform(numbers.begin(), numbers.end(), processed.begin(),
        bl::if_then_else_return(
            bl::_1 >= 0,
            bl::_1 * 2,        // Double positive numbers
            bl::_1 * bl::_1    // Square negative numbers
        )
    );
    
    std::cout << "Processed numbers: ";
    std::for_each(processed.begin(), processed.end(),
                  std::cout << bl::_1 << " ");
    std::cout << "\n";
    
    // Accumulate with lambda
    int sum = 0;
    std::for_each(numbers.begin(), numbers.end(),
                  bl::var(sum) += bl::_1);
    std::cout << "Sum using lambda: " << sum << "\n";
}
```

### Boost.Phoenix - Advanced Functional Programming
```cpp
#include <boost/phoenix.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

namespace phx = boost::phoenix;
using namespace phx::arg_names;

class Person {
public:
    Person(const std::string& name, int age) : name_(name), age_(age) {}
    
    const std::string& name() const { return name_; }
    int age() const { return age_; }
    void set_age(int age) { age_ = age; }
    
    bool operator<(const Person& other) const {
        return age_ < other.age_;
    }
    
private:
    std::string name_;
    int age_;
};

std::ostream& operator<<(std::ostream& os, const Person& p) {
    return os << p.name() << " (" << p.age() << ")";
}

void demonstrate_phoenix_basics() {
    std::vector<int> numbers = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    
    // Phoenix lambda expressions
    std::cout << "Numbers: ";
    std::for_each(numbers.begin(), numbers.end(),
                  std::cout << arg1 << " ");
    std::cout << "\n";
    
    // Complex predicates
    auto is_even_and_gt_5 = (arg1 % 2 == 0) && (arg1 > 5);
    
    std::cout << "Even numbers > 5: ";
    std::for_each(numbers.begin(), numbers.end(),
        phx::if_(is_even_and_gt_5)[
            std::cout << arg1 << " "
        ]
    );
    std::cout << "\n";
    
    // Transformations
    std::vector<int> transformed(numbers.size());
    std::transform(numbers.begin(), numbers.end(), transformed.begin(),
                   phx::if_else(arg1 % 2 == 0, arg1 * arg1, arg1 * 2));
    
    std::cout << "Transformed (square evens, double odds): ";
    std::for_each(transformed.begin(), transformed.end(),
                  std::cout << arg1 << " ");
    std::cout << "\n";
}

void demonstrate_phoenix_advanced() {
    std::vector<Person> people = {
        Person("Alice", 25),
        Person("Bob", 30),
        Person("Charlie", 20),
        Person("Diana", 35)
    };
    
    std::cout << "Original people:\n";
    std::for_each(people.begin(), people.end(),
                  std::cout << arg1 << "\\n");
    
    // Sort by age (descending)
    std::sort(people.begin(), people.end(),
              phx::bind(&Person::age, arg1) > phx::bind(&Person::age, arg2));
    
    std::cout << "\nSorted by age (descending):\n";
    std::for_each(people.begin(), people.end(),
                  std::cout << arg1 << "\\n");
    
    // Find person by age
    auto it = std::find_if(people.begin(), people.end(),
                          phx::bind(&Person::age, arg1) == 30);
    if (it != people.end()) {
        std::cout << "\nFound person aged 30: " << *it << "\n";
    }
    
    // Count people in age range
    int adults = std::count_if(people.begin(), people.end(),
                              phx::bind(&Person::age, arg1) >= 25);
    std::cout << "Adults (25+): " << adults << "\n";
    
    // Modify ages
    std::for_each(people.begin(), people.end(),
        phx::if_(phx::bind(&Person::age, arg1) < 30)[
            phx::bind(&Person::set_age, arg1, 
                     phx::bind(&Person::age, arg1) + 1)
        ]
    );
    
    std::cout << "\nAfter aging people under 30:\n";
    std::for_each(people.begin(), people.end(),
                  std::cout << arg1 << "\\n");
}
```

### Custom Function Composition
```cpp
#include <boost/function.hpp>
#include <boost/bind/bind.hpp>
#include <iostream>
#include <vector>

using namespace boost::placeholders;

template<typename F, typename G>
class FunctionComposition {
public:
    FunctionComposition(F f, G g) : f_(f), g_(g) {}
    
    template<typename T>
    auto operator()(T&& arg) -> decltype(f_(g_(std::forward<T>(arg)))) {
        return f_(g_(std::forward<T>(arg)));
    }
    
private:
    F f_;
    G g_;
};

template<typename F, typename G>
FunctionComposition<F, G> compose(F f, G g) {
    return FunctionComposition<F, G>(f, g);
}

// Helper functions for demonstration
int double_value(int x) { return x * 2; }
int add_ten(int x) { return x + 10; }
bool is_greater_than_20(int x) { return x > 20; }

void demonstrate_function_composition() {
    // Compose functions manually
    auto double_then_add_ten = compose(add_ten, double_value);
    auto process_and_check = compose(is_greater_than_20, double_then_add_ten);
    
    std::vector<int> numbers = { 3, 5, 8, 12, 15 };
    
    std::cout << "Function composition results:\n";
    for (int num : numbers) {
        int doubled = double_value(num);
        int processed = double_then_add_ten(num);
        bool check = process_and_check(num);
        
        std::cout << num << " -> double -> " << doubled 
                  << " -> add 10 -> " << processed
                  << " -> > 20? " << (check ? "yes" : "no") << "\n";
    }
    
    // Using boost::bind for composition
    auto bind_composition = boost::bind(
        add_ten,
        boost::bind(double_value, _1)
    );
    
    std::cout << "\nUsing boost::bind composition:\n";
    for (int num : numbers) {
        std::cout << num << " -> " << bind_composition(num) << "\n";
    }
}
```

### Functional Pipeline Pattern
```cpp
#include <boost/function.hpp>
#include <boost/bind/bind.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace boost::placeholders;

template<typename T>
class Pipeline {
public:
    Pipeline(T value) : value_(value) {}
    
    template<typename F>
    auto then(F func) -> Pipeline<decltype(func(value_))> {
        return Pipeline<decltype(func(value_))>(func(value_));
    }
    
    T get() const { return value_; }
    
private:
    T value_;
};

template<typename T>
Pipeline<T> make_pipeline(T value) {
    return Pipeline<T>(value);
}

// Pipeline operations
std::string to_upper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

std::string add_prefix(const std::string& str) {
    return ">>> " + str;
}

std::string add_suffix(const std::string& str) {
    return str + " <<<";
}

int string_length(const std::string& str) {
    return static_cast<int>(str.length());
}

std::string int_to_string(int value) {
    return std::to_string(value);
}

void demonstrate_pipeline() {
    std::string input = "hello world";
    
    // Functional pipeline
    auto result = make_pipeline(input)
        .then(to_upper)
        .then(add_prefix)
        .then(add_suffix)
        .get();
    
    std::cout << "Pipeline result: " << result << "\n";
    
    // Pipeline with type transformation
    auto length_result = make_pipeline(input)
        .then(to_upper)
        .then(string_length)
        .then(boost::bind(std::multiplies<int>(), _1, 2))
        .then(int_to_string)
        .then(add_prefix)
        .get();
    
    std::cout << "Length pipeline result: " << length_result << "\n";
    
    // Multiple inputs through pipeline
    std::vector<std::string> inputs = { "hello", "world", "boost", "functional" };
    
    std::cout << "Processing multiple inputs:\n";
    for (const auto& inp : inputs) {
        auto processed = make_pipeline(inp)
            .then(to_upper)
            .then(add_prefix)
            .get();
        std::cout << "  " << inp << " -> " << processed << "\n";
    }
}
```

## Practical Exercises

1. **Event System Implementation**
   - Create a type-safe event system using Boost.Function
   - Support multiple event handlers and priority ordering
   - Implement event filtering and transformation

2. **Functional DSL**
   - Build a domain-specific language using Boost.Phoenix
   - Create custom operators and expressions
   - Implement lazy evaluation and optimization

3. **Pipeline Framework**
   - Design a data processing pipeline framework
   - Support parallel processing stages
   - Add error handling and recovery mechanisms

4. **Expression Evaluator**
   - Create a mathematical expression evaluator
   - Use functional composition for operation chaining
   - Support variables and function definitions

## Performance Considerations

### Function Object Overhead
- Virtual function call costs in boost::function
- Template instantiation and code bloat
- Memory allocation for closure state

### Binding Performance
- Argument copying vs reference binding
- Temporary object creation
- Optimization opportunities with modern C++

### Lambda Expression Efficiency
- Capture strategies and memory usage
- Template instantiation costs
- Comparison with hand-written functors

## Best Practices

1. **Function Wrapper Usage**
   - Prefer std::function for new code when available
   - Use boost::function for compatibility
   - Consider performance implications of type erasure

2. **Binding Strategies**
   - Use lambdas instead of bind for new code
   - Be careful with reference vs value capture
   - Understand argument evaluation timing

3. **Functional Design**
   - Favor immutable data and pure functions
   - Use composition over inheritance
   - Design for testability and reusability

## Migration to Modern C++

### C++11 and Later Features
```cpp
// Boost.Function -> std::function
boost::function<int(int)> bf = [](int x) { return x * 2; };
std::function<int(int)> sf = [](int x) { return x * 2; };

// Boost.Bind -> std::bind or lambdas
auto bound = boost::bind(func, _1, 42);
auto std_bound = std::bind(func, std::placeholders::_1, 42);
auto lambda = [](int x) { return func(x, 42); };

// Boost.Lambda -> C++ lambdas
boost::lambda::_1 * boost::lambda::_1
[](int x) { return x * x; }
```

## Assessment

- Can use function wrappers effectively for callback systems
- Understands functional composition and binding patterns
- Can implement domain-specific languages with functional constructs
- Knows when to use Boost vs modern C++ functional features

## Next Steps

Move on to [Generic Programming Utilities](09_Generic_Programming_Utilities.md) to explore Boost's metaprogramming capabilities.
