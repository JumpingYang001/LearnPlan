# Functional Programming with Boost

*Duration: 1 week*

## Overview

Functional programming is a programming paradigm that treats computation as the evaluation of mathematical functions and avoids changing-state and mutable data. Boost provides powerful libraries that enable functional programming patterns in C++, bridging the gap between object-oriented and functional paradigms.

This section covers Boost's functional programming utilities, including function wrappers, binding, lambda expressions, and advanced functional programming constructs that were essential before C++11 introduced native lambda support.

### Why Learn Boost Functional Programming?

1. **Historical Context**: Understanding pre-C++11 functional programming techniques
2. **Legacy Code**: Many existing codebases still use Boost functional libraries
3. **Advanced Patterns**: Some Boost libraries offer features beyond standard C++
4. **Interoperability**: Boost functional components work seamlessly together
5. **Performance**: Some Boost implementations may be more optimized for specific use cases

### Functional Programming Core Concepts

```
Functional Programming Principles:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Pure        │  │ Immutable   │  │ Higher-Order        │  │
│  │ Functions   │  │ Data        │  │ Functions           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Function    │  │ Lazy        │  │ Function            │  │
│  │ Composition │  │ Evaluation  │  │ Currying            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Learning Topics

### Boost.Function - Function Wrappers and Type Erasure

**What is Type Erasure?**
Type erasure is a technique that allows you to store objects of different types in the same container by hiding their specific types behind a common interface. `boost::function` is a perfect example of type erasure for callable objects.

```cpp
// Type erasure in action
boost::function<int(int)> func;

func = [](int x) { return x * 2; };        // Lambda
func = std::bind(std::plus<int>(), 42, std::placeholders::_1); // Bound function
func = SomeCallableClass();                 // Function object

// All different types, but stored in the same variable!
```

#### Key Features:
- **Universal Callable Storage**: Store any callable (function, lambda, functor, bound function)
- **Type Safety**: Compile-time type checking for function signatures
- **Performance**: Optimized virtual dispatch mechanism
- **Empty State**: Can be empty (null) and checked for validity
- **Exception Safety**: Strong exception safety guarantees

#### Comparison with std::function:
| Feature | boost::function | std::function (C++11+) |
|---------|-----------------|------------------------|
| **Availability** | Pre-C++11 compatible | C++11 and later |
| **Performance** | Highly optimized | Standard implementation |
| **Small Object Optimization** | Yes | Yes (implementation-defined) |
| **Target Access** | Limited | target() and target_type() |
| **Allocator Support** | Yes | Yes (C++17 removed it) |

### Boost.Bind - Function Composition and Partial Application

**What is Partial Application?**
Partial application is the process of fixing some arguments of a function, producing another function with fewer arguments.

```cpp
// Original function: f(x, y, z)
// Partial application: g(y, z) where x is fixed to some value
auto multiply = [](int a, int b, int c) { return a * b * c; };
auto multiply_by_2 = boost::bind(multiply, 2, _1, _2); // Fix first argument to 2
// Now multiply_by_2(3, 4) == multiply(2, 3, 4) == 24
```

#### Key Concepts:
- **Placeholders**: `_1`, `_2`, `_3`, ... represent argument positions
- **Argument Reordering**: Change the order of arguments
- **Nested Binding**: Bind the result of one bind to another
- **Member Function Binding**: Bind class member functions and data members

#### Comparison with std::bind and Lambdas:
```cpp
// Boost.Bind (Pre-C++11)
auto add_5 = boost::bind(std::plus<int>(), _1, 5);

// std::bind (C++11)
auto add_5_std = std::bind(std::plus<int>(), std::placeholders::_1, 5);

// Lambda (C++11) - Often preferred for readability
auto add_5_lambda = [](int x) { return x + 5; };
```

### Boost.Lambda - Expression Templates and Lazy Evaluation

**What are Expression Templates?**
Expression templates are a C++ template technique used to eliminate temporaries and optimize mathematical expressions by building the entire expression as a type at compile time.

```cpp
// Traditional approach - creates temporaries
vector<int> a, b, c, result;
result = a + b * c;  // Creates temporary for (b * c), then temporary for (a + temp)

// Expression template approach - no temporaries
// The entire expression becomes a single type that evaluates lazily
```

#### Boost.Lambda Features:
- **Lazy Evaluation**: Expressions are built but not evaluated until needed
- **STL Integration**: Works seamlessly with STL algorithms
- **Control Structures**: Support for if-then-else, loops, etc.
- **Variable Binding**: Capture and modify local variables

#### Limitations:
- **Complex Syntax**: Can become hard to read with complex expressions
- **Compile-Time Overhead**: Heavy template instantiation
- **C++11 Superseded**: Modern lambdas are generally preferred

### Boost.Phoenix - Advanced Functional Programming

**What is Phoenix?**
Phoenix is a functional programming library that provides a framework for creating function objects. It's more powerful and flexible than Boost.Lambda, offering:

- **Actor Framework**: Function objects that can be composed and transformed
- **Lazy Evaluation**: Expressions are evaluated only when needed
- **Custom Operators**: Define your own operators for domain-specific languages
- **Statement Support**: Full C++ statement support within lambda expressions

```cpp
// Phoenix vs Lambda comparison
// Boost.Lambda
boost::lambda::_1 * boost::lambda::_1 + boost::lambda::_2

// Boost.Phoenix (more readable)
using namespace boost::phoenix::arg_names;
arg1 * arg1 + arg2
```

## Code Examples and Detailed Explanations

### Boost.Function - Basic Usage and Type Erasure

The power of `boost::function` lies in its ability to store any callable object with a compatible signature, regardless of its actual type. This is called **type erasure**.

```cpp
#include <boost/function.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// Define function types for different operations (type aliases for readability)
typedef boost::function<int(int, int)> BinaryIntOp;      // Takes 2 ints, returns int
typedef boost::function<bool(int)> IntPredicate;         // Takes int, returns bool
typedef boost::function<void(const std::string&)> StringProcessor; // Takes string, returns void

// Regular functions - these have fixed types at compile time
int add(int a, int b) { 
    std::cout << "  [Function] Adding " << a << " + " << b << std::endl;
    return a + b; 
}

int multiply(int a, int b) { 
    std::cout << "  [Function] Multiplying " << a << " * " << b << std::endl;
    return a * b; 
}

bool is_even(int n) { 
    std::cout << "  [Function] Checking if " << n << " is even" << std::endl;
    return n % 2 == 0; 
}

bool is_positive(int n) { 
    std::cout << "  [Function] Checking if " << n << " is positive" << std::endl;
    return n > 0; 
}

// Function object (functor) - has state and behavior
class Accumulator {
public:
    Accumulator(int initial = 0) : sum_(initial) {
        std::cout << "  [Accumulator] Created with initial value: " << initial << std::endl;
    }
    
    // operator() makes this a callable object
    int operator()(int value) {
        sum_ += value;
        std::cout << "  [Accumulator] Added " << value << ", total: " << sum_ << std::endl;
        return sum_;
    }
    
    int get_sum() const { return sum_; }
    
private:
    int sum_;
};

void demonstrate_function_wrappers() {
    std::cout << "=== Boost.Function Type Erasure Demonstration ===\n\n";
    
    // Store different types of callables in the same container
    // This is type erasure - different types (function pointers, lambdas) 
    // are stored behind the same interface
    std::vector<BinaryIntOp> operations;
    
    operations.push_back(add);        // Function pointer
    operations.push_back(multiply);   // Function pointer
    operations.push_back([](int a, int b) { // Lambda function
        std::cout << "  [Lambda] Subtracting " << a << " - " << b << std::endl;
        return a - b; 
    });
    
    // Use the operations polymorphically
    int x = 10, y = 5;
    std::cout << "Applying operations to " << x << " and " << y << ":\n";
    for (size_t i = 0; i < operations.size(); ++i) {
        int result = operations[i](x, y);  // Same interface, different implementations
        std::cout << "Operation " << i << " result: " << result << "\n\n";
    }
    
    // Demonstrate predicates (functions returning bool)
    std::vector<IntPredicate> predicates = { is_even, is_positive };
    std::vector<int> numbers = { -4, -1, 0, 3, 8 };
    
    std::cout << "Testing predicates on numbers: ";
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << "\n\n";
    
    for (size_t i = 0; i < predicates.size(); ++i) {
        std::cout << "Predicate " << i << " results: ";
        for (int num : numbers) {
            bool result = predicates[i](num);
            std::cout << num << ":" << (result ? "T" : "F") << " ";
        }
        std::cout << "\n\n";
    }
    
    // Function object with state
    std::cout << "Using stateful function object (Accumulator):\n";
    Accumulator acc(100);  // Start with 100
    boost::function<int(int)> accumulate_func = acc;  // Type erasure of function object
    
    std::cout << "Accumulating values 1 through 5:\n";
    for (int i = 1; i <= 5; ++i) {
        int result = accumulate_func(i);
        std::cout << "After adding " << i << ": " << result << "\n";
    }
    std::cout << "\n";
}

### Boost.Function - Advanced Features and Practical Applications

Understanding the advanced features of `boost::function` is crucial for building robust callback systems and event-driven architectures.

```cpp
#include <boost/function.hpp>
#include <iostream>
#include <memory>
#include <map>
#include <string>

// Event-driven system example
class EventHandler {
public:
    typedef boost::function<void(const std::string&)> EventCallback;
    
    void set_callback(EventCallback callback) {
        callback_ = callback;
        std::cout << "Callback registered\n";
    }
    
    void trigger_event(const std::string& event_data) {
        if (callback_) {  // Check if callback is valid (not empty)
            std::cout << "Triggering event with data: " << event_data << "\n";
            callback_(event_data);
        } else {
            std::cout << "No callback registered for event: " << event_data << "\n";
        }
    }
    
    bool has_callback() const {
        return static_cast<bool>(callback_);  // Convert to bool (true if not empty)
    }
    
    void clear_callback() {
        callback_ = EventCallback();  // Assign empty function
        std::cout << "Callback cleared\n";
    }
    
private:
    EventCallback callback_;
};

// Different types of loggers
class Logger {
public:
    void log_info(const std::string& message) {
        std::cout << "[INFO] " << message << "\n";
    }
    
    void log_error(const std::string& message) {
        std::cout << "[ERROR] " << message << "\n";
    }
    
    void log_debug(const std::string& message) {
        std::cout << "[DEBUG] " << message << "\n";
    }
};

// Multi-callback event system
class MultiEventHandler {
public:
    typedef boost::function<void(const std::string&)> EventCallback;
    typedef std::map<std::string, EventCallback> CallbackMap;
    
    void register_handler(const std::string& event_type, EventCallback callback) {
        callbacks_[event_type] = callback;
        std::cout << "Handler registered for event type: " << event_type << "\n";
    }
    
    void trigger_event(const std::string& event_type, const std::string& data) {
        auto it = callbacks_.find(event_type);
        if (it != callbacks_.end() && it->second) {
            std::cout << "Triggering " << event_type << " event\n";
            it->second(data);
        } else {
            std::cout << "No handler for event type: " << event_type << "\n";
        }
    }
    
    void remove_handler(const std::string& event_type) {
        callbacks_.erase(event_type);
        std::cout << "Handler removed for event type: " << event_type << "\n";
    }
    
    size_t handler_count() const { return callbacks_.size(); }
    
private:
    CallbackMap callbacks_;
};

void demonstrate_function_advanced() {
    std::cout << "=== Advanced Boost.Function Features ===\n\n";
    
    EventHandler handler;
    Logger logger;
    
    // Test empty function state
    std::cout << "1. Empty Function State:\n";
    std::cout << "Has callback: " << (handler.has_callback() ? "yes" : "no") << "\n";
    handler.trigger_event("Test event 1");
    std::cout << "\n";
    
    // Set member function as callback using lambda
    std::cout << "2. Member Function Callback:\n";
    handler.set_callback([&logger](const std::string& msg) {
        logger.log_info(msg);
    });
    
    std::cout << "Has callback: " << (handler.has_callback() ? "yes" : "no") << "\n";
    handler.trigger_event("Important system message");
    std::cout << "\n";
    
    // Change callback dynamically
    std::cout << "3. Dynamic Callback Change:\n";
    handler.set_callback([&logger](const std::string& msg) {
        logger.log_error(msg);
    });
    
    handler.trigger_event("Error occurred in system");
    std::cout << "\n";
    
    // Clear callback
    std::cout << "4. Callback Clearing:\n";
    handler.clear_callback();
    std::cout << "Has callback: " << (handler.has_callback() ? "yes" : "no") << "\n";
    handler.trigger_event("This won't be processed");
    std::cout << "\n";
    
    // Multi-event handler demonstration
    std::cout << "5. Multi-Event Handler System:\n";
    MultiEventHandler multi_handler;
    
    // Register different handlers for different event types
    multi_handler.register_handler("error", [&logger](const std::string& msg) {
        logger.log_error("System Error: " + msg);
    });
    
    multi_handler.register_handler("info", [&logger](const std::string& msg) {
        logger.log_info("System Info: " + msg);
    });
    
    multi_handler.register_handler("debug", [&logger](const std::string& msg) {
        logger.log_debug("Debug Info: " + msg);
    });
    
    std::cout << "Registered handlers: " << multi_handler.handler_count() << "\n\n";
    
    // Trigger different events
    multi_handler.trigger_event("error", "Database connection failed");
    multi_handler.trigger_event("info", "User logged in successfully");
    multi_handler.trigger_event("debug", "Processing user request");
    multi_handler.trigger_event("warning", "This event type is not registered");
    
    std::cout << "\n";
    
    // Remove a handler
    multi_handler.remove_handler("debug");
    std::cout << "Remaining handlers: " << multi_handler.handler_count() << "\n";
    multi_handler.trigger_event("debug", "This won't be processed");
}

// Function comparison and performance example
void demonstrate_function_comparison() {
    std::cout << "\n=== Function Types Comparison ===\n\n";
    
    // Regular function pointer
    int (*func_ptr)(int, int) = add;
    
    // boost::function
    boost::function<int(int, int)> boost_func = add;
    
    // Function object
    std::plus<int> plus_obj;
    boost::function<int(int, int)> boost_func_obj = plus_obj;
    
    // Lambda
    auto lambda = [](int a, int b) { return a + b; };
    boost::function<int(int, int)> boost_lambda = lambda;
    
    int x = 10, y = 20;
    
    std::cout << "Testing different function types with values " << x << " and " << y << ":\n";
    std::cout << "Function pointer result: " << func_ptr(x, y) << "\n";
    std::cout << "boost::function with regular function: " << boost_func(x, y) << "\n";
    std::cout << "boost::function with function object: " << boost_func_obj(x, y) << "\n";
    std::cout << "boost::function with lambda: " << boost_lambda(x, y) << "\n";
    
    // Demonstrate empty function checking
    boost::function<int(int, int)> empty_func;
    std::cout << "\nEmpty function test:\n";
    std::cout << "Is empty: " << (empty_func ? "no" : "yes") << "\n";
    
    // Trying to call empty function would throw boost::bad_function_call
    try {
        empty_func(1, 2);
    } catch (const boost::bad_function_call& e) {
        std::cout << "Caught exception: " << e.what() << "\n";
    }
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

## Learning Objectives

By the end of this section, you should be able to:

### Core Understanding
- **Explain the concept of type erasure** and how `boost::function` implements it
- **Understand functional programming paradigms** and their benefits in C++
- **Distinguish between different callable types** (functions, lambdas, functors) and when to use each
- **Implement callback systems** using function wrappers effectively

### Practical Skills
- **Create and use boost::function objects** for various callback scenarios
- **Apply function binding and composition** using Boost.Bind
- **Write functional-style code** using Boost.Lambda and Phoenix
- **Build domain-specific languages** using functional programming constructs
- **Optimize functional code** for performance and readability

### Advanced Concepts
- **Design event-driven architectures** using functional programming patterns
- **Implement lazy evaluation** and expression templates
- **Create function pipelines** for data processing
- **Understand migration paths** from Boost to modern C++ functional features

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can:

□ Create a `boost::function` wrapper and explain what type erasure means  
□ Write a callback system that can handle different types of callable objects  
□ Use `boost::bind` to create partial applications and function compositions  
□ Implement a simple functional pipeline for data transformation  
□ Explain the differences between Boost functional libraries and C++11+ features  
□ Debug and optimize functional programming code  
□ Create a domain-specific language using Phoenix expressions  

### Practical Exercises

**Exercise 1: Event System Implementation**
```cpp
// TODO: Create a complete event system with priority queues
class PriorityEventSystem {
    // Implement:
    // - Event registration with priorities
    // - Asynchronous event processing
    // - Event filtering and transformation
    // - Error handling and recovery
};
```

**Exercise 2: Functional Data Processing Pipeline**
```cpp
// TODO: Build a data processing pipeline
template<typename T>
class DataPipeline {
    // Implement:
    // - Chainable transformations
    // - Parallel processing stages
    // - Error propagation
    // - Result accumulation
};
```

**Exercise 3: Mathematical Expression DSL**
```cpp
// TODO: Create a mathematical expression evaluator using Phoenix
// Support: variables, functions, operators, lazy evaluation
// Example: auto expr = _1 * _1 + 2 * _2 - 5;
//          double result = expr(3, 4); // 3*3 + 2*4 - 5 = 12
```

## Performance Considerations and Best Practices

### Function Object Overhead Analysis

Understanding the performance implications of different functional programming approaches is crucial for production code.

```cpp
#include <boost/function.hpp>
#include <chrono>
#include <iostream>
#include <vector>

// Performance test framework
template<typename Func>
double measure_time(Func func, int iterations = 1000000) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // Convert to milliseconds
}

// Different function call approaches
int simple_add(int a, int b) { return a + b; }

class AddFunctor {
public:
    int operator()(int a, int b) const { return a + b; }
};

void performance_comparison() {
    std::cout << "=== Performance Comparison ===\n\n";
    
    const int a = 10, b = 20;
    const int iterations = 10000000;
    
    // 1. Direct function call
    double direct_time = measure_time([&]() {
        volatile int result = simple_add(a, b);
        (void)result; // Prevent optimization
    }, iterations);
    
    // 2. Function pointer
    int (*func_ptr)(int, int) = simple_add;
    double func_ptr_time = measure_time([&]() {
        volatile int result = func_ptr(a, b);
        (void)result;
    }, iterations);
    
    // 3. boost::function with regular function
    boost::function<int(int, int)> boost_func = simple_add;
    double boost_func_time = measure_time([&]() {
        volatile int result = boost_func(a, b);
        (void)result;
    }, iterations);
    
    // 4. boost::function with lambda
    boost::function<int(int, int)> boost_lambda = [](int x, int y) { return x + y; };
    double boost_lambda_time = measure_time([&]() {
        volatile int result = boost_lambda(a, b);
        (void)result;
    }, iterations);
    
    // 5. boost::function with functor
    boost::function<int(int, int)> boost_functor = AddFunctor{};
    double boost_functor_time = measure_time([&]() {
        volatile int result = boost_functor(a, b);
        (void)result;
    }, iterations);
    
    // 6. Direct lambda call
    auto direct_lambda = [](int x, int y) { return x + y; };
    double direct_lambda_time = measure_time([&]() {
        volatile int result = direct_lambda(a, b);
        (void)result;
    }, iterations);
    
    // Results
    std::cout << "Performance Results (" << iterations << " iterations):\n";
    std::cout << "1. Direct function call:     " << direct_time << " ms\n";
    std::cout << "2. Function pointer:         " << func_ptr_time << " ms\n";
    std::cout << "3. boost::function(func):    " << boost_func_time << " ms\n";
    std::cout << "4. boost::function(lambda):  " << boost_lambda_time << " ms\n";
    std::cout << "5. boost::function(functor): " << boost_functor_time << " ms\n";
    std::cout << "6. Direct lambda call:       " << direct_lambda_time << " ms\n\n";
    
    // Calculate relative overhead
    std::cout << "Relative Overhead (compared to direct call):\n";
    std::cout << "Function pointer:        " << (func_ptr_time / direct_time) << "x\n";
    std::cout << "boost::function(func):   " << (boost_func_time / direct_time) << "x\n";
    std::cout << "boost::function(lambda): " << (boost_lambda_time / direct_time) << "x\n";
    std::cout << "boost::function(functor):" << (boost_functor_time / direct_time) << "x\n";
    std::cout << "Direct lambda:           " << (direct_lambda_time / direct_time) << "x\n";
}
```

### Memory Usage and Small Object Optimization

```cpp
#include <boost/function.hpp>
#include <iostream>

void memory_usage_analysis() {
    std::cout << "\n=== Memory Usage Analysis ===\n\n";
    
    // Size comparison
    std::cout << "Object Sizes:\n";
    std::cout << "Function pointer:     " << sizeof(int(*)(int, int)) << " bytes\n";
    std::cout << "boost::function:      " << sizeof(boost::function<int(int, int)>) << " bytes\n";
    std::cout << "Lambda (empty):       " << sizeof([](int, int) { return 0; }) << " bytes\n";
    
    // Small object optimization test
    struct SmallFunctor {
        int operator()(int a, int b) const { return a + b + offset; }
        int offset = 0;
    };
    
    struct LargeFunctor {
        int operator()(int a, int b) const { return a + b; }
        char padding[1000] = {}; // Large object
    };
    
    std::cout << "\nSmall Object Optimization:\n";
    std::cout << "SmallFunctor size:    " << sizeof(SmallFunctor) << " bytes\n";
    std::cout << "LargeFunctor size:    " << sizeof(LargeFunctor) << " bytes\n";
    
    // boost::function will likely use small object optimization for SmallFunctor
    // but will allocate on heap for LargeFunctor
    boost::function<int(int, int)> small_func = SmallFunctor{};
    boost::function<int(int, int)> large_func = LargeFunctor{};
    
    std::cout << "Both stored in boost::function of size: " 
              << sizeof(boost::function<int(int, int)>) << " bytes\n";
    std::cout << "(Large functors may cause heap allocation)\n";
}
```

### Best Practices and Guidelines

```cpp
// BEST PRACTICES EXAMPLES

namespace best_practices {

// ✅ DO: Use specific function types
typedef boost::function<void(const std::string&)> LogCallback;
typedef boost::function<bool(int)> ValidationPredicate;
typedef boost::function<std::string(const std::string&)> StringTransform;

// ✅ DO: Check function validity before calling
void safe_callback_call(const LogCallback& callback, const std::string& message) {
    if (callback) {  // Always check!
        callback(message);
    } else {
        std::cerr << "Warning: Null callback called with: " << message << "\n";
    }
}

// ✅ DO: Use const references for parameters when possible
void process_data(const std::vector<int>& data, 
                  const boost::function<int(int)>& transformer) {
    for (int value : data) {
        if (transformer) {
            std::cout << transformer(value) << " ";
        }
    }
}

// ✅ DO: Provide clear error handling
class CallbackManager {
public:
    void set_error_callback(boost::function<void(const std::string&)> callback) {
        error_callback_ = callback;
    }
    
    void report_error(const std::string& error) {
        if (error_callback_) {
            try {
                error_callback_(error);
            } catch (const std::exception& e) {
                std::cerr << "Error in error callback: " << e.what() << "\n";
            }
        } else {
            std::cerr << "No error callback set for: " << error << "\n";
        }
    }
    
private:
    boost::function<void(const std::string&)> error_callback_;
};

// ❌ DON'T: Use boost::function for simple cases where lambda suffices
// Instead of:
// boost::function<int(int)> simple = [](int x) { return x * 2; };
// Use:
// auto simple = [](int x) { return x * 2; };

// ❌ DON'T: Store expensive-to-copy objects by value in closures
// class HeavyObject { char data[10000]; };
// boost::function<void()> bad = [heavy_obj]() { /* uses heavy_obj */ };
// Better:
// boost::function<void()> good = [&heavy_obj]() { /* uses reference */ };

} // namespace best_practices
```

### Migration Guide: Boost to Modern C++

```cpp
#include <boost/function.hpp>
#include <boost/bind/bind.hpp>
#include <functional>
#include <iostream>

void migration_examples() {
    std::cout << "\n=== Migration from Boost to Modern C++ ===\n\n";
    
    // Function wrappers
    std::cout << "1. Function Wrappers:\n";
    
    // Boost way
    boost::function<int(int)> boost_func = [](int x) { return x * 2; };
    std::cout << "Boost: " << boost_func(5) << "\n";
    
    // Modern C++ way
    std::function<int(int)> std_func = [](int x) { return x * 2; };
    std::cout << "Modern: " << std_func(5) << "\n\n";
    
    // Function binding
    std::cout << "2. Function Binding:\n";
    
    auto multiply = [](int a, int b, int c) { return a * b * c; };
    
    // Boost way
    using namespace boost::placeholders;
    auto boost_bound = boost::bind(multiply, _1, 2, _2);
    std::cout << "Boost bind: " << boost_bound(3, 4) << "\n";
    
    // Modern C++ ways
    auto std_bound = std::bind(multiply, std::placeholders::_1, 2, std::placeholders::_2);
    std::cout << "std::bind: " << std_bound(3, 4) << "\n";
    
    // Better: Use lambda (more readable)
    auto lambda_bound = [&multiply](int a, int c) { return multiply(a, 2, c); };
    std::cout << "Lambda: " << lambda_bound(3, 4) << "\n\n";
    
    // Recommendations
    std::cout << "Migration Recommendations:\n";
    std::cout << "• boost::function -> std::function (C++11+)\n";
    std::cout << "• boost::bind -> std::bind or lambda (C++11+)\n";
    std::cout << "• boost::lambda -> C++ lambda expressions (C++11+)\n";
    std::cout << "• Keep Boost.Phoenix for complex DSL scenarios\n";
}
```

## Study Materials and Resources

### Recommended Reading

**Primary Sources:**
- **"Boost C++ Libraries"** by Boris Schäling - Chapters on Functional Programming
- **"C++ Template Metaprogramming"** by David Abrahams and Aleksey Gurtovoy - Expression Templates
- **"Functional Programming in C++"** by Ivan Čukić - Modern Functional Programming Concepts

**Online Resources:**
- [Boost.Function Documentation](https://www.boost.org/doc/libs/1_82_0/doc/html/function.html) - Official documentation
- [Boost.Bind Documentation](https://www.boost.org/doc/libs/1_82_0/libs/bind/doc/html/index.html) - Complete reference
- [Boost.Phoenix Documentation](https://www.boost.org/doc/libs/1_82_0/libs/phoenix/doc/html/index.html) - Advanced functional programming

**Academic Papers:**
- "Expression Templates" by Todd Veldhuizen - Foundational paper on expression templates
- "Functional Programming with C++ Templates" - Template metaprogramming techniques

### Video Resources

- **"Functional Programming in C++"** - CppCon talks on functional programming
- **"Expression Templates and Lazy Evaluation"** - Template metaprogramming presentations
- **"Boost Libraries Deep Dive"** - Comprehensive Boost library tutorials

### Hands-on Labs and Projects

**Lab 1: Event-Driven Calculator**
```cpp
// Build a calculator with event-driven architecture
class Calculator {
    // Events: number_entered, operator_selected, equals_pressed
    // Use boost::function for event handlers
    // Implement undo/redo functionality
};
```

**Lab 2: Functional Data Processing Library**
```cpp
// Create a data processing library with functional approach
template<typename T>
class DataProcessor {
    // Chain operations: filter, map, reduce, sort
    // Use boost::function for transformations
    // Support parallel processing
};
```

**Lab 3: Domain-Specific Language (DSL)**
```cpp
// Build a query language using Boost.Phoenix
// Example: auto query = where(_1 > 100 && _2 == "active")
//          auto results = database.query(users, query);
```

### Practice Questions and Challenges

**Conceptual Questions:**
1. What is type erasure and how does `boost::function` implement it?
2. Explain the difference between partial application and currying.
3. What are expression templates and why are they useful?
4. How does lazy evaluation work in functional programming?
5. When would you choose Boost.Phoenix over simple lambdas?

**Technical Challenges:**
6. Implement a thread-safe callback system using `boost::function`
7. Create a functional pipeline that can be parallelized
8. Design a caching mechanism using function composition
9. Build a retry mechanism for function calls with exponential backoff
10. Implement a simple functional reactive programming system

**Coding Exercises:**
```cpp
// Challenge 1: Implement a functional Option/Maybe type
template<typename T>
class Optional {
    // Implement: map, flatMap, filter, orElse
    // Use boost::function for transformations
};

// Challenge 2: Create a functional Either type for error handling
template<typename L, typename R>
class Either {
    // Implement: map, mapLeft, fold, isLeft, isRight
    // Use functional composition for error propagation
};

// Challenge 3: Build a functional parser combinator library
template<typename T>
class Parser {
    // Implement: sequence, choice, many, optional
    // Use Boost.Phoenix for complex parsing rules
};
```

### Development Environment Setup

**Required Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get install libboost-dev libboost-system-dev

# Using vcpkg (Windows/Linux/macOS)
vcpkg install boost-function boost-bind boost-lambda boost-phoenix

# Using conan
conan install boost/1.82.0@
```

**CMake Configuration:**
```cmake
cmake_minimum_required(VERSION 3.10)
project(FunctionalProgramming)

find_package(Boost REQUIRED COMPONENTS system)

add_executable(functional_demo main.cpp)
target_link_libraries(functional_demo ${Boost_LIBRARIES})
target_include_directories(functional_demo PRIVATE ${Boost_INCLUDE_DIRS})

# Enable C++11 for modern features comparison
set_property(TARGET functional_demo PROPERTY CXX_STANDARD 11)
```

**Compilation Commands:**
```bash
# Basic compilation
g++ -std=c++11 -I/path/to/boost -o program program.cpp

# With optimization
g++ -std=c++11 -O3 -DNDEBUG -I/path/to/boost -o program program.cpp

# Debug build
g++ -std=c++11 -g -DDEBUG -I/path/to/boost -o program program.cpp

# With sanitizers
g++ -std=c++11 -fsanitize=address -fsanitize=undefined -I/path/to/boost -o program program.cpp
```

### Debugging and Profiling

**Debugging Functional Code:**
```cpp
// Add debug output to function objects
class DebugFunction {
    boost::function<int(int)> func_;
    std::string name_;
public:
    DebugFunction(boost::function<int(int)> f, const std::string& n) 
        : func_(f), name_(n) {}
    
    int operator()(int x) {
        std::cout << "Calling " << name_ << " with " << x << std::endl;
        int result = func_(x);
        std::cout << "Result: " << result << std::endl;
        return result;
    }
};
```

**Performance Profiling:**
```bash
# Use perf for performance analysis
perf record ./functional_program
perf report

# Use valgrind for memory analysis
valgrind --tool=memcheck --leak-check=full ./functional_program

# Use gprof for call graph analysis
g++ -pg -o program program.cpp
./program
gprof program gmon.out > analysis.txt
```

### Testing Functional Code

**Unit Testing Framework:**
```cpp
#include <boost/test/unit_test.hpp>
#include <boost/function.hpp>

BOOST_AUTO_TEST_SUITE(FunctionalTests)

BOOST_AUTO_TEST_CASE(TestFunctionWrapper) {
    boost::function<int(int)> func = [](int x) { return x * 2; };
    
    BOOST_CHECK_EQUAL(func(5), 10);
    BOOST_CHECK(func); // Not empty
    
    func = boost::function<int(int)>(); // Clear
    BOOST_CHECK(!func); // Empty
}

BOOST_AUTO_TEST_CASE(TestFunctionComposition) {
    auto add_5 = [](int x) { return x + 5; };
    auto multiply_2 = [](int x) { return x * 2; };
    
    // Compose functions
    auto composed = [=](int x) { return multiply_2(add_5(x)); };
    
    BOOST_CHECK_EQUAL(composed(3), 16); // (3 + 5) * 2 = 16
}

BOOST_AUTO_TEST_SUITE_END()
```

## Assessment and Knowledge Validation

### Self-Assessment Quiz

**Multiple Choice Questions:**

1. **What is the primary purpose of `boost::function`?**
   - a) To create lambda expressions
   - b) To provide type erasure for callable objects
   - c) To optimize function calls
   - d) To implement template metaprogramming

2. **Which of the following demonstrates partial application with `boost::bind`?**
   - a) `boost::bind(func, _1, _2)`
   - b) `boost::bind(func, 42, _1)`
   - c) `boost::bind(func)`
   - d) `boost::bind(_1, func)`

3. **What happens when you call an empty `boost::function`?**
   - a) Returns a default value
   - b) Compiles but does nothing
   - c) Throws `boost::bad_function_call`
   - d) Causes undefined behavior

**True/False Questions:**

4. `boost::function` has zero overhead compared to function pointers. (T/F)
5. Expression templates in Boost.Lambda eliminate temporary objects. (T/F)
6. Boost.Phoenix is more powerful than Boost.Lambda for complex expressions. (T/F)
7. Modern C++ lambdas completely replace all Boost functional libraries. (T/F)

**Practical Programming Challenges:**

**Challenge 1: Event System Design**
Design and implement a complete event system that demonstrates mastery of `boost::function`:

```cpp
// Requirements:
// - Support multiple event types
// - Allow multiple handlers per event
// - Implement handler priorities
// - Provide thread-safety
// - Support event filtering
// - Handle errors gracefully

class EventSystem {
public:
    typedef boost::function<void(const EventData&)> EventHandler;
    
    // Your implementation here
    void subscribe(const std::string& event_type, EventHandler handler, int priority = 0);
    void unsubscribe(const std::string& event_type, int handler_id);
    void publish(const std::string& event_type, const EventData& data);
    void add_filter(const std::string& event_type, boost::function<bool(const EventData&)> filter);
};
```

**Challenge 2: Functional Data Pipeline**
Create a functional data processing pipeline:

```cpp
// Requirements:
// - Chain multiple transformations
// - Support parallel processing
// - Implement lazy evaluation
// - Handle errors functionally
// - Provide performance metrics

template<typename T>
class Pipeline {
public:
    // Your implementation here
    template<typename F> Pipeline</* result type */> map(F func);
    template<typename F> Pipeline<T> filter(F predicate);
    template<typename F, typename U> U reduce(F func, U initial);
    Pipeline<T> parallel(size_t thread_count = std::thread::hardware_concurrency());
    
    std::vector<T> execute();
};
```

**Challenge 3: DSL Implementation**
Build a domain-specific language using Boost.Phoenix:

```cpp
// Create a query language for data structures
// Example usage:
// auto query = where(age_ > 25 && status_ == "active");
// auto results = database.query(users, query);

// Requirements:
// - Support comparison operators
// - Support logical operators (&&, ||, !)
// - Support field access
// - Compile-time type checking
// - Runtime evaluation
```

### Project-Based Assessment

**Capstone Project: Functional Web Server**

Build a simple HTTP server using functional programming principles:

```cpp
class FunctionalWebServer {
    typedef boost::function<Response(const Request&)> RequestHandler;
    typedef boost::function<void(const std::string&)> Logger;
    typedef boost::function<bool(const Request&)> RequestFilter;
    
public:
    // Route registration with functional handlers
    void route(const std::string& path, const std::string& method, RequestHandler handler);
    
    // Middleware support
    void use(RequestFilter filter);
    void use(boost::function<Request(const Request&)> transformer);
    
    // Functional composition for complex handlers
    RequestHandler compose(const std::vector<RequestHandler>& handlers);
    
    // Error handling
    void set_error_handler(boost::function<Response(const std::exception&)> handler);
    
    // Logging with functional approach
    void set_logger(Logger logger);
    
    void start(int port);
};

// Example usage:
server.route("/api/users", "GET", [](const Request& req) -> Response {
    return json_response(get_all_users());
});

server.use([](const Request& req) -> bool { 
    return authenticate(req); 
});

server.set_error_handler([](const std::exception& e) -> Response {
    return error_response(500, e.what());
});
```

### Grading Rubric

**Knowledge Understanding (25%)**
- Explains type erasure and its benefits
- Understands functional programming concepts
- Compares Boost vs modern C++ features
- Demonstrates awareness of performance implications

**Implementation Skills (35%)**
- Creates working `boost::function` implementations
- Uses binding and composition effectively
- Implements error handling properly
- Writes clean, maintainable functional code

**Design and Architecture (25%)**
- Designs appropriate functional abstractions
- Uses composition over inheritance
- Creates reusable functional components
- Applies appropriate design patterns

**Testing and Documentation (15%)**
- Writes comprehensive unit tests
- Documents API clearly
- Provides usage examples
- Includes performance considerations

### Common Mistakes and Solutions

**Mistake 1: Not checking for empty functions**
```cpp
// ❌ Wrong
boost::function<void()> callback;
callback(); // Throws exception!

// ✅ Correct
if (callback) {
    callback();
}
```

**Mistake 2: Expensive copying in closures**
```cpp
// ❌ Wrong - copies large object
LargeObject obj;
boost::function<void()> func = [obj]() { /* use obj */ };

// ✅ Correct - uses reference
boost::function<void()> func = [&obj]() { /* use obj */ };
```

**Mistake 3: Binding with temporary objects**
```cpp
// ❌ Wrong - temporary string destroyed
auto func = boost::bind(process_string, std::string("temp"), _1);

// ✅ Correct - proper lifetime management
std::string str = "persistent";
auto func = boost::bind(process_string, str, _1);
```

**Next Steps After Mastery:**
1. Explore advanced template metaprogramming
2. Study functional reactive programming (FRP)
3. Learn about monad patterns in C++
4. Investigate parallel functional programming
5. Examine functional programming in other languages for broader perspective

## Next Steps

Move on to [Generic Programming Utilities](09_Generic_Programming_Utilities.md) to explore Boost's metaprogramming capabilities.
