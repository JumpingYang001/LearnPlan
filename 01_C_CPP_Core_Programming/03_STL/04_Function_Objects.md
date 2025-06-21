# Function Objects (Functors)

*Part of STL Learning Track - 1 week*

## Overview

Function objects (functors) are objects that can be called as if they were functions. They provide a way to customize the behavior of STL algorithms and containers. C++ provides several types of function objects: predefined function objects, custom function objects, std::function, std::bind, and lambda expressions.

## Predefined Function Objects

### Arithmetic Function Objects
```cpp
#include <functional>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

void arithmetic_functors_examples() {
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = {10, 20, 30, 40, 50};
    std::vector<int> result(vec1.size());
    
    // std::plus - addition
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), 
                   std::plus<int>());
    
    std::cout << "Addition result: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // std::minus - subtraction
    std::transform(vec2.begin(), vec2.end(), vec1.begin(), result.begin(), 
                   std::minus<int>());
    
    std::cout << "Subtraction result: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // std::multiplies - multiplication
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), 
                   std::multiplies<int>());
    
    std::cout << "Multiplication result: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // std::divides - division
    std::transform(vec2.begin(), vec2.end(), vec1.begin(), result.begin(), 
                   std::divides<int>());
    
    std::cout << "Division result: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // std::modulus - modulo
    std::transform(vec2.begin(), vec2.end(), vec1.begin(), result.begin(), 
                   std::modulus<int>());
    
    std::cout << "Modulo result: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // std::negate - negation (unary)
    std::transform(vec1.begin(), vec1.end(), result.begin(), 
                   std::negate<int>());
    
    std::cout << "Negation result: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### Comparison Function Objects
```cpp
#include <functional>
#include <vector>
#include <algorithm>
#include <iostream>

void comparison_functors_examples() {
    std::vector<int> vec = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    // Sort in ascending order (default)
    std::vector<int> asc_vec = vec;
    std::sort(asc_vec.begin(), asc_vec.end(), std::less<int>());
    
    std::cout << "Ascending order: ";
    for (const auto& item : asc_vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Sort in descending order
    std::vector<int> desc_vec = vec;
    std::sort(desc_vec.begin(), desc_vec.end(), std::greater<int>());
    
    std::cout << "Descending order: ";
    for (const auto& item : desc_vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Other comparison functors
    std::cout << "Comparison results:" << std::endl;
    std::cout << "5 < 3: " << std::less<int>()(5, 3) << std::endl;
    std::cout << "5 > 3: " << std::greater<int>()(5, 3) << std::endl;
    std::cout << "5 <= 5: " << std::less_equal<int>()(5, 5) << std::endl;
    std::cout << "5 >= 5: " << std::greater_equal<int>()(5, 5) << std::endl;
    std::cout << "5 == 5: " << std::equal_to<int>()(5, 5) << std::endl;
    std::cout << "5 != 3: " << std::not_equal_to<int>()(5, 3) << std::endl;
}
```

### Logical Function Objects
```cpp
#include <functional>
#include <vector>
#include <algorithm>
#include <iostream>

void logical_functors_examples() {
    std::vector<bool> vec1 = {true, false, true, false};
    std::vector<bool> vec2 = {true, true, false, false};
    std::vector<bool> result(vec1.size());
    
    // std::logical_and
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), 
                   std::logical_and<bool>());
    
    std::cout << "Logical AND: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // std::logical_or
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), 
                   std::logical_or<bool>());
    
    std::cout << "Logical OR: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // std::logical_not (unary)
    std::transform(vec1.begin(), vec1.end(), result.begin(), 
                   std::logical_not<bool>());
    
    std::cout << "Logical NOT: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

## Custom Function Objects

### Basic Custom Functor
```cpp
#include <vector>
#include <algorithm>
#include <iostream>

// Custom functor class
class Multiply {
private:
    int factor_;
    
public:
    Multiply(int factor) : factor_(factor) {}
    
    int operator()(int value) const {
        return value * factor_;
    }
};

// Predicate functor
class IsEven {
public:
    bool operator()(int value) const {
        return value % 2 == 0;
    }
};

// Comparison functor
class CompareAbsolute {
public:
    bool operator()(int a, int b) const {
        return std::abs(a) < std::abs(b);
    }
};

void custom_functors_examples() {
    std::vector<int> vec = {1, -3, 2, -4, 5, -6, 7, -8};
    std::vector<int> result(vec.size());
    
    // Use custom multiplication functor
    Multiply multiply_by_3(3);
    std::transform(vec.begin(), vec.end(), result.begin(), multiply_by_3);
    
    std::cout << "Multiplied by 3: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Use predicate functor
    IsEven is_even;
    int even_count = std::count_if(vec.begin(), vec.end(), is_even);
    std::cout << "Even numbers count: " << even_count << std::endl;
    
    // Use comparison functor
    std::vector<int> abs_sorted = vec;
    CompareAbsolute comp_abs;
    std::sort(abs_sorted.begin(), abs_sorted.end(), comp_abs);
    
    std::cout << "Sorted by absolute value: ";
    for (const auto& item : abs_sorted) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### Stateful Functors
```cpp
#include <vector>
#include <algorithm>
#include <iostream>

// Functor with state
class Counter {
private:
    mutable int count_; // mutable because operator() is const
    
public:
    Counter() : count_(0) {}
    
    void operator()(int value) const {
        count_++;
        std::cout << "Processing element #" << count_ << ": " << value << std::endl;
    }
    
    int getCount() const { return count_; }
};

// Accumulator functor
class Accumulator {
private:
    mutable double sum_;
    mutable int count_;
    
public:
    Accumulator() : sum_(0.0), count_(0) {}
    
    void operator()(double value) const {
        sum_ += value;
        count_++;
    }
    
    double getAverage() const {
        return count_ > 0 ? sum_ / count_ : 0.0;
    }
    
    double getSum() const { return sum_; }
    int getCount() const { return count_; }
};

void stateful_functors_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // Use stateful counter
    Counter counter = std::for_each(vec.begin(), vec.end(), Counter());
    std::cout << "Total elements processed: " << counter.getCount() << std::endl;
    
    // Use accumulator
    std::vector<double> values = {1.5, 2.3, 3.7, 4.1, 5.9, 6.2};
    Accumulator acc = std::for_each(values.begin(), values.end(), Accumulator());
    
    std::cout << "Sum: " << acc.getSum() << std::endl;
    std::cout << "Count: " << acc.getCount() << std::endl;
    std::cout << "Average: " << acc.getAverage() << std::endl;
}
```

## std::function

### Basic std::function Usage
```cpp
#include <functional>
#include <vector>
#include <algorithm>
#include <iostream>

// Regular function
int square(int x) {
    return x * x;
}

// Function object
struct Cube {
    int operator()(int x) const {
        return x * x * x;
    }
};

void std_function_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::vector<int> result(vec.size());
    
    // std::function can hold different callable types
    
    // 1. Regular function
    std::function<int(int)> func1 = square;
    std::transform(vec.begin(), vec.end(), result.begin(), func1);
    
    std::cout << "Squares: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // 2. Function object
    std::function<int(int)> func2 = Cube();
    std::transform(vec.begin(), vec.end(), result.begin(), func2);
    
    std::cout << "Cubes: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // 3. Lambda expression
    std::function<int(int)> func3 = [](int x) { return x * x * x * x; };
    std::transform(vec.begin(), vec.end(), result.begin(), func3);
    
    std::cout << "Fourth powers: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // 4. Member function (with std::bind)
    struct Calculator {
        int multiply(int x, int factor) const {
            return x * factor;
        }
    };
    
    Calculator calc;
    std::function<int(int)> func4 = std::bind(&Calculator::multiply, &calc, std::placeholders::_1, 10);
    std::transform(vec.begin(), vec.end(), result.begin(), func4);
    
    std::cout << "Multiplied by 10: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### std::function as Class Member
```cpp
#include <functional>
#include <vector>
#include <iostream>

class EventProcessor {
private:
    std::function<void(int)> event_handler_;
    
public:
    void setEventHandler(std::function<void(int)> handler) {
        event_handler_ = handler;
    }
    
    void processEvents(const std::vector<int>& events) {
        if (event_handler_) {
            for (const auto& event : events) {
                event_handler_(event);
            }
        }
    }
};

void std_function_class_examples() {
    EventProcessor processor;
    std::vector<int> events = {1, 2, 3, 4, 5};
    
    // Set different handlers
    
    // Lambda handler
    processor.setEventHandler([](int event) {
        std::cout << "Processing event: " << event << std::endl;
    });
    processor.processEvents(events);
    
    // Function object handler
    class EventLogger {
    public:
        void operator()(int event) const {
            std::cout << "Logged event: " << event << std::endl;
        }
    };
    
    processor.setEventHandler(EventLogger());
    processor.processEvents(events);
}
```

## std::bind

### Basic std::bind Usage
```cpp
#include <functional>
#include <vector>
#include <algorithm>
#include <iostream>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b, int c) {
    return a * b * c;
}

void std_bind_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::vector<int> result(vec.size());
    
    // Bind to create unary function from binary function
    auto add_10 = std::bind(add, std::placeholders::_1, 10);
    std::transform(vec.begin(), vec.end(), result.begin(), add_10);
    
    std::cout << "Add 10: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Bind with multiple placeholders
    auto multiply_by_2_and_3 = std::bind(multiply, std::placeholders::_1, 2, 3);
    std::transform(vec.begin(), vec.end(), result.begin(), multiply_by_2_and_3);
    
    std::cout << "Multiply by 2 and 3: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Bind with reordered arguments
    auto subtract = [](int a, int b) { return a - b; };
    auto subtract_from_100 = std::bind(subtract, 100, std::placeholders::_1);
    std::transform(vec.begin(), vec.end(), result.begin(), subtract_from_100);
    
    std::cout << "Subtract from 100: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### Binding Member Functions
```cpp
#include <functional>
#include <vector>
#include <algorithm>
#include <iostream>

class Calculator {
private:
    double factor_;
    
public:
    Calculator(double factor) : factor_(factor) {}
    
    double multiply(double value) const {
        return value * factor_;
    }
    
    double power(double base, double exponent) const {
        return std::pow(base, exponent);
    }
    
    void print(const std::string& message) const {
        std::cout << "Calculator says: " << message << std::endl;
    }
};

void bind_member_functions_examples() {
    Calculator calc(2.5);
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> result(values.size());
    
    // Bind member function
    auto multiply_func = std::bind(&Calculator::multiply, &calc, std::placeholders::_1);
    std::transform(values.begin(), values.end(), result.begin(), multiply_func);
    
    std::cout << "Multiplied by factor: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Bind member function with multiple parameters
    auto square_func = std::bind(&Calculator::power, &calc, std::placeholders::_1, 2.0);
    std::transform(values.begin(), values.end(), result.begin(), square_func);
    
    std::cout << "Squared: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Bind void member function
    auto print_func = std::bind(&Calculator::print, &calc, std::placeholders::_1);
    std::vector<std::string> messages = {"Hello", "World", "From", "Calculator"};
    std::for_each(messages.begin(), messages.end(), print_func);
}
```

## Lambda Expressions as Function Objects

### Basic Lambda Usage
```cpp
#include <vector>
#include <algorithm>
#include <iostream>

void lambda_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Simple lambda with algorithm
    std::cout << "Even numbers: ";
    std::copy_if(vec.begin(), vec.end(), 
                 std::ostream_iterator<int>(std::cout, " "),
                 [](int n) { return n % 2 == 0; });
    std::cout << std::endl;
    
    // Lambda with capture
    int threshold = 5;
    std::cout << "Numbers > " << threshold << ": ";
    std::copy_if(vec.begin(), vec.end(), 
                 std::ostream_iterator<int>(std::cout, " "),
                 [threshold](int n) { return n > threshold; });
    std::cout << std::endl;
    
    // Lambda modifying captured variable
    int sum = 0;
    std::for_each(vec.begin(), vec.end(), 
                  [&sum](int n) { sum += n; });
    std::cout << "Sum: " << sum << std::endl;
    
    // Lambda with mutable capture
    int multiplier = 2;
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [multiplier](int n) mutable { 
                       return n * multiplier++; 
                   });
    
    std::cout << "Transformed: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    std::cout << "Original multiplier: " << multiplier << std::endl;
}
```

### Advanced Lambda Features
```cpp
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

void advanced_lambda_examples() {
    std::vector<std::string> words = {"apple", "banana", "cherry", "date", "elderberry"};
    
    // Generic lambda (C++14)
    auto print_with_length = [](const auto& item) {
        std::cout << item << " (length: " << item.size() << ")" << std::endl;
    };
    
    std::cout << "Words with lengths:" << std::endl;
    std::for_each(words.begin(), words.end(), print_with_length);
    
    // Lambda with complex capture
    std::string prefix = "Fruit: ";
    int counter = 1;
    
    std::for_each(words.begin(), words.end(), 
                  [prefix, &counter](const std::string& word) {
                      std::cout << counter++ << ". " << prefix << word << std::endl;
                  });
    
    // Lambda returning lambda (higher-order function)
    auto make_multiplier = [](int factor) {
        return [factor](int value) {
            return value * factor;
        };
    };
    
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::vector<int> result(numbers.size());
    
    auto triple = make_multiplier(3);
    std::transform(numbers.begin(), numbers.end(), result.begin(), triple);
    
    std::cout << "Tripled: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Lambda with explicit return type
    auto divide_with_check = [](double a, double b) -> double {
        if (b == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        return a / b;
    };
    
    try {
        std::cout << "10 / 3 = " << divide_with_check(10.0, 3.0) << std::endl;
        std::cout << "10 / 0 = " << divide_with_check(10.0, 0.0) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}
```

## Function Object Adapters and Utilities

### std::not_fn (C++17)
```cpp
#include <functional>
#include <vector>
#include <algorithm>
#include <iostream>

void not_fn_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    auto is_even = [](int n) { return n % 2 == 0; };
    
    // Count even numbers
    int even_count = std::count_if(vec.begin(), vec.end(), is_even);
    std::cout << "Even numbers: " << even_count << std::endl;
    
    // Count odd numbers using std::not_fn
    int odd_count = std::count_if(vec.begin(), vec.end(), std::not_fn(is_even));
    std::cout << "Odd numbers: " << odd_count << std::endl;
    
    // Copy odd numbers
    std::vector<int> odd_numbers;
    std::copy_if(vec.begin(), vec.end(), std::back_inserter(odd_numbers), 
                 std::not_fn(is_even));
    
    std::cout << "Odd numbers: ";
    for (const auto& item : odd_numbers) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### Function Composition
```cpp
#include <functional>
#include <vector>
#include <algorithm>
#include <iostream>

// Compose two functions
template<typename F, typename G>
auto compose(F f, G g) {
    return [f, g](auto x) { return f(g(x)); };
}

void function_composition_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::vector<int> result(vec.size());
    
    auto square = [](int x) { return x * x; };
    auto add_one = [](int x) { return x + 1; };
    
    // Square then add one
    auto square_then_add = compose(add_one, square);
    std::transform(vec.begin(), vec.end(), result.begin(), square_then_add);
    
    std::cout << "Square then add 1: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Add one then square
    auto add_then_square = compose(square, add_one);
    std::transform(vec.begin(), vec.end(), result.begin(), add_then_square);
    
    std::cout << "Add 1 then square: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

## Performance Considerations

### Function Object vs Lambda Performance
```cpp
#include <functional>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

// Function object
struct MultiplyFunctor {
    int factor;
    MultiplyFunctor(int f) : factor(f) {}
    int operator()(int x) const { return x * factor; }
};

// Regular function
int multiply_function(int x, int factor) {
    return x * factor;
}

void performance_comparison() {
    const size_t SIZE = 1000000;
    std::vector<int> vec(SIZE, 1);
    std::vector<int> result(SIZE);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Function object (often fastest due to inlining)
    MultiplyFunctor mult_functor(2);
    std::transform(vec.begin(), vec.end(), result.begin(), mult_functor);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Function object: " << duration.count() << " microseconds" << std::endl;
    
    // Lambda (usually equivalent to function object)
    start = std::chrono::high_resolution_clock::now();
    
    std::transform(vec.begin(), vec.end(), result.begin(), 
                   [](int x) { return x * 2; });
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Lambda: " << duration.count() << " microseconds" << std::endl;
    
    // std::function (usually slowest due to type erasure)
    start = std::chrono::high_resolution_clock::now();
    
    std::function<int(int)> func = [](int x) { return x * 2; };
    std::transform(vec.begin(), vec.end(), result.begin(), func);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "std::function: " << duration.count() << " microseconds" << std::endl;
    
    // std::bind (can be slower due to additional overhead)
    start = std::chrono::high_resolution_clock::now();
    
    auto bound_func = std::bind(multiply_function, std::placeholders::_1, 2);
    std::transform(vec.begin(), vec.end(), result.begin(), bound_func);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "std::bind: " << duration.count() << " microseconds" << std::endl;
}
```

## Real-world Examples

### Event System with Function Objects
```cpp
#include <functional>
#include <vector>
#include <string>
#include <iostream>
#include <map>

class EventSystem {
public:
    using EventHandler = std::function<void(const std::string&)>;
    
private:
    std::map<std::string, std::vector<EventHandler>> handlers_;
    
public:
    void subscribe(const std::string& event_type, EventHandler handler) {
        handlers_[event_type].push_back(handler);
    }
    
    void emit(const std::string& event_type, const std::string& data) {
        auto it = handlers_.find(event_type);
        if (it != handlers_.end()) {
            for (const auto& handler : it->second) {
                handler(data);
            }
        }
    }
};

// Event handlers
class Logger {
public:
    void log(const std::string& message) {
        std::cout << "[LOG] " << message << std::endl;
    }
};

class EmailNotifier {
private:
    std::string email_;
    
public:
    EmailNotifier(const std::string& email) : email_(email) {}
    
    void notify(const std::string& message) {
        std::cout << "[EMAIL to " << email_ << "] " << message << std::endl;
    }
};

void event_system_example() {
    EventSystem event_system;
    Logger logger;
    EmailNotifier admin_notifier("admin@example.com");
    EmailNotifier user_notifier("user@example.com");
    
    // Subscribe with member function
    event_system.subscribe("error", 
                          std::bind(&Logger::log, &logger, std::placeholders::_1));
    
    event_system.subscribe("error", 
                          std::bind(&EmailNotifier::notify, &admin_notifier, 
                                   std::placeholders::_1));
    
    // Subscribe with lambda
    event_system.subscribe("user_login", 
                          [&logger](const std::string& data) {
                              logger.log("User logged in: " + data);
                          });
    
    event_system.subscribe("user_login", 
                          [&user_notifier](const std::string& data) {
                              user_notifier.notify("Welcome back!");
                          });
    
    // Emit events
    event_system.emit("error", "Database connection failed");
    event_system.emit("user_login", "john_doe");
}
```

### Functional Programming Pipeline
```cpp
#include <functional>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <string>

template<typename Container, typename Predicate>
auto filter(const Container& container, Predicate pred) {
    Container result;
    std::copy_if(container.begin(), container.end(), 
                 std::back_inserter(result), pred);
    return result;
}

template<typename Container, typename Transform>
auto map(const Container& container, Transform trans) {
    using ValueType = decltype(trans(*container.begin()));
    std::vector<ValueType> result;
    result.reserve(container.size());
    std::transform(container.begin(), container.end(), 
                   std::back_inserter(result), trans);
    return result;
}

template<typename Container, typename BinaryOp>
auto reduce(const Container& container, 
           typename Container::value_type init, 
           BinaryOp op) {
    return std::accumulate(container.begin(), container.end(), init, op);
}

void functional_pipeline_example() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Functional pipeline: filter even numbers, square them, sum the result
    auto even_numbers = filter(numbers, [](int n) { return n % 2 == 0; });
    
    std::cout << "Even numbers: ";
    for (const auto& n : even_numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    auto squared = map(even_numbers, [](int n) { return n * n; });
    
    std::cout << "Squared: ";
    for (const auto& n : squared) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    auto sum = reduce(squared, 0, [](int a, int b) { return a + b; });
    
    std::cout << "Sum of squares of even numbers: " << sum << std::endl;
    
    // One-liner version
    auto result = reduce(
        map(filter(numbers, [](int n) { return n % 2 == 0; }),
            [](int n) { return n * n; }),
        0,
        [](int a, int b) { return a + b; }
    );
    
    std::cout << "One-liner result: " << result << std::endl;
}
```

## Complete Example Program

```cpp
#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>

int main() {
    std::cout << "=== Function Objects (Functors) Examples ===" << std::endl;
    
    std::cout << "\n--- Predefined Functors ---" << std::endl;
    arithmetic_functors_examples();
    comparison_functors_examples();
    logical_functors_examples();
    
    std::cout << "\n--- Custom Functors ---" << std::endl;
    custom_functors_examples();
    stateful_functors_examples();
    
    std::cout << "\n--- std::function ---" << std::endl;
    std_function_examples();
    std_function_class_examples();
    
    std::cout << "\n--- std::bind ---" << std::endl;
    std_bind_examples();
    bind_member_functions_examples();
    
    std::cout << "\n--- Lambda Expressions ---" << std::endl;
    lambda_examples();
    advanced_lambda_examples();
    
    std::cout << "\n--- Function Utilities ---" << std::endl;
    not_fn_examples();
    function_composition_examples();
    
    std::cout << "\n--- Performance Comparison ---" << std::endl;
    performance_comparison();
    
    std::cout << "\n--- Real-world Examples ---" << std::endl;
    event_system_example();
    functional_pipeline_example();
    
    return 0;
}
```

## Best Practices

1. **Choose the right tool**:
   - Lambdas for simple, local operations
   - Function objects for reusable, stateful operations
   - std::function for type erasure and polymorphism
   - std::bind for partial application (though lambdas are often preferred)

2. **Performance considerations**:
   - Lambdas and function objects are usually fastest (inlined)
   - std::function has overhead due to type erasure
   - std::bind can have additional overhead

3. **Prefer lambdas over std::bind**:
   - Lambdas are more readable and performant
   - Better error messages
   - More flexible capture semantics

4. **Use const correctness**:
   - Make operator() const when possible
   - Use appropriate capture modes in lambdas

5. **Consider generic lambdas** (C++14):
   - Use `auto` parameters for more flexible code
   - Reduces template instantiation bloat

## Key Concepts Summary

1. **Function Objects**: Objects that can be called like functions
2. **Predefined Functors**: Standard library provides common operations
3. **Custom Functors**: Create reusable, stateful callable objects
4. **std::function**: Type-erased function wrapper
5. **std::bind**: Partial function application and argument binding
6. **Lambdas**: Anonymous functions with capture capabilities
7. **Performance**: Consider overhead of different approaches

## Exercises

1. Create a custom functor that maintains statistics (min, max, avg) as it processes elements
2. Implement a higher-order function that takes two functions and returns their composition
3. Create a configurable validator using function objects for different data types
4. Build a simple command pattern using std::function
5. Implement a functional programming library with map, filter, and reduce operations
6. Create a caching decorator using function objects that memoizes expensive computations
7. Build a pipeline processing system using function objects and algorithms
