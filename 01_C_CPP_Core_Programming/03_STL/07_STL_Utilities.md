# STL Utilities

*Part of STL Learning Track - 1 week*

## Overview

STL utilities provide fundamental building blocks and helper types that support the STL and general C++ programming. This includes std::pair, std::tuple, optional types, variant types, smart pointers, time utilities, and mathematical utilities like std::ratio.

## std::pair and std::tuple

### std::pair Basics
```cpp
#include <utility>
#include <iostream>
#include <string>
#include <map>
#include <algorithm>

void pair_basic_examples() {
    std::cout << "=== std::pair Basic Examples ===" << std::endl;
    
    // Construction
    std::pair<int, std::string> p1(42, "answer");
    std::pair<int, std::string> p2 = std::make_pair(10, "ten");
    auto p3 = std::make_pair(3.14, "pi");
    
    std::cout << "p1: (" << p1.first << ", " << p1.second << ")" << std::endl;
    std::cout << "p2: (" << p2.first << ", " << p2.second << ")" << std::endl;
    std::cout << "p3: (" << p3.first << ", " << p3.second << ")" << std::endl;
    
    // Assignment
    p1 = p2;
    std::cout << "After assignment p1 = p2: (" << p1.first << ", " << p1.second << ")" << std::endl;
    
    // Comparison
    std::pair<int, int> pa(1, 2);
    std::pair<int, int> pb(1, 3);
    std::pair<int, int> pc(2, 1);
    
    std::cout << "pa < pb: " << (pa < pb) << std::endl;
    std::cout << "pa < pc: " << (pa < pc) << std::endl;
    std::cout << "pa == pa: " << (pa == pa) << std::endl;
    
    // Using with containers
    std::map<std::string, int> word_count;
    word_count.insert(std::make_pair("hello", 1));
    word_count.insert({"world", 1});
    
    // Find and access
    auto it = word_count.find("hello");
    if (it != word_count.end()) {
        std::cout << "Found: " << it->first << " -> " << it->second << std::endl;
    }
}
```

### std::pair with Algorithms
```cpp
#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>

void pair_with_algorithms() {
    std::cout << "\n=== std::pair with Algorithms ===" << std::endl;
    
    // Vector of pairs
    std::vector<std::pair<std::string, int>> students = {
        {"Alice", 85},
        {"Bob", 90},
        {"Charlie", 78},
        {"Diana", 92},
        {"Eve", 88}
    };
    
    std::cout << "Original list:" << std::endl;
    for (const auto& student : students) {
        std::cout << student.first << ": " << student.second << std::endl;
    }
    
    // Sort by grade (second element)
    std::sort(students.begin(), students.end(), 
              [](const auto& a, const auto& b) {
                  return a.second > b.second; // Descending order
              });
    
    std::cout << "\nSorted by grade (descending):" << std::endl;
    for (const auto& student : students) {
        std::cout << student.first << ": " << student.second << std::endl;
    }
    
    // Find student with highest grade
    auto max_grade = std::max_element(students.begin(), students.end(),
                                      [](const auto& a, const auto& b) {
                                          return a.second < b.second;
                                      });
    
    std::cout << "\nTop student: " << max_grade->first 
              << " with grade " << max_grade->second << std::endl;
    
    // Count students with grade >= 85
    int high_performers = std::count_if(students.begin(), students.end(),
                                        [](const auto& student) {
                                            return student.second >= 85;
                                        });
    
    std::cout << "Students with grade >= 85: " << high_performers << std::endl;
}
```

### std::tuple Basics
```cpp
#include <tuple>
#include <iostream>
#include <string>

void tuple_basic_examples() {
    std::cout << "\n=== std::tuple Basic Examples ===" << std::endl;
    
    // Construction
    std::tuple<int, std::string, double> t1(42, "hello", 3.14);
    auto t2 = std::make_tuple(10, "world", 2.71);
    std::tuple<int, std::string, double> t3{100, "tuple", 1.41};
    
    // Access elements
    std::cout << "t1: (" << std::get<0>(t1) << ", " 
              << std::get<1>(t1) << ", " << std::get<2>(t1) << ")" << std::endl;
    
    // Modify elements
    std::get<0>(t1) = 99;
    std::get<1>(t1) = "modified";
    
    std::cout << "Modified t1: (" << std::get<0>(t1) << ", " 
              << std::get<1>(t1) << ", " << std::get<2>(t1) << ")" << std::endl;
    
    // Structured bindings (C++17)
    auto [value, name, pi] = t2;
    std::cout << "Structured binding: " << value << ", " << name << ", " << pi << std::endl;
    
    // Tuple size and type
    constexpr size_t tuple_size = std::tuple_size_v<decltype(t1)>;
    std::cout << "Tuple size: " << tuple_size << std::endl;
    
    // Tie for unpacking
    int val;
    std::string str;
    double dbl;
    std::tie(val, str, dbl) = t1;
    std::cout << "Unpacked: " << val << ", " << str << ", " << dbl << std::endl;
    
    // Ignore some values
    std::tie(val, std::ignore, dbl) = t1;
    std::cout << "Partial unpack: " << val << ", " << dbl << std::endl;
}
```

### Advanced Tuple Operations
```cpp
#include <tuple>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

// Function returning multiple values
std::tuple<int, double, std::string> get_stats(const std::vector<int>& data) {
    if (data.empty()) {
        return std::make_tuple(0, 0.0, "empty");
    }
    
    int sum = 0;
    for (int val : data) {
        sum += val;
    }
    
    double average = static_cast<double>(sum) / data.size();
    std::string status = average > 50 ? "good" : "needs improvement";
    
    return std::make_tuple(sum, average, status);
}

// Tuple concatenation
template<typename... Tuples>
auto tuple_cat_example(Tuples&&... tuples) {
    return std::tuple_cat(std::forward<Tuples>(tuples)...);
}

void advanced_tuple_operations() {
    std::cout << "\n=== Advanced Tuple Operations ===" << std::endl;
    
    // Function returning tuple
    std::vector<int> grades = {85, 90, 78, 92, 88};
    auto [sum, avg, status] = get_stats(grades);
    
    std::cout << "Stats - Sum: " << sum << ", Average: " << avg 
              << ", Status: " << status << std::endl;
    
    // Tuple concatenation
    auto t1 = std::make_tuple(1, 2);
    auto t2 = std::make_tuple("hello", 3.14);
    auto t3 = std::make_tuple(true);
    
    auto combined = tuple_cat_example(t1, t2, t3);
    
    std::cout << "Combined tuple: (" 
              << std::get<0>(combined) << ", "
              << std::get<1>(combined) << ", "
              << std::get<2>(combined) << ", "
              << std::get<3>(combined) << ", "
              << std::get<4>(combined) << ")" << std::endl;
    
    // Tuple comparison
    auto ta = std::make_tuple(1, "a", 3.0);
    auto tb = std::make_tuple(1, "b", 3.0);
    auto tc = std::make_tuple(1, "a", 3.0);
    
    std::cout << "ta < tb: " << (ta < tb) << std::endl;
    std::cout << "ta == tc: " << (ta == tc) << std::endl;
    
    // Using tuples with containers
    std::vector<std::tuple<std::string, int, double>> employees = {
        {"Alice", 30, 75000.0},
        {"Bob", 25, 65000.0},
        {"Charlie", 35, 85000.0}
    };
    
    // Sort by salary (third element)
    std::sort(employees.begin(), employees.end(),
              [](const auto& a, const auto& b) {
                  return std::get<2>(a) > std::get<2>(b);
              });
    
    std::cout << "Employees sorted by salary:" << std::endl;
    for (const auto& [name, age, salary] : employees) {
        std::cout << name << " (age " << age << "): $" << salary << std::endl;
    }
}
```

## Optional Types (C++17)

### std::optional Basics
```cpp
#include <optional>
#include <iostream>
#include <string>
#include <vector>

std::optional<int> find_first_even(const std::vector<int>& numbers) {
    for (int num : numbers) {
        if (num % 2 == 0) {
            return num;
        }
    }
    return std::nullopt; // No even number found
}

std::optional<double> safe_divide(double a, double b) {
    if (b == 0.0) {
        return std::nullopt;
    }
    return a / b;
}

void optional_basic_examples() {
    std::cout << "\n=== std::optional Basic Examples ===" << std::endl;
    
    // Construction
    std::optional<int> opt1;              // Empty
    std::optional<int> opt2 = 42;         // Has value
    std::optional<int> opt3 = std::nullopt; // Explicitly empty
    std::optional<int> opt4{100};         // Has value
    
    // Check if has value
    std::cout << "opt1 has value: " << opt1.has_value() << std::endl;
    std::cout << "opt2 has value: " << opt2.has_value() << std::endl;
    
    // Access value
    if (opt2) {
        std::cout << "opt2 value: " << *opt2 << std::endl;
        std::cout << "opt2 value (method): " << opt2.value() << std::endl;
    }
    
    // Value or default
    std::cout << "opt1 value or default: " << opt1.value_or(-1) << std::endl;
    std::cout << "opt2 value or default: " << opt2.value_or(-1) << std::endl;
    
    // Function returning optional
    std::vector<int> numbers1 = {1, 3, 5, 7, 9};
    std::vector<int> numbers2 = {1, 3, 4, 7, 9};
    
    auto even1 = find_first_even(numbers1);
    auto even2 = find_first_even(numbers2);
    
    if (even1) {
        std::cout << "First even in numbers1: " << *even1 << std::endl;
    } else {
        std::cout << "No even number in numbers1" << std::endl;
    }
    
    if (even2) {
        std::cout << "First even in numbers2: " << *even2 << std::endl;
    } else {
        std::cout << "No even number in numbers2" << std::endl;
    }
    
    // Safe division
    auto result1 = safe_divide(10.0, 2.0);
    auto result2 = safe_divide(10.0, 0.0);
    
    std::cout << "10/2 = " << result1.value_or(0.0) << std::endl;
    std::cout << "10/0 = " << result2.value_or(0.0) << " (division by zero)" << std::endl;
}
```

### std::optional Advanced Usage
```cpp
#include <optional>
#include <iostream>
#include <string>
#include <map>

class UserDatabase {
private:
    std::map<int, std::string> users_;
    
public:
    UserDatabase() {
        users_[1] = "Alice";
        users_[2] = "Bob";
        users_[3] = "Charlie";
    }
    
    std::optional<std::string> get_user(int id) const {
        auto it = users_.find(id);
        if (it != users_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    bool add_user(int id, const std::string& name) {
        if (users_.find(id) != users_.end()) {
            return false; // User already exists
        }
        users_[id] = name;
        return true;
    }
};

// Chaining optionals
std::optional<int> string_to_int(const std::string& str) {
    try {
        return std::stoi(str);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<double> int_to_sqrt(int value) {
    if (value < 0) {
        return std::nullopt;
    }
    return std::sqrt(value);
}

void optional_advanced_examples() {
    std::cout << "\n=== std::optional Advanced Examples ===" << std::endl;
    
    UserDatabase db;
    
    // Query users
    for (int id = 1; id <= 5; ++id) {
        if (auto user = db.get_user(id)) {
            std::cout << "User " << id << ": " << *user << std::endl;
        } else {
            std::cout << "User " << id << ": not found" << std::endl;
        }
    }
    
    // Chaining operations
    std::vector<std::string> inputs = {"16", "25", "-4", "abc", "9"};
    
    for (const auto& input : inputs) {
        std::cout << "Input: " << input << " -> ";
        
        auto opt_int = string_to_int(input);
        if (opt_int) {
            auto opt_sqrt = int_to_sqrt(*opt_int);
            if (opt_sqrt) {
                std::cout << "sqrt = " << *opt_sqrt;
            } else {
                std::cout << "negative number";
            }
        } else {
            std::cout << "invalid number";
        }
        
        std::cout << std::endl;
    }
    
    // Using optional with emplace
    std::optional<std::string> opt_str;
    opt_str.emplace("constructed in place");
    std::cout << "Emplaced string: " << *opt_str << std::endl;
    
    // Reset optional
    opt_str.reset();
    std::cout << "After reset, has value: " << opt_str.has_value() << std::endl;
}
```

## std::variant (C++17)

### std::variant Basics
```cpp
#include <variant>
#include <iostream>
#include <string>
#include <vector>

void variant_basic_examples() {
    std::cout << "\n=== std::variant Basic Examples ===" << std::endl;
    
    // Construction
    std::variant<int, double, std::string> var1 = 42;
    std::variant<int, double, std::string> var2 = 3.14;
    std::variant<int, double, std::string> var3 = "hello";
    
    // Check which type is active
    std::cout << "var1 index: " << var1.index() << std::endl;
    std::cout << "var2 index: " << var2.index() << std::endl;
    std::cout << "var3 index: " << var3.index() << std::endl;
    
    // Check if holds specific type
    std::cout << "var1 holds int: " << std::holds_alternative<int>(var1) << std::endl;
    std::cout << "var1 holds string: " << std::holds_alternative<std::string>(var1) << std::endl;
    
    // Access value
    if (std::holds_alternative<int>(var1)) {
        std::cout << "var1 int value: " << std::get<int>(var1) << std::endl;
    }
    
    // Access by index
    std::cout << "var2 double value: " << std::get<1>(var2) << std::endl;
    
    // Safe access with get_if
    if (auto ptr = std::get_if<std::string>(&var3)) {
        std::cout << "var3 string value: " << *ptr << std::endl;
    }
    
    // Change variant value
    var1 = "now a string";
    std::cout << "var1 after assignment: " << std::get<std::string>(var1) << std::endl;
}
```

### std::variant with std::visit
```cpp
#include <variant>
#include <iostream>
#include <string>

// Visitor using function object
struct PrintVisitor {
    void operator()(int value) const {
        std::cout << "Integer: " << value << std::endl;
    }
    
    void operator()(double value) const {
        std::cout << "Double: " << value << std::endl;
    }
    
    void operator()(const std::string& value) const {
        std::cout << "String: " << value << std::endl;
    }
};

// Generic visitor
struct ToStringVisitor {
    std::string operator()(int value) const {
        return "int(" + std::to_string(value) + ")";
    }
    
    std::string operator()(double value) const {
        return "double(" + std::to_string(value) + ")";
    }
    
    std::string operator()(const std::string& value) const {
        return "string(" + value + ")";
    }
};

void variant_visitor_examples() {
    std::cout << "\n=== std::variant with std::visit ===" << std::endl;
    
    std::vector<std::variant<int, double, std::string>> variants = {
        42,
        3.14159,
        "hello world",
        100,
        2.71828
    };
    
    std::cout << "Using PrintVisitor:" << std::endl;
    for (const auto& var : variants) {
        std::visit(PrintVisitor{}, var);
    }
    
    std::cout << "\nUsing ToStringVisitor:" << std::endl;
    for (const auto& var : variants) {
        std::string result = std::visit(ToStringVisitor{}, var);
        std::cout << result << std::endl;
    }
    
    // Using lambda visitor
    std::cout << "\nUsing lambda visitor:" << std::endl;
    for (const auto& var : variants) {
        std::visit([](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, int>) {
                std::cout << "Processing integer: " << value << std::endl;
            } else if constexpr (std::is_same_v<T, double>) {
                std::cout << "Processing double: " << value << std::endl;
            } else if constexpr (std::is_same_v<T, std::string>) {
                std::cout << "Processing string: " << value << std::endl;
            }
        }, var);
    }
}
```

## std::any (C++17)

### std::any Usage
```cpp
#include <any>
#include <iostream>
#include <string>
#include <vector>
#include <typeinfo>

void any_examples() {
    std::cout << "\n=== std::any Examples ===" << std::endl;
    
    // Construction
    std::any any1 = 42;
    std::any any2 = 3.14;
    std::any any3 = std::string("hello");
    std::any any4; // Empty
    
    // Check if has value
    std::cout << "any1 has value: " << any1.has_value() << std::endl;
    std::cout << "any4 has value: " << any4.has_value() << std::endl;
    
    // Type information
    std::cout << "any1 type: " << any1.type().name() << std::endl;
    std::cout << "any2 type: " << any2.type().name() << std::endl;
    std::cout << "any3 type: " << any3.type().name() << std::endl;
    
    // Cast to specific type
    try {
        int value1 = std::any_cast<int>(any1);
        std::cout << "any1 as int: " << value1 << std::endl;
        
        double value2 = std::any_cast<double>(any2);
        std::cout << "any2 as double: " << value2 << std::endl;
        
        std::string value3 = std::any_cast<std::string>(any3);
        std::cout << "any3 as string: " << value3 << std::endl;
        
        // This will throw
        std::string wrong = std::any_cast<std::string>(any1);
        
    } catch (const std::bad_any_cast& e) {
        std::cout << "Bad any_cast: " << e.what() << std::endl;
    }
    
    // Safe cast with pointer
    if (auto ptr = std::any_cast<int>(&any1)) {
        std::cout << "Safe cast any1: " << *ptr << std::endl;
    }
    
    if (auto ptr = std::any_cast<std::string>(&any1)) {
        std::cout << "This won't print (wrong type)" << std::endl;
    } else {
        std::cout << "any1 is not a string" << std::endl;
    }
    
    // Reset any
    any1.reset();
    std::cout << "After reset, any1 has value: " << any1.has_value() << std::endl;
    
    // Container of any
    std::vector<std::any> mixed_data = {
        42,
        3.14,
        std::string("text"),
        true,
        'A'
    };
    
    std::cout << "\nMixed container:" << std::endl;
    for (const auto& item : mixed_data) {
        std::cout << "Type: " << item.type().name() << std::endl;
    }
}
```

## Smart Pointers

### std::unique_ptr
```cpp
#include <memory>
#include <iostream>
#include <vector>

class Resource {
private:
    std::string name_;
    
public:
    Resource(const std::string& name) : name_(name) {
        std::cout << "Resource " << name_ << " created" << std::endl;
    }
    
    ~Resource() {
        std::cout << "Resource " << name_ << " destroyed" << std::endl;
    }
    
    void use() {
        std::cout << "Using resource " << name_ << std::endl;
    }
    
    const std::string& name() const { return name_; }
};

std::unique_ptr<Resource> create_resource(const std::string& name) {
    return std::make_unique<Resource>(name);
}

void unique_ptr_examples() {
    std::cout << "\n=== std::unique_ptr Examples ===" << std::endl;
    
    // Basic usage
    std::unique_ptr<Resource> ptr1 = std::make_unique<Resource>("Resource1");
    ptr1->use();
    
    // Move semantics
    std::unique_ptr<Resource> ptr2 = std::move(ptr1);
    
    if (!ptr1) {
        std::cout << "ptr1 is now empty" << std::endl;
    }
    
    if (ptr2) {
        std::cout << "ptr2 now owns the resource" << std::endl;
        ptr2->use();
    }
    
    // Factory function
    auto ptr3 = create_resource("Resource3");
    ptr3->use();
    
    // Array version
    std::unique_ptr<int[]> arr = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; ++i) {
        arr[i] = i * i;
    }
    
    std::cout << "Array contents: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    
    // Custom deleter
    auto custom_deleter = [](Resource* ptr) {
        std::cout << "Custom deleter called for " << ptr->name() << std::endl;
        delete ptr;
    };
    
    std::unique_ptr<Resource, decltype(custom_deleter)> ptr4(
        new Resource("Resource4"), custom_deleter);
    
    ptr4->use();
    
    // Container of unique_ptr
    std::vector<std::unique_ptr<Resource>> resources;
    resources.push_back(std::make_unique<Resource>("Vec1"));
    resources.push_back(std::make_unique<Resource>("Vec2"));
    resources.push_back(std::make_unique<Resource>("Vec3"));
    
    std::cout << "Using resources in vector:" << std::endl;
    for (auto& res : resources) {
        res->use();
    }
    
    std::cout << "Exiting scope - resources will be destroyed" << std::endl;
}
```

### std::shared_ptr and std::weak_ptr
```cpp
#include <memory>
#include <iostream>
#include <vector>

class Node {
public:
    std::string data;
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> parent; // Avoid circular reference
    
    Node(const std::string& d) : data(d) {
        std::cout << "Node " << data << " created" << std::endl;
    }
    
    ~Node() {
        std::cout << "Node " << data << " destroyed" << std::endl;
    }
};

void shared_ptr_examples() {
    std::cout << "\n=== std::shared_ptr Examples ===" << std::endl;
    
    // Basic usage
    std::shared_ptr<Resource> ptr1 = std::make_shared<Resource>("Shared1");
    std::cout << "Reference count: " << ptr1.use_count() << std::endl;
    
    {
        std::shared_ptr<Resource> ptr2 = ptr1; // Copy, increases count
        std::cout << "Reference count after copy: " << ptr1.use_count() << std::endl;
        
        ptr2->use();
    } // ptr2 goes out of scope, count decreases
    
    std::cout << "Reference count after ptr2 destroyed: " << ptr1.use_count() << std::endl;
    
    // Weak pointer
    std::weak_ptr<Resource> weak_ptr = ptr1;
    std::cout << "Weak ptr expired: " << weak_ptr.expired() << std::endl;
    
    // Lock weak pointer to get shared_ptr
    if (auto locked = weak_ptr.lock()) {
        std::cout << "Locked weak pointer successfully" << std::endl;
        locked->use();
        std::cout << "Reference count with locked weak_ptr: " << locked.use_count() << std::endl;
    }
    
    // Shared ownership example
    std::vector<std::shared_ptr<Resource>> shared_resources;
    
    auto shared_res = std::make_shared<Resource>("SharedResource");
    shared_resources.push_back(shared_res);
    shared_resources.push_back(shared_res);
    shared_resources.push_back(shared_res);
    
    std::cout << "Shared resource count: " << shared_res.use_count() << std::endl;
    
    // Avoid circular references with weak_ptr
    auto root = std::make_shared<Node>("Root");
    auto child1 = std::make_shared<Node>("Child1");
    auto child2 = std::make_shared<Node>("Child2");
    
    root->next = child1;
    child1->parent = root; // weak_ptr prevents circular reference
    child1->next = child2;
    child2->parent = child1;
    
    std::cout << "Tree structure created" << std::endl;
    
    // Access parent through weak_ptr
    if (auto parent = child1->parent.lock()) {
        std::cout << "Child1's parent: " << parent->data << std::endl;
    }
    
    std::cout << "Exiting scope - shared resources will be cleaned up" << std::endl;
}
```

## Time Utilities

### std::chrono Basics
```cpp
#include <chrono>
#include <iostream>
#include <thread>

void chrono_basic_examples() {
    std::cout << "\n=== std::chrono Basic Examples ===" << std::endl;
    
    // Different duration types
    std::chrono::seconds sec(1);
    std::chrono::milliseconds ms(1000);
    std::chrono::microseconds us(1000000);
    std::chrono::nanoseconds ns(1000000000);
    
    std::cout << "1 second = " << ms.count() << " milliseconds" << std::endl;
    std::cout << "1 second = " << us.count() << " microseconds" << std::endl;
    std::cout << "1 second = " << ns.count() << " nanoseconds" << std::endl;
    
    // Duration arithmetic
    auto total_ms = std::chrono::milliseconds(500) + std::chrono::milliseconds(300);
    std::cout << "500ms + 300ms = " << total_ms.count() << "ms" << std::endl;
    
    // Duration conversion
    auto seconds_from_ms = std::chrono::duration_cast<std::chrono::seconds>(total_ms);
    std::cout << total_ms.count() << "ms = " << seconds_from_ms.count() << " seconds" << std::endl;
    
    // Time points
    auto now = std::chrono::steady_clock::now();
    auto system_now = std::chrono::system_clock::now();
    
    // Sleep for a while
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto later = std::chrono::steady_clock::now();
    auto elapsed = later - now;
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    std::cout << "Elapsed time: " << elapsed_ms.count() << "ms" << std::endl;
    
    // System time
    auto time_t_now = std::chrono::system_clock::to_time_t(system_now);
    std::cout << "System time: " << std::ctime(&time_t_now);
}
```

### Performance Timing
```cpp
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

class Timer {
private:
    std::chrono::steady_clock::time_point start_time_;
    
public:
    Timer() : start_time_(std::chrono::steady_clock::now()) {}
    
    void reset() {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    template<typename Duration = std::chrono::milliseconds>
    auto elapsed() const {
        auto end_time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<Duration>(end_time - start_time_);
    }
    
    auto elapsed_ms() const {
        return elapsed<std::chrono::milliseconds>().count();
    }
    
    auto elapsed_us() const {
        return elapsed<std::chrono::microseconds>().count();
    }
};

void performance_timing_examples() {
    std::cout << "\n=== Performance Timing Examples ===" << std::endl;
    
    const size_t size = 1000000;
    std::vector<int> data(size);
    
    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000);
    
    Timer timer;
    
    std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
    std::cout << "Data generation took: " << timer.elapsed_ms() << "ms" << std::endl;
    
    // Time sorting
    timer.reset();
    std::sort(data.begin(), data.end());
    std::cout << "Sorting took: " << timer.elapsed_ms() << "ms" << std::endl;
    
    // Time search
    timer.reset();
    auto it = std::binary_search(data.begin(), data.end(), 500);
    std::cout << "Binary search took: " << timer.elapsed_us() << "μs" << std::endl;
    std::cout << "Found 500: " << it << std::endl;
    
    // Benchmark function
    auto benchmark = [](const std::string& name, auto func) {
        Timer t;
        func();
        std::cout << name << " took: " << t.elapsed_ms() << "ms" << std::endl;
    };
    
    std::vector<int> test_data(100000);
    std::iota(test_data.begin(), test_data.end(), 1);
    
    benchmark("Vector copy", [&]() {
        std::vector<int> copy = test_data;
    });
    
    benchmark("Vector sum", [&]() {
        long long sum = 0;
        for (int val : test_data) {
            sum += val;
        }
    });
}
```

## std::ratio

### Compile-time Rational Arithmetic
```cpp
#include <ratio>
#include <iostream>

void ratio_examples() {
    std::cout << "\n=== std::ratio Examples ===" << std::endl;
    
    // Define ratios
    using half = std::ratio<1, 2>;
    using third = std::ratio<1, 3>;
    using quarter = std::ratio<1, 4>;
    
    std::cout << "half = " << half::num << "/" << half::den << std::endl;
    std::cout << "third = " << third::num << "/" << third::den << std::endl;
    std::cout << "quarter = " << quarter::num << "/" << quarter::den << std::endl;
    
    // Ratio arithmetic
    using sum = std::ratio_add<half, third>;
    using diff = std::ratio_subtract<half, third>;
    using product = std::ratio_multiply<half, third>;
    using quotient = std::ratio_divide<half, third>;
    
    std::cout << "\nArithmetic operations:" << std::endl;
    std::cout << "1/2 + 1/3 = " << sum::num << "/" << sum::den << std::endl;
    std::cout << "1/2 - 1/3 = " << diff::num << "/" << diff::den << std::endl;
    std::cout << "1/2 * 1/3 = " << product::num << "/" << product::den << std::endl;
    std::cout << "1/2 / 1/3 = " << quotient::num << "/" << quotient::den << std::endl;
    
    // Ratio comparisons
    std::cout << "\nComparisons:" << std::endl;
    std::cout << "1/2 == 1/3: " << std::ratio_equal_v<half, third> << std::endl;
    std::cout << "1/2 != 1/3: " << std::ratio_not_equal_v<half, third> << std::endl;
    std::cout << "1/2 < 1/3: " << std::ratio_less_v<half, third> << std::endl;
    std::cout << "1/2 > 1/3: " << std::ratio_greater_v<half, third> << std::endl;
    
    // SI prefixes (predefined ratios)
    std::cout << "\nSI prefixes:" << std::endl;
    std::cout << "kilo = " << std::kilo::num << "/" << std::kilo::den << std::endl;
    std::cout << "mega = " << std::mega::num << "/" << std::mega::den << std::endl;
    std::cout << "milli = " << std::milli::num << "/" << std::milli::den << std::endl;
    std::cout << "micro = " << std::micro::num << "/" << std::micro::den << std::endl;
    
    // Using ratios with chrono
    using custom_duration = std::chrono::duration<double, std::ratio<1, 100>>; // centiseconds
    custom_duration cs(150); // 150 centiseconds
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(cs);
    std::cout << "\n150 centiseconds = " << ms.count() << " milliseconds" << std::endl;
}
```

## Complete Example Program

```cpp
#include <iostream>
#include <utility>
#include <tuple>
#include <optional>
#include <variant>
#include <any>
#include <memory>
#include <chrono>
#include <ratio>

int main() {
    std::cout << "=== STL Utilities Examples ===" << std::endl;
    
    std::cout << "\n--- std::pair ---" << std::endl;
    pair_basic_examples();
    pair_with_algorithms();
    
    std::cout << "\n--- std::tuple ---" << std::endl;
    tuple_basic_examples();
    advanced_tuple_operations();
    
    std::cout << "\n--- std::optional ---" << std::endl;
    optional_basic_examples();
    optional_advanced_examples();
    
    std::cout << "\n--- std::variant ---" << std::endl;
    variant_basic_examples();
    variant_visitor_examples();
    
    std::cout << "\n--- std::any ---" << std::endl;
    any_examples();
    
    std::cout << "\n--- Smart Pointers ---" << std::endl;
    unique_ptr_examples();
    shared_ptr_examples();
    
    std::cout << "\n--- Time Utilities ---" << std::endl;
    chrono_basic_examples();
    performance_timing_examples();
    
    std::cout << "\n--- std::ratio ---" << std::endl;
    ratio_examples();
    
    return 0;
}
```

## Best Practices

1. **Use std::pair for simple two-element structures**
2. **Prefer std::tuple over custom structs for temporary data**
3. **Use std::optional instead of special values for "no value"**
4. **Choose std::variant over unions for type-safe alternatives**
5. **Use std::any sparingly - prefer type-safe alternatives**
6. **Prefer std::make_unique and std::make_shared**
7. **Use std::weak_ptr to break circular references**
8. **Use std::chrono for all time-related operations**
9. **Leverage std::ratio for compile-time arithmetic**

## Key Concepts Summary

1. **std::pair**: Two-element tuple for simple pairs
2. **std::tuple**: Multiple-element tuple with structured bindings
3. **std::optional**: Type-safe nullable values
4. **std::variant**: Type-safe union alternative
5. **std::any**: Type-erased value container
6. **Smart Pointers**: Automatic memory management
7. **std::chrono**: Time and duration utilities
8. **std::ratio**: Compile-time rational arithmetic

## Exercises

1. Create a function that returns multiple values using std::tuple
2. Implement a configuration system using std::variant for different value types
3. Build a cache system using std::optional for cache misses
4. Create a polymorphic container using std::any
5. Implement a resource pool using smart pointers
6. Build a performance profiler using std::chrono
7. Create a unit conversion system using std::ratio
8. Implement a visitor pattern using std::variant and std::visit
