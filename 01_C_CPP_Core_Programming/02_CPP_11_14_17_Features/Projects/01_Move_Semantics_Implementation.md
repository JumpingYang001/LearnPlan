# Move Semantics Implementation Project

## Project Overview

Build a comprehensive system demonstrating move semantics in C++11, including custom classes with proper move constructors, move assignment operators, and performance comparisons between copy and move operations.

## Learning Objectives

- Understand the difference between lvalues and rvalues
- Implement proper move constructors and move assignment operators
- Use std::move and std::forward correctly
- Measure performance improvements from move semantics
- Handle edge cases and exception safety in move operations

## Project Structure

```
move_semantics_project/
├── src/
│   ├── main.cpp
│   ├── resource_manager.cpp
│   ├── string_class.cpp
│   ├── container_class.cpp
│   └── performance_test.cpp
├── include/
│   ├── resource_manager.h
│   ├── string_class.h
│   ├── container_class.h
│   └── performance_test.h
├── tests/
│   ├── test_move_semantics.cpp
│   └── test_performance.cpp
└── CMakeLists.txt
```

## Implementation Details

### 1. Custom String Class with Move Semantics

Create a custom string class that demonstrates proper move semantics implementation.

```cpp
// include/string_class.h
#pragma once
#include <iostream>
#include <cstring>
#include <utility>

class MyString {
private:
    char* data_;
    size_t size_;
    size_t capacity_;
    
    void allocate(size_t capacity);
    void deallocate();
    
public:
    // Constructors
    MyString();
    MyString(const char* str);
    MyString(size_t count, char ch);
    
    // Copy constructor
    MyString(const MyString& other);
    
    // Move constructor
    MyString(MyString&& other) noexcept;
    
    // Copy assignment operator
    MyString& operator=(const MyString& other);
    
    // Move assignment operator
    MyString& operator=(MyString&& other) noexcept;
    
    // Destructor
    ~MyString();
    
    // Utility methods
    const char* c_str() const;
    size_t size() const;
    size_t capacity() const;
    bool empty() const;
    
    void reserve(size_t new_capacity);
    void resize(size_t new_size);
    void push_back(char ch);
    void append(const char* str);
    void clear();
    
    // Operators
    char& operator[](size_t index);
    const char& operator[](size_t index) const;
    MyString operator+(const MyString& other) const;
    MyString& operator+=(const MyString& other);
    
    bool operator==(const MyString& other) const;
    bool operator!=(const MyString& other) const;
    
    // Friend functions
    friend std::ostream& operator<<(std::ostream& os, const MyString& str);
    friend std::istream& operator>>(std::istream& is, MyString& str);
    
    // Debug methods
    void print_debug_info() const;
    static size_t get_construction_count();
    static size_t get_copy_count();
    static size_t get_move_count();
    static void reset_counters();
    
private:
    static size_t construction_count_;
    static size_t copy_count_;
    static size_t move_count_;
};
```

```cpp
// src/string_class.cpp
#include "string_class.h"
#include <algorithm>
#include <stdexcept>

// Static member initialization
size_t MyString::construction_count_ = 0;
size_t MyString::copy_count_ = 0;
size_t MyString::move_count_ = 0;

void MyString::allocate(size_t capacity) {
    if (capacity > 0) {
        data_ = new char[capacity];
        capacity_ = capacity;
    } else {
        data_ = nullptr;
        capacity_ = 0;
    }
}

void MyString::deallocate() {
    delete[] data_;
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
}

// Default constructor
MyString::MyString() : data_(nullptr), size_(0), capacity_(0) {
    ++construction_count_;
    std::cout << "MyString default constructor" << std::endl;
}

// C-string constructor
MyString::MyString(const char* str) : data_(nullptr), size_(0), capacity_(0) {
    ++construction_count_;
    if (str) {
        size_ = strlen(str);
        allocate(size_ + 1);
        strcpy(data_, str);
    }
    std::cout << "MyString c-string constructor: \"" << (str ? str : "") << "\"" << std::endl;
}

// Fill constructor
MyString::MyString(size_t count, char ch) : data_(nullptr), size_(count), capacity_(0) {
    ++construction_count_;
    if (count > 0) {
        allocate(count + 1);
        std::fill(data_, data_ + count, ch);
        data_[count] = '\0';
    }
    std::cout << "MyString fill constructor: " << count << " chars of '" << ch << "'" << std::endl;
}

// Copy constructor
MyString::MyString(const MyString& other) : data_(nullptr), size_(other.size_), capacity_(0) {
    ++construction_count_;
    ++copy_count_;
    std::cout << "MyString COPY constructor from: \"" << other.c_str() << "\"" << std::endl;
    
    if (other.data_) {
        allocate(size_ + 1);
        strcpy(data_, other.data_);
    }
}

// Move constructor
MyString::MyString(MyString&& other) noexcept 
    : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    ++construction_count_;
    ++move_count_;
    std::cout << "MyString MOVE constructor from: \"" << (other.data_ ? other.data_ : "") << "\"" << std::endl;
    
    // Reset other object
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
}

// Copy assignment operator
MyString& MyString::operator=(const MyString& other) {
    ++copy_count_;
    std::cout << "MyString COPY assignment from: \"" << other.c_str() << "\"" << std::endl;
    
    if (this != &other) {
        deallocate();
        size_ = other.size_;
        
        if (other.data_) {
            allocate(size_ + 1);
            strcpy(data_, other.data_);
        }
    }
    return *this;
}

// Move assignment operator
MyString& MyString::operator=(MyString&& other) noexcept {
    ++move_count_;
    std::cout << "MyString MOVE assignment from: \"" << (other.data_ ? other.data_ : "") << "\"" << std::endl;
    
    if (this != &other) {
        deallocate();
        
        // Take ownership of other's resources
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        
        // Reset other object
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    return *this;
}

// Destructor
MyString::~MyString() {
    std::cout << "MyString destructor: \"" << (data_ ? data_ : "") << "\"" << std::endl;
    deallocate();
}

// Utility methods implementation
const char* MyString::c_str() const {
    return data_ ? data_ : "";
}

size_t MyString::size() const {
    return size_;
}

size_t MyString::capacity() const {
    return capacity_;
}

bool MyString::empty() const {
    return size_ == 0;
}

void MyString::print_debug_info() const {
    std::cout << "MyString Debug Info:" << std::endl;
    std::cout << "  Address: " << static_cast<const void*>(this) << std::endl;
    std::cout << "  Data ptr: " << static_cast<const void*>(data_) << std::endl;
    std::cout << "  Size: " << size_ << std::endl;
    std::cout << "  Capacity: " << capacity_ << std::endl;
    std::cout << "  Content: \"" << c_str() << "\"" << std::endl;
}

// Static methods
size_t MyString::get_construction_count() { return construction_count_; }
size_t MyString::get_copy_count() { return copy_count_; }
size_t MyString::get_move_count() { return move_count_; }

void MyString::reset_counters() {
    construction_count_ = 0;
    copy_count_ = 0;
    move_count_ = 0;
}

// Operators
std::ostream& operator<<(std::ostream& os, const MyString& str) {
    os << str.c_str();
    return os;
}
```

### 2. Resource Manager Class

```cpp
// include/resource_manager.h
#pragma once
#include <iostream>
#include <memory>
#include <vector>

class ResourceManager {
private:
    std::unique_ptr<int[]> data_;
    size_t size_;
    std::string name_;
    
    static size_t next_id_;
    size_t id_;
    
    static size_t total_constructions_;
    static size_t total_copies_;
    static size_t total_moves_;
    
public:
    // Constructors
    ResourceManager(const std::string& name, size_t size);
    
    // Copy constructor
    ResourceManager(const ResourceManager& other);
    
    // Move constructor
    ResourceManager(ResourceManager&& other) noexcept;
    
    // Copy assignment
    ResourceManager& operator=(const ResourceManager& other);
    
    // Move assignment
    ResourceManager& operator=(ResourceManager&& other) noexcept;
    
    // Destructor
    ~ResourceManager();
    
    // Accessors
    const std::string& name() const { return name_; }
    size_t size() const { return size_; }
    size_t id() const { return id_; }
    const int* data() const { return data_.get(); }
    
    // Utility methods
    void fill_with_value(int value);
    void print_info() const;
    bool is_valid() const;
    
    // Factory methods
    static ResourceManager create_small(const std::string& name);
    static ResourceManager create_large(const std::string& name);
    static std::vector<ResourceManager> create_multiple(const std::string& base_name, size_t count);
    
    // Statistics
    static void print_statistics();
    static void reset_statistics();
};
```

### 3. Performance Testing Framework

```cpp
// include/performance_test.h
#pragma once
#include <chrono>
#include <vector>
#include <functional>
#include <string>

class PerformanceTest {
public:
    using TestFunction = std::function<void()>;
    
    struct TestResult {
        std::string test_name;
        double execution_time_ms;
        size_t operations_count;
        double operations_per_second;
    };
    
    static TestResult run_test(const std::string& name, TestFunction test_func, size_t operations = 1);
    static void compare_tests(const std::vector<std::pair<std::string, TestFunction>>& tests, size_t operations = 1);
    
    // Specific move semantics tests
    static void test_string_copy_vs_move();
    static void test_vector_operations();
    static void test_resource_manager_operations();
    static void test_perfect_forwarding();
    
private:
    static double get_execution_time_ms(TestFunction func);
};
```

### 4. Main Application

```cpp
// src/main.cpp
#include <iostream>
#include <vector>
#include <string>
#include "string_class.h"
#include "resource_manager.h"
#include "performance_test.h"

void demonstrate_basic_move_semantics() {
    std::cout << "\n=== Basic Move Semantics Demo ===" << std::endl;
    
    MyString::reset_counters();
    
    {
        std::cout << "\n1. Creating strings:" << std::endl;
        MyString str1("Hello");
        MyString str2("World");
        
        std::cout << "\n2. Copy operations:" << std::endl;
        MyString str3 = str1;  // Copy constructor
        MyString str4;
        str4 = str2;           // Copy assignment
        
        std::cout << "\n3. Move operations:" << std::endl;
        MyString str5 = std::move(str1);  // Move constructor
        MyString str6;
        str6 = std::move(str2);           // Move assignment
        
        std::cout << "\n4. After moves:" << std::endl;
        str1.print_debug_info();
        str2.print_debug_info();
        str5.print_debug_info();
        str6.print_debug_info();
    }
    
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "Constructions: " << MyString::get_construction_count() << std::endl;
    std::cout << "Copies: " << MyString::get_copy_count() << std::endl;
    std::cout << "Moves: " << MyString::get_move_count() << std::endl;
}

void demonstrate_resource_manager() {
    std::cout << "\n=== Resource Manager Demo ===" << std::endl;
    
    ResourceManager::reset_statistics();
    
    {
        std::cout << "\n1. Creating resource managers:" << std::endl;
        auto rm1 = ResourceManager::create_small("Resource1");
        auto rm2 = ResourceManager::create_large("Resource2");
        
        std::cout << "\n2. Copy operations:" << std::endl;
        ResourceManager rm3 = rm1;  // Copy constructor
        ResourceManager rm4("Empty", 0);
        rm4 = rm2;                  // Copy assignment
        
        std::cout << "\n3. Move operations:" << std::endl;
        ResourceManager rm5 = std::move(rm1);  // Move constructor
        ResourceManager rm6("Temp", 10);
        rm6 = std::move(rm2);                  // Move assignment
        
        std::cout << "\n4. After operations:" << std::endl;
        rm1.print_info();
        rm2.print_info();
        rm3.print_info();
        rm4.print_info();
        rm5.print_info();
        rm6.print_info();
    }
    
    ResourceManager::print_statistics();
}

template<typename T>
void perfect_forward_demo(T&& arg) {
    std::cout << "Perfect forwarding: " << std::forward<T>(arg) << std::endl;
}

void demonstrate_perfect_forwarding() {
    std::cout << "\n=== Perfect Forwarding Demo ===" << std::endl;
    
    MyString str("Forwarded");
    
    std::cout << "1. Forwarding lvalue:" << std::endl;
    perfect_forward_demo(str);
    
    std::cout << "\n2. Forwarding rvalue:" << std::endl;
    perfect_forward_demo(MyString("Temporary"));
    
    std::cout << "\n3. Forwarding moved value:" << std::endl;
    perfect_forward_demo(std::move(str));
}

int main() {
    try {
        demonstrate_basic_move_semantics();
        demonstrate_resource_manager();
        demonstrate_perfect_forwarding();
        
        std::cout << "\n=== Performance Tests ===" << std::endl;
        PerformanceTest::test_string_copy_vs_move();
        PerformanceTest::test_vector_operations();
        PerformanceTest::test_resource_manager_operations();
        
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
project(MoveSemantics)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/string_class.cpp
    src/resource_manager.cpp
    src/performance_test.cpp
)

# Create executable
add_executable(move_semantics ${SOURCES})

# Compiler-specific options
if(MSVC)
    target_compile_options(move_semantics PRIVATE /W4)
else()
    target_compile_options(move_semantics PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Optional: Add tests if you have a testing framework
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(move_semantics_tests
        tests/test_move_semantics.cpp
        src/string_class.cpp
        src/resource_manager.cpp
    )
    target_link_libraries(move_semantics_tests GTest::gtest_main)
    
    enable_testing()
    add_test(NAME MoveSemantics_Tests COMMAND move_semantics_tests)
endif()
```

## Testing Framework

```cpp
// tests/test_move_semantics.cpp
#include <gtest/gtest.h>
#include "string_class.h"
#include "resource_manager.h"

class MoveSemanticTests : public ::testing::Test {
protected:
    void SetUp() override {
        MyString::reset_counters();
        ResourceManager::reset_statistics();
    }
};

TEST_F(MoveSemanticTests, StringMoveConstructor) {
    MyString original("Test String");
    const char* original_data = original.c_str();
    
    MyString moved(std::move(original));
    
    EXPECT_STREQ(moved.c_str(), "Test String");
    EXPECT_TRUE(original.empty());
    EXPECT_EQ(MyString::get_move_count(), 1);
}

TEST_F(MoveSemanticTests, StringMoveAssignment) {
    MyString str1("First");
    MyString str2("Second");
    
    str1 = std::move(str2);
    
    EXPECT_STREQ(str1.c_str(), "Second");
    EXPECT_TRUE(str2.empty());
    EXPECT_EQ(MyString::get_move_count(), 1);
}

TEST_F(MoveSemanticTests, ResourceManagerMove) {
    auto rm1 = ResourceManager::create_small("Test");
    size_t original_id = rm1.id();
    
    ResourceManager rm2 = std::move(rm1);
    
    EXPECT_EQ(rm2.id(), original_id);
    EXPECT_FALSE(rm1.is_valid());
}

TEST_F(MoveSemanticTests, PerformanceImprovement) {
    const size_t test_size = 1000;
    
    // Test copy performance
    auto copy_time = PerformanceTest::run_test("Copy Test", [test_size]() {
        std::vector<MyString> strings;
        strings.reserve(test_size);
        
        for (size_t i = 0; i < test_size; ++i) {
            MyString str("Test string " + std::to_string(i));
            strings.push_back(str);  // Copy
        }
    });
    
    MyString::reset_counters();
    
    // Test move performance
    auto move_time = PerformanceTest::run_test("Move Test", [test_size]() {
        std::vector<MyString> strings;
        strings.reserve(test_size);
        
        for (size_t i = 0; i < test_size; ++i) {
            MyString str("Test string " + std::to_string(i));
            strings.push_back(std::move(str));  // Move
        }
    });
    
    // Move should be significantly faster
    EXPECT_LT(move_time.execution_time_ms, copy_time.execution_time_ms);
}
```

## Expected Learning Outcomes

After completing this project, you should understand:

1. **Move Semantics Fundamentals**
   - Difference between lvalues and rvalues
   - When move constructors and assignment operators are called
   - Resource transfer vs. resource copying

2. **Implementation Best Practices**
   - Exception safety in move operations (noexcept)
   - Proper resource cleanup and object state management
   - Self-assignment protection

3. **Performance Benefits**
   - Measurable performance improvements
   - Memory allocation reduction
   - Optimal resource utilization

4. **Integration with Standard Library**
   - How std::move works
   - Perfect forwarding with std::forward
   - Move-aware containers and algorithms

## Extensions and Improvements

1. **Advanced Features**
   - Implement move-only types
   - Add custom allocators
   - Implement move semantics for container classes

2. **Performance Analysis**
   - Memory profiling tools integration
   - Benchmark different scenarios
   - Compare with standard library implementations

3. **Real-world Applications**
   - File handling with move semantics
   - Network buffer management
   - Game object systems

This project provides a comprehensive foundation for understanding and implementing move semantics in C++11, with practical applications and performance validation.
