# Memory Leak Detection Techniques

*Duration: 1-2 weeks*

## Overview

Memory leaks are one of the most common and critical issues in C/C++ programming. They occur when dynamically allocated memory is not properly deallocated, leading to gradual memory consumption that can eventually exhaust system resources. This guide covers comprehensive techniques for detecting, analyzing, and preventing memory leaks using static analysis, dynamic analysis, and manual review methodologies.

### What Are Memory Leaks?

A **memory leak** occurs when:
1. Memory is allocated dynamically (`malloc`, `new`, etc.)
2. The program loses all references to that memory
3. The memory is never freed (`free`, `delete`, etc.)
4. The memory remains allocated until program termination

#### Types of Memory Issues

| Issue Type | Description | Example |
|------------|-------------|---------|
| **Memory Leak** | Allocated memory never freed | `malloc()` without `free()` |
| **Double Free** | Freeing same memory twice | `free(ptr); free(ptr);` |
| **Use After Free** | Accessing freed memory | `free(ptr); *ptr = 5;` |
| **Buffer Overflow** | Writing beyond allocated bounds | `arr[10] = x;` when `arr` has 10 elements |
| **Uninitialized Memory** | Reading uninitialized values | `int x; return x;` |

## Static Analysis Techniques

Static analysis examines source code without executing it, identifying potential memory leaks through code patterns and flow analysis.

### 1. Compiler Warnings and Static Analyzers

#### GCC/Clang Built-in Analysis
```bash
# Enable comprehensive warnings
gcc -Wall -Wextra -Wunused -Wunreachable-code -Wmissing-declarations \
    -Wmissing-prototypes -Wstrict-prototypes -g -O2 program.c

# Clang static analyzer
clang --analyze -Xanalyzer -analyzer-output=html program.c
```

#### Example: Basic Memory Leak Detection
```cpp
#include <iostream>
#include <memory>

// BAD: Static analyzer will detect this leak
int* create_leaky_array() {
    int* p = new int[10];
    
    // Static analyzer detects: allocated memory never freed
    // Path: allocation -> return -> memory lost
    return p;  // Caller responsibility unclear
}

// GOOD: Clear ownership with smart pointers
std::unique_ptr<int[]> create_safe_array() {
    return std::make_unique<int[]>(10);
    // Automatic cleanup when unique_ptr goes out of scope
}

// GOOD: Clear ownership documentation
int* create_array_with_docs() {
    // IMPORTANT: Caller must delete[] the returned pointer
    int* p = new int[10];
    return p;
}

void example_usage() {
    // Leak detection in action
    int* leaky = create_leaky_array();
    // Static analyzer warns: 'leaky' never freed
    
    // Safe usage
    auto safe = create_safe_array();
    // No warning - automatic cleanup
    
    // Manual but documented
    int* manual = create_array_with_docs();
    // Must remember to clean up
    delete[] manual;
}
```

### 2. Advanced Static Analysis Tools

#### Clang Static Analyzer
```bash
# Comprehensive analysis with HTML output
scan-build make

# Or for single file
clang --analyze -Xanalyzer -analyzer-output=text \
      -Xanalyzer -analyzer-checker=core,cplusplus,deadcode,security \
      program.cpp
```

**Example Configuration (.clang-tidy):**
```yaml
Checks: >
  clang-analyzer-*,
  cppcoreguidelines-*,
  performance-*,
  readability-*,
  -readability-magic-numbers
  
CheckOptions:
  - key: cppcoreguidelines-avoid-magic-numbers.IgnoredIntegerValues
    value: '0;1;2;3;4;5;6;7;8;9;10'
```

#### PC-lint/PC-lint Plus
```cpp
// Configure PC-lint for memory leak detection
// lint-config.lnt
-e429    // Custodial pointer not freed or returned
-e769    // Custodial pointer possibly not freed
-e593    // Custodial pointer possibly not freed on error path

// Example code that triggers PC-lint warnings
void problematic_function(bool condition) {
    char* buffer = new char[1024];  // +429: Custodial pointer
    
    if (condition) {
        // Error path - buffer not freed
        return;  // Warning: Custodial pointer not freed
    }
    
    process_buffer(buffer);
    // Missing delete[] buffer; - Warning triggered
}
```

#### Coverity Static Analysis
```cpp
// Coverity detects complex leak patterns
class ResourceManager {
private:
    char* data_;
    size_t size_;
    
public:
    ResourceManager(size_t sz) : size_(sz) {
        data_ = new char[sz];  // Coverity tracks this allocation
    }
    
    // BAD: Missing destructor - Coverity detects leak
    // ~ResourceManager() { delete[] data_; }  // Should have this
    
    // BAD: Copy constructor can cause double-delete
    ResourceManager(const ResourceManager& other) : size_(other.size_) {
        data_ = other.data_;  // Shallow copy - Coverity warns
    }
    
    // GOOD: Proper copy constructor
    ResourceManager(const ResourceManager& other) : size_(other.size_) {
        data_ = new char[size_];
        memcpy(data_, other.data_, size_);
    }
};
```

### 3. Manual Code Review Patterns

#### Common Leak Patterns to Look For

**Pattern 1: Exception Safety Issues**
```cpp
// BAD: Exception can cause leak
void risky_function() {
    char* buffer = new char[1024];
    
    // If this throws, buffer leaks
    risky_operation_that_may_throw();
    
    delete[] buffer;  // May never be reached
}

// GOOD: RAII (Resource Acquisition Is Initialization)
void safe_function() {
    std::unique_ptr<char[]> buffer(new char[1024]);
    
    // Exception safe - automatic cleanup
    risky_operation_that_may_throw();
    
    // No explicit delete needed
}
```

**Pattern 2: Conditional Cleanup**
```cpp
// BAD: Complex conditional logic with potential leaks
int* allocate_conditionally(int type, bool extra_data) {
    int* base = new int[100];
    
    if (type == 1) {
        int* extra = new int[50];
        if (!extra_data) {
            delete[] base;  // Cleanup base
            return nullptr; // But 'extra' leaks!
        }
        // Process both base and extra
        merge_arrays(base, extra);
        delete[] extra;
    }
    
    return base;  // Caller must delete
}

// GOOD: Early returns with proper cleanup
std::unique_ptr<int[]> allocate_conditionally_safe(int type, bool extra_data) {
    auto base = std::make_unique<int[]>(100);
    
    if (type == 1) {
        auto extra = std::make_unique<int[]>(50);
        if (!extra_data) {
            return nullptr;  // Both automatically cleaned up
        }
        merge_arrays(base.get(), extra.get());
    }
    
    return base;  // Transfer ownership
}
```

**Pattern 3: Loop Allocation Issues**
```cpp
// BAD: Leak in loop with early exit
std::vector<char*> process_files(const std::vector<std::string>& filenames) {
    std::vector<char*> buffers;
    
    for (const auto& filename : filenames) {
        char* buffer = new char[1024];
        
        if (!read_file(filename, buffer)) {
            // Early return leaks all allocated buffers!
            return {};
        }
        
        buffers.push_back(buffer);
    }
    
    return buffers;  // Caller must clean up all buffers
}

// GOOD: Exception-safe loop with RAII
std::vector<std::unique_ptr<char[]>> process_files_safe(
    const std::vector<std::string>& filenames) {
    
    std::vector<std::unique_ptr<char[]>> buffers;
    
    for (const auto& filename : filenames) {
        auto buffer = std::make_unique<char[]>(1024);
        
        if (!read_file(filename, buffer.get())) {
            // Early return automatically cleans up all buffers
            return {};
        }
        
        buffers.push_back(std::move(buffer));
    }
    
    return buffers;  // Automatic cleanup for caller too
}
```

## Dynamic Analysis Techniques

Dynamic analysis detects memory leaks by monitoring memory allocation and deallocation during program execution.

### 1. Valgrind (Linux/macOS)

Valgrind is the gold standard for memory leak detection on Unix-like systems.

#### Basic Valgrind Usage
```bash
# Compile with debug information
gcc -g -O0 program.c -o program

# Run with Valgrind
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./program

# Advanced options
valgrind --tool=memcheck \
         --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         --log-file=valgrind-output.txt \
         ./program
```

#### Example Program with Leaks
```cpp
#include <iostream>
#include <cstdlib>
#include <cstring>

class LeakyClass {
private:
    char* data_;
    int* numbers_;
    
public:
    LeakyClass(size_t size) {
        data_ = new char[size];           // Will be leaked
        numbers_ = new int[100];          // Will be leaked
        
        // Initialize data
        memset(data_, 0, size);
        for (int i = 0; i < 100; i++) {
            numbers_[i] = i;
        }
    }
    
    // Missing destructor - causes leaks!
    // ~LeakyClass() {
    //     delete[] data_;
    //     delete[] numbers_;
    // }
    
    void process() {
        // Temporary allocation
        char* temp = new char[50];        // Will be leaked
        strcpy(temp, "temporary data");
        
        // Process data but forget to free temp
        std::cout << "Processing: " << temp << std::endl;
        // delete[] temp;  // Missing!
    }
};

void demonstrate_leaks() {
    // 1. Simple malloc leak
    char* buffer = (char*)malloc(1024);
    // Missing free(buffer);
    
    // 2. C++ new leak
    int* array = new int[500];
    // Missing delete[] array;
    
    // 3. Class instance leak
    LeakyClass* obj = new LeakyClass(2048);
    obj->process();
    // Missing delete obj; (which would still leak internal data)
    
    // 4. Conditional leak
    if (true) {
        double* temp_data = new double[100];
        // Conditional cleanup missing
        // delete[] temp_data;
    }
}

int main() {
    demonstrate_leaks();
    
    std::cout << "Program completed (with leaks)" << std::endl;
    return 0;
}
```

#### Valgrind Output Analysis
```bash
# Valgrind will output something like:
==12345== HEAP SUMMARY:
==12345==     in use at exit: 10,448 bytes in 6 blocks
==12345==   total heap usage: 8 allocs, 2 frees, 11,272 bytes allocated
==12345== 
==12345== 1,024 bytes in 1 blocks are definitely lost in loss record 1 of 6
==12345==    at 0x4C2FB0F: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x108A1B: demonstrate_leaks() (program.cpp:45)
==12345==    by 0x108B2C: main (program.cpp:67)
==12345== 
==12345== 2,048 bytes in 1 blocks are definitely lost in loss record 2 of 6
==12345==    at 0x4C3089F: operator new[](unsigned long) (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x108756: LeakyClass::LeakyClass(unsigned long) (program.cpp:12)
==12345==    by 0x108A45: demonstrate_leaks() (program.cpp:52)
==12345==    by 0x108B2C: main (program.cpp:67)
```

### 2. AddressSanitizer (ASan)

AddressSanitizer is a fast memory error detector built into GCC and Clang.

#### Compilation and Usage
```bash
# Compile with AddressSanitizer
gcc -fsanitize=address -fno-omit-frame-pointer -g program.c -o program

# Run the program
./program

# For C++
g++ -fsanitize=address -fno-omit-frame-pointer -g -std=c++17 program.cpp -o program
```

#### ASan Example
```cpp
#include <iostream>
#include <vector>

void demonstrate_asan_detection() {
    // 1. Heap buffer overflow
    int* arr = new int[10];
    arr[15] = 42;  // ASan detects: heap-buffer-overflow
    delete[] arr;
    
    // 2. Use after free
    char* buffer = new char[100];
    delete[] buffer;
    buffer[0] = 'x';  // ASan detects: heap-use-after-free
    
    // 3. Double free
    int* data = new int[5];
    delete[] data;
    delete[] data;  // ASan detects: attempting double-free
    
    // 4. Stack buffer overflow
    char local_array[10];
    local_array[20] = 'a';  // ASan detects: stack-buffer-overflow
}

// ASan with custom memory allocation
class TrackedAllocator {
public:
    static void* allocate(size_t size) {
        void* ptr = malloc(size);
        std::cout << "Allocated " << size << " bytes at " << ptr << std::endl;
        return ptr;
    }
    
    static void deallocate(void* ptr) {
        std::cout << "Deallocating " << ptr << std::endl;
        free(ptr);
    }
};

int main() {
    // Enable ASan options via environment variables
    // export ASAN_OPTIONS="abort_on_error=1:detect_leaks=1:check_initialization_order=1"
    
    demonstrate_asan_detection();
    return 0;
}
```

### 3. Dr. Memory (Windows)

Dr. Memory is a powerful memory debugger for Windows applications.

#### Dr. Memory Usage
```cmd
REM Compile with debug information
cl /Zi /Od program.cpp

REM Run with Dr. Memory
drmemory.exe -- program.exe

REM Advanced options
drmemory.exe -brief -batch -logdir logs -- program.exe
```

#### Example for Dr. Memory
```cpp
#include <windows.h>
#include <iostream>

void windows_memory_issues() {
    // 1. Heap allocation leak
    LPVOID heap_mem = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, 1024);
    // Missing HeapFree(GetProcessHeap(), 0, heap_mem);
    
    // 2. VirtualAlloc leak
    LPVOID virtual_mem = VirtualAlloc(NULL, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    // Missing VirtualFree(virtual_mem, 0, MEM_RELEASE);
    
    // 3. C++ allocation
    char* cpp_buffer = new char[2048];
    // Missing delete[] cpp_buffer;
    
    // 4. Buffer overflow
    char stack_buffer[10];
    strcpy_s(stack_buffer, sizeof(stack_buffer), "This string is too long for the buffer");  // Dr. Memory detects overflow
}

int main() {
    windows_memory_issues();
    
    std::cout << "Windows program with memory issues" << std::endl;
    return 0;
}
```

### 4. Custom Memory Tracking

Implementing your own memory tracking system for debugging.

```cpp
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <cstdlib>

class MemoryTracker {
private:
    struct AllocationInfo {
        size_t size;
        const char* file;
        int line;
        const char* function;
    };
    
    static std::unordered_map<void*, AllocationInfo> allocations_;
    static std::mutex mutex_;
    static size_t total_allocated_;
    static size_t total_deallocated_;
    
public:
    static void* track_malloc(size_t size, const char* file, int line, const char* func) {
        void* ptr = malloc(size);
        if (ptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            allocations_[ptr] = {size, file, line, func};
            total_allocated_ += size;
            
            std::cout << "ALLOC: " << ptr << " (" << size << " bytes) at " 
                      << file << ":" << line << " in " << func << std::endl;
        }
        return ptr;
    }
    
    static void track_free(void* ptr, const char* file, int line, const char* func) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            total_deallocated_ += it->second.size;
            std::cout << "FREE:  " << ptr << " (" << it->second.size << " bytes) at " 
                      << file << ":" << line << " in " << func << std::endl;
            allocations_.erase(it);
        } else {
            std::cout << "ERROR: Attempting to free untracked pointer " << ptr 
                      << " at " << file << ":" << line << " in " << func << std::endl;
        }
        
        free(ptr);
    }
    
    static void report_leaks() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cout << "\n=== MEMORY LEAK REPORT ===" << std::endl;
        std::cout << "Total allocated: " << total_allocated_ << " bytes" << std::endl;
        std::cout << "Total deallocated: " << total_deallocated_ << " bytes" << std::endl;
        std::cout << "Leaked: " << (total_allocated_ - total_deallocated_) << " bytes" << std::endl;
        
        if (!allocations_.empty()) {
            std::cout << "\nLEAKED ALLOCATIONS:" << std::endl;
            for (const auto& alloc : allocations_) {
                std::cout << "  " << alloc.first << " (" << alloc.second.size << " bytes) "
                          << "allocated at " << alloc.second.file << ":" << alloc.second.line
                          << " in " << alloc.second.function << std::endl;
            }
        } else {
            std::cout << "\nNo memory leaks detected!" << std::endl;
        }
    }
};

// Static member definitions
std::unordered_map<void*, MemoryTracker::AllocationInfo> MemoryTracker::allocations_;
std::mutex MemoryTracker::mutex_;
size_t MemoryTracker::total_allocated_ = 0;
size_t MemoryTracker::total_deallocated_ = 0;

// Macros for easy tracking
#define TRACKED_MALLOC(size) MemoryTracker::track_malloc(size, __FILE__, __LINE__, __FUNCTION__)
#define TRACKED_FREE(ptr) MemoryTracker::track_free(ptr, __FILE__, __LINE__, __FUNCTION__)

// Example usage
void test_custom_tracking() {
    // This will be tracked
    char* buffer1 = (char*)TRACKED_MALLOC(1024);
    char* buffer2 = (char*)TRACKED_MALLOC(2048);
    
    // Free one buffer
    TRACKED_FREE(buffer1);
    
    // buffer2 will be reported as leaked
}

int main() {
    test_custom_tracking();
    
    // Report leaks before exit
    MemoryTracker::report_leaks();
    
    return 0;
}
```

### 5. Heap Profiling Tools

#### tcmalloc and Heap Profiler
```bash
# Install tcmalloc
sudo apt-get install libtcmalloc-minimal4

# Compile with tcmalloc
g++ -ltcmalloc program.cpp -o program

# Run with heap profiling
export HEAPPROFILE=/tmp/myprogram.heap
export HEAPCHECK=normal
./program

# Analyze heap profile
google-pprof --text ./program /tmp/myprogram.heap.0001.heap
```

#### jemalloc Profiling
```bash
# Compile with jemalloc
g++ -ljemalloc program.cpp -o program

# Run with profiling
export MALLOC_CONF="prof:true,prof_final:true"
./program

# Generate profile
jeprof --show_bytes --pdf ./program jeprof.*.heap > profile.pdf
```

## Prevention Strategies and Best Practices

### 1. RAII (Resource Acquisition Is Initialization)

RAII is the most effective technique for preventing memory leaks in C++.

#### Smart Pointers
```cpp
#include <memory>
#include <vector>
#include <iostream>

// BAD: Manual memory management
class OldStyleManager {
private:
    int* data_;
    size_t size_;
    
public:
    OldStyleManager(size_t sz) : size_(sz) {
        data_ = new int[sz];  // Manual allocation
    }
    
    ~OldStyleManager() {
        delete[] data_;  // Must remember to delete
    }
    
    // Need to implement copy constructor and assignment operator
    // to avoid double-delete issues (Rule of Three/Five)
};

// GOOD: Modern C++ with smart pointers
class ModernManager {
private:
    std::unique_ptr<int[]> data_;
    size_t size_;
    
public:
    ModernManager(size_t sz) : size_(sz) {
        data_ = std::make_unique<int[]>(sz);  // Automatic cleanup
    }
    
    // No destructor needed - automatic cleanup
    // Copy semantics handled automatically or can be explicitly deleted
    
    // Move semantics work out of the box
    ModernManager(ModernManager&& other) noexcept = default;
    ModernManager& operator=(ModernManager&& other) noexcept = default;
    
    // Disable copying if not needed
    ModernManager(const ModernManager&) = delete;
    ModernManager& operator=(const ModernManager&) = delete;
};

// Example: Smart pointer usage patterns
void demonstrate_smart_pointers() {
    // 1. unique_ptr for single ownership
    auto single_owner = std::make_unique<int>(42);
    // Automatically deleted when single_owner goes out of scope
    
    // 2. shared_ptr for shared ownership
    auto shared1 = std::make_shared<std::vector<int>>(1000);
    auto shared2 = shared1;  // Reference count increases
    // Deleted when last shared_ptr is destroyed
    
    // 3. weak_ptr to break circular references
    std::shared_ptr<Node> parent = std::make_shared<Node>();
    std::shared_ptr<Node> child = std::make_shared<Node>();
    parent->child = child;
    child->parent = std::weak_ptr<Node>(parent);  // Break cycle
    
    // 4. Custom deleters
    auto file_ptr = std::unique_ptr<FILE, decltype(&fclose)>(
        fopen("test.txt", "w"), &fclose
    );
    // File automatically closed when file_ptr goes out of scope
}
```

### 2. Exception Safety and Error Handling

```cpp
#include <exception>
#include <memory>
#include <iostream>

// BAD: Exception unsafe
void unsafe_function() {
    char* buffer1 = new char[1024];
    char* buffer2 = new char[2048];
    
    try {
        risky_operation();  // May throw exception
    } catch (...) {
        // If exception occurs, buffers leak!
        delete[] buffer1;
        delete[] buffer2;
        throw;  // Re-throw
    }
    
    delete[] buffer1;
    delete[] buffer2;
}

// GOOD: Exception safe with RAII
void safe_function() {
    auto buffer1 = std::make_unique<char[]>(1024);
    auto buffer2 = std::make_unique<char[]>(2048);
    
    risky_operation();  // If exception occurs, automatic cleanup
    
    // No explicit cleanup needed
}
```

### 3. Modern C++ Alternatives

#### Use Standard Containers
```cpp
// BAD: Manual array management
char* create_buffer(size_t size) {
    return new char[size];  // Caller must remember to delete[]
}

// GOOD: Use vector
std::vector<char> create_buffer_safe(size_t size) {
    return std::vector<char>(size);  // Automatic memory management
}

// GOOD: Use string for text data
std::string create_text_buffer(size_t size) {
    return std::string(size, '\0');  // Automatic memory management
}
```

## Advanced Detection Techniques

### 1. Memory Leak Detection in Production

#### Custom Allocation Tracking
```cpp
#include <atomic>
#include <unordered_map>
#include <mutex>

class ProductionMemoryTracker {
private:
    static std::atomic<size_t> bytes_allocated_;
    static std::atomic<size_t> bytes_deallocated_;
    static std::atomic<size_t> allocation_count_;
    static std::atomic<size_t> deallocation_count_;
    
public:
    static void record_allocation(size_t size) {
        bytes_allocated_.fetch_add(size);
        allocation_count_.fetch_add(1);
    }
    
    static void record_deallocation(size_t size) {
        bytes_deallocated_.fetch_add(size);
        deallocation_count_.fetch_add(1);
    }
    
    static void print_stats() {
        size_t allocated = bytes_allocated_.load();
        size_t deallocated = bytes_deallocated_.load();
        size_t alloc_count = allocation_count_.load();
        size_t dealloc_count = deallocation_count_.load();
        
        std::cout << "Memory Stats:" << std::endl;
        std::cout << "  Total allocated: " << allocated << " bytes (" << alloc_count << " allocations)" << std::endl;
        std::cout << "  Total deallocated: " << deallocated << " bytes (" << dealloc_count << " deallocations)" << std::endl;
        std::cout << "  Current usage: " << (allocated - deallocated) << " bytes" << std::endl;
        std::cout << "  Potential leaks: " << (alloc_count - dealloc_count) << " allocations" << std::endl;
    }
};

// Static member definitions
std::atomic<size_t> ProductionMemoryTracker::bytes_allocated_{0};
std::atomic<size_t> ProductionMemoryTracker::bytes_deallocated_{0};
std::atomic<size_t> ProductionMemoryTracker::allocation_count_{0};
std::atomic<size_t> ProductionMemoryTracker::deallocation_count_{0};
```

### 2. Integration with CI/CD Pipeline

#### Automated Memory Leak Testing
```bash
#!/bin/bash
# ci_memory_test.sh - Memory leak testing in CI pipeline

set -e

echo "Building with memory leak detection..."
make clean
make CFLAGS="-g -O0 -fsanitize=address" LDFLAGS="-fsanitize=address"

echo "Running memory leak tests..."
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1"

# Run test suite
./run_tests

# Run with Valgrind if available
if command -v valgrind &> /dev/null; then
    echo "Running Valgrind analysis..."
    valgrind --leak-check=full --error-exitcode=1 ./run_tests
fi

echo "Memory leak testing completed successfully"
```

#### Jenkins/GitHub Actions Integration
```yaml
# .github/workflows/memory-check.yml
name: Memory Leak Detection

on: [push, pull_request]

jobs:
  memory-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Install Dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y valgrind clang
    
    - name: Build with AddressSanitizer
      run: |
        clang++ -fsanitize=address -g -O1 -fno-omit-frame-pointer src/*.cpp -o test_program
    
    - name: Run Memory Leak Tests
      run: |
        export ASAN_OPTIONS="detect_leaks=1"
        ./test_program
    
    - name: Run Valgrind Analysis
      run: |
        valgrind --leak-check=full --error-exitcode=1 ./test_program
```

## Learning Objectives

By the end of this section, you should be able to:

- **Identify different types of memory issues** (leaks, double-free, use-after-free, buffer overflows)
- **Use static analysis tools** effectively (Clang Static Analyzer, PC-lint, Coverity)
- **Apply dynamic analysis tools** (Valgrind, AddressSanitizer, Dr. Memory)
- **Implement custom memory tracking** for debugging purposes
- **Apply RAII principles** to prevent memory leaks
- **Use smart pointers** correctly in modern C++
- **Design exception-safe code** that prevents resource leaks
- **Integrate memory leak detection** into development and CI/CD workflows

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Set up and use Valgrind to detect memory leaks  
□ Compile and run programs with AddressSanitizer  
□ Interpret memory leak reports from various tools  
□ Implement RAII using smart pointers  
□ Write exception-safe code that prevents leaks  
□ Design custom memory tracking systems  
□ Configure static analysis tools for your projects  
□ Integrate memory leak detection into CI/CD pipelines  

## Practical Exercises

### Exercise 1: Tool Comparison
Create a program with intentional memory leaks and test it with:
- Valgrind
- AddressSanitizer  
- Static analysis tools
Compare the output and detection capabilities.

### Exercise 2: Custom Memory Tracker
Implement a thread-safe memory tracking system that:
- Records all allocations with stack traces
- Detects double-free attempts
- Reports leaks at program exit
- Provides memory usage statistics

### Exercise 3: Legacy Code Modernization
Take a C-style program with manual memory management and:
- Identify all potential memory issues
- Convert to modern C++ with RAII
- Add proper exception safety
- Verify leak-free operation

## Study Materials

### Essential Reading
- **"Effective Modern C++"** by Scott Meyers - Smart pointers and RAII
- **"C++ Core Guidelines"** - Memory management best practices
- **Valgrind User Manual** - Comprehensive tool documentation
- **AddressSanitizer Documentation** - Google's memory error detector

### Tools Documentation
- [Valgrind Manual](http://valgrind.org/docs/manual/)
- [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html)
- [Clang Static Analyzer](https://clang-analyzer.llvm.org/)
- [Dr. Memory Documentation](http://drmemory.org/)

### Practice Resources
- **Memory leak detection labs**
- **Static analysis tool tutorials**
- **CI/CD integration examples**
- **Performance profiling exercises**

### Advanced Topics
- Memory pool design patterns
- Custom allocator implementation
- Garbage collection techniques
- Real-time memory management
