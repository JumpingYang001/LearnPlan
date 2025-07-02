# Memory Profiling Tools

*Duration: 1 week*

## Overview
Memory profiling tools are essential for detecting, analyzing, and fixing memory-related issues in C/C++ applications. These tools help identify memory leaks, buffer overflows, use-after-free errors, and other memory corruption issues that can lead to crashes, security vulnerabilities, and performance problems.

## Types of Memory Issues These Tools Detect

### Common Memory Problems
1. **Memory Leaks** - Allocated memory not freed
2. **Buffer Overflows** - Writing beyond allocated memory bounds
3. **Use-After-Free** - Accessing freed memory
4. **Double Free** - Freeing the same memory twice
5. **Uninitialized Memory Access** - Reading uninitialized variables
6. **Stack Buffer Overflows** - Overwriting stack memory
7. **Heap Corruption** - Corrupting heap metadata

### Example of Common Memory Issues
```cpp
#include <iostream>
#include <cstring>

// Example program with multiple memory issues for demonstration
class MemoryIssuesDemo {
private:
    int* data;
    size_t size;

public:
    MemoryIssuesDemo(size_t s) : size(s) {
        data = new int[size];
        // Issue 1: Uninitialized memory - data array not initialized
    }
    
    ~MemoryIssuesDemo() {
        delete[] data;
    }
    
    void demonstrateIssues() {
        // Issue 2: Buffer overflow
        for (size_t i = 0; i <= size; i++) {  // Note: <= instead of <
            data[i] = i;  // Writing beyond array bounds
        }
        
        // Issue 3: Memory leak
        int* temp = new int[100];
        // temp is never deleted - memory leak!
        
        // Issue 4: Use after free
        delete[] data;
        data[0] = 42;  // Accessing freed memory
        
        // Issue 5: Double free (will crash)
        // delete[] data;  // Commented out to prevent immediate crash
    }
    
    static void stackOverflow() {
        // Issue 6: Stack buffer overflow
        char buffer[10];
        strcpy(buffer, "This string is much longer than 10 characters");  // Buffer overflow
        std::cout << buffer << std::endl;
    }
};

int main() {
    MemoryIssuesDemo demo(10);
    demo.demonstrateIssues();
    MemoryIssuesDemo::stackOverflow();
    return 0;
}
```

## Valgrind - The Swiss Army Knife of Memory Debugging

**Valgrind** is a powerful instrumentation framework for building dynamic analysis tools. Its most popular tool, Memcheck, detects memory management problems.

### Installation and Setup

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install valgrind
```

**Linux (Red Hat/CentOS):**
```bash
sudo yum install valgrind
# or on newer versions:
sudo dnf install valgrind
```

**macOS:**
```bash
brew install valgrind
```

**Windows:**
Valgrind doesn't run natively on Windows. Use WSL (Windows Subsystem for Linux) or a Linux VM.

### Basic Valgrind Usage

**Simple Memory Leak Detection:**
```bash
# Compile with debug symbols
gcc -g -o program program.c

# Run with Valgrind
valgrind --leak-check=full ./program
```

**Comprehensive Example:**

First, let's create a program with memory issues:

```cpp
// memory_issues.cpp
#include <iostream>
#include <cstdlib>
#include <cstring>

void memory_leak_example() {
    int* arr = new int[100];
    arr[0] = 42;
    // Memory leak: arr is never deleted
}

void buffer_overflow_example() {
    int* arr = new int[10];
    arr[15] = 99;  // Buffer overflow - writing beyond bounds
    delete[] arr;
}

void use_after_free_example() {
    int* ptr = new int(42);
    delete ptr;
    std::cout << *ptr << std::endl;  // Use after free
}

void double_free_example() {
    int* ptr = new int(42);
    delete ptr;
    delete ptr;  // Double free
}

void uninitialized_read_example() {
    int* arr = new int[10];
    // arr is not initialized
    std::cout << arr[5] << std::endl;  // Reading uninitialized memory
    delete[] arr;
}

int main() {
    std::cout << "Demonstrating memory issues...\n";
    
    memory_leak_example();
    buffer_overflow_example();
    // use_after_free_example();  // Commented to prevent crash
    // double_free_example();     // Commented to prevent crash
    uninitialized_read_example();
    
    return 0;
}
```

**Compilation and Analysis:**
```bash
# Compile with debug information
g++ -g -o memory_issues memory_issues.cpp

# Run comprehensive Valgrind analysis
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         --log-file=valgrind-output.txt \
         ./memory_issues
```

**Understanding Valgrind Output:**

```
==12345== Memcheck, a memory error detector
==12345== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==12345== Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info
==12345== Command: ./memory_issues
==12345== 

Demonstrating memory issues...
==12345== Conditional jump or move depends on uninitialised value(s)
==12345==    at 0x4008B2: uninitialized_read_example() (memory_issues.cpp:30)
==12345==    by 0x4008F1: main (memory_issues.cpp:38)
==12345==  Uninitialised value was created by a heap allocation
==12345==    at 0x4005A1: operator new[](unsigned long) (vg_replace_malloc.c:433)
==12345==    at 0x400891: uninitialized_read_example() (memory_issues.cpp:28)

==12345== Invalid write of size 4
==12345==    at 0x400851: buffer_overflow_example() (memory_issues.cpp:18)
==12345==    by 0x4008E8: main (memory_issues.cpp:36)
==12345==  Address 0x5204064 is 20 bytes inside a block of size 40 alloc'd
==12345==    at 0x4005A1: operator new[](unsigned long) (vg_replace_malloc.c:433)
==12345==    at 0x400841: buffer_overflow_example() (memory_issues.cpp:17)

==12345== HEAP SUMMARY:
==12345==     in use at exit: 400 bytes in 1 blocks
==12345==   total heap usage: 3 allocs, 2 frees, 72,752 bytes allocated
==12345== 
==12345== 400 bytes in 1 blocks are definitely lost in loss record 1 of 1
==12345==    at 0x4005A1: operator new[](unsigned long) (vg_replace_malloc.c:433)
==12345==    at 0x400821: memory_leak_example() (memory_issues.cpp:8)
==12345==    by 0x4008E4: main (memory_issues.cpp:35)
```

**Valgrind Command Line Options:**

```bash
# Essential options
valgrind --leak-check=full          # Detailed leak information
valgrind --show-leak-kinds=all      # Show all types of leaks
valgrind --track-origins=yes        # Track origins of uninitialized values
valgrind --verbose                  # Verbose output
valgrind --log-file=output.txt      # Save output to file

# Performance options
valgrind --cache-sim=yes            # Cache simulation
valgrind --branch-sim=yes           # Branch prediction simulation

# Suppression options
valgrind --suppressions=my.supp     # Use custom suppression file
valgrind --gen-suppressions=all     # Generate suppression entries
```

### Advanced Valgrind Features

**1. Custom Suppression Files:**
```
# my_suppressions.supp
{
   ignore_std_string_leak
   Memcheck:Leak
   fun:_Znwm
   fun:_ZNSs4_Rep9_S_createEmmRKSaIcE
   ...
}
```

**2. Valgrind with GDB Integration:**
```bash
# Start program under Valgrind with GDB server
valgrind --vgdb=yes --vgdb-error=0 ./program

# In another terminal, connect GDB
gdb ./program
(gdb) target remote | vgdb
```

**3. Automated Testing with Valgrind:**
```bash
#!/bin/bash
# test_with_valgrind.sh

PROGRAM="./my_program"
VALGRIND_OPTIONS="--leak-check=full --error-exitcode=1"

echo "Running $PROGRAM with Valgrind..."
valgrind $VALGRIND_OPTIONS $PROGRAM

if [ $? -eq 0 ]; then
    echo "✓ No memory errors detected"
else
    echo "✗ Memory errors found"
    exit 1
fi
```

## AddressSanitizer (ASan) - Fast and Efficient

**AddressSanitizer** is a fast memory error detector built into GCC and Clang. It's much faster than Valgrind but requires recompilation of your program.

### What AddressSanitizer Detects

- **Buffer overflows** (stack and heap)
- **Use-after-free** errors
- **Use-after-return** errors
- **Double-free** errors
- **Memory leaks**
- **Stack buffer overflow**
- **Global buffer overflow**

### Basic Usage

**Compilation with AddressSanitizer:**
```bash
# GCC
gcc -fsanitize=address -fno-omit-frame-pointer -g -o program program.c

# Clang
clang -fsanitize=address -fno-omit-frame-pointer -g -o program program.c

# C++
g++ -fsanitize=address -fno-omit-frame-pointer -g -o program program.cpp
```

### Comprehensive Example

**Sample Program with Memory Issues:**
```cpp
// asan_demo.cpp
#include <iostream>
#include <vector>
#include <cstring>

class AsanDemo {
public:
    // 1. Heap buffer overflow
    static void heap_buffer_overflow() {
        std::cout << "\n=== Heap Buffer Overflow ===" << std::endl;
        int* arr = new int[10];
        arr[10] = 42;  // Writing one past the end
        std::cout << "Value: " << arr[10] << std::endl;
        delete[] arr;
    }
    
    // 2. Stack buffer overflow
    static void stack_buffer_overflow() {
        std::cout << "\n=== Stack Buffer Overflow ===" << std::endl;
        int arr[10];
        arr[10] = 42;  // Writing one past the end
        std::cout << "Value: " << arr[10] << std::endl;
    }
    
    // 3. Use after free
    static void use_after_free() {
        std::cout << "\n=== Use After Free ===" << std::endl;
        int* ptr = new int(42);
        delete ptr;
        std::cout << "Value: " << *ptr << std::endl;  // Use after free
    }
    
    // 4. Memory leak
    static void memory_leak() {
        std::cout << "\n=== Memory Leak ===" << std::endl;
        int* leaked = new int[1000];
        leaked[0] = 42;
        std::cout << "Allocated memory, but never freed" << std::endl;
        // leaked is never deleted
    }
    
    // 5. Global buffer overflow
    static void global_buffer_overflow() {
        std::cout << "\n=== Global Buffer Overflow ===" << std::endl;
        static int global_array[10];
        global_array[10] = 42;  // Writing past the end
        std::cout << "Value: " << global_array[10] << std::endl;
    }
    
    // 6. Use after return (stack use after scope)
    static int* return_stack_address() {
        int local_var = 42;
        return &local_var;  // Returning address of local variable
    }
    
    static void use_after_return() {
        std::cout << "\n=== Use After Return ===" << std::endl;
        int* ptr = return_stack_address();
        std::cout << "Value: " << *ptr << std::endl;  // Use after return
    }
};

int main(int argc, char* argv[]) {
    std::cout << "AddressSanitizer Demo Program" << std::endl;
    
    if (argc > 1) {
        int test_case = std::atoi(argv[1]);
        
        switch (test_case) {
            case 1:
                AsanDemo::heap_buffer_overflow();
                break;
            case 2:
                AsanDemo::stack_buffer_overflow();
                break;
            case 3:
                AsanDemo::use_after_free();
                break;
            case 4:
                AsanDemo::memory_leak();
                break;
            case 5:
                AsanDemo::global_buffer_overflow();
                break;
            case 6:
                AsanDemo::use_after_return();
                break;
            default:
                std::cout << "Usage: " << argv[0] << " [1-6]" << std::endl;
                std::cout << "1: Heap buffer overflow" << std::endl;
                std::cout << "2: Stack buffer overflow" << std::endl;
                std::cout << "3: Use after free" << std::endl;
                std::cout << "4: Memory leak" << std::endl;
                std::cout << "5: Global buffer overflow" << std::endl;
                std::cout << "6: Use after return" << std::endl;
        }
    } else {
        // Run a safe demonstration
        std::cout << "Running safe demonstration..." << std::endl;
        AsanDemo::memory_leak();  // This won't crash, just leak
    }
    
    return 0;
}
```

**Compilation and Testing:**
```bash
# Compile with AddressSanitizer
g++ -fsanitize=address -fno-omit-frame-pointer -g -O1 -o asan_demo asan_demo.cpp

# Test different error types
./asan_demo 1  # Heap buffer overflow
./asan_demo 2  # Stack buffer overflow
./asan_demo 3  # Use after free
./asan_demo 4  # Memory leak (use with leak detection)
```

### Understanding AddressSanitizer Output

**Heap Buffer Overflow Output:**
```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x60200000eff8 at pc 0x000000401234 bp 0x7fff12345678 sp 0x7fff12345670
WRITE of size 4 at 0x60200000eff8 thread T0
    #0 0x401233 in AsanDemo::heap_buffer_overflow() asan_demo.cpp:12
    #1 0x401456 in main asan_demo.cpp:67
    #2 0x7f8b12345678 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x20830)
    #3 0x401089 in _start (asan_demo+0x401089)

0x60200000eff8 is located 0 bytes to the right of 40-byte region [0x60200000efd0,0x60200000eff8)
allocated by thread T0 here:
    #0 0x7f8b12345678 in operator new[](unsigned long) (/usr/lib/x86_64-linux-gnu/libasan.so.4+0xe1234)
    #1 0x401210 in AsanDemo::heap_buffer_overflow() asan_demo.cpp:11
    #2 0x401456 in main asan_demo.cpp:67
    #3 0x7f8b12345678 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x20830)

SUMMARY: AddressSanitizer: heap-buffer-overflow asan_demo.cpp:12 in AsanDemo::heap_buffer_overflow()
Shadow bytes around the buggy address:
  0x0c047fff9df0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff9e00: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff9e10: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff9e20: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff9e30: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
=>0x0c047fff9e40: fa fa fa fa fa fa fa fa fa fa 00 00 00 00 00[fa]
  0x0c047fff9e50: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff9e60: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff9e70: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff9e80: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x0c047fff9e90: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
Legend: fa = poisoned, 00 = valid allocated memory
```

### Advanced AddressSanitizer Features

**1. Environment Variables:**
```bash
# Enable leak detection (disabled by default on Linux)
export ASAN_OPTIONS=detect_leaks=1

# Abort on first error
export ASAN_OPTIONS=abort_on_error=1

# Continue after error
export ASAN_OPTIONS=halt_on_error=0

# Detailed output
export ASAN_OPTIONS=verbosity=1

# Combined options
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=0:abort_on_error=0
```

**2. Suppressions:**
```bash
# Create suppression file
cat > asan_suppressions.txt << EOF
# Suppress known leak in third-party library
leak:libthirdparty.so
# Suppress specific function
leak:SomeKnownLeakyFunction
EOF

# Use suppression file
export ASAN_OPTIONS=suppressions=asan_suppressions.txt
```

**3. Integration with Testing:**
```bash
#!/bin/bash
# test_with_asan.sh

# Compile with AddressSanitizer
g++ -fsanitize=address -g -o test_program test_program.cpp

# Set environment
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1

# Run tests
echo "Running tests with AddressSanitizer..."
./test_program

if [ $? -eq 0 ]; then
    echo "✓ All tests passed with no memory errors"
else
    echo "✗ Memory errors detected"
    exit 1
fi
```

**4. Performance Considerations:**
```bash
# For better performance, use -O1 optimization
g++ -fsanitize=address -O1 -g -o program program.cpp

# AddressSanitizer typically adds 2x slowdown and 2x memory usage
# Much faster than Valgrind but requires recompilation
```

### AddressSanitizer vs Valgrind Comparison

| Feature | AddressSanitizer | Valgrind |
|---------|------------------|----------|
| **Speed** | ~2x slowdown | ~10-50x slowdown |
| **Memory Usage** | ~2x increase | ~3-5x increase |
| **Recompilation** | Required | Not required |
| **Detection Coverage** | Comprehensive | More comprehensive |
| **Platform Support** | Linux, macOS, Windows | Linux, macOS |
| **Ease of Use** | Very easy | Easy |
| **CI/CD Integration** | Excellent | Good |

## Dr. Memory - Windows Memory Debugger

**Dr. Memory** is a memory monitoring tool for Windows, Linux, and Mac that detects memory-related programming errors such as accesses of uninitialized memory, accesses to unaddressable memory, and memory leaks.

### Installation

**Windows:**
```bash
# Download from https://drmemory.org/
# Extract to C:\Program Files\DrMemory
# Add to PATH: C:\Program Files\DrMemory\bin

# Verify installation
drmemory -version
```

**Linux:**
```bash
# Download and extract
wget https://github.com/DynamoRIO/drmemory/releases/download/release_2.3.0/DrMemory-Linux-2.3.0-1.tar.gz
tar -xzf DrMemory-Linux-2.3.0-1.tar.gz
export PATH=$PATH:$PWD/DrMemory-Linux-2.3.0-1/bin

# Or install via package manager (Ubuntu)
sudo apt-get install drmemory
```

### Basic Usage

**Simple Memory Analysis:**
```bash
# Compile your program
gcc -g -o program program.c

# Run with Dr. Memory
drmemory -- ./program

# Windows example
drmemory -- program.exe
```

### Comprehensive Example

**Sample Program for Dr. Memory Analysis:**
```cpp
// drmemory_demo.cpp
#include <iostream>
#include <cstdlib>
#include <cstring>

class DrMemoryDemo {
public:
    // 1. Uninitialized memory read
    static void uninitialized_read() {
        std::cout << "\n=== Uninitialized Memory Read ===" << std::endl;
        int* arr = new int[10];
        // arr is not initialized
        std::cout << "Uninitialized value: " << arr[5] << std::endl;
        delete[] arr;
    }
    
    // 2. Buffer overflow
    static void buffer_overflow() {
        std::cout << "\n=== Buffer Overflow ===" << std::endl;
        char* buffer = new char[10];
        strcpy(buffer, "This string is longer than 10 characters");
        std::cout << "Buffer content: " << buffer << std::endl;
        delete[] buffer;
    }
    
    // 3. Use after free
    static void use_after_free() {
        std::cout << "\n=== Use After Free ===" << std::endl;
        int* ptr = new int(42);
        delete ptr;
        *ptr = 100;  // Use after free
        std::cout << "Modified freed memory" << std::endl;
    }
    
    // 4. Memory leak
    static void memory_leak() {
        std::cout << "\n=== Memory Leak ===" << std::endl;
        for (int i = 0; i < 5; i++) {
            int* leaked = new int[100];
            leaked[0] = i;
            // Never deleted - memory leak
        }
        std::cout << "Created memory leaks" << std::endl;
    }
    
    // 5. Invalid free
    static void invalid_free() {
        std::cout << "\n=== Invalid Free ===" << std::endl;
        int stack_var = 42;
        int* ptr = &stack_var;
        // delete ptr;  // This would be invalid free - commented out
        std::cout << "Attempted invalid free operation" << std::endl;
    }
    
    // 6. Double free
    static void double_free() {
        std::cout << "\n=== Double Free ===" << std::endl;
        int* ptr = new int(42);
        delete ptr;
        // delete ptr;  // Double free - commented out to prevent crash
        std::cout << "Attempted double free operation" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "Dr. Memory Demo Program" << std::endl;
    
    if (argc > 1) {
        int test_case = std::atoi(argv[1]);
        
        switch (test_case) {
            case 1:
                DrMemoryDemo::uninitialized_read();
                break;
            case 2:
                DrMemoryDemo::buffer_overflow();
                break;
            case 3:
                DrMemoryDemo::use_after_free();
                break;
            case 4:
                DrMemoryDemo::memory_leak();
                break;
            case 5:
                DrMemoryDemo::invalid_free();
                break;
            case 6:
                DrMemoryDemo::double_free();
                break;
            default:
                std::cout << "Usage: " << argv[0] << " [1-6]" << std::endl;
        }
    } else {
        // Run safe operations
        DrMemoryDemo::memory_leak();
        DrMemoryDemo::uninitialized_read();
    }
    
    return 0;
}
```

**Compilation and Analysis:**
```bash
# Windows (Visual Studio)
cl /Zi /EHsc drmemory_demo.cpp

# Linux/GCC
g++ -g -o drmemory_demo drmemory_demo.cpp

# Run with Dr. Memory
drmemory -logdir logs -- ./drmemory_demo 1
```

### Understanding Dr. Memory Output

**Sample Dr. Memory Report:**
```
Dr. Memory version 2.3.0 build 1
Running "drmemory_demo.exe"

Error #1: UNINITIALIZED READ: reading register eax
# 0 DrMemoryDemo::uninitialized_read  [drmemory_demo.cpp:15]
# 1 main                              [drmemory_demo.cpp:85]
Note: @0:00:01.234 in thread 1234
Note: instruction: mov    (%eax) -> %ecx

Error #2: LEAK 400 direct bytes + 0 indirect bytes
# 0 replace_operator_new_array        [drmemory_demo.cpp:13]
# 1 DrMemoryDemo::uninitialized_read  [drmemory_demo.cpp:13]
# 2 main                              [drmemory_demo.cpp:85]

===========================================================================
FINAL SUMMARY:
      2 unique,     2 total uninitialized access(es)
      0 unique,     0 total invalid heap argument(s)
      0 unique,     0 total GDI usage error(s)
      0 unique,     0 total handle leak(s)
      0 unique,     0 total warning(s)
      1 unique,     1 total,    400 byte(s) of leak(s)
      0 unique,     0 total,      0 byte(s) of possible leak(s)
ERRORS FOUND:
      2 unique,     2 total error(s)
```

### Advanced Dr. Memory Features

**1. Command Line Options:**
```bash
# Detailed memory leak information
drmemory -show_reachable -- program.exe

# Light mode (faster, less detection)
drmemory -light -- program.exe

# Check uninitialized reads
drmemory -check_uninitialized -- program.exe

# Specify log directory
drmemory -logdir C:\DrMemoryLogs -- program.exe

# Suppress certain errors
drmemory -suppress suppress.txt -- program.exe

# Batch mode (no interaction)
drmemory -batch -- program.exe
```

**2. Suppression Files:**
```
# suppress.txt
# Suppress known issues in system libraries
UNINITIALIZED READ
name=suppress_system_uninit
system.dll!*

LEAK
name=suppress_known_leak
MyApp.exe!KnownLeakyFunction
```

**3. Integration with Visual Studio:**

Create a custom build configuration:
```xml
<!-- Dr. Memory integration in Visual Studio -->
<PropertyGroup>
  <LocalDebuggerCommand>C:\Program Files\DrMemory\bin\drmemory.exe</LocalDebuggerCommand>
  <LocalDebuggerCommandArguments>-logdir $(OutDir)DrMemoryLogs -- $(TargetPath)</LocalDebuggerCommandArguments>
</PropertyGroup>
```

**4. Automated Testing Script:**
```batch
@echo off
REM test_with_drmemory.bat

echo Running Dr. Memory analysis...
drmemory -batch -logdir logs -- %1

if %ERRORLEVEL% EQU 0 (
    echo ✓ No memory errors detected
) else (
    echo ✗ Memory errors found - check logs directory
    exit /b 1
)
```

## Other Essential Memory Profiling Tools

### 1. **Clang Static Analyzer**

Static analysis tool that finds bugs without running the program.

```bash
# Run static analysis
clang --analyze -Xanalyzer -analyzer-output=html -o analysis_results source.cpp

# View results
open analysis_results/index.html
```

**Example Analysis:**
```cpp
// static_analysis_example.cpp
#include <iostream>

void potential_null_dereference() {
    int* ptr = nullptr;
    *ptr = 42;  // Static analyzer will catch this
}

void potential_memory_leak() {
    int* data = new int[100];
    if (some_condition()) {
        return;  // Leak if condition is true
    }
    delete[] data;
}
```

### 2. **Heap Profilers**

**gperftools (Google Performance Tools):**
```bash
# Install
sudo apt-get install google-perftools libgoogle-perftools-dev

# Compile with profiling
g++ -g -o program program.cpp -ltcmalloc -lprofiler

# Run with heap profiling
env HEAPPROFILE=./heap_profile ./program

# Analyze results
google-pprof --web ./program ./heap_profile.0001.heap
```

**jemalloc:**
```bash
# Compile with jemalloc
g++ -g -o program program.cpp -ljemalloc

# Run with profiling
export MALLOC_CONF="prof:true,prof_gdump:true"
./program

# Analyze with jeprof
jeprof --show_bytes --pdf ./program jeprof.*.heap > heap_profile.pdf
```

### 3. **Platform-Specific Tools**

**Windows: Application Verifier**
```cmd
REM Enable Application Verifier
appverif /verify myapp.exe

REM Run the application
myapp.exe

REM Check results in Event Viewer
```

**Windows: CRT Debug Heap**
```cpp
#ifdef _DEBUG
#include <crtdbg.h>

int main() {
    // Enable memory leak detection
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    
    // Set breakpoint on specific allocation
    _CrtSetBreakAlloc(123);  // Break on allocation #123
    
    // Your program code here
    
    return 0;
}
#endif
```

**macOS: Instruments**
```bash
# Run with Instruments
instruments -t Leaks myapp

# Command line memory analysis
leaks myapp
```

## Tool Selection Guide

### When to Use Each Tool

| Scenario | Recommended Tool | Reason |
|----------|------------------|--------|
| **Development Phase** | AddressSanitizer | Fast feedback, easy integration |
| **CI/CD Pipeline** | AddressSanitizer | Good performance, reliable |
| **Legacy Code Analysis** | Valgrind | No recompilation needed |
| **Windows Development** | Dr. Memory | Native Windows support |
| **Production Debugging** | Lightweight profilers | Minimal performance impact |
| **Static Analysis** | Clang Static Analyzer | Catch issues before runtime |
| **Memory Usage Profiling** | Heap profilers | Understand allocation patterns |

### Performance Comparison

```cpp
// benchmark_memory_tools.cpp
#include <chrono>
#include <iostream>
#include <vector>

class PerformanceBenchmark {
public:
    static void memory_intensive_task() {
        const int iterations = 1000000;
        std::vector<int*> ptrs;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Allocate memory
        for (int i = 0; i < iterations; i++) {
            ptrs.push_back(new int(i));
        }
        
        // Access memory
        int sum = 0;
        for (auto ptr : ptrs) {
            sum += *ptr;
        }
        
        // Free memory
        for (auto ptr : ptrs) {
            delete ptr;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Task completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Sum: " << sum << std::endl;
    }
};

int main() {
    std::cout << "Running memory-intensive benchmark..." << std::endl;
    PerformanceBenchmark::memory_intensive_task();
    return 0;
}
```

**Performance Results (typical):**
```bash
# Normal execution
g++ -O2 -o benchmark benchmark.cpp
time ./benchmark
# Real: 0.5s

# With AddressSanitizer
g++ -fsanitize=address -O1 -o benchmark benchmark.cpp
time ./benchmark
# Real: 1.2s (2.4x slower)

# With Valgrind
g++ -g -o benchmark benchmark.cpp
time valgrind --tool=memcheck ./benchmark
# Real: 15s (30x slower)
```

## Integration with Development Workflow

### 1. **CMake Integration**

```cmake
# CMakeLists.txt
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_VALGRIND "Enable Valgrind testing" OFF)

if(ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
endif()

# Custom target for Valgrind
if(ENABLE_VALGRIND)
    find_program(VALGRIND_PROGRAM valgrind)
    if(VALGRIND_PROGRAM)
        add_custom_target(valgrind
            COMMAND ${VALGRIND_PROGRAM} --leak-check=full $<TARGET_FILE:myapp>
            DEPENDS myapp
        )
    endif()
endif()
```

### 2. **GitHub Actions CI/CD**

```yaml
# .github/workflows/memory_check.yml
name: Memory Check

on: [push, pull_request]

jobs:
  memory-sanitizer:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y valgrind
    
    - name: Build with AddressSanitizer
      run: |
        g++ -fsanitize=address -g -o test_program test_program.cpp
    
    - name: Run AddressSanitizer tests
      env:
        ASAN_OPTIONS: detect_leaks=1:abort_on_error=1
      run: ./test_program
    
    - name: Run Valgrind tests
      run: |
        g++ -g -o test_program_valgrind test_program.cpp
        valgrind --error-exitcode=1 --leak-check=full ./test_program_valgrind
```

### 3. **Pre-commit Hooks**

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running memory checks..."

# Build with sanitizers
make clean
make ASAN=1

# Run quick tests
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1
./run_tests

if [ $? -ne 0 ]; then
    echo "❌ Memory errors detected. Please fix before committing."
    exit 1
fi

echo "✅ Memory checks passed."
```

## Best Practices and Recommendations

### Development Workflow

1. **During Development:**
   - Use AddressSanitizer for fast feedback
   - Enable all compiler warnings (`-Wall -Wextra`)
   - Use static analysis tools regularly

2. **Before Code Review:**
   - Run comprehensive Valgrind analysis
   - Check with multiple sanitizers
   - Verify no new memory leaks

3. **In CI/CD:**
   - Automated AddressSanitizer builds
   - Nightly Valgrind runs for comprehensive checking
   - Fail builds on memory errors

4. **Production Monitoring:**
   - Use lightweight profiling in production
   - Monitor memory usage patterns
   - Set up alerts for unusual memory consumption

### Common Pitfalls to Avoid

❌ **Don't:**
- Ignore memory errors in "unimportant" code paths
- Disable memory checks in release builds without testing
- Use only one tool - different tools catch different issues
- Run memory tools only when problems are suspected

✅ **Do:**
- Integrate memory checking into your regular development workflow
- Fix memory errors as soon as they're detected
- Use multiple tools for comprehensive coverage
- Educate your team about memory safety best practices

## Learning Objectives

By the end of this section, you should be able to:

### Technical Skills
- **Configure and use** Valgrind, AddressSanitizer, and Dr. Memory effectively
- **Interpret output** from memory profiling tools and identify root causes
- **Integrate memory checking** into your development workflow and CI/CD pipeline
- **Choose the appropriate tool** for different scenarios and platforms
- **Write memory-safe code** by understanding common memory error patterns

### Practical Applications
- **Debug complex memory issues** in real-world applications
- **Set up automated memory testing** for continuous integration
- **Profile memory usage** to optimize application performance
- **Create suppression files** for known false positives
- **Benchmark and compare** different memory debugging approaches

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can:

□ Install and configure at least two different memory profiling tools  
□ Identify and fix heap buffer overflows using tool output  
□ Detect and resolve memory leaks in a multi-file C++ project  
□ Set up AddressSanitizer in a CMake project  
□ Create Valgrind suppression files for third-party library issues  
□ Integrate memory checking into a CI/CD pipeline  
□ Explain the trade-offs between different memory debugging tools  
□ Use static analysis tools to catch issues before runtime  

## Study Materials and Resources

### Essential Reading
- **"Effective C++"** by Scott Meyers - Chapters on memory management
- **"More Effective C++"** by Scott Meyers - Advanced memory techniques
- **AddressSanitizer documentation**: https://clang.llvm.org/docs/AddressSanitizer.html
- **Valgrind User Manual**: https://valgrind.org/docs/manual/manual.html

### Online Resources
- **Google Sanitizers Wiki**: https://github.com/google/sanitizers/wiki
- **Dr. Memory Documentation**: https://drmemory.org/docs/
- **Intel Inspector User Guide**: https://software.intel.com/content/www/us/en/develop/tools/inspector.html

### Practical Exercises

**Exercise 1: Multi-tool Analysis**
```cpp
// Create a program with various memory issues
// Analyze with all three tools and compare results
// Document which tool catches which issues best
```

**Exercise 2: CI/CD Integration**
```bash
# Set up a GitHub Actions workflow
# That runs multiple memory tools
# And fails the build on memory errors
```

**Exercise 3: Performance Benchmarking**
```cpp
// Create a benchmark program
// Measure overhead of different tools
// Create a recommendation matrix for your team
```

### Advanced Topics for Further Study
- **Custom memory allocators** and their debugging challenges
- **Embedded systems** memory debugging techniques
- **Multi-threaded** memory debugging with sanitizers
- **Kernel-level** memory debugging tools
- **Memory debugging** in containerized environments

