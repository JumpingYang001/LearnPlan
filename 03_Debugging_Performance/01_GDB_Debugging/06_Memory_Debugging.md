# Memory Debugging with GDB

*Duration: 2-3 days*

## Overview

Memory debugging is crucial for developing robust C/C++ applications. Memory-related bugs like buffer overflows, use-after-free, double-free, and memory leaks can cause crashes, security vulnerabilities, and performance issues. This guide covers comprehensive memory debugging techniques using GDB combined with powerful tools like Valgrind, AddressSanitizer, and static analysis.

### Common Memory Issues
- **Buffer Overflows**: Writing beyond allocated memory boundaries
- **Memory Leaks**: Failing to free allocated memory
- **Use-After-Free**: Accessing memory after it's been freed
- **Double-Free**: Freeing the same memory twice
- **Uninitialized Memory**: Reading from uninitialized variables
- **Stack Corruption**: Overwriting stack variables or return addresses

## Buffer Overflow Detection

### Understanding Buffer Overflows

A buffer overflow occurs when data written to a buffer exceeds its allocated size, potentially overwriting adjacent memory locations. This can lead to crashes, unpredictable behavior, or security vulnerabilities.

### Example 1: Stack Buffer Overflow
```c
// buffer_overflow.c
#include <stdio.h>
#include <string.h>

int main() {
    char buf[8];
    char safe_data = 'X';
    
    printf("Before overflow:\n");
    printf("buf address: %p\n", buf);
    printf("safe_data: %c (at %p)\n", safe_data, &safe_data);
    
    // DANGEROUS: This will overflow the buffer
    strcpy(buf, "This string is definitely too long for an 8-byte buffer!");
    
    printf("\nAfter overflow:\n");
    printf("buf content: %s\n", buf);
    printf("safe_data: %c (corrupted!)\n", safe_data);
    
    return 0;
}
```

### Compilation and GDB Setup
```bash
# Compile with debug symbols and stack protection disabled to see the overflow
gcc -g -fno-stack-protector -o buffer_overflow buffer_overflow.c

# Alternative: Enable stack protection to catch overflow
gcc -g -fstack-protector-all -o buffer_overflow_protected buffer_overflow.c
```

### GDB Debugging Session
```bash
gdb ./buffer_overflow
```

**GDB Commands for Buffer Overflow Analysis:**
```gdb
# Set breakpoints
(gdb) break main
(gdb) break strcpy

# Run the program
(gdb) run

# Examine memory layout before overflow
(gdb) info registers
(gdb) x/16xw $rsp        # Examine 16 words on stack
(gdb) print &buf
(gdb) print &safe_data

# Continue to strcpy
(gdb) continue

# Step through the strcpy operation
(gdb) step

# Examine memory after overflow
(gdb) x/32xb buf         # Examine 32 bytes from buf
(gdb) x/16xw $rsp        # Check stack corruption

# Check for stack canary corruption (if enabled)
(gdb) info registers
(gdb) backtrace
```

### Advanced Buffer Overflow Example
```c
// advanced_overflow.c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void vulnerable_function(char* input) {
    char local_buffer[64];
    int important_var = 0x12345678;
    
    printf("Before strcpy:\n");
    printf("local_buffer: %p\n", local_buffer);
    printf("important_var: 0x%x (at %p)\n", important_var, &important_var);
    
    // Vulnerable copy - no bounds checking
    strcpy(local_buffer, input);
    
    printf("After strcpy:\n");
    printf("local_buffer: %s\n", local_buffer);
    printf("important_var: 0x%x (should be 0x12345678)\n", important_var);
    
    if (important_var != 0x12345678) {
        printf("MEMORY CORRUPTION DETECTED!\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_string>\n", argv[0]);
        return 1;
    }
    
    vulnerable_function(argv[1]);
    return 0;
}
```

**Testing with Different Input Lengths:**
```bash
# Safe input
./advanced_overflow "Hello World"

# Overflow input (72+ characters to overwrite important_var)
./advanced_overflow "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBB"
```

**GDB Analysis Commands:**
```gdb
# Start debugging
(gdb) file advanced_overflow
(gdb) set args "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBB"

# Set breakpoints
(gdb) break vulnerable_function
(gdb) break strcpy

# Run and analyze
(gdb) run
(gdb) info frame              # Show stack frame info
(gdb) info locals            # Show local variables
(gdb) x/20xw local_buffer    # Examine buffer memory
(gdb) x/20xw &important_var  # Examine important_var location

# After strcpy
(gdb) continue
(gdb) x/100xb local_buffer   # See the overflow
(gdb) print important_var    # Check if corrupted
```

## Memory Leak Detection

### Understanding Memory Leaks

Memory leaks occur when dynamically allocated memory is not freed, causing the program to consume increasing amounts of memory over time. This can lead to performance degradation and system resource exhaustion.

### Example 1: Simple Memory Leak
```c
// memory_leak.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void create_leak() {
    char* leaked_memory = malloc(1024);
    strcpy(leaked_memory, "This memory will never be freed!");
    printf("Allocated memory: %s\n", leaked_memory);
    // Missing free(leaked_memory) - MEMORY LEAK!
}

void proper_allocation() {
    char* proper_memory = malloc(1024);
    strcpy(proper_memory, "This memory will be properly freed");
    printf("Allocated memory: %s\n", proper_memory);
    free(proper_memory);  // Properly freed
}

int main() {
    printf("Creating memory leaks...\n");
    for (int i = 0; i < 10; i++) {
        create_leak();  // Each call leaks 1024 bytes
    }
    
    printf("Proper memory management...\n");
    for (int i = 0; i < 10; i++) {
        proper_allocation();  // No leaks here
    }
    
    return 0;
}
```

### Example 2: Complex Memory Leak with Data Structures
```c
// complex_leak.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
    char* data;
    struct Node* next;
} Node;

typedef struct {
    Node* head;
    int count;
} LinkedList;

LinkedList* create_list() {
    LinkedList* list = malloc(sizeof(LinkedList));
    list->head = NULL;
    list->count = 0;
    return list;
}

void add_node(LinkedList* list, const char* data) {
    Node* new_node = malloc(sizeof(Node));
    new_node->data = malloc(strlen(data) + 1);
    strcpy(new_node->data, data);
    new_node->next = list->head;
    list->head = new_node;
    list->count++;
}

// BUGGY: Incomplete cleanup function
void destroy_list_buggy(LinkedList* list) {
    Node* current = list->head;
    while (current) {
        Node* next = current->next;
        // BUG: Not freeing current->data
        free(current);
        current = next;
    }
    free(list);
}

// CORRECT: Complete cleanup function
void destroy_list_correct(LinkedList* list) {
    Node* current = list->head;
    while (current) {
        Node* next = current->next;
        free(current->data);  // Free the string data
        free(current);        // Free the node
        current = next;
    }
    free(list);  // Free the list structure
}

int main() {
    printf("Creating linked list with memory leaks...\n");
    LinkedList* leaky_list = create_list();
    add_node(leaky_list, "First node data");
    add_node(leaky_list, "Second node data");
    add_node(leaky_list, "Third node data");
    destroy_list_buggy(leaky_list);  // Leaks the string data
    
    printf("Creating linked list with proper cleanup...\n");
    LinkedList* proper_list = create_list();
    add_node(proper_list, "First node data");
    add_node(proper_list, "Second node data");
    add_node(proper_list, "Third node data");
    destroy_list_correct(proper_list);  // No leaks
    
    return 0;
}
```

### GDB Commands for Memory Analysis
```gdb
# Load the program
(gdb) file memory_leak

# Set breakpoints
(gdb) break create_leak
(gdb) break malloc
(gdb) break free

# Run and track allocations
(gdb) run

# When malloc is called, examine the heap
(gdb) info proc mappings     # Show memory mappings
(gdb) print $rax            # Return value of malloc (allocated address)

# Continue execution and track memory usage
(gdb) continue

# Use GDB's heap analysis (if available)
(gdb) info heap
(gdb) heap chunks           # Show heap chunks
```

### Advanced Memory Tracking with GDB
```c
// memory_tracker.c - Example with manual tracking
#include <stdio.h>
#include <stdlib.h>

static int allocation_count = 0;
static size_t total_allocated = 0;

void* tracked_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr) {
        allocation_count++;
        total_allocated += size;
        printf("ALLOC: %p (%zu bytes) - Total: %d allocations, %zu bytes\n", 
               ptr, size, allocation_count, total_allocated);
    }
    return ptr;
}

void tracked_free(void* ptr) {
    if (ptr) {
        allocation_count--;
        printf("FREE: %p - Remaining: %d allocations\n", ptr, allocation_count);
        free(ptr);
    }
}

int main() {
    void* ptr1 = tracked_malloc(100);
    void* ptr2 = tracked_malloc(200);
    void* ptr3 = tracked_malloc(300);
    
    tracked_free(ptr1);
    tracked_free(ptr2);
    // Intentionally not freeing ptr3 to create a leak
    
    printf("Final allocation count: %d (should be 0)\n", allocation_count);
    return 0;
}
```

## Use-After-Free and Double-Free Detection

### Use-After-Free Vulnerabilities

Use-after-free occurs when a program continues to use a memory location after it has been freed, leading to undefined behavior and potential security exploits.

### Example: Use-After-Free Bug
```c
// use_after_free.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* name;
    int age;
} Person;

Person* create_person(const char* name, int age) {
    Person* p = malloc(sizeof(Person));
    p->name = malloc(strlen(name) + 1);
    strcpy(p->name, name);
    p->age = age;
    return p;
}

void free_person(Person* p) {
    if (p) {
        free(p->name);
        free(p);
    }
}

int main() {
    Person* john = create_person("John Doe", 30);
    printf("Created person: %s, age %d\n", john->name, john->age);
    
    free_person(john);
    printf("Person freed\n");
    
    // USE-AFTER-FREE BUG: Accessing freed memory
    printf("Accessing freed memory: %s, age %d\n", john->name, john->age);
    
    // DOUBLE-FREE BUG: Freeing already freed memory
    free_person(john);
    
    return 0;
}
```

### Double-Free Example
```c
// double_free.c
#include <stdio.h>
#include <stdlib.h>

void problematic_function() {
    char* buffer = malloc(100);
    
    // Some operations
    strcpy(buffer, "Hello World");
    printf("Buffer: %s\n", buffer);
    
    // First free
    free(buffer);
    printf("Buffer freed once\n");
    
    // DOUBLE-FREE ERROR
    free(buffer);  // This will cause undefined behavior
    printf("Buffer freed twice - ERROR!\n");
}

int main() {
    problematic_function();
    return 0;
}
```

### GDB Analysis for Use-After-Free
```bash
# Compile with debug info
gcc -g -o use_after_free use_after_free.c
```

**GDB Debugging Session:**
```gdb
# Start GDB
(gdb) file use_after_free

# Set breakpoints
(gdb) break create_person
(gdb) break free_person
(gdb) break main

# Run the program
(gdb) run

# When at create_person, examine allocation
(gdb) step
(gdb) print p
(gdb) print p->name

# When at free_person, examine deallocation
(gdb) continue
(gdb) step
(gdb) watch p->name     # Watch for changes to freed memory

# After freeing, examine the memory content
(gdb) continue
(gdb) x/20xb john       # Examine freed memory (may show garbage)
(gdb) print john->name  # This will show garbage or cause error
```

### Advanced Use-After-Free Detection
```c
// advanced_uaf.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Safe pointer wrapper
typedef struct {
    void* ptr;
    int is_valid;
    size_t size;
} SafePointer;

SafePointer* safe_malloc(size_t size) {
    SafePointer* safe_ptr = malloc(sizeof(SafePointer));
    safe_ptr->ptr = malloc(size);
    safe_ptr->is_valid = (safe_ptr->ptr != NULL);
    safe_ptr->size = size;
    
    printf("SAFE_MALLOC: %p (%zu bytes)\n", safe_ptr->ptr, size);
    return safe_ptr;
}

void safe_free(SafePointer* safe_ptr) {
    if (safe_ptr && safe_ptr->is_valid) {
        printf("SAFE_FREE: %p\n", safe_ptr->ptr);
        free(safe_ptr->ptr);
        safe_ptr->ptr = NULL;
        safe_ptr->is_valid = 0;
        safe_ptr->size = 0;
    }
}

void* safe_access(SafePointer* safe_ptr) {
    if (!safe_ptr || !safe_ptr->is_valid) {
        printf("ERROR: Attempting to access freed or invalid memory!\n");
        return NULL;
    }
    return safe_ptr->ptr;
}

int main() {
    SafePointer* data = safe_malloc(100);
    
    // Safe access
    char* buffer = (char*)safe_access(data);
    if (buffer) {
        strcpy(buffer, "Hello Safe World");
        printf("Data: %s\n", buffer);
    }
    
    // Free the memory
    safe_free(data);
    
    // Attempt unsafe access - this will be caught
    buffer = (char*)safe_access(data);
    if (buffer) {
        printf("This shouldn't print: %s\n", buffer);
    }
    
    free(data);  // Free the wrapper
    return 0;
}
```

## Integration with Valgrind

Valgrind is a powerful suite of tools for memory debugging, memory leak detection, and profiling. It's particularly effective when combined with GDB for comprehensive memory analysis.

### Basic Val

**Installation:**
```bash
# Ubuntu/Debian
sudo apt-get install valgrind

# CentOS/RHEL
sudo yum install valgrind

# macOS (with Homebrew)
brew install valgrind
```

**Basic Memory Check:**
```bash
# Compile with debug symbols
gcc -g -o memory_test memory_test.c

# Run with Valgrind
valgrind --tool=memcheck ./memory_test

# Detailed output
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./memory_test

# Track origins of uninitialized values
valgrind --tool=memcheck --track-origins=yes ./memory_test
```

### Comprehensive Valgrind Example
```c
// valgrind_test.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void memory_leak_function() {
    // Memory leak
    char* leaked = malloc(100);
    strcpy(leaked, "This memory will leak");
    // No free() call
}

void buffer_overflow_function() {
    char* buffer = malloc(10);
    // Buffer overflow
    strcpy(buffer, "This string is too long for the buffer");
    free(buffer);
}

void use_after_free_function() {
    char* ptr = malloc(50);
    strcpy(ptr, "Hello");
    free(ptr);
    
    // Use after free
    printf("Use after free: %s\n", ptr);
}

void double_free_function() {
    char* ptr = malloc(30);
    strcpy(ptr, "Double free test");
    free(ptr);
    free(ptr);  // Double free
}

void uninitialized_memory_function() {
    int* array = malloc(10 * sizeof(int));
    // Using uninitialized memory
    printf("Uninitialized value: %d\n", array[5]);
    free(array);
}

int main() {
    printf("Testing various memory errors...\n");
    
    memory_leak_function();
    buffer_overflow_function();
    use_after_free_function();
    double_free_function();
    uninitialized_memory_function();
    
    return 0;
}
```

**Valgrind Output Analysis:**
```bash
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes ./valgrind_test
```

**Expected Output Explanation:**
```
==12345== Memcheck, a memory error detector
==12345== ERROR SUMMARY: X errors from Y contexts

# Memory leaks
==12345== HEAP SUMMARY:
==12345==     definitely lost: 100 bytes in 1 blocks
==12345==     possibly lost: 0 bytes in 0 blocks

# Invalid read/write
==12345== Invalid write of size 1
==12345==    at 0x4C2E0F0: strcpy
==12345==    by 0x40059A: buffer_overflow_function

# Use after free
==12345== Invalid read of size 1
==12345==    at 0x4005D2: use_after_free_function
==12345==    Address 0x520d040 is 0 bytes inside a block of size 50 free'd

# Double free
==12345== Invalid free() / delete / delete[] / realloc()
==12345==    at 0x4C2A82E: free
==12345==    by 0x400612: double_free_function
```

### Valgrind Suppression Files

Create a suppression file for known false positives:
```
# valgrind_suppressions.supp
{
   known_library_leak
   Memcheck:Leak
   fun:malloc
   fun:library_init_function
}
```

Usage:
```bash
valgrind --suppressions=valgrind_suppressions.supp ./program
```

### Integration with GDB
```bash
# Start Valgrind with GDB server
valgrind --tool=memcheck --vgdb=yes --vgdb-error=0 ./program

# In another terminal, connect GDB
gdb ./program
(gdb) target remote | vgdb

# Now you can use GDB commands while Valgrind monitors memory
(gdb) break main
(gdb) continue
(gdb) monitor leak_check full
```

### Helgrind for Thread Debugging
```bash
# Detect race conditions
valgrind --tool=helgrind ./threaded_program

# Example output for race conditions
==12345== Possible data race during read of size 4 at 0x60104C by thread #2
==12345== Locks held: none
==12345==    at 0x400123: worker_thread
==12345== 
==12345== This conflicts with a previous write of size 4 by thread #1
==12345== Locks held: none
==12345==    at 0x400145: main
```

### Advanced Valgrind Features

**Custom Memory Allocators:**
```c
// custom_allocator.c
#include <valgrind/memcheck.h>

void* my_malloc(size_t size) {
    void* ptr = malloc(size);
    VALGRIND_MAKE_MEM_UNDEFINED(ptr, size);
    return ptr;
}

void my_free(void* ptr, size_t size) {
    VALGRIND_MAKE_MEM_NOACCESS(ptr, size);
    free(ptr);
}
```

**Compile with Valgrind support:**
```bash
gcc -g -I/usr/include/valgrind -o custom_allocator custom_allocator.c
```

## AddressSanitizer (ASan) Integration

AddressSanitizer is a fast memory error detector built into GCC and Clang. It's particularly useful for catching memory errors during development and testing.

### Basic AddressSanitizer Usage

**Compilation with ASan:**
```bash
# GCC
gcc -g -fsanitize=address -fno-omit-frame-pointer -o program program.c

# Clang
clang -g -fsanitize=address -fno-omit-frame-pointer -o program program.c

# With additional options
gcc -g -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -O1 -o program program.c
```

### ASan Example Program
```c
// asan_test.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void heap_buffer_overflow() {
    printf("\n=== Heap Buffer Overflow ===\n");
    char* buffer = malloc(10);
    buffer[10] = 'X';  // One byte past the end
    free(buffer);
}

void stack_buffer_overflow() {
    printf("\n=== Stack Buffer Overflow ===\n");
    char buffer[10];
    buffer[10] = 'X';  // One byte past the end
}

void use_after_free_demo() {
    printf("\n=== Use After Free ===\n");
    char* ptr = malloc(10);
    free(ptr);
    ptr[0] = 'X';  // Use after free
}

void double_free_demo() {
    printf("\n=== Double Free ===\n");
    char* ptr = malloc(10);
    free(ptr);
    free(ptr);  // Double free
}

void memory_leak_demo() {
    printf("\n=== Memory Leak (detected at exit) ===\n");
    char* leaked = malloc(100);
    strcpy(leaked, "This memory will leak");
    // No free() call
}

int main() {
    printf("AddressSanitizer Memory Error Demo\n");
    
    // Uncomment one at a time to test different errors
    // heap_buffer_overflow();
    // stack_buffer_overflow();
    // use_after_free_demo();
    // double_free_demo();
    memory_leak_demo();
    
    return 0;
}
```

### ASan Runtime Options
```bash
# Set environment variables for ASan behavior
export ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:print_stats=1"

# Detect leaks at program exit
export ASAN_OPTIONS="detect_leaks=1"

# Symbolize addresses in error reports
export ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer

# Run the program
./asan_test
```

### ASan Error Report Analysis

**Heap Buffer Overflow Report:**
```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x60200000eff0 at pc 0x000000400567 bp 0x7fff8c6c4440 sp 0x7fff8c6c4438
WRITE of size 1 at 0x60200000eff0 thread T0
    #0 0x400566 in heap_buffer_overflow asan_test.c:8
    #1 0x4005a0 in main asan_test.c:35
    #2 0x7f7b8c0c3830 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x20830)

0x60200000eff0 is located 0 bytes to the right of 10-byte region [0x60200000efe0,0x60200000efea)
allocated by thread T0 here:
    #0 0x7f7b8c9b5d28 in malloc (/usr/lib/x86_64-linux-gnu/libasan.so.4+0xded28)
    #1 0x400557 in heap_buffer_overflow asan_test.c:7
    #2 0x4005a0 in main asan_test.c:35
```

**Key Information:**
- **Error Type**: heap-buffer-overflow
- **Location**: 0 bytes to the right of allocated region
- **Stack Trace**: Shows exact location of error
- **Allocation Stack**: Shows where memory was allocated

### ASan with GDB Integration
```bash
# Compile with ASan and debug symbols
gcc -g -fsanitize=address -O1 -fno-omit-frame-pointer -o program program.c

# Run with GDB
gdb ./program
```

**GDB Commands for ASan:**
```gdb
# Set environment for better stack traces
(gdb) set environment ASAN_OPTIONS=abort_on_error=1

# Set breakpoint on ASan error handler
(gdb) break __asan_report_error

# Run the program
(gdb) run

# When ASan detects an error, examine the state
(gdb) backtrace
(gdb) info registers
(gdb) x/20xb $rdi  # Examine memory around error location
```

### Advanced ASan Features

**Custom Error Handlers:**
```c
// asan_custom.c
#include <stdio.h>
#include <sanitizer/asan_interface.h>

void custom_error_handler() {
    printf("Custom ASan error handler called!\n");
    // Log error, send notification, etc.
}

int main() {
    // Set custom error callback
    __asan_set_error_report_callback(custom_error_handler);
    
    // Trigger an error
    char* ptr = malloc(10);
    ptr[15] = 'X';  // Buffer overflow
    
    return 0;
}
```

**Manual Memory Poisoning:**
```c
// asan_manual.c
#include <stdio.h>
#include <stdlib.h>
#include <sanitizer/asan_interface.h>

int main() {
    char* buffer = malloc(100);
    
    // Manually poison part of the buffer
    __asan_poison_memory_region(buffer + 50, 50);
    
    // This is fine
    buffer[25] = 'A';
    
    // This will trigger ASan error
    buffer[75] = 'B';  // Accessing poisoned memory
    
    // Unpoison before freeing
    __asan_unpoison_memory_region(buffer + 50, 50);
    free(buffer);
    
    return 0;
}
```

### Performance Considerations

**ASan Overhead:**
- Memory usage: ~2x increase
- CPU overhead: ~2x slowdown
- Compile time: Slightly increased

**Optimization Tips:**
```bash
# Use -O1 for better performance while keeping debugging info
gcc -g -O1 -fsanitize=address -fno-omit-frame-pointer program.c

# Disable specific checks for performance
export ASAN_OPTIONS="detect_stack_use_after_return=false:check_initialization_order=false"
```

## Learning Objectives and Study Materials

### Learning Objectives

By the end of this section, you should be able to:

**Core Competencies:**
- Identify and classify different types of memory errors (buffer overflows, leaks, use-after-free, double-free)
- Use GDB effectively for memory analysis and debugging
- Integrate multiple tools (GDB + Valgrind + AddressSanitizer) for comprehensive memory debugging
- Interpret error reports from memory debugging tools and trace errors to source code
- Implement preventive measures to avoid common memory issues in your code

**Advanced Skills:**
- Set up automated memory testing in development workflows
- Analyze complex memory corruption scenarios in multi-threaded applications
- Create custom memory debugging utilities and wrappers
- Optimize debugging performance while maintaining error detection capability

### Self-Assessment Checklist

□ Detect buffer overflows using GDB and explain the memory layout  
□ Find memory leaks with Valgrind and fix them systematically  
□ Identify use-after-free bugs and implement safe pointer practices  
□ Compile programs with AddressSanitizer and interpret error reports  
□ Use GDB watchpoints to track memory corruption in real-time  
□ Integrate static analysis tools into your development workflow  

### Essential Study Materials

**Books:**
- "Debugging with GDB" - GNU Documentation (Chapter 10: Examining Memory)
- "Valgrind User Manual" - Official Documentation
- "Secure Programming Cookbook" - O'Reilly (Memory Management chapters)

**Online Resources:**
- Valgrind Quick Start Guide: http://valgrind.org/docs/manual/quick-start.html
- AddressSanitizer Wiki: https://github.com/google/sanitizers/wiki/AddressSanitizer
- GDB Memory Commands: https://sourceware.org/gdb/current/onlinedocs/gdb/Memory.html

**Development Environment Setup:**
```bash
# Ubuntu/Debian setup
sudo apt-get install gdb valgrind clang clang-tools cppcheck

# Enable core dumps
ulimit -c unlimited

# GDB configuration (~/.gdbinit)
set print pretty on
set print array on
set confirm off
set history save on
```

