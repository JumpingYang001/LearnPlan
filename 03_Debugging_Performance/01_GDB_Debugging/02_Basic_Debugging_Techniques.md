# Basic Debugging Techniques

*Duration: 2-3 days*

## Overview

Mastering basic debugging techniques is essential for effective software development. This section covers fundamental GDB operations including running programs, setting breakpoints and watchpoints, examining program state, and navigating through code execution. You'll learn to identify and fix common programming errors using systematic debugging approaches.

## Core Debugging Concepts

### What is Debugging?
Debugging is the systematic process of:
1. **Identifying** that a bug exists
2. **Isolating** the source of the bug
3. **Fixing** the bug
4. **Testing** that the fix works
5. **Ensuring** no new bugs were introduced

### Types of Bugs
- **Syntax Errors**: Caught by compiler
- **Runtime Errors**: Crashes, segmentation faults
- **Logic Errors**: Program runs but produces wrong results
- **Performance Issues**: Program is too slow or uses too much memory

## Fundamental GDB Operations

### Starting GDB and Loading Programs

**Compilation for Debugging:**
```bash
# Always compile with debug symbols for effective debugging
gcc -g -o program program.c

# With additional debugging info and warnings
gcc -g -Wall -Wextra -O0 -o program program.c

# For C++ programs
g++ -g -std=c++17 -Wall -o program program.cpp
```

**Starting GDB:**
```bash
# Method 1: Start GDB with program
gdb ./program

# Method 2: Start GDB then load program
gdb
(gdb) file ./program

# Method 3: Start GDB with core dump
gdb ./program core

# Method 4: Attach to running process
gdb -p <process_id>
```

### Running Programs in GDB

```bash
# Basic commands for program execution
(gdb) run                    # Run program from beginning
(gdb) run arg1 arg2         # Run with command line arguments
(gdb) run < input.txt       # Run with input redirection
(gdb) start                 # Run and stop at main()
(gdb) continue              # Continue execution after breakpoint
(gdb) step                  # Step into functions (source level)
(gdb) next                  # Step over functions (source level)
(gdb) stepi                 # Step one assembly instruction
(gdb) nexti                 # Next assembly instruction
(gdb) finish                # Run until current function returns
(gdb) until                 # Run until line number or function
(gdb) kill                  # Terminate the program
(gdb) quit                  # Exit GDB
```

## Breakpoints: Controlling Program Execution

### Setting Breakpoints

**Basic Breakpoint Commands:**
```bash
# Set breakpoint by function name
(gdb) break main
(gdb) break printf
(gdb) break MyClass::method

# Set breakpoint by line number
(gdb) break 15              # Line 15 in current file
(gdb) break program.c:15    # Line 15 in specific file

# Set breakpoint by address
(gdb) break *0x400526

# Conditional breakpoints
(gdb) break main if argc > 1
(gdb) break 25 if x == 0
(gdb) break loop_function if i > 100

# Temporary breakpoints (deleted after first hit)
(gdb) tbreak main           # Temporary breakpoint
```

**Managing Breakpoints:**
```bash
# List all breakpoints
(gdb) info breakpoints
(gdb) info break           # Short form

# Disable/Enable breakpoints
(gdb) disable 1            # Disable breakpoint #1
(gdb) enable 1             # Enable breakpoint #1
(gdb) disable              # Disable all breakpoints

# Delete breakpoints
(gdb) delete 1             # Delete breakpoint #1
(gdb) delete               # Delete all breakpoints
(gdb) clear                # Delete breakpoint at current line
(gdb) clear function_name  # Delete breakpoint at function
```

### Practical Breakpoint Example

**buggy_program.c:**
```c
#include <stdio.h>
#include <stdlib.h>

int calculate_factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * calculate_factorial(n - 1);
}

int process_array(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i <= size; i++) {  // BUG: should be i < size
        sum += arr[i];
    }
    return sum;
}

int main(int argc, char *argv[]) {
    printf("Starting program...\n");
    
    // Test factorial
    int fact = calculate_factorial(5);
    printf("Factorial of 5: %d\n", fact);
    
    // Test array processing
    int numbers[] = {1, 2, 3, 4, 5};
    int result = process_array(numbers, 5);
    printf("Sum: %d\n", result);
    
    return 0;
}
```

**Debugging Session:**
```bash
$ gcc -g -o buggy_program buggy_program.c
$ gdb ./buggy_program

(gdb) break main
Breakpoint 1 at 0x400647: file buggy_program.c, line 17.

(gdb) run
Breakpoint 1, main (argc=1, argv=0x7fffffffddb8) at buggy_program.c:17
17          printf("Starting program...\n");

(gdb) break process_array
Breakpoint 2 at 0x4005f0: file buggy_program.c, line 10.

(gdb) continue
Starting program...
Factorial of 5: 120

Breakpoint 2, process_array (arr=0x7fffffffdd90, size=5) at buggy_program.c:10
10          int sum = 0;

# Now we can examine the problematic loop
(gdb) break 11              # Break at for loop
(gdb) continue
```

## Watchpoints: Monitoring Variable Changes

### Understanding Watchpoints

Watchpoints are special breakpoints that trigger when a variable's value changes, helping you track down when and where variables are being modified unexpectedly.

**Types of Watchpoints:**
- **watch**: Break when variable is written
- **rwatch**: Break when variable is read
- **awatch**: Break when variable is accessed (read or write)

### Setting Watchpoints

```bash
# Basic watchpoint syntax
(gdb) watch variable_name
(gdb) watch global_var
(gdb) watch *pointer        # Watch what pointer points to
(gdb) watch array[index]    # Watch specific array element

# Read watchpoints
(gdb) rwatch variable_name

# Access watchpoints (read or write)
(gdb) awatch variable_name

# Conditional watchpoints
(gdb) watch x if x > 100
```

### Watchpoint Example

**watch_demo.c:**
```c
#include <stdio.h>

int global_counter = 0;

void increment_counter() {
    global_counter++;
}

void suspicious_function() {
    global_counter = 999;  // Unexpected modification
}

int main() {
    printf("Initial counter: %d\n", global_counter);
    
    for (int i = 0; i < 5; i++) {
        increment_counter();
        printf("Counter after increment %d: %d\n", i+1, global_counter);
        
        if (i == 2) {
            suspicious_function();  // This will trigger watchpoint
        }
    }
    
    return 0;
}
```

**Debugging with Watchpoints:**
```bash
$ gcc -g -o watch_demo watch_demo.c
$ gdb ./watch_demo

(gdb) break main
(gdb) run
Breakpoint 1, main () at watch_demo.c:12
12          printf("Initial counter: %d\n", global_counter);

(gdb) watch global_counter
Hardware watchpoint 2: global_counter

(gdb) continue
Initial counter: 0
Hardware watchpoint 2: global_counter
Old value = 0
New value = 1
increment_counter () at watch_demo.c:6
6       }

(gdb) continue
Counter after increment 1: 1
Hardware watchpoint 2: global_counter
Old value = 1
New value = 2
# ... continues until suspicious_function modifies it unexpectedly
```

## Examining Program State

### Viewing Variables and Memory

```bash
# Print variables
(gdb) print variable_name
(gdb) print *pointer
(gdb) print array[0]
(gdb) print struct_var.member

# Print with different formats
(gdb) print/x variable      # Hexadecimal
(gdb) print/d variable      # Decimal
(gdb) print/o variable      # Octal
(gdb) print/t variable      # Binary
(gdb) print/c variable      # Character
(gdb) print/f variable      # Floating point

# Examine memory
(gdb) x/10dw &variable      # Examine 10 decimal words
(gdb) x/5i function_name    # Examine 5 instructions
(gdb) x/s string_pointer    # Examine as string
```

### Examining Complex Data Structures

**data_structures.c:**
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    char name[50];
    float salary;
} Employee;

typedef struct Node {
    int data;
    struct Node* next;
} Node;

int main() {
    // Create an employee
    Employee emp = {101, "John Doe", 50000.0};
    
    // Create a linked list
    Node* head = malloc(sizeof(Node));
    head->data = 10;
    head->next = malloc(sizeof(Node));
    head->next->data = 20;
    head->next->next = NULL;
    
    printf("Employee: %d, %s, %.2f\n", emp.id, emp.name, emp.salary);
    
    // Process linked list
    Node* current = head;
    while (current != NULL) {
        printf("Data: %d\n", current->data);
        current = current->next;
    }
    
    // Clean up
    free(head->next);
    free(head);
    
    return 0;
}
```

**Debugging Complex Data:**
```bash
(gdb) break main
(gdb) run
(gdb) next 6                # Move to after emp initialization

# Examine the struct
(gdb) print emp
$1 = {id = 101, name = "John Doe", salary = 50000}

(gdb) print emp.name
$2 = "John Doe"

(gdb) print &emp
$3 = (Employee *) 0x7fffffffdd50

# After linked list creation
(gdb) print head
$4 = (Node *) 0x602010

(gdb) print *head
$5 = {data = 10, next = 0x602030}

(gdb) print head->next->data
$6 = 20

# Navigate through linked list
(gdb) set $node = head
(gdb) while ($node != 0)
 >print $node->data
 >set $node = $node->next
 >end
```

### Stack Examination and Function Calls

**Backtrace Commands:**
```bash
# Show call stack
(gdb) backtrace            # Full backtrace
(gdb) bt                   # Short form
(gdb) bt 5                 # Show only 5 frames
(gdb) bt full              # Show local variables in each frame

# Navigate stack frames
(gdb) frame 0              # Go to frame 0 (current)
(gdb) frame 2              # Go to frame 2
(gdb) up                   # Move up one frame
(gdb) down                 # Move down one frame

# Show current frame info
(gdb) info frame           # Current frame details
(gdb) info locals          # Local variables in current frame
(gdb) info args            # Function arguments
```

**Stack Example:**
```c
#include <stdio.h>

void function_c(int value) {
    int local_c = value * 2;
    printf("In function_c: %d\n", local_c);
    
    // Trigger breakpoint here to examine stack
    int *null_ptr = NULL;
    *null_ptr = 42;  // This will cause segmentation fault
}

void function_b(int value) {
    int local_b = value + 10;
    printf("In function_b: %d\n", local_b);
    function_c(local_b);
}

void function_a(int value) {
    int local_a = value + 5;
    printf("In function_a: %d\n", local_a);
    function_b(local_a);
}

int main() {
    int start_value = 1;
    printf("Starting with: %d\n", start_value);
    function_a(start_value);
    return 0;
}
```

**Debugging Stack Example:**
```bash
$ gcc -g -o stack_example stack_example.c
$ gdb ./stack_example

(gdb) run
Starting with: 1
In function_a: 6
In function_b: 16
In function_c: 32

Program received signal SIGSEGV, Segmentation fault.
0x000000000040054a in function_c (value=32) at stack_example.c:8
8           *null_ptr = 42;

(gdb) bt
#0  0x000000000040054a in function_c (value=32) at stack_example.c:8
#1  0x000000000040057a in function_b (value=16) at stack_example.c:14
#2  0x00000000004005a4 in function_a (value=6) at stack_example.c:20
#3  0x00000000004005c8 in main () at stack_example.c:25

(gdb) bt full
#0  0x000000000040054a in function_c (value=32) at stack_example.c:8
        local_c = 32
        null_ptr = 0x0
#1  0x000000000040057a in function_b (value=16) at stack_example.c:14
        local_b = 16
#2  0x00000000004005a4 in function_a (value=6) at stack_example.c:20
        local_a = 6
#3  0x00000000004005c8 in main () at stack_example.c:25
        start_value = 1

# Examine each frame
(gdb) frame 1
(gdb) print local_b
$1 = 16

(gdb) frame 2
(gdb) print local_a
$2 = 6
```

## Debugging Different Types of Errors

### Segmentation Faults

**Common Causes:**
- Dereferencing NULL pointers
- Buffer overflows
- Use after free
- Stack overflow

**segfault_demo.c:**
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void null_pointer_error() {
    int *ptr = NULL;
    *ptr = 42;  // Segmentation fault
}

void buffer_overflow_error() {
    char buffer[10];
    strcpy(buffer, "This string is too long for the buffer");  // Buffer overflow
}

void use_after_free_error() {
    int *ptr = malloc(sizeof(int));
    *ptr = 42;
    free(ptr);
    *ptr = 100;  // Use after free
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <error_type>\n", argv[0]);
        printf("Error types: 1=null_pointer, 2=buffer_overflow, 3=use_after_free\n");
        return 1;
    }
    
    int error_type = atoi(argv[1]);
    
    switch (error_type) {
        case 1:
            null_pointer_error();
            break;
        case 2:
            buffer_overflow_error();
            break;
        case 3:
            use_after_free_error();
            break;
        default:
            printf("Unknown error type\n");
    }
    
    return 0;
}
```

**Debugging Segfaults:**
```bash
$ gcc -g -o segfault_demo segfault_demo.c
$ gdb ./segfault_demo

(gdb) set args 1
(gdb) run
Program received signal SIGSEGV, Segmentation fault.
0x0000000000400566 in null_pointer_error () at segfault_demo.c:7
7           *ptr = 42;

(gdb) print ptr
$1 = (int *) 0x0

(gdb) bt
#0  0x0000000000400566 in null_pointer_error () at segfault_demo.c:7
#1  0x00000000004005f5 in main (argc=2, argv=0x7fffffffdd88) at segfault_demo.c:33
```

### Memory Errors with Valgrind Integration

```bash
# Compile with debug info
gcc -g -o program program.c

# Run with Valgrind
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./program

# Run GDB with Valgrind
valgrind --vgdb=yes --vgdb-error=0 ./program

# In another terminal
gdb ./program
(gdb) target remote | vgdb
```

### Logic Errors

**logic_error.c:**
```c
#include <stdio.h>

int calculate_average(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i <= size; i++) {  // BUG: should be i < size
        sum += arr[i];
    }
    return sum / size;
}

int binary_search(int *arr, int size, int target) {
    int left = 0, right = size - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;  // Potential overflow for large values
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid + 1;  // BUG: should be mid - 1
        }
    }
    
    return -1;  // Not found
}

int main() {
    int numbers[] = {1, 3, 5, 7, 9, 11, 13, 15};
    int size = sizeof(numbers) / sizeof(numbers[0]);
    
    // Test average calculation
    int avg = calculate_average(numbers, size);
    printf("Average: %d\n", avg);
    
    // Test binary search
    int target = 7;
    int index = binary_search(numbers, size, target);
    printf("Index of %d: %d\n", target, index);
    
    return 0;
}
```

**Debugging Logic Errors:**
```bash
(gdb) break calculate_average
(gdb) run
(gdb) next
(gdb) print i
(gdb) print size
(gdb) watch i
(gdb) continue

# When i becomes equal to size, we have our bug!
```

## Advanced Debugging Techniques

### Debugging Optimized Code

```bash
# Compile with minimal optimization and debug info
gcc -g -O1 -o program program.c

# In GDB, handle optimization issues
(gdb) set print pretty on
(gdb) set print array on
(gdb) set confirm off

# Some variables may be optimized out
(gdb) info locals
# May show: value = <optimized out>
```

### Debugging with Core Dumps

```bash
# Enable core dumps
ulimit -c unlimited

# Run program that crashes
./program
# Segmentation fault (core dumped)

# Debug with core dump
gdb ./program core

# Or
gdb ./program
(gdb) core core

# Examine crash state
(gdb) bt
(gdb) print variable_name
```

### Remote Debugging

```bash
# On target machine (embedded system, server, etc.)
gdbserver :1234 ./program

# On development machine
gdb ./program
(gdb) target remote target_ip:1234
(gdb) continue
```

### Multi-threaded Debugging

```bash
# Show all threads
(gdb) info threads

# Switch between threads
(gdb) thread 2

# Set scheduler locking
(gdb) set scheduler-locking on    # Only current thread runs
(gdb) set scheduler-locking step  # Only current thread steps
(gdb) set scheduler-locking off   # All threads run

# Thread-specific breakpoints
(gdb) break function_name thread 2
```

## Debugging Best Practices

### Systematic Debugging Approach

1. **Reproduce the Bug Consistently**
   ```c
   // Create minimal test case
   int main() {
       // Minimal code to reproduce the issue
       return 0;
   }
   ```

2. **Use Assertions for Debugging**
   ```c
   #include <assert.h>
   
   void process_array(int *arr, int size) {
       assert(arr != NULL);        // Check preconditions
       assert(size > 0);
       
       for (int i = 0; i < size; i++) {
           assert(i >= 0 && i < size);  // Check loop bounds
           // Process arr[i]
       }
   }
   ```

3. **Add Debug Prints Strategically**
   ```c
   #ifdef DEBUG
   #define DEBUG_PRINT(fmt, ...) \
       fprintf(stderr, "DEBUG: %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
   #else
   #define DEBUG_PRINT(fmt, ...)
   #endif
   
   void function() {
       DEBUG_PRINT("Entering function with param: %d", param);
       // Function code
       DEBUG_PRINT("Exiting function with result: %d", result);
   }
   ```

4. **Use Static Analysis Tools**
   ```bash
   # Use compiler warnings
   gcc -Wall -Wextra -Wpedantic -Werror program.c
   
   # Use static analyzers
   clang-tidy program.c
   cppcheck program.c
   ```

### GDB Configuration Tips

**Create ~/.gdbinit file:**
```bash
# Pretty printing
set print pretty on
set print array on
set print array-indexes on

# History
set history save on
set history size 10000
set history filename ~/.gdb_history

# Convenience functions
define plist
  set $node = $arg0
  while $node
    print *$node
    set $node = $node->next
  end
end

# Auto-load local .gdbinit files
set auto-load safe-path /
```

## Learning Objectives

By the end of this section, you should be able to:
- **Compile programs with appropriate debug symbols** for effective debugging
- **Set and manage breakpoints** including conditional and temporary breakpoints
- **Use watchpoints effectively** to monitor variable changes
- **Navigate through program execution** using step, next, continue commands
- **Examine program state** including variables, memory, and call stack
- **Debug different types of errors** (segfaults, logic errors, memory issues)
- **Apply systematic debugging approaches** to isolate and fix bugs
- **Use advanced debugging techniques** for optimized code and multi-threaded programs

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can:

□ Compile a program with debug symbols and start it in GDB  
□ Set breakpoints at functions, line numbers, and with conditions  
□ Use watchpoints to track variable modifications  
□ Navigate the call stack and examine local variables in each frame  
□ Print variables in different formats (hex, decimal, binary)  
□ Identify the cause of a segmentation fault using GDB  
□ Debug a simple logic error in a loop or conditional statement  
□ Use GDB to examine memory contents and pointer values  
□ Apply systematic debugging methodology to unknown bugs  

### Practical Exercises

**Exercise 1: Basic Debugging Workflow**
```c
// debug_exercise1.c - Find and fix the bugs
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    for (int i = 1; i <= 10; i++) {
        printf("Factorial of %d is %d\n", i, factorial(i));
    }
    return 0;
}
// TODO: What happens with factorial(0)? Use GDB to trace execution
```

**Exercise 2: Segmentation Fault Investigation**
```c
// debug_exercise2.c - Fix the segmentation fault
#include <stdio.h>
#include <string.h>

void copy_string(char *dest, const char *src) {
    strcpy(dest, src);  // Potential issue here
}

int main() {
    char *buffer;  // Uninitialized pointer
    copy_string(buffer, "Hello, World!");
    printf("Result: %s\n", buffer);
    return 0;
}
// TODO: Use GDB to identify why this crashes
```

**Exercise 3: Logic Error Detection**
```c
// debug_exercise3.c - Find the logic error
#include <stdio.h>

int find_max(int arr[], int size) {
    int max = 0;  // Potential issue with initialization
    for (int i = 1; i < size; i++) {  // Starting from index 1?
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

int main() {
    int numbers[] = {-5, -2, -8, -1, -10};
    int max = find_max(numbers, 5);
    printf("Maximum value: %d\n", max);
    return 0;
}
// TODO: Why doesn't this work correctly with negative numbers?
```

## Study Materials

### Recommended Reading
- **Primary:** "The Art of Debugging" by Norman Matloff and Peter Jay Salzman
- **Reference:** GDB User Manual (https://sourceware.org/gdb/current/onlinedocs/gdb/)
- **Online:** "Debugging with GDB" tutorial series

### Essential GDB Commands Quick Reference

```bash
# Program Control
run [args]          # Start program
start               # Start and break at main
continue            # Continue execution
step                # Step into functions
next                # Step over functions
finish              # Run until function returns
kill                # Terminate program

# Breakpoints
break function      # Break at function
break file:line     # Break at line in file
break *address      # Break at memory address
info breakpoints    # List breakpoints
disable/enable N    # Disable/enable breakpoint N
delete N           # Delete breakpoint N

# Watchpoints
watch variable      # Break when variable changes
rwatch variable     # Break when variable is read
awatch variable     # Break when variable is accessed

# Examining Data
print variable      # Print variable value
print/x variable    # Print in hexadecimal
x/fmt address      # Examine memory
info locals        # Show local variables
info args          # Show function arguments

# Stack Navigation
backtrace          # Show call stack
frame N            # Switch to frame N
up/down            # Move up/down stack
info frame         # Show current frame info

# Program Information
info program       # Program status
info registers     # CPU registers
info memory        # Memory regions
info threads       # List threads (for multi-threaded)
```

### Practice Scenarios

1. **Buffer Overflow Detection**: Debug a program with array bounds violation
2. **Memory Leak Investigation**: Use GDB with Valgrind to find memory leaks  
3. **Infinite Loop Debugging**: Identify why a loop never terminates
4. **Pointer Arithmetic Errors**: Debug incorrect pointer manipulations
5. **Race Condition Debugging**: Debug multi-threaded synchronization issues

### Next Steps

After mastering these basic debugging techniques, you'll be ready to move on to:
- Advanced GDB features (custom commands, scripting)
- Memory debugging with Valgrind
- Performance profiling techniques
- Specialized debugging for concurrent programs
