# GDB Fundamentals

*Duration: 1 week*

## Overview

**GDB (GNU Debugger)** is the most powerful and widely-used debugger for C, C++, and other programming languages. It allows you to examine what your program is doing at runtime, find bugs, analyze crashes, and understand program behavior step by step.

### What GDB Can Do
- **Step through code** line by line or instruction by instruction
- **Set breakpoints** to pause execution at specific locations
- **Examine variables** and memory contents in real-time
- **Analyze stack traces** to understand call hierarchies
- **Debug crashes** and core dumps
- **Attach to running processes** for live debugging
- **Remote debugging** across networks
- **Multi-threaded debugging** with thread control

### Why Learn GDB?
- **Essential skill** for any C/C++ developer
- **Faster debugging** than printf-style debugging
- **Deep program understanding** through runtime inspection
- **Production debugging** capabilities
- **Memory debugging** for finding leaks and corruption
- **Performance analysis** through execution tracing

## Installation and Setup

### Installing GDB

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install gdb
```

**Linux (CentOS/RHEL/Fedora):**
```bash
# CentOS/RHEL
sudo yum install gdb
# Fedora
sudo dnf install gdb
```

**macOS:**
```bash
# Using Homebrew
brew install gdb

# Note: macOS requires additional setup for code signing
# Create a certificate and sign gdb for proper functionality
```

**Windows:**
```bash
# Using MinGW-w64 or MSYS2
pacman -S gdb

# Or install via Visual Studio Build Tools
# GDB is included with MinGW installations
```

### Verifying Installation
```bash
gdb --version
# Should output version information
```

### Essential GDB Configuration

Create a `.gdbinit` file in your home directory for customization:

```bash
# ~/.gdbinit
set confirm off
set verbose off
set prompt (gdb) 
set history save on
set history size 10000
set history filename ~/.gdb_history
set print pretty on
set print array-indexes on
set print demangle on
set print asm-demangle on
set print static-members on
set print vtbl on
set print object on
set print elements 200
set disassembly-flavor intel
```

## Compiling for Debugging

### Debug Information Levels

**Basic Debug Info (`-g`):**
```bash
gcc -g program.c -o program
```

**Enhanced Debug Info (`-g3`):**
```bash
# Includes macro definitions and more detailed info
gcc -g3 program.c -o program
```

**Optimized Code with Debug Info:**
```bash
# Be careful: optimization can make debugging confusing
gcc -g -O2 program.c -o program
```

**Debug-Friendly Compilation:**
```bash
# Best for debugging - no optimization, extra warnings
gcc -g -O0 -Wall -Wextra -fno-omit-frame-pointer program.c -o program
```

### Compilation Flags Explanation

| Flag | Purpose |
|------|---------|
| `-g` | Include debug symbols |
| `-g3` | Include debug symbols + macro definitions |
| `-O0` | No optimization (easier debugging) |
| `-Wall` | Enable common warnings |
| `-Wextra` | Enable extra warnings |
| `-fno-omit-frame-pointer` | Keep frame pointers (better stack traces) |
| `-fsanitize=address` | Enable AddressSanitizer for memory debugging |
| `-fsanitize=thread` | Enable ThreadSanitizer for race conditions |

## Example: Complete Debugging Session

Let's create a more comprehensive example to demonstrate GDB capabilities:

```c
// debug_example.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    char name[50];
    float score;
} Student;

void print_student(Student *s) {
    printf("Student ID: %d, Name: %s, Score: %.2f\n", 
           s->id, s->name, s->score);
}

int find_max_score(Student students[], int count) {
    int max_index = 0;
    for (int i = 1; i < count; i++) {
        if (students[i].score > students[max_index].score) {
            max_index = i;
        }
    }
    return max_index;
}

void buggy_function(char *input) {
    char buffer[10];
    strcpy(buffer, input);  // Potential buffer overflow!
    printf("Processed: %s\n", buffer);
}

int main() {
    printf("GDB Debugging Example\n");
    
    // Create array of students
    Student students[] = {
        {1, "Alice", 85.5},
        {2, "Bob", 92.0},
        {3, "Charlie", 78.3},
        {4, "Diana", 95.7}
    };
    
    int count = sizeof(students) / sizeof(students[0]);
    
    printf("Students:\n");
    for (int i = 0; i < count; i++) {
        print_student(&students[i]);
    }
    
    int best = find_max_score(students, count);
    printf("\nBest student: ");
    print_student(&students[best]);
    
    // This will cause problems with long input
    char *test_input = "This is a very long string that will overflow";
    buggy_function(test_input);
    
    return 0;
}
```

### Compilation and Initial Run

```bash
# Compile with debug information
gcc -g -O0 -Wall -Wextra -o debug_example debug_example.c

# Run normally first to see the issue
./debug_example
```

## Starting GDB and Basic Navigation

### Launching GDB

**Method 1: Start with Program**
```bash
gdb ./debug_example
```

**Method 2: Start GDB then Load Program**
```bash
gdb
(gdb) file ./debug_example
```

**Method 3: Debug with Arguments**
```bash
gdb --args ./debug_example arg1 arg2
```

**Method 4: Attach to Running Process**
```bash
# Find process ID
ps aux | grep debug_example
# Attach to PID
gdb -p <PID>
```

### Essential GDB Commands

#### Program Control Commands

**Starting and Stopping:**
```bash
(gdb) run                    # Start program execution
(gdb) run arg1 arg2          # Start with command line arguments
(gdb) kill                   # Terminate current program
(gdb) quit                   # Exit GDB
```

**Execution Control:**
```bash
(gdb) continue (or c)        # Continue execution
(gdb) next (or n)            # Execute next line (step over functions)
(gdb) step (or s)            # Execute next line (step into functions)
(gdb) nexti (or ni)          # Next assembly instruction
(gdb) stepi (or si)          # Step into assembly instruction
(gdb) finish                 # Execute until current function returns
(gdb) until                  # Execute until current loop ends
```

#### Breakpoint Commands

**Setting Breakpoints:**
```bash
(gdb) break main             # Break at function main
(gdb) break 25               # Break at line 25
(gdb) break debug_example.c:25   # Break at line 25 in specific file
(gdb) break print_student    # Break at function print_student
(gdb) break *0x400526        # Break at memory address
```

**Conditional Breakpoints:**
```bash
(gdb) break 30 if i == 5     # Break at line 30 only when i equals 5
(gdb) break print_student if s->score > 90.0
```

**Managing Breakpoints:**
```bash
(gdb) info breakpoints       # List all breakpoints
(gdb) disable 1              # Disable breakpoint #1
(gdb) enable 1               # Enable breakpoint #1
(gdb) delete 1               # Delete breakpoint #1
(gdb) clear                  # Delete all breakpoints at current location
(gdb) delete                 # Delete all breakpoints
```

#### Variable and Memory Inspection

**Printing Variables:**
```bash
(gdb) print i                # Print variable i
(gdb) print students[0]      # Print array element
(gdb) print students[0].name # Print struct member
(gdb) print *s               # Dereference pointer
(gdb) print &i               # Print address of variable
```

**Advanced Printing:**
```bash
(gdb) print/x i              # Print in hexadecimal
(gdb) print/d i              # Print in decimal
(gdb) print/o i              # Print in octal
(gdb) print/t i              # Print in binary
(gdb) print/c i              # Print as character
(gdb) print/f i              # Print as floating point
```

**Examining Memory:**
```bash
(gdb) x/10i main             # Examine 10 instructions at main
(gdb) x/10x 0x400000         # Examine 10 hex words at address
(gdb) x/s 0x400000           # Examine string at address
(gdb) x/10c buffer           # Examine 10 characters
```

**Display Commands (Auto-print):**
```bash
(gdb) display i              # Auto-print i after each step
(gdb) display students[i]    # Auto-print array element
(gdb) info display           # Show active displays
(gdb) delete display 1       # Remove display #1
```

## Practical GDB Session Walkthrough

Let's debug our example program step by step:

### Session 1: Basic Debugging

```bash
$ gdb ./debug_example
GNU gdb (Ubuntu 9.2-0ubuntu1~20.04) 9.2
Reading symbols from ./debug_example...

(gdb) list                   # Show source code
1    #include <stdio.h>
2    #include <stdlib.h>
3    #include <string.h>
4    
5    typedef struct {
6        int id;
7        char name[50];
8        float score;
9    } Student;
10   

(gdb) list main              # Show main function
24   int main() {
25       printf("GDB Debugging Example\n");
26       
27       // Create array of students
28       Student students[] = {
29           {1, "Alice", 85.5},
30           {2, "Bob", 92.0},
31           {3, "Charlie", 78.3},
32           {4, "Diana", 95.7}
33       };
34       

(gdb) break main             # Set breakpoint at main
Breakpoint 1 at 0x11a9: file debug_example.c, line 25.

(gdb) run                    # Start execution
Starting program: /path/to/debug_example 

Breakpoint 1, main () at debug_example.c:25
25       printf("GDB Debugging Example\n");

(gdb) next                   # Execute printf
GDB Debugging Example
28       Student students[] = {

(gdb) next                   # Skip array initialization
35       int count = sizeof(students) / sizeof(students[0]);

(gdb) print count            # Check count value
$1 = 4

(gdb) print students         # Examine students array
$2 = {{id = 1, name = "Alice", score = 85.5}, 
      {id = 2, name = "Bob", score = 92}, 
      {id = 3, name = "Charlie", score = 78.3000031}, 
      {id = 4, name = "Diana", score = 95.6999969}}

(gdb) break print_student    # Break at print_student function
Breakpoint 2 at 0x1169: file debug_example.c, line 12.

(gdb) continue               # Continue to next breakpoint
Continuing.
Students:

Breakpoint 2, print_student (s=0x7fffffffe010) at debug_example.c:12
12       printf("Student ID: %d, Name: %s, Score: %.2f\n", 

(gdb) print s                # Examine student pointer
$3 = (Student *) 0x7fffffffe010

(gdb) print *s               # Dereference to see student data
$4 = {id = 1, name = "Alice", score = 85.5}

(gdb) continue               # Continue through all students
# ... (breakpoint hits for each student)
```

### Session 2: Finding the Buffer Overflow

```bash
(gdb) break buggy_function   # Set breakpoint at problematic function
Breakpoint 3 at 0x11f0: file debug_example.c, line 21.

(gdb) continue
Continuing.
Best student: Student ID: 4, Name: Diana, Score: 95.70

Breakpoint 3, buggy_function (input=0x402010 "This is a very long string...") 
at debug_example.c:21
21       strcpy(buffer, input);

(gdb) print input            # Check input length
$5 = 0x402010 "This is a very long string that will overflow"

(gdb) print strlen(input)    # Get string length
$6 = 45

(gdb) print sizeof(buffer)   # This won't work - buffer is local
# Alternative: examine the local variables

(gdb) info locals            # Show local variables
buffer = "\000\000\000\000\000\000\000\000\000"
input = 0x402010 "This is a very long string that will overflow"

(gdb) print &buffer          # Get buffer address
$7 = (char (*)[10]) 0x7fffffffe006

(gdb) x/10c 0x7fffffffe006   # Examine buffer before strcpy
0x7fffffffe006: 0 '\000'     0 '\000'     0 '\000'     0 '\000'
0x7fffffffe00a: 0 '\000'     0 '\000'     0 '\000'     0 '\000'
0x7fffffffe00e: 0 '\000'     0 '\000'

(gdb) next                   # Execute the dangerous strcpy
22       printf("Processed: %s\n", buffer);

(gdb) x/50c 0x7fffffffe006   # Examine buffer after strcpy
# You'll see the overflow has occurred!

(gdb) backtrace              # Show call stack
#0  buggy_function (input=0x402010 "This is a very long string...") 
    at debug_example.c:22
#1  0x00005555555552a5 in main () at debug_example.c:46
```

## Advanced GDB Features

### Stack Frame Navigation

```bash
(gdb) backtrace (or bt)      # Show call stack
(gdb) frame 1                # Switch to frame 1
(gdb) up                     # Move up one frame
(gdb) down                   # Move down one frame
(gdb) info frame             # Show current frame info
(gdb) info args              # Show function arguments
(gdb) info locals            # Show local variables
```

### Watchpoints (Data Breakpoints)

```bash
(gdb) watch variable         # Break when variable changes
(gdb) rwatch variable        # Break when variable is read
(gdb) awatch variable        # Break when variable is accessed
(gdb) info watchpoints       # List watchpoints
```

### Multi-threaded Debugging

```bash
(gdb) info threads           # List all threads
(gdb) thread 2               # Switch to thread 2
(gdb) thread apply all bt    # Show backtrace for all threads
(gdb) set scheduler-locking on   # Control thread execution
```

### Core Dump Analysis

```bash
# When program crashes, examine core dump
gdb ./debug_example core

# Or generate core dump manually
(gdb) generate-core-file
```

### Remote Debugging

```bash
# On target machine
gdbserver localhost:2345 ./debug_example

# On development machine
gdb ./debug_example
(gdb) target remote target_ip:2345
```

## GDB Text User Interface (TUI)

GDB includes a text-based user interface for better visualization:

```bash
(gdb) tui enable             # Enable TUI mode
# Or start with: gdb -tui ./debug_example

# TUI Commands:
# Ctrl+L : Refresh screen
# Ctrl+X+A : Toggle TUI mode
# Ctrl+X+1 : Single window
# Ctrl+X+2 : Two windows
# PgUp/PgDn : Scroll source window
```

**TUI Layout Options:**
```bash
(gdb) layout src             # Source code window
(gdb) layout asm             # Assembly window
(gdb) layout split           # Source and assembly
(gdb) layout regs            # Show registers
(gdb) focus cmd              # Focus on command window
(gdb) focus src              # Focus on source window
```

## Debugging Tips and Best Practices

### 1. Effective Breakpoint Strategy
```bash
# Start broad, then narrow down
(gdb) break main
(gdb) break function_with_bug
(gdb) break problematic_line if condition
```

### 2. Use Conditional Logic
```bash
# Only break when problem occurs
(gdb) break malloc if size > 1000000
(gdb) break strcpy if strlen(src) > 10
```

### 3. Automation with GDB Scripts
Create a `.gdb` script file:
```bash
# debug_session.gdb
break main
run
break buggy_function
continue
print input
print strlen(input)
next
x/50c buffer
```

Run with: `gdb -x debug_session.gdb ./debug_example`

### 4. Debugging Optimized Code
```bash
# When debugging optimized code
(gdb) set print symbol-loading on
(gdb) set debug infrun on
(gdb) info registers         # Check register values
```

### 5. Memory Debugging Integration
```bash
# Compile with AddressSanitizer
gcc -g -fsanitize=address -o debug_example debug_example.c

# Run in GDB - will catch buffer overflows automatically
(gdb) run
```

### Common GDB Pitfalls and Solutions

**Problem: "No debugging symbols found"**
```bash
# Solution: Recompile with -g flag
gcc -g -O0 program.c -o program
```

**Problem: Variables optimized away**
```bash
# Solution: Compile with -O0 or use -Og
gcc -g -O0 program.c -o program
```

**Problem: Can't see source code**
```bash
# Solution: Ensure source file paths are correct
(gdb) directory /path/to/source
(gdb) list
```

**Problem: Shared library debugging**
```bash
# Solution: Load shared library symbols
(gdb) info sharedlibrary
(gdb) sharedlibrary library_name
```

## Learning Objectives

By the end of this section, you should be able to:

- **Install and configure GDB** on your development system
- **Compile programs with debugging information** using appropriate flags
- **Navigate GDB interface** efficiently using essential commands
- **Set and manage breakpoints** including conditional breakpoints
- **Inspect variables, memory, and program state** during execution
- **Debug common programming errors** like buffer overflows and segmentation faults
- **Use advanced GDB features** like watchpoints, TUI mode, and stack navigation
- **Analyze core dumps** for post-mortem debugging
- **Debug multi-threaded applications** with thread-specific commands
- **Create debugging scripts** for automated debugging sessions

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can:

□ Compile a C program with debug symbols and run it in GDB  
□ Set breakpoints at functions, line numbers, and with conditions  
□ Step through code using `next`, `step`, and `continue` commands  
□ Print and examine variables, arrays, and struct members  
□ Navigate the call stack using `backtrace`, `up`, and `down`  
□ Use GDB to find a buffer overflow or segmentation fault  
□ Enable and use GDB's TUI mode for better visualization  
□ Create and use watchpoints to monitor variable changes  
□ Debug a program that crashes and analyze the crash location  
□ Write a simple GDB script to automate debugging tasks  

### Practical Exercises

**Exercise 1: Basic Debugging Session**
```c
// TODO: Debug this program and find why it crashes
#include <stdio.h>
#include <stdlib.h>

int main() {
    int *ptr = NULL;
    *ptr = 42;  // This will crash - use GDB to confirm
    printf("Value: %d\n", *ptr);
    return 0;
}
```

**Exercise 2: Array Bounds Investigation**
```c
// TODO: Use GDB to investigate array access patterns
#include <stdio.h>

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    
    for (int i = 0; i <= 10; i++) {  // Bug: goes beyond array bounds
        printf("arr[%d] = %d\n", i, arr[i]);
    }
    
    return 0;
}
```

**Exercise 3: Function Call Analysis**
```c
// TODO: Use GDB to trace function calls and parameter values
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    int result = factorial(5);
    printf("5! = %d\n", result);
    return 0;
}
```

**Exercise 4: Memory Layout Exploration**
```c
// TODO: Use GDB to examine memory layout and variable addresses
#include <stdio.h>

int global_var = 100;

int main() {
    int local_var = 200;
    static int static_var = 300;
    int *heap_var = malloc(sizeof(int));
    *heap_var = 400;
    
    // Use GDB to examine addresses and memory layout
    printf("Variables created\n");
    
    free(heap_var);
    return 0;
}
```

## Study Materials

### Essential Documentation
- **Official GDB Manual**: [GNU GDB Documentation](https://www.gnu.org/software/gdb/documentation/)
- **GDB Quick Reference**: `man gdb` or `info gdb`
- **GDB Command Reference**: [GDB Commands](https://sourceware.org/gdb/onlinedocs/gdb/Command-Index.html)

### Recommended Reading
- **"The Art of Debugging with GDB, DDD, and Eclipse"** by Norman Matloff
- **"Debugging with GDB"** - Free GNU Manual
- **"Linux Programming Interface"** - Chapter 22 (Debugging)
- **"Effective Debugging"** by Diomidis Spinellis

### Video Resources
- **"GDB Tutorial"** - CS107 Stanford Course
- **"Advanced GDB"** - CppCon presentations
- **"Linux Debugging Tools"** - YouTube tutorials

### Interactive Learning
- **GDB Online Tutorial**: [GDB Tutorial by Tutorials Point](https://www.tutorialspoint.com/gnu_debugger/index.htm)
- **Hands-on Labs**: Practice with real debugging scenarios
- **CTF Challenges**: Reverse engineering and binary exploitation

### Practice Questions

**Conceptual Questions:**
1. What information is added to an executable when compiled with `-g`?
2. What's the difference between `next` and `step` commands?
3. When would you use a watchpoint vs a breakpoint?
4. How does GDB handle debugging optimized code?
5. What are the advantages of using GDB's TUI mode?

**Technical Questions:**
6. How do you set a conditional breakpoint that only triggers when a variable equals a specific value?
7. What command shows you the call stack from the current function to main?
8. How can you examine the assembly code for a specific function?
9. What's the difference between `print` and `display` commands?
10. How do you debug a program that takes command-line arguments?

**Debugging Scenarios:**
11. Your program crashes with a segmentation fault. What GDB commands would you use to find the crash location?
12. You suspect a variable is being modified unexpectedly. How would you monitor it?
13. Your program hangs in an infinite loop. How would you use GDB to identify the problem?
14. You need to debug a program that crashes only after running for several minutes. What approach would you take?
15. How would you debug a program that works correctly when run normally but fails under GDB?

### Development Environment Setup

**Required Tools Installation:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install gdb build-essential

# Additional useful tools
sudo apt install valgrind strace ltrace

# Enhanced GDB with Python support
sudo apt install gdb-python3
```

**GDB Configuration File (`~/.gdbinit`):**
```bash
# Essential settings for better debugging experience
set confirm off
set history save on
set history size 10000
set print pretty on
set print array-indexes on
set print static-members on
set disassembly-flavor intel
set auto-load safe-path /

# Custom aliases for common commands
define cls
    shell clear
end

define bpl
    info breakpoints
end

define spr
    set print repeats 0
    set print elements 0
    set print null-stop on
end
```

**Compilation Scripts:**
```bash
#!/bin/bash
# debug_build.sh
gcc -g -O0 -Wall -Wextra -fno-omit-frame-pointer \
    -fsanitize=address -fsanitize=undefined \
    "$1" -o "${1%.*}_debug"

echo "Built ${1%.*}_debug with full debugging support"
```

**Useful Aliases:**
```bash
# Add to ~/.bashrc
alias gdb='gdb -tui'
alias gdbs='gdb -batch -ex run -ex bt -ex quit --args'
alias memcheck='valgrind --tool=memcheck --leak-check=full'
```

### Advanced Topics for Further Study

1. **GDB Python Scripting** - Automate complex debugging tasks
2. **Remote Debugging** - Debug programs running on different machines
3. **Kernel Debugging** - Debug Linux kernel with KGDB
4. **Embedded Debugging** - Debug microcontroller programs
5. **Core Dump Analysis** - Post-mortem debugging techniques
6. **Integration with IDEs** - Use GDB with Visual Studio Code, CLion, etc.
7. **Performance Debugging** - Combine GDB with profiling tools
8. **Reverse Debugging** - Debug backwards through program execution

### Next Steps

After mastering GDB fundamentals, consider exploring:
- **Advanced Memory Debugging** with Valgrind
- **Performance Profiling** with gprof and perf
- **Static Analysis Tools** like Clang Static Analyzer
- **Dynamic Analysis** with sanitizers
- **Specialized Debuggers** for specific domains (LLDB, WinDbg, etc.)

