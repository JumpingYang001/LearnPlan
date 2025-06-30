# Controlling Program Execution in GDB

*Duration: 2-3 hours*

## Overview
Master the art of controlling program execution in GDB by learning to step through code line by line, navigate call stacks, manage multiple threads and frames, handle signals, and control program flow. This section focuses on the essential execution control commands that make GDB a powerful debugging tool.

## Learning Objectives

By the end of this section, you will be able to:
- **Step through code** using different stepping modes (step, next, stepi, nexti)
- **Navigate call stacks** and examine different stack frames
- **Control program flow** with continue, finish, and until commands
- **Handle signals** and interrupts during debugging
- **Debug multi-threaded programs** by switching between threads
- **Use advanced execution control** features like conditional breakpoints and watchpoints
- **Optimize debugging workflow** with efficient command combinations

## Stepping Through Code

### Understanding Different Step Commands

GDB provides several commands for stepping through code execution, each serving different purposes:

#### 1. `step` (s) - Step Into Functions
Steps **into** function calls, allowing you to debug inside called functions.

#### 2. `next` (n) - Step Over Functions  
Steps **over** function calls, treating them as single statements.

#### 3. `stepi` (si) - Step One Machine Instruction
Steps one assembly instruction at a time.

#### 4. `nexti` (ni) - Step Over One Machine Instruction
Steps over one assembly instruction, useful for assembly-level debugging.

### Comprehensive Stepping Example

```c
#include <stdio.h>
#include <stdlib.h>

int multiply(int a, int b) {
    printf("Inside multiply: a=%d, b=%d\n", a, b);
    int result = a * b;
    printf("Result: %d\n", result);
    return result;
}

int calculate_sum(int x, int y) {
    printf("Inside calculate_sum: x=%d, y=%d\n", x, y);
    int product = multiply(x, y);  // Function call here
    int sum = x + y + product;
    return sum;
}

int main() {
    printf("Starting program\n");
    
    int a = 5, b = 3;
    printf("Variables: a=%d, b=%d\n", a, b);
    
    int result = calculate_sum(a, b);  // Function call here
    printf("Final result: %d\n", result);
    
    return 0;
}
```

### GDB Stepping Session Walkthrough

**Compilation:**
```bash
gcc -g -o stepping_demo stepping_demo.c
gdb stepping_demo
```

**Step-by-step debugging session:**

```gdb
# Start debugging session
(gdb) break main
Breakpoint 1 at 0x4005f7: file stepping_demo.c, line 16.

(gdb) run
Starting program: /path/to/stepping_demo 

Breakpoint 1, main () at stepping_demo.c:16
16      printf("Starting program\n");

# Step to next line (step over printf)
(gdb) next
Starting program
17      
18      int a = 5, b = 3;

# Step to next line
(gdb) next  
19      printf("Variables: a=%d, b=%d\n", a, b);

# Examine variables
(gdb) print a
$1 = 5
(gdb) print b  
$2 = 3

# Step over the printf
(gdb) next
Variables: a=5, b=3
21      int result = calculate_sum(a, b);

# Now use 'step' to step INTO the function call
(gdb) step
calculate_sum (x=5, y=3) at stepping_demo.c:9
9       printf("Inside calculate_sum: x=%d, y=%d\n", x, y);

# We're now inside calculate_sum function
(gdb) next
Inside calculate_sum: x=5, y=3
10      int product = multiply(x, y);

# Step INTO the multiply function call
(gdb) step
multiply (a=5, b=3) at stepping_demo.c:4
4       printf("Inside multiply: a=%d, b=%d\n", a, b);

# Continue stepping through multiply function
(gdb) next
Inside multiply: a=5, b=3
5       int result = a * b;

(gdb) next
6       printf("Result: %d\n", result);

(gdb) print result
$3 = 15

(gdb) next
Result: 15
7       return result;

# Step out of function (return to caller)
(gdb) next
calculate_sum (x=5, y=3) at stepping_demo.c:11
11      int sum = x + y + product;

# Check the returned value
(gdb) print product
$4 = 15
```

### Stepping Command Comparison Table

| Command | Alias | Description | Use Case |
|---------|-------|-------------|----------|
| `step` | `s` | Step into function calls | Debug inside functions |
| `next` | `n` | Step over function calls | Skip function internals |
| `stepi` | `si` | Step one machine instruction | Assembly debugging |
| `nexti` | `ni` | Step over one instruction | Assembly debugging |
| `step N` | `s N` | Step N times | Repeat stepping |
| `next N` | `n N` | Next N times | Skip multiple lines |

### Assembly-Level Stepping Example

```c
int simple_add(int x, int y) {
    return x + y;
}

int main() {
    int result = simple_add(10, 20);
    return 0;
}
```

**Assembly debugging session:**
```gdb
(gdb) break simple_add
(gdb) run
(gdb) disassemble
Dump of assembler code for function simple_add:
=> 0x0000555555555149 <+0>:     push   %rbp
   0x000055555555514a <+1>:     mov    %rsp,%rbp
   0x000055555555514d <+4>:     mov    %edi,-0x4(%rbp)
   0x0000555555555150 <+7>:     mov    %esi,-0x8(%rbp)
   0x0000555555555153 <+10>:    mov    -0x4(%rbp),%edx
   0x0000555555555156 <+13>:    mov    -0x8(%rbp),%eax
   0x0000555555555159 <+16>:    add    %edx,%eax
   0x000055555555515b <+18>:    pop    %rbp
   0x000055555555515c <+19>:    retq

# Step one assembly instruction
(gdb) stepi
0x000055555555514a in simple_add ()

(gdb) stepi  
0x000055555555514d in simple_add ()

# Check registers
(gdb) info registers
rax            0x555555555155   93824992235861
rbx            0x0              0
rcx            0x555555555170   93824992235888
rdx            0x7fffffffe498   140737488348312
```

## Managing Call Stacks and Frames

### Understanding the Call Stack

The **call stack** is a data structure that stores information about active function calls. Each function call creates a new **stack frame** containing:
- Function parameters
- Local variables  
- Return address
- Saved registers

### Call Stack Visualization

```
Stack grows downward ↓

┌─────────────────────┐ ← Frame 0 (Current/Top)
│   current_function  │
│   - local vars      │
│   - parameters      │
│   - return address  │
├─────────────────────┤ ← Frame 1
│   caller_function   │
│   - local vars      │
│   - parameters      │
│   - return address  │
├─────────────────────┤ ← Frame 2
│   main()            │
│   - local vars      │
│   - parameters      │
│   - return address  │
└─────────────────────┘ ← Frame N (Bottom)
```

### Stack Navigation Commands

| Command | Description | Example |
|---------|-------------|---------|
| `backtrace` (`bt`) | Show full call stack | `bt` |
| `frame N` | Switch to frame N | `frame 1` |
| `up` | Move up one frame | `up` |
| `down` | Move down one frame | `down` |
| `info frame` | Show current frame details | `info frame` |
| `info args` | Show function arguments | `info args` |
| `info locals` | Show local variables | `info locals` |

### Comprehensive Stack Navigation Example

```c
#include <stdio.h>
#include <string.h>

void level3_function(char* message, int count) {
    char local_buffer[100];
    strcpy(local_buffer, message);
    
    printf("Level 3: %s (count=%d)\n", local_buffer, count);
    
    // Intentional bug - let's debug here
    int* null_ptr = NULL;
    *null_ptr = 42;  // This will cause segmentation fault
}

void level2_function(const char* prefix, int value) {
    char formatted_msg[200];
    snprintf(formatted_msg, sizeof(formatted_msg), "%s: Value is %d", prefix, value);
    
    printf("Level 2: Processing...\n");
    level3_function(formatted_msg, value * 2);
}

void level1_function(int initial_value) {
    printf("Level 1: Starting with value %d\n", initial_value);
    
    int processed_value = initial_value + 10;
    level2_function("Debug Message", processed_value);
}

int main(int argc, char* argv[]) {
    printf("Main: Program starting\n");
    
    int start_value = 5;
    level1_function(start_value);
    
    printf("Main: Program ending\n");
    return 0;
}
```

### GDB Stack Navigation Session

**Compilation and execution:**
```bash
gcc -g -o stack_demo stack_demo.c
gdb stack_demo
```

**Debugging session:**
```gdb
(gdb) run
Starting program: /path/to/stack_demo
Main: Program starting
Level 1: Starting with value 5
Level 2: Processing...
Level 3: Debug Message: Value is 15 (count=10)

Program received signal SIGSEGV, Segmentation fault.
0x0000555555555234 in level3_function (message=0x7fffffffe1b0 "Debug Message: Value is 15", count=10) at stack_demo.c:11
11      *null_ptr = 42;  // This will cause segmentation fault

# Show the complete call stack
(gdb) backtrace
#0  0x0000555555555234 in level3_function (message=0x7fffffffe1b0 "Debug Message: Value is 15", count=10) at stack_demo.c:11
#1  0x0000555555555287 in level2_function (prefix=0x555555556008 "Debug Message", value=15) at stack_demo.c:18
#2  0x00005555555552b8 in level1_function (initial_value=5) at stack_demo.c:24
#3  0x00005555555552d8 in main (argc=1, argv=0x7fffffffe3a8) at stack_demo.c:30

# Examine current frame (frame 0)
(gdb) info frame
Stack level 0, frame at 0x7fffffffe110:
 rip = 0x555555555234 in level3_function (stack_demo.c:11); saved rip = 0x555555555287
 called by frame at 0x7fffffffe140
 source language c.
 Arglist at 0x7fffffffe100, args: message=0x7fffffffe1b0 "Debug Message: Value is 15", count=10
 Locals at 0x7fffffffe100, Previous frame's sp is 0x7fffffffe110

# Show arguments and local variables in current frame
(gdb) info args
message = 0x7fffffffe1b0 "Debug Message: Value is 15"
count = 10

(gdb) info locals
local_buffer = "Debug Message: Value is 15", '\000' <repeats 73 times>
null_ptr = 0x0

# Move to frame 1 (the caller)
(gdb) frame 1
#1  0x0000555555555287 in level2_function (prefix=0x555555556008 "Debug Message", value=15) at stack_demo.c:18
18      level3_function(formatted_msg, value * 2);

(gdb) info args
prefix = 0x555555556008 "Debug Message"
value = 15

(gdb) info locals
formatted_msg = "Debug Message: Value is 15", '\000' <repeats 175 times>

# Move to frame 2
(gdb) frame 2
#2  0x00005555555552b8 in level1_function (initial_value=5) at stack_demo.c:24
24      level2_function("Debug Message", processed_value);

(gdb) info args
initial_value = 5

(gdb) info locals
processed_value = 15

# Move to main function (frame 3)
(gdb) frame 3
#3  0x00005555555552d8 in main (argc=1, argv=0x7fffffffe3a8) at stack_demo.c:30
30      level1_function(start_value);

(gdb) info args
argc = 1
argv = 0x7fffffffe3a8

(gdb) info locals
start_value = 5

# Use 'up' and 'down' commands to navigate
(gdb) down 2  # Go down 2 frames
#1  0x0000555555555287 in level2_function (prefix=0x555555556008 "Debug Message", value=15) at stack_demo.c:18

(gdb) up 1    # Go up 1 frame
#2  0x00005555555552b8 in level1_function (initial_value=5) at stack_demo.c:24
```

### Advanced Stack Analysis

**Print variables from different frames:**
```gdb
# Print variable from specific frame
(gdb) frame 0
(gdb) print local_buffer
$1 = "Debug Message: Value is 15", '\000' <repeats 73 times>

# Print variable from another frame without switching
(gdb) print *(int*)($rsp + 16)  # Access stack memory directly

# Show detailed backtrace with local variables
(gdb) backtrace full
#0  0x0000555555555234 in level3_function (message=0x7fffffffe1b0 "Debug Message: Value is 15", count=10) at stack_demo.c:11
        local_buffer = "Debug Message: Value is 15", '\000' <repeats 73 times>
        null_ptr = 0x0
#1  0x0000555555555287 in level2_function (prefix=0x555555556008 "Debug Message", value=15) at stack_demo.c:18
        formatted_msg = "Debug Message: Value is 15", '\000' <repeats 175 times>
#2  0x00005555555552b8 in level1_function (initial_value=5) at stack_demo.c:24
        processed_value = 15
#3  0x00005555555552d8 in main (argc=1, argv=0x7fffffffe3a8) at stack_demo.c:30
        start_value = 5
```

### Stack Corruption Detection

```c
// Example: Stack buffer overflow
void vulnerable_function(char* input) {
    char buffer[10];  // Small buffer
    strcpy(buffer, input);  // Potential overflow
    printf("Buffer: %s\n", buffer);
}

int main() {
    char large_input[100];
    memset(large_input, 'A', 50);
    large_input[50] = '\0';
    
    vulnerable_function(large_input);  // Will corrupt stack
    return 0;
}
```

**Debugging stack corruption:**
```gdb
(gdb) run
Program received signal SIGSEGV, Segmentation fault.

(gdb) backtrace
#0  0x4141414141414141 in ?? ()
#1  0x4141414141414141 in ?? ()
#2  0x4141414141414141 in ?? ()
# Stack is corrupted - return addresses overwritten with 'A' (0x41)

# Use stack canaries and address sanitizer to catch these:
gcc -fstack-protector-all -fsanitize=address -g -o program program.c
```

## Managing Program Execution

### Advanced Execution Control Commands

### Continue and Flow Control

Beyond stepping, GDB provides several commands to control program execution flow:

#### Flow Control Commands Reference

| Command | Alias | Description | Use Case |
|---------|-------|-------------|----------|
| `continue` | `c` | Continue execution until next breakpoint | Resume normal execution |
| `finish` | `fin` | Run until current function returns | Exit current function |
| `until` | `u` | Run until next line or specific line | Skip loops |
| `jump` | `j` | Jump to specific line/address | Skip problematic code |
| `return` | `ret` | Force function to return | Early function exit |

### Practical Flow Control Examples

#### Example 1: Using `finish` to Exit Functions

```c
#include <stdio.h>

int factorial(int n) {
    printf("Computing factorial of %d\n", n);
    
    if (n <= 1) {
        return 1;
    }
    
    int result = n * factorial(n - 1);  // Recursive call
    printf("factorial(%d) = %d\n", n, result);
    return result;
}

int main() {
    int value = 5;
    int result = factorial(value);
    printf("Final result: %d\n", result);
    return 0;
}
```

**GDB session using `finish`:**
```gdb
(gdb) break factorial
(gdb) run

# First call: factorial(5)
Breakpoint 1, factorial (n=5) at factorial.c:4
4       printf("Computing factorial of %d\n", n);

(gdb) continue
Computing factorial of %d

# We're now in factorial(4) due to recursion
Breakpoint 1, factorial (n=4) at factorial.c:4

# Use finish to complete this function and return to caller
(gdb) finish
Run till exit from #0  factorial (n=4) at factorial.c:4
factorial(4) = 24
factorial (n=5) at factorial.c:9
9       printf("factorial(%d) = %d\n", n, result);

# Now we're back in factorial(5), result contains factorial(4)
(gdb) print result
$1 = 24

(gdb) print n
$2 = 5
```

#### Example 2: Using `until` to Skip Loops

```c
#include <stdio.h>

int main() {
    printf("Before loop\n");
    
    // Long loop we want to skip during debugging
    for (int i = 0; i < 1000000; i++) {
        if (i % 100000 == 0) {
            printf("Progress: %d\n", i);
        }
    }
    
    printf("After loop\n");
    
    // Another section we want to debug
    int result = 42;
    printf("Result: %d\n", result);
    
    return 0;
}
```

**GDB session using `until`:**
```gdb
(gdb) break main
(gdb) run

Breakpoint 1, main () at loop_demo.c:4
4       printf("Before loop\n");

(gdb) next
Before loop
6       for (int i = 0; i < 1000000; i++) {

# Instead of stepping through 1,000,000 iterations, use until
(gdb) until 11
Progress: 0
Progress: 100000
Progress: 200000
Progress: 300000
Progress: 400000
Progress: 500000
Progress: 600000
Progress: 700000
Progress: 800000
Progress: 900000
After loop
11      int result = 42;

# Now we can debug the interesting part
(gdb) next
12      printf("Result: %d\n", result);

(gdb) print result
$1 = 42
```

#### Example 3: Using `jump` to Skip Problematic Code

```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Starting program\n");
    
    int* ptr = malloc(100);
    if (ptr == NULL) {
        printf("Memory allocation failed\n");
        return 1;  // Problematic exit we want to skip
    }
    
    printf("Memory allocated successfully\n");
    
    // Do some work with ptr
    *ptr = 42;
    printf("Value: %d\n", *ptr);
    
    free(ptr);
    printf("Program completed successfully\n");
    return 0;
}
```

**GDB session using `jump`:**
```gdb
(gdb) break main
(gdb) run

# Let's say we want to test the success path even if malloc fails
(gdb) break 8  # Line with "return 1;"
(gdb) continue

# If malloc fails and we hit this breakpoint:
# We can jump over the problematic return statement
(gdb) jump 11  # Jump to "printf("Memory allocated successfully\n");"

# Now we can test the rest of the code even though malloc failed
# (Note: This is for testing purposes - in real scenarios, 
#  jumping over error handling can cause crashes)
```

### Signal Handling

Programs often receive signals (interrupts) during execution. GDB allows you to control how these signals are handled.

#### Common Signals

| Signal | Description | Default Action |
|--------|-------------|----------------|
| SIGINT | Interrupt (Ctrl+C) | Terminate |
| SIGSEGV | Segmentation fault | Terminate + core dump |
| SIGFPE | Floating point exception | Terminate + core dump |
| SIGTERM | Termination request | Terminate |
| SIGUSR1/SIGUSR2 | User-defined signals | Terminate |

#### Signal Handling Commands

```gdb
# Show current signal handling settings
(gdb) info signals

# Handle specific signals
(gdb) handle SIGINT stop print pass    # Stop on SIGINT, print message, pass to program
(gdb) handle SIGINT nostop noprint nopass  # Ignore SIGINT completely
(gdb) handle SIGSEGV stop print nopass     # Stop on SIGSEGV, don't pass to program

# Send signal to program
(gdb) signal SIGUSR1  # Send SIGUSR1 to the program
```

#### Signal Handling Example

```c
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

void signal_handler(int sig) {
    printf("Received signal %d\n", sig);
}

int main() {
    // Install signal handler
    signal(SIGUSR1, signal_handler);
    signal(SIGINT, signal_handler);
    
    printf("Process ID: %d\n", getpid());
    printf("Waiting for signals... (Press Ctrl+C or send SIGUSR1)\n");
    
    // Infinite loop waiting for signals
    while (1) {
        sleep(1);
        printf("Still running...\n");
    }
    
    return 0;
}
```

**GDB signal debugging session:**
```gdb
(gdb) break signal_handler
(gdb) run
Process ID: 12345
Waiting for signals... (Press Ctrl+C or send SIGUSR1)
Still running...
Still running...

# In another terminal: kill -USR1 12345
# Or press Ctrl+C in the GDB terminal

Breakpoint 1, signal_handler (sig=10) at signal_demo.c:5
5       printf("Received signal %d\n", sig);

(gdb) print sig
$1 = 10  # SIGUSR1

(gdb) backtrace
#0  signal_handler (sig=10) at signal_demo.c:5
#1  <signal handler called>
#2  0x00007ffff7b94e2b in __GI___libc_nanosleep () from /lib/x86_64-linux-gnu/libc.so.6
#3  0x00007ffff7b94c5a in sleep () from /lib/x86_64-linux-gnu/libc.so.6
#4  0x0000555555555201 in main () at signal_demo.c:17

# Continue execution
(gdb) continue
Received signal 10
Still running...
```

### Return Value Manipulation

You can force functions to return specific values for testing purposes:

```c
#include <stdio.h>

int check_permission() {
    // Simulate permission check that might fail
    printf("Checking permissions...\n");
    return 0;  // 0 = no permission, 1 = has permission
}

int main() {
    if (check_permission()) {
        printf("Access granted\n");
        // Perform privileged operation
    } else {
        printf("Access denied\n");
        return 1;
    }
    
    printf("Operation completed\n");
    return 0;
}
```

**Force function return value:**
```gdb
(gdb) break check_permission
(gdb) run

Breakpoint 1, check_permission () at permission.c:5
5       printf("Checking permissions...\n");

# Force the function to return 1 (success) instead of 0
(gdb) return 1
Make check_permission return now? (y or n) y
#0  0x0000555555555187 in main () at permission.c:10
10      if (check_permission()) {

(gdb) continue
Access granted
Operation completed
```

### Conditional Execution

You can set up conditional breakpoints and commands:

```gdb
# Break only when condition is true
(gdb) break factorial if n == 3

# Execute commands when breakpoint is hit
(gdb) break main
(gdb) commands 1
Type commands for breakpoint(s) 1, one per line.
End with a line saying just "end".
>print "Starting main function"
>info args
>continue
>end

# Breakpoint will automatically execute these commands
```

## Multi-threaded Debugging

Debugging multi-threaded programs requires special attention to thread management and synchronization.

### Thread Control Commands

| Command | Description | Example |
|---------|-------------|---------|
| `info threads` | List all threads | `info threads` |
| `thread N` | Switch to thread N | `thread 2` |
| `break func thread N` | Break in specific thread | `break worker thread 3` |
| `thread apply all bt` | Run command on all threads | `thread apply all bt` |
| `set scheduler-locking on/off` | Control thread scheduling | `set scheduler-locking on` |

### Multi-threaded Debugging Example

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

typedef struct {
    int thread_id;
    int iterations;
    int* shared_counter;
    pthread_mutex_t* mutex;
} thread_data_t;

void* worker_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    
    printf("Thread %d starting\n", data->thread_id);
    
    for (int i = 0; i < data->iterations; i++) {
        // Critical section
        pthread_mutex_lock(data->mutex);
        
        int old_value = *(data->shared_counter);
        usleep(1000);  // Simulate work (and potential race condition)
        *(data->shared_counter) = old_value + 1;
        
        printf("Thread %d: Counter = %d (iteration %d)\n", 
               data->thread_id, *(data->shared_counter), i);
        
        pthread_mutex_unlock(data->mutex);
        
        usleep(500000);  // Sleep for 0.5 seconds
    }
    
    printf("Thread %d finishing\n", data->thread_id);
    return NULL;
}

int main() {
    const int NUM_THREADS = 3;
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    int shared_counter = 0;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    
    printf("Creating %d threads\n", NUM_THREADS);
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i + 1;
        thread_data[i].iterations = 3;
        thread_data[i].shared_counter = &shared_counter;
        thread_data[i].mutex = &mutex;
        
        if (pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            exit(1);
        }
    }
    
    printf("All threads created, waiting for completion\n");
    
    // Wait for all threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Final counter value: %d\n", shared_counter);
    return 0;
}
```

### Multi-threaded GDB Session

**Compilation:**
```bash
gcc -g -pthread -o threaded_demo threaded_demo.c
gdb threaded_demo
```

**Debugging session:**
```gdb
# Set breakpoint in worker function
(gdb) break worker_thread
(gdb) run

# Program will stop when first thread hits the breakpoint
Breakpoint 1, worker_thread (arg=0x7fffffffe0e0) at threaded_demo.c:14
14      printf("Thread %d starting\n", data->thread_id);

# Check which threads exist
(gdb) info threads
  Id   Target Id         Frame 
* 1    Thread 0x7ffff7fc9740 (LWP 12345) "threaded_demo" worker_thread (arg=0x7fffffffe0e0) at threaded_demo.c:14
  2    Thread 0x7ffff77c8700 (LWP 12346) "threaded_demo" 0x00007ffff7bc20f0 in start_thread () from /lib/x86_64-linux-gnu/libpthread.so.0
  3    Thread 0x7ffff6fc7700 (LWP 12347) "threaded_demo" 0x00007ffff7bc20f0 in start_thread () from /lib/x86_64-linux-gnu/libpthread.so.0
  4    Thread 0x7ffff67c6700 (LWP 12348) "threaded_demo" 0x00007ffff7bc20f0 in start_thread () from /lib/x86_64-linux-gnu/libpthread.so.0

# Continue, all threads will hit the breakpoint
(gdb) continue
Thread %d starting
1

# Continue again
(gdb) continue
Thread %d starting
2

# Switch to a specific thread
(gdb) thread 3
[Switching to thread 3 (Thread 0x7ffff6fc7700 (LWP 12347))]
#0  worker_thread (arg=0x7fffffffe100) at threaded_demo.c:14

# Check thread-specific data
(gdb) print *(thread_data_t*)arg
$1 = {thread_id = 3, iterations = 3, shared_counter = 0x7fffffffe13c, mutex = 0x7fffffffe140}

# Run commands on all threads
(gdb) thread apply all backtrace

Thread 4 (Thread 0x7ffff67c6700 (LWP 12348)):
#0  worker_thread (arg=0x7fffffffe120) at threaded_demo.c:14
#1  0x00007ffff7bc1609 in start_thread () from /lib/x86_64-linux-gnu/libpthread.so.0
#2  0x00007ffff78f8293 in clone () from /lib/x86_64-linux-gnu/libc.so.6

Thread 3 (Thread 0x7ffff6fc7700 (LWP 12347)):
#0  worker_thread (arg=0x7fffffffe100) at threaded_demo.c:14
#1  0x00007ffff7bc1609 in start_thread () from /lib/x86_64-linux-gnu/libpthread.so.0
#2  0x00007ffff78f8293 in clone () from /lib/x86_64-linux-gnu/libc.so.6

Thread 2 (Thread 0x7ffff77c8700 (LWP 12346)):
#0  worker_thread (arg=0x7fffffffe0e0) at threaded_demo.c:14
#1  0x00007ffff7bc1609 in start_thread () from /lib/x86_64-linux-gnu/libpthread.so.0
#2  0x00007ffff78f8293 in clone () from /lib/x86_64-linux-gnu/libc.so.6

Thread 1 (Thread 0x7ffff7fc9740 (LWP 12345)):
#0  0x00007ffff7bc1129 in pthread_join () from /lib/x86_64-linux-gnu/libpthread.so.0
#1  0x00005555555552e7 in main () at threaded_demo.c:52

# Control thread scheduling
(gdb) set scheduler-locking on
# Now only current thread will run when stepping

# Step through current thread without other threads interfering
(gdb) next
Thread 3 starting

(gdb) next
16      for (int i = 0; i < data->iterations; i++) {

# Turn scheduler locking off to allow other threads to run
(gdb) set scheduler-locking off
```

### Thread-specific Breakpoints

```gdb
# Break in specific thread only
(gdb) break worker_thread thread 2

# Break when any thread hits a specific condition
(gdb) break 25 if data->thread_id == 1

# Break when shared variable changes
(gdb) watch shared_counter
```

### Debugging Race Conditions

```c
// Race condition example
int global_var = 0;

void* racy_function(void* arg) {
    for (int i = 0; i < 1000; i++) {
        // Race condition: not atomic
        global_var = global_var + 1;
    }
    return NULL;
}
```

**Race condition debugging:**
```gdb
# Set a watchpoint on the global variable
(gdb) watch global_var

# The program will stop every time global_var changes
# This helps identify when multiple threads access it simultaneously

# Use conditional watchpoints for specific values
(gdb) watch global_var if global_var > 500

# Check which thread modified the variable
(gdb) info threads
(gdb) thread apply all where
```

## Best Practices and Debugging Workflow

### Efficient GDB Workflow

#### 1. Setup and Preparation
```bash
# Compile with debug information and optimizations off
gcc -g -O0 -Wall -Wextra -o program program.c

# For multi-threaded programs
gcc -g -O0 -pthread -o program program.c

# With address sanitizer for memory bugs
gcc -g -O0 -fsanitize=address -o program program.c

# With thread sanitizer for race conditions  
gcc -g -O0 -fsanitize=thread -o program program.c
```

#### 2. Initial Investigation Strategy
```gdb
# Start with understanding the problem
(gdb) run
# If it crashes:
(gdb) backtrace
(gdb) info registers
(gdb) disassemble

# If it hangs:
(gdb) interrupt
(gdb) backtrace
(gdb) info threads
```

#### 3. Systematic Debugging Approach

**Step 1: Reproduce the Issue**
```gdb
# Set breakpoints at key locations
(gdb) break main
(gdb) break function_where_bug_occurs

# Run with same conditions that cause the bug
(gdb) run arg1 arg2
```

**Step 2: Narrow Down the Problem**
```gdb
# Use binary search approach
# Set breakpoints at middle points of suspicious code
(gdb) break 50
(gdb) continue
# If bug hasn't occurred yet, problem is after line 50
# If bug has occurred, problem is before line 50
```

**Step 3: Examine State**
```gdb
# Check variable values
(gdb) print variable
(gdb) print *pointer
(gdb) print array[0]@10  # Print first 10 elements

# Check memory
(gdb) x/10x address     # Examine memory as hex
(gdb) x/10i address     # Examine as instructions
(gdb) x/s string_ptr    # Examine as string
```

### Common Debugging Scenarios

#### Scenario 1: Segmentation Fault

```c
// Common segfault causes
int* ptr = NULL;
*ptr = 42;              // Dereferencing NULL pointer

char buffer[10];
buffer[20] = 'x';       // Buffer overflow

int* ptr = malloc(sizeof(int));
free(ptr);
*ptr = 42;              // Use after free
```

**Debugging approach:**
```gdb
(gdb) run
Program received signal SIGSEGV, Segmentation fault.

(gdb) backtrace
(gdb) info registers
(gdb) print $pc          # Program counter
(gdb) x/i $pc           # Instruction at crash

# Check the faulting address
(gdb) info signal SIGSEGV
```

#### Scenario 2: Infinite Loop

```c
// Infinite loop example
int main() {
    int i = 0;
    while (i < 10) {
        printf("i = %d\n", i);
        // Missing: i++;
    }
    return 0;
}
```

**Debugging approach:**
```gdb
(gdb) run
# Program hangs, interrupt it
^C
Program received signal SIGINT, Interrupt.

(gdb) backtrace
(gdb) list              # Show current code
(gdb) print i           # Check loop variable
(gdb) set var i = 10    # Force exit condition
(gdb) continue
```

#### Scenario 3: Memory Corruption

```c
// Memory corruption example
int main() {
    int* arr = malloc(10 * sizeof(int));
    
    // Buffer overflow
    for (int i = 0; i <= 15; i++) {  // Should be i < 10
        arr[i] = i;
    }
    
    free(arr);
    return 0;
}
```

**Debugging with Valgrind:**
```bash
# Use Valgrind to detect memory errors
valgrind --tool=memcheck --leak-check=full ./program

# Use AddressSanitizer
gcc -g -fsanitize=address -o program program.c
./program
```

**GDB debugging:**
```gdb
# Set watchpoint on heap metadata
(gdb) watch *(void**)((char*)arr - 16)  # Watch heap header

# Use memory checking commands
(gdb) print arr
(gdb) x/20x arr         # Examine memory around allocation
```

## Practice Exercises

### Exercise 1: Basic Stepping Practice

**Goal**: Master the difference between `step`, `next`, and other stepping commands.

```c
// exercise1.c
#include <stdio.h>

int add(int a, int b) {
    printf("Adding %d + %d\n", a, b);
    return a + b;
}

int multiply(int x, int y) {
    printf("Multiplying %d * %d\n", x, y);
    return x * y;
}

int calculate(int val) {
    int sum = add(val, 10);
    int product = multiply(sum, 2);
    return product;
}

int main() {
    int result = calculate(5);
    printf("Final result: %d\n", result);
    return 0;
}
```

**Tasks**:
1. Compile and debug the program
2. Use different stepping commands to navigate through execution:
   - Use `next` to step over function calls
   - Use `step` to step into function calls
   - Use `finish` to complete current function
   - Use `until` with line numbers

**Expected Learning**: Understanding when to use each stepping command for efficient debugging.

### Exercise 2: Stack Navigation Challenge

**Goal**: Practice navigating call stacks and examining variables in different frames.

```c
// exercise2.c
#include <stdio.h>

void level_3(int depth, char* message) {
    char local_msg[100];
    sprintf(local_msg, "Level 3: %s (depth=%d)", message, depth);
    
    printf("%s\n", local_msg);
    
    // Force a breakpoint here for practice
    int* debug_ptr = NULL;  // Set breakpoint on this line
    printf("Debug point reached\n");
}

void level_2(int count, const char* prefix) {
    char buffer[50];
    sprintf(buffer, "%s-%d", prefix, count * 2);
    
    level_3(count + 1, buffer);
    
    printf("Back in level_2\n");
}

void level_1(int start_val) {
    int processed = start_val * 3;
    
    level_2(processed, "TEST");
    
    printf("Back in level_1\n");
}

int main(int argc, char* argv[]) {
    int initial = 5;
    
    printf("Starting with value: %d\n", initial);
    level_1(initial);
    printf("Program completed\n");
    
    return 0;
}
```

**Tasks**:
1. Set a breakpoint at the `debug_ptr` line in `level_3`
2. When the program stops, use `backtrace` to see the full call stack
3. Navigate through all stack frames using `frame`, `up`, and `down`
4. In each frame, examine local variables and function arguments
5. Print the value of variables from different frames without switching to them

**Expected Learning**: Mastery of stack navigation and variable inspection across multiple call levels.

### Exercise 3: Multi-threaded Debugging

**Goal**: Debug a multi-threaded program with synchronization issues.

```c
// exercise3.c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

typedef struct {
    int id;
    int* shared_resource;
    pthread_mutex_t* mutex;
} worker_data_t;

void* worker(void* arg) {
    worker_data_t* data = (worker_data_t*)arg;
    
    for (int i = 0; i < 5; i++) {
        printf("Worker %d: iteration %d\n", data->id, i);
        
        // Intentional race condition for debugging practice
        pthread_mutex_lock(data->mutex);
        
        int old_val = *(data->shared_resource);
        usleep(10000);  // Simulate work
        *(data->shared_resource) = old_val + data->id;
        
        printf("Worker %d: shared_resource = %d\n", data->id, *(data->shared_resource));
        
        pthread_mutex_unlock(data->mutex);
        
        usleep(100000);
    }
    
    return NULL;
}

int main() {
    const int NUM_WORKERS = 3;
    pthread_t workers[NUM_WORKERS];
    worker_data_t worker_data[NUM_WORKERS];
    int shared_resource = 0;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    
    // Create workers
    for (int i = 0; i < NUM_WORKERS; i++) {
        worker_data[i].id = i + 1;
        worker_data[i].shared_resource = &shared_resource;
        worker_data[i].mutex = &mutex;
        
        pthread_create(&workers[i], NULL, worker, &worker_data[i]);
    }
    
    // Wait for completion
    for (int i = 0; i < NUM_WORKERS; i++) {
        pthread_join(workers[i], NULL);
    }
    
    printf("Final shared_resource value: %d\n", shared_resource);
    return 0;
}
```

**Tasks**:
1. Compile with `-pthread` flag
2. Debug the program and practice thread switching
3. Set breakpoints that trigger in specific threads only
4. Use `info threads` to monitor all thread states
5. Use `thread apply all backtrace` to see all thread stacks
6. Practice using scheduler locking to control thread execution

**Expected Learning**: Understanding multi-threaded debugging techniques and thread synchronization inspection.

### Exercise 4: Signal Handling Debug

**Goal**: Practice debugging programs that handle signals.

```c
// exercise4.c
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>

volatile sig_atomic_t signal_count = 0;
volatile sig_atomic_t should_exit = 0;

void signal_handler(int sig) {
    signal_count++;
    printf("Received signal %d (count: %d)\n", sig, signal_count);
    
    if (signal_count >= 3) {
        should_exit = 1;
    }
}

int main() {
    signal(SIGINT, signal_handler);
    signal(SIGUSR1, signal_handler);
    
    printf("PID: %d\n", getpid());
    printf("Send SIGINT (Ctrl+C) or SIGUSR1 signals...\n");
    
    int counter = 0;
    while (!should_exit) {
        printf("Main loop iteration %d\n", ++counter);
        sleep(2);
    }
    
    printf("Exiting after %d signals\n", signal_count);
    return 0;
}
```

**Tasks**:
1. Debug the program and set breakpoints in the signal handler
2. Practice sending signals while debugging:
   - Use Ctrl+C to send SIGINT
   - From another terminal: `kill -USR1 <pid>`
3. Examine the call stack when inside the signal handler
4. Practice controlling signal handling with GDB commands
5. Use watchpoints on `signal_count` to track changes

**Expected Learning**: Understanding signal debugging and asynchronous event handling in GDB.

### Exercise 5: Memory Debugging Challenge

**Goal**: Debug memory-related issues using GDB and external tools.

```c
// exercise5.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    char* name;
    int* values;
} data_item_t;

data_item_t* create_item(int id, const char* name, int num_values) {
    data_item_t* item = malloc(sizeof(data_item_t));
    
    item->id = id;
    item->name = malloc(strlen(name) + 1);
    strcpy(item->name, name);
    
    item->values = malloc(num_values * sizeof(int));
    for (int i = 0; i < num_values; i++) {
        item->values[i] = i * id;
    }
    
    return item;
}

void process_item(data_item_t* item, int extra_processing) {
    if (item == NULL) return;
    
    printf("Processing item %d: %s\n", item->id, item->name);
    
    // Intentional buffer overflow for debugging practice
    if (extra_processing) {
        for (int i = 0; i <= 10; i++) {  // Should be i < some_limit
            item->values[i] = item->values[i] * 2;
        }
    }
}

void free_item(data_item_t* item) {
    if (item) {
        free(item->name);
        free(item->values);
        free(item);
    }
}

int main() {
    data_item_t* items[3];
    
    // Create items
    items[0] = create_item(1, "First", 5);
    items[1] = create_item(2, "Second", 8);
    items[2] = create_item(3, "Third", 3);
    
    // Process items
    for (int i = 0; i < 3; i++) {
        process_item(items[i], i % 2);  // Every other item gets extra processing
    }
    
    // Clean up
    for (int i = 0; i < 3; i++) {
        free_item(items[i]);
    }
    
    return 0;
}
```

**Tasks**:
1. Compile with AddressSanitizer: `gcc -g -fsanitize=address`
2. Debug the memory issues using GDB
3. Set watchpoints on allocated memory
4. Practice examining heap memory with `x` command
5. Use Valgrind to detect memory leaks: `valgrind --leak-check=full`
6. Fix the buffer overflow and verify the fix

**Expected Learning**: Memory debugging techniques and integration of GDB with memory analysis tools.

## Assessment Checklist

Before proceeding to the next section, ensure you can:

□ **Step through code** using all stepping commands (`step`, `next`, `stepi`, `nexti`)  
□ **Navigate call stacks** efficiently using `frame`, `up`, `down`, and `backtrace`  
□ **Control program flow** with `continue`, `finish`, `until`, and `jump`  
□ **Debug multi-threaded programs** by switching threads and examining thread-specific state  
□ **Handle signals** during debugging and understand signal-related debugging  
□ **Set and use conditional breakpoints** and commands  
□ **Examine variables and memory** from different stack frames  
□ **Use advanced execution control** features like return value manipulation  
□ **Apply systematic debugging approaches** to different types of bugs  
□ **Create and use GDB scripts** for automation  

### Self-Test Questions

1. What's the difference between `step` and `next` when encountering a function call?
2. How do you examine local variables in a stack frame other than the current one?
3. What command would you use to skip the rest of a long-running loop?
4. How can you debug only one thread while others continue running?
5. What's the purpose of `scheduler-locking` in multi-threaded debugging?
6. How do you force a function to return a specific value during debugging?
7. What's the difference between a breakpoint and a watchpoint?
8. How can you automatically execute commands when a breakpoint is hit?

## Additional Resources

### Books
- "The Art of Debugging with GDB, DDD, and Eclipse" by Norman S. Matloff
- "Debugging with GDB" - Official GNU Manual

### Online Resources
- [GDB Documentation](https://sourceware.org/gdb/current/onlinedocs/gdb/)
- [GDB Tutorial Series](https://www.sourceware.org/gdb/wiki/GDBTutorial)

### Tools Integration
- **IDE Integration**: VSCode GDB extension, CLion debugger
- **Memory Tools**: Valgrind, AddressSanitizer, ThreadSanitizer
- **Profiling**: gprof, perf, Intel VTune
