# Advanced GDB Features

*Duration: 2-3 days*

## Overview
This section covers advanced GDB capabilities that make debugging complex programs more efficient and powerful. You'll learn to leverage scripting, reverse debugging, remote debugging, and core dump analysis to tackle challenging debugging scenarios.

## Learning Objectives

By the end of this section, you should be able to:
- **Write and execute GDB scripts** using both GDB commands and Python
- **Use reverse debugging** to step backwards through program execution
- **Debug programs remotely** across different machines or containers
- **Analyze core dumps** to diagnose crashed programs
- **Create custom debugging commands** and automation scripts
- **Debug multi-threaded applications** effectively
- **Use advanced breakpoint techniques** for complex debugging scenarios

## GDB Scripting and Automation

### Python Scripting in GDB

GDB includes a powerful Python scripting engine that allows you to create sophisticated debugging tools and automate repetitive tasks.

#### Basic Python Integration

**Simple Python Commands:**
```python
# In GDB, you can execute Python directly
(gdb) python print("Hello from GDB Python!")
(gdb) python import sys; print(f"Python version: {sys.version}")
(gdb) python help(gdb)  # Show GDB Python API help
```

**Creating Python Functions in GDB:**
```python
# Define a function in GDB
(gdb) python
>def hello_gdb():
>    print("Hello from GDB Python!")
>    print(f"Current frame: {gdb.selected_frame()}")
>end

# Call the function
(gdb) python hello_gdb()
```

#### Advanced Python Scripting Examples

**Example 1: Custom Memory Dumper**
```python
# Create a script file: memory_dumper.py
import gdb

class MemoryDumper(gdb.Command):
    """Dump memory in a formatted way"""
    
    def __init__(self):
        super(MemoryDumper, self).__init__("memdump", gdb.COMMAND_USER)
    
    def invoke(self, args, from_tty):
        argv = gdb.string_to_argv(args)
        if len(argv) != 2:
            print("Usage: memdump <address> <size>")
            return
        
        address = int(argv[0], 0)  # Support hex addresses
        size = int(argv[1])
        
        try:
            # Read memory
            inferior = gdb.selected_inferior()
            memory = inferior.read_memory(address, size)
            
            # Format output
            print(f"Memory dump at 0x{address:x} ({size} bytes):")
            for i in range(0, size, 16):
                # Hex representation
                hex_part = ""
                ascii_part = ""
                for j in range(16):
                    if i + j < size:
                        byte = memory[i + j]
                        hex_part += f"{byte:02x} "
                        ascii_part += chr(byte) if 32 <= byte <= 126 else '.'
                    else:
                        hex_part += "   "
                
                print(f"0x{address + i:08x}: {hex_part} | {ascii_part}")
                
        except gdb.MemoryError:
            print(f"Cannot access memory at 0x{address:x}")

# Register the command
MemoryDumper()
```

**Example 2: Stack Trace Analyzer**
```python
# stack_analyzer.py
import gdb

class StackAnalyzer(gdb.Command):
    """Analyze stack frames for debugging"""
    
    def __init__(self):
        super(StackAnalyzer, self).__init__("analyze-stack", gdb.COMMAND_USER)
    
    def invoke(self, args, from_tty):
        frame = gdb.selected_frame()
        frame_count = 0
        
        print("Stack Analysis:")
        print("=" * 60)
        
        while frame:
            print(f"Frame #{frame_count}:")
            print(f"  Function: {frame.name() or '<unknown>'}")
            print(f"  PC: 0x{frame.pc():x}")
            
            # Show local variables
            try:
                block = frame.block()
                print("  Local variables:")
                for symbol in block:
                    if symbol.is_variable:
                        try:
                            value = symbol.value(frame)
                            print(f"    {symbol.name} = {value}")
                        except:
                            print(f"    {symbol.name} = <unavailable>")
            except:
                print("  Local variables: <unavailable>")
            
            print()
            frame = frame.older()
            frame_count += 1
            
            if frame_count > 20:  # Prevent infinite loops
                print("... (truncated)")
                break

StackAnalyzer()
```

**Example 3: Breakpoint with Conditions and Actions**
```python
# conditional_breakpoint.py
import gdb

class ConditionalBreakpoint(gdb.Command):
    """Set breakpoint with custom condition and action"""
    
    def __init__(self):
        super(ConditionalBreakpoint, self).__init__("smart-break", gdb.COMMAND_USER)
    
    def invoke(self, args, from_tty):
        argv = gdb.string_to_argv(args)
        if len(argv) < 1:
            print("Usage: smart-break <location> [condition] [action]")
            return
        
        location = argv[0]
        condition = argv[1] if len(argv) > 1 else None
        action = argv[2] if len(argv) > 2 else None
        
        # Create breakpoint
        bp = gdb.Breakpoint(location)
        
        if condition:
            bp.condition = condition
            print(f"Set conditional breakpoint at {location}: {condition}")
        
        if action:
            # Custom action when breakpoint hits
            class BreakpointAction(gdb.Breakpoint):
                def stop(self):
                    print(f"Breakpoint hit: executing '{action}'")
                    gdb.execute(action)
                    return True  # Stop execution
            
            # Replace simple breakpoint with action breakpoint
            bp.delete()
            BreakpointAction(location)

ConditionalBreakpoint()
```

#### Loading and Using Python Scripts

**Method 1: Load script file**
```bash
# In GDB
(gdb) source memory_dumper.py
(gdb) memdump 0x7fff12345678 64
```

**Method 2: Auto-load script**
```bash
# Create .gdbinit file in your project directory
echo "source /path/to/your/scripts/debug_helpers.py" > .gdbinit
```

**Method 3: GDB Python API in standalone script**
```python
#!/usr/bin/env python3
# standalone_debug.py
import gdb
import sys

def analyze_program():
    """Standalone debugging script"""
    # This script can be run with: gdb -x standalone_debug.py ./program
    
    print("Starting automated debugging session...")
    
    # Set breakpoints
    gdb.execute("break main")
    gdb.execute("break malloc")
    
    # Run program
    gdb.execute("run")
    
    # Collect information
    while True:
        try:
            frame = gdb.selected_frame()
            print(f"Stopped at: {frame.name()} at 0x{frame.pc():x}")
            
            # Continue execution
            gdb.execute("continue")
        except gdb.error:
            break
    
    print("Debugging session completed.")

# Execute if running with GDB
if __name__ == "__main__" and "gdb" in sys.modules:
    analyze_program()
```

### GDB Command Scripting

**Creating Custom GDB Commands:**
```bash
# Create a .gdb script file: debug_helpers.gdb

# Define macro for printing array
define parray
    if $argc != 2
        help parray
    else
        set $i = 0
        while $i < $arg1
            print $arg0[$i]
            set $i = $i + 1
        end
    end
end
document parray
Print array elements
Usage: parray array_name size
end

# Define macro for hex dump
define hexdump
    if $argc != 2
        help hexdump
    else
        dump binary memory /tmp/gdb_dump.bin $arg0 $arg0+$arg1
        shell hexdump -C /tmp/gdb_dump.bin
        shell rm /tmp/gdb_dump.bin
    end
end
document hexdump
Hex dump memory region
Usage: hexdump start_address size
end

# Define macro for stack analysis
define stack-locals
    set $frame = 0
    while $frame < 10
        frame $frame
        info locals
        set $frame = $frame + 1
    end
    frame 0
end
document stack-locals
Show local variables for all stack frames
end
```

**Loading and using the script:**
```bash
# In GDB
(gdb) source debug_helpers.gdb
(gdb) parray my_array 10
(gdb) hexdump &buffer 256
(gdb) stack-locals
```

## Reverse Debugging

Reverse debugging allows you to step backwards through program execution, making it easier to understand how your program reached a particular state or to find the root cause of bugs.

### Understanding Reverse Debugging

**How It Works:**
- GDB records the execution history
- You can step backwards through time
- Useful for understanding complex state changes
- Helps track down elusive bugs

**Limitations:**
- Requires recording mode (overhead)
- Not all operations are reversible
- Memory usage increases during recording
- Performance impact during execution

### Basic Reverse Debugging Commands

```bash
# Start recording execution
(gdb) record
(gdb) record full        # More complete recording (slower)

# Basic reverse commands
(gdb) reverse-continue   # Continue backwards until breakpoint
(gdb) reverse-step      # Step backwards one instruction
(gdb) reverse-next      # Step backwards over function calls
(gdb) reverse-finish    # Step backwards out of current function

# Check recording status
(gdb) info record
(gdb) record stop       # Stop recording
```

### Practical Reverse Debugging Example

**Sample Program with Bug:**
```c
// buggy_program.c
#include <stdio.h>
#include <stdlib.h>

int calculate_sum(int* arr, int size) {
    int sum = 0;
    for (int i = 0; i <= size; i++) {  // BUG: should be i < size
        sum += arr[i];  // Buffer overflow!
    }
    return sum;
}

int main() {
    int numbers[] = {1, 2, 3, 4, 5};
    int size = sizeof(numbers) / sizeof(numbers[0]);
    
    printf("Array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", numbers[i]);
    }
    printf("\n");
    
    int result = calculate_sum(numbers, size);
    printf("Sum: %d\n", result);
    
    return 0;
}
```

**Debugging Session with Reverse Debugging:**
```bash
# Compile with debug info
$ gcc -g -o buggy_program buggy_program.c

# Start GDB
$ gdb ./buggy_program

# Set up reverse debugging
(gdb) break main
(gdb) run
(gdb) record                    # Start recording

# Set breakpoint where we suspect issues
(gdb) break calculate_sum
(gdb) continue

# Step through the problematic loop
(gdb) step
(gdb) print i                   # i = 0
(gdb) print arr[i]             # arr[0] = 1
(gdb) next
(gdb) print sum                # sum = 1

# Continue until we hit the bug
(gdb) continue
# ... loop continues ...
# Eventually we get garbage value or crash

# Now use reverse debugging to understand what happened
(gdb) reverse-step             # Go back one step
(gdb) print i                  # What was i when things went wrong?
(gdb) print &arr[i]           # Is this address valid?

# Step backwards through the loop to see pattern
(gdb) reverse-next
(gdb) print i
(gdb) reverse-next
(gdb) print i

# Go back to beginning of function to understand the issue
(gdb) reverse-finish           # Go back to caller
(gdb) reverse-step             # Step back into function call
(gdb) print size               # Check the size parameter
```

### Advanced Reverse Debugging Techniques

**Example: Finding Memory Corruption Source**
```c
// memory_corruption.c
#include <stdio.h>
#include <string.h>

char global_buffer[100];

void function_a() {
    strcpy(global_buffer, "Hello from function A");
}

void function_b() {
    // BUG: Writing beyond buffer bounds
    strcpy(global_buffer, "This is a very long string that exceeds the buffer size and causes memory corruption issues");
}

void function_c() {
    printf("Buffer contents: %s\n", global_buffer);
}

int main() {
    function_a();
    printf("After A: %s\n", global_buffer);
    
    function_b();  // This corrupts memory
    printf("After B: %s\n", global_buffer);
    
    function_c();
    return 0;
}
```

**Reverse debugging session:**
```bash
(gdb) break main
(gdb) run
(gdb) record

# Set watchpoint on the buffer
(gdb) watch global_buffer
(gdb) continue

# When watchpoint hits, use reverse debugging
(gdb) info registers           # Check current state
(gdb) reverse-step            # Go back to see what modified buffer
(gdb) print $pc               # Where were we?
(gdb) disassemble             # What instruction caused the change?

# Continue reverse stepping to find root cause
(gdb) reverse-continue        # Go back to previous watchpoint hit
(gdb) backtrace              # What call stack led to this?
```

### Reverse Debugging with Core Dumps

**Analyzing crash with reverse debugging:**
```bash
# If program crashes and produces core dump
$ gdb ./program core

# Even with core dump, you can use some reverse debugging
(gdb) info registers
(gdb) backtrace
(gdb) frame 0

# Look at assembly around crash
(gdb) disassemble
(gdb) info line            # Source line info

# Examine memory state
(gdb) x/10i $pc-20         # Instructions before crash
(gdb) info locals          # Local variables at crash
```

### Performance Considerations

**Recording Overhead:**
```bash
# Check recording statistics
(gdb) info record
# Shows: instruction count, memory usage, etc.

# Optimize recording for specific scenarios
(gdb) record full          # Complete but slow
(gdb) record btrace        # Branch trace only (faster)
(gdb) record btrace pt     # Intel Processor Trace (fastest)
```

**Best Practices:**
- Enable recording only when needed
- Use `record btrace` for better performance
- Set recording limits to manage memory usage
- Stop recording after capturing the problematic area

```bash
# Set recording limits
(gdb) set record full insn-number-max 200000
(gdb) set record full memory-query on
```

## Remote Debugging

Remote debugging allows you to debug programs running on different machines, containers, or embedded systems. This is essential for debugging server applications, embedded systems, or containerized applications.

### Setting Up Remote Debugging

#### Method 1: Using gdbserver

**On the target machine (where program runs):**
```bash
# Install gdbserver
sudo apt-get install gdbserver

# Start program under gdbserver
gdbserver localhost:1234 ./my_program arg1 arg2

# Or attach to running process
gdbserver localhost:1234 --attach <PID>

# For network debugging (listen on all interfaces)
gdbserver 0.0.0.0:1234 ./my_program
```

**On the development machine (where you debug):**
```bash
# Start GDB with your program
gdb ./my_program

# Connect to remote target
(gdb) target remote localhost:1234

# Or connect over network
(gdb) target remote 192.168.1.100:1234

# Now debug normally
(gdb) break main
(gdb) continue
```

#### Method 2: SSH Tunneling for Secure Remote Debugging

```bash
# Create SSH tunnel (on development machine)
ssh -L 1234:localhost:1234 user@remote-server

# On remote server
gdbserver localhost:1234 ./program

# On development machine
gdb ./program
(gdb) target remote localhost:1234
```

#### Method 3: Docker Container Debugging

**Dockerfile with debugging support:**
```dockerfile
FROM ubuntu:20.04

# Install debugging tools
RUN apt-get update && apt-get install -y \
    gdb \
    gdbserver \
    build-essential

# Copy your program
COPY my_program /usr/local/bin/
COPY debug_symbols /usr/local/bin/

# Expose debugging port
EXPOSE 1234

# Start with gdbserver
CMD ["gdbserver", "0.0.0.0:1234", "/usr/local/bin/my_program"]
```

**Running and debugging the container:**
```bash
# Build and run container
docker build -t my-debug-app .
docker run -p 1234:1234 my-debug-app

# Connect from host
gdb ./my_program
(gdb) target remote localhost:1234
```

### Advanced Remote Debugging Scenarios

#### Cross-Platform Debugging

**Debugging ARM binary from x86 machine:**
```bash
# On ARM target
gdbserver localhost:1234 ./arm_program

# On x86 development machine with cross-GDB
arm-linux-gnueabihf-gdb ./arm_program
(gdb) target remote arm-device-ip:1234
(gdb) set architecture arm
(gdb) break main
(gdb) continue
```

#### Multi-Process Remote Debugging

```bash
# Debug parent and child processes
(gdb) set follow-fork-mode child    # Follow child processes
(gdb) set detach-on-fork off        # Keep debugging parent too
(gdb) target remote localhost:1234
(gdb) info inferiors                # List all processes being debugged
(gdb) inferior 2                    # Switch to process 2
```

#### Remote Core Dump Analysis

```bash
# Copy core dump from remote machine
scp user@remote:/path/to/core.dump ./

# Copy program binary and libraries
scp user@remote:/path/to/program ./
scp -r user@remote:/lib/x86_64-linux-gnu/ ./libs/

# Debug core dump
gdb ./program core.dump
(gdb) set solib-search-path ./libs  # Tell GDB where to find libraries
(gdb) backtrace
```

## Core Dump Analysis

Core dumps are snapshots of a program's memory when it crashes. They're invaluable for post-mortem debugging, especially in production environments.

### Enabling Core Dumps

**System Configuration:**
```bash
# Check current core dump settings
ulimit -c

# Enable unlimited core dumps
ulimit -c unlimited

# Make permanent (add to ~/.bashrc)
echo "ulimit -c unlimited" >> ~/.bashrc

# System-wide core dump configuration
sudo vim /etc/security/limits.conf
# Add: * soft core unlimited

# Configure core dump location and naming
echo '/tmp/core.%e.%p.%t' | sudo tee /proc/sys/kernel/core_pattern
# %e = executable name, %p = PID, %t = timestamp
```

**Program-specific core dump generation:**
```c
// Force core dump in program
#include <signal.h>
#include <stdlib.h>

void crash_handler(int sig) {
    printf("Received signal %d, generating core dump\n", sig);
    abort();  // This will generate core dump
}

int main() {
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    
    // Your program logic
    return 0;
}
```

### Analyzing Core Dumps

#### Basic Core Dump Analysis

```bash
# Load core dump in GDB
gdb ./program core.12345

# Basic information
(gdb) info program          # Why did it crash?
(gdb) info registers        # CPU state at crash
(gdb) backtrace            # Call stack
(gdb) info locals          # Local variables
(gdb) info args            # Function arguments
```

#### Advanced Core Dump Analysis

**Example: Analyzing a Segmentation Fault**
```c
// crash_program.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[50];
    int age;
    double salary;
} Person;

void process_person(Person* p) {
    printf("Processing: %s, age %d\n", p->name, p->age);
    
    // Bug: p might be NULL
    p->salary *= 1.1;  // Potential segfault here
}

int main() {
    Person* people[5];
    
    // Initialize some people
    for (int i = 0; i < 3; i++) {
        people[i] = malloc(sizeof(Person));
        sprintf(people[i]->name, "Person%d", i);
        people[i]->age = 25 + i;
        people[i]->salary = 50000.0;
    }
    
    // Bug: indices 3 and 4 are uninitialized (NULL)
    
    // Process all people (will crash on NULL pointer)
    for (int i = 0; i < 5; i++) {
        process_person(people[i]);  // Crash here when i >= 3
    }
    
    return 0;
}
```

**Comprehensive core dump analysis:**
```bash
# Compile with debug info
gcc -g -o crash_program crash_program.c

# Run and generate core
./crash_program
# Segmentation fault (core dumped)

# Analyze core dump
gdb ./crash_program core

# Detailed analysis
(gdb) info program
Program terminated with signal SIGSEGV, Segmentation fault.
#0  0x0000000000401234 in process_person (p=0x0) at crash_program.c:15

# Examine the crash point
(gdb) list
(gdb) print p                    # p = 0x0 (NULL pointer)
(gdb) backtrace full            # Full stack trace with locals

# Examine the calling context
(gdb) frame 1                   # Go to caller frame
(gdb) print i                   # What was the loop index?
(gdb) print people              # Examine the array
(gdb) print people[0]@5         # Print all 5 elements

# Memory examination
(gdb) x/5gx people              # Examine people array as pointers
(gdb) info heap                 # If available, show heap info
```

#### Automated Core Dump Analysis Script

```python
# core_analyzer.py - Python script for automated analysis
import gdb
import re

class CoreAnalyzer(gdb.Command):
    """Automated core dump analysis"""
    
    def __init__(self):
        super(CoreAnalyzer, self).__init__("analyze-core", gdb.COMMAND_USER)
    
    def invoke(self, args, from_tty):
        print("=== Automated Core Dump Analysis ===")
        
        # 1. Basic crash information
        print("\n1. CRASH INFORMATION:")
        try:
            gdb.execute("info program")
        except:
            print("No program information available")
        
        # 2. Register state
        print("\n2. REGISTER STATE:")
        try:
            gdb.execute("info registers")
        except:
            print("No register information available")
        
        # 3. Stack trace
        print("\n3. STACK TRACE:")
        try:
            gdb.execute("backtrace full")
        except:
            print("No stack trace available")
        
        # 4. Memory around crash
        print("\n4. MEMORY ANALYSIS:")
        try:
            frame = gdb.selected_frame()
            pc = frame.pc()
            print(f"Crash at PC: 0x{pc:x}")
            
            # Disassemble around crash point
            gdb.execute(f"disassemble {pc-32},{pc+32}")
        except:
            print("Cannot analyze memory around crash")
        
        # 5. Thread information
        print("\n5. THREAD INFORMATION:")
        try:
            gdb.execute("info threads")
        except:
            print("No thread information")
        
        # 6. Shared libraries
        print("\n6. SHARED LIBRARIES:")
        try:
            gdb.execute("info sharedlibrary")
        except:
            print("No shared library information")
        
        print("\n=== Analysis Complete ===")

CoreAnalyzer()
```

**Usage:**
```bash
gdb ./program core
(gdb) source core_analyzer.py
(gdb) analyze-core
```

### Core Dump Best Practices

#### Production Environment Setup

```bash
# 1. Configure core dump collection
# /etc/systemd/system.conf
DefaultLimitCORE=infinity

# 2. Set up core dump processing with systemd-coredump
sudo vim /etc/systemd/coredump.conf
# Storage=external
# Compress=yes
# ProcessSizeMax=2G

# 3. Create core dump analysis script
#!/bin/bash
# /usr/local/bin/analyze-crash.sh
PROGRAM=$1
CORE=$2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="/var/log/crash-reports/crash_${TIMESTAMP}.txt"

echo "Analyzing crash: $PROGRAM with core $CORE" | tee $REPORT
gdb -batch -ex "bt" -ex "info registers" -ex "quit" $PROGRAM $CORE >> $REPORT
```

#### Debugging Strategies for Different Crash Types

**NULL Pointer Dereference:**
```bash
(gdb) print variable_that_crashed
(gdb) print &variable_that_crashed
(gdb) backtrace
(gdb) frame 1
(gdb) print *pointer_name  # Check if pointer is valid
```

**Buffer Overflow:**
```bash
(gdb) info locals
(gdb) print sizeof(buffer)
(gdb) x/50c buffer_address  # Examine buffer contents
(gdb) print strlen(string_var)  # Check string lengths
```

**Stack Overflow:**
```bash
(gdb) backtrace 100        # Check for recursive calls
(gdb) info frame           # Check stack pointer
(gdb) print $sp            # Stack pointer value
```

**Double Free / Use After Free:**
```bash
(gdb) print heap_pointer
(gdb) x/10gx heap_pointer   # Examine heap memory
(gdb) info heap            # If available
```

## Advanced Breakpoint Techniques

### Conditional Breakpoints with Complex Logic

```bash
# Break only when specific conditions are met
(gdb) break function_name if variable > 100 && other_var != NULL

# Break on specific thread
(gdb) break main thread 2

# Break with hit count
(gdb) break function_name
(gdb) condition 1 ($bpnum > 10)  # Break only after 10th hit

# Temporary breakpoint
(gdb) tbreak function_name  # Deleted after first hit
```

### Watchpoints for Memory Debugging

```bash
# Watch for memory changes
(gdb) watch global_variable           # Break when value changes
(gdb) rwatch global_variable         # Break when value is read
(gdb) awatch global_variable         # Break on read OR write

# Watch memory range
(gdb) watch *((int*)0x12345678)      # Watch specific memory address
(gdb) watch *array@10                # Watch array of 10 elements
```

### Tracepoints for Non-Intrusive Debugging

```bash
# Collect data without stopping execution
(gdb) trace function_name
(gdb) actions 1
> collect parameter1, parameter2, $locals
> end

# Start tracing
(gdb) tstart

# View collected data
(gdb) tfind start
(gdb) print parameter1
(gdb) tfind next
```

## Multi-threaded Debugging

### Thread-Specific Debugging

```bash
# List all threads
(gdb) info threads

# Switch to specific thread
(gdb) thread 3

# Apply command to all threads
(gdb) thread apply all backtrace

# Set breakpoints for specific threads
(gdb) break file.c:123 thread 2

# Control thread execution
(gdb) set scheduler-locking on      # Only current thread runs
(gdb) set scheduler-locking step    # Only during stepping
(gdb) set scheduler-locking off     # All threads run
```

### Debugging Race Conditions

```c
// race_condition.c
#include <pthread.h>
#include <stdio.h>

int shared_counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* worker_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < 10000; i++) {
        // Race condition here
        pthread_mutex_lock(&mutex);
        shared_counter++;
        pthread_mutex_unlock(&mutex);
    }
    
    printf("Thread %d finished\n", thread_id);
    return NULL;
}

int main() {
    pthread_t threads[4];
    int thread_ids[4];
    
    for (int i = 0; i < 4; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, worker_thread, &thread_ids[i]);
    }
    
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Final counter: %d\n", shared_counter);
    return 0;
}
```

**Debugging the race condition:**
```bash
# Compile with thread debugging
gcc -g -pthread -o race_condition race_condition.c

# Debug
gdb ./race_condition

# Set up thread debugging
(gdb) set print thread-events on
(gdb) break worker_thread
(gdb) run

# When breakpoint hits
(gdb) info threads
(gdb) thread apply all where

# Watch the shared variable
(gdb) watch shared_counter
(gdb) continue

# Analyze thread interactions
(gdb) thread apply all print shared_counter
(gdb) thread apply all backtrace
```

## Study Materials and Practice

### Recommended Reading
- **Primary:** "Debugging with GDB" - Official GDB Manual (Chapter 23-27: Advanced Features)
- **Python Scripting:** "GDB Python API Documentation" 
- **Core Dumps:** "Advanced Linux Programming" by CodeSourcery LLC
- **Remote Debugging:** "Linux System Programming" by Robert Love

### Hands-on Exercises

**Exercise 1: Create a Python GDB Extension**
```python
# TODO: Create a GDB extension that:
# 1. Finds all string variables in current frame
# 2. Checks for potential buffer overflows
# 3. Displays a security analysis report
```

**Exercise 2: Reverse Debugging Challenge**
```c
// TODO: Debug this program using reverse debugging
// Find why the array gets corrupted
int mysterious_bug() {
    int arr[10] = {1,2,3,4,5,6,7,8,9,10};
    for (int i = 0; i < 15; i++) {  // Bug is here
        process_element(arr[i]);
    }
    return calculate_checksum(arr);
}
```

**Exercise 3: Remote Debugging Setup**
```bash
# TODO: Set up remote debugging environment
# 1. Create Docker container with gdbserver
# 2. Debug a web server application remotely
# 3. Analyze performance bottlenecks
```

### Assessment Checklist

□ Can write and execute Python scripts in GDB  
□ Understand reverse debugging concepts and limitations  
□ Can set up remote debugging with gdbserver  
□ Know how to analyze core dumps effectively  
□ Can debug multi-threaded applications  
□ Understand advanced breakpoint and watchpoint techniques  
□ Can create custom GDB commands and automation  

## Next Section
[Memory Analysis and Leak Detection](../02_Memory_Leak_Detection/01_Valgrind_Basics.md)
