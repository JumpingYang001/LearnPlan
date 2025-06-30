# Debugging Multi-threaded Programs

*Duration: 1-2 weeks*

## Overview

Debugging multi-threaded programs presents unique challenges that don't exist in single-threaded applications. This guide covers comprehensive techniques for debugging thread creation, synchronization issues, race conditions, deadlocks, and performance problems using GDB and other specialized tools.

### Key Debugging Challenges in Multi-threaded Programs

1. **Non-deterministic behavior** - Bugs may not reproduce consistently
2. **Race conditions** - Timing-dependent errors
3. **Deadlocks** - Threads waiting indefinitely for each other
4. **Thread interference** - One thread affecting another's execution
5. **Complex state management** - Multiple execution contexts simultaneously

## Essential GDB Commands for Thread Debugging

### Basic Thread Navigation

```bash
# List all threads in the program
(gdb) info threads

# Switch to a specific thread
(gdb) thread <thread_number>

# Show current thread
(gdb) thread

# Apply command to all threads
(gdb) thread apply all <command>

# Show backtrace for all threads
(gdb) thread apply all bt

# Show detailed thread information
(gdb) info threads verbose
```

### Thread-Specific Breakpoints and Watchpoints

```bash
# Set breakpoint for specific thread only
(gdb) break <function> thread <thread_number>

# Conditional breakpoint based on thread ID
(gdb) break <function> if pthread_self() == <thread_id>

# Set breakpoint that stops all threads
(gdb) break <function>

# Set breakpoint that only stops current thread
(gdb) set scheduler-locking on
(gdb) break <function>
```

## Comprehensive Threading Debug Examples

### Example 1: Basic Thread Creation and Debugging

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

typedef struct {
    int thread_id;
    int iterations;
    int* shared_counter;
} thread_data_t;

void* thread_func(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    
    printf("Thread %d starting with %d iterations\n", 
           data->thread_id, data->iterations);
    
    for (int i = 0; i < data->iterations; i++) {
        // Simulate some work
        usleep(100000); // 100ms
        
        // Update shared counter (potential race condition)
        (*data->shared_counter)++;
        
        printf("Thread %d: iteration %d, counter = %d\n", 
               data->thread_id, i, *data->shared_counter);
    }
    
    printf("Thread %d finished\n", data->thread_id);
    return NULL;
}

int main() {
    const int NUM_THREADS = 3;
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    int shared_counter = 0;
    
    printf("Creating %d threads...\n", NUM_THREADS);
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i + 1;
        thread_data[i].iterations = 5;
        thread_data[i].shared_counter = &shared_counter;
        
        if (pthread_create(&threads[i], NULL, thread_func, &thread_data[i]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            exit(1);
        }
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Final counter value: %d\n", shared_counter);
    return 0;
}
```

**GDB Debugging Session:**
```bash
# Compile with debug symbols
gcc -g -pthread -o thread_debug thread_debug.c

# Start GDB
gdb ./thread_debug

# Set breakpoints
(gdb) break thread_func
(gdb) break main

# Run the program
(gdb) run

# When it hits main breakpoint
(gdb) continue

# When it hits thread_func, check which thread
(gdb) info threads
(gdb) print data->thread_id

# Set conditional breakpoint for specific thread
(gdb) break thread_func if data->thread_id == 2

# Continue and observe different threads
(gdb) continue

# Check all thread backtraces
(gdb) thread apply all bt

# Step through code in current thread only
(gdb) set scheduler-locking on
(gdb) step

# Allow all threads to run
(gdb) set scheduler-locking off
```

### Example 2: Race Condition Detection

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 4
#define INCREMENTS_PER_THREAD 100000

int global_counter = 0;
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

// Unsafe version - race condition
void* unsafe_increment(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < INCREMENTS_PER_THREAD; i++) {
        // RACE CONDITION: Multiple threads can read-modify-write simultaneously
        int temp = global_counter;  // READ
        temp++;                     // MODIFY  
        global_counter = temp;      // WRITE
        
        // Add some debugging info
        if (i % 10000 == 0) {
            printf("Thread %d: i=%d, counter=%d\n", thread_id, i, global_counter);
        }
    }
    
    return NULL;
}

// Safe version - with mutex
void* safe_increment(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < INCREMENTS_PER_THREAD; i++) {
        pthread_mutex_lock(&counter_mutex);
        
        int temp = global_counter;
        temp++;
        global_counter = temp;
        
        pthread_mutex_unlock(&counter_mutex);
        
        if (i % 10000 == 0) {
            printf("Thread %d: i=%d, counter=%d\n", thread_id, i, global_counter);
        }
    }
    
    return NULL;
}

int main(int argc, char* argv[]) {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    int use_safe_version = (argc > 1) ? atoi(argv[1]) : 0;
    
    printf("Using %s version\n", use_safe_version ? "safe" : "unsafe");
    printf("Expected final value: %d\n", NUM_THREADS * INCREMENTS_PER_THREAD);
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i + 1;
        
        void* (*thread_func)(void*) = use_safe_version ? safe_increment : unsafe_increment;
        
        if (pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            exit(1);
        }
    }
    
    // Wait for threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Final counter value: %d\n", global_counter);
    printf("Expected: %d, Actual: %d, Difference: %d\n", 
           NUM_THREADS * INCREMENTS_PER_THREAD, 
           global_counter,
           (NUM_THREADS * INCREMENTS_PER_THREAD) - global_counter);
    
    return 0;
}
```

**Advanced GDB Race Condition Debugging:**
```bash
# Compile with debug info and no optimization
gcc -g -O0 -pthread -o race_debug race_debug.c

# Start GDB
gdb ./race_debug

# Set watchpoint on global variable
(gdb) watch global_counter

# Set breakpoint in the critical section
(gdb) break unsafe_increment
(gdb) condition 1 i == 50000  # Stop at specific iteration

# Run with unsafe version
(gdb) run 0

# When watchpoint triggers, examine the situation
(gdb) info threads
(gdb) thread apply all bt
(gdb) print global_counter
(gdb) print temp

# Step through the race condition
(gdb) set scheduler-locking on
(gdb) step
(gdb) print temp
(gdb) step
(gdb) print global_counter

# Switch to another thread to see interference
(gdb) thread 2
(gdb) step
```

### Example 3: Deadlock Detection and Analysis

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void* thread1_function(void* arg) {
    printf("Thread 1: Trying to acquire mutex1...\n");
    pthread_mutex_lock(&mutex1);
    printf("Thread 1: Acquired mutex1\n");
    
    // Simulate some work
    printf("Thread 1: Working with mutex1...\n");
    sleep(2);
    
    printf("Thread 1: Trying to acquire mutex2...\n");
    pthread_mutex_lock(&mutex2);  // POTENTIAL DEADLOCK
    printf("Thread 1: Acquired mutex2\n");
    
    printf("Thread 1: Working with both mutexes...\n");
    sleep(1);
    
    printf("Thread 1: Releasing mutexes...\n");
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    
    printf("Thread 1: Finished\n");
    return NULL;
}

void* thread2_function(void* arg) {
    printf("Thread 2: Trying to acquire mutex2...\n");
    pthread_mutex_lock(&mutex2);
    printf("Thread 2: Acquired mutex2\n");
    
    // Simulate some work
    printf("Thread 2: Working with mutex2...\n");
    sleep(2);
    
    printf("Thread 2: Trying to acquire mutex1...\n");
    pthread_mutex_lock(&mutex1);  // POTENTIAL DEADLOCK
    printf("Thread 2: Acquired mutex1\n");
    
    printf("Thread 2: Working with both mutexes...\n");
    sleep(1);
    
    printf("Thread 2: Releasing mutexes...\n");
    pthread_mutex_unlock(&mutex1);
    pthread_mutex_unlock(&mutex2);
    
    printf("Thread 2: Finished\n");
    return NULL;
}

int main() {
    pthread_t thread1, thread2;
    
    printf("Starting deadlock demonstration...\n");
    
    // Create threads that will deadlock
    pthread_create(&thread1, NULL, thread1_function, NULL);
    pthread_create(&thread2, NULL, thread2_function, NULL);
    
    // Wait for threads (this will hang due to deadlock)
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    
    printf("Program completed\n");  // This line will never execute
    return 0;
}
```

**Deadlock Debugging in GDB:**
```bash
# Compile the deadlock program
gcc -g -pthread -o deadlock_debug deadlock_debug.c

# Start GDB
gdb ./deadlock_debug

# Set breakpoints at critical sections
(gdb) break thread1_function
(gdb) break thread2_function
(gdb) break pthread_mutex_lock

# Run the program
(gdb) run

# Let it run until it hangs (deadlock occurs)
(gdb) continue
# ... wait for deadlock ...

# Interrupt the program (Ctrl+C)
^C

# Check all threads
(gdb) info threads

# Examine each thread's stack
(gdb) thread 1
(gdb) bt
(gdb) frame 0
(gdb) print mutex1
(gdb) print mutex2

(gdb) thread 2  
(gdb) bt
(gdb) frame 0

# Check mutex ownership
(gdb) print mutex1.__data.__owner
(gdb) print mutex2.__data.__owner

# Examine the deadlock situation
(gdb) thread apply all bt
```

## Advanced Debugging Techniques

### Scheduler Locking in GDB

Scheduler locking controls how GDB handles thread execution during debugging:

```bash
# Show current scheduler locking mode
(gdb) show scheduler-locking

# Lock scheduler - only current thread runs
(gdb) set scheduler-locking on

# Step mode - only current thread runs during stepping
(gdb) set scheduler-locking step  

# Unlock scheduler - all threads run freely
(gdb) set scheduler-locking off
```

**Practical Example:**
```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

void* worker_thread(void* arg) {
    int id = *(int*)arg;
    
    for (int i = 0; i < 5; i++) {
        printf("Thread %d: step %d\n", id, i);
        sleep(1);  // Breakpoint here for stepping demo
    }
    
    return NULL;
}

int main() {
    pthread_t threads[3];
    int ids[] = {1, 2, 3};
    
    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, worker_thread, &ids[i]);
    }
    
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }
    
    return 0;
}
```

**Debugging with Scheduler Control:**
```bash
(gdb) break worker_thread
(gdb) run

# When all threads hit breakpoint
(gdb) info threads
  Id   Target Id         Frame 
* 1    Thread 0x7f... main () at main.c:20
  2    Thread 0x7f... worker_thread (arg=0x...) at main.c:6
  3    Thread 0x7f... worker_thread (arg=0x...) at main.c:6
  4    Thread 0x7f... worker_thread (arg=0x...) at main.c:6

# Control execution of specific threads
(gdb) thread 2
(gdb) set scheduler-locking on
(gdb) step    # Only thread 2 steps forward
(gdb) step
(gdb) print id

(gdb) thread 3
(gdb) step    # Only thread 3 steps forward
(gdb) print id

# Resume all threads
(gdb) set scheduler-locking off
(gdb) continue
```

### Thread-Safe Debugging with Logging

```c
#include <pthread.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <sys/time.h>

pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

void thread_safe_log(const char* format, ...) {
    pthread_mutex_lock(&log_mutex);
    
    // Get timestamp
    struct timeval tv;
    gettimeofday(&tv, NULL);
    
    // Get thread ID  
    pthread_t thread_id = pthread_self();
    
    // Print timestamp and thread ID
    printf("[%ld.%06ld] Thread %lu: ", tv.tv_sec, tv.tv_usec, thread_id);
    
    // Print user message
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    
    printf("\n");
    fflush(stdout);
    
    pthread_mutex_unlock(&log_mutex);
}

// Example usage in threaded code
void* debug_worker(void* arg) {
    int worker_id = *(int*)arg;
    
    thread_safe_log("Worker %d started", worker_id);
    
    for (int i = 0; i < 3; i++) {
        thread_safe_log("Worker %d: processing item %d", worker_id, i);
        usleep(100000);  // 100ms
    }
    
    thread_safe_log("Worker %d finished", worker_id);
    return NULL;
}
```

### Memory Debugging with Valgrind

**Helgrind - Race Condition Detection:**
```bash
# Compile with debug info
gcc -g -pthread -o program program.c

# Run with Helgrind to detect race conditions
valgrind --tool=helgrind ./program

# More verbose output
valgrind --tool=helgrind --verbose ./program

# Save output to file
valgrind --tool=helgrind --log-file=helgrind.log ./program
```

**Example Helgrind Output:**
```
==12345== Helgrind, a thread error detector
==12345== Copyright (C) 2007-2017, and GNU GPL'd, by OpenWorks LLP et al.
==12345== Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info
==12345== Command: ./race_debug 0
==12345== 
==12345== ---Thread-Announcement------------------------------------------
==12345== 
==12345== Thread #1 is the program's root thread
==12345== 
==12345== Thread #2 was created
==12345==    at 0x4E4C6B2: clone (clone.S:71)
==12345==    by 0x4E3EE99: create_thread (createthread.c:101)
==12345==    by 0x4E406D3: pthread_create@@GLIBC_2.2.5 (pthread_create.c:805)
==12345==    by 0x108A89: main (race_debug.c:67)
==12345== 
==12345== ----------------------------------------------------------------
==12345== 
==12345== Possible data race during read of size 4 at 0x109034 by thread #1
==12345== Locks held: none
==12345==    at 0x1089E1: unsafe_increment (race_debug.c:17)
==12345==    by 0x4E3F608: start_thread (pthread_create.c:477)
==12345==    by 0x4E4C6B2: clone (clone.S:71)
==12345== 
==12345== This conflicts with a previous write of size 4 by thread #2
==12345== Locks held: none
==12345==    at 0x1089F1: unsafe_increment (race_debug.c:19)
==12345==    by 0x4E3F608: start_thread (pthread_create.c:477)
==12345==    by 0x4E4C6B2: clone (clone.S:71)
```

**DRD - Another Race Detection Tool:**
```bash
# Alternative to Helgrind
valgrind --tool=drd ./program

# With additional options
valgrind --tool=drd --check-stack-var=yes --trace-mutex=yes ./program
```

### Thread Sanitizer (ThreadSanitizer)

**Compile with ThreadSanitizer:**
```bash
# GCC/Clang with ThreadSanitizer
gcc -g -fsanitize=thread -fPIE -pie -pthread -o program program.c

# Run the program
./program
```

**Example ThreadSanitizer Output:**
```
==================
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 4 at 0x7b0400000000 by thread T2:
    #0 unsafe_increment /path/to/race_debug.c:19:20
    #1 <null> <null>

  Previous read of size 4 at 0x7b0400000000 by thread T1:
    #0 unsafe_increment /path/to/race_debug.c:17:16
    #1 <null> <null>

  Location is global 'global_counter' of size 4 at 0x7b0400000000 (race_debug+0x000000000000)

  Thread T2 (tid=12347, running) created by main thread at:
    #0 pthread_create ../../../../src/libsanitizer/tsan/tsan_interceptors.cc:969
    #1 main /path/to/race_debug.c:67:12

  Thread T1 (tid=12346, running) created by main thread at:
    #0 pthread_create ../../../../src/libsanitizer/tsan/tsan_interceptors.cc:969
    #1 main /path/to/race_debug.c:67:12

SUMMARY: ThreadSanitizer: data race /path/to/race_debug.c:19:20 in unsafe_increment
==================
```

## Performance Debugging

### CPU Profiling with Perf

```bash
# Record performance data for multi-threaded program
perf record -g ./program

# Analyze the recorded data
perf report

# Record specific events
perf record -e cache-misses,context-switches -g ./program

# Real-time monitoring
perf top -p $(pgrep program_name)
```

### Thread Contention Analysis

```c
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

pthread_mutex_t contended_mutex = PTHREAD_MUTEX_INITIALIZER;
int critical_section_counter = 0;

void* contending_thread(void* arg) {
    int thread_id = *(int*)arg;
    struct timespec start, end;
    
    for (int i = 0; i < 10; i++) {
        // Measure time waiting for mutex
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        pthread_mutex_lock(&contended_mutex);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double wait_time = (end.tv_sec - start.tv_sec) + 
                          (end.tv_nsec - start.tv_nsec) / 1e9;
        
        // Hold mutex for varying amounts of time
        usleep((thread_id % 3 + 1) * 10000);  // 10-30ms
        
        critical_section_counter++;
        printf("Thread %d: waited %.6f seconds, counter=%d\n", 
               thread_id, wait_time, critical_section_counter);
        
        pthread_mutex_unlock(&contended_mutex);
        
        // Do some work outside critical section
        usleep(5000);  // 5ms
    }
    
    return NULL;
}

int main() {
    const int NUM_THREADS = 5;
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    
    printf("Starting contention analysis with %d threads\n", NUM_THREADS);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i + 1;
        pthread_create(&threads[i], NULL, contending_thread, &thread_ids[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Contention analysis completed\n");
    return 0;
}
```

### Debugging with Core Dumps

```bash
# Enable core dumps
ulimit -c unlimited

# Run program that crashes
./buggy_program

# Debug with core dump
gdb ./buggy_program core

# In GDB, examine the crash
(gdb) bt
(gdb) info threads
(gdb) thread apply all bt
(gdb) print variable_name
(gdb) frame 1
(gdb) list
```

## Common Debugging Scenarios

### Scenario 1: Thread Not Starting

**Problem:** Thread creation appears to succeed but thread never executes

**Debug Approach:**
```bash
(gdb) break pthread_create
(gdb) run
(gdb) print $rax  # Check return value
(gdb) break thread_function
(gdb) continue
# If breakpoint not hit, check thread creation parameters
```

### Scenario 2: Intermittent Crashes

**Problem:** Program crashes randomly, hard to reproduce

**Debug Approach:**
```bash
# Run with core dumps enabled
ulimit -c unlimited

# Use AddressSanitizer
gcc -g -fsanitize=address -pthread -o program program.c

# Run multiple times with different conditions
for i in {1..100}; do
    echo "Run $i"
    ./program || break
done
```

### Scenario 3: Performance Degradation

**Problem:** Multi-threaded program slower than expected

**Debug Approach:**
```bash
# Check for lock contention
perf record -e context-switches -g ./program
perf report

# Monitor thread activity
top -H -p $(pgrep program)

# Use strace to see system calls
strace -f -e trace=futex ./program
```

## Learning Objectives

By the end of this section, you should be able to:

- **Navigate multi-threaded programs** in GDB effectively
- **Set and use thread-specific breakpoints** and watchpoints
- **Detect and analyze race conditions** using various tools
- **Identify and debug deadlock situations** step by step
- **Use scheduler locking** to control thread execution during debugging
- **Apply thread sanitizers** to catch threading bugs automatically
- **Profile multi-threaded applications** for performance issues
- **Debug thread synchronization problems** systematically

### Self-Assessment Checklist

□ Can switch between threads in GDB and examine their state  
□ Can set conditional breakpoints for specific threads  
□ Can identify race conditions in code and fix them  
□ Can detect deadlocks and understand their cause  
□ Can use Valgrind/Helgrind to find threading bugs  
□ Can interpret ThreadSanitizer output  
□ Can measure and optimize thread performance  
□ Can debug core dumps from multi-threaded crashes  

## Practice Exercises

### Exercise 1: Debug Race Condition
```c
// TODO: Find and fix the race condition in this code
int shared_balance = 1000;

void* withdraw_money(void* amount) {
    int withdrawal = *(int*)amount;
    
    if (shared_balance >= withdrawal) {
        printf("Withdrawing %d...\n", withdrawal);
        shared_balance -= withdrawal;  // Race condition here!
        printf("New balance: %d\n", shared_balance);
    } else {
        printf("Insufficient funds\n");
    }
    
    return NULL;
}
```

### Exercise 2: Deadlock Prevention
```c
// TODO: Modify this code to prevent deadlock
pthread_mutex_t account1_mutex;
pthread_mutex_t account2_mutex;

void transfer_money(int from_account, int to_account, int amount) {
    if (from_account == 1) {
        pthread_mutex_lock(&account1_mutex);
        pthread_mutex_lock(&account2_mutex);  // Potential deadlock
    } else {
        pthread_mutex_lock(&account2_mutex);
        pthread_mutex_lock(&account1_mutex);  // Potential deadlock
    }
    
    // Transfer logic here
    
    pthread_mutex_unlock(&account2_mutex);
    pthread_mutex_unlock(&account1_mutex);
}
```

### Exercise 3: Performance Analysis
```c
// TODO: Analyze why this multi-threaded program is slow
pthread_mutex_t global_mutex = PTHREAD_MUTEX_INITIALIZER;

void* worker_thread(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        pthread_mutex_lock(&global_mutex);
        // Very short critical section
        int temp = i * 2;
        pthread_mutex_unlock(&global_mutex);
    }
    return NULL;
}
```

## Study Materials and References

### Essential Reading
- **"Programming with POSIX Threads"** by David R. Butenhof - Chapters 8-9
- **"The Art of Debugging with GDB, DDD, and Eclipse"** - Multi-threading chapter
- **GDB Manual** - Thread debugging sections

### Online Resources
- [GDB Thread Debugging Documentation](https://sourceware.org/gdb/onlinedocs/gdb/Threads.html)
- [Valgrind Helgrind Manual](https://valgrind.org/docs/manual/hg-manual.html)
- [ThreadSanitizer Documentation](https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual)

### Tools Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install gdb valgrind
sudo apt-get install linux-tools-common linux-tools-generic  # For perf
```

**Compilation Flags:**
```bash
# Debug build
gcc -g -pthread -Wall -Wextra -o program program.c

# With ThreadSanitizer
gcc -g -fsanitize=thread -pthread -o program program.c

# With AddressSanitizer  
gcc -g -fsanitize=address -pthread -o program program.c

# Optimized build for performance testing
gcc -O2 -pthread -o program program.c
```

## Next Steps

After mastering multi-threaded debugging:
1. **Advanced synchronization primitives** (condition variables, semaphores)
2. **Lock-free programming** and atomic operations
3. **High-performance parallel algorithms**
4. **Memory models** and memory ordering
5. **Scalability analysis** and optimization techniques

---

*Remember: Multi-threaded debugging requires patience and systematic approaches. Use the right tools for each type of problem, and always reproduce issues in controlled environments before applying fixes.*
