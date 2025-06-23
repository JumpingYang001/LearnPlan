# Fundamentals of Threading Concepts

*Duration: 1 week*

## Concepts to Master

### Process vs Thread

#### What is a Process?
A **process** is an independent program in execution that has its own:
- Memory space (code, data, heap, stack)
- Process ID (PID)
- File descriptors
- Security context
- Environment variables

#### What is a Thread?
A **thread** is a lightweight execution unit within a process that shares:
- Memory space (code, data, heap) with other threads in the same process
- File descriptors
- Signal handlers
- Current working directory

But has its own:
- Stack
- Program counter (PC)
- Registers
- Thread ID

#### Key Differences Comparison

| Aspect | Process | Thread |
|--------|---------|---------|
| **Memory** | Isolated address space | Shared address space |
| **Creation Cost** | High (fork() overhead) | Low (pthread_create() overhead) |
| **Context Switch** | Expensive | Cheap |
| **Communication** | IPC (pipes, sockets, shared memory) | Direct memory access |
| **Crash Impact** | Isolated (one crash doesn't affect others) | Shared (one thread crash can kill process) |
| **Resource Usage** | High memory footprint | Low memory footprint |

#### Visual Representation
```
Process A                    Process B
┌─────────────────┐         ┌─────────────────┐
│ Thread 1        │         │ Thread 1        │
│ ┌─────────────┐ │         │ ┌─────────────┐ │
│ │   Stack 1   │ │         │ │   Stack 1   │ │
│ └─────────────┘ │         │ └─────────────┘ │
│ Thread 2        │         │                 │
│ ┌─────────────┐ │         │                 │
│ │   Stack 2   │ │         │                 │
│ └─────────────┘ │         │                 │
│                 │         │                 │
│ Shared Memory:  │         │ Shared Memory:  │
│ ┌─────────────┐ │         │ ┌─────────────┐ │
│ │    Code     │ │         │ │    Code     │ │
│ │    Data     │ │         │ │    Data     │ │
│ │    Heap     │ │         │ │    Heap     │ │
│ └─────────────┘ │         │ └─────────────┘ │
└─────────────────┘         └─────────────────┘
```

#### Performance Implications

**Process Creation Example:**
```c
#include <unistd.h>
#include <sys/wait.h>
#include <stdio.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process
        printf("Child process PID: %d\n", getpid());
    } else if (pid > 0) {
        // Parent process
        printf("Parent process PID: %d, Child PID: %d\n", getpid(), pid);
        wait(NULL); // Wait for child to complete
    }
    
    return 0;
}
```

**Thread Creation Example:**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void* thread_function(void* arg) {
    int thread_num = *(int*)arg;
    printf("Thread %d is running\n", thread_num);
    return NULL;
}

int main() {
    pthread_t thread1, thread2;
    int arg1 = 1, arg2 = 2;
    
    // Create threads (much faster than fork())
    pthread_create(&thread1, NULL, thread_function, &arg1);
    pthread_create(&thread2, NULL, thread_function, &arg2);
    
    // Wait for threads to complete
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    
    return 0;
}
```

### Thread Creation and Termination

#### Thread Lifecycle Management
Understanding the complete lifecycle of a thread is crucial for effective multi-threaded programming:

1. **Creation** → 2. **Initialization** → 3. **Execution** → 4. **Termination** → 5. **Cleanup**

#### Creating Threads Programmatically

**POSIX Threads (pthreads) Example:**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Thread function prototype
void* worker_thread(void* arg);

typedef struct {
    int thread_id;
    char* message;
} thread_data_t;

void* worker_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    
    printf("Thread %d started: %s\n", data->thread_id, data->message);
    
    // Simulate some work
    sleep(2);
    
    printf("Thread %d finished\n", data->thread_id);
    
    // Return some result
    int* result = malloc(sizeof(int));
    *result = data->thread_id * 10;
    return result;
}

int main() {
    const int NUM_THREADS = 3;
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i + 1;
        thread_data[i].message = "Hello from thread";
        
        int result = pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
        if (result != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            exit(1);
        }
    }
    
    // Wait for all threads to complete and collect results
    for (int i = 0; i < NUM_THREADS; i++) {
        void* return_value;
        pthread_join(threads[i], &return_value);
        
        if (return_value != NULL) {
            int* result = (int*)return_value;
            printf("Thread %d returned: %d\n", i + 1, *result);
            free(result); // Clean up allocated memory
        }
    }
    
    printf("All threads completed\n");
    return 0;
}
```

#### Proper Thread Termination Techniques

**Method 1: Natural Termination (Recommended)**
```c
void* thread_function(void* arg) {
    // Do work
    printf("Thread doing work...\n");
    
    // Thread naturally terminates when function returns
    return NULL;
}
```

**Method 2: Explicit Exit**
```c
void* thread_function(void* arg) {
    // Do some work
    if (some_condition) {
        pthread_exit(NULL); // Explicit exit
    }
    
    // More work
    return NULL;
}
```

**Method 3: Cancellation (Use with caution)**
```c
void* cancellable_thread(void* arg) {
    // Enable cancellation
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
    
    while (1) {
        // Check for cancellation point
        pthread_testcancel();
        
        // Do work
        printf("Working...\n");
        sleep(1);
    }
    
    return NULL;
}

// In main thread
pthread_cancel(thread_id); // Request cancellation
```

#### Resource Cleanup Considerations

**Cleanup Handler Example:**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void cleanup_handler(void* arg) {
    printf("Cleanup handler called\n");
    if (arg) {
        free(arg);
        printf("Freed allocated memory\n");
    }
}

void* thread_with_cleanup(void* arg) {
    char* buffer = malloc(1024);
    
    // Install cleanup handler
    pthread_cleanup_push(cleanup_handler, buffer);
    
    // Simulate work that might be cancelled
    for (int i = 0; i < 10; i++) {
        printf("Working... %d\n", i);
        sleep(1);
        pthread_testcancel(); // Cancellation point
    }
    
    // Remove cleanup handler (won't execute if thread completes normally)
    pthread_cleanup_pop(1); // 1 = execute cleanup handler anyway
    
    return NULL;
}
```

#### Best Practices for Thread Management

1. **Always join detached threads** unless you specifically need detached threads
2. **Use proper error checking** for all pthread functions
3. **Clean up resources** in cleanup handlers
4. **Avoid thread cancellation** when possible - use cooperative termination instead
5. **Set appropriate thread attributes** for stack size, scheduling, etc.

**Example with Error Checking:**
```c
int create_thread_safe(pthread_t* thread, void* (*start_routine)(void*), void* arg) {
    int result = pthread_create(thread, NULL, start_routine, arg);
    
    switch (result) {
        case 0:
            printf("Thread created successfully\n");
            break;
        case EAGAIN:
            fprintf(stderr, "Insufficient resources to create thread\n");
            break;
        case EPERM:
            fprintf(stderr, "No permission to create thread\n");
            break;
        default:
            fprintf(stderr, "Unknown error creating thread: %d\n", result);
            break;
    }
    
    return result;
}
```

### Thread States and Lifecycle

Understanding thread states is crucial for debugging and optimizing multi-threaded applications. Here's a detailed breakdown of each state:

#### Thread State Diagram
```
    [NEW] ──────> [RUNNABLE] ──────> [RUNNING]
       │              │                  │
       │              │                  ├──> [BLOCKED]
       │              │                  │        │
       │              │                  │        ├──> [WAITING]
       │              │                  │        │        │
       │              │<─────────────────┼────────┴────────┘
       │                                 │
       └─────────────────────────────────┼──> [TERMINATED]
                                         │
                                    [TIME_WAITING]
```

#### Detailed State Explanations

**1. NEW State**
- Thread object created but `pthread_create()` not yet called
- Thread is not yet scheduled for execution
- No system resources allocated yet

```c
pthread_t thread;  // Thread is in NEW state
// Thread object exists but thread hasn't started
```

**2. RUNNABLE State**
- Thread is ready to run and waiting for CPU time
- Thread has been created and is in the scheduler's ready queue
- May be waiting for the OS scheduler to assign CPU time

```c
pthread_t thread;
pthread_create(&thread, NULL, thread_function, NULL);
// Thread is now RUNNABLE (waiting for CPU)
```

**3. RUNNING State**
- Thread is currently executing on a CPU core
- Thread has the CPU and is actively running instructions
- Can transition to other states based on events

```c
void* thread_function(void* arg) {
    // When this code executes, thread is in RUNNING state
    printf("Thread is running!\n");
    return NULL;
}
```

**4. BLOCKED State**
- Thread is waiting for a resource to become available
- Common blocking scenarios:
  - Waiting for mutex lock
  - Waiting for I/O operation
  - Waiting for memory allocation

```c
void* blocked_thread(void* arg) {
    pthread_mutex_t* mutex = (pthread_mutex_t*)arg;
    
    printf("Trying to acquire lock...\n");
    pthread_mutex_lock(mutex);  // Thread may become BLOCKED here
    
    printf("Lock acquired, doing work...\n");
    // Critical section work
    
    pthread_mutex_unlock(mutex);
    return NULL;
}
```

**5. WAITING State**
- Thread is waiting indefinitely for another thread to perform a specific action
- Examples: waiting for condition variable, joining another thread

```c
void* waiting_thread(void* arg) {
    pthread_cond_t* condition = (pthread_cond_t*)arg;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    
    pthread_mutex_lock(&mutex);
    pthread_cond_wait(condition, &mutex);  // Thread enters WAITING state
    pthread_mutex_unlock(&mutex);
    
    return NULL;
}
```

**6. TIMED_WAITING State**
- Thread is waiting for a specified period of time
- Examples: `sleep()`, `usleep()`, timed mutex operations

```c
void* timed_waiting_thread(void* arg) {
    printf("Thread going to sleep...\n");
    sleep(5);  // Thread enters TIMED_WAITING state for 5 seconds
    printf("Thread woke up!\n");
    return NULL;
}
```

**7. TERMINATED State**
- Thread has completed execution
- Thread function has returned or `pthread_exit()` was called
- Thread resources are being cleaned up

```c
void* terminating_thread(void* arg) {
    printf("Thread about to terminate...\n");
    return NULL;  // Thread enters TERMINATED state
}
```

#### Practical State Monitoring Example

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>

// Function to get thread state (Linux-specific example)
void print_thread_info(const char* stage) {
    pid_t tid = syscall(SYS_gettid);
    printf("[%s] Thread ID: %d, Process ID: %d\n", stage, tid, getpid());
}

void* state_demo_thread(void* arg) {
    print_thread_info("RUNNING - Thread started");
    
    // Simulate different states
    printf("Entering TIMED_WAITING state...\n");
    sleep(2);  // TIMED_WAITING
    
    print_thread_info("RUNNING - After sleep");
    
    // Simulate some work
    for (int i = 0; i < 1000000; i++) {
        // RUNNING state during computation
    }
    
    print_thread_info("RUNNING - Before termination");
    return NULL;  // TERMINATED
}

int main() {
    pthread_t thread;
    
    printf("Main thread creating new thread...\n");
    pthread_create(&thread, NULL, state_demo_thread, NULL);
    // New thread is now RUNNABLE
    
    printf("Main thread waiting for child thread...\n");
    pthread_join(thread, NULL);  // Main thread enters WAITING state
    // Child thread is now TERMINATED
    
    printf("All threads completed\n");
    return 0;
}
```

#### State Transition Triggers

| From State | To State | Trigger |
|------------|----------|---------|
| NEW | RUNNABLE | `pthread_create()` called |
| RUNNABLE | RUNNING | OS scheduler assigns CPU |
| RUNNING | BLOCKED | Waiting for mutex, I/O, etc. |
| RUNNING | WAITING | `pthread_cond_wait()`, `pthread_join()` |
| RUNNING | TIMED_WAITING | `sleep()`, `usleep()`, timed operations |
| RUNNING | TERMINATED | Function returns, `pthread_exit()` |
| BLOCKED | RUNNABLE | Resource becomes available |
| WAITING | RUNNABLE | Condition signaled, thread joined |
| TIMED_WAITING | RUNNABLE | Timeout expires |

#### Debugging Thread States

**Using GDB to inspect thread states:**
```bash
# In GDB
(gdb) info threads        # List all threads
(gdb) thread 2           # Switch to thread 2
(gdb) bt                 # Show backtrace
(gdb) where              # Show current location
```

**Using system tools:**
```bash
# Linux: Check thread states
ps -eLf | grep your_program

# Show thread states in /proc
cat /proc/PID/status | grep State
```

### Benefits and Challenges of Multi-threading

#### Benefits

**1. Improved Performance on Multi-core Systems**

Multi-threading allows your program to utilize multiple CPU cores simultaneously, leading to significant performance improvements for CPU-intensive tasks.

```c
#include <pthread.h>
#include <stdio.h>
#include <time.h>

// Single-threaded approach
void single_threaded_computation() {
    clock_t start = clock();
    
    long long sum = 0;
    for (long long i = 0; i < 1000000000; i++) {
        sum += i;
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Single-threaded result: %lld, Time: %.2f seconds\n", sum, time_taken);
}

// Multi-threaded approach
typedef struct {
    long long start;
    long long end;
    long long result;
} thread_data_t;

void* compute_sum(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    data->result = 0;
    
    for (long long i = data->start; i < data->end; i++) {
        data->result += i;
    }
    
    return NULL;
}

void multi_threaded_computation() {
    clock_t start = clock();
    
    const int NUM_THREADS = 4;
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    
    long long total_range = 1000000000;
    long long range_per_thread = total_range / NUM_THREADS;
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].start = i * range_per_thread;
        thread_data[i].end = (i + 1) * range_per_thread;
        pthread_create(&threads[i], NULL, compute_sum, &thread_data[i]);
    }
    
    // Wait for threads and collect results
    long long total_sum = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total_sum += thread_data[i].result;
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Multi-threaded result: %lld, Time: %.2f seconds\n", total_sum, time_taken);
}
```

**2. Better Resource Utilization**

While one thread waits for I/O operations, other threads can continue working, maximizing CPU utilization.

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

void* io_worker(void* arg) {
    int worker_id = *(int*)arg;
    
    printf("Worker %d: Starting I/O operation...\n", worker_id);
    sleep(2);  // Simulate I/O wait
    printf("Worker %d: I/O completed, processing data...\n", worker_id);
    
    // Simulate CPU work
    for (int i = 0; i < 100000000; i++) {
        // Some computation
    }
    
    printf("Worker %d: Processing completed\n", worker_id);
    return NULL;
}

// Without threading: Total time = 3 * (I/O time + CPU time)
// With threading: Total time ≈ max(I/O time + CPU time) (parallel execution)
```

**3. Enhanced Program Responsiveness**

GUI applications remain responsive while background tasks execute.

```c
// Pseudo-code for GUI responsiveness
void* background_task(void* arg) {
    // Long-running operation (file processing, network request, etc.)
    process_large_file();
    
    // Update GUI when done (thread-safe)
    post_gui_update("Task completed!");
    return NULL;
}

void on_button_click() {
    pthread_t background_thread;
    pthread_create(&background_thread, NULL, background_task, NULL);
    
    // GUI remains responsive - user can still interact
    // Background work continues in parallel
}
```

**4. Concurrent Task Execution**

Multiple independent tasks can execute simultaneously.

```c
#include <pthread.h>
#include <stdio.h>

void* download_file(void* url) {
    printf("Downloading from %s...\n", (char*)url);
    sleep(3);  // Simulate download
    printf("Downloaded %s\n", (char*)url);
    return NULL;
}

int main() {
    char* urls[] = {"http://site1.com", "http://site2.com", "http://site3.com"};
    pthread_t threads[3];
    
    // Start all downloads concurrently instead of sequentially
    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, download_file, urls[i]);
    }
    
    // Wait for all downloads to complete
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }
    
    return 0;
}
```

#### Challenges

**1. Race Conditions and Data Races**

Multiple threads accessing shared data simultaneously can lead to unpredictable results.

```c
#include <pthread.h>
#include <stdio.h>

int shared_counter = 0;  // Shared resource

// PROBLEMATIC: Race condition example
void* increment_counter_unsafe(void* arg) {
    for (int i = 0; i < 100000; i++) {
        shared_counter++;  // NOT ATOMIC - Race condition!
        // Assembly: LOAD counter, ADD 1, STORE counter
        // Another thread can interrupt between these operations
    }
    return NULL;
}

// SOLUTION: Using mutex for thread-safe access
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment_counter_safe(void* arg) {
    for (int i = 0; i < 100000; i++) {
        pthread_mutex_lock(&counter_mutex);
        shared_counter++;  // Now atomic with respect to other threads
        pthread_mutex_unlock(&counter_mutex);
    }
    return NULL;
}

void demonstrate_race_condition() {
    pthread_t threads[2];
    
    // Reset counter
    shared_counter = 0;
    
    // Unsafe version - race condition
    pthread_create(&threads[0], NULL, increment_counter_unsafe, NULL);
    pthread_create(&threads[1], NULL, increment_counter_unsafe, NULL);
    
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
    
    printf("Unsafe result: %d (should be 200000)\n", shared_counter);
    
    // Reset counter
    shared_counter = 0;
    
    // Safe version - with mutex
    pthread_create(&threads[0], NULL, increment_counter_safe, NULL);
    pthread_create(&threads[1], NULL, increment_counter_safe, NULL);
    
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
    
    printf("Safe result: %d (should be 200000)\n", shared_counter);
}
```

**2. Deadlocks and Livelocks**

Threads can get stuck waiting for each other to release resources.

```c
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

// DEADLOCK EXAMPLE
void* thread1_function(void* arg) {
    pthread_mutex_lock(&mutex1);
    printf("Thread 1: Acquired mutex1\n");
    
    sleep(1);  // Simulate work
    
    printf("Thread 1: Trying to acquire mutex2...\n");
    pthread_mutex_lock(&mutex2);  // Will block if thread2 has mutex2
    printf("Thread 1: Acquired mutex2\n");
    
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

void* thread2_function(void* arg) {
    pthread_mutex_lock(&mutex2);
    printf("Thread 2: Acquired mutex2\n");
    
    sleep(1);  // Simulate work
    
    printf("Thread 2: Trying to acquire mutex1...\n");
    pthread_mutex_lock(&mutex1);  // Will block if thread1 has mutex1
    printf("Thread 2: Acquired mutex1\n");
    
    pthread_mutex_unlock(&mutex1);
    pthread_mutex_unlock(&mutex2);
    return NULL;
}

// DEADLOCK PREVENTION: Always acquire locks in the same order
void* thread1_safe(void* arg) {
    pthread_mutex_lock(&mutex1);  // Always acquire mutex1 first
    pthread_mutex_lock(&mutex2);  // Then mutex2
    
    // Do work
    
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

void* thread2_safe(void* arg) {
    pthread_mutex_lock(&mutex1);  // Same order: mutex1 first
    pthread_mutex_lock(&mutex2);  // Then mutex2
    
    // Do work
    
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}
```

**3. Synchronization Complexity**

Coordinating multiple threads requires careful design and can be error-prone.

```c
// Producer-Consumer problem - complex synchronization
#include <pthread.h>
#include <semaphore.h>

#define BUFFER_SIZE 10

int buffer[BUFFER_SIZE];
int in = 0, out = 0;

sem_t empty_slots;  // Counts empty slots
sem_t full_slots;   // Counts full slots
pthread_mutex_t buffer_mutex = PTHREAD_MUTEX_INITIALIZER;

void* producer(void* arg) {
    for (int i = 0; i < 20; i++) {
        int item = i;
        
        sem_wait(&empty_slots);           // Wait for empty slot
        pthread_mutex_lock(&buffer_mutex); // Acquire buffer access
        
        buffer[in] = item;
        in = (in + 1) % BUFFER_SIZE;
        printf("Produced: %d\n", item);
        
        pthread_mutex_unlock(&buffer_mutex);
        sem_post(&full_slots);            // Signal full slot available
    }
    return NULL;
}

void* consumer(void* arg) {
    for (int i = 0; i < 20; i++) {
        sem_wait(&full_slots);            // Wait for full slot
        pthread_mutex_lock(&buffer_mutex); // Acquire buffer access
        
        int item = buffer[out];
        out = (out + 1) % BUFFER_SIZE;
        printf("Consumed: %d\n", item);
        
        pthread_mutex_unlock(&buffer_mutex);
        sem_post(&empty_slots);           // Signal empty slot available
    }
    return NULL;
}
```

**4. Debugging Difficulties**

Multi-threaded bugs can be non-deterministic and hard to reproduce.

```c
// Debugging tips for multi-threaded programs:

// 1. Use thread-safe debugging output
pthread_mutex_t debug_mutex = PTHREAD_MUTEX_INITIALIZER;

void thread_safe_printf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    pthread_mutex_lock(&debug_mutex);
    vprintf(format, args);
    fflush(stdout);
    pthread_mutex_unlock(&debug_mutex);
    
    va_end(args);
}

// 2. Add thread identification to debug output
void* debug_thread(void* arg) {
    pthread_t thread_id = pthread_self();
    
    thread_safe_printf("Thread %lu: Starting work\n", thread_id);
    
    // Your thread work here
    
    thread_safe_printf("Thread %lu: Finished work\n", thread_id);
    return NULL;
}

// 3. Use tools like Valgrind's Helgrind for race condition detection
// Compile with: gcc -g -pthread program.c
// Run with: valgrind --tool=helgrind ./program
```

#### Best Practices Summary

✅ **DO:**
- Use mutexes to protect shared data
- Acquire locks in consistent order to prevent deadlocks
- Use condition variables for complex synchronization
- Keep critical sections small
- Use atomic operations for simple counters
- Test with stress testing and race condition detectors

❌ **DON'T:**
- Access shared data without synchronization
- Hold locks longer than necessary
- Create too many threads (overhead)
- Ignore return values from pthread functions
- Use busy-waiting instead of proper synchronization

## Learning Objectives

By the end of this section, you should be able to:
- **Explain the fundamental differences** between processes and threads with concrete examples
- **Create and manage threads** using POSIX threads (pthreads) API
- **Identify and handle different thread states** during program execution
- **Recognize when to use threading** vs single-threaded approaches based on performance requirements
- **Understand the basic challenges** in multi-threaded programming and apply basic solutions
- **Implement proper thread synchronization** using mutexes and condition variables
- **Debug multi-threaded programs** using appropriate tools and techniques

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can:

□ Write a simple multi-threaded program that creates and joins threads  
□ Explain what happens during a context switch between threads  
□ Identify potential race conditions in code  
□ Implement thread-safe access to shared variables  
□ Describe the pros and cons of using threads vs processes for a given scenario  
□ Use debugging tools to inspect thread states  
□ Handle thread creation and termination errors properly  

### Practical Exercises

**Exercise 1: Basic Thread Creation**
```c
// TODO: Complete this program to create 5 threads that print their ID
#include <pthread.h>
#include <stdio.h>

void* print_thread_id(void* arg) {
    // Your code here
}

int main() {
    // Your code here
    return 0;
}
```

**Exercise 2: Race Condition Detection**
```c
// TODO: Identify and fix the race condition in this code
int global_counter = 0;

void* increment_counter(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        global_counter++;  // Race condition here!
    }
    return NULL;
}
```

**Exercise 3: Producer-Consumer Implementation**
```c
// TODO: Implement a simple producer-consumer using threads
// Producer generates numbers 1-100
// Consumer processes and prints them
// Use proper synchronization
```

## Study Materials

### Recommended Reading
- **Primary:** Chapter 1-2 of "Programming with POSIX Threads" by David R. Butenhof
- **Alternative:** "The Linux Programming Interface" - Chapters 29-30 (Thread concepts)
- **Online:** [POSIX Threads Programming](https://computing.llnl.gov/tutorials/pthreads/) - LLNL Tutorial
- **Reference:** `man pthread_create`, `man pthread_join`, `man pthread_mutex_lock`

### Video Resources
- "Introduction to Threads" - MIT OpenCourseWare
- "Concurrent Programming" - Carnegie Mellon University lectures
- "Linux System Programming" - Threading fundamentals

### Hands-on Labs
- **Lab 1:** Create a multi-threaded file processor
- **Lab 2:** Implement thread-safe data structures
- **Lab 3:** Build a simple thread pool

### Practice Questions

**Conceptual Questions:**
1. What are the main differences between a process and a thread? Give 3 specific examples.
2. When would you choose multi-threading over multi-processing? Provide a real-world scenario.
3. What resources are shared between threads in the same process? What resources are private?
4. What are the potential risks of concurrent execution? How can they be mitigated?
5. Explain the difference between race conditions and deadlocks with examples.

**Technical Questions:**
6. What happens if you don't call `pthread_join()` for a thread?
7. How does the OS scheduler decide which thread to run next?
8. What's the difference between `pthread_exit()` and returning from the thread function?
9. Why is `printf()` potentially unsafe in multi-threaded programs?
10. How can you detect race conditions in your code?

**Coding Challenges:**
```c
// Challenge 1: Fix the race condition
int shared_data = 0;
void* worker(void* arg) {
    for (int i = 0; i < 10000; i++) {
        shared_data = shared_data + 1;  // Fix this
    }
    return NULL;
}

// Challenge 2: Prevent the deadlock
pthread_mutex_t lock1, lock2;
void* thread1(void* arg) {
    pthread_mutex_lock(&lock1);
    // ... work ...
    pthread_mutex_lock(&lock2);  // Potential deadlock
    // ... work ...
    pthread_mutex_unlock(&lock2);
    pthread_mutex_unlock(&lock1);
    return NULL;
}

// Challenge 3: Implement thread-safe queue
typedef struct {
    int* data;
    int front, rear, size, capacity;
    pthread_mutex_t mutex;
} ThreadSafeQueue;

// Implement: enqueue(), dequeue(), isEmpty(), isFull()
```

### Development Environment Setup

**Required Tools:**
```bash
# Install development tools
sudo apt-get install build-essential
sudo apt-get install gdb
sudo apt-get install valgrind

# For thread debugging
sudo apt-get install helgrind
```

**Compilation Commands:**
```bash
# Basic compilation
gcc -pthread -o program program.c

# Debug build
gcc -g -pthread -Wall -Wextra -o program program.c

# With sanitizers
gcc -g -pthread -fsanitize=thread -o program program.c
```

**Debugging Commands:**
```bash
# Run with thread sanitizer
./program

# Use GDB for debugging
gdb ./program
(gdb) set scheduler-locking on  # Control thread execution
(gdb) info threads             # List threads
(gdb) thread 2                 # Switch to thread 2

# Use Helgrind for race detection
valgrind --tool=helgrind ./program
```

## Next Section
[POSIX Threads (pthreads)](02_POSIX_Threads.md)
