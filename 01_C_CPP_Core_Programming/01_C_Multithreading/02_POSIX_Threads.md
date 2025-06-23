# POSIX Threads (pthreads)

*Duration: 2 weeks*

## Overview

POSIX Threads (pthreads) is the POSIX standard for threads, providing a comprehensive API for creating and managing threads in Unix-like systems including Linux, macOS, and other POSIX-compliant systems. This section provides an in-depth coverage of the pthread API, from basic thread creation to advanced thread management techniques.

### What is POSIX Threads?

POSIX Threads, commonly known as **pthreads**, is a standardized C language threads programming interface specified by the IEEE POSIX.1c standard (Threads extensions). It defines a set of C programming language types, functions, and constants for:

- Thread creation and management
- Mutual exclusion (mutexes)
- Condition variables
- Thread-specific data
- Thread synchronization primitives

### Why Use pthreads?

1. **Portability**: Code works across different Unix-like systems
2. **Performance**: Low-level control over thread behavior
3. **Flexibility**: Rich API with fine-grained control
4. **Industry Standard**: Widely adopted and well-documented
5. **Integration**: Works seamlessly with C/C++ applications

### pthread vs Other Threading Libraries

| Feature | pthreads | std::thread (C++11) | OpenMP | Windows Threads |
|---------|----------|-------------------|---------|-----------------|
| **Language** | C/C++ | C++ | C/C++/Fortran | C/C++ |
| **Portability** | Unix/Linux/POSIX | Cross-platform | Cross-platform | Windows only |
| **Learning Curve** | Moderate | Easy | Easy | Moderate |
| **Performance** | High | High | High | High |
| **Control Level** | Low-level | Mid-level | High-level | Low-level |

## Core pthread Functions

### Thread Creation with `pthread_create()`

The `pthread_create()` function is the foundation of pthread programming, used to create new threads.

```c
#include <pthread.h>

int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine) (void *), void *arg);
```

**Parameters Explained:**
- `thread`: Pointer to `pthread_t` variable where the thread ID will be stored
- `attr`: Thread attributes (use `NULL` for default attributes)
- `start_routine`: Function pointer to the thread's entry point
- `arg`: Single argument passed to the thread function (use struct for multiple args)

**Return Value:**
- `0` on success
- Error code on failure (e.g., `EAGAIN`, `EINVAL`, `EPERM`)

#### Detailed Example with Error Handling

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

// Thread function signature must match: void* func(void* arg)
void* worker_thread(void* arg) {
    int worker_id = *(int*)arg;
    printf("Worker %d: Starting execution\n", worker_id);
    
    // Simulate work
    for (int i = 0; i < 5; i++) {
        printf("Worker %d: Step %d\n", worker_id, i + 1);
        sleep(1);
    }
    
    printf("Worker %d: Completed\n", worker_id);
    return NULL;
}

int main() {
    pthread_t thread;
    int worker_id = 42;
    int result;
    
    printf("Main thread: Creating worker thread\n");
    
    result = pthread_create(&thread, NULL, worker_thread, &worker_id);
    if (result != 0) {
        fprintf(stderr, "pthread_create failed: %s\n", strerror(result));
        exit(EXIT_FAILURE);
    }
    
    printf("Main thread: Worker thread created successfully\n");
    
    // Wait for thread to complete
    pthread_join(thread, NULL);
    
    printf("Main thread: Worker thread joined\n");
    return 0;
}
```

#### Advanced Thread Creation Patterns

**Pattern 1: Multiple Threads with Unique Arguments**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int thread_id;
    char* task_name;
    int iterations;
} thread_args_t;

void* task_worker(void* arg) {
    thread_args_t* args = (thread_args_t*)arg;
    
    printf("Thread %d: Starting task '%s' with %d iterations\n", 
           args->thread_id, args->task_name, args->iterations);
    
    for (int i = 0; i < args->iterations; i++) {
        printf("Thread %d: %s - iteration %d\n", 
               args->thread_id, args->task_name, i + 1);
        usleep(100000); // 100ms delay
    }
    
    return NULL;
}

int main() {
    const int NUM_THREADS = 3;
    pthread_t threads[NUM_THREADS];
    thread_args_t args[NUM_THREADS];
    
    // Initialize thread arguments
    char* tasks[] = {"Processing", "Computing", "Analyzing"};
    int iterations[] = {3, 5, 4};
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i + 1;
        args[i].task_name = tasks[i];
        args[i].iterations = iterations[i];
        
        if (pthread_create(&threads[i], NULL, task_worker, &args[i]) != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    
    // Wait for all threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    return 0;
}
```

**Pattern 2: Thread Factory Function**
```c
// Thread factory for creating threads with common patterns
typedef struct {
    pthread_t thread_id;
    int (*work_function)(void* data);
    void* work_data;
    int status;
} managed_thread_t;

void* thread_wrapper(void* arg) {
    managed_thread_t* thread_info = (managed_thread_t*)arg;
    
    printf("Thread starting work function\n");
    thread_info->status = thread_info->work_function(thread_info->work_data);
    printf("Thread completed with status: %d\n", thread_info->status);
    
    return NULL;
}

int create_managed_thread(managed_thread_t* thread_info, 
                         int (*work_func)(void*), void* data) {
    thread_info->work_function = work_func;
    thread_info->work_data = data;
    thread_info->status = -1;
    
    return pthread_create(&thread_info->thread_id, NULL, thread_wrapper, thread_info);
}
```

### Thread Joining with `pthread_join()`

The `pthread_join()` function waits for a thread to terminate and optionally retrieves its return value.

```c
int pthread_join(pthread_t thread, void **retval);
```

**Parameters:**
- `thread`: Thread ID to wait for
- `retval`: Pointer to store the thread's return value (can be `NULL`)

**Important Notes:**
- Calling thread blocks until specified thread terminates
- Can only join threads created in joinable state (default)
- Each thread can only be joined once
- Calling `pthread_join()` on already joined or detached thread is undefined behavior

#### Comprehensive Joining Examples

**Example 1: Collecting Return Values**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int start;
    int end;
    long long sum;
} sum_task_t;

void* calculate_sum(void* arg) {
    sum_task_t* task = (sum_task_t*)arg;
    task->sum = 0;
    
    printf("Thread calculating sum from %d to %d\n", task->start, task->end);
    
    for (int i = task->start; i <= task->end; i++) {
        task->sum += i;
    }
    
    printf("Thread completed: sum = %lld\n", task->sum);
    return task; // Return the task structure
}

int main() {
    const int NUM_THREADS = 4;
    pthread_t threads[NUM_THREADS];
    sum_task_t tasks[NUM_THREADS];
    
    int range_size = 1000000 / NUM_THREADS;
    
    // Create threads with different ranges
    for (int i = 0; i < NUM_THREADS; i++) {
        tasks[i].start = i * range_size + 1;
        tasks[i].end = (i + 1) * range_size;
        
        if (i == NUM_THREADS - 1) {
            tasks[i].end = 1000000; // Handle remainder
        }
        
        pthread_create(&threads[i], NULL, calculate_sum, &tasks[i]);
    }
    
    // Join threads and collect results
    long long total_sum = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        sum_task_t* result;
        pthread_join(threads[i], (void**)&result);
        
        printf("Thread %d result: %lld\n", i, result->sum);
        total_sum += result->sum;
    }
    
    printf("Total sum: %lld\n", total_sum);
    printf("Expected: %lld\n", (1000000LL * 1000001LL) / 2);
    
    return 0;
}
```

**Example 2: Dynamic Memory Return Values**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* result_message;
    int result_code;
    double computation_time;
} thread_result_t;

void* complex_computation(void* arg) {
    int task_id = *(int*)arg;
    
    // Allocate result structure
    thread_result_t* result = malloc(sizeof(thread_result_t));
    result->result_message = malloc(256);
    
    clock_t start_time = clock();
    
    // Simulate complex computation
    long long computation = 0;
    for (int i = 0; i < 1000000; i++) {
        computation += i * task_id;
    }
    
    clock_t end_time = clock();
    result->computation_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    snprintf(result->result_message, 256, 
             "Task %d completed: result=%lld", task_id, computation);
    result->result_code = 0; // Success
    
    return result;
}

int main() {
    pthread_t threads[3];
    int task_ids[3] = {1, 2, 3};
    
    // Create threads
    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, complex_computation, &task_ids[i]);
    }
    
    // Join and process results
    for (int i = 0; i < 3; i++) {
        thread_result_t* result;
        pthread_join(threads[i], (void**)&result);
        
        printf("Thread %d: %s\n", i, result->result_message);
        printf("  Execution time: %.3f seconds\n", result->computation_time);
        printf("  Result code: %d\n", result->result_code);
        
        // Clean up allocated memory
        free(result->result_message);
        free(result);
    }
    
    return 0;
}
```

### Thread Detachment with `pthread_detach()`

Detached threads automatically clean up their resources when they terminate, without requiring `pthread_join()`.

```c
int pthread_detach(pthread_t thread);
```

**When to Use Detached Threads:**
- Fire-and-forget operations
- Background tasks that don't need to return results
- Server applications with many short-lived threads
- When you don't need to wait for thread completion

#### Detached Thread Examples

**Example 1: Background Logger Thread**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

void* background_logger(void* arg) {
    char* log_message = (char*)arg;
    FILE* log_file = fopen("app.log", "a");
    
    if (log_file == NULL) {
        perror("Failed to open log file");
        return NULL;
    }
    
    // Get current time
    time_t now;
    time(&now);
    char* time_str = ctime(&now);
    time_str[strlen(time_str) - 1] = '\0'; // Remove newline
    
    fprintf(log_file, "[%s] %s\n", time_str, log_message);
    fclose(log_file);
    
    printf("Background logger: Message logged\n");
    free(log_message); // Clean up passed argument
    
    return NULL;
}

void log_message_async(const char* message) {
    pthread_t logger_thread;
    char* message_copy = strdup(message); // Create copy for thread
    
    // Create detached thread
    pthread_create(&logger_thread, NULL, background_logger, message_copy);
    pthread_detach(logger_thread);
    
    // Thread will clean up automatically when it finishes
}

int main() {
    printf("Main program starting\n");
    
    log_message_async("Application started");
    log_message_async("Processing user request");
    log_message_async("Database connection established");
    
    // Continue main program work
    printf("Main program doing other work...\n");
    sleep(2); // Give detached threads time to complete
    
    log_message_async("Application shutting down");
    sleep(1); // Give final log time to complete
    
    return 0;
}
```

**Example 2: Detached vs Joinable Comparison**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void* worker_function(void* arg) {
    int worker_id = *(int*)arg;
    printf("Worker %d: Starting\n", worker_id);
    
    sleep(2); // Simulate work
    
    printf("Worker %d: Finished\n", worker_id);
    return NULL;
}

int main() {
    pthread_t joinable_thread, detached_thread;
    int id1 = 1, id2 = 2;
    
    printf("=== Joinable Thread Example ===\n");
    
    // Create joinable thread (default)
    pthread_create(&joinable_thread, NULL, worker_function, &id1);
    
    printf("Main: Created joinable thread, waiting for completion\n");
    pthread_join(joinable_thread, NULL);
    printf("Main: Joinable thread completed\n\n");
    
    printf("=== Detached Thread Example ===\n");
    
    // Create and immediately detach thread
    pthread_create(&detached_thread, NULL, worker_function, &id2);
    pthread_detach(detached_thread);
    
    printf("Main: Created detached thread, continuing without waiting\n");
    printf("Main: Doing other work while detached thread runs\n");
    
    // Give detached thread time to complete
    sleep(3);
    
    printf("Main: Program ending (detached thread cleaned up automatically)\n");
    return 0;
}
```

### Thread Identification

Understanding thread identity is crucial for debugging and thread-specific operations.

```c
pthread_t pthread_self(void);
int pthread_equal(pthread_t t1, pthread_t t2);
```

#### Thread Identification Examples

**Example 1: Thread Self-Identification**
```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

void* identified_worker(void* arg) {
    pthread_t my_id = pthread_self();
    int task_number = *(int*)arg;
    
    printf("Thread %lu: Starting task %d\n", (unsigned long)my_id, task_number);
    
    // Simulate work
    sleep(task_number);
    
    printf("Thread %lu: Completed task %d\n", (unsigned long)my_id, task_number);
    return NULL;
}

int main() {
    pthread_t main_thread = pthread_self();
    printf("Main thread ID: %lu\n", (unsigned long)main_thread);
    
    pthread_t workers[3];
    int task_numbers[3] = {1, 2, 3};
    
    // Create worker threads
    for (int i = 0; i < 3; i++) {
        pthread_create(&workers[i], NULL, identified_worker, &task_numbers[i]);
        printf("Main: Created worker thread %lu\n", (unsigned long)workers[i]);
    }
    
    // Wait for all workers
    for (int i = 0; i < 3; i++) {
        pthread_join(workers[i], NULL);
    }
    
    printf("Main thread %lu: All workers completed\n", (unsigned long)main_thread);
    return 0;
}
```

**Example 2: Thread Comparison and Management**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    pthread_t thread_id;
    int status;
    char name[32];
} thread_info_t;

void* managed_worker(void* arg) {
    thread_info_t* info = (thread_info_t*)arg;
    
    // Verify thread ID matches
    if (pthread_equal(pthread_self(), info->thread_id)) {
        printf("Thread %s: ID verification successful\n", info->name);
    } else {
        printf("Thread %s: ID verification FAILED\n", info->name);
    }
    
    info->status = 1; // Mark as running
    
    // Simulate work
    for (int i = 0; i < 5; i++) {
        printf("Thread %s: Working step %d\n", info->name, i + 1);
        usleep(500000); // 500ms
    }
    
    info->status = 2; // Mark as completed
    return NULL;
}

int main() {
    const int NUM_THREADS = 3;
    thread_info_t threads[NUM_THREADS];
    
    // Initialize thread info
    char names[][32] = {"Alpha", "Beta", "Gamma"};
    for (int i = 0; i < NUM_THREADS; i++) {
        strcpy(threads[i].name, names[i]);
        threads[i].status = 0; // Not started
    }
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i].thread_id, NULL, managed_worker, &threads[i]);
        printf("Main: Created thread %s with ID %lu\n", 
               threads[i].name, (unsigned long)threads[i].thread_id);
    }
    
    // Monitor thread status
    printf("\nMain: Monitoring thread status...\n");
    int all_completed = 0;
    while (!all_completed) {
        all_completed = 1;
        for (int i = 0; i < NUM_THREADS; i++) {
            printf("Thread %s: Status %d\n", threads[i].name, threads[i].status);
            if (threads[i].status < 2) {
                all_completed = 0;
            }
        }
        if (!all_completed) {
            printf("---\n");
            sleep(1);
        }
    }
    
    // Join all threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i].thread_id, NULL);
    }
    
    printf("Main: All threads completed\n");
    return 0;
}
```

### Thread Attributes

Thread attributes allow fine-grained control over thread behavior and resource usage. Understanding attributes is crucial for optimizing thread performance and resource management.

```c
// Core attribute functions
int pthread_attr_init(pthread_attr_t *attr);
int pthread_attr_destroy(pthread_attr_t *attr);
int pthread_attr_setdetachstate(pthread_attr_t *attr, int detachstate);
int pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize);
int pthread_attr_setguardsize(pthread_attr_t *attr, size_t guardsize);
int pthread_attr_setschedpolicy(pthread_attr_t *attr, int policy);
int pthread_attr_setschedparam(pthread_attr_t *attr, const struct sched_param *param);
```

#### Common Thread Attributes

| Attribute | Description | Default Value | Use Cases |
|-----------|-------------|---------------|-----------|
| **Detach State** | Joinable vs Detached | `PTHREAD_CREATE_JOINABLE` | Background tasks vs Result collection |
| **Stack Size** | Thread stack size | System-dependent (usually 8MB) | Memory optimization, deep recursion |
| **Guard Size** | Stack overflow protection | System page size | Debugging, safety-critical apps |
| **Scheduling Policy** | Thread scheduling behavior | `SCHED_OTHER` | Real-time applications |
| **Priority** | Thread execution priority | 0 | Performance tuning |
| **Scope** | Contention scope | `PTHREAD_SCOPE_SYSTEM` | Thread competition level |

#### Comprehensive Attribute Examples

**Example 1: Custom Stack Size**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Recursive function to test stack usage
int recursive_function(int depth, int max_depth) {
    char large_array[1024]; // 1KB local variable
    
    if (depth >= max_depth) {
        printf("Thread reached maximum recursion depth: %d\n", depth);
        return depth;
    }
    
    // Initialize array to prevent optimization
    for (int i = 0; i < 1024; i++) {
        large_array[i] = (char)(depth % 256);
    }
    
    return recursive_function(depth + 1, max_depth);
}

void* stack_test_thread(void* arg) {
    int max_depth = *(int*)arg;
    pthread_attr_t current_attr;
    size_t stack_size;
    
    // Get current thread attributes
    pthread_getattr_np(pthread_self(), &current_attr);
    pthread_attr_getstacksize(&current_attr, &stack_size);
    
    printf("Thread: Current stack size: %zu bytes (%.2f MB)\n", 
           stack_size, (double)stack_size / (1024 * 1024));
    
    // Test recursion with current stack
    int result = recursive_function(0, max_depth);
    printf("Thread: Successfully completed %d recursive calls\n", result);
    
    pthread_attr_destroy(&current_attr);
    return NULL;
}

int main() {
    pthread_t threads[3];
    pthread_attr_t attr1, attr2, attr3;
    int test_depth = 1000;
    
    // Test 1: Default stack size
    printf("=== Test 1: Default Stack Size ===\n");
    pthread_create(&threads[0], NULL, stack_test_thread, &test_depth);
    pthread_join(threads[0], NULL);
    
    // Test 2: Small stack size (1MB)
    printf("\n=== Test 2: Small Stack Size (1MB) ===\n");
    pthread_attr_init(&attr1);
    pthread_attr_setstacksize(&attr1, 1024 * 1024); // 1MB
    pthread_create(&threads[1], &attr1, stack_test_thread, &test_depth);
    pthread_join(threads[1], NULL);
    pthread_attr_destroy(&attr1);
    
    // Test 3: Large stack size (16MB)
    printf("\n=== Test 3: Large Stack Size (16MB) ===\n");
    pthread_attr_init(&attr2);
    pthread_attr_setstacksize(&attr2, 16 * 1024 * 1024); // 16MB
    pthread_create(&threads[2], &attr2, stack_test_thread, &test_depth);
    pthread_join(threads[2], NULL);
    pthread_attr_destroy(&attr2);
    
    return 0;
}
```

**Example 2: Detached Thread with Custom Attributes**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    int worker_id;
    int work_duration;
    char task_name[64];
} background_task_t;

void* background_worker(void* arg) {
    background_task_t* task = (background_task_t*)arg;
    
    printf("Background Worker %d: Starting task '%s'\n", 
           task->worker_id, task->task_name);
    
    // Simulate work
    for (int i = 0; i < task->work_duration; i++) {
        printf("Worker %d: Progress %d/%d\n", 
               task->worker_id, i + 1, task->work_duration);
        sleep(1);
    }
    
    printf("Background Worker %d: Task '%s' completed\n", 
           task->worker_id, task->task_name);
    
    // Clean up task data
    free(task);
    return NULL;
}

pthread_t create_background_task(int worker_id, const char* task_name, 
                                int duration, size_t stack_size) {
    pthread_t thread;
    pthread_attr_t attr;
    background_task_t* task;
    
    // Allocate task data
    task = malloc(sizeof(background_task_t));
    task->worker_id = worker_id;
    task->work_duration = duration;
    strncpy(task->task_name, task_name, sizeof(task->task_name) - 1);
    task->task_name[sizeof(task->task_name) - 1] = '\0';
    
    // Initialize attributes
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    
    if (stack_size > 0) {
        pthread_attr_setstacksize(&attr, stack_size);
    }
    
    // Create detached thread
    int result = pthread_create(&thread, &attr, background_worker, task);
    if (result != 0) {
        fprintf(stderr, "Failed to create background task: %s\n", strerror(result));
        free(task);
        thread = 0;
    }
    
    pthread_attr_destroy(&attr);
    return thread;
}

int main() {
    printf("Main: Starting background tasks\n");
    
    // Create various background tasks with different configurations
    create_background_task(1, "Database Cleanup", 3, 0); // Default stack
    create_background_task(2, "Log Processing", 5, 512 * 1024); // 512KB stack
    create_background_task(3, "Cache Refresh", 4, 2 * 1024 * 1024); // 2MB stack
    
    printf("Main: All background tasks started\n");
    printf("Main: Continuing with other work...\n");
    
    // Simulate main program work
    for (int i = 0; i < 8; i++) {
        printf("Main: Main work step %d\n", i + 1);
        sleep(1);
    }
    
    printf("Main: Program completed (background tasks may still be running)\n");
    return 0;
}
```

**Example 3: Real-time Thread Attributes**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <sys/time.h>

void* realtime_worker(void* arg) {
    int worker_id = *(int*)arg;
    struct sched_param param;
    int policy;
    
    // Get current scheduling parameters
    pthread_getschedparam(pthread_self(), &policy, &param);
    
    printf("RT Worker %d: Policy=%s, Priority=%d\n", 
           worker_id,
           (policy == SCHED_FIFO) ? "FIFO" : 
           (policy == SCHED_RR) ? "Round Robin" : "Other",
           param.sched_priority);
    
    // Perform time-critical work
    for (int i = 0; i < 10; i++) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        
        // Simulate critical work (busy wait)
        volatile long count = 0;
        for (long j = 0; j < 1000000; j++) {
            count++;
        }
        
        gettimeofday(&end, NULL);
        long duration = (end.tv_sec - start.tv_sec) * 1000000 + 
                       (end.tv_usec - start.tv_usec);
        
        printf("RT Worker %d: Iteration %d completed in %ld microseconds\n", 
               worker_id, i + 1, duration);
        
        usleep(100000); // 100ms between iterations
    }
    
    return NULL;
}

int main() {
    // Note: Real-time scheduling typically requires root privileges
    pthread_t rt_thread, normal_thread;
    pthread_attr_t rt_attr;
    struct sched_param param;
    int worker1 = 1, worker2 = 2;
    
    printf("=== Normal Priority Thread ===\n");
    pthread_create(&normal_thread, NULL, realtime_worker, &worker1);
    
    printf("\n=== High Priority Real-time Thread ===\n");
    
    // Initialize real-time attributes
    pthread_attr_init(&rt_attr);
    pthread_attr_setschedpolicy(&rt_attr, SCHED_FIFO);
    
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_attr_setschedparam(&rt_attr, &param);
    
    // Try to create real-time thread
    int result = pthread_create(&rt_thread, &rt_attr, realtime_worker, &worker2);
    if (result == EPERM) {
        printf("Note: Real-time scheduling requires root privileges\n");
        printf("Creating thread with normal priority instead\n");
        pthread_create(&rt_thread, NULL, realtime_worker, &worker2);
    } else if (result != 0) {
        fprintf(stderr, "Failed to create real-time thread: %s\n", strerror(result));
        exit(EXIT_FAILURE);
    }
    
    // Wait for both threads
    pthread_join(normal_thread, NULL);
    pthread_join(rt_thread, NULL);
    
    pthread_attr_destroy(&rt_attr);
    
    printf("Main: Both threads completed\n");
    return 0;
}
```

#### Attribute Best Practices

**✅ Good Practices:**
```c
// Always initialize and destroy attributes
pthread_attr_t attr;
pthread_attr_init(&attr);
// ... use attributes ...
pthread_attr_destroy(&attr);

// Check attribute setting return values
if (pthread_attr_setstacksize(&attr, new_size) != 0) {
    fprintf(stderr, "Failed to set stack size\n");
}

// Use appropriate stack sizes
// Small for simple tasks: 64KB - 512KB  
// Default for normal tasks: system default
// Large for deep recursion: 4MB - 16MB
```

**❌ Common Mistakes:**
```c
// Don't forget to destroy attributes
pthread_attr_t attr;
pthread_attr_init(&attr);
// Missing: pthread_attr_destroy(&attr);

// Don't set unreasonable stack sizes
pthread_attr_setstacksize(&attr, 100); // Too small!
pthread_attr_setstacksize(&attr, 1GB); // Too large!

// Don't ignore return values
pthread_attr_setstacksize(&attr, size); // Should check return value
```

## Advanced pthread Concepts

### Thread-Local Storage (TLS)

Thread-local storage allows each thread to have its own copy of global variables, preventing interference between threads.

```c
#include <pthread.h>

// Methods for thread-local storage
int pthread_key_create(pthread_key_t *key, void (*destructor)(void*));
int pthread_key_delete(pthread_key_t key);
int pthread_setspecific(pthread_key_t key, const void *value);
void *pthread_getspecific(pthread_key_t key);
```

#### Thread-Local Storage Examples

**Example 1: Thread-Local Counter**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

pthread_key_t counter_key;

void counter_destructor(void* value) {
    printf("Thread exiting: counter value was %d\n", *(int*)value);
    free(value);
}

void increment_counter() {
    int* counter = (int*)pthread_getspecific(counter_key);
    
    if (counter == NULL) {
        // First time accessing counter in this thread
        counter = malloc(sizeof(int));
        *counter = 0;
        pthread_setspecific(counter_key, counter);
        printf("Thread %lu: Initialized counter\n", pthread_self());
    }
    
    (*counter)++;
    printf("Thread %lu: Counter = %d\n", pthread_self(), *counter);
}

void* worker_with_counter(void* arg) {
    int worker_id = *(int*)arg;
    
    printf("Worker %d starting\n", worker_id);
    
    // Each thread has its own counter
    for (int i = 0; i < 5; i++) {
        increment_counter();
        usleep(100000); // 100ms
    }
    
    printf("Worker %d completed\n", worker_id);
    return NULL;
}

int main() {
    // Create thread-local storage key
    pthread_key_create(&counter_key, counter_destructor);
    
    pthread_t threads[3];
    int worker_ids[3] = {1, 2, 3};
    
    // Create worker threads
    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, worker_with_counter, &worker_ids[i]);
    }
    
    // Wait for completion
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Clean up
    pthread_key_delete(counter_key);
    
    return 0;
}
```

**Example 2: Thread-Local Error Handling**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define MAX_ERROR_MSG 256

pthread_key_t error_key;

typedef struct {
    int error_code;
    char error_message[MAX_ERROR_MSG];
    char function_name[64];
} thread_error_t;

void error_destructor(void* value) {
    free(value);
}

void set_thread_error(int code, const char* message, const char* function) {
    thread_error_t* error = (thread_error_t*)pthread_getspecific(error_key);
    
    if (error == NULL) {
        error = malloc(sizeof(thread_error_t));
        pthread_setspecific(error_key, error);
    }
    
    error->error_code = code;
    strncpy(error->error_message, message, MAX_ERROR_MSG - 1);
    error->error_message[MAX_ERROR_MSG - 1] = '\0';
    strncpy(error->function_name, function, sizeof(error->function_name) - 1);
    error->function_name[sizeof(error->function_name) - 1] = '\0';
}

thread_error_t* get_thread_error() {
    return (thread_error_t*)pthread_getspecific(error_key);
}

// Simulate functions that might fail
int risky_operation_1() {
    // Simulate random failure
    if (rand() % 3 == 0) {
        set_thread_error(EINVAL, "Invalid parameter in operation 1", __func__);
        return -1;
    }
    return 0;
}

int risky_operation_2() {
    if (rand() % 4 == 0) {
        set_thread_error(ENOMEM, "Out of memory in operation 2", __func__);
        return -1;
    }
    return 0;
}

void* error_prone_worker(void* arg) {
    int worker_id = *(int*)arg;
    
    printf("Worker %d: Starting error-prone operations\n", worker_id);
    
    for (int i = 0; i < 5; i++) {
        printf("Worker %d: Attempt %d\n", worker_id, i + 1);
        
        if (risky_operation_1() != 0) {
            thread_error_t* error = get_thread_error();
            printf("Worker %d: Error in %s: %s (code: %d)\n", 
                   worker_id, error->function_name, error->error_message, error->error_code);
            continue;
        }
        
        if (risky_operation_2() != 0) {
            thread_error_t* error = get_thread_error();
            printf("Worker %d: Error in %s: %s (code: %d)\n", 
                   worker_id, error->function_name, error->error_message, error->error_code);
            continue;
        }
        
        printf("Worker %d: Attempt %d succeeded\n", worker_id, i + 1);
    }
    
    return NULL;
}

int main() {
    srand(time(NULL));
    
    // Initialize thread-local error storage
    pthread_key_create(&error_key, error_destructor);
    
    pthread_t threads[3];
    int worker_ids[3] = {1, 2, 3};
    
    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, error_prone_worker, &worker_ids[i]);
    }
    
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }
    
    pthread_key_delete(error_key);
    return 0;
}
```

### Thread Cancellation

Thread cancellation allows one thread to request the termination of another thread.

```c
// Cancellation functions
int pthread_cancel(pthread_t thread);
int pthread_setcancelstate(int state, int *oldstate);
int pthread_setcanceltype(int type, int *oldtype);
void pthread_testcancel(void);
void pthread_cleanup_push(void (*routine)(void*), void *arg);
void pthread_cleanup_pop(int execute);
```

#### Thread Cancellation Examples

**Example 1: Cooperative Cancellation**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void cleanup_handler(void* arg) {
    char* resource_name = (char*)arg;
    printf("Cleanup: Releasing %s\n", resource_name);
    free(resource_name);
}

void* cancellable_worker(void* arg) {
    int worker_id = *(int*)arg;
    char* resource_name = malloc(64);
    snprintf(resource_name, 64, "Resource-%d", worker_id);
    
    // Install cleanup handler
    pthread_cleanup_push(cleanup_handler, resource_name);
    
    // Enable cancellation
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);
    
    printf("Worker %d: Starting long-running task\n", worker_id);
    
    for (int i = 0; i < 100; i++) {
        // Check for cancellation requests
        pthread_testcancel();
        
        printf("Worker %d: Step %d/100\n", worker_id, i + 1);
        usleep(200000); // 200ms
        
        // Another cancellation point
        pthread_testcancel();
    }
    
    printf("Worker %d: Task completed normally\n", worker_id);
    
    // Remove cleanup handler (and execute it)
    pthread_cleanup_pop(1);
    
    return NULL;
}

int main() {
    pthread_t workers[3];
    int worker_ids[3] = {1, 2, 3};
    
    // Start workers
    for (int i = 0; i < 3; i++) {
        pthread_create(&workers[i], NULL, cancellable_worker, &worker_ids[i]);
    }
    
    // Let them run for a while
    printf("Main: Letting workers run for 3 seconds\n");
    sleep(3);
    
    // Cancel worker 2
    printf("Main: Cancelling worker 2\n");
    pthread_cancel(workers[1]);
    
    // Wait for all workers
    for (int i = 0; i < 3; i++) {
        void* result;
        int join_result = pthread_join(workers[i], &result);
        
        if (result == PTHREAD_CANCELED) {
            printf("Main: Worker %d was cancelled\n", i + 1);
        } else {
            printf("Main: Worker %d completed normally\n", i + 1);
        }
    }
    
    return 0;
}
```

**Example 2: Asynchronous vs Deferred Cancellation**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

void* async_cancellable_worker(void* arg) {
    // Enable asynchronous cancellation (dangerous!)
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
    
    printf("Async worker: Starting (can be cancelled at any time)\n");
    
    // Simulate work without cancellation points
    volatile long counter = 0;
    for (long i = 0; i < 1000000000L; i++) {
        counter += i;
        if (i % 100000000 == 0) {
            printf("Async worker: Progress %ld/1000000000\n", i);
        }
    }
    
    printf("Async worker: Completed\n");
    return NULL;
}

void* deferred_cancellable_worker(void* arg) {
    // Enable deferred cancellation (safe)
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);
    
    printf("Deferred worker: Starting (cancelled only at cancellation points)\n");
    
    for (int i = 0; i < 100; i++) {
        // Cancellation point
        pthread_testcancel();
        
        printf("Deferred worker: Step %d\n", i);
        
        // Simulate work
        usleep(100000); // This is also a cancellation point
    }
    
    printf("Deferred worker: Completed\n");
    return NULL;
}

int main() {
    pthread_t async_thread, deferred_thread;
    
    printf("=== Testing Asynchronous Cancellation ===\n");
    pthread_create(&async_thread, NULL, async_cancellable_worker, NULL);
    
    sleep(2); // Let it run briefly
    printf("Main: Cancelling async worker\n");
    pthread_cancel(async_thread);
    
    void* result;
    pthread_join(async_thread, &result);
    if (result == PTHREAD_CANCELED) {
        printf("Main: Async worker was cancelled\n");
    }
    
    printf("\n=== Testing Deferred Cancellation ===\n");
    pthread_create(&deferred_thread, NULL, deferred_cancellable_worker, NULL);
    
    sleep(2); // Let it run briefly  
    printf("Main: Cancelling deferred worker\n");
    pthread_cancel(deferred_thread);
    
    pthread_join(deferred_thread, &result);
    if (result == PTHREAD_CANCELED) {
        printf("Main: Deferred worker was cancelled\n");
    }
    
    return 0;
}
```

### Thread Pools

Thread pools are a common pattern for managing a fixed number of worker threads to handle tasks efficiently.

**Example: Simple Thread Pool Implementation**
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

#define MAX_TASKS 100

typedef struct {
    void (*function)(void* arg);
    void* argument;
} task_t;

typedef struct {
    pthread_t* threads;
    int thread_count;
    task_t task_queue[MAX_TASKS];
    int queue_front;
    int queue_rear;
    int queue_size;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_condition;
    bool shutdown;
} thread_pool_t;

thread_pool_t* create_thread_pool(int thread_count) {
    thread_pool_t* pool = malloc(sizeof(thread_pool_t));
    pool->threads = malloc(thread_count * sizeof(pthread_t));
    pool->thread_count = thread_count;
    pool->queue_front = 0;
    pool->queue_rear = 0;
    pool->queue_size = 0;
    pool->shutdown = false;
    
    pthread_mutex_init(&pool->queue_mutex, NULL);
    pthread_cond_init(&pool->queue_condition, NULL);
    
    return pool;
}

void* worker_thread(void* arg) {
    thread_pool_t* pool = (thread_pool_t*)arg;
    
    while (true) {
        pthread_mutex_lock(&pool->queue_mutex);
        
        // Wait for tasks or shutdown signal
        while (pool->queue_size == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->queue_condition, &pool->queue_mutex);
        }
        
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->queue_mutex);
            break;
        }
        
        // Get task from queue
        task_t task = pool->task_queue[pool->queue_front];
        pool->queue_front = (pool->queue_front + 1) % MAX_TASKS;
        pool->queue_size--;
        
        pthread_mutex_unlock(&pool->queue_mutex);
        
        // Execute task
        printf("Worker %lu: Executing task\n", pthread_self());
        task.function(task.argument);
    }
    
    printf("Worker %lu: Shutting down\n", pthread_self());
    return NULL;
}

void start_thread_pool(thread_pool_t* pool) {
    for (int i = 0; i < pool->thread_count; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }
}

bool add_task(thread_pool_t* pool, void (*function)(void*), void* argument) {
    pthread_mutex_lock(&pool->queue_mutex);
    
    if (pool->queue_size >= MAX_TASKS) {
        pthread_mutex_unlock(&pool->queue_mutex);
        return false; // Queue full
    }
    
    pool->task_queue[pool->queue_rear].function = function;
    pool->task_queue[pool->queue_rear].argument = argument;
    pool->queue_rear = (pool->queue_rear + 1) % MAX_TASKS;
    pool->queue_size++;
    
    pthread_cond_signal(&pool->queue_condition);
    pthread_mutex_unlock(&pool->queue_mutex);
    
    return true;
}

void shutdown_thread_pool(thread_pool_t* pool) {
    pthread_mutex_lock(&pool->queue_mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->queue_condition);
    pthread_mutex_unlock(&pool->queue_mutex);
    
    // Wait for all threads to finish
    for (int i = 0; i < pool->thread_count; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    // Cleanup
    pthread_mutex_destroy(&pool->queue_mutex);
    pthread_cond_destroy(&pool->queue_condition);
    free(pool->threads);
    free(pool);
}

// Example task functions
void simple_task(void* arg) {
    int task_id = *(int*)arg;
    printf("Task %d: Starting work\n", task_id);
    sleep(1 + rand() % 3); // 1-3 seconds of work
    printf("Task %d: Completed\n", task_id);
}

int main() {
    srand(time(NULL));
    
    // Create thread pool with 4 workers
    thread_pool_t* pool = create_thread_pool(4);
    start_thread_pool(pool);
    
    printf("Main: Thread pool started with 4 workers\n");
    
    // Add tasks to the pool
    int task_ids[10];
    for (int i = 0; i < 10; i++) {
        task_ids[i] = i + 1;
        add_task(pool, simple_task, &task_ids[i]);
        printf("Main: Added task %d\n", i + 1);
    }
    
    // Let tasks complete
    printf("Main: Waiting for tasks to complete...\n");
    sleep(10);
    
    // Shutdown pool
    printf("Main: Shutting down thread pool\n");
    shutdown_thread_pool(pool);
    
    printf("Main: Program completed\n");
    return 0;
}
```

## Comprehensive Code Examples

### Basic Thread Creation Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

void* thread_function(void* arg) {
    int thread_num = *(int*)arg;
    printf("Thread %d is running on system\n", thread_num);
    
    // Simulate work with different durations
    int work_time = 1 + (thread_num % 3);
    sleep(work_time);
    
    printf("Thread %d finished after %d seconds\n", thread_num, work_time);
    return NULL;
}

int main() {
    const int NUM_THREADS = 5;
    pthread_t threads[NUM_THREADS];
    int thread_args[NUM_THREADS];
    int result;
    
    printf("Main: Creating %d threads\n", NUM_THREADS);
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i] = i + 1;
        result = pthread_create(&threads[i], NULL, thread_function, &thread_args[i]);
        
        if (result != 0) {
            fprintf(stderr, "Error creating thread %d: %s\n", i, strerror(result));
            exit(EXIT_FAILURE);
        }
        
        printf("Main: Created thread %d with ID %lu\n", i + 1, threads[i]);
    }
    
    printf("Main: All threads created, waiting for completion\n");
    
    // Join threads
    for (int i = 0; i < NUM_THREADS; i++) {
        result = pthread_join(threads[i], NULL);
        if (result != 0) {
            fprintf(stderr, "Error joining thread %d: %s\n", i, strerror(result));
        } else {
            printf("Main: Thread %d joined successfully\n", i + 1);
        }
    }
    
    printf("Main: All threads completed\n");
    return 0;
}
```

### Thread with Return Value Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

typedef struct {
    int number;
    double square_root;
    int is_prime;
    long long factorial;
} calculation_result_t;

void* mathematical_calculator(void* arg) {
    int num = *(int*)arg;
    calculation_result_t* result = malloc(sizeof(calculation_result_t));
    
    result->number = num;
    
    // Calculate square root
    result->square_root = sqrt((double)num);
    
    // Check if prime
    result->is_prime = 1;
    if (num < 2) {
        result->is_prime = 0;
    } else {
        for (int i = 2; i <= sqrt(num); i++) {
            if (num % i == 0) {
                result->is_prime = 0;
                break;
            }
        }
    }
    
    // Calculate factorial (limited to prevent overflow)
    result->factorial = 1;
    if (num <= 20) { // Limit to prevent overflow
        for (int i = 1; i <= num; i++) {
            result->factorial *= i;
        }
    } else {
        result->factorial = -1; // Indicate overflow
    }
    
    printf("Thread: Calculated results for %d\n", num);
    return result;
}

int main() {
    const int NUM_CALCULATIONS = 6;
    pthread_t threads[NUM_CALCULATIONS];
    int numbers[NUM_CALCULATIONS] = {5, 7, 10, 13, 15, 25};
    
    printf("Main: Starting mathematical calculations\n");
    
    // Create threads for calculations
    for (int i = 0; i < NUM_CALCULATIONS; i++) {
        pthread_create(&threads[i], NULL, mathematical_calculator, &numbers[i]);
    }
    
    // Collect results
    for (int i = 0; i < NUM_CALCULATIONS; i++) {
        calculation_result_t* result;
        pthread_join(threads[i], (void**)&result);
        
        printf("\n--- Results for %d ---\n", result->number);
        printf("Square root: %.2f\n", result->square_root);
        printf("Is prime: %s\n", result->is_prime ? "Yes" : "No");
        
        if (result->factorial == -1) {
            printf("Factorial: Too large (overflow)\n");
        } else {
            printf("Factorial: %lld\n", result->factorial);
        }
        
        free(result);
    }
    
    printf("\nMain: All calculations completed\n");
    return 0;
}
```

### Producer-Consumer Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>

#define BUFFER_SIZE 10
#define NUM_ITEMS 20

typedef struct {
    int buffer[BUFFER_SIZE];
    int in;  // Next free position
    int out; // Next item to consume
    sem_t empty; // Count empty slots
    sem_t full;  // Count full slots
    pthread_mutex_t mutex; // Mutual exclusion for buffer access
} shared_buffer_t;

shared_buffer_t shared_buffer;

void* producer(void* arg) {
    int producer_id = *(int*)arg;
    
    for (int i = 0; i < NUM_ITEMS; i++) {
        int item = producer_id * 100 + i;
        
        // Wait for empty slot
        sem_wait(&shared_buffer.empty);
        
        // Acquire exclusive access to buffer
        pthread_mutex_lock(&shared_buffer.mutex);
        
        // Add item to buffer
        shared_buffer.buffer[shared_buffer.in] = item;
        printf("Producer %d: Produced item %d at position %d\n", 
               producer_id, item, shared_buffer.in);
        shared_buffer.in = (shared_buffer.in + 1) % BUFFER_SIZE;
        
        // Release exclusive access
        pthread_mutex_unlock(&shared_buffer.mutex);
        
        // Signal that buffer has one more item
        sem_post(&shared_buffer.full);
        
        // Simulate production time
        usleep(100000 + rand() % 200000); // 100-300ms
    }
    
    printf("Producer %d: Finished producing\n", producer_id);
    return NULL;
}

void* consumer(void* arg) {
    int consumer_id = *(int*)arg;
    int items_consumed = 0;
    
    while (items_consumed < NUM_ITEMS) {
        // Wait for item to be available
        sem_wait(&shared_buffer.full);
        
        // Acquire exclusive access to buffer
        pthread_mutex_lock(&shared_buffer.mutex);
        
        // Remove item from buffer
        int item = shared_buffer.buffer[shared_buffer.out];
        printf("Consumer %d: Consumed item %d from position %d\n", 
               consumer_id, item, shared_buffer.out);
        shared_buffer.out = (shared_buffer.out + 1) % BUFFER_SIZE;
        
        // Release exclusive access
        pthread_mutex_unlock(&shared_buffer.mutex);
        
        // Signal that buffer has one more empty slot
        sem_post(&shared_buffer.empty);
        
        items_consumed++;
        
        // Simulate consumption time
        usleep(150000 + rand() % 100000); // 150-250ms
    }
    
    printf("Consumer %d: Finished consuming\n", consumer_id);
    return NULL;
}

int main() {
    srand(time(NULL));
    
    // Initialize shared buffer
    shared_buffer.in = 0;
    shared_buffer.out = 0;
    sem_init(&shared_buffer.empty, 0, BUFFER_SIZE); // Initially all empty
    sem_init(&shared_buffer.full, 0, 0);            // Initially none full
    pthread_mutex_init(&shared_buffer.mutex, NULL);
    
    pthread_t producer_thread, consumer_thread;
    int producer_id = 1, consumer_id = 1;
    
    printf("Starting Producer-Consumer simulation\n");
    printf("Buffer size: %d, Items to produce: %d\n\n", BUFFER_SIZE, NUM_ITEMS);
    
    // Create producer and consumer threads
    pthread_create(&producer_thread, NULL, producer, &producer_id);
    pthread_create(&consumer_thread, NULL, consumer, &consumer_id);
    
    // Wait for both threads to complete
    pthread_join(producer_thread, NULL);
    pthread_join(consumer_thread, NULL);
    
    // Cleanup
    sem_destroy(&shared_buffer.empty);
    sem_destroy(&shared_buffer.full);
    pthread_mutex_destroy(&shared_buffer.mutex);
    
    printf("\nProducer-Consumer simulation completed\n");
    return 0;
}
```

### Multi-threaded File Processing Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    char input_filename[256];
    char output_filename[256];
    int thread_id;
    int lines_processed;
    int characters_processed;
} file_processor_t;

void* process_file(void* arg) {
    file_processor_t* processor = (file_processor_t*)arg;
    FILE* input_file = fopen(processor->input_filename, "r");
    FILE* output_file = fopen(processor->output_filename, "w");
    
    if (!input_file || !output_file) {
        printf("Thread %d: Error opening files\n", processor->thread_id);
        processor->lines_processed = -1;
        return processor;
    }
    
    printf("Thread %d: Processing %s -> %s\n", 
           processor->thread_id, processor->input_filename, processor->output_filename);
    
    char line[1024];
    processor->lines_processed = 0;
    processor->characters_processed = 0;
    
    while (fgets(line, sizeof(line), input_file)) {
        // Convert to uppercase and count characters
        int line_length = strlen(line);
        for (int i = 0; i < line_length; i++) {
            if (line[i] >= 'a' && line[i] <= 'z') {
                line[i] = line[i] - 'a' + 'A';
            }
            processor->characters_processed++;
        }
        
        // Write processed line
        fputs(line, output_file);
        processor->lines_processed++;
        
        // Simulate processing time
        usleep(10000); // 10ms per line
    }
    
    fclose(input_file);
    fclose(output_file);
    
    printf("Thread %d: Completed processing %d lines, %d characters\n", 
           processor->thread_id, processor->lines_processed, processor->characters_processed);
    
    return processor;
}

// Helper function to create test files
void create_test_file(const char* filename, int num_lines) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    
    for (int i = 0; i < num_lines; i++) {
        fprintf(file, "This is line %d with some sample text for processing.\n", i + 1);
    }
    fclose(file);
}

int main() {
    const int NUM_FILES = 4;
    pthread_t threads[NUM_FILES];
    file_processor_t processors[NUM_FILES];
    
    // Create test input files
    printf("Creating test files...\n");
    create_test_file("input1.txt", 50);
    create_test_file("input2.txt", 75);
    create_test_file("input3.txt", 100);
    create_test_file("input4.txt", 125);
    
    // Initialize processors
    char input_files[][256] = {"input1.txt", "input2.txt", "input3.txt", "input4.txt"};
    char output_files[][256] = {"output1.txt", "output2.txt", "output3.txt", "output4.txt"};
    
    printf("Starting parallel file processing...\n");
    
    // Create processing threads
    for (int i = 0; i < NUM_FILES; i++) {
        processors[i].thread_id = i + 1;
        strcpy(processors[i].input_filename, input_files[i]);
        strcpy(processors[i].output_filename, output_files[i]);
        
        pthread_create(&threads[i], NULL, process_file, &processors[i]);
    }
    
    // Wait for all processing to complete
    int total_lines = 0, total_characters = 0;
    for (int i = 0; i < NUM_FILES; i++) {
        file_processor_t* result;
        pthread_join(threads[i], (void**)&result);
        
        if (result->lines_processed > 0) {
            total_lines += result->lines_processed;
            total_characters += result->characters_processed;
        }
    }
    
    printf("\n=== Processing Summary ===\n");
    printf("Total files processed: %d\n", NUM_FILES);
    printf("Total lines processed: %d\n", total_lines);
    printf("Total characters processed: %d\n", total_characters);
    
    // Cleanup test files
    printf("Cleaning up test files...\n");
    remove("input1.txt"); remove("output1.txt");
    remove("input2.txt"); remove("output2.txt");
    remove("input3.txt"); remove("output3.txt");
    remove("input4.txt"); remove("output4.txt");
    
    return 0;
}
```

## Key Concepts and Best Practices

### Thread Lifecycle Management

Understanding the complete thread lifecycle is essential for effective pthread programming:

```
Thread Creation → Initialization → Execution → Synchronization → Termination → Cleanup
      ↓               ↓              ↓              ↓               ↓            ↓
pthread_create()  start_routine   Running      pthread_join()  return/exit   Resource
                     begins        state       or detached      from func     cleanup
```

**Detailed Lifecycle Phases:**

1. **Creation Phase**
   ```c
   pthread_t thread;
   int result = pthread_create(&thread, NULL, thread_function, arg);
   // Thread is now in system's ready queue
   ```

2. **Initialization Phase**
   ```c
   void* thread_function(void* arg) {
       // Thread-specific initialization
       pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
       // Setup thread-local storage if needed
       // ...
   }
   ```

3. **Execution Phase**
   ```c
   void* thread_function(void* arg) {
       // Main thread work
       while (work_to_do) {
           do_work();
           pthread_testcancel(); // Check for cancellation
       }
   }
   ```

4. **Termination Phase**
   ```c
   void* thread_function(void* arg) {
       // Cleanup before return
       cleanup_resources();
       return result_value; // or pthread_exit(result_value)
   }
   ```

5. **Cleanup Phase**
   ```c
   // In main thread
   void* result;
   pthread_join(thread, &result); // Collect resources
   // or thread auto-cleans if detached
   ```

### Error Handling Best Practices

Proper error handling is crucial for robust pthread applications:

```c
#include <pthread.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

// Comprehensive error checking wrapper
int safe_pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                       void* (*start_routine)(void*), void* arg, 
                       const char* context) {
    int result = pthread_create(thread, attr, start_routine, arg);
    
    if (result != 0) {
        fprintf(stderr, "pthread_create failed in %s: %s (error code: %d)\n",
                context, strerror(result), result);
        
        switch (result) {
            case EAGAIN:
                fprintf(stderr, "  - System lacks resources or thread limit reached\n");
                break;
            case EINVAL:
                fprintf(stderr, "  - Invalid thread attributes\n");
                break;
            case EPERM:
                fprintf(stderr, "  - No permission to set scheduling parameters\n");
                break;
            default:
                fprintf(stderr, "  - Unknown error occurred\n");
        }
        return -1;
    }
    
    return 0;
}

// Error-safe thread joining
int safe_pthread_join(pthread_t thread, void** retval, const char* context) {
    int result = pthread_join(thread, retval);
    
    if (result != 0) {
        fprintf(stderr, "pthread_join failed in %s: %s\n", context, strerror(result));
        return -1;
    }
    
    return 0;
}

// Example usage
void* worker_thread(void* arg) {
    // Simulate potential failure
    if (rand() % 10 == 0) {
        fprintf(stderr, "Worker thread encountered an error\n");
        return (void*)-1;
    }
    
    printf("Worker thread completed successfully\n");
    return (void*)0;
}

int main() {
    pthread_t thread;
    void* result;
    
    if (safe_pthread_create(&thread, NULL, worker_thread, NULL, "main") != 0) {
        exit(EXIT_FAILURE);
    }
    
    if (safe_pthread_join(thread, &result, "main") != 0) {
        exit(EXIT_FAILURE);
    }
    
    if (result == (void*)-1) {
        printf("Main: Worker thread reported an error\n");
    } else {
        printf("Main: Worker thread completed successfully\n");
    }
    
    return 0;
}
```

### Memory Management in Multi-threaded Programs

Proper memory management is critical in pthread applications:

```c
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Thread-safe memory pool
typedef struct memory_block {
    void* data;
    size_t size;
    int in_use;
    struct memory_block* next;
} memory_block_t;

typedef struct {
    memory_block_t* blocks;
    pthread_mutex_t mutex;
} memory_pool_t;

memory_pool_t* create_memory_pool() {
    memory_pool_t* pool = malloc(sizeof(memory_pool_t));
    pool->blocks = NULL;
    pthread_mutex_init(&pool->mutex, NULL);
    return pool;
}

void* pool_allocate(memory_pool_t* pool, size_t size) {
    pthread_mutex_lock(&pool->mutex);
    
    // Find available block or create new one
    memory_block_t* block = pool->blocks;
    while (block) {
        if (!block->in_use && block->size >= size) {
            block->in_use = 1;
            pthread_mutex_unlock(&pool->mutex);
            return block->data;
        }
        block = block->next;
    }
    
    // Create new block
    block = malloc(sizeof(memory_block_t));
    block->data = malloc(size);
    block->size = size;
    block->in_use = 1;
    block->next = pool->blocks;
    pool->blocks = block;
    
    pthread_mutex_unlock(&pool->mutex);
    return block->data;
}

void pool_free(memory_pool_t* pool, void* ptr) {
    pthread_mutex_lock(&pool->mutex);
    
    memory_block_t* block = pool->blocks;
    while (block) {
        if (block->data == ptr) {
            block->in_use = 0;
            break;
        }
        block = block->next;
    }
    
    pthread_mutex_unlock(&pool->mutex);
}

// Thread argument management
typedef struct {
    int thread_id;
    char* message;
    int* shared_counter;
    pthread_mutex_t* counter_mutex;
} thread_context_t;

thread_context_t* create_thread_context(int id, const char* msg, 
                                       int* counter, pthread_mutex_t* mutex) {
    thread_context_t* ctx = malloc(sizeof(thread_context_t));
    ctx->thread_id = id;
    ctx->message = strdup(msg); // Allocate copy of string
    ctx->shared_counter = counter;
    ctx->counter_mutex = mutex;
    return ctx;
}

void destroy_thread_context(thread_context_t* ctx) {
    if (ctx) {
        free(ctx->message);
        free(ctx);
    }
}

void* memory_aware_worker(void* arg) {
    thread_context_t* ctx = (thread_context_t*)arg;
    
    printf("Thread %d: %s\n", ctx->thread_id, ctx->message);
    
    // Thread-safe counter update
    pthread_mutex_lock(ctx->counter_mutex);
    (*ctx->shared_counter)++;
    printf("Thread %d: Counter updated to %d\n", ctx->thread_id, *ctx->shared_counter);
    pthread_mutex_unlock(ctx->counter_mutex);
    
    // Clean up context before returning
    destroy_thread_context(ctx);
    return NULL;
}
```

### Performance Optimization Techniques

Optimizing pthread performance requires understanding system behavior:

```c
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

// Performance measurement utilities
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// CPU affinity setting (Linux-specific)
#ifdef __linux__
#include <sched.h>

void set_thread_affinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        fprintf(stderr, "Failed to set CPU affinity: %s\n", strerror(result));
    } else {
        printf("Thread bound to CPU %d\n", cpu_id);
    }
}
#endif

// Cache-friendly data structure
typedef struct {
    int data[16]; // Cache line size aligned
    pthread_mutex_t mutex;
    char padding[64 - sizeof(pthread_mutex_t) - 64]; // Prevent false sharing
} cache_aligned_data_t;

// Lock-free atomic operations example
#include <stdatomic.h>

atomic_int lock_free_counter = 0;

void* lock_free_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < 100000; i++) {
        atomic_fetch_add(&lock_free_counter, 1);
    }
    
    printf("Lock-free worker %d completed\n", thread_id);
    return NULL;
}

// Performance comparison: mutex vs atomic
void compare_synchronization_performance() {
    struct timespec start, end;
    const int NUM_THREADS = 4;
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    
    // Test 1: Atomic operations
    printf("=== Testing Lock-Free Atomic Operations ===\n");
    lock_free_counter = 0;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, lock_free_worker, &thread_ids[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double atomic_time = get_time_diff(start, end);
    
    printf("Atomic result: %d, Time: %.3f seconds\n", 
           atomic_load(&lock_free_counter), atomic_time);
    
    // Test 2: Mutex-based operations (previous example)
    // ... (implementation would be similar to earlier mutex examples)
    
    printf("Performance comparison completed\n");
}
```

### Debugging and Profiling pthread Applications

Essential debugging techniques for pthread programs:

```c
#include <pthread.h>
#include <stdio.h>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

// Debug-friendly thread creation with naming
void set_thread_name(const char* name) {
#ifdef __linux__
    pthread_setname_np(pthread_self(), name);
#endif
    printf("Thread name set to: %s (ID: %lu)\n", name, pthread_self());
}

// Signal handler for debugging
void debug_signal_handler(int sig) {
    void* array[10];
    size_t size;
    
    printf("Thread %lu received signal %d\n", pthread_self(), sig);
    
    // Get stack trace
    size = backtrace(array, 10);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    
    exit(1);
}

// Debug-enabled thread function
void* debug_thread(void* arg) {
    char thread_name[32];
    int thread_id = *(int*)arg;
    
    snprintf(thread_name, sizeof(thread_name), "Worker-%d", thread_id);
    set_thread_name(thread_name);
    
    // Install signal handler for debugging
    signal(SIGUSR1, debug_signal_handler);
    
    printf("Debug thread %d: Starting work\n", thread_id);
    
    // Simulate work with periodic status updates
    for (int i = 0; i < 10; i++) {
        printf("Debug thread %d: Step %d/10\n", thread_id, i + 1);
        sleep(1);
        
        // Artificial crash condition for testing
        if (thread_id == 2 && i == 5) {
            printf("Debug thread %d: Triggering debug condition\n", thread_id);
            raise(SIGUSR1);
        }
    }
    
    printf("Debug thread %d: Completed\n", thread_id);
    return NULL;
}

// Thread monitoring function
void* monitor_thread(void* arg) {
    set_thread_name("Monitor");
    
    while (1) {
        printf("Monitor: System status check\n");
        
        // Monitor system resources, thread states, etc.
        // This would typically check /proc/self/task/ on Linux
        
        sleep(5);
    }
    
    return NULL;
}
```

## Hands-on Exercises and Projects

### Exercise 1: Multi-threaded Hello World
**Objective:** Master basic thread creation and management

**Requirements:**
```c
// Complete this program
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void* hello_thread(void* arg) {
    // TODO: Print "Hello from thread X" where X is the thread number
    // TODO: Include thread ID in the output
    // TODO: Sleep for a random time (1-3 seconds)
    // TODO: Print completion message
}

int main() {
    // TODO: Create 5 threads
    // TODO: Pass unique identifiers to each thread
    // TODO: Join all threads before program exit
    // TODO: Print summary of completed threads
    return 0;
}
```

**Expected Output:**
```
Main: Creating 5 threads
Hello from thread 1 (ID: 140234567891200)
Hello from thread 3 (ID: 140234567892200)
...
Thread 1 completed after 2 seconds
Main: All 5 threads completed successfully
```

### Exercise 2: Thread with Complex Parameters
**Objective:** Practice passing structured data to threads

**Challenge:** Create a program that processes different mathematical operations in parallel:

```c
typedef struct {
    int thread_id;
    char operation[16]; // "factorial", "fibonacci", "prime_check"
    int input_number;
    long long result;
    double execution_time;
} math_task_t;

void* math_worker(void* arg) {
    // TODO: Implement factorial, fibonacci, and prime checking
    // TODO: Measure execution time
    // TODO: Store results in the task structure
}

int main() {
    // TODO: Create tasks for different operations
    // TODO: Create threads to process tasks
    // TODO: Collect and display results
}
```

### Exercise 3: Producer-Consumer with Multiple Producers
**Objective:** Implement complex synchronization scenarios

**Challenge:** Extend the basic producer-consumer to handle:
- 3 producer threads (different production rates)
- 2 consumer threads (different consumption patterns)
- Bounded buffer with statistics tracking

```c
typedef struct {
    int item_id;
    int producer_id;
    time_t timestamp;
} item_t;

typedef struct {
    item_t* buffer;
    int capacity;
    int count;
    int in, out;
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
    // TODO: Add statistics fields
} buffer_t;

// TODO: Implement buffer operations
// TODO: Add statistics tracking
// TODO: Handle graceful shutdown
```

### Exercise 4: Thread Pool Implementation
**Objective:** Master advanced threading patterns

**Challenge:** Build a thread pool that supports:
- Dynamic task submission
- Priority-based task scheduling
- Thread pool resizing
- Graceful shutdown

```c
typedef enum {
    PRIORITY_LOW = 1,
    PRIORITY_NORMAL = 2,
    PRIORITY_HIGH = 3
} task_priority_t;

typedef struct task {
    void (*function)(void*);
    void* arg;
    task_priority_t priority;
    struct task* next;
} task_t;

typedef struct {
    pthread_t* threads;
    int thread_count;
    int active_threads;
    task_t* task_queue;
    pthread_mutex_t queue_mutex;
    pthread_cond_t work_available;
    pthread_cond_t work_complete;
    int shutdown;
} thread_pool_t;

// TODO: Implement thread pool functions
// thread_pool_t* create_pool(int size);
// int add_task(thread_pool_t* pool, void (*func)(void*), void* arg, task_priority_t priority);
// void resize_pool(thread_pool_t* pool, int new_size);
// void shutdown_pool(thread_pool_t* pool);
```

### Project: Multi-threaded Web Log Analyzer
**Objective:** Apply pthread concepts to a real-world problem

**Requirements:**
1. Parse multiple web log files concurrently
2. Extract and analyze different metrics in separate threads
3. Use thread-safe data structures for results aggregation
4. Generate a comprehensive report

**Architecture:**
```
File Reader Threads → Processing Queue → Analysis Threads → Results Aggregator
       ↓                    ↓                   ↓                ↓
   log1.txt              Shared              IP Stats        Final Report
   log2.txt              Buffer              URL Stats           ↓
   log3.txt                ↓               Error Stats      report.txt
```

**Implementation Framework:**
```c
typedef struct log_entry {
    char ip[16];
    char timestamp[32];
    char method[8];
    char url[256];
    int status_code;
    int bytes_sent;
} log_entry_t;

typedef struct analysis_result {
    // IP address statistics
    char top_ips[10][16];
    int ip_counts[10];
    
    // URL statistics  
    char top_urls[10][256];
    int url_counts[10];
    
    // Status code statistics
    int status_codes[600]; // Index = status code
    
    // General statistics
    long total_requests;
    long total_bytes;
    double avg_response_size;
    
    pthread_mutex_t mutex;
} analysis_result_t;

// TODO: Implement the complete log analyzer
```

### Exercise 5: Thread Synchronization Debugging
**Objective:** Practice identifying and fixing threading bugs

**Buggy Code to Fix:**
```c
// This code has multiple threading bugs - find and fix them!
int shared_counter = 0;
pthread_mutex_t mutex;

void* buggy_worker(void* arg) {
    int iterations = *(int*)arg;
    
    for (int i = 0; i < iterations; i++) {
        // Bug 1: Missing mutex lock
        shared_counter++;
        
        if (shared_counter % 100 == 0) {
            printf("Counter: %d\n", shared_counter);
        }
        
        // Bug 2: Potential stack variable issue
        int local_var = i;
        pthread_create(/* ... */); // New thread created with &local_var
    }
    
    // Bug 3: Missing return statement or wrong return type
}

int main() {
    pthread_t threads[5];
    
    // Bug 4: Mutex not initialized
    
    for (int i = 0; i < 5; i++) {
        int arg = 1000;
        // Bug 5: All threads get same argument address
        pthread_create(&threads[i], NULL, buggy_worker, &arg);
    }
    
    // Bug 6: Not joining threads
    
    printf("Final counter: %d\n", shared_counter);
    return 0;
}
```

### Advanced Challenge: Parallel Matrix Multiplication
**Objective:** Optimize computational workloads with threading

**Challenge:** Implement parallel matrix multiplication with:
- Block-based threading strategy
- Cache-friendly data access patterns
- Performance measurement and comparison
- Load balancing between threads

```c
typedef struct {
    int** matrix_a;
    int** matrix_b;
    int** result;
    int start_row;
    int end_row;
    int n; // Matrix dimension
    int thread_id;
    double execution_time;
} matrix_thread_data_t;

// TODO: Implement parallel matrix multiplication
// TODO: Compare performance with single-threaded version
// TODO: Experiment with different thread counts
// TODO: Measure cache performance impact
```

### Assessment Criteria

For each exercise, ensure you can demonstrate:

**✅ Basic Requirements:**
- [ ] Code compiles without warnings
- [ ] Proper error handling for all pthread functions
- [ ] No memory leaks (use valgrind to verify)
- [ ] Correct thread synchronization
- [ ] Clear, readable code with comments

**✅ Advanced Requirements:**
- [ ] Performance benchmarking
- [ ] Race condition testing
- [ ] Resource usage optimization
- [ ] Graceful error recovery
- [ ] Comprehensive debugging output

**✅ Professional Standards:**
- [ ] Modular code design
- [ ] Consistent naming conventions
- [ ] Proper resource cleanup
- [ ] Documentation and usage examples
- [ ] Unit tests for critical functions

## Common Pitfalls and Solutions

### 1. Passing Stack Variables to Threads

**❌ WRONG - Classic Bug:**
```c
void create_threads_wrong() {
    pthread_t threads[5];
    
    for (int i = 0; i < 5; i++) {
        // BUG: All threads will see the same value of i (probably 5)
        // because i is on the stack and changes before threads start
        pthread_create(&threads[i], NULL, worker_function, &i);
    }
    
    for (int i = 0; i < 5; i++) {
        pthread_join(threads[i], NULL);
    }
}
```

**✅ CORRECT Solutions:**
```c
// Solution 1: Use separate storage for each thread
void create_threads_correct_v1() {
    pthread_t threads[5];
    int thread_args[5]; // Separate storage
    
    for (int i = 0; i < 5; i++) {
        thread_args[i] = i; // Each thread gets its own copy
        pthread_create(&threads[i], NULL, worker_function, &thread_args[i]);
    }
    
    for (int i = 0; i < 5; i++) {
        pthread_join(threads[i], NULL);
    }
}

// Solution 2: Pass value directly (for small data)
void* worker_function_v2(void* arg) {
    int thread_id = (int)(intptr_t)arg; // Cast pointer to int
    printf("Thread %d running\n", thread_id);
    return NULL;
}

void create_threads_correct_v2() {
    pthread_t threads[5];
    
    for (int i = 0; i < 5; i++) {
        pthread_create(&threads[i], NULL, worker_function_v2, (void*)(intptr_t)i);
    }
    
    for (int i = 0; i < 5; i++) {
        pthread_join(threads[i], NULL);
    }
}

// Solution 3: Dynamic allocation for complex data
typedef struct {
    int thread_id;
    char* message;
    int* shared_data;
} thread_data_t;

void* worker_function_v3(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    printf("Thread %d: %s\n", data->thread_id, data->message);
    
    // Clean up allocated data
    free(data->message);
    free(data);
    return NULL;
}

void create_threads_correct_v3() {
    pthread_t threads[5];
    
    for (int i = 0; i < 5; i++) {
        thread_data_t* data = malloc(sizeof(thread_data_t));
        data->thread_id = i;
        data->message = strdup("Hello from thread");
        data->shared_data = NULL;
        
        pthread_create(&threads[i], NULL, worker_function_v3, data);
    }
    
    for (int i = 0; i < 5; i++) {
        pthread_join(threads[i], NULL);
    }
}
```

### 2. Forgetting to Join Threads

**❌ WRONG - Resource Leak:**
```c
void resource_leak_example() {
    for (int i = 0; i < 100; i++) {
        pthread_t thread;
        pthread_create(&thread, NULL, worker_function, NULL);
        // BUG: Never joining threads - resource leak!
    }
    // Program exits with 100 zombie threads
}
```

**✅ CORRECT Solutions:**
```c
// Solution 1: Always join threads
void proper_joining() {
    pthread_t threads[100];
    
    for (int i = 0; i < 100; i++) {
        pthread_create(&threads[i], NULL, worker_function, NULL);
    }
    
    // Join all threads
    for (int i = 0; i < 100; i++) {
        pthread_join(threads[i], NULL);
    }
}

// Solution 2: Use detached threads for fire-and-forget tasks
void proper_detached_threads() {
    for (int i = 0; i < 100; i++) {
        pthread_t thread;
        pthread_attr_t attr;
        
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        
        pthread_create(&thread, &attr, worker_function, NULL);
        
        pthread_attr_destroy(&attr);
        // No need to join - resources auto-cleaned
    }
}
```

### 3. Not Checking Return Values

**❌ WRONG - Silent Failures:**
```c
void ignore_errors() {
    pthread_t thread;
    pthread_mutex_t mutex;
    
    // BUG: Ignoring return values
    pthread_create(&thread, NULL, worker_function, NULL);
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_lock(&mutex);
    pthread_join(thread, NULL);
    pthread_mutex_destroy(&mutex);
}
```

**✅ CORRECT - Comprehensive Error Checking:**
```c
void proper_error_checking() {
    pthread_t thread;
    pthread_mutex_t mutex;
    int result;
    
    // Check thread creation
    result = pthread_create(&thread, NULL, worker_function, NULL);
    if (result != 0) {
        fprintf(stderr, "pthread_create failed: %s\n", strerror(result));
        exit(EXIT_FAILURE);
    }
    
    // Check mutex initialization
    result = pthread_mutex_init(&mutex, NULL);
    if (result != 0) {
        fprintf(stderr, "pthread_mutex_init failed: %s\n", strerror(result));
        exit(EXIT_FAILURE);
    }
    
    // Check mutex lock
    result = pthread_mutex_lock(&mutex);
    if (result != 0) {
        fprintf(stderr, "pthread_mutex_lock failed: %s\n", strerror(result));
        // Handle error appropriately
    }
    
    // Check thread join
    result = pthread_join(thread, NULL);
    if (result != 0) {
        fprintf(stderr, "pthread_join failed: %s\n", strerror(result));
    }
    
    // Check mutex cleanup
    result = pthread_mutex_destroy(&mutex);
    if (result != 0) {
        fprintf(stderr, "pthread_mutex_destroy failed: %s\n", strerror(result));
    }
}
```

### 4. Race Conditions in Shared Data

**❌ WRONG - Race Condition:**
```c
int global_counter = 0;

void* increment_counter_unsafe(void* arg) {
    for (int i = 0; i < 100000; i++) {
        // BUG: Race condition - not atomic
        global_counter++; // Read-Modify-Write operation
    }
    return NULL;
}

void race_condition_demo() {
    pthread_t threads[4];
    
    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, increment_counter_unsafe, NULL);
    }
    
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Counter: %d (should be 400000)\n", global_counter);
    // Will print less than 400000 due to race condition
}
```

**✅ CORRECT Solutions:**
```c
// Solution 1: Mutex protection
int global_counter = 0;
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment_counter_safe(void* arg) {
    for (int i = 0; i < 100000; i++) {
        pthread_mutex_lock(&counter_mutex);
        global_counter++; // Now atomic with respect to other threads
        pthread_mutex_unlock(&counter_mutex);
    }
    return NULL;
}

// Solution 2: Atomic operations (C11 or later)
#include <stdatomic.h>
atomic_int atomic_counter = 0;

void* increment_counter_atomic(void* arg) {
    for (int i = 0; i < 100000; i++) {
        atomic_fetch_add(&atomic_counter, 1);
    }
    return NULL;
}

// Solution 3: Local accumulation + final sum
int global_counter = 0;
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment_counter_optimized(void* arg) {
    int local_sum = 0;
    
    // Accumulate locally (no contention)
    for (int i = 0; i < 100000; i++) {
        local_sum++;
    }
    
    // Single critical section
    pthread_mutex_lock(&counter_mutex);
    global_counter += local_sum;
    pthread_mutex_unlock(&counter_mutex);
    
    return NULL;
}
```

### 5. Deadlock Prevention

**❌ WRONG - Potential Deadlock:**
```c
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void* thread1_function(void* arg) {
    pthread_mutex_lock(&mutex1);
    printf("Thread 1: Acquired mutex1\n");
    sleep(1);
    
    printf("Thread 1: Trying to acquire mutex2\n");
    pthread_mutex_lock(&mutex2); // May deadlock
    
    printf("Thread 1: Acquired both mutexes\n");
    
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

void* thread2_function(void* arg) {
    pthread_mutex_lock(&mutex2);
    printf("Thread 2: Acquired mutex2\n");
    sleep(1);
    
    printf("Thread 2: Trying to acquire mutex1\n");
    pthread_mutex_lock(&mutex1); // May deadlock
    
    printf("Thread 2: Acquired both mutexes\n");
    
    pthread_mutex_unlock(&mutex1);
    pthread_mutex_unlock(&mutex2);
    return NULL;
}
```

**✅ CORRECT Solutions:**
```c
// Solution 1: Consistent lock ordering
void* thread1_safe(void* arg) {
    // Always acquire mutex1 first, then mutex2
    pthread_mutex_lock(&mutex1);
    pthread_mutex_lock(&mutex2);
    
    printf("Thread 1: Working with both mutexes\n");
    
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

void* thread2_safe(void* arg) {
    // Same order: mutex1 first, then mutex2
    pthread_mutex_lock(&mutex1);
    pthread_mutex_lock(&mutex2);
    
    printf("Thread 2: Working with both mutexes\n");
    
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

// Solution 2: Try-lock with timeout
void* thread1_trylock(void* arg) {
    pthread_mutex_lock(&mutex1);
    
    if (pthread_mutex_trylock(&mutex2) == 0) {
        // Got both locks
        printf("Thread 1: Working with both mutexes\n");
        pthread_mutex_unlock(&mutex2);
    } else {
        printf("Thread 1: Could not acquire mutex2, continuing\n");
    }
    
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

// Solution 3: Single mutex for related resources
pthread_mutex_t combined_mutex = PTHREAD_MUTEX_INITIALIZER;

void* thread1_combined(void* arg) {
    pthread_mutex_lock(&combined_mutex);
    
    // Work with both resources under single lock
    printf("Thread 1: Working with both resources\n");
    
    pthread_mutex_unlock(&combined_mutex);
    return NULL;
}
```

### 6. Thread Cancellation Handling

**❌ WRONG - Unsafe Cancellation:**
```c
void* unsafe_cancellable_thread(void* arg) {
    FILE* file = fopen("important_data.txt", "w");
    char* buffer = malloc(1024);
    
    // BUG: No cleanup handlers - resources leak if cancelled
    
    for (int i = 0; i < 1000000; i++) {
        fprintf(file, "Data line %d\n", i);
        
        // BUG: Cancellation can occur here, leaving file open
        if (i % 1000 == 0) {
            printf("Progress: %d/1000000\n", i);
        }
    }
    
    fclose(file);
    free(buffer);
    return NULL;
}
```

**✅ CORRECT - Safe Cancellation:**
```c
typedef struct {
    FILE* file;
    char* buffer;
} cleanup_data_t;

void cleanup_handler(void* arg) {
    cleanup_data_t* data = (cleanup_data_t*)arg;
    
    if (data->file) {
        fclose(data->file);
        printf("Cleanup: File closed\n");
    }
    
    if (data->buffer) {
        free(data->buffer);
        printf("Cleanup: Buffer freed\n");
    }
    
    free(data);
}

void* safe_cancellable_thread(void* arg) {
    cleanup_data_t* cleanup_data = malloc(sizeof(cleanup_data_t));
    cleanup_data->file = fopen("important_data.txt", "w");
    cleanup_data->buffer = malloc(1024);
    
    // Install cleanup handler
    pthread_cleanup_push(cleanup_handler, cleanup_data);
    
    // Enable cancellation
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);
    
    for (int i = 0; i < 1000000; i++) {
        fprintf(cleanup_data->file, "Data line %d\n", i);
        
        // Safe cancellation point
        if (i % 1000 == 0) {
            printf("Progress: %d/1000000\n", i);
            pthread_testcancel(); // Check for cancellation
        }
    }
    
    // Normal cleanup (execute handler)
    pthread_cleanup_pop(1);
    
    return NULL;
}
```

### Debugging Tools and Techniques

**Compilation Flags for Debugging:**
```bash
# Debug build with all warnings
gcc -g -Wall -Wextra -pthread -o program program.c

# Thread sanitizer (detects race conditions)
gcc -g -fsanitize=thread -pthread -o program program.c

# Address sanitizer (detects memory errors)
gcc -g -fsanitize=address -pthread -o program program.c
```

**Runtime Debugging:**
```bash
# Use GDB for debugging
gdb ./program
(gdb) set scheduler-locking on
(gdb) info threads
(gdb) thread 2
(gdb) bt

# Use Valgrind for race detection
valgrind --tool=helgrind ./program

# Use Valgrind for memory leak detection
valgrind --leak-check=full ./program
```

## Assessment and Learning Outcomes

### Learning Objectives Checklist

By completing this section, you should be able to:

**✅ Basic pthread Operations:**
- [ ] Create threads using `pthread_create()` with proper error handling
- [ ] Join threads using `pthread_join()` and handle return values
- [ ] Create detached threads using `pthread_detach()` when appropriate
- [ ] Use thread attributes to customize thread behavior
- [ ] Identify thread IDs using `pthread_self()` and `pthread_equal()`

**✅ Advanced pthread Concepts:**
- [ ] Implement thread-local storage using pthread keys
- [ ] Handle thread cancellation safely with cleanup handlers
- [ ] Create and manage thread pools for efficient task processing
- [ ] Use atomic operations as alternatives to mutex locking
- [ ] Debug multi-threaded programs using appropriate tools

**✅ Best Practices and Problem Solving:**
- [ ] Avoid common pitfalls (stack variable passing, resource leaks)
- [ ] Implement proper error handling for all pthread functions
- [ ] Design thread-safe data structures and algorithms
- [ ] Optimize thread performance through careful design choices
- [ ] Debug race conditions and deadlocks effectively

### Self-Assessment Quiz

**Conceptual Questions:**

1. **Thread Creation:** What happens if `pthread_create()` is called with a NULL attribute parameter?

2. **Memory Management:** Why is it dangerous to pass the address of a loop variable to `pthread_create()`?

3. **Thread Lifecycle:** What's the difference between a joinable and detached thread in terms of resource management?

4. **Synchronization:** When would you choose atomic operations over mutex locks?

5. **Error Handling:** What information can you get from pthread function return values?

**Practical Challenges:**

6. **Code Review:** Find the bugs in this code snippet:
```c
void create_workers() {
    pthread_t threads[10];
    for (int i = 0; i < 10; i++) {
        pthread_create(&threads[i], NULL, worker, &i);
    }
}
```

7. **Performance:** Explain why this counter implementation might be slow:
```c
void* counter_thread(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        pthread_mutex_lock(&mutex);
        global_counter++;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}
```

8. **Design Decision:** When would you use thread-local storage instead of mutex-protected shared variables?

### Hands-on Validation Projects

**Project 1: Thread-Safe Data Structure**
Implement a thread-safe queue with the following operations:
```c
typedef struct thread_safe_queue thread_safe_queue_t;

thread_safe_queue_t* queue_create(int capacity);
int queue_enqueue(thread_safe_queue_t* queue, void* item);
void* queue_dequeue(thread_safe_queue_t* queue);
int queue_size(thread_safe_queue_t* queue);
void queue_destroy(thread_safe_queue_t* queue);
```

**Project 2: Parallel File Processor**
Create a program that:
- Reads multiple text files concurrently
- Processes each file in a separate thread
- Aggregates results (word count, line count, etc.)
- Handles files of different sizes efficiently

**Project 3: Thread Pool Web Server Simulation**
Implement a simplified web server that:
- Uses a thread pool to handle requests
- Processes different types of requests (GET, POST)
- Maintains connection statistics
- Handles graceful shutdown

### Performance Benchmarking

**Benchmark 1: Thread Creation Overhead**
```c
// Measure time to create and join N threads
void benchmark_thread_creation(int num_threads) {
    struct timespec start, end;
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, empty_worker, NULL);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_taken = (end.tv_sec - start.tv_sec) + 
                       (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Created %d threads in %.3f seconds (%.0f threads/sec)\n",
           num_threads, time_taken, num_threads / time_taken);
    
    free(threads);
}
```

**Benchmark 2: Synchronization Overhead**
Compare performance of:
- Mutex-protected operations
- Atomic operations
- Lock-free algorithms
- Thread-local storage

**Benchmark 3: Scalability Analysis**
Test how your programs perform with:
- 1, 2, 4, 8, 16 threads
- Different workload sizes
- Various synchronization strategies

### Code Quality Standards

**✅ Code Quality Checklist:**
- [ ] All pthread functions have return value checking
- [ ] Proper resource cleanup (no memory leaks)
- [ ] Consistent error handling strategy
- [ ] Thread-safe access to all shared data
- [ ] Clear separation of concerns between threads
- [ ] Comprehensive logging for debugging
- [ ] Performance considerations documented

**✅ Documentation Requirements:**
- [ ] Function-level documentation for all pthread operations
- [ ] Architecture diagram showing thread interactions
- [ ] Performance characteristics documented
- [ ] Known limitations and trade-offs explained
- [ ] Usage examples for all major features

### Debugging Proficiency Test

**Challenge:** Debug this problematic multi-threaded program:

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

int shared_data[1000];
pthread_mutex_t mutex;

void* worker(void* arg) {
    int id = (int)arg;
    
    for (int i = 0; i < 100; i++) {
        pthread_mutex_lock(&mutex);
        
        for (int j = 0; j < 1000; j++) {
            shared_data[j] += id;
        }
        
        if (i % 10 == 0) {
            printf("Worker %d: iteration %d\n", id, i);
        }
        
        pthread_mutex_unlock(&mutex);
    }
    
    return NULL;
}

int main() {
    pthread_t threads[10];
    
    for (int i = 0; i < 10; i++) {
        pthread_create(&threads[i], NULL, worker, (void*)i);
    }
    
    for (int i = 0; i < 10; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Sum: %d\n", shared_data[0]);
    return 0;
}
```

**Issues to identify:**
1. Mutex initialization missing
2. Potential performance issues
3. Argument passing problems
4. Missing error checking
5. Resource cleanup issues

### Final Competency Demonstration

**Capstone Project: Multi-threaded Application Framework**

Design and implement a complete multi-threaded application that demonstrates mastery of:

1. **Thread Management:** Dynamic thread creation/destruction
2. **Synchronization:** Multiple synchronization primitives
3. **Performance:** Efficient resource utilization
4. **Robustness:** Comprehensive error handling
5. **Maintainability:** Clean, documented code

**Example Applications:**
- Multi-threaded web crawler
- Parallel image processing pipeline
- Real-time data processing system
- Multi-user chat server
- Distributed computation framework

**Evaluation Criteria:**
- Correctness (no race conditions, deadlocks)
- Performance (efficient CPU and memory usage)
- Robustness (handles errors gracefully)
- Code quality (readable, maintainable)
- Documentation (clear usage and design explanation)

## Next Section
[Thread Synchronization Mechanisms](03_Thread_Synchronization.md)
