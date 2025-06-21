# Debugging Threaded Applications

*Duration: 1 week*

## Overview

Debugging multi-threaded applications presents unique challenges due to the non-deterministic nature of concurrent execution. This section covers tools, techniques, and strategies for effectively debugging threading issues.

## Common Threading Bugs

### Race Conditions

Race conditions occur when multiple threads access shared data without proper synchronization, leading to unpredictable results.

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

// Example of a race condition
int global_counter = 0;

void* increment_counter(void* arg) {
    int iterations = *(int*)arg;
    
    for (int i = 0; i < iterations; i++) {
        // RACE CONDITION: Non-atomic read-modify-write
        int temp = global_counter;  // Read
        temp = temp + 1;            // Modify
        global_counter = temp;      // Write
    }
    
    return NULL;
}

void demonstrate_race_condition() {
    const int num_threads = 4;
    const int iterations = 100000;
    
    pthread_t threads[num_threads];
    int thread_iterations = iterations;
    
    printf("Starting race condition demonstration...\n");
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, increment_counter, &thread_iterations);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Expected result: %d\n", num_threads * iterations);
    printf("Actual result: %d\n", global_counter);
    printf("Lost updates: %d\n", (num_threads * iterations) - global_counter);
}

// Fixed version with proper synchronization
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment_counter_safe(void* arg) {
    int iterations = *(int*)arg;
    
    for (int i = 0; i < iterations; i++) {
        pthread_mutex_lock(&counter_mutex);
        global_counter++;
        pthread_mutex_unlock(&counter_mutex);
    }
    
    return NULL;
}
```

### Deadlocks

Deadlocks occur when threads wait for each other indefinitely, creating a circular dependency.

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void* thread1_function(void* arg) {
    printf("Thread 1: Acquiring mutex1\n");
    pthread_mutex_lock(&mutex1);
    printf("Thread 1: Got mutex1\n");
    
    sleep(1); // Increase chance of deadlock
    
    printf("Thread 1: Trying to acquire mutex2\n");
    pthread_mutex_lock(&mutex2); // Will block if thread2 has mutex2
    printf("Thread 1: Got mutex2\n");
    
    // Critical section
    printf("Thread 1: In critical section\n");
    
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    
    return NULL;
}

void* thread2_function(void* arg) {
    printf("Thread 2: Acquiring mutex2\n");
    pthread_mutex_lock(&mutex2);
    printf("Thread 2: Got mutex2\n");
    
    sleep(1); // Increase chance of deadlock
    
    printf("Thread 2: Trying to acquire mutex1\n");
    pthread_mutex_lock(&mutex1); // Will block if thread1 has mutex1
    printf("Thread 2: Got mutex1\n");
    
    // Critical section
    printf("Thread 2: In critical section\n");
    
    pthread_mutex_unlock(&mutex1);
    pthread_mutex_unlock(&mutex2);
    
    return NULL;
}

void demonstrate_deadlock() {
    pthread_t thread1, thread2;
    
    printf("Starting deadlock demonstration...\n");
    printf("(This may hang - use Ctrl+C to interrupt)\n");
    
    pthread_create(&thread1, NULL, thread1_function, NULL);
    pthread_create(&thread2, NULL, thread2_function, NULL);
    
    // Set a timeout for the join
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += 5; // 5 second timeout
    
    int result1 = pthread_timedjoin_np(thread1, NULL, &timeout);
    int result2 = pthread_timedjoin_np(thread2, NULL, &timeout);
    
    if (result1 == ETIMEDOUT || result2 == ETIMEDOUT) {
        printf("Deadlock detected! Threads timed out.\n");
        
        // Force termination (not recommended in production)
        pthread_cancel(thread1);
        pthread_cancel(thread2);
        pthread_join(thread1, NULL);
        pthread_join(thread2, NULL);
    } else {
        printf("No deadlock occurred this time.\n");
    }
}

// Fixed version using lock ordering
void* thread1_fixed(void* arg) {
    printf("Thread 1: Acquiring locks in order\n");
    
    // Always acquire mutex1 first, then mutex2
    pthread_mutex_lock(&mutex1);
    printf("Thread 1: Got mutex1\n");
    
    pthread_mutex_lock(&mutex2);
    printf("Thread 1: Got mutex2\n");
    
    printf("Thread 1: In critical section\n");
    
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    
    return NULL;
}

void* thread2_fixed(void* arg) {
    printf("Thread 2: Acquiring locks in order\n");
    
    // Always acquire mutex1 first, then mutex2
    pthread_mutex_lock(&mutex1);
    printf("Thread 2: Got mutex1\n");
    
    pthread_mutex_lock(&mutex2);
    printf("Thread 2: Got mutex2\n");
    
    printf("Thread 2: In critical section\n");
    
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    
    return NULL;
}
```

### Livelocks and Starvation

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

// Livelock example: threads keep backing off and retrying
pthread_mutex_t resource1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t resource2 = PTHREAD_MUTEX_INITIALIZER;

void* polite_thread1(void* arg) {
    for (int i = 0; i < 10; i++) {
        printf("Thread 1: Attempting to acquire resources\n");
        
        if (pthread_mutex_trylock(&resource1) == 0) {
            printf("Thread 1: Got resource1\n");
            
            if (pthread_mutex_trylock(&resource2) == 0) {
                printf("Thread 1: Got both resources, working...\n");
                sleep(1);
                pthread_mutex_unlock(&resource2);
                pthread_mutex_unlock(&resource1);
                break;
            } else {
                printf("Thread 1: Can't get resource2, releasing resource1\n");
                pthread_mutex_unlock(&resource1);
                usleep(rand() % 1000); // Random backoff
            }
        } else {
            printf("Thread 1: Can't get resource1\n");
            usleep(rand() % 1000);
        }
    }
    
    return NULL;
}

void* polite_thread2(void* arg) {
    for (int i = 0; i < 10; i++) {
        printf("Thread 2: Attempting to acquire resources\n");
        
        if (pthread_mutex_trylock(&resource2) == 0) {
            printf("Thread 2: Got resource2\n");
            
            if (pthread_mutex_trylock(&resource1) == 0) {
                printf("Thread 2: Got both resources, working...\n");
                sleep(1);
                pthread_mutex_unlock(&resource1);
                pthread_mutex_unlock(&resource2);
                break;
            } else {
                printf("Thread 2: Can't get resource1, releasing resource2\n");
                pthread_mutex_unlock(&resource2);
                usleep(rand() % 1000); // Random backoff
            }
        } else {
            printf("Thread 2: Can't get resource2\n");
            usleep(rand() % 1000);
        }
    }
    
    return NULL;
}

void demonstrate_livelock() {
    pthread_t thread1, thread2;
    
    printf("Starting livelock demonstration...\n");
    
    pthread_create(&thread1, NULL, polite_thread1, NULL);
    pthread_create(&thread2, NULL, polite_thread2, NULL);
    
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    
    printf("Livelock demonstration completed\n");
}
```

## Thread-Aware Debugging with GDB

### Basic GDB Threading Commands

```bash
# Compile with debug information
gcc -g -pthread -o threaded_program program.c

# Start GDB
gdb ./threaded_program

# Set breakpoints before thread creation
(gdb) break main
(gdb) run

# Thread-specific commands
(gdb) info threads          # List all threads
(gdb) thread 2              # Switch to thread 2
(gdb) thread apply all bt   # Show backtrace for all threads
(gdb) thread apply 2 3 bt  # Show backtrace for threads 2 and 3

# Set thread-specific breakpoints
(gdb) break worker_function thread 2

# Debug mutex states
(gdb) print mutex_variable
(gdb) print *mutex_variable
```

### GDB Debugging Example

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t debug_mutex = PTHREAD_MUTEX_INITIALIZER;
int shared_value = 0;

void* debug_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < 5; i++) {
        pthread_mutex_lock(&debug_mutex);
        
        int old_value = shared_value;
        sleep(1); // Deliberate delay for debugging
        shared_value = old_value + 1;
        
        printf("Thread %d: Updated shared_value from %d to %d\n", 
               thread_id, old_value, shared_value);
        
        pthread_mutex_unlock(&debug_mutex);
        sleep(1);
    }
    
    return NULL;
}

int main() {
    const int num_threads = 3;
    pthread_t threads[num_threads];
    int thread_ids[num_threads];
    
    printf("Starting debug example with %d threads\n", num_threads);
    
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i + 1;
        pthread_create(&threads[i], NULL, debug_worker, &thread_ids[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Final shared_value: %d\n", shared_value);
    return 0;
}
```

## Thread Sanitizers

### Using ThreadSanitizer (TSan)

ThreadSanitizer is a powerful tool for detecting race conditions and other threading bugs.

```bash
# Compile with ThreadSanitizer
gcc -fsanitize=thread -g -o program program.c -lpthread

# Run the program
./program
```

### Example with TSan Detection

```c
#include <stdio.h>
#include <pthread.h>

int racy_variable = 0;

void* writer_thread(void* arg) {
    for (int i = 0; i < 100000; i++) {
        racy_variable++; // Race condition here
    }
    return NULL;
}

void* reader_thread(void* arg) {
    for (int i = 0; i < 100000; i++) {
        volatile int temp = racy_variable; // Race condition here
        (void)temp; // Suppress unused variable warning
    }
    return NULL;
}

int main() {
    pthread_t writer, reader;
    
    pthread_create(&writer, NULL, writer_thread, NULL);
    pthread_create(&reader, NULL, reader_thread, NULL);
    
    pthread_join(writer, NULL);
    pthread_join(reader, NULL);
    
    printf("Final value: %d\n", racy_variable);
    return 0;
}
```

### Helgrind (Valgrind's Thread Checker)

```bash
# Install valgrind
sudo apt-get install valgrind

# Run with Helgrind
valgrind --tool=helgrind ./program

# Run with detailed output
valgrind --tool=helgrind --read-var-info=yes ./program
```

## Custom Debugging Tools

### Thread State Monitor

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_THREADS 10
#define MAX_NAME_LEN 32

typedef enum {
    THREAD_CREATED,
    THREAD_RUNNING,
    THREAD_WAITING,
    THREAD_FINISHED
} ThreadState;

typedef struct {
    pthread_t thread_id;
    char name[MAX_NAME_LEN];
    ThreadState state;
    time_t last_update;
    void* current_function;
} ThreadInfo;

static ThreadInfo thread_registry[MAX_THREADS];
static int thread_count = 0;
static pthread_mutex_t registry_mutex = PTHREAD_MUTEX_INITIALIZER;

int register_debug_thread(const char* name) {
    pthread_mutex_lock(&registry_mutex);
    
    if (thread_count >= MAX_THREADS) {
        pthread_mutex_unlock(&registry_mutex);
        return -1;
    }
    
    int id = thread_count++;
    thread_registry[id].thread_id = pthread_self();
    strncpy(thread_registry[id].name, name, MAX_NAME_LEN - 1);
    thread_registry[id].state = THREAD_CREATED;
    thread_registry[id].last_update = time(NULL);
    
    pthread_mutex_unlock(&registry_mutex);
    return id;
}

void update_thread_state(int thread_id, ThreadState state) {
    if (thread_id < 0 || thread_id >= thread_count) return;
    
    pthread_mutex_lock(&registry_mutex);
    thread_registry[thread_id].state = state;
    thread_registry[thread_id].last_update = time(NULL);
    pthread_mutex_unlock(&registry_mutex);
}

void print_thread_status() {
    pthread_mutex_lock(&registry_mutex);
    
    printf("\n=== Thread Status ===\n");
    printf("%-15s %-12s %-10s %s\n", "Name", "Thread ID", "State", "Last Update");
    printf("%.60s\n", "------------------------------------------------------------");
    
    for (int i = 0; i < thread_count; i++) {
        const char* state_str;
        switch (thread_registry[i].state) {
            case THREAD_CREATED:  state_str = "Created"; break;
            case THREAD_RUNNING:  state_str = "Running"; break;
            case THREAD_WAITING:  state_str = "Waiting"; break;
            case THREAD_FINISHED: state_str = "Finished"; break;
            default: state_str = "Unknown"; break;
        }
        
        printf("%-15s %-12lu %-10s %ld\n",
               thread_registry[i].name,
               (unsigned long)thread_registry[i].thread_id,
               state_str,
               thread_registry[i].last_update);
    }
    
    pthread_mutex_unlock(&registry_mutex);
}

// Example worker function with debug support
void* debug_monitored_worker(void* arg) {
    char* name = (char*)arg;
    int thread_id = register_debug_thread(name);
    
    update_thread_state(thread_id, THREAD_RUNNING);
    
    for (int i = 0; i < 5; i++) {
        printf("%s: Working iteration %d\n", name, i + 1);
        sleep(2);
        
        if (i == 2) {
            update_thread_state(thread_id, THREAD_WAITING);
            printf("%s: Simulating wait condition\n", name);
            sleep(3);
            update_thread_state(thread_id, THREAD_RUNNING);
        }
    }
    
    update_thread_state(thread_id, THREAD_FINISHED);
    return NULL;
}

void demonstrate_thread_monitoring() {
    pthread_t threads[3];
    char* names[] = {"Worker-1", "Worker-2", "Worker-3"};
    
    // Create monitoring threads
    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, debug_monitored_worker, names[i]);
    }
    
    // Monitor thread status
    for (int i = 0; i < 10; i++) {
        sleep(2);
        print_thread_status();
    }
    
    // Wait for completion
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }
    
    print_thread_status();
}
```

### Deadlock Detection Tool

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAX_LOCKS 10
#define MAX_THREADS 10

typedef struct {
    pthread_mutex_t* mutex;
    pthread_t owner;
    time_t acquired_time;
    char name[32];
} LockInfo;

typedef struct {
    pthread_t thread_id;
    pthread_mutex_t* waiting_for;
    pthread_mutex_t* owned_locks[MAX_LOCKS];
    int owned_count;
} ThreadLockInfo;

static LockInfo lock_registry[MAX_LOCKS];
static ThreadLockInfo thread_locks[MAX_THREADS];
static int lock_count = 0;
static int thread_lock_count = 0;
static pthread_mutex_t detector_mutex = PTHREAD_MUTEX_INITIALIZER;

void register_lock(pthread_mutex_t* mutex, const char* name) {
    pthread_mutex_lock(&detector_mutex);
    
    if (lock_count < MAX_LOCKS) {
        lock_registry[lock_count].mutex = mutex;
        lock_registry[lock_count].owner = 0;
        lock_registry[lock_count].acquired_time = 0;
        strncpy(lock_registry[lock_count].name, name, 31);
        lock_count++;
    }
    
    pthread_mutex_unlock(&detector_mutex);
}

void record_lock_acquisition(pthread_mutex_t* mutex) {
    pthread_mutex_lock(&detector_mutex);
    
    pthread_t current_thread = pthread_self();
    
    // Update lock registry
    for (int i = 0; i < lock_count; i++) {
        if (lock_registry[i].mutex == mutex) {
            lock_registry[i].owner = current_thread;
            lock_registry[i].acquired_time = time(NULL);
            break;
        }
    }
    
    // Update thread lock info
    for (int i = 0; i < thread_lock_count; i++) {
        if (thread_locks[i].thread_id == current_thread) {
            if (thread_locks[i].owned_count < MAX_LOCKS) {
                thread_locks[i].owned_locks[thread_locks[i].owned_count++] = mutex;
            }
            thread_locks[i].waiting_for = NULL;
            break;
        }
    }
    
    pthread_mutex_unlock(&detector_mutex);
}

void record_lock_wait(pthread_mutex_t* mutex) {
    pthread_mutex_lock(&detector_mutex);
    
    pthread_t current_thread = pthread_self();
    
    // Find or create thread entry
    int thread_index = -1;
    for (int i = 0; i < thread_lock_count; i++) {
        if (thread_locks[i].thread_id == current_thread) {
            thread_index = i;
            break;
        }
    }
    
    if (thread_index == -1 && thread_lock_count < MAX_THREADS) {
        thread_index = thread_lock_count++;
        thread_locks[thread_index].thread_id = current_thread;
        thread_locks[thread_index].owned_count = 0;
    }
    
    if (thread_index != -1) {
        thread_locks[thread_index].waiting_for = mutex;
    }
    
    pthread_mutex_unlock(&detector_mutex);
}

void detect_potential_deadlock() {
    pthread_mutex_lock(&detector_mutex);
    
    printf("\n=== Potential Deadlock Detection ===\n");
    
    for (int i = 0; i < thread_lock_count; i++) {
        if (thread_locks[i].waiting_for == NULL) continue;
        
        pthread_t waiting_thread = thread_locks[i].thread_id;
        pthread_mutex_t* waiting_mutex = thread_locks[i].waiting_for;
        
        // Find who owns the mutex this thread is waiting for
        pthread_t blocking_thread = 0;
        for (int j = 0; j < lock_count; j++) {
            if (lock_registry[j].mutex == waiting_mutex) {
                blocking_thread = lock_registry[j].owner;
                break;
            }
        }
        
        if (blocking_thread == 0) continue;
        
        // Check if blocking thread is waiting for any mutex owned by waiting thread
        for (int j = 0; j < thread_lock_count; j++) {
            if (thread_locks[j].thread_id != blocking_thread) continue;
            
            if (thread_locks[j].waiting_for != NULL) {
                // Check if blocking thread is waiting for a mutex owned by waiting thread
                for (int k = 0; k < thread_locks[i].owned_count; k++) {
                    if (thread_locks[i].owned_locks[k] == thread_locks[j].waiting_for) {
                        printf("POTENTIAL DEADLOCK DETECTED:\n");
                        printf("  Thread %lu waiting for mutex owned by Thread %lu\n",
                               (unsigned long)waiting_thread, (unsigned long)blocking_thread);
                        printf("  Thread %lu waiting for mutex owned by Thread %lu\n",
                               (unsigned long)blocking_thread, (unsigned long)waiting_thread);
                    }
                }
            }
        }
    }
    
    pthread_mutex_unlock(&detector_mutex);
}

// Wrapper functions for instrumented locking
int debug_mutex_lock(pthread_mutex_t* mutex) {
    record_lock_wait(mutex);
    int result = pthread_mutex_lock(mutex);
    if (result == 0) {
        record_lock_acquisition(mutex);
    }
    return result;
}

int debug_mutex_unlock(pthread_mutex_t* mutex) {
    // Remove from owned locks
    pthread_mutex_lock(&detector_mutex);
    pthread_t current_thread = pthread_self();
    
    for (int i = 0; i < thread_lock_count; i++) {
        if (thread_locks[i].thread_id == current_thread) {
            for (int j = 0; j < thread_locks[i].owned_count; j++) {
                if (thread_locks[i].owned_locks[j] == mutex) {
                    // Shift remaining locks
                    for (int k = j; k < thread_locks[i].owned_count - 1; k++) {
                        thread_locks[i].owned_locks[k] = thread_locks[i].owned_locks[k + 1];
                    }
                    thread_locks[i].owned_count--;
                    break;
                }
            }
            break;
        }
    }
    pthread_mutex_unlock(&detector_mutex);
    
    return pthread_mutex_unlock(mutex);
}
```

## Logging and Trace Analysis

### Thread-Safe Logging System

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>

typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
} LogLevel;

typedef struct {
    FILE* log_file;
    LogLevel min_level;
    pthread_mutex_t log_mutex;
    char buffer[4096];
} ThreadLogger;

static ThreadLogger global_logger = {
    .log_file = NULL,
    .min_level = LOG_INFO,
    .log_mutex = PTHREAD_MUTEX_INITIALIZER
};

void init_thread_logger(const char* filename, LogLevel min_level) {
    pthread_mutex_lock(&global_logger.log_mutex);
    
    if (filename) {
        global_logger.log_file = fopen(filename, "a");
    } else {
        global_logger.log_file = stdout;
    }
    
    global_logger.min_level = min_level;
    
    pthread_mutex_unlock(&global_logger.log_mutex);
}

void thread_log(LogLevel level, const char* format, ...) {
    if (level < global_logger.min_level) return;
    
    pthread_mutex_lock(&global_logger.log_mutex);
    
    // Get timestamp
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);
    
    // Get thread ID
    pthread_t thread_id = pthread_self();
    
    // Level string
    const char* level_str;
    switch (level) {
        case LOG_DEBUG: level_str = "DEBUG"; break;
        case LOG_INFO:  level_str = "INFO"; break;
        case LOG_WARN:  level_str = "WARN"; break;
        case LOG_ERROR: level_str = "ERROR"; break;
        default: level_str = "UNKNOWN"; break;
    }
    
    // Format message
    va_list args;
    va_start(args, format);
    vsnprintf(global_logger.buffer, sizeof(global_logger.buffer), format, args);
    va_end(args);
    
    // Write log entry
    fprintf(global_logger.log_file, "[%s] [%s] [Thread %lu] %s\n",
            time_str, level_str, (unsigned long)thread_id, global_logger.buffer);
    
    fflush(global_logger.log_file);
    
    pthread_mutex_unlock(&global_logger.log_mutex);
}

#define LOG_DEBUG(...) thread_log(LOG_DEBUG, __VA_ARGS__)
#define LOG_INFO(...)  thread_log(LOG_INFO, __VA_ARGS__)
#define LOG_WARN(...)  thread_log(LOG_WARN, __VA_ARGS__)
#define LOG_ERROR(...) thread_log(LOG_ERROR, __VA_ARGS__)

// Example usage
void* logging_worker(void* arg) {
    int worker_id = *(int*)arg;
    
    LOG_INFO("Worker %d starting up", worker_id);
    
    for (int i = 0; i < 5; i++) {
        LOG_DEBUG("Worker %d: Processing item %d", worker_id, i);
        
        // Simulate some work
        sleep(1);
        
        if (i == 2) {
            LOG_WARN("Worker %d: Encountered minor issue at item %d", worker_id, i);
        }
    }
    
    LOG_INFO("Worker %d completed successfully", worker_id);
    return NULL;
}

void demonstrate_thread_logging() {
    init_thread_logger("thread_debug.log", LOG_DEBUG);
    
    const int num_workers = 3;
    pthread_t workers[num_workers];
    int worker_ids[num_workers];
    
    LOG_INFO("Starting %d worker threads", num_workers);
    
    for (int i = 0; i < num_workers; i++) {
        worker_ids[i] = i + 1;
        pthread_create(&workers[i], NULL, logging_worker, &worker_ids[i]);
    }
    
    for (int i = 0; i < num_workers; i++) {
        pthread_join(workers[i], NULL);
    }
    
    LOG_INFO("All workers completed");
}
```

## Debugging Exercises

### Exercise 1: Race Condition Detective

```c
// Find and fix the race conditions in this code
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

typedef struct {
    int* array;
    int size;
    int sum;
} ArraySum;

ArraySum global_sum = {NULL, 0, 0};

void* calculate_sum(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = thread_id; i < global_sum.size; i += 2) {
        global_sum.sum += global_sum.array[i]; // Race condition!
    }
    
    return NULL;
}

int main() {
    const int size = 1000;
    global_sum.array = malloc(size * sizeof(int));
    global_sum.size = size;
    
    for (int i = 0; i < size; i++) {
        global_sum.array[i] = i + 1;
    }
    
    pthread_t threads[2];
    int thread_ids[2] = {0, 1};
    
    for (int i = 0; i < 2; i++) {
        pthread_create(&threads[i], NULL, calculate_sum, &thread_ids[i]);
    }
    
    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Sum: %d\n", global_sum.sum);
    printf("Expected: %d\n", size * (size + 1) / 2);
    
    free(global_sum.array);
    return 0;
}
```

## Assessment

You should be able to:
- Identify common threading bugs (race conditions, deadlocks, livelocks)
- Use GDB effectively for multi-threaded debugging
- Apply ThreadSanitizer and other tools for bug detection
- Create custom debugging and monitoring tools
- Implement thread-safe logging systems
- Analyze and fix complex threading issues
- Design debugging strategies for large multi-threaded applications

## Next Section
[Project Examples](Projects/)
