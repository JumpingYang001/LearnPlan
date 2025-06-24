# Thread Local Storage

*Duration: 1 week*

## Overview

Thread Local Storage (TLS) allows each thread to have its own private copy of data, eliminating the need for synchronization when accessing thread-specific information. This is particularly useful for maintaining per-thread state and avoiding global variable conflicts.

## Concepts

### What is Thread Local Storage?
- Each thread gets its own copy of TLS variables
- No synchronization needed for thread-specific data
- Useful for per-thread state management
- Alternative to passing data through function parameters

### Use Cases
- Error codes and errno handling
- Per-thread caches and buffers
- Thread-specific configuration
- Performance counters per thread
- Random number generators with per-thread seeds

## POSIX Thread-Specific Data

### Key Management Functions

```c
#include <pthread.h>

// Create a new key
int pthread_key_create(pthread_key_t *key, void (*destructor)(void*));

// Delete a key
int pthread_key_delete(pthread_key_t key);

// Set thread-specific value
int pthread_setspecific(pthread_key_t key, const void *value);

// Get thread-specific value
void *pthread_getspecific(pthread_key_t key);
```

### Basic Usage Pattern

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

pthread_key_t thread_data_key;

// Destructor function called when thread exits
void cleanup_thread_data(void* data) {
    printf("Cleaning up thread data: %p\n", data);
    free(data);
}

void initialize_tls() {
    if (pthread_key_create(&thread_data_key, cleanup_thread_data) != 0) {
        fprintf(stderr, "Failed to create thread key\n");
        exit(1);
    }
}

void set_thread_data(int value) {
    int* data = malloc(sizeof(int));
    *data = value;
    pthread_setspecific(thread_data_key, data);
}

int get_thread_data() {
    int* data = (int*)pthread_getspecific(thread_data_key);
    return data ? *data : -1;
}

void* thread_function(void* arg) {
    int thread_id = *(int*)arg;
    
    // Set thread-specific data
    set_thread_data(thread_id * 100);
    
    printf("Thread %d: Set data to %d\n", thread_id, get_thread_data());
    
    // Simulate some work
    for (int i = 0; i < 5; i++) {
        printf("Thread %d: Data is %d\n", thread_id, get_thread_data());
        sleep(1);
    }
    
    return NULL;
}

int main() {
    initialize_tls();
    
    pthread_t threads[3];
    int thread_ids[3] = {1, 2, 3};
    
    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, thread_function, &thread_ids[i]);
    }
    
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }
    
    pthread_key_delete(thread_data_key);
    return 0;
}
```

## Advanced TLS Patterns

### Thread-Specific Error Handling

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MAX_ERROR_MSG 256

typedef struct {
    int error_code;
    char error_message[MAX_ERROR_MSG];
} ThreadError;

pthread_key_t error_key;

void cleanup_error_data(void* data) {
    free(data);
}

void init_error_system() {
    pthread_key_create(&error_key, cleanup_error_data);
}

void set_thread_error(int code, const char* message) {
    ThreadError* error = (ThreadError*)pthread_getspecific(error_key);
    
    if (!error) {
        error = malloc(sizeof(ThreadError));
        pthread_setspecific(error_key, error);
    }
    
    error->error_code = code;
    strncpy(error->error_message, message, MAX_ERROR_MSG - 1);
    error->error_message[MAX_ERROR_MSG - 1] = '\0';
}

int get_thread_error_code() {
    ThreadError* error = (ThreadError*)pthread_getspecific(error_key);
    return error ? error->error_code : 0;
}

const char* get_thread_error_message() {
    ThreadError* error = (ThreadError*)pthread_getspecific(error_key);
    return error ? error->error_message : "No error";
}

// Example API functions that set thread-specific errors
int divide_numbers(int a, int b, double* result) {
    if (b == 0) {
        set_thread_error(1, "Division by zero");
        return -1;
    }
    
    *result = (double)a / b;
    set_thread_error(0, "Success");
    return 0;
}

void* worker_thread(void* arg) {
    int thread_id = *(int*)arg;
    double result;
    
    // Test successful operation
    if (divide_numbers(10, 2, &result) == 0) {
        printf("Thread %d: 10/2 = %.2f (Error: %s)\n", 
               thread_id, result, get_thread_error_message());
    }
    
    // Test error condition
    if (divide_numbers(10, 0, &result) != 0) {
        printf("Thread %d: Error %d: %s\n", 
               thread_id, get_thread_error_code(), get_thread_error_message());
    }
    
    return NULL;
}
```

### Thread-Specific Random Number Generator

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

pthread_key_t rng_key;

typedef struct {
    unsigned int seed;
    int initialized;
} ThreadRNG;

void cleanup_rng(void* data) {
    free(data);
}

void init_rng_system() {
    pthread_key_create(&rng_key, cleanup_rng);
}

void init_thread_rng() {
    ThreadRNG* rng = (ThreadRNG*)pthread_getspecific(rng_key);
    
    if (!rng) {
        rng = malloc(sizeof(ThreadRNG));
        rng->seed = time(NULL) ^ (pthread_self() << 16);
        rng->initialized = 1;
        pthread_setspecific(rng_key, rng);
        printf("Thread %lu: Initialized RNG with seed %u\n", 
               pthread_self(), rng->seed);
    }
}

int thread_rand() {
    ThreadRNG* rng = (ThreadRNG*)pthread_getspecific(rng_key);
    
    if (!rng || !rng->initialized) {
        init_thread_rng();
        rng = (ThreadRNG*)pthread_getspecific(rng_key);
    }
    
    return rand_r(&rng->seed);
}

void* random_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    printf("Thread %d: Random numbers: ", thread_id);
    for (int i = 0; i < 5; i++) {
        printf("%d ", thread_rand() % 100);
    }
    printf("\n");
    
    return NULL;
}
```

### Thread-Specific Performance Counters

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

typedef struct {
    unsigned long operations_count;
    clock_t start_time;
    double total_time;
} ThreadStats;

pthread_key_t stats_key;

void cleanup_stats(void* data) {
    ThreadStats* stats = (ThreadStats*)data;
    printf("Thread statistics:\n");
    printf("  Operations: %lu\n", stats->operations_count);
    printf("  Total time: %.6f seconds\n", stats->total_time);
    printf("  Ops/sec: %.2f\n", stats->operations_count / stats->total_time);
    free(data);
}

void init_stats_system() {
    pthread_key_create(&stats_key, cleanup_stats);
}

ThreadStats* get_thread_stats() {
    ThreadStats* stats = (ThreadStats*)pthread_getspecific(stats_key);
    
    if (!stats) {
        stats = calloc(1, sizeof(ThreadStats));
        stats->start_time = clock();
        pthread_setspecific(stats_key, stats);
    }
    
    return stats;
}

void increment_operation_count() {
    ThreadStats* stats = get_thread_stats();
    stats->operations_count++;
}

void update_thread_time() {
    ThreadStats* stats = get_thread_stats();
    clock_t end_time = clock();
    stats->total_time = ((double)(end_time - stats->start_time)) / CLOCKS_PER_SEC;
}

void* performance_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    // Simulate work with performance tracking
    for (int i = 0; i < 100000; i++) {
        increment_operation_count();
        
        // Simulate some computation
        volatile int sum = 0;
        for (int j = 0; j < 100; j++) {
            sum += j;
        }
    }
    
    update_thread_time();
    printf("Thread %d completed work\n", thread_id);
    
    return NULL;
}
```

## Compiler-Specific TLS

### GCC `__thread` Keyword

```c
#include <stdio.h>
#include <pthread.h>

// Thread-local variable using __thread
__thread int thread_local_counter = 0;
__thread char thread_local_buffer[256];

void* thread_work(void* arg) {
    int thread_id = *(int*)arg;
    
    // Each thread has its own copy
    thread_local_counter = thread_id * 10;
    snprintf(thread_local_buffer, sizeof(thread_local_buffer), 
             "Thread %d data", thread_id);
    
    for (int i = 0; i < 5; i++) {
        thread_local_counter++;
        printf("Thread %d: counter = %d, buffer = %s\n", 
               thread_id, thread_local_counter, thread_local_buffer);
        sleep(1);
    }
    
    return NULL;
}
```

### C11 `_Thread_local` Keyword

```c
#include <stdio.h>
#include <threads.h>

_Thread_local int tls_value = 0;

int thread_func(void* arg) {
    int thread_id = *(int*)arg;
    
    tls_value = thread_id * 100;
    
    printf("Thread %d: TLS value = %d\n", thread_id, tls_value);
    
    return 0;
}
```

## Best Practices

### 1. Always Use Destructors for Dynamic Memory

```c
void* thread_data_destructor(void* data) {
    // Clean up any allocated resources
    if (data) {
        free(data);
    }
}

pthread_key_create(&key, thread_data_destructor);
```

### 2. Initialize Keys Once

```c
pthread_once_t key_once = PTHREAD_ONCE_INIT;
pthread_key_t global_key;

void init_key() {
    pthread_key_create(&global_key, cleanup_function);
}

void ensure_key_initialized() {
    pthread_once(&key_once, init_key);
}
```

### 3. Handle NULL Returns

```c
int get_thread_value() {
    int* value = (int*)pthread_getspecific(key);
    return value ? *value : DEFAULT_VALUE;
}
```

### 4. Consider Static TLS for Simple Cases

```c
// Simpler for basic types
__thread int simple_counter = 0;

// vs complex key management for the same purpose
```

## Common Use Cases and Examples

### 1. Thread-Safe errno Implementation

```c
__thread int thread_errno = 0;

#define set_errno(val) (thread_errno = (val))
#define get_errno() (thread_errno)
```

### 2. Per-Thread Memory Pool

```c
typedef struct {
    void* pool;
    size_t size;
    size_t used;
} ThreadMemPool;

pthread_key_t mempool_key;

void cleanup_mempool(void* data) {
    ThreadMemPool* pool = (ThreadMemPool*)data;
    free(pool->pool);
    free(pool);
}

ThreadMemPool* get_thread_mempool() {
    ThreadMemPool* pool = pthread_getspecific(mempool_key);
    if (!pool) {
        pool = malloc(sizeof(ThreadMemPool));
        pool->pool = malloc(4096); // 4KB pool
        pool->size = 4096;
        pool->used = 0;
        pthread_setspecific(mempool_key, pool);
    }
    return pool;
}
```

### 3. Thread-Specific Configuration

```c
typedef struct {
    int debug_level;
    char log_prefix[64];
    int max_retries;
} ThreadConfig;

__thread ThreadConfig config = {
    .debug_level = 0,
    .log_prefix = "",
    .max_retries = 3
};
```

## Performance Considerations

### TLS Access Performance
- Compiler-specific TLS (`__thread`) is typically faster than pthread keys
- pthread keys involve function calls and hash table lookups
- Static TLS allocation is fastest

### Memory Usage
- Each thread gets its own copy of TLS variables
- Consider memory usage with many threads
- Use destructors to prevent memory leaks

## Exercises

1. **Thread-Specific Logger**
   - Implement a logging system where each thread has its own log buffer
   - Flush buffers periodically or on thread termination

2. **Per-Thread Connection Pool**
   - Create a database connection pool per thread
   - Each thread maintains its own connections

3. **Thread-Local Cache**
   - Implement a simple cache using TLS
   - Compare performance with shared cache

4. **Error Handling System**
   - Design a comprehensive error handling system using TLS
   - Support error codes, messages, and stack traces

## Common Pitfalls

1. **Forgetting Destructors**
   ```c
   // Memory leak - no destructor
   pthread_key_create(&key, NULL);
   ```

2. **Accessing TLS After Thread Exit**
   ```c
   // Invalid - thread has exited
   void* data = pthread_getspecific(key);
   ```

3. **Not Checking for NULL**
   ```c
   // Potential segfault
   int value = *(int*)pthread_getspecific(key);
   ```

## Assessment

You should be able to:
- Implement thread-specific data using pthread keys
- Create proper destructors for cleanup
- Use compiler-specific TLS keywords effectively
- Design thread-local storage patterns for common use cases
- Understand performance implications of different TLS approaches
- Debug TLS-related memory issues

## Next Section
[Advanced Threading Patterns](06_Advanced_Threading_Patterns.md)
