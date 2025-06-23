# Thread Local Storage: Per-Thread Data Management

*Duration: 1 week*

## Overview

Thread Local Storage (TLS) is a sophisticated mechanism that provides each thread with its own private copy of data, eliminating the need for synchronization when accessing thread-specific information. This powerful feature enables efficient per-thread state management, improves performance by avoiding contention, and simplifies many multi-threaded programming patterns.

### The Problem TLS Solves

In multi-threaded applications, sharing global variables leads to race conditions:

```c
// PROBLEMATIC: Global variable shared by all threads
int global_counter = 0;
char global_buffer[256];

void* thread_function(void* arg) {
    // Race condition: multiple threads accessing same variables
    global_counter++;  // Needs synchronization
    sprintf(global_buffer, "Thread data"); // Potential corruption
    return NULL;
}
```

Traditional solutions require synchronization overhead:

```c
// TRADITIONAL: Synchronization approach
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;
int shared_counter = 0;

void* thread_function(void* arg) {
    pthread_mutex_lock(&counter_mutex);  // Contention point
    shared_counter++;
    pthread_mutex_unlock(&counter_mutex);
    return NULL;
}
```

TLS provides elegant solution without synchronization:

```c
// ELEGANT: Thread Local Storage approach
__thread int thread_counter = 0;      // Each thread has its own copy
__thread char thread_buffer[256];     // No contention, no synchronization needed

void* thread_function(void* arg) {
    thread_counter++;  // No synchronization needed!
    sprintf(thread_buffer, "Thread %lu data", pthread_self());
    return NULL;
}
```

### Benefits of Thread Local Storage

1. **Performance**: No synchronization overhead for thread-specific data
2. **Simplicity**: Cleaner code without explicit locking mechanisms
3. **Scalability**: No contention between threads accessing their own data
4. **Safety**: Automatic isolation prevents many race conditions
5. **Convenience**: Transparent access to per-thread state

### TLS Implementation Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Thread Local Storage Methods                 │
├─────────────────────────────────────────────────────────────────┤
│  Method          │  Performance  │  Portability  │  Flexibility │
├─────────────────────────────────────────────────────────────────┤
│  __thread        │  Excellent    │  GCC/Clang    │  Limited     │
│  _Thread_local   │  Excellent    │  C11 Standard │  Limited     │
│  pthread_key_*   │  Good         │  POSIX        │  High        │
│  compiler ext.   │  Variable     │  Specific     │  High        │
└─────────────────────────────────────────────────────────────────┘
```

## Advanced POSIX Thread-Specific Data

### Comprehensive Key Management Framework

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <assert.h>

// Enhanced TLS key management with error handling and statistics
typedef struct {
    pthread_key_t key;
    const char* name;
    void (*destructor)(void*);
    int active;
    
    // Statistics
    int set_count;
    int get_count;
    int destruction_count;
    struct timespec creation_time;
} tls_key_info_t;

#define MAX_TLS_KEYS 32
static tls_key_info_t tls_keys[MAX_TLS_KEYS];
static int tls_key_count = 0;
static pthread_mutex_t tls_mgmt_mutex = PTHREAD_MUTEX_INITIALIZER;

// Enhanced destructor wrapper for statistics
void tls_destructor_wrapper(void* data) {
    // Find the key info
    pthread_key_t current_key = 0; // We'll need to track this differently
    
    for (int i = 0; i < tls_key_count; i++) {
        if (tls_keys[i].active) {
            // Call original destructor
            if (tls_keys[i].destructor) {
                tls_keys[i].destructor(data);
            }
            
            pthread_mutex_lock(&tls_mgmt_mutex);
            tls_keys[i].destruction_count++;
            pthread_mutex_unlock(&tls_mgmt_mutex);
            
            printf("TLS: Destroyed data for key '%s' (total destructions: %d)\n",
                   tls_keys[i].name, tls_keys[i].destruction_count);
            break;
        }
    }
}

// Create TLS key with enhanced management
int tls_key_create(pthread_key_t* key, const char* name, void (*destructor)(void*)) {
    if (!key || !name) {
        errno = EINVAL;
        return -1;
    }
    
    pthread_mutex_lock(&tls_mgmt_mutex);
    
    if (tls_key_count >= MAX_TLS_KEYS) {
        pthread_mutex_unlock(&tls_mgmt_mutex);
        errno = EAGAIN;
        fprintf(stderr, "TLS: Maximum number of keys (%d) exceeded\n", MAX_TLS_KEYS);
        return -1;
    }
    
    int result = pthread_key_create(key, destructor);
    if (result != 0) {
        pthread_mutex_unlock(&tls_mgmt_mutex);
        fprintf(stderr, "TLS: Failed to create key '%s': %s\n", name, strerror(result));
        errno = result;
        return -1;
    }
    
    // Store key information
    tls_key_info_t* info = &tls_keys[tls_key_count];
    info->key = *key;
    info->name = strdup(name);
    info->destructor = destructor;
    info->active = 1;
    info->set_count = 0;
    info->get_count = 0;
    info->destruction_count = 0;
    clock_gettime(CLOCK_REALTIME, &info->creation_time);
    
    tls_key_count++;
    
    pthread_mutex_unlock(&tls_mgmt_mutex);
    
    printf("TLS: Created key '%s' (total keys: %d)\n", name, tls_key_count);
    return 0;
}

// Enhanced setspecific with error handling and statistics
int tls_setspecific(pthread_key_t key, const void* value, const char* context) {
    int result = pthread_setspecific(key, value);
    if (result != 0) {
        fprintf(stderr, "TLS: Failed to set value in %s: %s\n", 
                context ? context : "unknown", strerror(result));
        return -1;
    }
    
    // Update statistics
    pthread_mutex_lock(&tls_mgmt_mutex);
    for (int i = 0; i < tls_key_count; i++) {
        if (tls_keys[i].active && tls_keys[i].key == key) {
            tls_keys[i].set_count++;
            printf("TLS: Set value for key '%s' in %s (total sets: %d)\n",
                   tls_keys[i].name, context ? context : "unknown", tls_keys[i].set_count);
            break;
        }
    }
    pthread_mutex_unlock(&tls_mgmt_mutex);
    
    return 0;
}

// Enhanced getspecific with error handling and statistics
void* tls_getspecific(pthread_key_t key, const char* context) {
    void* value = pthread_getspecific(key);
    
    // Update statistics
    pthread_mutex_lock(&tls_mgmt_mutex);
    for (int i = 0; i < tls_key_count; i++) {
        if (tls_keys[i].active && tls_keys[i].key == key) {
            tls_keys[i].get_count++;
            if (context) {
                printf("TLS: Retrieved value for key '%s' in %s (total gets: %d)\n",
                       tls_keys[i].name, context, tls_keys[i].get_count);
            }
            break;
        }
    }
    pthread_mutex_unlock(&tls_mgmt_mutex);
    
    return value;
}

// Delete TLS key with cleanup
int tls_key_delete(pthread_key_t key) {
    int result = pthread_key_delete(key);
    if (result != 0) {
        fprintf(stderr, "TLS: Failed to delete key: %s\n", strerror(result));
        return -1;
    }
    
    pthread_mutex_lock(&tls_mgmt_mutex);
    for (int i = 0; i < tls_key_count; i++) {
        if (tls_keys[i].active && tls_keys[i].key == key) {
            printf("TLS: Deleted key '%s' (sets: %d, gets: %d, destructions: %d)\n",
                   tls_keys[i].name, tls_keys[i].set_count, 
                   tls_keys[i].get_count, tls_keys[i].destruction_count);
            
            free((void*)tls_keys[i].name);
            tls_keys[i].active = 0;
            break;
        }
    }
    pthread_mutex_unlock(&tls_mgmt_mutex);
    
    return 0;
}

// Get TLS system statistics
void tls_get_statistics() {
    pthread_mutex_lock(&tls_mgmt_mutex);
    
    printf("\n=== Thread Local Storage Statistics ===\n");
    printf("Total keys created: %d\n", tls_key_count);
    
    for (int i = 0; i < tls_key_count; i++) {
        if (tls_keys[i].active) {
            printf("\nKey: %s\n", tls_keys[i].name);
            printf("  Set operations: %d\n", tls_keys[i].set_count);
            printf("  Get operations: %d\n", tls_keys[i].get_count);
            printf("  Destructions: %d\n", tls_keys[i].destruction_count);
            
            struct timespec now;
            clock_gettime(CLOCK_REALTIME, &now);
            double age = (now.tv_sec - tls_keys[i].creation_time.tv_sec) +
                        (now.tv_nsec - tls_keys[i].creation_time.tv_nsec) / 1e9;
            printf("  Age: %.3f seconds\n", age);
        }
    }
    
    pthread_mutex_unlock(&tls_mgmt_mutex);
}
```

### Advanced Thread-Specific Error Handling System

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdarg.h>
#include <time.h>

#define MAX_ERROR_MSG 512
#define MAX_ERROR_STACK 10

typedef struct error_entry {
    int error_code;
    char error_message[MAX_ERROR_MSG];
    char function_name[64];
    char file_name[64];
    int line_number;
    struct timespec timestamp;
} error_entry_t;

typedef struct {
    error_entry_t error_stack[MAX_ERROR_STACK];
    int stack_depth;
    int total_errors;
    int warning_count;
    struct timespec first_error_time;
} thread_error_context_t;

static pthread_key_t error_context_key;
static pthread_once_t error_system_init = PTHREAD_ONCE_INIT;

// Destructor for error context
void cleanup_error_context(void* data) {
    thread_error_context_t* context = (thread_error_context_t*)data;
    
    if (context->total_errors > 0) {
        printf("Thread %lu error summary: %d errors, %d warnings\n",
               (unsigned long)pthread_self(), 
               context->total_errors, context->warning_count);
    }
    
    free(data);
}

// Initialize error handling system
void init_error_system() {
    if (tls_key_create(&error_context_key, "error_context", cleanup_error_context) != 0) {
        fprintf(stderr, "Failed to initialize error handling system\n");
        exit(1);
    }
    printf("Thread-local error handling system initialized\n");
}

// Get or create error context for current thread
thread_error_context_t* get_error_context() {
    pthread_once(&error_system_init, init_error_system);
    
    thread_error_context_t* context = 
        (thread_error_context_t*)tls_getspecific(error_context_key, "get_error_context");
    
    if (!context) {
        context = calloc(1, sizeof(thread_error_context_t));
        if (!context) {
            fprintf(stderr, "Failed to allocate error context\n");
            return NULL;
        }
        
        clock_gettime(CLOCK_REALTIME, &context->first_error_time);
        
        if (tls_setspecific(error_context_key, context, "get_error_context") != 0) {
            free(context);
            return NULL;
        }
        
        printf("Thread %lu: Created error context\n", (unsigned long)pthread_self());
    }
    
    return context;
}

// Set detailed error information
void set_thread_error_detailed(int code, const char* function, const char* file, 
                               int line, const char* format, ...) {
    thread_error_context_t* context = get_error_context();
    if (!context) return;
    
    // Shift stack if full
    if (context->stack_depth >= MAX_ERROR_STACK) {
        memmove(&context->error_stack[0], &context->error_stack[1],
                (MAX_ERROR_STACK - 1) * sizeof(error_entry_t));
        context->stack_depth = MAX_ERROR_STACK - 1;
    }
    
    error_entry_t* entry = &context->error_stack[context->stack_depth];
    entry->error_code = code;
    strncpy(entry->function_name, function, sizeof(entry->function_name) - 1);
    strncpy(entry->file_name, file, sizeof(entry->file_name) - 1);
    entry->line_number = line;
    clock_gettime(CLOCK_REALTIME, &entry->timestamp);
    
    // Format error message
    va_list args;
    va_start(args, format);
    vsnprintf(entry->error_message, sizeof(entry->error_message), format, args);
    va_end(args);
    
    context->stack_depth++;
    context->total_errors++;
    
    if (code < 1000) { // Warning threshold
        context->warning_count++;
    }
    
    printf("Thread %lu: Error #%d in %s:%d - Code %d: %s\n",
           (unsigned long)pthread_self(), context->total_errors,
           function, line, code, entry->error_message);
}

// Macro for convenient error setting
#define SET_THREAD_ERROR(code, format, ...) \
    set_thread_error_detailed(code, __FUNCTION__, __FILE__, __LINE__, format, ##__VA_ARGS__)

// Get the most recent error
int get_thread_error_code() {
    thread_error_context_t* context = get_error_context();
    if (!context || context->stack_depth == 0) {
        return 0;
    }
    return context->error_stack[context->stack_depth - 1].error_code;
}

const char* get_thread_error_message() {
    thread_error_context_t* context = get_error_context();
    if (!context || context->stack_depth == 0) {
        return "No error";
    }
    return context->error_stack[context->stack_depth - 1].error_message;
}

// Print complete error stack
void print_thread_error_stack() {
    thread_error_context_t* context = get_error_context();
    if (!context || context->stack_depth == 0) {
        printf("Thread %lu: No errors\n", (unsigned long)pthread_self());
        return;
    }
    
    printf("\n=== Thread %lu Error Stack ===\n", (unsigned long)pthread_self());
    printf("Total errors: %d, Warnings: %d\n", 
           context->total_errors, context->warning_count);
    
    for (int i = 0; i < context->stack_depth; i++) {
        error_entry_t* entry = &context->error_stack[i];
        
        struct tm* tm_info = localtime(&entry->timestamp.tv_sec);
        printf("[%02d:%02d:%02d] %s:%d in %s() - Code %d: %s\n",
               tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec,
               entry->file_name, entry->line_number, entry->function_name,
               entry->error_code, entry->error_message);
    }
}

// Clear error stack
void clear_thread_errors() {
    thread_error_context_t* context = get_error_context();
    if (context) {
        context->stack_depth = 0;
        printf("Thread %lu: Cleared error stack\n", (unsigned long)pthread_self());
    }
}

// Example API functions using the error system
int divide_with_error_handling(double a, double b, double* result) {
    if (result == NULL) {
        SET_THREAD_ERROR(1001, "Result pointer is NULL");
        return -1;
    }
    
    if (b == 0.0) {
        SET_THREAD_ERROR(1002, "Division by zero: %.2f / %.2f", a, b);
        return -1;
    }
    
    if (b < 1e-10 && b > -1e-10) {
        SET_THREAD_ERROR(1003, "Division by very small number: %.2e", b);
        return -1;
    }
    
    *result = a / b;
    return 0;
}

int parse_integer_with_error_handling(const char* str, int* result) {
    if (!str) {
        SET_THREAD_ERROR(2001, "Input string is NULL");
        return -1;
    }
    
    if (!result) {
        SET_THREAD_ERROR(2002, "Result pointer is NULL");
        return -1;
    }
    
    char* endptr;
    long val = strtol(str, &endptr, 10);
    
    if (endptr == str) {
        SET_THREAD_ERROR(2003, "No digits found in string: '%s'", str);
        return -1;
    }
    
    if (*endptr != '\0') {
        SET_THREAD_ERROR(2004, "Invalid characters in string: '%s'", str);
        return -1;
    }
    
    if (val > INT_MAX || val < INT_MIN) {
        SET_THREAD_ERROR(2005, "Value out of range: %ld", val);
        return -1;
    }
    
    *result = (int)val;
    return 0;
}
```

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

### GCC/Clang `__thread` Keyword

The `__thread` keyword provides the most efficient form of thread-local storage, implemented directly by the compiler and linker.

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

// Different types of __thread declarations
__thread int thread_local_counter = 0;           // Initialized to 0
__thread char thread_local_buffer[256];          // Uninitialized
__thread double computation_cache = -1.0;       // Initial value
__thread struct timespec thread_start_time;     // Complex types

// Thread-local structure for comprehensive data
typedef struct {
    unsigned long operations;
    double total_cpu_time;
    int error_count;
    char thread_name[64];
} thread_metrics_t;

__thread thread_metrics_t metrics = {0};

// Advanced usage with initialization function
void initialize_thread_metrics(const char* name) {
    metrics.operations = 0;
    metrics.total_cpu_time = 0.0;
    metrics.error_count = 0;
    strncpy(metrics.thread_name, name, sizeof(metrics.thread_name) - 1);
    
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &thread_start_time);
    
    printf("Thread %lu (%s): Metrics initialized\n", 
           (unsigned long)pthread_self(), name);
}

// Helper functions using __thread variables
void record_operation() {
    metrics.operations++;
    
    struct timespec current_time;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current_time);
    
    metrics.total_cpu_time = (current_time.tv_sec - thread_start_time.tv_sec) +
                            (current_time.tv_nsec - thread_start_time.tv_nsec) / 1e9;
}

void record_error() {
    metrics.error_count++;
    printf("Thread %s: Error #%d recorded\n", 
           metrics.thread_name, metrics.error_count);
}

void print_thread_metrics() {
    printf("=== Thread %s Metrics ===\n", metrics.thread_name);
    printf("Operations: %lu\n", metrics.operations);
    printf("CPU Time: %.6f seconds\n", metrics.total_cpu_time);
    printf("Errors: %d\n", metrics.error_count);
    if (metrics.total_cpu_time > 0) {
        printf("Operations/sec: %.2f\n", metrics.operations / metrics.total_cpu_time);
    }
}

// Comprehensive example using __thread
void* worker_thread_advanced(void* arg) {
    char* thread_name = (char*)arg;
    
    // Initialize thread-local data
    initialize_thread_metrics(thread_name);
    thread_local_counter = 1000;
    
    // Simulate different types of work
    for (int i = 0; i < 50000; i++) {
        // Simulate computation
        double result = 0.0;
        for (int j = 0; j < 100; j++) {
            result += j * 0.001;
        }
        
        // Store in thread-local cache
        computation_cache = result;
        
        record_operation();
        
        // Simulate occasional errors
        if (i % 10000 == 0 && i > 0) {
            record_error();
        }
        
        // Update counter
        thread_local_counter++;
        
        // Format data in thread-local buffer
        if (i % 5000 == 0) {
            snprintf(thread_local_buffer, sizeof(thread_local_buffer),
                    "Progress: %d/50000, Cache: %.3f, Counter: %d",
                    i, computation_cache, thread_local_counter);
            printf("Thread %s: %s\n", thread_name, thread_local_buffer);
        }
    }
    
    print_thread_metrics();
    return NULL;
}

// Example of __thread with dynamic initialization
__thread char* dynamic_thread_buffer = NULL;
__thread size_t buffer_size = 0;

void ensure_thread_buffer(size_t min_size) {
    if (!dynamic_thread_buffer || buffer_size < min_size) {
        free(dynamic_thread_buffer);
        buffer_size = min_size * 2; // Double the requested size
        dynamic_thread_buffer = malloc(buffer_size);
        
        printf("Thread %lu: Allocated buffer of size %zu\n",
               (unsigned long)pthread_self(), buffer_size);
    }
}

// Cleanup function (must be called explicitly with __thread)
void cleanup_thread_resources() {
    free(dynamic_thread_buffer);
    dynamic_thread_buffer = NULL;
    buffer_size = 0;
}
```

### C11 `_Thread_local` Keyword

The C11 standard introduces `_Thread_local` as the portable way to declare thread-local storage:

```c
#include <stdio.h>
#include <threads.h>
#include <time.h>
#include <string.h>

// C11 thread-local declarations
_Thread_local int tls_counter = 0;
_Thread_local char tls_message[128] = "Initial message";
_Thread_local struct timespec thread_birth_time;

// Complex thread-local structure
typedef struct {
    int worker_id;
    unsigned long tasks_completed;
    double efficiency_rating;
    char status_message[64];
} worker_state_t;

_Thread_local worker_state_t worker_state = {0};

// Thread initialization function
void init_worker_thread(int id) {
    worker_state.worker_id = id;
    worker_state.tasks_completed = 0;
    worker_state.efficiency_rating = 0.0;
    snprintf(worker_state.status_message, sizeof(worker_state.status_message),
             "Worker %d initialized", id);
    
    timespec_get(&thread_birth_time, TIME_UTC);
    
    printf("C11 Thread %d: Initialized at %ld\n", 
           id, thread_birth_time.tv_sec);
}

// Task execution with metrics
int execute_task(int task_complexity) {
    struct timespec start_time, end_time;
    timespec_get(&start_time, TIME_UTC);
    
    // Simulate work based on complexity
    volatile long sum = 0;
    for (int i = 0; i < task_complexity * 1000; i++) {
        sum += i * i;
    }
    
    timespec_get(&end_time, TIME_UTC);
    
    double task_time = (end_time.tv_sec - start_time.tv_sec) +
                      (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    worker_state.tasks_completed++;
    
    // Update efficiency rating (tasks per second)
    struct timespec current_time;
    timespec_get(&current_time, TIME_UTC);
    double total_time = (current_time.tv_sec - thread_birth_time.tv_sec) +
                       (current_time.tv_nsec - thread_birth_time.tv_nsec) / 1e9;
    
    if (total_time > 0) {
        worker_state.efficiency_rating = worker_state.tasks_completed / total_time;
    }
    
    snprintf(worker_state.status_message, sizeof(worker_state.status_message),
             "Completed task %lu (%.3fs)", worker_state.tasks_completed, task_time);
    
    return 0;
}

// Get worker statistics
void print_worker_stats() {
    printf("=== Worker %d Statistics ===\n", worker_state.worker_id);
    printf("Tasks completed: %lu\n", worker_state.tasks_completed);
    printf("Efficiency: %.2f tasks/sec\n", worker_state.efficiency_rating);
    printf("Status: %s\n", worker_state.status_message);
}

int worker_thread_func(void* arg) {
    int worker_id = *(int*)arg;
    init_worker_thread(worker_id);
    
    // Execute varying complexity tasks
    for (int i = 0; i < 20; i++) {
        int complexity = (i % 5) + 1; // 1-5 complexity levels
        execute_task(complexity);
        
        // Update local counter and message
        tls_counter++;
        snprintf(tls_message, sizeof(tls_message),
                "Worker %d: Task %d completed", worker_id, tls_counter);
        
        if (i % 5 == 0) {
            printf("Thread %d: %s\n", worker_id, tls_message);
        }
    }
    
    print_worker_stats();
    return thrd_success;
}
```

### Microsoft Visual C++ `__declspec(thread)`

```c
#ifdef _WIN32
#include <windows.h>
#include <stdio.h>

// MSVC thread-local storage
__declspec(thread) int msvc_tls_counter = 0;
__declspec(thread) char msvc_tls_buffer[256];
__declspec(thread) LARGE_INTEGER thread_start_time;

void init_msvc_thread() {
    QueryPerformanceCounter(&thread_start_time);
    msvc_tls_counter = GetCurrentThreadId() % 1000;
    sprintf_s(msvc_tls_buffer, sizeof(msvc_tls_buffer),
              "MSVC Thread %lu initialized", GetCurrentThreadId());
    
    printf("%s\n", msvc_tls_buffer);
}

DWORD WINAPI msvc_worker_thread(LPVOID param) {
    init_msvc_thread();
    
    for (int i = 0; i < 10; i++) {
        msvc_tls_counter += i;
        
        LARGE_INTEGER current_time;
        QueryPerformanceCounter(&current_time);
        
        sprintf_s(msvc_tls_buffer, sizeof(msvc_tls_buffer),
                 "Thread %lu: Counter=%d, Time=%lld",
                 GetCurrentThreadId(), msvc_tls_counter, 
                 current_time.QuadPart - thread_start_time.QuadPart);
        
        printf("%s\n", msvc_tls_buffer);
        Sleep(100);
    }
    
    return 0;
}
#endif
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

## Advanced TLS Usage Patterns

### Thread-Local Singleton Pattern

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

typedef struct {
    int instance_id;
    struct timespec creation_time;
    unsigned long access_count;
    char instance_data[128];
} singleton_instance_t;

__thread singleton_instance_t* thread_singleton = NULL;
static int global_instance_counter = 0;
static pthread_mutex_t instance_counter_mutex = PTHREAD_MUTEX_INITIALIZER;

// Get or create thread-local singleton
singleton_instance_t* get_thread_singleton() {
    if (!thread_singleton) {
        thread_singleton = malloc(sizeof(singleton_instance_t));
        
        pthread_mutex_lock(&instance_counter_mutex);
        thread_singleton->instance_id = ++global_instance_counter;
        pthread_mutex_unlock(&instance_counter_mutex);
        
        clock_gettime(CLOCK_REALTIME, &thread_singleton->creation_time);
        thread_singleton->access_count = 0;
        
        snprintf(thread_singleton->instance_data, 
                sizeof(thread_singleton->instance_data),
                "Singleton instance for thread %lu",
                (unsigned long)pthread_self());
        
        printf("Created singleton instance #%d for thread %lu\n",
               thread_singleton->instance_id, (unsigned long)pthread_self());
    }
    
    thread_singleton->access_count++;
    return thread_singleton;
}

// Cleanup function (manual call required)
void cleanup_thread_singleton() {
    if (thread_singleton) {
        printf("Destroying singleton instance #%d (accessed %lu times)\n",
               thread_singleton->instance_id, thread_singleton->access_count);
        free(thread_singleton);
        thread_singleton = NULL;
    }
}

void* singleton_test_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    // Multiple accesses to singleton
    for (int i = 0; i < 5; i++) {
        singleton_instance_t* instance = get_thread_singleton();
        printf("Thread %d: Access #%lu to instance #%d\n",
               thread_id, instance->access_count, instance->instance_id);
        usleep(100000); // 100ms
    }
    
    cleanup_thread_singleton();
    return NULL;
}
```

### Thread-Local Memory Pool

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>

#define POOL_SIZE 8192
#define ALIGNMENT 8
#define ALIGN(size) (((size) + ALIGNMENT - 1) & ~(ALIGNMENT - 1))

typedef struct memory_block {
    size_t size;
    int is_free;
    struct memory_block* next;
    uint8_t data[];
} memory_block_t;

typedef struct {
    uint8_t pool[POOL_SIZE];
    memory_block_t* first_block;
    size_t total_allocated;
    size_t peak_allocated;
    int allocation_count;
    int free_count;
} thread_memory_pool_t;

__thread thread_memory_pool_t* tls_pool = NULL;

// Initialize thread-local memory pool
thread_memory_pool_t* get_thread_pool() {
    if (!tls_pool) {
        tls_pool = malloc(sizeof(thread_memory_pool_t));
        memset(tls_pool, 0, sizeof(thread_memory_pool_t));
        
        // Initialize first block to cover entire pool
        tls_pool->first_block = (memory_block_t*)tls_pool->pool;
        tls_pool->first_block->size = POOL_SIZE - sizeof(memory_block_t);
        tls_pool->first_block->is_free = 1;
        tls_pool->first_block->next = NULL;
        
        printf("Thread %lu: Initialized memory pool (%d bytes)\n",
               (unsigned long)pthread_self(), POOL_SIZE);
    }
    return tls_pool;
}

// Thread-local malloc
void* tls_malloc(size_t size) {
    thread_memory_pool_t* pool = get_thread_pool();
    size = ALIGN(size);
    
    memory_block_t* current = pool->first_block;
    
    // Find suitable free block
    while (current) {
        if (current->is_free && current->size >= size) {
            // Split block if necessary
            if (current->size > size + sizeof(memory_block_t) + ALIGNMENT) {
                memory_block_t* new_block = (memory_block_t*)
                    ((uint8_t*)current + sizeof(memory_block_t) + size);
                new_block->size = current->size - size - sizeof(memory_block_t);
                new_block->is_free = 1;
                new_block->next = current->next;
                
                current->size = size;
                current->next = new_block;
            }
            
            current->is_free = 0;
            pool->total_allocated += current->size;
            pool->allocation_count++;
            
            if (pool->total_allocated > pool->peak_allocated) {
                pool->peak_allocated = pool->total_allocated;
            }
            
            return current->data;
        }
        current = current->next;
    }
    
    printf("Thread %lu: Pool allocation failed for %zu bytes\n",
           (unsigned long)pthread_self(), size);
    return NULL; // Pool exhausted
}

// Thread-local free
void tls_free(void* ptr) {
    if (!ptr || !tls_pool) return;
    
    thread_memory_pool_t* pool = tls_pool;
    
    // Find the block containing this pointer
    memory_block_t* current = pool->first_block;
    while (current) {
        if (current->data == ptr) {
            current->is_free = 1;
            pool->total_allocated -= current->size;
            pool->free_count++;
            
            // Coalesce with next block if it's free
            if (current->next && current->next->is_free) {
                current->size += current->next->size + sizeof(memory_block_t);
                current->next = current->next->next;
            }
            
            // Coalesce with previous block if it's free
            memory_block_t* prev = pool->first_block;
            while (prev && prev->next != current) {
                prev = prev->next;
            }
            
            if (prev && prev->is_free) {
                prev->size += current->size + sizeof(memory_block_t);
                prev->next = current->next;
            }
            
            return;
        }
        current = current->next;
    }
}

// Get pool statistics
void print_pool_stats() {
    if (!tls_pool) return;
    
    thread_memory_pool_t* pool = tls_pool;
    
    printf("=== Thread %lu Pool Statistics ===\n", (unsigned long)pthread_self());
    printf("Current allocated: %zu bytes\n", pool->total_allocated);
    printf("Peak allocated: %zu bytes\n", pool->peak_allocated);
    printf("Allocations: %d\n", pool->allocation_count);
    printf("Frees: %d\n", pool->free_count);
    printf("Pool utilization: %.1f%%\n", 
           (double)pool->peak_allocated / POOL_SIZE * 100);
}

// Cleanup pool
void cleanup_thread_pool() {
    if (tls_pool) {
        print_pool_stats();
        free(tls_pool);
        tls_pool = NULL;
    }
}

void* pool_test_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    // Test various allocation patterns
    void* ptrs[10];
    
    // Allocate various sizes
    for (int i = 0; i < 10; i++) {
        size_t size = (i + 1) * 64; // 64, 128, 192, ... bytes
        ptrs[i] = tls_malloc(size);
        
        if (ptrs[i]) {
            memset(ptrs[i], i, size); // Fill with pattern
            printf("Thread %d: Allocated %zu bytes at %p\n", 
                   thread_id, size, ptrs[i]);
        }
    }
    
    // Free every other allocation
    for (int i = 0; i < 10; i += 2) {
        if (ptrs[i]) {
            tls_free(ptrs[i]);
            printf("Thread %d: Freed allocation %d\n", thread_id, i);
        }
    }
    
    // Try to allocate large block
    void* large_ptr = tls_malloc(2048);
    if (large_ptr) {
        printf("Thread %d: Large allocation successful\n", thread_id);
        tls_free(large_ptr);
    }
    
    // Free remaining allocations
    for (int i = 1; i < 10; i += 2) {
        if (ptrs[i]) {
            tls_free(ptrs[i]);
        }
    }
    
    cleanup_thread_pool();
    return NULL;
}
```

### Thread-Local String Interning

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MAX_INTERNED_STRINGS 256
#define STRING_POOL_SIZE 4096

typedef struct string_entry {
    char* string;
    int ref_count;
    size_t length;
} string_entry_t;

typedef struct {
    string_entry_t entries[MAX_INTERNED_STRINGS];
    char string_pool[STRING_POOL_SIZE];
    size_t pool_used;
    int entry_count;
    int hit_count;
    int miss_count;
} thread_string_intern_t;

__thread thread_string_intern_t* tls_intern = NULL;

// Initialize string interning system
thread_string_intern_t* get_string_intern() {
    if (!tls_intern) {
        tls_intern = calloc(1, sizeof(thread_string_intern_t));
        printf("Thread %lu: Initialized string interning system\n", 
               (unsigned long)pthread_self());
    }
    return tls_intern;
}

// Intern a string (return existing or create new)
const char* intern_string(const char* str) {
    if (!str) return NULL;
    
    thread_string_intern_t* intern = get_string_intern();
    size_t len = strlen(str);
    
    // Search for existing string
    for (int i = 0; i < intern->entry_count; i++) {
        if (intern->entries[i].length == len &&
            strcmp(intern->entries[i].string, str) == 0) {
            intern->entries[i].ref_count++;
            intern->hit_count++;
            return intern->entries[i].string;
        }
    }
    
    // Add new string if space available
    if (intern->entry_count < MAX_INTERNED_STRINGS &&
        intern->pool_used + len + 1 <= STRING_POOL_SIZE) {
        
        string_entry_t* entry = &intern->entries[intern->entry_count];
        entry->string = &intern->string_pool[intern->pool_used];
        entry->length = len;
        entry->ref_count = 1;
        
        strcpy(entry->string, str);
        intern->pool_used += len + 1;
        intern->entry_count++;
        intern->miss_count++;
        
        return entry->string;
    }
    
    // Fallback: return original string
    intern->miss_count++;
    return str;
}

// Print interning statistics
void print_intern_stats() {
    if (!tls_intern) return;
    
    thread_string_intern_t* intern = tls_intern;
    
    printf("=== Thread %lu String Interning Stats ===\n", 
           (unsigned long)pthread_self());
    printf("Unique strings: %d\n", intern->entry_count);
    printf("Pool used: %zu/%d bytes\n", intern->pool_used, STRING_POOL_SIZE);
    printf("Cache hits: %d\n", intern->hit_count);
    printf("Cache misses: %d\n", intern->miss_count);
    
    if (intern->hit_count + intern->miss_count > 0) {
        printf("Hit rate: %.1f%%\n", 
               (double)intern->hit_count / 
               (intern->hit_count + intern->miss_count) * 100);
    }
    
    // Show most referenced strings
    for (int i = 0; i < intern->entry_count; i++) {
        if (intern->entries[i].ref_count > 1) {
            printf("  '%s': %d references\n", 
                   intern->entries[i].string, intern->entries[i].ref_count);
        }
    }
}

void cleanup_string_intern() {
    if (tls_intern) {
        print_intern_stats();
        free(tls_intern);
        tls_intern = NULL;
    }
}

void* string_intern_test_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    // Test repeated string interning
    const char* test_strings[] = {
        "Hello", "World", "Thread", "Local", "Storage",
        "Hello", "World", "Hello", "Performance", "Test"
    };
    
    printf("Thread %d: Testing string interning\n", thread_id);
    
    for (int i = 0; i < 10; i++) {
        const char* interned = intern_string(test_strings[i]);
        printf("Thread %d: Interned '%s' -> %p\n", 
               thread_id, test_strings[i], (void*)interned);
    }
    
    // Test with formatted strings
    char buffer[64];
    for (int i = 0; i < 5; i++) {
        snprintf(buffer, sizeof(buffer), "Generated_%d", i % 3);
        const char* interned = intern_string(buffer);
        printf("Thread %d: Dynamic string '%s' -> %p\n", 
               thread_id, buffer, (void*)interned);
    }
    
    cleanup_string_intern();
    return NULL;
}
```

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

### TLS Access Performance Comparison

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

#define ITERATIONS 10000000

// Different TLS methods for comparison
__thread int gcc_tls_var = 0;
pthread_key_t pthread_key;
pthread_once_t key_once = PTHREAD_ONCE_INIT;

void init_pthread_key() {
    pthread_key_create(&pthread_key, NULL);
}

// Performance measurement utilities
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + 
           (end.tv_nsec - start.tv_nsec) / 1e9;
}

void benchmark_gcc_tls() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < ITERATIONS; i++) {
        gcc_tls_var = i;
        volatile int temp = gcc_tls_var;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    printf("GCC __thread access: %.3f ns per operation\n",
           get_time_diff(start, end) * 1e9 / ITERATIONS);
}

void benchmark_pthread_tls() {
    pthread_once(&key_once, init_pthread_key);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < ITERATIONS; i++) {
        int* value = malloc(sizeof(int));
        *value = i;
        pthread_setspecific(pthread_key, value);
        
        int* retrieved = (int*)pthread_getspecific(pthread_key);
        volatile int temp = *retrieved;
        
        free(value);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    printf("pthread_setspecific/getspecific: %.3f ns per operation\n",
           get_time_diff(start, end) * 1e9 / ITERATIONS);
}

void benchmark_global_with_mutex() {
    static int global_var = 0;
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < ITERATIONS; i++) {
        pthread_mutex_lock(&mutex);
        global_var = i;
        volatile int temp = global_var;
        pthread_mutex_unlock(&mutex);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    printf("Global variable with mutex: %.3f ns per operation\n",
           get_time_diff(start, end) * 1e9 / ITERATIONS);
}

void* performance_test_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    printf("\n=== Thread %d Performance Tests ===\n", thread_id);
    
    benchmark_gcc_tls();
    benchmark_pthread_tls();
    benchmark_global_with_mutex();
    
    return NULL;
}
```

### Memory Usage Analysis

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

// Large TLS structures for memory analysis
__thread char large_buffer[16384];           // 16KB per thread
__thread double matrix[256][256];            // 512KB per thread
__thread int counters[1000];                 // 4KB per thread

typedef struct {
    char data[4096];
    int metadata[100];
    double calculations[50];
} complex_tls_structure;

__thread complex_tls_structure tls_struct = {0};

// Memory usage reporting
void report_tls_memory_usage(int thread_count) {
    size_t per_thread_tls = sizeof(large_buffer) + 
                           sizeof(matrix) + 
                           sizeof(counters) + 
                           sizeof(tls_struct);
    
    size_t total_tls_memory = per_thread_tls * thread_count;
    
    printf("\n=== TLS Memory Usage Analysis ===\n");
    printf("Memory per thread (TLS only): %zu bytes (%.2f KB)\n", 
           per_thread_tls, per_thread_tls / 1024.0);
    printf("Total TLS memory for %d threads: %zu bytes (%.2f MB)\n",
           thread_count, total_tls_memory, total_tls_memory / 1048576.0);
    
    // Calculate overhead compared to shared memory
    size_t shared_memory = per_thread_tls; // Only one copy if shared
    printf("Memory overhead vs shared: %.1fx\n", 
           (double)total_tls_memory / shared_memory);
}

// Memory initialization patterns
void init_tls_memory_pattern(int thread_id) {
    // Initialize with thread-specific patterns
    memset(large_buffer, thread_id & 0xFF, sizeof(large_buffer));
    
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            matrix[i][j] = thread_id * 1000.0 + i + j;
        }
    }
    
    for (int i = 0; i < 1000; i++) {
        counters[i] = thread_id * i;
    }
    
    snprintf(tls_struct.data, sizeof(tls_struct.data), 
             "Thread %d complex data structure", thread_id);
    
    printf("Thread %d: Initialized %.2f KB of TLS memory\n", 
           thread_id, (sizeof(large_buffer) + sizeof(matrix) + 
                      sizeof(counters) + sizeof(tls_struct)) / 1024.0);
}

void* memory_usage_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    init_tls_memory_pattern(thread_id);
    
    // Simulate memory access patterns
    for (int iter = 0; iter < 100; iter++) {
        // Access different parts of TLS memory
        large_buffer[iter % sizeof(large_buffer)] = thread_id;
        matrix[iter % 256][iter % 256] += 1.0;
        counters[iter % 1000]++;
        
        if (iter % 10 == 0) {
            snprintf(tls_struct.data, sizeof(tls_struct.data),
                    "Thread %d iteration %d", thread_id, iter);
        }
    }
    
    printf("Thread %d: Completed memory access patterns\n", thread_id);
    return NULL;
}
```

### Cache Performance Impact

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

#define CACHE_LINE_SIZE 64
#define TEST_ARRAY_SIZE 1024

// Aligned TLS data for cache efficiency
typedef struct {
    int data[CACHE_LINE_SIZE / sizeof(int)];
} cache_line_t;

__thread cache_line_t tls_cache_lines[TEST_ARRAY_SIZE / (CACHE_LINE_SIZE / sizeof(int))];

// Shared data structure (potential false sharing)
struct shared_counters {
    int counter1;
    char padding1[CACHE_LINE_SIZE - sizeof(int)];
    int counter2;
    char padding2[CACHE_LINE_SIZE - sizeof(int)];
    int counter3;
    char padding3[CACHE_LINE_SIZE - sizeof(int)];
} shared_data = {0};

// TLS counters (no false sharing)
__thread int tls_counter1 = 0;
__thread int tls_counter2 = 0;
__thread int tls_counter3 = 0;

void benchmark_false_sharing(int iterations) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < iterations; i++) {
        __sync_fetch_and_add(&shared_data.counter1, 1);
        __sync_fetch_and_add(&shared_data.counter2, 1);
        __sync_fetch_and_add(&shared_data.counter3, 1);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_taken = (end.tv_sec - start.tv_sec) + 
                       (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Shared counters (false sharing): %.3f ns per operation\n",
           time_taken * 1e9 / (iterations * 3));
}

void benchmark_tls_no_sharing(int iterations) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < iterations; i++) {
        tls_counter1++;
        tls_counter2++;
        tls_counter3++;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_taken = (end.tv_sec - start.tv_sec) + 
                       (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("TLS counters (no sharing): %.3f ns per operation\n",
           time_taken * 1e9 / (iterations * 3));
}

void* cache_performance_thread(void* arg) {
    int thread_id = *(int*)arg;
    int iterations = 1000000;
    
    printf("\nThread %d Cache Performance Tests:\n", thread_id);
    
    benchmark_false_sharing(iterations);
    benchmark_tls_no_sharing(iterations);
    
    // Initialize TLS cache lines with thread-specific data
    for (int i = 0; i < sizeof(tls_cache_lines) / sizeof(cache_line_t); i++) {
        for (int j = 0; j < CACHE_LINE_SIZE / sizeof(int); j++) {
            tls_cache_lines[i].data[j] = thread_id * 1000 + i * 16 + j;
        }
    }
    
    // Sequential access pattern (cache-friendly)
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    volatile int sum = 0;
    for (int iter = 0; iter < 1000; iter++) {
        for (int i = 0; i < sizeof(tls_cache_lines) / sizeof(cache_line_t); i++) {
            for (int j = 0; j < CACHE_LINE_SIZE / sizeof(int); j++) {
                sum += tls_cache_lines[i].data[j];
            }
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_taken = (end.tv_sec - start.tv_sec) + 
                       (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Thread %d sequential TLS access: %.3f ms (sum=%d)\n",
           thread_id, time_taken * 1000, sum);
    
    return NULL;
}
```

### Performance Guidelines

#### 1. Choose the Right TLS Method

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TLS Performance Comparison                     │
├─────────────────────────────────────────────────────────────────────┤
│  Method              │  Access Time  │  Memory Usage  │  Flexibility │
├─────────────────────────────────────────────────────────────────────┤
│  __thread           │  ~1-2 ns      │  Static        │  Limited     │
│  _Thread_local      │  ~1-2 ns      │  Static        │  Limited     │
│  pthread_getspecific│  ~20-50 ns    │  Dynamic       │  High        │
│  Manual TLS array  │  ~5-10 ns     │  Static        │  Medium      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2. Memory Considerations

- **Static TLS**: Allocated at thread creation, faster access
- **Dynamic TLS**: Allocated on demand, more flexible but slower
- **Memory per thread**: Each thread gets its own copy
- **Consider thread count**: TLS memory scales linearly with threads

#### 3. Cache Efficiency

- Align TLS structures to cache line boundaries
- Group frequently accessed TLS variables together
- Avoid unnecessary padding in TLS structures
- Consider NUMA effects in multi-socket systems

## Common Pitfalls and Solutions

### 1. Memory Leaks with pthread_key_t

**Problem:**
```c
// WRONG: No destructor specified
pthread_key_t key;
pthread_key_create(&key, NULL);

void* worker(void* arg) {
    char* data = malloc(256);
    pthread_setspecific(key, data);
    return NULL; // Memory leak - data never freed
}
```

**Solution:**
```c
// CORRECT: Proper destructor
void cleanup_data(void* data) {
    free(data);
    printf("Cleaned up thread-local data\n");
}

pthread_key_t key;
pthread_key_create(&key, cleanup_data);
```

### 2. Accessing TLS After Thread Exit

**Problem:**
```c
// WRONG: Accessing TLS from another thread
pthread_t thread;
__thread int tls_value = 42;

void* worker(void* arg) {
    tls_value = 100;
    return &tls_value; // DANGEROUS: returning address of TLS variable
}

void* main_thread() {
    pthread_create(&thread, NULL, worker, NULL);
    int* result;
    pthread_join(thread, (void**)&result);
    printf("Value: %d\n", *result); // UNDEFINED BEHAVIOR
}
```

**Solution:**
```c
// CORRECT: Copy TLS data before thread exit
void* worker(void* arg) {
    static __thread int tls_value = 42;
    tls_value = 100;
    
    int* result = malloc(sizeof(int));
    *result = tls_value;
    return result; // Safe: heap allocated
}
```

### 3. Initialization Race Conditions

**Problem:**
```c
// WRONG: Race condition in key creation
pthread_key_t global_key;

void* worker(void* arg) {
    // Multiple threads might call this simultaneously
    pthread_key_create(&global_key, cleanup_func);
    // Race condition here!
}
```

**Solution:**
```c
// CORRECT: Use pthread_once
pthread_key_t global_key;
pthread_once_t key_once = PTHREAD_ONCE_INIT;

void init_key() {
    pthread_key_create(&global_key, cleanup_func);
}

void* worker(void* arg) {
    pthread_once(&key_once, init_key);
    // Safe: key created exactly once
}
```

### 4. Assuming TLS is Zero-Initialized

**Problem:**
```c
// WRONG: Assuming uninitialized TLS is zero
__thread int counter; // May not be zero!

void increment() {
    counter++; // May start from garbage value
}
```

**Solution:**
```c
// CORRECT: Explicit initialization
__thread int counter = 0; // Explicitly zero-initialized

// Or use initialization flag
__thread int counter;
__thread int initialized = 0;

void ensure_initialized() {
    if (!initialized) {
        counter = 0;
        initialized = 1;
    }
}
```

### 5. TLS in Dynamically Loaded Libraries

**Problem:**
```c
// WRONG: TLS in library without proper initialization
// In shared library (.so/.dll)
__thread int lib_tls_var = 0;

void lib_function() {
    lib_tls_var++; // May not work correctly in all scenarios
}
```

**Solution:**
```c
// CORRECT: Use pthread_key_t for dynamic libraries
static pthread_key_t lib_key;
static pthread_once_t lib_once = PTHREAD_ONCE_INIT;

void init_lib_key() {
    pthread_key_create(&lib_key, NULL);
}

int* get_lib_counter() {
    pthread_once(&lib_once, init_lib_key);
    
    int* counter = pthread_getspecific(lib_key);
    if (!counter) {
        counter = calloc(1, sizeof(int));
        pthread_setspecific(lib_key, counter);
    }
    return counter;
}

void lib_function() {
    int* counter = get_lib_counter();
    (*counter)++;
}
```

## Comprehensive Exercises

### Exercise 1: Thread-Local HTTP Connection Pool

**Objective:** Implement a thread-local HTTP connection pool where each thread maintains its own set of persistent connections.

**Requirements:**
- Support multiple HTTP servers
- Implement connection reuse and timeout
- Track connection statistics per thread
- Implement proper cleanup on thread exit

**Starter Code Framework:**
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <netinet/in.h>
#include <sys/socket.h>

#define MAX_CONNECTIONS 10
#define MAX_HOSTNAME_LEN 256

typedef struct {
    int socket_fd;
    char hostname[MAX_HOSTNAME_LEN];
    int port;
    time_t last_used;
    int is_active;
} http_connection_t;

typedef struct {
    http_connection_t connections[MAX_CONNECTIONS];
    int connection_count;
    int total_requests;
    int cache_hits;
    int cache_misses;
} thread_http_pool_t;

// TODO: Implement these functions
__thread thread_http_pool_t* tls_http_pool = NULL;

thread_http_pool_t* get_http_pool();
http_connection_t* get_connection(const char* hostname, int port);
void release_connection(http_connection_t* conn);
void cleanup_expired_connections();
void print_pool_statistics();
void cleanup_http_pool();

// Test the implementation
void* http_worker_thread(void* arg);
```

**Advanced Requirements:**
- Implement connection pooling with LRU eviction
- Add connection health checking
- Support SSL/TLS connections
- Implement connection load balancing

### Exercise 2: Thread-Local Memory Allocator with Debugging

**Objective:** Create a thread-local memory allocator that tracks allocations, detects leaks, and provides debugging information.

**Requirements:**
- Implement malloc/free replacement functions
- Track allocation sizes and call stacks
- Detect memory leaks and double-free errors
- Provide per-thread memory usage statistics
- Support memory debugging with guards

**Starter Code Framework:**
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <execinfo.h>
#include <stdint.h>

#define MAX_ALLOCATIONS 1000
#define STACK_TRACE_DEPTH 10
#define GUARD_PATTERN 0xDEADBEEF
#define FREE_PATTERN 0xFEEDFACE

typedef struct {
    void* ptr;
    size_t size;
    const char* file;
    int line;
    void* stack_trace[STACK_TRACE_DEPTH];
    int stack_depth;
    time_t alloc_time;
    int is_active;
} allocation_record_t;

typedef struct {
    allocation_record_t allocations[MAX_ALLOCATIONS];
    int allocation_count;
    size_t total_allocated;
    size_t peak_allocated;
    int leak_count;
    int double_free_count;
} thread_debug_allocator_t;

// TODO: Implement these functions
__thread thread_debug_allocator_t* tls_allocator = NULL;

void* debug_malloc(size_t size, const char* file, int line);
void debug_free(void* ptr, const char* file, int line);
void* debug_realloc(void* ptr, size_t size, const char* file, int line);
void print_allocation_report();
void check_memory_leaks();
void cleanup_debug_allocator();

#define MALLOC(size) debug_malloc(size, __FILE__, __LINE__)
#define FREE(ptr) debug_free(ptr, __FILE__, __LINE__)
#define REALLOC(ptr, size) debug_realloc(ptr, size, __FILE__, __LINE__)

// Test the implementation
void* memory_test_thread(void* arg);
```

### Exercise 3: Thread-Local Configuration Management System

**Objective:** Build a hierarchical configuration system where threads can have their own configuration overrides while inheriting from global defaults.

**Requirements:**
- Support nested configuration sections
- Implement configuration inheritance and override mechanisms
- Support different data types (int, double, string, boolean)
- Provide configuration change notifications
- Implement configuration validation

**Starter Code Framework:**
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef enum {
    CONFIG_INT,
    CONFIG_DOUBLE,
    CONFIG_STRING,
    CONFIG_BOOL
} config_type_t;

typedef struct config_value {
    char* key;
    config_type_t type;
    union {
        int int_val;
        double double_val;
        char* string_val;
        int bool_val;
    } value;
    struct config_value* next;
} config_value_t;

typedef struct {
    config_value_t* values;
    config_value_t* parent_config; // Inheritance chain
    pthread_mutex_t config_mutex;
    int change_count;
} thread_config_t;

// TODO: Implement these functions
__thread thread_config_t* tls_config = NULL;
static config_value_t* global_config = NULL;

void init_config_system();
void set_config_int(const char* key, int value);
void set_config_double(const char* key, double value);
void set_config_string(const char* key, const char* value);
void set_config_bool(const char* key, int value);

int get_config_int(const char* key, int default_value);
double get_config_double(const char* key, double default_value);
const char* get_config_string(const char* key, const char* default_value);
int get_config_bool(const char* key, int default_value);

void print_config_hierarchy();
void cleanup_thread_config();

// Test the implementation
void* config_test_thread(void* arg);
```

### Exercise 4: Thread-Local Performance Profiler

**Objective:** Create a comprehensive performance profiling system that tracks function execution times, call counts, and performance hotspots on a per-thread basis.

**Requirements:**
- Implement function enter/exit tracking
- Calculate execution times with nanosecond precision
- Support nested function calls with proper stack management
- Generate performance reports with call graphs
- Implement sampling-based profiling to reduce overhead

**Advanced Features:**
- CPU cache miss tracking
- Memory allocation profiling integration
- Real-time performance monitoring
- Profile data export to standard formats

### Exercise 5: Thread-Local Database Connection Manager

**Objective:** Design a sophisticated database connection manager that maintains per-thread connection pools with advanced features.

**Requirements:**
- Support multiple database types (PostgreSQL, MySQL, SQLite)
- Implement connection pooling with configurable limits
- Add transaction management with proper rollback
- Support connection health monitoring and automatic reconnection
- Implement query caching with thread-local cache

**Advanced Features:**
- Connection load balancing across multiple database servers
- Read/write splitting for master-slave configurations
- Connection pooling statistics and monitoring
- Automatic failover and recovery mechanisms

## Assessment and Self-Evaluation

### Knowledge Check Questions

1. **Basic Understanding:**
   - What are the main differences between `__thread` and `pthread_key_t`?
   - When should you use TLS instead of synchronization primitives?
   - How does TLS affect memory usage in multi-threaded applications?

2. **Implementation Skills:**
   - Implement a thread-safe random number generator using TLS
   - Create a thread-local error handling system with stack traces
   - Design a thread-local memory pool with statistics tracking

3. **Performance Analysis:**
   - Measure and compare the performance of different TLS approaches
   - Analyze the cache implications of TLS vs. shared memory
   - Identify scenarios where TLS might hurt performance

4. **Advanced Concepts:**
   - Explain the relationship between TLS and NUMA architecture
   - Discuss TLS behavior in dynamically loaded libraries
   - Analyze TLS initialization and destruction ordering

### Practical Assessment Criteria

**Beginner Level (0-3 points each):**
- [ ] Can declare and use basic TLS variables with `__thread`
- [ ] Understands the concept of per-thread data isolation
- [ ] Can implement simple thread-local storage using pthread keys
- [ ] Recognizes when TLS is appropriate vs. synchronization

**Intermediate Level (0-5 points each):**
- [ ] Implements proper cleanup with pthread key destructors
- [ ] Creates thread-local data structures with initialization
- [ ] Handles TLS in library code correctly
- [ ] Optimizes TLS usage for performance

**Advanced Level (0-10 points each):**
- [ ] Designs complex TLS systems with inheritance and overrides
- [ ] Implements custom TLS allocators with debugging features
- [ ] Analyzes and optimizes TLS memory usage patterns
- [ ] Creates portable TLS code across different platforms

### Total Score Interpretation:
- **90-100 points:** Expert level - Ready for production TLS development
- **70-89 points:** Advanced level - Can handle complex TLS scenarios
- **50-69 points:** Intermediate level - Understands core concepts
- **30-49 points:** Beginner level - Needs more practice
- **Below 30 points:** Review fundamentals before proceeding

### Real-World Application Projects

1. **High-Performance Web Server**
   - Implement per-thread request processing with TLS
   - Thread-local connection pools and caching
   - Performance monitoring and statistics

2. **Multi-Threaded Game Engine**
   - Thread-local resource management
   - Per-thread scripting contexts
   - Thread-local memory allocators for performance

3. **Database Management System**
   - Thread-local transaction contexts
   - Per-thread query caches
   - Thread-local buffer pools

4. **Scientific Computing Framework**
   - Thread-local random number generators
   - Per-thread result accumulation
   - Thread-local algorithm state management

## Common Use Cases and Examples

### 1. Thread-Safe errno Implementation

```c
#include <stdio.h>
#include <errno.h>

// Traditional approach - not thread-safe
// extern int errno;

// Thread-safe approach
__thread int thread_errno = 0;

#define set_errno(val) (thread_errno = (val))
#define get_errno() (thread_errno)

// Example usage
int safe_divide(int a, int b, double* result) {
    if (b == 0) {
        set_errno(EINVAL);
        return -1;
    }
    
    *result = (double)a / b;
    set_errno(0);
    return 0;
}

void* worker_thread(void* arg) {
    int thread_id = *(int*)arg;
    double result;
    
    // Test successful operation
    if (safe_divide(10, 2, &result) == 0) {
        printf("Thread %d: 10/2 = %.2f (errno: %d)\n", 
               thread_id, result, get_errno());
    }
    
    // Test error condition
    if (safe_divide(10, 0, &result) != 0) {
        printf("Thread %d: Division error (errno: %d)\n", 
               thread_id, get_errno());
    }
    
    return NULL;
}
```

### 2. Per-Thread Memory Pool

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define POOL_SIZE 4096

typedef struct memory_chunk {
    size_t size;
    int is_free;
    struct memory_chunk* next;
    char data[];
} memory_chunk_t;

typedef struct {
    char pool[POOL_SIZE];
    memory_chunk_t* first_chunk;
    size_t allocated;
    size_t peak_usage;
    int allocations;
    int deallocations;
} thread_memory_pool_t;

__thread thread_memory_pool_t* tls_pool = NULL;

void cleanup_memory_pool(void* data) {
    thread_memory_pool_t* pool = (thread_memory_pool_t*)data;
    printf("Thread pool stats - Allocated: %zu, Peak: %zu, Allocs: %d, Frees: %d\n",
           pool->allocated, pool->peak_usage, pool->allocations, pool->deallocations);
    free(pool);
}

pthread_key_t pool_key;
pthread_once_t pool_init_once = PTHREAD_ONCE_INIT;

void init_pool_key() {
    pthread_key_create(&pool_key, cleanup_memory_pool);
}

thread_memory_pool_t* get_thread_pool() {
    pthread_once(&pool_init_once, init_pool_key);
    
    if (!tls_pool) {
        tls_pool = calloc(1, sizeof(thread_memory_pool_t));
        
        // Initialize first chunk to span entire pool
        tls_pool->first_chunk = (memory_chunk_t*)tls_pool->pool;
        tls_pool->first_chunk->size = POOL_SIZE - sizeof(memory_chunk_t);
        tls_pool->first_chunk->is_free = 1;
        tls_pool->first_chunk->next = NULL;
        
        pthread_setspecific(pool_key, tls_pool);
        printf("Thread %lu: Initialized memory pool\n", (unsigned long)pthread_self());
    }
    
    return tls_pool;
}

void* pool_malloc(size_t size) {
    thread_memory_pool_t* pool = get_thread_pool();
    
    // Align size to 8 bytes
    size = (size + 7) & ~7;
    
    memory_chunk_t* current = pool->first_chunk;
    
    while (current) {
        if (current->is_free && current->size >= size) {
            // Split chunk if necessary
            if (current->size > size + sizeof(memory_chunk_t) + 8) {
                memory_chunk_t* new_chunk = (memory_chunk_t*)
                    ((char*)current + sizeof(memory_chunk_t) + size);
                new_chunk->size = current->size - size - sizeof(memory_chunk_t);
                new_chunk->is_free = 1;
                new_chunk->next = current->next;
                
                current->size = size;
                current->next = new_chunk;
            }
            
            current->is_free = 0;
            pool->allocated += current->size;
            pool->allocations++;
            
            if (pool->allocated > pool->peak_usage) {
                pool->peak_usage = pool->allocated;
            }
            
            return current->data;
        }
        current = current->next;
    }
    
    return NULL; // Pool exhausted
}

void pool_free(void* ptr) {
    if (!ptr || !tls_pool) return;
    
    thread_memory_pool_t* pool = tls_pool;
    
    // Find the chunk
    memory_chunk_t* current = pool->first_chunk;
    while (current) {
        if (current->data == ptr) {
            current->is_free = 1;
            pool->allocated -= current->size;
            pool->deallocations++;
            
            // Coalesce with next free chunk
            if (current->next && current->next->is_free) {
                current->size += current->next->size + sizeof(memory_chunk_t);
                current->next = current->next->next;
            }
            
            // Coalesce with previous free chunk
            memory_chunk_t* prev = pool->first_chunk;
            while (prev && prev->next != current) {
                prev = prev->next;
            }
            
            if (prev && prev->is_free) {
                prev->size += current->size + sizeof(memory_chunk_t);
                prev->next = current->next;
            }
            
            return;
        }
        current = current->next;
    }
}

void* pool_test_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    // Test various allocations
    void* ptrs[10];
    for (int i = 0; i < 10; i++) {
        size_t size = (i + 1) * 64;
        ptrs[i] = pool_malloc(size);
        if (ptrs[i]) {
            memset(ptrs[i], i, size);
            printf("Thread %d: Allocated %zu bytes\n", thread_id, size);
        }
    }
    
    // Free some allocations
    for (int i = 0; i < 10; i += 2) {
        if (ptrs[i]) {
            pool_free(ptrs[i]);
            printf("Thread %d: Freed allocation %d\n", thread_id, i);
        }
    }
    
    return NULL;
}
```

### 3. Thread-Specific Configuration

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct {
    int debug_level;
    char log_prefix[64];
    int max_retries;
    double timeout_seconds;
    int enable_caching;
} thread_config_t;

// Default configuration
static const thread_config_t default_config = {
    .debug_level = 1,
    .log_prefix = "THREAD",
    .max_retries = 3,
    .timeout_seconds = 30.0,
    .enable_caching = 1
};

__thread thread_config_t config;
__thread int config_initialized = 0;

void init_thread_config() {
    if (!config_initialized) {
        config = default_config;
        snprintf(config.log_prefix, sizeof(config.log_prefix), 
                "THREAD_%lu", (unsigned long)pthread_self());
        config_initialized = 1;
        
        printf("%s: Configuration initialized\n", config.log_prefix);
    }
}

void set_debug_level(int level) {
    init_thread_config();
    config.debug_level = level;
    printf("%s: Debug level set to %d\n", config.log_prefix, level);
}

void set_timeout(double seconds) {
    init_thread_config();
    config.timeout_seconds = seconds;
    printf("%s: Timeout set to %.1f seconds\n", config.log_prefix, seconds);
}

void debug_log(int level, const char* message) {
    init_thread_config();
    
    if (level <= config.debug_level) {
        printf("[%s][DEBUG-%d]: %s\n", config.log_prefix, level, message);
    }
}

int perform_operation_with_retry(const char* operation) {
    init_thread_config();
    
    for (int attempt = 1; attempt <= config.max_retries; attempt++) {
        debug_log(2, operation);
        
        // Simulate operation that may fail
        if (rand() % 4 == 0) { // 25% success rate
            printf("%s: %s succeeded on attempt %d\n", 
                   config.log_prefix, operation, attempt);
            return 0;
        }
        
        printf("%s: %s failed, attempt %d/%d\n", 
               config.log_prefix, operation, attempt, config.max_retries);
        
        if (attempt < config.max_retries) {
            usleep(100000); // 100ms delay
        }
    }
    
    printf("%s: %s failed after %d attempts\n", 
           config.log_prefix, operation, config.max_retries);
    return -1;
}

void* config_test_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    // Customize configuration for this thread
    if (thread_id % 2 == 0) {
        set_debug_level(3);
        set_timeout(60.0);
    } else {
        set_debug_level(1);
        set_timeout(15.0);
    }
    
    // Perform operations with thread-specific configuration
    perform_operation_with_retry("Database connection");
    perform_operation_with_retry("File operation");
    perform_operation_with_retry("Network request");
    
    return NULL;
}
```

### 4. Thread-Local Performance Counters

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

typedef struct {
    unsigned long operations_count;
    unsigned long bytes_processed;
    unsigned long errors_count;
    struct timespec start_time;
    double total_execution_time;
    double min_operation_time;
    double max_operation_time;
} thread_stats_t;

__thread thread_stats_t stats = {0};
__thread int stats_initialized = 0;

void cleanup_thread_stats(void* data) {
    thread_stats_t* stats = (thread_stats_t*)data;
    
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double lifetime = (end_time.tv_sec - stats->start_time.tv_sec) +
                     (end_time.tv_nsec - stats->start_time.tv_nsec) / 1e9;
    
    printf("\n=== Thread %lu Final Statistics ===\n", (unsigned long)pthread_self());
    printf("Lifetime: %.3f seconds\n", lifetime);
    printf("Operations: %lu (%.2f ops/sec)\n", 
           stats->operations_count, stats->operations_count / lifetime);
    printf("Bytes processed: %lu (%.2f MB/sec)\n", 
           stats->bytes_processed, (stats->bytes_processed / 1048576.0) / lifetime);
    printf("Errors: %lu (%.2f%% error rate)\n", 
           stats->errors_count, 
           (double)stats->errors_count / stats->operations_count * 100);
    printf("Execution time: %.6f seconds\n", stats->total_execution_time);
    printf("Avg operation time: %.6f seconds\n", 
           stats->total_execution_time / stats->operations_count);
    printf("Min operation time: %.6f seconds\n", stats->min_operation_time);
    printf("Max operation time: %.6f seconds\n", stats->max_operation_time);
    
    free(data);
}

pthread_key_t stats_key;
pthread_once_t stats_init_once = PTHREAD_ONCE_INIT;

void init_stats_key() {
    pthread_key_create(&stats_key, cleanup_thread_stats);
}

void init_thread_stats() {
    if (!stats_initialized) {
        clock_gettime(CLOCK_MONOTONIC, &stats.start_time);
        stats.min_operation_time = 1e9; // Initialize to very large value
        stats_initialized = 1;
        
        pthread_once(&stats_init_once, init_stats_key);
        
        thread_stats_t* heap_stats = malloc(sizeof(thread_stats_t));
        *heap_stats = stats;
        pthread_setspecific(stats_key, heap_stats);
        
        printf("Thread %lu: Performance statistics initialized\n", 
               (unsigned long)pthread_self());
    }
}

void record_operation(size_t bytes, double execution_time, int had_error) {
    init_thread_stats();
    
    stats.operations_count++;
    stats.bytes_processed += bytes;
    stats.total_execution_time += execution_time;
    
    if (had_error) {
        stats.errors_count++;
    }
    
    if (execution_time < stats.min_operation_time) {
        stats.min_operation_time = execution_time;
    }
    
    if (execution_time > stats.max_operation_time) {
        stats.max_operation_time = execution_time;
    }
    
    // Update heap copy for cleanup
    thread_stats_t* heap_stats = pthread_getspecific(stats_key);
    if (heap_stats) {
        *heap_stats = stats;
    }
}

void print_current_stats() {
    if (!stats_initialized) return;
    
    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    
    double uptime = (current_time.tv_sec - stats.start_time.tv_sec) +
                   (current_time.tv_nsec - stats.start_time.tv_nsec) / 1e9;
    
    printf("Thread %lu: Ops=%lu, Bytes=%lu, Errors=%lu, Uptime=%.1fs\n",
           (unsigned long)pthread_self(),
           stats.operations_count, stats.bytes_processed, 
           stats.errors_count, uptime);
}

int simulate_work(size_t data_size) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Simulate variable work duration
    usleep((rand() % 1000) + 100); // 100-1100 microseconds
    
    // Simulate occasional errors
    int had_error = (rand() % 20 == 0); // 5% error rate
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double execution_time = (end.tv_sec - start.tv_sec) +
                           (end.tv_nsec - start.tv_nsec) / 1e9;
    
    record_operation(data_size, execution_time, had_error);
    
    return had_error ? -1 : 0;
}

void* performance_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    // Simulate workload
    for (int i = 0; i < 1000; i++) {
        size_t data_size = (rand() % 1024) + 64; // 64-1087 bytes
        simulate_work(data_size);
        
        if (i % 100 == 0) {
            print_current_stats();
        }
    }
    
    return NULL;
}
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
