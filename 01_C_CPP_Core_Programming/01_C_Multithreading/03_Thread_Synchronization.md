# Thread Synchronization Mechanisms

*Duration: 2 weeks*

## Overview

Thread synchronization is the cornerstone of safe multi-threaded programming. It encompasses the techniques and mechanisms used to coordinate the execution of multiple threads, ensuring data consistency, preventing race conditions, and maintaining program correctness in concurrent environments.

### Why Synchronization is Critical

Without proper synchronization, multi-threaded programs suffer from:

1. **Race Conditions**: Multiple threads access shared data simultaneously, leading to unpredictable results
2. **Data Corruption**: Inconsistent state due to concurrent modifications
3. **Lost Updates**: Thread modifications overwriting each other
4. **Visibility Issues**: Changes made by one thread not visible to others
5. **Atomicity Violations**: Complex operations being interrupted mid-execution

### Synchronization Hierarchy

Understanding the relationship between different synchronization primitives:

```
Low Level (Hardware)
├── Atomic Operations (Compare-and-Swap, Fetch-and-Add)
├── Memory Barriers/Fences
│
Mid Level (OS Primitives)
├── Mutexes (Mutual Exclusion)
├── Spinlocks (Busy-Wait Locks)
├── Read-Write Locks
├── Semaphores
│
High Level (Application Patterns)
├── Condition Variables
├── Barriers
├── Message Queues
└── Lock-Free Data Structures
```

### Performance vs Correctness Trade-offs

| Aspect | High Performance | High Correctness |
|--------|------------------|------------------|
| **Locking Strategy** | Lock-free, Fine-grained | Coarse-grained, Simple |
| **Synchronization Cost** | Minimal overhead | Safety first |
| **Complexity** | High implementation complexity | Lower complexity |
| **Debugging** | Difficult to debug | Easier to reason about |
| **Scalability** | Better scalability | May have bottlenecks |

### Synchronization Design Patterns

**Pattern 1: Producer-Consumer**
- Multiple producers generate data
- Multiple consumers process data
- Bounded buffer coordination

**Pattern 2: Reader-Writer**
- Many readers can access data simultaneously
- Writers need exclusive access
- Optimize for read-heavy workloads

**Pattern 3: Master-Worker**
- Master thread distributes work
- Worker threads process tasks
- Synchronize completion and results

**Pattern 4: Pipeline**
- Data flows through processing stages
- Each stage handled by different threads
- Synchronize stage transitions

## Understanding Race Conditions and Critical Sections

### Race Conditions Deep Dive

A **race condition** occurs when the correctness of a program depends on the relative timing or interleaving of multiple threads. The term "race" refers to threads "racing" to access shared resources.

#### Anatomy of a Race Condition

```c
// Global shared variable
int bank_balance = 1000;

// Thread function that withdraws money
void* withdraw_money(void* arg) {
    int amount = *(int*)arg;
    
    // Step 1: Read current balance
    int current_balance = bank_balance;
    
    // Step 2: Check if sufficient funds (could be interrupted here!)
    if (current_balance >= amount) {
        
        // Step 3: Simulate processing time
        usleep(1000);  // 1ms delay
        
        // Step 4: Update balance (could be interrupted here too!)
        bank_balance = current_balance - amount;
        
        printf("Withdrew %d, new balance: %d\n", amount, bank_balance);
    } else {
        printf("Insufficient funds for withdrawal of %d\n", amount);
    }
    
    return NULL;
}

// Demonstrating the race condition
void demonstrate_race_condition() {
    pthread_t thread1, thread2;
    int withdraw1 = 600, withdraw2 = 600;
    
    printf("Initial balance: %d\n", bank_balance);
    
    // Both threads try to withdraw 600 from 1000 balance
    pthread_create(&thread1, NULL, withdraw_money, &withdraw1);
    pthread_create(&thread2, NULL, withdraw_money, &withdraw2);
    
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    
    printf("Final balance: %d\n", bank_balance);
    // Result is unpredictable! Could be 400, -200, or other values
}
```

#### Types of Race Conditions

**1. Read-Modify-Write Race**
```c
// Classic counter increment race
int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 100000; i++) {
        counter++; // Not atomic! Three operations: read, add, write
    }
    return NULL;
}

// Assembly equivalent of counter++:
// LOAD R1, [counter]    // Read
// ADD  R1, R1, 1        // Modify  <- Context switch can happen here!
// STORE [counter], R1   // Write
```

**2. Check-Then-Act Race**
```c
// Checking and acting on shared state
if (resource_available) {        // Check
    use_resource();              // Act <- Resource might become unavailable here!
}
```

**3. Data Race vs Race Condition**
```c
// Data Race: Multiple threads access same memory location,
// at least one is a write, without synchronization
int shared_var = 0;

void* thread1(void* arg) {
    shared_var = 1;  // Write without synchronization
    return NULL;
}

void* thread2(void* arg) {
    printf("%d\n", shared_var);  // Read without synchronization
    return NULL;
}

// Race Condition: Timing-dependent incorrect behavior
// (Data races often cause race conditions, but not always)
```

### Critical Sections Explained

A **critical section** is a code segment that accesses shared resources (variables, files, hardware) that must not be accessed by more than one thread at a time.

#### Identifying Critical Sections

```c
#include <pthread.h>
#include <stdio.h>

// Shared resources
int shared_counter = 0;
FILE* shared_log_file = NULL;
char shared_buffer[1024];

void* worker_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    // Non-critical section - local variables only
    int local_counter = 0;
    char local_buffer[256];
    snprintf(local_buffer, sizeof(local_buffer), "Thread %d working", thread_id);
    
    // CRITICAL SECTION 1: Shared counter modification
    // ↓↓↓ START CRITICAL SECTION ↓↓↓
    shared_counter++;
    printf("Thread %d: Counter = %d\n", thread_id, shared_counter);
    // ↑↑↑ END CRITICAL SECTION ↑↑↑
    
    // Non-critical section - processing
    for (int i = 0; i < 1000; i++) {
        local_counter += i;
    }
    
    // CRITICAL SECTION 2: File writing
    // ↓↓↓ START CRITICAL SECTION ↓↓↓
    if (shared_log_file) {
        fprintf(shared_log_file, "Thread %d completed work\n", thread_id);
        fflush(shared_log_file);
    }
    // ↑↑↑ END CRITICAL SECTION ↑↑↑
    
    // CRITICAL SECTION 3: Shared buffer access
    // ↓↓↓ START CRITICAL SECTION ↓↓↓
    snprintf(shared_buffer, sizeof(shared_buffer), 
             "Last completed: Thread %d", thread_id);
    // ↑↑↑ END CRITICAL SECTION ↑↑↑
    
    return NULL;
}
```

#### Critical Section Properties

For correct synchronization, critical sections must guarantee:

1. **Mutual Exclusion**: At most one thread in critical section at any time
2. **Progress**: If no thread is in critical section, threads waiting to enter should be able to decide who enters next
3. **Bounded Waiting**: No thread should wait indefinitely to enter critical section
4. **Performance**: Critical sections should be as short as possible

#### Critical Section Design Principles

```c
// BAD: Critical section too large
pthread_mutex_lock(&mutex);
{
    // Large critical section reduces parallelism
    process_large_data_set();      // Could take seconds!
    update_shared_counter();       // Actually needs protection
    perform_complex_calculation(); // Doesn't need shared data
    write_to_shared_log();        // Needs protection
}
pthread_mutex_unlock(&mutex);

// GOOD: Minimize critical section size
{
    // Do non-shared work outside critical section
    process_large_data_set();
    int result = perform_complex_calculation();
    
    // Only protect the actual shared access
    pthread_mutex_lock(&mutex);
    update_shared_counter();
    pthread_mutex_unlock(&mutex);
    
    // Another minimal critical section
    pthread_mutex_lock(&mutex);
    write_to_shared_log();
    pthread_mutex_unlock(&mutex);
}
```

### Advanced Race Condition Examples

#### The ABA Problem
```c
// Lock-free stack implementation with ABA problem
typedef struct node {
    int data;
    struct node* next;
} node_t;

node_t* stack_top = NULL;

// This has the ABA problem!
int lock_free_pop() {
    node_t* top;
    node_t* next;
    
    do {
        top = stack_top;           // Read current top
        if (top == NULL) return -1; // Empty stack
        
        next = top->next;          // Read next node
        
        // Problem: Between here and the CAS below,
        // another thread could:
        // 1. Pop this node (A)
        // 2. Pop another node (B)  
        // 3. Push the first node back (A)
        // Now top still equals original value, but structure changed!
        
    } while (!__sync_bool_compare_and_swap(&stack_top, top, next));
    
    int data = top->data;
    free(top);
    return data;
}
```

#### Memory Reordering Issues
```c
// Without proper memory barriers, this can fail!
int data = 0;
int flag = 0;

// Thread 1 (Producer)
void* producer(void* arg) {
    data = 42;        // Write data
    flag = 1;         // Set flag - but this might be reordered before data write!
    return NULL;
}

// Thread 2 (Consumer)  
void* consumer(void* arg) {
    while (flag != 1); // Wait for flag
    
    // Data might still be 0 due to memory reordering!
    printf("Data: %d\n", data);
    return NULL;
}

// Solution: Use memory barriers or atomic operations
#include <stdatomic.h>
atomic_int atomic_data = 0;
atomic_int atomic_flag = 0;

void* safe_producer(void* arg) {
    atomic_store(&atomic_data, 42);
    atomic_store(&atomic_flag, 1);  // Guarantees ordering
    return NULL;
}

void* safe_consumer(void* arg) {
    while (atomic_load(&atomic_flag) != 1);
    printf("Data: %d\n", atomic_load(&atomic_data));
    return NULL;
}
```

## Mutex Locks (`pthread_mutex_t`) - The Foundation of Thread Safety

Mutex (Mutual Exclusion) locks are the most fundamental synchronization primitive, providing exclusive access to shared resources. They ensure that only one thread can hold the lock at any given time.

### Understanding Mutex Mechanics

#### How Mutexes Work Internally

```
Thread Attempts Lock:
┌─────────────────┐    Lock Available?    ┌─────────────────┐
│ Thread calls    │ ────Yes──────────────► │ Acquire lock    │
│ pthread_mutex_  │                        │ Enter critical  │
│ lock()          │                        │ section         │
└─────────────────┘                        └─────────────────┘
         │                                           │
         │                                           ▼
         │                ┌─────────────────┐    Execute     ┌─────────────────┐
         └───No──────────► │ Block thread    │ ◄──critical────│ pthread_mutex_  │
                          │ Add to wait     │    section     │ unlock()        │
                          │ queue           │                 │ Release lock    │
                          └─────────────────┘                 └─────────────────┘
                                   │                                   │
                                   └──────── Wake up waiting ─────────┘
                                            thread from queue
```

### Comprehensive Mutex Operations

#### Basic Mutex Initialization and Usage

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

// Method 1: Static initialization (preferred for global mutexes)
pthread_mutex_t global_mutex = PTHREAD_MUTEX_INITIALIZER;

// Method 2: Dynamic initialization
pthread_mutex_t dynamic_mutex;

int initialize_mutex_safe() {
    int result = pthread_mutex_init(&dynamic_mutex, NULL);
    if (result != 0) {
        fprintf(stderr, "Mutex initialization failed: %s\n", strerror(result));
        return -1;
    }
    return 0;
}

// Safe mutex operations with error checking
int safe_mutex_lock(pthread_mutex_t* mutex, const char* context) {
    int result = pthread_mutex_lock(mutex);
    if (result != 0) {
        fprintf(stderr, "Mutex lock failed in %s: %s\n", context, strerror(result));
    }
    return result;
}

int safe_mutex_unlock(pthread_mutex_t* mutex, const char* context) {
    int result = pthread_mutex_unlock(mutex);
    if (result != 0) {
        fprintf(stderr, "Mutex unlock failed in %s: %s\n", context, strerror(result));
    }
    return result;
}

// Example usage
void* worker_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    if (safe_mutex_lock(&global_mutex, "worker_thread") == 0) {
        // Critical section
        printf("Thread %d: In critical section\n", thread_id);
        sleep(1); // Simulate work
        
        safe_mutex_unlock(&global_mutex, "worker_thread");
    }
    
    return NULL;
}
```

### Advanced Mutex Types and Attributes

#### Mutex Types Comparison

```c
#include <pthread.h>

void demonstrate_mutex_types() {
    pthread_mutex_t normal_mutex;
    pthread_mutex_t recursive_mutex;
    pthread_mutex_t errorcheck_mutex;
    pthread_mutexattr_t attr;
    
    // Initialize attributes
    pthread_mutexattr_init(&attr);
    
    // 1. NORMAL MUTEX (Default)
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_NORMAL);
    pthread_mutex_init(&normal_mutex, &attr);
    // - Fastest performance
    // - No error checking
    // - Undefined behavior if locked twice by same thread
    
    // 2. RECURSIVE MUTEX
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&recursive_mutex, &attr);
    // - Same thread can lock multiple times
    // - Must unlock same number of times
    // - Slower than normal mutex
    
    // 3. ERROR CHECKING MUTEX
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutex_init(&errorcheck_mutex, &attr);
    // - Returns error on recursive lock attempts
    // - Returns error on unlock by non-owner
    // - Useful for debugging
    
    pthread_mutexattr_destroy(&attr);
}
```

#### Recursive Mutex Example

```c
#include <pthread.h>
#include <stdio.h>

pthread_mutex_t recursive_mutex;
int recursion_depth = 0;

void recursive_function(int depth) {
    pthread_mutex_lock(&recursive_mutex);
    recursion_depth++;
    
    printf("Recursion depth: %d (function depth: %d)\n", recursion_depth, depth);
    
    if (depth > 0) {
        recursive_function(depth - 1); // Recursive call - would deadlock with normal mutex!
    }
    
    recursion_depth--;
    pthread_mutex_unlock(&recursive_mutex);
}

void* recursive_worker(void* arg) {
    recursive_function(5);
    return NULL;
}

int main() {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&recursive_mutex, &attr);
    
    pthread_t thread;
    pthread_create(&thread, NULL, recursive_worker, NULL);
    pthread_join(thread, NULL);
    
    pthread_mutex_destroy(&recursive_mutex);
    pthread_mutexattr_destroy(&attr);
    
    return 0;
}
```

### Timed Mutex Operations

#### Non-blocking and Timeout Operations

```c
#include <pthread.h>
#include <time.h>
#include <errno.h>

// Try to acquire lock without blocking
int try_acquire_lock(pthread_mutex_t* mutex) {
    int result = pthread_mutex_trylock(mutex);
    
    switch (result) {
        case 0:
            printf("Lock acquired immediately\n");
            return 1; // Success
            
        case EBUSY:
            printf("Lock is currently held by another thread\n");
            return 0; // Lock not available
            
        default:
            fprintf(stderr, "Trylock error: %s\n", strerror(result));
            return -1; // Error
    }
}

// Acquire lock with timeout
int acquire_lock_with_timeout(pthread_mutex_t* mutex, int timeout_seconds) {
    struct timespec timeout;
    
    // Get current time
    if (clock_gettime(CLOCK_REALTIME, &timeout) != 0) {
        perror("clock_gettime");
        return -1;
    }
    
    // Add timeout
    timeout.tv_sec += timeout_seconds;
    
    int result = pthread_mutex_timedlock(mutex, &timeout);
    
    switch (result) {
        case 0:
            printf("Lock acquired within timeout\n");
            return 1; // Success
            
        case ETIMEDOUT:
            printf("Lock acquisition timed out after %d seconds\n", timeout_seconds);
            return 0; // Timeout
            
        default:
            fprintf(stderr, "Timedlock error: %s\n", strerror(result));
            return -1; // Error
    }
}

// Practical example: Resource manager with timeout
typedef struct {
    int resource_id;
    int in_use;
    pthread_mutex_t mutex;
} resource_t;

resource_t* create_resource(int id) {
    resource_t* resource = malloc(sizeof(resource_t));
    resource->resource_id = id;
    resource->in_use = 0;
    pthread_mutex_init(&resource->mutex, NULL);
    return resource;
}

int use_resource_with_timeout(resource_t* resource, int timeout_sec) {
    printf("Attempting to acquire resource %d...\n", resource->resource_id);
    
    if (acquire_lock_with_timeout(&resource->mutex, timeout_sec) == 1) {
        // Resource acquired
        resource->in_use = 1;
        printf("Using resource %d\n", resource->resource_id);
        
        // Simulate work
        sleep(2);
        
        resource->in_use = 0;
        pthread_mutex_unlock(&resource->mutex);
        printf("Released resource %d\n", resource->resource_id);
        return 1;
    }
    
    return 0; // Could not acquire resource
}
```

### Comprehensive Thread-Safe Data Structures

#### Thread-Safe Counter with Statistics

```c
#include <pthread.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    long long value;
    long long increment_count;
    long long decrement_count;
    double total_wait_time;
    pthread_mutex_t mutex;
} thread_safe_counter_t;

thread_safe_counter_t* counter_create() {
    thread_safe_counter_t* counter = malloc(sizeof(thread_safe_counter_t));
    counter->value = 0;
    counter->increment_count = 0;
    counter->decrement_count = 0;
    counter->total_wait_time = 0.0;
    pthread_mutex_init(&counter->mutex, NULL);
    return counter;
}

void counter_increment(thread_safe_counter_t* counter) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    pthread_mutex_lock(&counter->mutex);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double wait_time = (end.tv_sec - start.tv_sec) + 
                      (end.tv_nsec - start.tv_nsec) / 1e9;
    
    counter->value++;
    counter->increment_count++;
    counter->total_wait_time += wait_time;
    
    pthread_mutex_unlock(&counter->mutex);
}

void counter_decrement(thread_safe_counter_t* counter) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    pthread_mutex_lock(&counter->mutex);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double wait_time = (end.tv_sec - start.tv_sec) + 
                      (end.tv_nsec - start.tv_nsec) / 1e9;
    
    counter->value--;
    counter->decrement_count++;
    counter->total_wait_time += wait_time;
    
    pthread_mutex_unlock(&counter->mutex);
}

void counter_get_stats(thread_safe_counter_t* counter, 
                      long long* value, long long* increments, 
                      long long* decrements, double* avg_wait_time) {
    pthread_mutex_lock(&counter->mutex);
    
    *value = counter->value;
    *increments = counter->increment_count;
    *decrements = counter->decrement_count;
    
    long long total_operations = counter->increment_count + counter->decrement_count;
    *avg_wait_time = total_operations > 0 ? 
                    counter->total_wait_time / total_operations : 0.0;
    
    pthread_mutex_unlock(&counter->mutex);
}

// Example usage with multiple threads
void* counter_worker(void* arg) {
    thread_safe_counter_t* counter = (thread_safe_counter_t*)arg;
    
    for (int i = 0; i < 10000; i++) {
        if (i % 2 == 0) {
            counter_increment(counter);
        } else {
            counter_decrement(counter);
        }
        
        // Small delay to create contention
        usleep(1);
    }
    
    return NULL;
}

int main() {
    thread_safe_counter_t* counter = counter_create();
    pthread_t threads[8];
    
    printf("Starting counter test with 8 threads...\n");
    
    // Create threads
    for (int i = 0; i < 8; i++) {
        pthread_create(&threads[i], NULL, counter_worker, counter);
    }
    
    // Join threads
    for (int i = 0; i < 8; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Get final statistics
    long long value, increments, decrements;
    double avg_wait_time;
    counter_get_stats(counter, &value, &increments, &decrements, &avg_wait_time);
    
    printf("Final Results:\n");
    printf("  Counter value: %lld\n", value);
    printf("  Total increments: %lld\n", increments);
    printf("  Total decrements: %lld\n", decrements);
    printf("  Average wait time: %.6f seconds\n", avg_wait_time);
    
    free(counter);
    return 0;
}
```

#### Thread-Safe Hash Table

```c
#include <pthread.h>
#include <string.h>

#define HASH_TABLE_SIZE 1024

typedef struct hash_entry {
    char* key;
    void* value;
    struct hash_entry* next;
} hash_entry_t;

typedef struct {
    hash_entry_t* buckets[HASH_TABLE_SIZE];
    pthread_mutex_t bucket_mutexes[HASH_TABLE_SIZE]; // Fine-grained locking
    size_t size;
    pthread_mutex_t size_mutex;
} thread_safe_hash_table_t;

// Simple hash function
unsigned int hash(const char* key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % HASH_TABLE_SIZE;
}

thread_safe_hash_table_t* hash_table_create() {
    thread_safe_hash_table_t* table = malloc(sizeof(thread_safe_hash_table_t));
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        table->buckets[i] = NULL;
        pthread_mutex_init(&table->bucket_mutexes[i], NULL);
    }
    
    table->size = 0;
    pthread_mutex_init(&table->size_mutex, NULL);
    
    return table;
}

int hash_table_put(thread_safe_hash_table_t* table, const char* key, void* value) {
    unsigned int bucket_index = hash(key);
    
    pthread_mutex_lock(&table->bucket_mutexes[bucket_index]);
    
    // Check if key already exists
    hash_entry_t* entry = table->buckets[bucket_index];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            // Update existing entry
            entry->value = value;
            pthread_mutex_unlock(&table->bucket_mutexes[bucket_index]);
            return 1; // Updated existing
        }
        entry = entry->next;
    }
    
    // Create new entry
    hash_entry_t* new_entry = malloc(sizeof(hash_entry_t));
    new_entry->key = strdup(key);
    new_entry->value = value;
    new_entry->next = table->buckets[bucket_index];
    table->buckets[bucket_index] = new_entry;
    
    pthread_mutex_unlock(&table->bucket_mutexes[bucket_index]);
    
    // Update size (separate mutex to avoid holding bucket lock too long)
    pthread_mutex_lock(&table->size_mutex);
    table->size++;
    pthread_mutex_unlock(&table->size_mutex);
    
    return 0; // Created new entry
}

void* hash_table_get(thread_safe_hash_table_t* table, const char* key) {
    unsigned int bucket_index = hash(key);
    
    pthread_mutex_lock(&table->bucket_mutexes[bucket_index]);
    
    hash_entry_t* entry = table->buckets[bucket_index];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            void* value = entry->value;
            pthread_mutex_unlock(&table->bucket_mutexes[bucket_index]);
            return value;
        }
        entry = entry->next;
    }
    
    pthread_mutex_unlock(&table->bucket_mutexes[bucket_index]);
    return NULL; // Not found
}
```

## Read-Write Locks (`pthread_rwlock_t`) - Optimizing for Read-Heavy Workloads

Read-write locks (also known as shared-exclusive locks) are designed to optimize scenarios where data is read frequently but written infrequently. They allow multiple concurrent readers OR a single exclusive writer, but not both.

### Read-Write Lock Mechanics

#### Understanding the Access Patterns

```
Read-Write Lock States:
┌─────────────────┐
│   UNLOCKED      │ ──── Read Request ────► ┌─────────────────┐
│                 │                         │  READ LOCKED    │
│   No readers    │ ◄─── All readers ─────── │  (Shared)       │
│   No writers    │      release            │                 │
└─────────────────┘                         │  Multiple       │
         │                                  │  readers OK     │
         │                                  └─────────────────┘
         │                                           │
         │                                           │
    Write Request                              Write Request
         │                                      (blocks until
         ▼                                       all readers
┌─────────────────┐                             finish)
│  WRITE LOCKED   │                                  │
│  (Exclusive)    │                                  │
│                 │ ◄────────────────────────────────┘
│  Only one       │
│  writer allowed │
└─────────────────┘
```

### Comprehensive Read-Write Lock Implementation

#### Basic Operations with Error Handling

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

// Safe wrapper functions
int safe_rwlock_rdlock(pthread_rwlock_t* rwlock, const char* context) {
    int result = pthread_rwlock_rdlock(rwlock);
    if (result != 0) {
        fprintf(stderr, "Read lock failed in %s: %s\n", context, strerror(result));
    }
    return result;
}

int safe_rwlock_wrlock(pthread_rwlock_t* rwlock, const char* context) {
    int result = pthread_rwlock_wrlock(rwlock);
    if (result != 0) {
        fprintf(stderr, "Write lock failed in %s: %s\n", context, strerror(result));
    }
    return result;
}

int safe_rwlock_unlock(pthread_rwlock_t* rwlock, const char* context) {
    int result = pthread_rwlock_unlock(rwlock);
    if (result != 0) {
        fprintf(stderr, "Unlock failed in %s: %s\n", context, strerror(result));
    }
    return result;
}

// Advanced read-write lock with timeout
int safe_rwlock_rdlock_timeout(pthread_rwlock_t* rwlock, int timeout_sec) {
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_sec;
    
    int result = pthread_rwlock_timedrdlock(rwlock, &timeout);
    
    switch (result) {
        case 0:
            return 0; // Success
        case ETIMEDOUT:
            printf("Read lock timed out after %d seconds\n", timeout_sec);
            return result;
        default:
            fprintf(stderr, "Timed read lock failed: %s\n", strerror(result));
            return result;
    }
}

int safe_rwlock_wrlock_timeout(pthread_rwlock_t* rwlock, int timeout_sec) {
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_sec;
    
    int result = pthread_rwlock_timedwrlock(rwlock, &timeout);
    
    switch (result) {
        case 0:
            return 0; // Success
        case ETIMEDOUT:
            printf("Write lock timed out after %d seconds\n", timeout_sec);
            return result;
        default:
            fprintf(stderr, "Timed write lock failed: %s\n", strerror(result));
            return result;
    }
}
```

### Advanced Reader-Writer Examples

#### Example 1: Thread-Safe Database Simulation

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#define MAX_RECORDS 1000
#define MAX_KEY_LENGTH 32
#define MAX_VALUE_LENGTH 256

// Database record structure
typedef struct {
    char key[MAX_KEY_LENGTH];
    char value[MAX_VALUE_LENGTH];
    int valid;
} database_record_t;

// Database structure with read-write lock
typedef struct {
    database_record_t records[MAX_RECORDS];
    int record_count;
    pthread_rwlock_t rwlock;
    
    // Statistics
    long long read_count;
    long long write_count;
    double total_read_time;
    double total_write_time;
    pthread_mutex_t stats_mutex;
} thread_safe_database_t;

thread_safe_database_t* database_create() {
    thread_safe_database_t* db = malloc(sizeof(thread_safe_database_t));
    
    for (int i = 0; i < MAX_RECORDS; i++) {
        db->records[i].valid = 0;
    }
    
    db->record_count = 0;
    db->read_count = 0;
    db->write_count = 0;
    db->total_read_time = 0.0;
    db->total_write_time = 0.0;
    
    pthread_rwlock_init(&db->rwlock, NULL);
    pthread_mutex_init(&db->stats_mutex, NULL);
    
    return db;
}

// Read operation (multiple readers can execute concurrently)
int database_read(thread_safe_database_t* db, const char* key, char* value) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    if (safe_rwlock_rdlock(&db->rwlock, "database_read") != 0) {
        return -1;
    }
    
    printf("Reader [Thread %lu]: Searching for key '%s'\n", pthread_self(), key);
    
    int found = 0;
    for (int i = 0; i < db->record_count; i++) {
        if (db->records[i].valid && strcmp(db->records[i].key, key) == 0) {
            strcpy(value, db->records[i].value);
            found = 1;
            break;
        }
    }
    
    // Simulate read processing time
    usleep(100000 + rand() % 200000); // 100-300ms
    
    if (found) {
        printf("Reader [Thread %lu]: Found '%s' = '%s'\n", pthread_self(), key, value);
    } else {
        printf("Reader [Thread %lu]: Key '%s' not found\n", pthread_self(), key);
    }
    
    safe_rwlock_unlock(&db->rwlock, "database_read");
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double read_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Update statistics
    pthread_mutex_lock(&db->stats_mutex);
    db->read_count++;
    db->total_read_time += read_time;
    pthread_mutex_unlock(&db->stats_mutex);
    
    return found ? 1 : 0;
}

// Write operation (exclusive access required)
int database_write(thread_safe_database_t* db, const char* key, const char* value) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    if (safe_rwlock_wrlock(&db->rwlock, "database_write") != 0) {
        return -1;
    }
    
    printf("Writer [Thread %lu]: Writing '%s' = '%s'\n", pthread_self(), key, value);
    
    // Check if key already exists (update)
    int updated = 0;
    for (int i = 0; i < db->record_count; i++) {
        if (db->records[i].valid && strcmp(db->records[i].key, key) == 0) {
            strcpy(db->records[i].value, value);
            updated = 1;
            printf("Writer [Thread %lu]: Updated existing record\n", pthread_self());
            break;
        }
    }
    
    // Add new record if not found and space available
    if (!updated && db->record_count < MAX_RECORDS) {
        strcpy(db->records[db->record_count].key, key);
        strcpy(db->records[db->record_count].value, value);
        db->records[db->record_count].valid = 1;
        db->record_count++;
        printf("Writer [Thread %lu]: Added new record (total: %d)\n", 
               pthread_self(), db->record_count);
    }
    
    // Simulate write processing time
    usleep(300000 + rand() % 400000); // 300-700ms (longer than reads)
    
    safe_rwlock_unlock(&db->rwlock, "database_write");
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double write_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Update statistics
    pthread_mutex_lock(&db->stats_mutex);
    db->write_count++;
    db->total_write_time += write_time;
    pthread_mutex_unlock(&db->stats_mutex);
    
    return updated ? 1 : 0;
}

void database_print_stats(thread_safe_database_t* db) {
    pthread_mutex_lock(&db->stats_mutex);
    
    printf("\n=== Database Statistics ===\n");
    printf("Total records: %d\n", db->record_count);
    printf("Read operations: %lld\n", db->read_count);
    printf("Write operations: %lld\n", db->write_count);
    printf("Average read time: %.3f ms\n", 
           db->read_count > 0 ? (db->total_read_time / db->read_count) * 1000 : 0);
    printf("Average write time: %.3f ms\n",
           db->write_count > 0 ? (db->total_write_time / db->write_count) * 1000 : 0);
    printf("Total read time: %.3f seconds\n", db->total_read_time);
    printf("Total write time: %.3f seconds\n", db->total_write_time);
    
    pthread_mutex_unlock(&db->stats_mutex);
}

// Thread functions
typedef struct {
    thread_safe_database_t* db;
    int thread_id;
    int operation_count;
} thread_args_t;

void* reader_thread(void* arg) {
    thread_args_t* args = (thread_args_t*)arg;
    char keys[][16] = {"user1", "user2", "user3", "user4", "user5"};
    char value[MAX_VALUE_LENGTH];
    
    printf("Reader %d: Starting %d read operations\n", 
           args->thread_id, args->operation_count);
    
    for (int i = 0; i < args->operation_count; i++) {
        char* key = keys[rand() % 5];
        database_read(args->db, key, value);
        
        // Brief pause between operations
        usleep(50000 + rand() % 100000); // 50-150ms
    }
    
    printf("Reader %d: Completed all operations\n", args->thread_id);
    return NULL;
}

void* writer_thread(void* arg) {
    thread_args_t* args = (thread_args_t*)arg;
    char keys[][16] = {"user1", "user2", "user3", "user4", "user5"};
    
    printf("Writer %d: Starting %d write operations\n", 
           args->thread_id, args->operation_count);
    
    for (int i = 0; i < args->operation_count; i++) {
        char* key = keys[rand() % 5];
        char value[MAX_VALUE_LENGTH];
        snprintf(value, sizeof(value), "Data-%d-%d", args->thread_id, i);
        
        database_write(args->db, key, value);
        
        // Longer pause between write operations
        usleep(200000 + rand() % 300000); // 200-500ms
    }
    
    printf("Writer %d: Completed all operations\n", args->thread_id);
    return NULL;
}

int main() {
    srand(time(NULL));
    
    thread_safe_database_t* db = database_create();
    
    const int NUM_READERS = 6;
    const int NUM_WRITERS = 2;
    const int OPERATIONS_PER_THREAD = 8;
    
    pthread_t readers[NUM_READERS];
    pthread_t writers[NUM_WRITERS];
    thread_args_t reader_args[NUM_READERS];
    thread_args_t writer_args[NUM_WRITERS];
    
    printf("Starting database simulation:\n");
    printf("  Readers: %d (each doing %d operations)\n", NUM_READERS, OPERATIONS_PER_THREAD);
    printf("  Writers: %d (each doing %d operations)\n", NUM_WRITERS, OPERATIONS_PER_THREAD);
    printf("  Expected read-heavy workload...\n\n");
    
    // Create reader threads
    for (int i = 0; i < NUM_READERS; i++) {
        reader_args[i].db = db;
        reader_args[i].thread_id = i + 1;
        reader_args[i].operation_count = OPERATIONS_PER_THREAD;
        pthread_create(&readers[i], NULL, reader_thread, &reader_args[i]);
    }
    
    // Create writer threads
    for (int i = 0; i < NUM_WRITERS; i++) {
        writer_args[i].db = db;
        writer_args[i].thread_id = i + 1;
        writer_args[i].operation_count = OPERATIONS_PER_THREAD;
        pthread_create(&writers[i], NULL, writer_thread, &writer_args[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < NUM_READERS; i++) {
        pthread_join(readers[i], NULL);
    }
    
    for (int i = 0; i < NUM_WRITERS; i++) {
        pthread_join(writers[i], NULL);
    }
    
    database_print_stats(db);
    
    // Cleanup
    pthread_rwlock_destroy(&db->rwlock);
    pthread_mutex_destroy(&db->stats_mutex);
    free(db);
    
    return 0;
}
```

#### Example 2: Configuration Manager with Read-Write Locks

```c
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define MAX_CONFIG_ENTRIES 100
#define MAX_KEY_LENGTH 64
#define MAX_VALUE_LENGTH 256

typedef struct {
    char key[MAX_KEY_LENGTH];
    char value[MAX_VALUE_LENGTH];
} config_entry_t;

typedef struct {
    config_entry_t entries[MAX_CONFIG_ENTRIES];
    int entry_count;
    pthread_rwlock_t rwlock;
    int version; // Configuration version number
} config_manager_t;

config_manager_t* config_create() {
    config_manager_t* config = malloc(sizeof(config_manager_t));
    config->entry_count = 0;
    config->version = 1;
    pthread_rwlock_init(&config->rwlock, NULL);
    return config;
}

// Read configuration value (concurrent reads allowed)
int config_get(config_manager_t* config, const char* key, char* value) {
    pthread_rwlock_rdlock(&config->rwlock);
    
    int found = 0;
    for (int i = 0; i < config->entry_count; i++) {
        if (strcmp(config->entries[i].key, key) == 0) {
            strcpy(value, config->entries[i].value);
            found = 1;
            break;
        }
    }
    
    printf("Config read [v%d]: %s = %s\n", 
           config->version, key, found ? value : "NOT_FOUND");
    
    pthread_rwlock_unlock(&config->rwlock);
    return found;
}

// Update configuration (exclusive access required)
void config_set(config_manager_t* config, const char* key, const char* value) {
    pthread_rwlock_wrlock(&config->rwlock);
    
    // Check if key exists (update)
    int updated = 0;
    for (int i = 0; i < config->entry_count; i++) {
        if (strcmp(config->entries[i].key, key) == 0) {
            strcpy(config->entries[i].value, value);
            updated = 1;
            break;
        }
    }
    
    // Add new entry if not found
    if (!updated && config->entry_count < MAX_CONFIG_ENTRIES) {
        strcpy(config->entries[config->entry_count].key, key);
        strcpy(config->entries[config->entry_count].value, value);
        config->entry_count++;
    }
    
    config->version++; // Increment version on any change
    
    printf("Config write [v%d]: %s = %s %s\n", 
           config->version, key, value, updated ? "(updated)" : "(new)");
    
    pthread_rwlock_unlock(&config->rwlock);
}

// Bulk configuration reload (exclusive access)
void config_reload_from_file(config_manager_t* config, const char* filename) {
    pthread_rwlock_wrlock(&config->rwlock);
    
    printf("Config reload: Starting bulk reload from %s\n", filename);
    
    // Clear existing configuration
    config->entry_count = 0;
    
    // Simulate loading from file
    const char* sample_config[][2] = {
        {"database_host", "localhost"},
        {"database_port", "5432"},
        {"max_connections", "100"},
        {"timeout_seconds", "30"},
        {"debug_enabled", "false"}
    };
    
    for (int i = 0; i < 5; i++) {
        strcpy(config->entries[i].key, sample_config[i][0]);
        strcpy(config->entries[i].value, sample_config[i][1]);
        config->entry_count++;
        
        // Simulate file I/O time
        usleep(50000);
    }
    
    config->version++;
    
    printf("Config reload: Completed reload, version %d, %d entries\n", 
           config->version, config->entry_count);
    
    pthread_rwlock_unlock(&config->rwlock);
}
```

### Read-Write Lock Performance Analysis

#### Comparing Mutex vs RWLock Performance

```c
#include <pthread.h>
#include <time.h>

typedef struct {
    int data[1000];
    pthread_mutex_t mutex;
    pthread_rwlock_t rwlock;
} shared_data_t;

// Benchmark function for mutex-only approach
double benchmark_mutex_reads(shared_data_t* data, int read_count) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < read_count; i++) {
        pthread_mutex_lock(&data->mutex);
        
        // Simulate read operation
        volatile int sum = 0;
        for (int j = 0; j < 1000; j++) {
            sum += data->data[j];
        }
        
        pthread_mutex_unlock(&data->mutex);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// Benchmark function for read-write lock approach
double benchmark_rwlock_reads(shared_data_t* data, int read_count) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < read_count; i++) {
        pthread_rwlock_rdlock(&data->rwlock);
        
        // Simulate read operation
        volatile int sum = 0;
        for (int j = 0; j < 1000; j++) {
            sum += data->data[j];
        }
        
        pthread_rwlock_unlock(&data->rwlock);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

void performance_comparison() {
    shared_data_t data;
    pthread_mutex_init(&data.mutex, NULL);
    pthread_rwlock_init(&data.rwlock, NULL);
    
    // Initialize data
    for (int i = 0; i < 1000; i++) {
        data.data[i] = i;
    }
    
    const int READ_COUNT = 10000;
    
    printf("Performance Comparison (Single Thread):\n");
    
    double mutex_time = benchmark_mutex_reads(&data, READ_COUNT);
    printf("  Mutex-only approach: %.3f seconds\n", mutex_time);
    
    double rwlock_time = benchmark_rwlock_reads(&data, READ_COUNT);
    printf("  Read-write lock approach: %.3f seconds\n", rwlock_time);
    
    printf("  Performance difference: %.1f%%\n", 
           ((mutex_time - rwlock_time) / mutex_time) * 100);
    
    pthread_mutex_destroy(&data.mutex);
    pthread_rwlock_destroy(&data.rwlock);
}
```

## Spinlocks - High-Performance Locking for Short Critical Sections

Spinlocks are a type of lock where threads actively "spin" (busy-wait) in a loop until the lock becomes available, rather than being put to sleep. They are optimal for very short critical sections where the overhead of blocking and waking threads would be greater than spinning.

### When to Use Spinlocks

**✅ Good for:**
- Very short critical sections (few instructions)
- High-frequency locking scenarios
- Real-time systems where blocking is unacceptable
- Multi-core systems with available CPU cycles
- Kernel-level code

**❌ Bad for:**
- Long critical sections
- Single-core systems
- I/O operations within critical sections
- Memory allocation within critical sections
- When lock contention is high

### Spinlock Implementation and Usage

#### Basic Spinlock Operations

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// Basic spinlock usage
pthread_spinlock_t spinlock;

int initialize_spinlock() {
    // PTHREAD_PROCESS_PRIVATE: Only threads within same process
    // PTHREAD_PROCESS_SHARED: Threads across different processes
    int result = pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);
    if (result != 0) {
        fprintf(stderr, "Spinlock initialization failed: %s\n", strerror(result));
        return -1;
    }
    return 0;
}

void safe_spinlock_operation() {
    // Acquire spinlock
    if (pthread_spin_lock(&spinlock) != 0) {
        fprintf(stderr, "Failed to acquire spinlock\n");
        return;
    }
    
    // CRITICAL SECTION - KEEP VERY SHORT!
    // Only a few simple operations here
    static int counter = 0;
    counter++;
    printf("Counter: %d (Thread: %lu)\n", counter, pthread_self());
    
    // Release spinlock
    if (pthread_spin_unlock(&spinlock) != 0) {
        fprintf(stderr, "Failed to release spinlock\n");
    }
}

void cleanup_spinlock() {
    pthread_spin_destroy(&spinlock);
}
```

#### Spinlock vs Mutex Performance Comparison

```c
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define NUM_ITERATIONS 1000000
#define NUM_THREADS 4

// Global variables for testing
volatile int spinlock_counter = 0;
volatile int mutex_counter = 0;
pthread_spinlock_t test_spinlock;
pthread_mutex_t test_mutex = PTHREAD_MUTEX_INITIALIZER;

// Performance measurement structure
typedef struct {
    double total_time;
    int iterations_per_thread;
    int thread_id;
} perf_result_t;

// Spinlock performance test
void* spinlock_worker(void* arg) {
    perf_result_t* result = (perf_result_t*)arg;
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < result->iterations_per_thread; i++) {
        pthread_spin_lock(&test_spinlock);
        
        // Very short critical section - just increment
        spinlock_counter++;
        
        pthread_spin_unlock(&test_spinlock);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    result->total_time = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    
    return result;
}

// Mutex performance test
void* mutex_worker(void* arg) {
    perf_result_t* result = (perf_result_t*)arg;
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < result->iterations_per_thread; i++) {
        pthread_mutex_lock(&test_mutex);
        
        // Same short critical section
        mutex_counter++;
        
        pthread_mutex_unlock(&test_mutex);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    result->total_time = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    
    return result;
}

void performance_comparison() {
    pthread_t threads[NUM_THREADS];
    perf_result_t results[NUM_THREADS];
    int iterations_per_thread = NUM_ITERATIONS / NUM_THREADS;
    
    // Initialize spinlock
    pthread_spin_init(&test_spinlock, PTHREAD_PROCESS_PRIVATE);
    
    printf("Performance Comparison: Spinlock vs Mutex\n");
    printf("Total iterations: %d, Threads: %d, Iterations per thread: %d\n\n", 
           NUM_ITERATIONS, NUM_THREADS, iterations_per_thread);
    
    // Test Spinlock Performance
    printf("Testing Spinlock Performance...\n");
    spinlock_counter = 0;
    
    struct timespec spinlock_start, spinlock_end;
    clock_gettime(CLOCK_MONOTONIC, &spinlock_start);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        results[i].iterations_per_thread = iterations_per_thread;
        results[i].thread_id = i;
        pthread_create(&threads[i], NULL, spinlock_worker, &results[i]);
    }
    
    double total_spinlock_time = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        perf_result_t* result;
        pthread_join(threads[i], (void**)&result);
        total_spinlock_time += result->total_time;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &spinlock_end);
    double spinlock_wall_time = (spinlock_end.tv_sec - spinlock_start.tv_sec) + 
                               (spinlock_end.tv_nsec - spinlock_start.tv_nsec) / 1e9;
    
    printf("Spinlock Results:\n");
    printf("  Final counter: %d (expected: %d)\n", spinlock_counter, NUM_ITERATIONS);
    printf("  Wall clock time: %.3f seconds\n", spinlock_wall_time);
    printf("  Total thread time: %.3f seconds\n", total_spinlock_time);
    printf("  Average time per operation: %.0f nanoseconds\n", 
           (total_spinlock_time / NUM_ITERATIONS) * 1e9);
    
    // Test Mutex Performance
    printf("\nTesting Mutex Performance...\n");
    mutex_counter = 0;
    
    struct timespec mutex_start, mutex_end;
    clock_gettime(CLOCK_MONOTONIC, &mutex_start);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        results[i].iterations_per_thread = iterations_per_thread;
        results[i].thread_id = i;
        pthread_create(&threads[i], NULL, mutex_worker, &results[i]);
    }
    
    double total_mutex_time = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        perf_result_t* result;
        pthread_join(threads[i], (void**)&result);
        total_mutex_time += result->total_time;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &mutex_end);
    double mutex_wall_time = (mutex_end.tv_sec - mutex_start.tv_sec) + 
                            (mutex_end.tv_nsec - mutex_start.tv_nsec) / 1e9;
    
    printf("Mutex Results:\n");
    printf("  Final counter: %d (expected: %d)\n", mutex_counter, NUM_ITERATIONS);
    printf("  Wall clock time: %.3f seconds\n", mutex_wall_time);
    printf("  Total thread time: %.3f seconds\n", total_mutex_time);
    printf("  Average time per operation: %.0f nanoseconds\n", 
           (total_mutex_time / NUM_ITERATIONS) * 1e9);
    
    // Performance comparison
    printf("\nPerformance Analysis:\n");
    printf("  Spinlock vs Mutex speedup: %.2fx\n", mutex_wall_time / spinlock_wall_time);
    printf("  CPU efficiency (spinlock): %.1f%% (lower is more spinning)\n", 
           (spinlock_wall_time / total_spinlock_time) * 100);
    printf("  CPU efficiency (mutex): %.1f%%\n", 
           (mutex_wall_time / total_mutex_time) * 100);
    
    // Cleanup
    pthread_spin_destroy(&test_spinlock);
    pthread_mutex_destroy(&test_mutex);
}
```

#### Advanced Spinlock Example: Lock-Free Queue with Spinlock Fallback

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>

#define QUEUE_SIZE 1024

// Lock-free queue with spinlock fallback for debugging
typedef struct {
    int data[QUEUE_SIZE];
    atomic_int head;
    atomic_int tail;
    pthread_spinlock_t debug_lock; // For statistics only
    
    // Statistics
    long enqueue_count;
    long dequeue_count;
    long contention_count;
} lockfree_queue_t;

lockfree_queue_t* queue_create() {
    lockfree_queue_t* queue = malloc(sizeof(lockfree_queue_t));
    atomic_init(&queue->head, 0);
    atomic_init(&queue->tail, 0);
    queue->enqueue_count = 0;
    queue->dequeue_count = 0;
    queue->contention_count = 0;
    pthread_spin_init(&queue->debug_lock, PTHREAD_PROCESS_PRIVATE);
    return queue;
}

int queue_enqueue(lockfree_queue_t* queue, int value) {
    int current_tail, next_tail;
    int attempts = 0;
    
    do {
        current_tail = atomic_load(&queue->tail);
        next_tail = (current_tail + 1) % QUEUE_SIZE;
        
        // Check if queue is full
        if (next_tail == atomic_load(&queue->head)) {
            return 0; // Queue full
        }
        
        attempts++;
        if (attempts > 10) {
            // High contention detected
            pthread_spin_lock(&queue->debug_lock);
            queue->contention_count++;
            pthread_spin_unlock(&queue->debug_lock);
        }
        
    } while (!atomic_compare_exchange_weak(&queue->tail, &current_tail, next_tail));
    
    // Store the data
    queue->data[current_tail] = value;
    
    // Update statistics (using spinlock for simplicity)
    pthread_spin_lock(&queue->debug_lock);
    queue->enqueue_count++;
    pthread_spin_unlock(&queue->debug_lock);
    
    return 1; // Success
}

int queue_dequeue(lockfree_queue_t* queue, int* value) {
    int current_head, next_head;
    int attempts = 0;
    
    do {
        current_head = atomic_load(&queue->head);
        
        // Check if queue is empty
        if (current_head == atomic_load(&queue->tail)) {
            return 0; // Queue empty
        }
        
        next_head = (current_head + 1) % QUEUE_SIZE;
        
        attempts++;
        if (attempts > 10) {
            pthread_spin_lock(&queue->debug_lock);
            queue->contention_count++;
            pthread_spin_unlock(&queue->debug_lock);
        }
        
    } while (!atomic_compare_exchange_weak(&queue->head, &current_head, next_head));
    
    // Retrieve the data
    *value = queue->data[current_head];
    
    // Update statistics
    pthread_spin_lock(&queue->debug_lock);
    queue->dequeue_count++;
    pthread_spin_unlock(&queue->debug_lock);
    
    return 1; // Success
}

void queue_print_stats(lockfree_queue_t* queue) {
    pthread_spin_lock(&queue->debug_lock);
    
    printf("Queue Statistics:\n");
    printf("  Enqueue operations: %ld\n", queue->enqueue_count);
    printf("  Dequeue operations: %ld\n", queue->dequeue_count);
    printf("  Contention events: %ld\n", queue->contention_count);
    printf("  Current size: %d\n", 
           (atomic_load(&queue->tail) - atomic_load(&queue->head) + QUEUE_SIZE) % QUEUE_SIZE);
    
    pthread_spin_unlock(&queue->debug_lock);
}

// Example usage with producer-consumer
typedef struct {
    lockfree_queue_t* queue;
    int thread_id;
    int operations;
} queue_worker_args_t;

void* producer_thread(void* arg) {
    queue_worker_args_t* args = (queue_worker_args_t*)arg;
    
    for (int i = 0; i < args->operations; i++) {
        int value = args->thread_id * 1000 + i;
        
        while (!queue_enqueue(args->queue, value)) {
            // Queue full, brief spin
            usleep(1);
        }
        
        // Brief pause to simulate work
        usleep(10);
    }
    
    printf("Producer %d completed %d enqueues\n", args->thread_id, args->operations);
    return NULL;
}

void* consumer_thread(void* arg) {
    queue_worker_args_t* args = (queue_worker_args_t*)arg;
    int consumed = 0;
    
    while (consumed < args->operations) {
        int value;
        if (queue_dequeue(args->queue, &value)) {
            consumed++;
            // Brief pause to simulate processing
            usleep(15);
        } else {
            // Queue empty, brief spin
            usleep(1);
        }
    }
    
    printf("Consumer %d completed %d dequeues\n", args->thread_id, consumed);
    return NULL;
}

int main() {
    lockfree_queue_t* queue = queue_create();
    
    const int NUM_PRODUCERS = 3;
    const int NUM_CONSUMERS = 2;
    const int OPERATIONS_PER_THREAD = 1000;
    
    pthread_t producers[NUM_PRODUCERS];
    pthread_t consumers[NUM_CONSUMERS];
    queue_worker_args_t producer_args[NUM_PRODUCERS];
    queue_worker_args_t consumer_args[NUM_CONSUMERS];
    
    printf("Starting lock-free queue test with spinlock statistics\n");
    printf("Producers: %d, Consumers: %d, Operations each: %d\n\n", 
           NUM_PRODUCERS, NUM_CONSUMERS, OPERATIONS_PER_THREAD);
    
    // Create producers
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        producer_args[i].queue = queue;
        producer_args[i].thread_id = i + 1;
        producer_args[i].operations = OPERATIONS_PER_THREAD;
        pthread_create(&producers[i], NULL, producer_thread, &producer_args[i]);
    }
    
    // Create consumers
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        consumer_args[i].queue = queue;
        consumer_args[i].thread_id = i + 1;
        consumer_args[i].operations = (NUM_PRODUCERS * OPERATIONS_PER_THREAD) / NUM_CONSUMERS;
        pthread_create(&consumers[i], NULL, consumer_thread, &consumer_args[i]);
    }
    
    // Wait for completion
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        pthread_join(producers[i], NULL);
    }
    
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        pthread_join(consumers[i], NULL);
    }
    
    printf("\nTest completed!\n");
    queue_print_stats(queue);
    
    // Cleanup
    pthread_spin_destroy(&queue->debug_lock);
    free(queue);
    
    return 0;
}
```

### Spinlock Design Considerations

#### CPU Architecture Impact

```c
// Different spinlock implementations based on architecture
#if defined(__x86_64__) || defined(__i386__)
    // x86/x64: Use PAUSE instruction to reduce power consumption
    #define CPU_RELAX() __asm__ __volatile__("pause" ::: "memory")
#elif defined(__aarch64__) || defined(__arm__)
    // ARM: Use YIELD instruction
    #define CPU_RELAX() __asm__ __volatile__("yield" ::: "memory")
#else
    // Generic: Compiler memory barrier
    #define CPU_RELAX() __asm__ __volatile__("" ::: "memory")
#endif

// Custom spinlock with CPU-specific optimizations
typedef struct {
    atomic_flag locked;
    int spin_count;
} optimized_spinlock_t;

void optimized_spin_lock(optimized_spinlock_t* lock) {
    int spins = 0;
    
    while (atomic_flag_test_and_set_explicit(&lock->locked, memory_order_acquire)) {
        // Adaptive spinning strategy
        if (spins < 1000) {
            // Short spins: just pause
            CPU_RELAX();
            spins++;
        } else if (spins < 10000) {
            // Medium spins: pause longer
            for (int i = 0; i < 10; i++) {
                CPU_RELAX();
            }
            spins++;
        } else {
            // Long contention: yield to other threads
            sched_yield();
            spins = 0; // Reset spin count
        }
    }
    
    lock->spin_count += spins;
}

void optimized_spin_unlock(optimized_spinlock_t* lock) {
    atomic_flag_clear_explicit(&lock->locked, memory_order_release);
}
```

## Barriers

Barriers are synchronization primitives that force a group of threads to wait until all threads in the group have reached a specific synchronization point. They're essential for implementing parallel algorithms where computation must proceed in phases.

### How Barriers Work

```
Thread 1: ------>|     |------>
Thread 2: ---->  |     |    -->
Thread 3: ------> |WAIT |------>
Thread 4: -->     |     |      >
                 Barrier Point
```

When a thread reaches a barrier:
1. It decrements the barrier counter
2. If it's not the last thread, it blocks
3. When the last thread arrives, all threads are released simultaneously

### Barrier API and Usage

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

// Barrier initialization with error handling
int safe_barrier_init(pthread_barrier_t* barrier, int count, const char* context) {
    int result = pthread_barrier_init(barrier, NULL, count);
    if (result != 0) {
        fprintf(stderr, "Barrier initialization failed in %s: %s\n", 
                context, strerror(result));
        return -1;
    }
    printf("Barrier initialized for %d threads in %s\n", count, context);
    return 0;
}

// Safe barrier wait with error handling
int safe_barrier_wait(pthread_barrier_t* barrier, int thread_id, const char* phase) {
    printf("Thread %d waiting at barrier (%s)\n", thread_id, phase);
    
    int result = pthread_barrier_wait(barrier);
    
    if (result == 0) {
        printf("Thread %d released from barrier (%s)\n", thread_id, phase);
        return 0;
    } else if (result == PTHREAD_BARRIER_SERIAL_THREAD) {
        printf("Thread %d was the last to reach barrier (%s) - serial thread\n", 
               thread_id, phase);
        return 1; // Special return value for serial thread
    } else {
        fprintf(stderr, "Barrier wait failed for thread %d (%s): %s\n", 
                thread_id, phase, strerror(result));
        return -1;
    }
}

// Barrier cleanup
int safe_barrier_destroy(pthread_barrier_t* barrier, const char* context) {
    int result = pthread_barrier_destroy(barrier);
    if (result != 0) {
        fprintf(stderr, "Barrier destruction failed in %s: %s\n", 
                context, strerror(result));
        return -1;
    }
    printf("Barrier destroyed in %s\n", context);
    return 0;
}
```

### Advanced Barrier Examples

#### Example 1: Multi-Phase Parallel Algorithm

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#define NUM_THREADS 4
#define ARRAY_SIZE 1000
#define NUM_PHASES 3

typedef struct {
    int thread_id;
    int* data;
    int data_size;
    pthread_barrier_t* barrier;
    
    // Phase-specific data
    double* phase_results;
    int phase_count;
} thread_data_t;

// Simulate different computational phases
void* multi_phase_computation(void* arg) {
    thread_data_t* tdata = (thread_data_t*)arg;
    int id = tdata->thread_id;
    int chunk_size = tdata->data_size / NUM_THREADS;
    int start = id * chunk_size;
    int end = (id == NUM_THREADS - 1) ? tdata->data_size : (id + 1) * chunk_size;
    
    printf("Thread %d processing range [%d, %d)\n", id, start, end);
    
    // Phase 1: Data preprocessing
    printf("Thread %d starting Phase 1: Data preprocessing\n", id);
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        tdata->data[i] = tdata->data[i] * 2; // Simple preprocessing
        sum += tdata->data[i];
    }
    tdata->phase_results[id * NUM_PHASES + 0] = sum;
    
    // Synchronize after Phase 1
    int barrier_result = safe_barrier_wait(tdata->barrier, id, "Phase 1 Complete");
    
    // Only the serial thread performs global operations
    if (barrier_result == 1) {
        printf("Serial thread performing global Phase 1 operations\n");
        // Global operation could go here
    }
    
    // Phase 2: Complex computation
    printf("Thread %d starting Phase 2: Complex computation\n", id);
    double variance = 0.0;
    double mean = sum / (end - start);
    for (int i = start; i < end; i++) {
        double diff = tdata->data[i] - mean;
        variance += diff * diff;
    }
    tdata->phase_results[id * NUM_PHASES + 1] = variance / (end - start);
    
    // Synchronize after Phase 2
    barrier_result = safe_barrier_wait(tdata->barrier, id, "Phase 2 Complete");
    
    if (barrier_result == 1) {
        printf("Serial thread performing global Phase 2 operations\n");
    }
    
    // Phase 3: Final processing
    printf("Thread %d starting Phase 3: Final processing\n", id);
    double final_result = 0.0;
    for (int i = start; i < end; i++) {
        final_result += sqrt(abs(tdata->data[i]));
    }
    tdata->phase_results[id * NUM_PHASES + 2] = final_result;
    
    // Final synchronization
    barrier_result = safe_barrier_wait(tdata->barrier, id, "Phase 3 Complete");
    
    if (barrier_result == 1) {
        printf("All phases completed. Computing final results...\n");
        
        // Aggregate results from all threads
        for (int phase = 0; phase < NUM_PHASES; phase++) {
            double total = 0.0;
            printf("Phase %d results: ", phase + 1);
            for (int t = 0; t < NUM_THREADS; t++) {
                double result = tdata->phase_results[t * NUM_PHASES + phase];
                printf("T%d=%.2f ", t, result);
                total += result;
            }
            printf("Total=%.2f\n", total);
        }
    }
    
    printf("Thread %d completed all phases\n", id);
    return NULL;
}

// Demo function for multi-phase barrier synchronization
void demo_multi_phase_barriers() {
    printf("\n=== Multi-Phase Barrier Synchronization Demo ===\n");
    
    pthread_t threads[NUM_THREADS];
    pthread_barrier_t barrier;
    thread_data_t thread_data[NUM_THREADS];
    
    // Initialize data
    int* shared_data = malloc(ARRAY_SIZE * sizeof(int));
    double* phase_results = malloc(NUM_THREADS * NUM_PHASES * sizeof(double));
    
    // Fill with sample data
    srand(time(NULL));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        shared_data[i] = rand() % 100 + 1;
    }
    
    // Initialize barrier
    if (safe_barrier_init(&barrier, NUM_THREADS, "Multi-phase computation") != 0) {
        free(shared_data);
        free(phase_results);
        return;
    }
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].data = shared_data;
        thread_data[i].data_size = ARRAY_SIZE;
        thread_data[i].barrier = &barrier;
        thread_data[i].phase_results = phase_results;
        thread_data[i].phase_count = NUM_PHASES;
        
        if (pthread_create(&threads[i], NULL, multi_phase_computation, &thread_data[i]) != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            break;
        }
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Cleanup
    safe_barrier_destroy(&barrier, "Multi-phase computation");
    free(shared_data);
    free(phase_results);
    
    printf("Multi-phase barrier demo completed\n");
}
```

#### Example 2: Iterative Parallel Algorithm with Barriers

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define GRID_SIZE 100
#define NUM_THREADS 4
#define MAX_ITERATIONS 50
#define CONVERGENCE_THRESHOLD 0.001

typedef struct {
    int thread_id;
    double** grid;
    double** new_grid;
    int start_row;
    int end_row;
    pthread_barrier_t* iteration_barrier;
    pthread_barrier_t* convergence_barrier;
    double* thread_max_diff;
    int* global_converged;
    int iteration;
} iterative_thread_data_t;

// Jacobi iteration for solving heat equation
void* jacobi_iteration(void* arg) {
    iterative_thread_data_t* tdata = (iterative_thread_data_t*)arg;
    int id = tdata->thread_id;
    
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        tdata->iteration = iter;
        double max_diff = 0.0;
        
        // Compute new values for assigned rows
        for (int i = tdata->start_row; i < tdata->end_row; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {
                double old_val = tdata->grid[i][j];
                double new_val = 0.25 * (tdata->grid[i-1][j] + tdata->grid[i+1][j] + 
                                        tdata->grid[i][j-1] + tdata->grid[i][j+1]);
                tdata->new_grid[i][j] = new_val;
                
                double diff = fabs(new_val - old_val);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }
        
        // Store thread's maximum difference
        tdata->thread_max_diff[id] = max_diff;
        
        // Wait for all threads to complete their computation
        int barrier_result = safe_barrier_wait(tdata->iteration_barrier, id, "Iteration computation");
        
        // Serial thread checks for convergence
        if (barrier_result == 1) {
            double global_max_diff = 0.0;
            for (int t = 0; t < NUM_THREADS; t++) {
                if (tdata->thread_max_diff[t] > global_max_diff) {
                    global_max_diff = tdata->thread_max_diff[t];
                }
            }
            
            printf("Iteration %d: Max difference = %f\n", iter, global_max_diff);
            
            if (global_max_diff < CONVERGENCE_THRESHOLD) {
                *(tdata->global_converged) = 1;
                printf("Converged after %d iterations!\n", iter + 1);
            }
            
            // Swap grids
            double** temp = tdata->grid;
            tdata->grid = tdata->new_grid;
            tdata->new_grid = temp;
        }
        
        // Wait for convergence check and grid swap
        safe_barrier_wait(tdata->convergence_barrier, id, "Convergence check");
        
        // Check if converged
        if (*(tdata->global_converged)) {
            printf("Thread %d exiting due to convergence\n", id);
            break;
        }
    }
    
    return NULL;
}
```

### Barrier Performance Considerations

#### Performance Characteristics

| Aspect | Impact | Optimization |
|--------|--------|--------------|
| **Thread Count** | Linear wait time increase | Use hierarchical barriers for large thread counts |
| **Barrier Frequency** | High overhead if frequent | Batch operations between barriers |
| **Load Imbalance** | Fastest thread waits longest | Balance work distribution |
| **Memory Access** | False sharing on barrier structure | Align barrier to cache line |

#### Optimized Barrier Implementation

```c
#include <stdatomic.h>

// Cache-aligned barrier to prevent false sharing
typedef struct {
    alignas(64) atomic_int count;
    alignas(64) atomic_int generation;
    int total_threads;
    alignas(64) char padding[64];
} optimized_barrier_t;

void optimized_barrier_init(optimized_barrier_t* barrier, int thread_count) {
    atomic_store(&barrier->count, thread_count);
    atomic_store(&barrier->generation, 0);
    barrier->total_threads = thread_count;
}

void optimized_barrier_wait(optimized_barrier_t* barrier) {
    int gen = atomic_load(&barrier->generation);
    
    if (atomic_fetch_sub(&barrier->count, 1) == 1) {
        // Last thread: reset and wake others
        atomic_store(&barrier->count, barrier->total_threads);
        atomic_fetch_add(&barrier->generation, 1);
    } else {
        // Wait for generation change
        while (atomic_load(&barrier->generation) == gen) {
            // Spin wait with pause instruction
            __builtin_ia32_pause();
        }
    }
}
```

### Real-World Barrier Applications

#### 1. Parallel Sorting Algorithms
```c
// Merge sort with barriers between merge phases
void parallel_merge_sort_with_barriers(int* array, int size, int num_threads);
```

#### 2. Matrix Operations
```c
// Matrix multiplication with row-wise thread distribution
void parallel_matrix_multiply_with_barriers(double** A, double** B, double** C, int n);
```

#### 3. Simulation Timesteps
```c
// Physics simulation with synchronized timesteps
void physics_simulation_timestep_with_barriers(particle_t* particles, int count);
```

### Common Barrier Pitfalls

#### Pitfall 1: Wrong Thread Count
```c
// WRONG: Barrier initialized for wrong number of threads
pthread_barrier_init(&barrier, NULL, 3); // But 4 threads try to use it
// One thread will wait forever

// CORRECT: Match barrier count to actual thread count
pthread_barrier_init(&barrier, NULL, NUM_THREADS);
```

#### Pitfall 2: Barrier Reuse Without Proper Cleanup
```c
// WRONG: Reusing barrier without considering waiting threads
pthread_barrier_destroy(&barrier); // While threads might still be waiting

// CORRECT: Ensure all threads have passed barrier before destruction
```

#### Pitfall 3: Deadlock with Multiple Barriers
```c
// WRONG: Different barrier ordering can cause deadlock
// Thread 1: barrier_wait(&barrier1); barrier_wait(&barrier2);
// Thread 2: barrier_wait(&barrier2); barrier_wait(&barrier1);

// CORRECT: Consistent barrier ordering across all threads
```
```

## Condition Variables

Condition variables are synchronization primitives that allow threads to wait for certain conditions to become true. They're always used in conjunction with mutexes and provide an efficient way to implement the "wait for condition" pattern without busy waiting.

### How Condition Variables Work

```
Thread 1 (Consumer):
1. Lock mutex
2. Check condition
3. If false: wait on condition variable (releases mutex atomically)
4. When signaled: reacquire mutex
5. Process data
6. Unlock mutex

Thread 2 (Producer):
1. Lock mutex
2. Modify shared data
3. Signal condition variable
4. Unlock mutex
```

### The Classic Producer-Consumer Problem

Condition variables solve the producer-consumer problem elegantly:

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>

#define BUFFER_SIZE 10
#define NUM_PRODUCERS 2
#define NUM_CONSUMERS 3
#define ITEMS_TO_PRODUCE 20

typedef struct {
    int buffer[BUFFER_SIZE];
    int count;          // Number of items in buffer
    int in;             // Index for next item to produce
    int out;            // Index for next item to consume
    
    pthread_mutex_t mutex;
    pthread_cond_t not_full;    // Signaled when buffer is not full
    pthread_cond_t not_empty;   // Signaled when buffer is not empty
    
    // Statistics
    int total_produced;
    int total_consumed;
    int producer_waits;
    int consumer_waits;
} bounded_buffer_t;

// Initialize the bounded buffer
int buffer_init(bounded_buffer_t* buf) {
    buf->count = 0;
    buf->in = 0;
    buf->out = 0;
    buf->total_produced = 0;
    buf->total_consumed = 0;
    buf->producer_waits = 0;
    buf->consumer_waits = 0;
    
    if (pthread_mutex_init(&buf->mutex, NULL) != 0) {
        perror("Failed to initialize mutex");
        return -1;
    }
    
    if (pthread_cond_init(&buf->not_full, NULL) != 0) {
        perror("Failed to initialize not_full condition");
        pthread_mutex_destroy(&buf->mutex);
        return -1;
    }
    
    if (pthread_cond_init(&buf->not_empty, NULL) != 0) {
        perror("Failed to initialize not_empty condition");
        pthread_cond_destroy(&buf->not_full);
        pthread_mutex_destroy(&buf->mutex);
        return -1;
    }
    
    return 0;
}

// Produce an item (add to buffer)
int buffer_produce(bounded_buffer_t* buf, int item, int producer_id) {
    if (pthread_mutex_lock(&buf->mutex) != 0) {
        perror("Producer: Failed to lock mutex");
        return -1;
    }
    
    // Wait while buffer is full
    while (buf->count == BUFFER_SIZE) {
        printf("Producer %d: Buffer full, waiting...\n", producer_id);
        buf->producer_waits++;
        
        if (pthread_cond_wait(&buf->not_full, &buf->mutex) != 0) {
            perror("Producer: Failed to wait on condition");
            pthread_mutex_unlock(&buf->mutex);
            return -1;
        }
    }
    
    // Add item to buffer
    buf->buffer[buf->in] = item;
    buf->in = (buf->in + 1) % BUFFER_SIZE;
    buf->count++;
    buf->total_produced++;
    
    printf("Producer %d: Produced item %d, buffer count: %d\n", 
           producer_id, item, buf->count);
    
    // Signal that buffer is not empty
    if (pthread_cond_signal(&buf->not_empty) != 0) {
        perror("Producer: Failed to signal not_empty");
    }
    
    if (pthread_mutex_unlock(&buf->mutex) != 0) {
        perror("Producer: Failed to unlock mutex");
        return -1;
    }
    
    return 0;
}

// Consume an item (remove from buffer)
int buffer_consume(bounded_buffer_t* buf, int* item, int consumer_id) {
    if (pthread_mutex_lock(&buf->mutex) != 0) {
        perror("Consumer: Failed to lock mutex");
        return -1;
    }
    
    // Wait while buffer is empty
    while (buf->count == 0) {
        printf("Consumer %d: Buffer empty, waiting...\n", consumer_id);
        buf->consumer_waits++;
        
        if (pthread_cond_wait(&buf->not_empty, &buf->mutex) != 0) {
            perror("Consumer: Failed to wait on condition");
            pthread_mutex_unlock(&buf->mutex);
            return -1;
        }
    }
    
    // Remove item from buffer
    *item = buf->buffer[buf->out];
    buf->out = (buf->out + 1) % BUFFER_SIZE;
    buf->count--;
    buf->total_consumed++;
    
    printf("Consumer %d: Consumed item %d, buffer count: %d\n", 
           consumer_id, *item, buf->count);
    
    // Signal that buffer is not full
    if (pthread_cond_signal(&buf->not_full) != 0) {
        perror("Consumer: Failed to signal not_full");
    }
    
    if (pthread_mutex_unlock(&buf->mutex) != 0) {
        perror("Consumer: Failed to unlock mutex");
        return -1;
    }
    
    return 0;
}

// Cleanup buffer
void buffer_destroy(bounded_buffer_t* buf) {
    pthread_cond_destroy(&buf->not_empty);
    pthread_cond_destroy(&buf->not_full);
    pthread_mutex_destroy(&buf->mutex);
}

// Producer thread function
void* producer_thread(void* arg) {
    bounded_buffer_t* buf = ((void**)arg)[0];
    int producer_id = *((int*)((void**)arg)[1]);
    
    for (int i = 0; i < ITEMS_TO_PRODUCE / NUM_PRODUCERS; i++) {
        int item = producer_id * 1000 + i; // Unique item number
        
        if (buffer_produce(buf, item, producer_id) != 0) {
            fprintf(stderr, "Producer %d failed to produce item\n", producer_id);
            break;
        }
        
        // Simulate production time
        usleep(rand() % 100000); // 0-100ms
    }
    
    printf("Producer %d finished\n", producer_id);
    return NULL;
}

// Consumer thread function
void* consumer_thread(void* arg) {
    bounded_buffer_t* buf = ((void**)arg)[0];
    int consumer_id = *((int*)((void**)arg)[1]);
    int items_consumed = 0;
    
    while (items_consumed < ITEMS_TO_PRODUCE / NUM_CONSUMERS) {
        int item;
        
        if (buffer_consume(buf, &item, consumer_id) == 0) {
            items_consumed++;
            
            // Simulate consumption time
            usleep(rand() % 150000); // 0-150ms
        } else {
            fprintf(stderr, "Consumer %d failed to consume item\n", consumer_id);
            break;
        }
    }
    
    printf("Consumer %d finished, consumed %d items\n", consumer_id, items_consumed);
    return NULL;
}
```

### Advanced Condition Variable Patterns

#### Pattern 1: Multiple Conditions with Single Mutex

```c
typedef struct {
    int readers_count;
    int writers_count;
    int writers_waiting;
    
    pthread_mutex_t mutex;
    pthread_cond_t readers_ok;  // Readers can proceed
    pthread_cond_t writers_ok;  // Writers can proceed
} read_write_control_t;

// Reader entry
void reader_enter(read_write_control_t* ctrl) {
    pthread_mutex_lock(&ctrl->mutex);
    
    // Wait while there are writers or writers waiting
    while (ctrl->writers_count > 0 || ctrl->writers_waiting > 0) {
        pthread_cond_wait(&ctrl->readers_ok, &ctrl->mutex);
    }
    
    ctrl->readers_count++;
    pthread_mutex_unlock(&ctrl->mutex);
}

// Reader exit
void reader_exit(read_write_control_t* ctrl) {
    pthread_mutex_lock(&ctrl->mutex);
    
    ctrl->readers_count--;
    
    // If last reader, signal waiting writers
    if (ctrl->readers_count == 0 && ctrl->writers_waiting > 0) {
        pthread_cond_signal(&ctrl->writers_ok);
    }
    
    pthread_mutex_unlock(&ctrl->mutex);
}

// Writer entry
void writer_enter(read_write_control_t* ctrl) {
    pthread_mutex_lock(&ctrl->mutex);
    
    ctrl->writers_waiting++;
    
    // Wait while there are readers or other writers
    while (ctrl->readers_count > 0 || ctrl->writers_count > 0) {
        pthread_cond_wait(&ctrl->writers_ok, &ctrl->mutex);
    }
    
    ctrl->writers_waiting--;
    ctrl->writers_count++;
    
    pthread_mutex_unlock(&ctrl->mutex);
}

// Writer exit
void writer_exit(read_write_control_t* ctrl) {
    pthread_mutex_lock(&ctrl->mutex);
    
    ctrl->writers_count--;
    
    // Signal waiting writers first (writer preference)
    if (ctrl->writers_waiting > 0) {
        pthread_cond_signal(&ctrl->writers_ok);
    } else {
        // No waiting writers, signal all readers
        pthread_cond_broadcast(&ctrl->readers_ok);
    }
    
    pthread_mutex_unlock(&ctrl->mutex);
}
```

#### Pattern 2: Timed Condition Waits

```c
#include <time.h>

// Wait with timeout
int timed_condition_wait(pthread_cond_t* cond, pthread_mutex_t* mutex, int timeout_sec) {
    struct timespec timeout;
    int result;
    
    // Get current time
    if (clock_gettime(CLOCK_REALTIME, &timeout) != 0) {
        perror("Failed to get current time");
        return -1;
    }
    
    // Add timeout
    timeout.tv_sec += timeout_sec;
    
    // Wait with timeout
    result = pthread_cond_timedwait(cond, mutex, &timeout);
    
    switch (result) {
        case 0:
            printf("Condition signaled within timeout\n");
            return 0;
            
        case ETIMEDOUT:
            printf("Condition wait timed out after %d seconds\n", timeout_sec);
            return 1;
            
        default:
            fprintf(stderr, "Timed wait failed: %s\n", strerror(result));
            return -1;
    }
}

// Example usage in bounded buffer with timeout
int buffer_consume_timeout(bounded_buffer_t* buf, int* item, int consumer_id, int timeout_sec) {
    if (pthread_mutex_lock(&buf->mutex) != 0) {
        return -1;
    }
    
    // Wait while buffer is empty, but with timeout
    while (buf->count == 0) {
        int wait_result = timed_condition_wait(&buf->not_empty, &buf->mutex, timeout_sec);
        
        if (wait_result != 0) {
            pthread_mutex_unlock(&buf->mutex);
            return wait_result; // Timeout or error
        }
    }
    
    // Consume item (same as before)
    *item = buf->buffer[buf->out];
    buf->out = (buf->out + 1) % BUFFER_SIZE;
    buf->count--;
    
    pthread_cond_signal(&buf->not_full);
    pthread_mutex_unlock(&buf->mutex);
    
    return 0;
}
```

### Spurious Wakeups and Proper Condition Checking

#### Why Always Use While Loops

```c
// WRONG: Using if statement
if (buffer_empty) {
    pthread_cond_wait(&not_empty, &mutex);
}
// Problem: Spurious wakeup might occur when buffer is still empty

// CORRECT: Using while loop
while (buffer_empty) {
    pthread_cond_wait(&not_empty, &mutex);
}
// Always recheck condition after wakeup
```

#### Comprehensive Example with Error Handling

```c
typedef struct {
    int value;
    int ready;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
} thread_safe_value_t;

int safe_value_init(thread_safe_value_t* tsv) {
    tsv->value = 0;
    tsv->ready = 0;
    
    int result = pthread_mutex_init(&tsv->mutex, NULL);
    if (result != 0) {
        fprintf(stderr, "Mutex init failed: %s\n", strerror(result));
        return -1;
    }
    
    result = pthread_cond_init(&tsv->condition, NULL);
    if (result != 0) {
        fprintf(stderr, "Condition init failed: %s\n", strerror(result));
        pthread_mutex_destroy(&tsv->mutex);
        return -1;
    }
    
    return 0;
}

// Set value and notify waiters
int safe_value_set(thread_safe_value_t* tsv, int new_value) {
    int result = pthread_mutex_lock(&tsv->mutex);
    if (result != 0) {
        fprintf(stderr, "Lock failed in set: %s\n", strerror(result));
        return -1;
    }
    
    tsv->value = new_value;
    tsv->ready = 1;
    
    printf("Value set to %d, notifying waiters\n", new_value);
    
    // Notify all waiting threads
    result = pthread_cond_broadcast(&tsv->condition);
    if (result != 0) {
        fprintf(stderr, "Broadcast failed: %s\n", strerror(result));
    }
    
    result = pthread_mutex_unlock(&tsv->mutex);
    if (result != 0) {
        fprintf(stderr, "Unlock failed in set: %s\n", strerror(result));
        return -1;
    }
    
    return 0;
}

// Wait for value to be ready and retrieve it
int safe_value_get(thread_safe_value_t* tsv, int* value) {
    int result = pthread_mutex_lock(&tsv->mutex);
    if (result != 0) {
        fprintf(stderr, "Lock failed in get: %s\n", strerror(result));
        return -1;
    }
    
    // Wait until value is ready
    while (!tsv->ready) {
        printf("Waiting for value to be set...\n");
        
        result = pthread_cond_wait(&tsv->condition, &tsv->mutex);
        if (result != 0) {
            fprintf(stderr, "Condition wait failed: %s\n", strerror(result));
            pthread_mutex_unlock(&tsv->mutex);
            return -1;
        }
        
        // Always recheck condition after wait
        printf("Woke up, rechecking condition...\n");
    }
    
    *value = tsv->value;
    printf("Retrieved value: %d\n", *value);
    
    result = pthread_mutex_unlock(&tsv->mutex);
    if (result != 0) {
        fprintf(stderr, "Unlock failed in get: %s\n", strerror(result));
        return -1;
    }
    
    return 0;
}

void safe_value_destroy(thread_safe_value_t* tsv) {
    pthread_cond_destroy(&tsv->condition);
    pthread_mutex_destroy(&tsv->mutex);
}
```

## Semaphores

Semaphores are counting synchronization primitives that maintain a count of available resources. They're more general than mutexes (which are binary semaphores) and are particularly useful for resource management and flow control.

### Semaphore Types and Operations

#### POSIX Semaphores

```c
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>

// Named semaphore (inter-process)
sem_t* named_sem = sem_open("/my_semaphore", O_CREAT | O_EXCL, 0644, 1);

// Unnamed semaphore (intra-process)
sem_t unnamed_sem;
sem_init(&unnamed_sem, 0, 1); // 0 = thread-shared, 1 = initial value

// Semaphore operations
sem_wait(&sem);    // P operation (decrement, block if 0)
sem_post(&sem);    // V operation (increment, wake waiters)
sem_trywait(&sem); // Non-blocking wait
sem_timedwait(&sem, &timeout); // Wait with timeout

// Cleanup
sem_destroy(&unnamed_sem);
sem_close(named_sem);
sem_unlink("/my_semaphore");
```

### Resource Pool Management with Semaphores

```c
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define POOL_SIZE 3
#define NUM_CLIENTS 8
#define MAX_USE_TIME 2

typedef struct {
    int resource_id;
    int in_use;
    pthread_mutex_t mutex;
} resource_t;

typedef struct {
    resource_t resources[POOL_SIZE];
    sem_t available_count;      // Count of available resources
    pthread_mutex_t pool_mutex; // Protects resource allocation
    
    // Statistics
    int total_acquisitions;
    int total_releases;
    int max_wait_time;
} resource_pool_t;

// Initialize resource pool
int resource_pool_init(resource_pool_t* pool) {
    // Initialize semaphore with pool size
    if (sem_init(&pool->available_count, 0, POOL_SIZE) != 0) {
        perror("Failed to initialize semaphore");
        return -1;
    }
    
    if (pthread_mutex_init(&pool->pool_mutex, NULL) != 0) {
        perror("Failed to initialize pool mutex");
        sem_destroy(&pool->available_count);
        return -1;
    }
    
    // Initialize resources
    for (int i = 0; i < POOL_SIZE; i++) {
        pool->resources[i].resource_id = i;
        pool->resources[i].in_use = 0;
        
        if (pthread_mutex_init(&pool->resources[i].mutex, NULL) != 0) {
            perror("Failed to initialize resource mutex");
            // Cleanup previously initialized resources
            for (int j = 0; j < i; j++) {
                pthread_mutex_destroy(&pool->resources[j].mutex);
            }
            pthread_mutex_destroy(&pool->pool_mutex);
            sem_destroy(&pool->available_count);
            return -1;
        }
    }
    
    pool->total_acquisitions = 0;
    pool->total_releases = 0;
    pool->max_wait_time = 0;
    
    printf("Resource pool initialized with %d resources\n", POOL_SIZE);
    return 0;
}

// Acquire a resource from the pool
resource_t* resource_pool_acquire(resource_pool_t* pool, int client_id) {
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    printf("Client %d: Requesting resource...\n", client_id);
    
    // Wait for available resource (decrements semaphore)
    if (sem_wait(&pool->available_count) != 0) {
        perror("Failed to wait on semaphore");
        return NULL;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    int wait_time = (end_time.tv_sec - start_time.tv_sec) * 1000 + 
                    (end_time.tv_nsec - start_time.tv_nsec) / 1000000;
    
    printf("Client %d: Got semaphore after %d ms, looking for resource...\n", 
           client_id, wait_time);
    
    // Find and allocate an available resource
    pthread_mutex_lock(&pool->pool_mutex);
    
    resource_t* allocated_resource = NULL;
    for (int i = 0; i < POOL_SIZE; i++) {
        if (!pool->resources[i].in_use) {
            pool->resources[i].in_use = 1;
            allocated_resource = &pool->resources[i];
            pool->total_acquisitions++;
            
            if (wait_time > pool->max_wait_time) {
                pool->max_wait_time = wait_time;
            }
            
            printf("Client %d: Allocated resource %d (total acquisitions: %d)\n", 
                   client_id, i, pool->total_acquisitions);
            break;
        }
    }
    
    pthread_mutex_unlock(&pool->pool_mutex);
    
    if (allocated_resource == NULL) {
        // This shouldn't happen if semaphore works correctly
        fprintf(stderr, "ERROR: Semaphore allowed access but no resource available!\n");
        sem_post(&pool->available_count); // Release semaphore
    }
    
    return allocated_resource;
}

// Release a resource back to the pool
int resource_pool_release(resource_pool_t* pool, resource_t* resource, int client_id) {
    if (resource == NULL) {
        return -1;
    }
    
    printf("Client %d: Releasing resource %d\n", client_id, resource->resource_id);
    
    // Mark resource as available
    pthread_mutex_lock(&pool->pool_mutex);
    resource->in_use = 0;
    pool->total_releases++;
    printf("Client %d: Resource %d released (total releases: %d)\n", 
           client_id, resource->resource_id, pool->total_releases);
    pthread_mutex_unlock(&pool->pool_mutex);
    
    // Signal that a resource is now available (increments semaphore)
    if (sem_post(&pool->available_count) != 0) {
        perror("Failed to post semaphore");
        return -1;
    }
    
    return 0;
}

// Get current semaphore value
int resource_pool_get_available_count(resource_pool_t* pool) {
    int value;
    if (sem_getvalue(&pool->available_count, &value) == 0) {
        return value;
    }
    return -1;
}

// Cleanup resource pool
void resource_pool_destroy(resource_pool_t* pool) {
    printf("Pool statistics: Acquisitions=%d, Releases=%d, Max wait time=%d ms\n",
           pool->total_acquisitions, pool->total_releases, pool->max_wait_time);
    
    for (int i = 0; i < POOL_SIZE; i++) {
        pthread_mutex_destroy(&pool->resources[i].mutex);
    }
    
    pthread_mutex_destroy(&pool->pool_mutex);
    sem_destroy(&pool->available_count);
}

// Client thread function
void* client_thread(void* arg) {
    resource_pool_t* pool = ((void**)arg)[0];
    int client_id = *((int*)((void**)arg)[1]);
    
    // Simulate multiple resource uses
    for (int use = 0; use < 2; use++) {
        printf("Client %d: Use #%d - attempting to acquire resource\n", client_id, use + 1);
        
        // Acquire resource
        resource_t* resource = resource_pool_acquire(pool, client_id);
        
        if (resource != NULL) {
            // Use the resource
            printf("Client %d: Using resource %d for %d seconds\n", 
                   client_id, resource->resource_id, MAX_USE_TIME);
            
            sleep(rand() % MAX_USE_TIME + 1);
            
            // Release resource
            resource_pool_release(pool, resource, client_id);
        } else {
            fprintf(stderr, "Client %d: Failed to acquire resource\n", client_id);
        }
        
        // Rest between uses
        sleep(rand() % 2);
    }
    
    printf("Client %d: Finished all resource uses\n", client_id);
    return NULL;
}
```

### Advanced Semaphore Patterns

#### Pattern 1: Producer-Consumer with Counting Semaphores

```c
#define BUFFER_SIZE 5

typedef struct {
    int buffer[BUFFER_SIZE];
    int in, out;
    
    sem_t empty_slots;  // Counts empty buffer slots
    sem_t full_slots;   // Counts full buffer slots
    sem_t mutex;        // Binary semaphore for mutual exclusion
    
    int items_produced;
    int items_consumed;
} semaphore_buffer_t;

int semaphore_buffer_init(semaphore_buffer_t* buf) {
    buf->in = buf->out = 0;
    buf->items_produced = buf->items_consumed = 0;
    
    // Initialize counting semaphores
    if (sem_init(&buf->empty_slots, 0, BUFFER_SIZE) != 0 ||
        sem_init(&buf->full_slots, 0, 0) != 0 ||
        sem_init(&buf->mutex, 0, 1) != 0) {
        return -1;
    }
    
    return 0;
}

void semaphore_buffer_produce(semaphore_buffer_t* buf, int item, int producer_id) {
    // Wait for empty slot
    sem_wait(&buf->empty_slots);
    
    // Acquire mutex for buffer access
    sem_wait(&buf->mutex);
    
    // Add item to buffer
    buf->buffer[buf->in] = item;
    buf->in = (buf->in + 1) % BUFFER_SIZE;
    buf->items_produced++;
    
    printf("Producer %d: Produced item %d (total: %d)\n", 
           producer_id, item, buf->items_produced);
    
    // Release mutex
    sem_post(&buf->mutex);
    
    // Signal that there's a full slot
    sem_post(&buf->full_slots);
}

int semaphore_buffer_consume(semaphore_buffer_t* buf, int consumer_id) {
    // Wait for full slot
    sem_wait(&buf->full_slots);
    
    // Acquire mutex for buffer access
    sem_wait(&buf->mutex);
    
    // Remove item from buffer
    int item = buf->buffer[buf->out];
    buf->out = (buf->out + 1) % BUFFER_SIZE;
    buf->items_consumed++;
    
    printf("Consumer %d: Consumed item %d (total: %d)\n", 
           consumer_id, item, buf->items_consumed);
    
    // Release mutex
    sem_post(&buf->mutex);
    
    // Signal that there's an empty slot
    sem_post(&buf->empty_slots);
    
    return item;
}
```

#### Pattern 2: Dining Philosophers with Semaphores

```c
#define NUM_PHILOSOPHERS 5

typedef struct {
    sem_t forks[NUM_PHILOSOPHERS];  // One semaphore per fork
    sem_t dining_room;              // Limits concurrent diners
    int meal_counts[NUM_PHILOSOPHERS];
    pthread_mutex_t stats_mutex;
} dining_table_t;

int dining_table_init(dining_table_t* table) {
    // Initialize fork semaphores (binary)
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        if (sem_init(&table->forks[i], 0, 1) != 0) {
            return -1;
        }
        table->meal_counts[i] = 0;
    }
    
    // Limit concurrent diners to prevent deadlock
    if (sem_init(&table->dining_room, 0, NUM_PHILOSOPHERS - 1) != 0) {
        return -1;
    }
    
    pthread_mutex_init(&table->stats_mutex, NULL);
    return 0;
}

void philosopher_dine(dining_table_t* table, int philosopher_id) {
    int left_fork = philosopher_id;
    int right_fork = (philosopher_id + 1) % NUM_PHILOSOPHERS;
    
    // Enter dining room (prevents deadlock)
    sem_wait(&table->dining_room);
    
    printf("Philosopher %d: Entered dining room\n", philosopher_id);
    
    // Pick up forks (always in same order to prevent deadlock)
    int first_fork = (left_fork < right_fork) ? left_fork : right_fork;
    int second_fork = (left_fork < right_fork) ? right_fork : left_fork;
    
    printf("Philosopher %d: Trying to pick up fork %d\n", philosopher_id, first_fork);
    sem_wait(&table->forks[first_fork]);
    
    printf("Philosopher %d: Picked up fork %d, trying fork %d\n", 
           philosopher_id, first_fork, second_fork);
    sem_wait(&table->forks[second_fork]);
    
    // Eat
    printf("Philosopher %d: Eating with forks %d and %d\n", 
           philosopher_id, left_fork, right_fork);
    sleep(1 + rand() % 2);
    
    // Update meal count
    pthread_mutex_lock(&table->stats_mutex);
    table->meal_counts[philosopher_id]++;
    pthread_mutex_unlock(&table->stats_mutex);
    
    // Put down forks
    sem_post(&table->forks[first_fork]);
    sem_post(&table->forks[second_fork]);
    
    printf("Philosopher %d: Finished eating (meals: %d)\n", 
           philosopher_id, table->meal_counts[philosopher_id]);
    
    // Leave dining room
    sem_post(&table->dining_room);
}
```

### Semaphore Performance and Best Practices

#### Performance Comparison

| Synchronization | Use Case | Performance | Complexity |
|-----------------|----------|-------------|------------|
| **Binary Semaphore** | Simple mutual exclusion | Good | Low |
| **Counting Semaphore** | Resource counting | Good | Medium |
| **Mutex** | Mutual exclusion with ownership | Better | Low |
| **Condition Variable** | Wait for condition | Best for waiting | Medium |

#### Best Practices

1. **Always pair wait and post operations**
2. **Initialize with correct initial count**
3. **Use consistent ordering for multiple semaphores**
4. **Consider using mutexes for simple mutual exclusion**
5. **Handle errors properly**

```c
// Good semaphore usage pattern
int safe_semaphore_operation(sem_t* sem, int operation) {
    int result;
    
    if (operation == 0) { // wait
        do {
            result = sem_wait(sem);
        } while (result == -1 && errno == EINTR); // Retry on interrupt
    } else { // post
        result = sem_post(sem);
    }
    
    if (result != 0) {
        perror("Semaphore operation failed");
        return -1;
    }
    
    return 0;
}
```

## Advanced Synchronization Patterns

### Lock-Free Programming Techniques

Lock-free programming uses atomic operations to achieve synchronization without traditional locks, providing better performance and avoiding deadlocks.

#### Compare-and-Swap (CAS) Operations

```c
#include <stdatomic.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

// Lock-free counter using atomic operations
typedef struct {
    atomic_int value;
    atomic_long operations;
} lockfree_counter_t;

void lockfree_counter_init(lockfree_counter_t* counter) {
    atomic_store(&counter->value, 0);
    atomic_store(&counter->operations, 0);
}

// Lock-free increment
int lockfree_counter_increment(lockfree_counter_t* counter) {
    int old_value, new_value;
    
    do {
        old_value = atomic_load(&counter->value);
        new_value = old_value + 1;
    } while (!atomic_compare_exchange_weak(&counter->value, &old_value, new_value));
    
    atomic_fetch_add(&counter->operations, 1);
    return new_value;
}

// Lock-free decrement with minimum check
int lockfree_counter_decrement(lockfree_counter_t* counter, int minimum) {
    int old_value, new_value;
    
    do {
        old_value = atomic_load(&counter->value);
        if (old_value <= minimum) {
            return old_value; // Cannot decrement below minimum
        }
        new_value = old_value - 1;
    } while (!atomic_compare_exchange_weak(&counter->value, &old_value, new_value));
    
    atomic_fetch_add(&counter->operations, 1);
    return new_value;
}

// Lock-free stack implementation
typedef struct stack_node {
    int data;
    struct stack_node* next;
} stack_node_t;

typedef struct {
    atomic_intptr_t top;
    atomic_long push_count;
    atomic_long pop_count;
} lockfree_stack_t;

void lockfree_stack_init(lockfree_stack_t* stack) {
    atomic_store(&stack->top, (intptr_t)NULL);
    atomic_store(&stack->push_count, 0);
    atomic_store(&stack->pop_count, 0);
}

void lockfree_stack_push(lockfree_stack_t* stack, int data) {
    stack_node_t* new_node = malloc(sizeof(stack_node_t));
    new_node->data = data;
    
    stack_node_t* old_top;
    do {
        old_top = (stack_node_t*)atomic_load(&stack->top);
        new_node->next = old_top;
    } while (!atomic_compare_exchange_weak(&stack->top, (intptr_t*)&old_top, (intptr_t)new_node));
    
    atomic_fetch_add(&stack->push_count, 1);
    printf("Pushed %d onto stack\n", data);
}

int lockfree_stack_pop(lockfree_stack_t* stack, int* data) {
    stack_node_t* old_top;
    stack_node_t* new_top;
    
    do {
        old_top = (stack_node_t*)atomic_load(&stack->top);
        if (old_top == NULL) {
            return 0; // Stack is empty
        }
        new_top = old_top->next;
    } while (!atomic_compare_exchange_weak(&stack->top, (intptr_t*)&old_top, (intptr_t)new_top));
    
    *data = old_top->data;
    free(old_top);
    
    atomic_fetch_add(&stack->pop_count, 1);
    printf("Popped %d from stack\n", *data);
    return 1;
}
```

### The Monitor Pattern

Monitors encapsulate both data and the synchronization mechanisms to access it, providing a high-level abstraction for thread-safe operations.

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Monitor for thread-safe bounded queue
typedef struct {
    int* buffer;
    int capacity;
    int size;
    int front;
    int rear;
    
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
    
    // Monitor statistics
    int total_enqueued;
    int total_dequeued;
    int max_size_reached;
    double avg_wait_time;
    int wait_count;
} bounded_queue_monitor_t;

// Initialize monitor
int monitor_init(bounded_queue_monitor_t* monitor, int capacity) {
    monitor->buffer = malloc(capacity * sizeof(int));
    if (!monitor->buffer) {
        return -1;
    }
    
    monitor->capacity = capacity;
    monitor->size = 0;
    monitor->front = 0;
    monitor->rear = 0;
    monitor->total_enqueued = 0;
    monitor->total_dequeued = 0;
    monitor->max_size_reached = 0;
    monitor->avg_wait_time = 0.0;
    monitor->wait_count = 0;
    
    if (pthread_mutex_init(&monitor->mutex, NULL) != 0) {
        free(monitor->buffer);
        return -1;
    }
    
    if (pthread_cond_init(&monitor->not_full, NULL) != 0) {
        pthread_mutex_destroy(&monitor->mutex);
        free(monitor->buffer);
        return -1;
    }
    
    if (pthread_cond_init(&monitor->not_empty, NULL) != 0) {
        pthread_cond_destroy(&monitor->not_full);
        pthread_mutex_destroy(&monitor->mutex);
        free(monitor->buffer);
        return -1;
    }
    
    printf("Monitor initialized with capacity %d\n", capacity);
    return 0;
}

// Monitor method: enqueue with timeout
int monitor_enqueue_timeout(bounded_queue_monitor_t* monitor, int item, int timeout_ms) {
    struct timespec start_time, end_time, timeout_spec;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    if (pthread_mutex_lock(&monitor->mutex) != 0) {
        return -1;
    }
    
    // Calculate timeout
    clock_gettime(CLOCK_REALTIME, &timeout_spec);
    timeout_spec.tv_sec += timeout_ms / 1000;
    timeout_spec.tv_nsec += (timeout_ms % 1000) * 1000000;
    if (timeout_spec.tv_nsec >= 1000000000) {
        timeout_spec.tv_sec++;
        timeout_spec.tv_nsec -= 1000000000;
    }
    
    // Wait while queue is full
    while (monitor->size == monitor->capacity) {
        printf("Queue full, waiting to enqueue %d...\n", item);
        
        int wait_result = pthread_cond_timedwait(&monitor->not_full, &monitor->mutex, &timeout_spec);
        
        if (wait_result == ETIMEDOUT) {
            pthread_mutex_unlock(&monitor->mutex);
            printf("Enqueue timeout after %d ms\n", timeout_ms);
            return 0; // Timeout
        } else if (wait_result != 0) {
            pthread_mutex_unlock(&monitor->mutex);
            return -1; // Error
        }
    }
    
    // Enqueue item
    monitor->buffer[monitor->rear] = item;
    monitor->rear = (monitor->rear + 1) % monitor->capacity;
    monitor->size++;
    monitor->total_enqueued++;
    
    if (monitor->size > monitor->max_size_reached) {
        monitor->max_size_reached = monitor->size;
    }
    
    printf("Enqueued %d, queue size: %d\n", item, monitor->size);
    
    // Signal waiting consumers
    pthread_cond_signal(&monitor->not_empty);
    
    // Calculate wait time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double wait_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    monitor->avg_wait_time = (monitor->avg_wait_time * monitor->wait_count + wait_time) / 
                             (monitor->wait_count + 1);
    monitor->wait_count++;
    
    pthread_mutex_unlock(&monitor->mutex);
    return 1; // Success
}

// Monitor method: dequeue with priority
int monitor_dequeue_priority(bounded_queue_monitor_t* monitor, int* item, int high_priority) {
    if (pthread_mutex_lock(&monitor->mutex) != 0) {
        return -1;
    }
    
    // High priority threads get preference
    if (high_priority) {
        while (monitor->size == 0) {
            printf("High priority thread waiting to dequeue...\n");
            if (pthread_cond_wait(&monitor->not_empty, &monitor->mutex) != 0) {
                pthread_mutex_unlock(&monitor->mutex);
                return -1;
            }
        }
    } else {
        // Low priority threads check once and return if empty
        if (monitor->size == 0) {
            pthread_mutex_unlock(&monitor->mutex);
            return 0; // Queue empty
        }
    }
    
    // Dequeue item
    *item = monitor->buffer[monitor->front];
    monitor->front = (monitor->front + 1) % monitor->capacity;
    monitor->size--;
    monitor->total_dequeued++;
    
    printf("Dequeued %d (%s priority), queue size: %d\n", 
           *item, high_priority ? "high" : "low", monitor->size);
    
    // Signal waiting producers
    pthread_cond_signal(&monitor->not_full);
    
    pthread_mutex_unlock(&monitor->mutex);
    return 1; // Success
}

// Monitor method: get statistics
void monitor_get_stats(bounded_queue_monitor_t* monitor) {
    if (pthread_mutex_lock(&monitor->mutex) != 0) {
        return;
    }
    
    printf("\n=== Queue Monitor Statistics ===\n");
    printf("Capacity: %d\n", monitor->capacity);
    printf("Current size: %d\n", monitor->size);
    printf("Total enqueued: %d\n", monitor->total_enqueued);
    printf("Total dequeued: %d\n", monitor->total_dequeued);
    printf("Max size reached: %d\n", monitor->max_size_reached);
    printf("Average wait time: %.2f ms\n", monitor->avg_wait_time);
    printf("Total waits: %d\n", monitor->wait_count);
    printf("Utilization: %.1f%%\n", 
           (double)monitor->max_size_reached / monitor->capacity * 100);
    
    pthread_mutex_unlock(&monitor->mutex);
}

// Cleanup monitor
void monitor_destroy(bounded_queue_monitor_t* monitor) {
    pthread_cond_destroy(&monitor->not_empty);
    pthread_cond_destroy(&monitor->not_full);
    pthread_mutex_destroy(&monitor->mutex);
    free(monitor->buffer);
}
```

### Message Passing Patterns

#### Thread-Safe Message Queue

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_MESSAGE_SIZE 256

typedef struct message {
    char data[MAX_MESSAGE_SIZE];
    int priority;
    int sender_id;
    struct timespec timestamp;
    struct message* next;
} message_t;

typedef struct {
    message_t* head;
    message_t* tail;
    int count;
    int max_count;
    
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    
    // Statistics
    int messages_sent;
    int messages_received;
    int high_priority_count;
    int low_priority_count;
} message_queue_t;

int message_queue_init(message_queue_t* queue, int max_count) {
    queue->head = NULL;
    queue->tail = NULL;
    queue->count = 0;
    queue->max_count = max_count;
    queue->messages_sent = 0;
    queue->messages_received = 0;
    queue->high_priority_count = 0;
    queue->low_priority_count = 0;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        return -1;
    }
    
    if (pthread_cond_init(&queue->not_empty, NULL) != 0) {
        pthread_mutex_destroy(&queue->mutex);
        return -1;
    }
    
    if (pthread_cond_init(&queue->not_full, NULL) != 0) {
        pthread_cond_destroy(&queue->not_empty);
        pthread_mutex_destroy(&queue->mutex);
        return -1;
    }
    
    return 0;
}

// Send message with priority
int message_queue_send(message_queue_t* queue, const char* data, int priority, int sender_id) {
    if (pthread_mutex_lock(&queue->mutex) != 0) {
        return -1;
    }
    
    // Wait while queue is full
    while (queue->count >= queue->max_count) {
        printf("Message queue full, sender %d waiting...\n", sender_id);
        if (pthread_cond_wait(&queue->not_full, &queue->mutex) != 0) {
            pthread_mutex_unlock(&queue->mutex);
            return -1;
        }
    }
    
    // Create new message
    message_t* new_message = malloc(sizeof(message_t));
    if (!new_message) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    
    strncpy(new_message->data, data, MAX_MESSAGE_SIZE - 1);
    new_message->data[MAX_MESSAGE_SIZE - 1] = '\0';
    new_message->priority = priority;
    new_message->sender_id = sender_id;
    clock_gettime(CLOCK_REALTIME, &new_message->timestamp);
    new_message->next = NULL;
    
    // Insert based on priority (higher priority = lower number)
    if (queue->head == NULL || priority < queue->head->priority) {
        // Insert at head
        new_message->next = queue->head;
        queue->head = new_message;
        if (queue->tail == NULL) {
            queue->tail = new_message;
        }
    } else {
        // Find insertion point
        message_t* current = queue->head;
        while (current->next != NULL && current->next->priority <= priority) {
            current = current->next;
        }
        new_message->next = current->next;
        current->next = new_message;
        if (new_message->next == NULL) {
            queue->tail = new_message;
        }
    }
    
    queue->count++;
    queue->messages_sent++;
    
    if (priority == 0) {
        queue->high_priority_count++;
    } else {
        queue->low_priority_count++;
    }
    
    printf("Sent message (priority %d) from sender %d: \"%.50s%s\"\n", 
           priority, sender_id, data, strlen(data) > 50 ? "..." : "");
    
    // Signal waiting receivers
    pthread_cond_signal(&queue->not_empty);
    
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

// Receive message
int message_queue_receive(message_queue_t* queue, char* data, int* priority, int* sender_id, int receiver_id) {
    if (pthread_mutex_lock(&queue->mutex) != 0) {
        return -1;
    }
    
    // Wait while queue is empty
    while (queue->count == 0) {
        printf("Message queue empty, receiver %d waiting...\n", receiver_id);
        if (pthread_cond_wait(&queue->not_empty, &queue->mutex) != 0) {
            pthread_mutex_unlock(&queue->mutex);
            return -1;
        }
    }
    
    // Remove message from head (highest priority)
    message_t* message = queue->head;
    queue->head = message->next;
    if (queue->head == NULL) {
        queue->tail = NULL;
    }
    
    // Copy message data
    strncpy(data, message->data, MAX_MESSAGE_SIZE);
    *priority = message->priority;
    *sender_id = message->sender_id;
    
    queue->count--;
    queue->messages_received++;
    
    printf("Received message (priority %d) by receiver %d from sender %d: \"%.50s%s\"\n", 
           *priority, receiver_id, *sender_id, data, strlen(data) > 50 ? "..." : "");
    
    free(message);
    
    // Signal waiting senders
    pthread_cond_signal(&queue->not_full);
    
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

// Get queue statistics
void message_queue_stats(message_queue_t* queue) {
    if (pthread_mutex_lock(&queue->mutex) != 0) {
        return;
    }
    
    printf("\n=== Message Queue Statistics ===\n");
    printf("Current messages: %d / %d\n", queue->count, queue->max_count);
    printf("Total sent: %d\n", queue->messages_sent);
    printf("Total received: %d\n", queue->messages_received);
    printf("High priority messages: %d\n", queue->high_priority_count);
    printf("Low priority messages: %d\n", queue->low_priority_count);
    printf("Messages in transit: %d\n", queue->messages_sent - queue->messages_received);
    
    pthread_mutex_unlock(&queue->mutex);
}

void message_queue_destroy(message_queue_t* queue) {
    pthread_mutex_lock(&queue->mutex);
    
    // Free remaining messages
    message_t* current = queue->head;
    while (current != NULL) {
        message_t* next = current->next;
        free(current);
        current = next;
    }
    
    pthread_mutex_unlock(&queue->mutex);
    
    pthread_cond_destroy(&queue->not_full);
    pthread_cond_destroy(&queue->not_empty);
    pthread_mutex_destroy(&queue->mutex);
}
```

## Error Handling and Debugging

### Comprehensive Error Handling Framework

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <stdarg.h>

// Error logging levels
typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARNING = 2,
    LOG_ERROR = 3,
    LOG_FATAL = 4
} log_level_t;

// Thread-safe error logging
typedef struct {
    FILE* log_file;
    log_level_t min_level;
    pthread_mutex_t log_mutex;
    int log_count[5]; // Count per log level
} error_logger_t;

static error_logger_t global_logger = {NULL, LOG_INFO, PTHREAD_MUTEX_INITIALIZER, {0}};

// Initialize error logger
int error_logger_init(const char* filename, log_level_t min_level) {
    if (filename) {
        global_logger.log_file = fopen(filename, "a");
        if (!global_logger.log_file) {
            perror("Failed to open log file");
            return -1;
        }
    } else {
        global_logger.log_file = stderr;
    }
    
    global_logger.min_level = min_level;
    
    // Log initialization
    log_message(LOG_INFO, "Error logger initialized at level %d", min_level);
    return 0;
}

// Thread-safe logging function
void log_message(log_level_t level, const char* format, ...) {
    if (level < global_logger.min_level) {
        return;
    }
    
    pthread_mutex_lock(&global_logger.log_mutex);
    
    // Get timestamp
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    struct tm* tm_info = localtime(&now.tv_sec);
    
    // Get thread ID
    pthread_t thread_id = pthread_self();
    
    // Log level strings
    const char* level_str[] = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
    
    // Print timestamp, thread ID, and level
    if (global_logger.log_file) {
        fprintf(global_logger.log_file, "[%04d-%02d-%02d %02d:%02d:%02d.%03ld] [%lu] [%s] ",
                tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday,
                tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec,
                now.tv_nsec / 1000000, (unsigned long)thread_id, level_str[level]);
        
        // Print message
        va_list args;
        va_start(args, format);
        vfprintf(global_logger.log_file, format, args);
        va_end(args);
        
        fprintf(global_logger.log_file, "\n");
        fflush(global_logger.log_file);
    }
    
    global_logger.log_count[level]++;
    
    pthread_mutex_unlock(&global_logger.log_mutex);
}

// Enhanced error checking functions
int check_pthread_error(int result, const char* function_name, const char* context) {
    if (result != 0) {
        log_message(LOG_ERROR, "pthread error in %s (%s): %s (code: %d)", 
                   function_name, context, strerror(result), result);
        return -1;
    }
    return 0;
}

int safe_mutex_lock(pthread_mutex_t* mutex, const char* context) {
    int result = pthread_mutex_lock(mutex);
    if (result != 0) {
        log_message(LOG_ERROR, "Mutex lock failed in %s: %s", context, strerror(result));
        return -1;
    }
    log_message(LOG_DEBUG, "Mutex locked in %s", context);
    return 0;
}

int safe_mutex_unlock(pthread_mutex_t* mutex, const char* context) {
    int result = pthread_mutex_unlock(mutex);
    if (result != 0) {
        log_message(LOG_ERROR, "Mutex unlock failed in %s: %s", context, strerror(result));
        return -1;
    }
    log_message(LOG_DEBUG, "Mutex unlocked in %s", context);
    return 0;
}

int safe_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex, const char* context) {
    log_message(LOG_DEBUG, "Waiting on condition in %s", context);
    int result = pthread_cond_wait(cond, mutex);
    if (result != 0) {
        log_message(LOG_ERROR, "Condition wait failed in %s: %s", context, strerror(result));
        return -1;
    }
    log_message(LOG_DEBUG, "Condition signaled in %s", context);
    return 0;
}

int safe_cond_signal(pthread_cond_t* cond, const char* context) {
    int result = pthread_cond_signal(cond);
    if (result != 0) {
        log_message(LOG_ERROR, "Condition signal failed in %s: %s", context, strerror(result));
        return -1;
    }
    log_message(LOG_DEBUG, "Condition signaled in %s", context);
    return 0;
}

// Deadlock detection utilities
typedef struct {
    pthread_t thread_id;
    const char* waiting_for;
    const char* holding;
    struct timespec wait_start;
} thread_wait_info_t;

#define MAX_THREADS 100
static thread_wait_info_t wait_info[MAX_THREADS];
static int wait_count = 0;
static pthread_mutex_t wait_info_mutex = PTHREAD_MUTEX_INITIALIZER;

void register_wait(const char* resource_name, const char* held_resource) {
    pthread_mutex_lock(&wait_info_mutex);
    
    if (wait_count < MAX_THREADS) {
        wait_info[wait_count].thread_id = pthread_self();
        wait_info[wait_count].waiting_for = resource_name;
        wait_info[wait_count].holding = held_resource;
        clock_gettime(CLOCK_MONOTONIC, &wait_info[wait_count].wait_start);
        wait_count++;
        
        log_message(LOG_DEBUG, "Thread %lu waiting for %s while holding %s", 
                   (unsigned long)pthread_self(), resource_name, 
                   held_resource ? held_resource : "nothing");
    }
    
    pthread_mutex_unlock(&wait_info_mutex);
}

void unregister_wait() {
    pthread_mutex_lock(&wait_info_mutex);
    
    pthread_t current_thread = pthread_self();
    for (int i = 0; i < wait_count; i++) {
        if (pthread_equal(wait_info[i].thread_id, current_thread)) {
            // Remove this entry by shifting others down
            for (int j = i; j < wait_count - 1; j++) {
                wait_info[j] = wait_info[j + 1];
            }
            wait_count--;
            break;
        }
    }
    
    pthread_mutex_unlock(&wait_info_mutex);
}

void check_for_deadlocks() {
    pthread_mutex_lock(&wait_info_mutex);
    
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    
    log_message(LOG_INFO, "=== Deadlock Detection Report ===");
    log_message(LOG_INFO, "Currently waiting threads: %d", wait_count);
    
    for (int i = 0; i < wait_count; i++) {
        double wait_time = (now.tv_sec - wait_info[i].wait_start.tv_sec) +
                          (now.tv_nsec - wait_info[i].wait_start.tv_nsec) / 1e9;
        
        log_message(LOG_WARNING, "Thread %lu: waiting %.2fs for %s, holding %s",
                   (unsigned long)wait_info[i].thread_id, wait_time,
                   wait_info[i].waiting_for, 
                   wait_info[i].holding ? wait_info[i].holding : "nothing");
        
        if (wait_time > 5.0) { // Potential deadlock if waiting > 5 seconds
            log_message(LOG_ERROR, "POTENTIAL DEADLOCK: Thread %lu has been waiting for %.2fs",
                       (unsigned long)wait_info[i].thread_id, wait_time);
        }
    }
    
    pthread_mutex_unlock(&wait_info_mutex);
}

// Error recovery mechanisms
typedef struct {
    int max_retries;
    int retry_delay_ms;
    int exponential_backoff;
} retry_config_t;

int retry_operation(int (*operation)(void*), void* arg, retry_config_t* config, const char* operation_name) {
    int attempt = 0;
    int delay = config->retry_delay_ms;
    
    while (attempt < config->max_retries) {
        log_message(LOG_DEBUG, "Attempting %s (attempt %d/%d)", 
                   operation_name, attempt + 1, config->max_retries);
        
        int result = operation(arg);
        if (result == 0) {
            if (attempt > 0) {
                log_message(LOG_INFO, "%s succeeded after %d retries", operation_name, attempt);
            }
            return 0;
        }
        
        attempt++;
        if (attempt < config->max_retries) {
            log_message(LOG_WARNING, "%s failed (attempt %d), retrying in %d ms", 
                       operation_name, attempt, delay);
            
            usleep(delay * 1000); // Convert to microseconds
            
            if (config->exponential_backoff) {
                delay *= 2;
            }
        }
    }
    
    log_message(LOG_ERROR, "%s failed after %d attempts", operation_name, config->max_retries);
    return -1;
}

// Resource leak detection
typedef struct {
    void* resource;
    const char* type;
    const char* allocated_at;
    struct timespec allocation_time;
    pthread_t owner_thread;
} resource_tracker_t;

#define MAX_RESOURCES 1000
static resource_tracker_t tracked_resources[MAX_RESOURCES];
static int resource_count = 0;
static pthread_mutex_t resource_tracker_mutex = PTHREAD_MUTEX_INITIALIZER;

void track_resource(void* resource, const char* type, const char* location) {
    pthread_mutex_lock(&resource_tracker_mutex);
    
    if (resource_count < MAX_RESOURCES) {
        tracked_resources[resource_count].resource = resource;
        tracked_resources[resource_count].type = type;
        tracked_resources[resource_count].allocated_at = location;
        clock_gettime(CLOCK_MONOTONIC, &tracked_resources[resource_count].allocation_time);
        tracked_resources[resource_count].owner_thread = pthread_self();
        resource_count++;
        
        log_message(LOG_DEBUG, "Tracking %s resource %p allocated at %s by thread %lu",
                   type, resource, location, (unsigned long)pthread_self());
    }
    
    pthread_mutex_unlock(&resource_tracker_mutex);
}

void untrack_resource(void* resource) {
    pthread_mutex_lock(&resource_tracker_mutex);
    
    for (int i = 0; i < resource_count; i++) {
        if (tracked_resources[i].resource == resource) {
            log_message(LOG_DEBUG, "Untracking %s resource %p", 
                       tracked_resources[i].type, resource);
            
            // Remove by shifting
            for (int j = i; j < resource_count - 1; j++) {
                tracked_resources[j] = tracked_resources[j + 1];
            }
            resource_count--;
            break;
        }
    }
    
    pthread_mutex_unlock(&resource_tracker_mutex);
}

void report_resource_leaks() {
    pthread_mutex_lock(&resource_tracker_mutex);
    
    if (resource_count == 0) {
        log_message(LOG_INFO, "No resource leaks detected");
    } else {
        log_message(LOG_WARNING, "Potential resource leaks detected: %d resources", resource_count);
        
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        
        for (int i = 0; i < resource_count; i++) {
            double age = (now.tv_sec - tracked_resources[i].allocation_time.tv_sec) +
                        (now.tv_nsec - tracked_resources[i].allocation_time.tv_nsec) / 1e9;
            
            log_message(LOG_WARNING, "Leaked %s resource %p (age: %.2fs) allocated at %s by thread %lu",
                       tracked_resources[i].type, tracked_resources[i].resource, age,
                       tracked_resources[i].allocated_at, 
                       (unsigned long)tracked_resources[i].owner_thread);
        }
    }
    
    pthread_mutex_unlock(&resource_tracker_mutex);
}

// Get error statistics
void get_error_stats() {
    pthread_mutex_lock(&global_logger.log_mutex);
    
    printf("\n=== Error Statistics ===\n");
    printf("Debug messages: %d\n", global_logger.log_count[LOG_DEBUG]);
    printf("Info messages: %d\n", global_logger.log_count[LOG_INFO]);
    printf("Warning messages: %d\n", global_logger.log_count[LOG_WARNING]);
    printf("Error messages: %d\n", global_logger.log_count[LOG_ERROR]);
    printf("Fatal messages: %d\n", global_logger.log_count[LOG_FATAL]);
    
    int total = 0;
    for (int i = 0; i < 5; i++) {
        total += global_logger.log_count[i];
    }
    printf("Total messages: %d\n", total);
    
    pthread_mutex_unlock(&global_logger.log_mutex);
}

// Cleanup error logger
void error_logger_cleanup() {
    get_error_stats();
    report_resource_leaks();
    
    if (global_logger.log_file && global_logger.log_file != stderr) {
        fclose(global_logger.log_file);
    }
}
```

## Performance Considerations and Optimization

### Synchronization Performance Analysis

Understanding the performance characteristics of different synchronization primitives is crucial for building efficient multi-threaded applications.

#### Performance Comparison Table

| Primitive | Overhead | Scalability | Fairness | Deadlock Risk | Best Use Case |
|-----------|----------|-------------|----------|---------------|---------------|
| **Mutex** | Medium | Good | Fair | Medium | General mutual exclusion |
| **Spinlock** | Low | Poor | Poor | Low | Very short critical sections |
| **RWLock** | High | Excellent | Good | Medium | Read-heavy workloads |
| **Semaphore** | Medium | Good | Good | Low | Resource counting |
| **Condition Variable** | Low | Excellent | Fair | Low | Event waiting |
| **Atomic Operations** | Very Low | Excellent | N/A | None | Simple operations |

#### Micro-benchmark Framework

```c
#include <time.h>
#include <sys/time.h>

typedef struct {
    const char* test_name;
    double min_time;
    double max_time;
    double avg_time;
    double total_time;
    int iterations;
    int operations_per_iteration;
} benchmark_result_t;

// High-resolution timing
static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Benchmark different synchronization methods
void benchmark_synchronization() {
    const int ITERATIONS = 10000;
    const int OPERATIONS = 1000;
    
    benchmark_result_t results[5];
    int test_count = 0;
    
    // Test 1: Mutex performance
    {
        pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
        double start_time = get_time();
        
        for (int i = 0; i < ITERATIONS; i++) {
            for (int j = 0; j < OPERATIONS; j++) {
                pthread_mutex_lock(&mutex);
                // Minimal critical section
                __asm__ volatile("" ::: "memory"); // Prevent optimization
                pthread_mutex_unlock(&mutex);
            }
        }
        
        double end_time = get_time();
        results[test_count++] = (benchmark_result_t){
            "Mutex Lock/Unlock", 0, 0, 0, end_time - start_time, ITERATIONS, OPERATIONS
        };
        
        pthread_mutex_destroy(&mutex);
    }
    
    // Test 2: Spinlock performance
    {
        pthread_spinlock_t spinlock;
        pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);
        
        double start_time = get_time();
        
        for (int i = 0; i < ITERATIONS; i++) {
            for (int j = 0; j < OPERATIONS; j++) {
                pthread_spin_lock(&spinlock);
                __asm__ volatile("" ::: "memory");
                pthread_spin_unlock(&spinlock);
            }
        }
        
        double end_time = get_time();
        results[test_count++] = (benchmark_result_t){
            "Spinlock Lock/Unlock", 0, 0, 0, end_time - start_time, ITERATIONS, OPERATIONS
        };
        
        pthread_spin_destroy(&spinlock);
    }
    
    // Test 3: Atomic operations
    {
        atomic_int counter = ATOMIC_VAR_INIT(0);
        double start_time = get_time();
        
        for (int i = 0; i < ITERATIONS; i++) {
            for (int j = 0; j < OPERATIONS; j++) {
                atomic_fetch_add(&counter, 1);
            }
        }
        
        double end_time = get_time();
        results[test_count++] = (benchmark_result_t){
            "Atomic Increment", 0, 0, 0, end_time - start_time, ITERATIONS, OPERATIONS
        };
    }
    
    // Test 4: Read-write lock (read-heavy)
    {
        pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
        double start_time = get_time();
        
        for (int i = 0; i < ITERATIONS; i++) {
            for (int j = 0; j < OPERATIONS; j++) {
                pthread_rwlock_rdlock(&rwlock);
                __asm__ volatile("" ::: "memory");
                pthread_rwlock_unlock(&rwlock);
            }
        }
        
        double end_time = get_time();
        results[test_count++] = (benchmark_result_t){
            "RWLock Read Lock/Unlock", 0, 0, 0, end_time - start_time, ITERATIONS, OPERATIONS
        };
        
        pthread_rwlock_destroy(&rwlock);
    }
    
    // Test 5: Condition variable signaling
    {
        pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
        pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
        
        double start_time = get_time();
        
        for (int i = 0; i < ITERATIONS; i++) {
            for (int j = 0; j < OPERATIONS; j++) {
                pthread_mutex_lock(&mutex);
                pthread_cond_signal(&cond);
                pthread_mutex_unlock(&mutex);
            }
        }
        
        double end_time = get_time();
        results[test_count++] = (benchmark_result_t){
            "Condition Signal", 0, 0, 0, end_time - start_time, ITERATIONS, OPERATIONS
        };
        
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&cond);
    }
    
    // Print results
    printf("\n=== Synchronization Performance Benchmark ===\n");
    printf("%-25s %12s %15s %15s\n", "Operation", "Total Time", "Ops/Second", "ns/Operation");
    printf("%-25s %12s %15s %15s\n", "-------------------------", "------------", "---------------", "---------------");
    
    for (int i = 0; i < test_count; i++) {
        long long total_ops = (long long)results[i].iterations * results[i].operations_per_iteration;
        double ops_per_sec = total_ops / results[i].total_time;
        double ns_per_op = (results[i].total_time * 1e9) / total_ops;
        
        printf("%-25s %10.3fs %12.0f %12.1f\n", 
               results[i].test_name, results[i].total_time, ops_per_sec, ns_per_op);
    }
}
```

### Memory Model and Cache Considerations

#### False Sharing Prevention

```c
// Bad: False sharing between threads
struct bad_counters {
    int counter1;  // Same cache line
    int counter2;  // Same cache line
};

// Good: Prevent false sharing
struct aligned_counters {
    alignas(64) int counter1;  // Own cache line
    alignas(64) int counter2;  // Own cache line
};

// Alternative: Padding
struct padded_counters {
    int counter1;
    char padding1[60];  // Fill rest of cache line
    int counter2;
    char padding2[60];
};
```

#### NUMA-Aware Synchronization

```c
#include <numa.h>
#include <numaif.h>

// NUMA-aware thread-local storage
typedef struct {
    int node_id;
    void* local_data;
    size_t data_size;
} numa_thread_data_t;

numa_thread_data_t* create_numa_local_data(size_t size) {
    numa_thread_data_t* tdata = malloc(sizeof(numa_thread_data_t));
    
    // Get current NUMA node
    tdata->node_id = numa_node_of_cpu(sched_getcpu());
    
    // Allocate memory on local NUMA node
    tdata->local_data = numa_alloc_onnode(size, tdata->node_id);
    tdata->data_size = size;
    
    printf("Allocated %zu bytes on NUMA node %d\n", size, tdata->node_id);
    return tdata;
}

void destroy_numa_local_data(numa_thread_data_t* tdata) {
    numa_free(tdata->local_data, tdata->data_size);
    free(tdata);
}
```

## Common Pitfalls and Solutions

### Deadlock Prevention and Detection

#### Ordered Lock Acquisition

```c
// WRONG: Can cause deadlock
void transfer_funds_wrong(account_t* from, account_t* to, double amount) {
    pthread_mutex_lock(&from->mutex);
    pthread_mutex_lock(&to->mutex);  // Potential deadlock if another thread locks in reverse order
    
    if (from->balance >= amount) {
        from->balance -= amount;
        to->balance += amount;
    }
    
    pthread_mutex_unlock(&to->mutex);
    pthread_mutex_unlock(&from->mutex);
}

// CORRECT: Always acquire locks in consistent order
void transfer_funds_correct(account_t* from, account_t* to, double amount) {
    // Order locks by address to prevent deadlock
    account_t* first = (from < to) ? from : to;
    account_t* second = (from < to) ? to : from;
    
    pthread_mutex_lock(&first->mutex);
    pthread_mutex_lock(&second->mutex);
    
    if (from->balance >= amount) {
        from->balance -= amount;
        to->balance += amount;
        printf("Transferred %.2f from account %p to %p\n", amount, from, to);
    } else {
        printf("Insufficient funds: %.2f available, %.2f requested\n", from->balance, amount);
    }
    
    pthread_mutex_unlock(&second->mutex);
    pthread_mutex_unlock(&first->mutex);
}
```

#### Timeout-Based Deadlock Recovery

```c
int safe_double_lock(pthread_mutex_t* mutex1, pthread_mutex_t* mutex2, int timeout_ms) {
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_ms / 1000;
    timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
    
    // Try first lock with timeout
    int result1 = pthread_mutex_timedlock(mutex1, &timeout);
    if (result1 != 0) {
        if (result1 == ETIMEDOUT) {
            printf("First lock timed out - potential deadlock avoided\n");
        }
        return result1;
    }
    
    // Try second lock with timeout
    int result2 = pthread_mutex_timedlock(mutex2, &timeout);
    if (result2 != 0) {
        pthread_mutex_unlock(mutex1); // Release first lock
        if (result2 == ETIMEDOUT) {
            printf("Second lock timed out - backing off\n");
        }
        return result2;
    }
    
    return 0; // Success
}
```

### Priority Inversion Solutions

```c
// Mutex with priority inheritance
pthread_mutexattr_t attr;
pthread_mutex_t priority_mutex;

void init_priority_mutex() {
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    pthread_mutex_init(&priority_mutex, &attr);
}

// Thread with real-time priority
void set_thread_priority(pthread_t thread, int priority) {
    struct sched_param param;
    param.sched_priority = priority;
    
    int result = pthread_setschedparam(thread, SCHED_FIFO, &param);
    if (result != 0) {
        fprintf(stderr, "Failed to set thread priority: %s\n", strerror(result));
    } else {
        printf("Thread priority set to %d\n", priority);
    }
}
```

## Comprehensive Exercises

### Exercise 1: Thread-Safe Bank System (Advanced)

**Objective**: Implement a complete banking system with multiple account types, transaction logging, and deadlock prevention.

**Requirements**:
1. Support checking and savings accounts with different interest rates
2. Implement thread-safe transfers between any accounts
3. Maintain transaction history with timestamps
4. Prevent deadlocks using ordered locking
5. Handle insufficient funds gracefully
6. Generate account statements

**Starter Code Framework**:
```c
typedef enum {
    ACCOUNT_CHECKING,
    ACCOUNT_SAVINGS
} account_type_t;

typedef struct transaction {
    double amount;
    account_type_t from_type;
    account_type_t to_type;
    struct timespec timestamp;
    char description[128];
    struct transaction* next;
} transaction_t;

typedef struct {
    int account_number;
    account_type_t type;
    double balance;
    double interest_rate;
    transaction_t* transaction_history;
    pthread_mutex_t mutex;
    int transaction_count;
} bank_account_t;

typedef struct {
    bank_account_t* accounts;
    int account_count;
    int max_accounts;
    pthread_rwlock_t bank_lock;
    double total_assets;
    int total_transactions;
} bank_system_t;

// Functions to implement:
// - bank_system_init()
// - create_account()
// - transfer_funds()
// - calculate_interest()
// - generate_statement()
// - get_bank_summary()
```

### Exercise 2: Producer-Consumer with Priority Queue (Intermediate)

**Objective**: Implement a multi-priority producer-consumer system with different service levels.

**Requirements**:
1. Support 3 priority levels: HIGH, MEDIUM, LOW
2. High priority items are always processed first
3. Implement starvation prevention for low priority items
4. Track processing statistics per priority level
5. Support multiple producers and consumers
6. Implement graceful shutdown

### Exercise 3: Readers-Writers with Writer Preference (Advanced)

**Objective**: Implement a readers-writers solution that prevents writer starvation.

**Requirements**:
1. Multiple readers can access data simultaneously
2. Only one writer can access data at a time
3. Writers have preference over readers to prevent starvation
4. Track access patterns and waiting times
5. Support timed operations with timeout
6. Implement fair scheduling

### Exercise 4: Dining Philosophers with Monitoring (Advanced)

**Objective**: Solve the dining philosophers problem with comprehensive monitoring and deadlock detection.

**Requirements**:
1. Prevent deadlock using resource ordering or limiting
2. Monitor philosopher states (thinking, hungry, eating)
3. Track eating frequency and duration
4. Detect and report potential starvation
5. Implement different strategies (waiter, resource hierarchy)
6. Generate real-time status reports

### Exercise 5: Lock-Free Data Structure (Expert)

**Objective**: Implement a lock-free stack or queue using atomic operations.

**Requirements**:
1. Use only atomic operations (no locks)
2. Handle ABA problem correctly
3. Implement memory reclamation strategy
4. Compare performance with lock-based version
5. Test with high contention scenarios
6. Document memory ordering requirements

## Assessment Criteria and Self-Evaluation

### Knowledge Assessment

**Level 1: Basic Understanding (Can you...)**
- [ ] Explain what race conditions are and why they occur?
- [ ] Use mutex correctly to protect critical sections?
- [ ] Understand the difference between mutex and semaphore?
- [ ] Implement simple producer-consumer pattern?
- [ ] Handle basic pthread errors?

**Level 2: Intermediate Application (Can you...)**
- [ ] Choose appropriate synchronization primitive for different scenarios?
- [ ] Implement readers-writers pattern with read-write locks?
- [ ] Use condition variables effectively for thread coordination?
- [ ] Avoid common deadlock scenarios?
- [ ] Design thread-safe data structures?

**Level 3: Advanced Mastery (Can you...)**
- [ ] Implement lock-free algorithms using atomic operations?
- [ ] Analyze and optimize synchronization performance?
- [ ] Handle complex synchronization patterns (barriers, monitors)?
- [ ] Debug and resolve synchronization issues in production code?
- [ ] Design scalable concurrent systems?

### Practical Assessment Rubric

| Criterion | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|-----------|---------------|----------|------------------|-----------------------|
| **Correctness** | No race conditions, handles all edge cases | Minor edge cases missed | Some race conditions present | Major correctness issues |
| **Performance** | Optimal synchronization choice, minimal overhead | Good performance with minor inefficiencies | Acceptable performance | Significant performance issues |
| **Error Handling** | Comprehensive error checking and recovery | Good error handling with minor gaps | Basic error handling | Poor or missing error handling |
| **Code Quality** | Clean, well-documented, maintainable | Well-structured with minor issues | Adequate structure | Poor structure and documentation |
| **Scalability** | Scales well under high contention | Good scalability with minor bottlenecks | Adequate scalability | Poor scalability |

### Performance Benchmarking Tasks

1. **Mutex vs Spinlock Comparison**
   - Measure performance under different critical section lengths
   - Test with varying thread counts
   - Analyze cache performance impact

2. **Lock Granularity Analysis**
   - Compare coarse-grained vs fine-grained locking
   - Measure throughput and latency
   - Identify optimal granularity for your workload

3. **Condition Variable Efficiency**
   - Compare with polling-based approaches
   - Measure wakeup latency
   - Test spurious wakeup handling

4. **Read-Write Lock Optimization**
   - Test with different read/write ratios
   - Compare with mutex for write-heavy workloads
   - Measure fairness characteristics

### Code Review Checklist

**Synchronization Design**
- [ ] Appropriate primitive chosen for each use case
- [ ] Consistent lock ordering to prevent deadlocks
- [ ] Minimal critical section length
- [ ] Proper cleanup of synchronization objects

**Error Handling**
- [ ] All pthread function return values checked
- [ ] Appropriate error recovery mechanisms
- [ ] Resource cleanup in error paths
- [ ] Meaningful error messages and logging

**Performance Considerations**
- [ ] False sharing avoided
- [ ] Lock contention minimized
- [ ] Memory barriers used correctly
- [ ] NUMA awareness where applicable

**Testing and Debugging**
- [ ] Race conditions tested under high contention
- [ ] Deadlock scenarios verified
- [ ] Memory leaks checked
- [ ] Stress testing performed

## Next Steps and Advanced Topics

### Recommended Next Learning Paths

1. **Advanced Concurrency Patterns**
   - Actor model implementation
   - Software transactional memory
   - Coroutines and fiber systems
   - Message passing architectures

2. **Lock-Free Programming**
   - Memory models and ordering
   - ABA problem and solutions
   - Hazard pointers and RCU
   - Performance analysis tools

3. **High-Performance Computing**
   - NUMA-aware programming
   - Cache-conscious algorithms
   - Parallel algorithm design
   - GPU programming (CUDA/OpenCL)

4. **Distributed Systems**
   - Distributed consensus algorithms
   - Replication and consistency
   - Network partitions and CAP theorem
   - Microservices synchronization

### Tools and Resources for Continued Learning

**Static Analysis Tools**
- ThreadSanitizer (TSan) for race condition detection
- Helgrind (Valgrind) for synchronization error detection
- Clang Static Analyzer for code quality

**Performance Profiling**
- Intel VTune for performance analysis
- perf for Linux performance monitoring
- Google Benchmark for microbenchmarking
- Flame graphs for visualization

**Books and References**
- "The Art of Multiprocessor Programming" by Herlihy & Shavit
- "Concurrent Programming on Windows" by Joe Duffy
- "Programming with POSIX Threads" by David Butenhof
- "Memory Models for C/C++ Programmers" by McKenney

**Online Resources**
- POSIX threads documentation
- Intel Threading Building Blocks (TBB)
- C++ concurrency reference
- Linux kernel synchronization primitives

## Next Section
[Condition Variables](04_Condition_Variables.md)
