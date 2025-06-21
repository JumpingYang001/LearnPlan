# Thread Synchronization Mechanisms

*Duration: 2 weeks*

## Overview

Thread synchronization is crucial for preventing race conditions and ensuring data consistency in multi-threaded programs. This section covers the primary synchronization primitives in pthreads.

## Race Conditions and Critical Sections

### Race Conditions
A race condition occurs when multiple threads access shared data concurrently, and the outcome depends on the timing of their execution.

### Critical Sections
Code sections that access shared resources and must be executed atomically by only one thread at a time.

## Mutex Locks (`pthread_mutex_t`)

### Basic Mutex Operations

```c
#include <pthread.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Or dynamic initialization
pthread_mutex_t mutex;
pthread_mutex_init(&mutex, NULL);

// Locking and unlocking
pthread_mutex_lock(&mutex);
// Critical section
pthread_mutex_unlock(&mutex);

// Cleanup
pthread_mutex_destroy(&mutex);
```

### Mutex Types

```c
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);

// Types:
// PTHREAD_MUTEX_NORMAL - Basic mutex (default)
// PTHREAD_MUTEX_RECURSIVE - Can be locked multiple times by same thread
// PTHREAD_MUTEX_ERRORCHECK - Error checking mutex
pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);

pthread_mutex_init(&mutex, &attr);
```

### Timed Mutex Operations

```c
#include <time.h>

struct timespec timeout;
clock_gettime(CLOCK_REALTIME, &timeout);
timeout.tv_sec += 5; // 5 second timeout

int result = pthread_mutex_timedlock(&mutex, &timeout);
if (result == ETIMEDOUT) {
    printf("Mutex lock timed out\n");
}
```

### Example: Thread-Safe Counter

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

int counter = 0;
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment_counter(void* arg) {
    int iterations = *(int*)arg;
    
    for (int i = 0; i < iterations; i++) {
        pthread_mutex_lock(&counter_mutex);
        counter++;
        pthread_mutex_unlock(&counter_mutex);
    }
    
    return NULL;
}

int main() {
    pthread_t threads[4];
    int iterations = 250000;
    
    // Create threads
    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, increment_counter, &iterations);
    }
    
    // Join threads
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Final counter value: %d\n", counter);
    printf("Expected value: %d\n", 4 * iterations);
    
    pthread_mutex_destroy(&counter_mutex);
    return 0;
}
```

## Read-Write Locks (`pthread_rwlock_t`)

Read-write locks allow multiple readers or a single writer, but not both simultaneously.

```c
#include <pthread.h>

pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;

// Reader lock
pthread_rwlock_rdlock(&rwlock);
// Read operation
pthread_rwlock_unlock(&rwlock);

// Writer lock
pthread_rwlock_wrlock(&rwlock);
// Write operation
pthread_rwlock_unlock(&rwlock);

// Cleanup
pthread_rwlock_destroy(&rwlock);
```

### Example: Reader-Writer Problem

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

int shared_data = 0;
pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;

void* reader(void* arg) {
    int id = *(int*)arg;
    
    for (int i = 0; i < 5; i++) {
        pthread_rwlock_rdlock(&rwlock);
        printf("Reader %d: Read value %d\n", id, shared_data);
        sleep(1);
        pthread_rwlock_unlock(&rwlock);
        usleep(100000);
    }
    
    return NULL;
}

void* writer(void* arg) {
    int id = *(int*)arg;
    
    for (int i = 0; i < 3; i++) {
        pthread_rwlock_wrlock(&rwlock);
        shared_data += 10;
        printf("Writer %d: Wrote value %d\n", id, shared_data);
        sleep(2);
        pthread_rwlock_unlock(&rwlock);
        sleep(1);
    }
    
    return NULL;
}
```

## Spinlocks

Spinlocks are useful for short critical sections where blocking would be more expensive than spinning.

```c
#include <pthread.h>

pthread_spinlock_t spinlock;

// Initialize
pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);

// Lock/unlock
pthread_spin_lock(&spinlock);
// Critical section (should be very short)
pthread_spin_unlock(&spinlock);

// Cleanup
pthread_spin_destroy(&spinlock);
```

## Barriers

Barriers allow threads to synchronize at specific points in execution.

```c
#include <pthread.h>

pthread_barrier_t barrier;

// Initialize for N threads
pthread_barrier_init(&barrier, NULL, N);

// Wait at barrier
int result = pthread_barrier_wait(&barrier);
if (result == PTHREAD_BARRIER_SERIAL_THREAD) {
    // Only one thread gets this return value
    printf("Last thread reached barrier\n");
}

// Cleanup
pthread_barrier_destroy(&barrier);
```

### Example: Parallel Computation with Barrier

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define NUM_THREADS 4
#define ARRAY_SIZE 1000

int array[ARRAY_SIZE];
int partial_sums[NUM_THREADS];
int total_sum = 0;
pthread_barrier_t barrier;
pthread_mutex_t sum_mutex = PTHREAD_MUTEX_INITIALIZER;

void* compute_partial_sum(void* arg) {
    int thread_id = *(int*)arg;
    int start = thread_id * (ARRAY_SIZE / NUM_THREADS);
    int end = (thread_id + 1) * (ARRAY_SIZE / NUM_THREADS);
    
    // Phase 1: Compute partial sum
    partial_sums[thread_id] = 0;
    for (int i = start; i < end; i++) {
        partial_sums[thread_id] += array[i];
    }
    
    printf("Thread %d: Partial sum = %d\n", thread_id, partial_sums[thread_id]);
    
    // Wait for all threads to complete Phase 1
    pthread_barrier_wait(&barrier);
    
    // Phase 2: One thread computes total sum
    if (thread_id == 0) {
        for (int i = 0; i < NUM_THREADS; i++) {
            total_sum += partial_sums[i];
        }
        printf("Total sum: %d\n", total_sum);
    }
    
    return NULL;
}
```

## Error Handling Best Practices

### Comprehensive Error Checking

```c
int check_pthread_error(int result, const char* function_name) {
    if (result != 0) {
        fprintf(stderr, "Error in %s: %s\n", function_name, strerror(result));
        return -1;
    }
    return 0;
}

// Usage
if (check_pthread_error(pthread_mutex_lock(&mutex), "pthread_mutex_lock") != 0) {
    // Handle error
}
```

## Common Synchronization Patterns

### 1. Producer-Consumer with Mutex
```c
pthread_mutex_t buffer_mutex = PTHREAD_MUTEX_INITIALIZER;
// Protect shared buffer with mutex
```

### 2. Reader-Writer Pattern
```c
pthread_rwlock_t data_lock = PTHREAD_RWLOCK_INITIALIZER;
// Multiple readers, single writer
```

### 3. Thread Pool Synchronization
```c
pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t work_available = PTHREAD_COND_INITIALIZER;
// Work queue with condition variables
```

## Performance Considerations

### Mutex vs Spinlock Trade-offs
- **Mutex**: Better for longer critical sections, thread sleeps when blocked
- **Spinlock**: Better for very short critical sections, thread spins when blocked

### Lock Granularity
- **Fine-grained**: More concurrent access, higher overhead
- **Coarse-grained**: Less concurrent access, lower overhead

### False Sharing
```c
// Bad: False sharing
struct {
    int counter1;
    int counter2;
} counters;

// Good: Prevent false sharing
struct {
    int counter1;
    char padding[64 - sizeof(int)];
    int counter2;
} counters;
```

## Exercises

1. **Bank Account Transfer**
   - Implement thread-safe money transfer between accounts
   - Prevent race conditions and ensure consistency

2. **Reader-Writer Database**
   - Simulate a database with multiple readers and writers
   - Use read-write locks for optimization

3. **Parallel Array Processing**
   - Process large arrays using multiple threads
   - Use barriers for synchronization between phases

4. **Lock Performance Comparison**
   - Compare mutex vs spinlock performance
   - Measure impact on different critical section sizes

## Common Pitfalls

1. **Deadlocks**
   ```c
   // Thread 1
   pthread_mutex_lock(&mutex1);
   pthread_mutex_lock(&mutex2);
   
   // Thread 2
   pthread_mutex_lock(&mutex2);
   pthread_mutex_lock(&mutex1); // Potential deadlock
   ```

2. **Forgetting to Unlock**
   ```c
   pthread_mutex_lock(&mutex);
   if (error_condition) {
       return; // BUG: Forgot to unlock!
   }
   pthread_mutex_unlock(&mutex);
   ```

3. **Using Wrong Lock Type**
   - Using spinlocks for long critical sections
   - Using mutex for very short operations

## Assessment

You should be able to:
- Implement thread-safe data structures using mutexes
- Choose appropriate synchronization primitives for different scenarios
- Avoid common synchronization pitfalls like deadlocks
- Analyze performance implications of different locking strategies
- Debug synchronization-related issues

## Next Section
[Condition Variables](04_Condition_Variables.md)
