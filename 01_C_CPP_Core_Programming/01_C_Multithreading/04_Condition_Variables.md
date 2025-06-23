# Condition Variables: Advanced Thread Coordination

*Duration: 1 week*

## Overview

Condition variables are sophisticated synchronization primitives that provide an efficient mechanism for threads to wait for specific conditions to become true and to signal other threads when conditions change. They are fundamental to implementing complex coordination patterns in multi-threaded applications and are essential for avoiding busy-waiting scenarios.

### The Problem Condition Variables Solve

Without condition variables, threads would need to use inefficient polling:

```c
// INEFFICIENT: Busy waiting (polling)
pthread_mutex_lock(&mutex);
while (!condition_met) {
    pthread_mutex_unlock(&mutex);
    usleep(1000); // Waste CPU cycles
    pthread_mutex_lock(&mutex);
}
// Condition is true, proceed...
pthread_mutex_unlock(&mutex);
```

With condition variables, threads can efficiently wait:

```c
// EFFICIENT: Event-driven waiting
pthread_mutex_lock(&mutex);
while (!condition_met) {
    pthread_cond_wait(&condition, &mutex); // Sleep until signaled
}
// Condition is true, proceed...
pthread_mutex_unlock(&mutex);
```

### Why Condition Variables Matter

1. **Efficiency**: Threads sleep instead of consuming CPU cycles
2. **Responsiveness**: Immediate notification when conditions change
3. **Scalability**: No performance degradation with waiting threads
4. **Flexibility**: Support complex coordination patterns
5. **Standard**: Part of POSIX threading standard

### The Condition Variable Ecosystem

```
┌─────────────────────────────────────────────────────────────┐
│                    Condition Variable System                │
├─────────────────────────────────────────────────────────────┤
│  Condition Variable  │  Associated Mutex  │  Shared State    │
│  (pthread_cond_t)   │  (pthread_mutex_t) │  (application)   │
├─────────────────────────────────────────────────────────────┤
│  Operations:        │  Protection:       │  Condition:      │
│  • wait()          │  • lock()          │  • predicate     │
│  • signal()        │  • unlock()        │  • shared data   │
│  • broadcast()     │  • try_lock()      │  • flags/counts  │
│  • timedwait()     │  • timedlock()     │  • state info    │
└─────────────────────────────────────────────────────────────┘
```

## Fundamental Concepts and Mechanics

### The Atomic Wait Operation

The key insight of condition variables is the atomic "unlock and wait" operation:

```c
// This is what pthread_cond_wait() does atomically:
// 1. Unlock the mutex
// 2. Put thread to sleep on condition variable
// 3. When signaled: wake up and relock mutex
// 4. Return to caller (with mutex locked)

void atomic_wait_demonstration() {
    pthread_mutex_lock(&mutex);
    
    while (!condition_is_true) {
        // These steps happen ATOMICALLY in pthread_cond_wait():
        // Step 1: Unlock mutex (other threads can now modify condition)
        // Step 2: Thread goes to sleep waiting for signal
        // Step 3: When signaled, thread wakes up
        // Step 4: Thread re-acquires mutex before returning
        
        pthread_cond_wait(&condition, &mutex);
        
        // At this point:
        // - Thread is awake
        // - Mutex is locked
        // - Condition might have changed (need to recheck)
    }
    
    // Condition is true AND mutex is locked
    // Safe to proceed with work
    
    pthread_mutex_unlock(&mutex);
}
```

### Spurious Wakeups: Why Always Use While Loops

Spurious wakeups can occur due to:
1. **System-level optimizations**
2. **Signal handling**
3. **Hardware interrupts**
4. **Implementation details**

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>

// Demonstration of spurious wakeup handling
typedef struct {
    int value;
    int target;
    int spurious_wakeups;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
} spurious_demo_t;

void* waiter_thread(void* arg) {
    spurious_demo_t* demo = (spurious_demo_t*)arg;
    int wakeup_count = 0;
    
    pthread_mutex_lock(&demo->mutex);
    
    printf("Waiter: Starting to wait for value to reach %d\n", demo->target);
    
    while (demo->value < demo->target) {
        printf("Waiter: Value is %d, waiting... (wakeup #%d)\n", 
               demo->value, ++wakeup_count);
        
        int wait_result = pthread_cond_wait(&demo->condition, &demo->mutex);
        
        if (wait_result != 0) {
            fprintf(stderr, "Condition wait failed: %s\n", strerror(wait_result));
            break;
        }
        
        printf("Waiter: Woke up! Value is now %d\n", demo->value);
        
        // Check if this was a spurious wakeup
        if (demo->value < demo->target) {
            demo->spurious_wakeups++;
            printf("Waiter: Spurious wakeup detected! (total: %d)\n", 
                   demo->spurious_wakeups);
        }
    }
    
    printf("Waiter: Condition met! Value reached %d after %d wakeups (%d spurious)\n",
           demo->value, wakeup_count, demo->spurious_wakeups);
    
    pthread_mutex_unlock(&demo->mutex);
    return NULL;
}

void* modifier_thread(void* arg) {
    spurious_demo_t* demo = (spurious_demo_t*)arg;
    
    // Gradually increase value with some randomness
    for (int i = 1; i <= demo->target; i++) {
        usleep(100000 + rand() % 200000); // Random delay 100-300ms
        
        pthread_mutex_lock(&demo->mutex);
        demo->value = i;
        printf("Modifier: Set value to %d\n", demo->value);
        
        // Signal the waiting thread
        pthread_cond_signal(&demo->condition);
        
        pthread_mutex_unlock(&demo->mutex);
    }
    
    return NULL;
}
```

## Core API and Advanced Usage

### Comprehensive Error Handling Framework

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

// Enhanced condition variable operations with error handling
typedef struct {
    pthread_cond_t cond;
    pthread_mutex_t mutex;
    const char* name;
    
    // Statistics
    int wait_count;
    int signal_count;
    int broadcast_count;
    int timeout_count;
    double total_wait_time;
} enhanced_condition_t;

// Initialize condition variable with error checking
int enhanced_cond_init(enhanced_condition_t* econd, const char* name) {
    econd->name = name;
    econd->wait_count = 0;
    econd->signal_count = 0;
    econd->broadcast_count = 0;
    econd->timeout_count = 0;
    econd->total_wait_time = 0.0;
    
    // Initialize mutex first
    int result = pthread_mutex_init(&econd->mutex, NULL);
    if (result != 0) {
        fprintf(stderr, "Failed to initialize mutex for %s: %s\n", 
                name, strerror(result));
        return -1;
    }
    
    // Initialize condition variable
    result = pthread_cond_init(&econd->cond, NULL);
    if (result != 0) {
        fprintf(stderr, "Failed to initialize condition variable %s: %s\n", 
                name, strerror(result));
        pthread_mutex_destroy(&econd->mutex);
        return -1;
    }
    
    printf("Enhanced condition variable '%s' initialized successfully\n", name);
    return 0;
}

// Safe condition wait with timing and error handling
int enhanced_cond_wait(enhanced_condition_t* econd, int* condition, 
                      const char* context) {
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    int result = pthread_mutex_lock(&econd->mutex);
    if (result != 0) {
        fprintf(stderr, "Failed to lock mutex in %s for %s: %s\n", 
                context, econd->name, strerror(result));
        return -1;
    }
    
    printf("%s: Checking condition for %s\n", context, econd->name);
    
    while (!(*condition)) {
        printf("%s: Condition not met for %s, waiting...\n", context, econd->name);
        
        result = pthread_cond_wait(&econd->cond, &econd->mutex);
        if (result != 0) {
            fprintf(stderr, "Condition wait failed in %s for %s: %s\n", 
                    context, econd->name, strerror(result));
            pthread_mutex_unlock(&econd->mutex);
            return -1;
        }
        
        printf("%s: Woke up, rechecking condition for %s\n", context, econd->name);
    }
    
    // Update statistics
    econd->wait_count++;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double wait_time = (end_time.tv_sec - start_time.tv_sec) + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    econd->total_wait_time += wait_time;
    
    printf("%s: Condition met for %s (wait time: %.3fs)\n", 
           context, econd->name, wait_time);
    
    pthread_mutex_unlock(&econd->mutex);
    return 0;
}

// Safe condition signal with error handling
int enhanced_cond_signal(enhanced_condition_t* econd, const char* context) {
    int result = pthread_cond_signal(&econd->cond);
    if (result != 0) {
        fprintf(stderr, "Failed to signal condition %s from %s: %s\n", 
                econd->name, context, strerror(result));
        return -1;
    }
    
    econd->signal_count++;
    printf("%s: Signaled condition %s (total signals: %d)\n", 
           context, econd->name, econd->signal_count);
    return 0;
}

// Safe condition broadcast with error handling
int enhanced_cond_broadcast(enhanced_condition_t* econd, const char* context) {
    int result = pthread_cond_broadcast(&econd->cond);
    if (result != 0) {
        fprintf(stderr, "Failed to broadcast condition %s from %s: %s\n", 
                econd->name, context, strerror(result));
        return -1;
    }
    
    econd->broadcast_count++;
    printf("%s: Broadcasted condition %s (total broadcasts: %d)\n", 
           context, econd->name, econd->broadcast_count);
    return 0;
}

// Timed condition wait with comprehensive error handling
int enhanced_cond_timedwait(enhanced_condition_t* econd, int* condition, 
                           int timeout_seconds, const char* context) {
    struct timespec timeout, start_time, end_time;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_seconds;
    
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    int result = pthread_mutex_lock(&econd->mutex);
    if (result != 0) {
        fprintf(stderr, "Failed to lock mutex in %s for %s: %s\n", 
                context, econd->name, strerror(result));
        return -1;
    }
    
    while (!(*condition)) {
        printf("%s: Condition not met for %s, waiting with %ds timeout...\n", 
               context, econd->name, timeout_seconds);
        
        result = pthread_cond_timedwait(&econd->cond, &econd->mutex, &timeout);
        
        if (result == ETIMEDOUT) {
            econd->timeout_count++;
            printf("%s: Timeout waiting for condition %s (total timeouts: %d)\n", 
                   context, econd->name, econd->timeout_count);
            pthread_mutex_unlock(&econd->mutex);
            return 1; // Timeout
        } else if (result != 0) {
            fprintf(stderr, "Timed wait failed in %s for %s: %s\n", 
                    context, econd->name, strerror(result));
            pthread_mutex_unlock(&econd->mutex);
            return -1;
        }
        
        printf("%s: Woke up within timeout, rechecking condition for %s\n", 
               context, econd->name);
    }
    
    // Update statistics
    econd->wait_count++;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double wait_time = (end_time.tv_sec - start_time.tv_sec) + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    econd->total_wait_time += wait_time;
    
    printf("%s: Condition met for %s within timeout (wait time: %.3fs)\n", 
           context, econd->name, wait_time);
    
    pthread_mutex_unlock(&econd->mutex);
    return 0; // Success
}

// Get condition variable statistics
void enhanced_cond_get_stats(enhanced_condition_t* econd) {
    printf("\n=== Statistics for Condition Variable '%s' ===\n", econd->name);
    printf("Total waits: %d\n", econd->wait_count);
    printf("Total signals: %d\n", econd->signal_count);
    printf("Total broadcasts: %d\n", econd->broadcast_count);
    printf("Total timeouts: %d\n", econd->timeout_count);
    printf("Total wait time: %.3f seconds\n", econd->total_wait_time);
    if (econd->wait_count > 0) {
        printf("Average wait time: %.3f seconds\n", 
               econd->total_wait_time / econd->wait_count);
    }
    printf("Signal/Wait ratio: %.2f\n", 
           econd->wait_count > 0 ? (double)econd->signal_count / econd->wait_count : 0);
}

// Cleanup enhanced condition variable
void enhanced_cond_destroy(enhanced_condition_t* econd) {
    enhanced_cond_get_stats(econd);
    
    pthread_cond_destroy(&econd->cond);
    pthread_mutex_destroy(&econd->mutex);
    
    printf("Enhanced condition variable '%s' destroyed\n", econd->name);
}
```
## Advanced Producer-Consumer Patterns

### Multi-Priority Producer-Consumer System

Real-world applications often need priority-based processing. Here's a comprehensive implementation:

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_BUFFER_SIZE 100
#define NUM_PRIORITIES 3
#define HIGH_PRIORITY 0
#define MEDIUM_PRIORITY 1
#define LOW_PRIORITY 2

typedef struct work_item {
    int id;
    int priority;
    char description[128];
    struct timespec created_time;
    struct timespec start_time;
    struct timespec end_time;
    int processing_thread_id;
    struct work_item* next;
} work_item_t;

typedef struct {
    work_item_t* head[NUM_PRIORITIES];  // Priority queues
    work_item_t* tail[NUM_PRIORITIES];
    int count[NUM_PRIORITIES];
    int total_count;
    int max_size;
    
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
    pthread_cond_t priority_available[NUM_PRIORITIES];
    
    // Statistics
    int total_produced;
    int total_consumed;
    int items_by_priority[NUM_PRIORITIES];
    int processed_by_priority[NUM_PRIORITIES];
    double avg_wait_time[NUM_PRIORITIES];
    int starvation_prevention_count;
    
    // Fairness mechanism
    int low_priority_starved_count;
    int fairness_threshold;
} priority_buffer_t;

// Initialize priority buffer
int priority_buffer_init(priority_buffer_t* buffer, int max_size, int fairness_threshold) {
    buffer->max_size = max_size;
    buffer->total_count = 0;
    buffer->total_produced = 0;
    buffer->total_consumed = 0;
    buffer->low_priority_starved_count = 0;
    buffer->fairness_threshold = fairness_threshold;
    buffer->starvation_prevention_count = 0;
    
    // Initialize priority queues
    for (int i = 0; i < NUM_PRIORITIES; i++) {
        buffer->head[i] = NULL;
        buffer->tail[i] = NULL;
        buffer->count[i] = 0;
        buffer->items_by_priority[i] = 0;
        buffer->processed_by_priority[i] = 0;
        buffer->avg_wait_time[i] = 0.0;
    }
    
    // Initialize synchronization primitives
    if (pthread_mutex_init(&buffer->mutex, NULL) != 0) {
        return -1;
    }
    
    if (pthread_cond_init(&buffer->not_full, NULL) != 0) {
        pthread_mutex_destroy(&buffer->mutex);
        return -1;
    }
    
    if (pthread_cond_init(&buffer->not_empty, NULL) != 0) {
        pthread_cond_destroy(&buffer->not_full);
        pthread_mutex_destroy(&buffer->mutex);
        return -1;
    }
    
    for (int i = 0; i < NUM_PRIORITIES; i++) {
        if (pthread_cond_init(&buffer->priority_available[i], NULL) != 0) {
            // Cleanup previously initialized condition variables
            for (int j = 0; j < i; j++) {
                pthread_cond_destroy(&buffer->priority_available[j]);
            }
            pthread_cond_destroy(&buffer->not_empty);
            pthread_cond_destroy(&buffer->not_full);
            pthread_mutex_destroy(&buffer->mutex);
            return -1;
        }
    }
    
    printf("Priority buffer initialized: max_size=%d, fairness_threshold=%d\n", 
           max_size, fairness_threshold);
    return 0;
}

// Produce item with priority
int priority_buffer_produce(priority_buffer_t* buffer, work_item_t* item, int producer_id) {
    if (item->priority < 0 || item->priority >= NUM_PRIORITIES) {
        fprintf(stderr, "Invalid priority: %d\n", item->priority);
        return -1;
    }
    
    pthread_mutex_lock(&buffer->mutex);
    
    // Wait while buffer is full
    while (buffer->total_count >= buffer->max_size) {
        printf("Producer %d: Buffer full (%d/%d), waiting...\n", 
               producer_id, buffer->total_count, buffer->max_size);
        pthread_cond_wait(&buffer->not_full, &buffer->mutex);
    }
    
    // Add timestamp
    clock_gettime(CLOCK_REALTIME, &item->created_time);
    
    // Add to appropriate priority queue
    int priority = item->priority;
    if (buffer->tail[priority]) {
        buffer->tail[priority]->next = item;
    } else {
        buffer->head[priority] = item;
    }
    buffer->tail[priority] = item;
    item->next = NULL;
    
    // Update counts
    buffer->count[priority]++;
    buffer->total_count++;
    buffer->total_produced++;
    buffer->items_by_priority[priority]++;
    
    const char* priority_names[] = {"HIGH", "MEDIUM", "LOW"};
    printf("Producer %d: Added %s priority item %d '%s' (queue: %d/%d total: %d/%d)\n",
           producer_id, priority_names[priority], item->id, item->description,
           buffer->count[priority], buffer->total_count, 
           buffer->items_by_priority[priority], buffer->total_produced);
    
    // Signal appropriate conditions
    pthread_cond_signal(&buffer->not_empty);
    pthread_cond_signal(&buffer->priority_available[priority]);
    
    pthread_mutex_unlock(&buffer->mutex);
    return 0;
}

// Consume item with priority preference
work_item_t* priority_buffer_consume(priority_buffer_t* buffer, int consumer_id) {
    pthread_mutex_lock(&buffer->mutex);
    
    // Wait while buffer is empty
    while (buffer->total_count == 0) {
        printf("Consumer %d: Buffer empty, waiting...\n", consumer_id);
        pthread_cond_wait(&buffer->not_empty, &buffer->mutex);
    }
    
    work_item_t* item = NULL;
    int selected_priority = -1;
    
    // Fairness mechanism: check if low priority items are being starved
    if (buffer->count[LOW_PRIORITY] > 0 && 
        buffer->low_priority_starved_count >= buffer->fairness_threshold) {
        
        // Force processing of low priority item
        selected_priority = LOW_PRIORITY;
        buffer->starvation_prevention_count++;
        buffer->low_priority_starved_count = 0;
        
        printf("Consumer %d: Starvation prevention - processing LOW priority item\n", consumer_id);
    } else {
        // Normal priority-based selection
        for (int p = HIGH_PRIORITY; p < NUM_PRIORITIES; p++) {
            if (buffer->count[p] > 0) {
                selected_priority = p;
                break;
            }
        }
        
        // Track potential starvation
        if (selected_priority != LOW_PRIORITY && buffer->count[LOW_PRIORITY] > 0) {
            buffer->low_priority_starved_count++;
        }
    }
    
    if (selected_priority >= 0) {
        // Remove item from selected priority queue
        item = buffer->head[selected_priority];
        buffer->head[selected_priority] = item->next;
        if (!buffer->head[selected_priority]) {
            buffer->tail[selected_priority] = NULL;
        }
        
        // Update counts
        buffer->count[selected_priority]--;
        buffer->total_count--;
        buffer->total_consumed++;
        buffer->processed_by_priority[selected_priority]++;
        
        // Add processing timestamps
        clock_gettime(CLOCK_REALTIME, &item->start_time);
        item->processing_thread_id = consumer_id;
        
        // Calculate wait time
        double wait_time = 
            (item->start_time.tv_sec - item->created_time.tv_sec) +
            (item->start_time.tv_nsec - item->created_time.tv_nsec) / 1e9;
        
        // Update average wait time
        int processed = buffer->processed_by_priority[selected_priority];
        buffer->avg_wait_time[selected_priority] = 
            (buffer->avg_wait_time[selected_priority] * (processed - 1) + wait_time) / processed;
        
        const char* priority_names[] = {"HIGH", "MEDIUM", "LOW"};
        printf("Consumer %d: Processing %s priority item %d (wait: %.3fs, queue: %d)\n",
               consumer_id, priority_names[selected_priority], item->id, 
               wait_time, buffer->count[selected_priority]);
    }
    
    // Signal that buffer is not full
    if (buffer->total_count < buffer->max_size) {
        pthread_cond_signal(&buffer->not_full);
    }
    
    pthread_mutex_unlock(&buffer->mutex);
    return item;
}

// Get buffer statistics
void priority_buffer_get_stats(priority_buffer_t* buffer) {
    pthread_mutex_lock(&buffer->mutex);
    
    printf("\n=== Priority Buffer Statistics ===\n");
    printf("Buffer capacity: %d\n", buffer->max_size);
    printf("Current total items: %d\n", buffer->total_count);
    printf("Total produced: %d\n", buffer->total_produced);
    printf("Total consumed: %d\n", buffer->total_consumed);
    printf("Starvation prevention activations: %d\n", buffer->starvation_prevention_count);
    printf("Current low priority starvation count: %d\n", buffer->low_priority_starved_count);
    
    const char* priority_names[] = {"HIGH", "MEDIUM", "LOW"};
    printf("\nPer-Priority Statistics:\n");
    for (int i = 0; i < NUM_PRIORITIES; i++) {
        printf("  %s Priority:\n", priority_names[i]);
        printf("    Current queue size: %d\n", buffer->count[i]);
        printf("    Total produced: %d\n", buffer->items_by_priority[i]);
        printf("    Total processed: %d\n", buffer->processed_by_priority[i]);
        printf("    Average wait time: %.3f seconds\n", buffer->avg_wait_time[i]);
        if (buffer->items_by_priority[i] > 0) {
            double throughput = (double)buffer->processed_by_priority[i] / buffer->items_by_priority[i] * 100;
            printf("    Throughput: %.1f%%\n", throughput);
        }
    }
    
    pthread_mutex_unlock(&buffer->mutex);
}

// Producer thread function
void* priority_producer_thread(void* arg) {
    struct {
        priority_buffer_t* buffer;
        int producer_id;
        int items_to_produce;
    } *data = arg;
    
    srand(time(NULL) + data->producer_id);
    
    for (int i = 0; i < data->items_to_produce; i++) {
        work_item_t* item = malloc(sizeof(work_item_t));
        if (!item) {
            fprintf(stderr, "Producer %d: Failed to allocate memory\n", data->producer_id);
            continue;
        }
        
        item->id = data->producer_id * 1000 + i;
        item->priority = rand() % NUM_PRIORITIES;
        snprintf(item->description, sizeof(item->description), 
                "Task from Producer %d, Item %d", data->producer_id, i);
        
        if (priority_buffer_produce(data->buffer, item, data->producer_id) != 0) {
            free(item);
            fprintf(stderr, "Producer %d: Failed to produce item %d\n", 
                    data->producer_id, i);
        }
        
        // Variable production rate
        usleep(50000 + rand() % 100000); // 50-150ms
    }
    
    printf("Producer %d: Finished producing %d items\n", 
           data->producer_id, data->items_to_produce);
    return NULL;
}

// Consumer thread function
void* priority_consumer_thread(void* arg) {
    struct {
        priority_buffer_t* buffer;
        int consumer_id;
        int items_to_consume;
    } *data = arg;
    
    int items_consumed = 0;
    
    while (items_consumed < data->items_to_consume) {
        work_item_t* item = priority_buffer_consume(data->buffer, data->consumer_id);
        
        if (item) {
            // Simulate processing time based on priority
            int processing_time = 0;
            switch (item->priority) {
                case HIGH_PRIORITY:   processing_time = 50000 + rand() % 50000; break;   // 50-100ms
                case MEDIUM_PRIORITY: processing_time = 100000 + rand() % 100000; break; // 100-200ms
                case LOW_PRIORITY:    processing_time = 200000 + rand() % 200000; break; // 200-400ms
            }
            
            usleep(processing_time);
            
            // Mark completion
            clock_gettime(CLOCK_REALTIME, &item->end_time);
            double processing_duration = 
                (item->end_time.tv_sec - item->start_time.tv_sec) +
                (item->end_time.tv_nsec - item->start_time.tv_nsec) / 1e9;
            
            printf("Consumer %d: Completed item %d in %.3fs\n", 
                   data->consumer_id, item->id, processing_duration);
            
            free(item);
            items_consumed++;
        } else {
            // This shouldn't happen in normal operation
            printf("Consumer %d: No item received, retrying...\n", data->consumer_id);
            usleep(10000);
        }
    }
    
    printf("Consumer %d: Finished consuming %d items\n", 
           data->consumer_id, items_consumed);
    return NULL;
}

// Cleanup priority buffer
void priority_buffer_destroy(priority_buffer_t* buffer) {
    pthread_mutex_lock(&buffer->mutex);
    
    // Free any remaining items
    for (int p = 0; p < NUM_PRIORITIES; p++) {
        work_item_t* current = buffer->head[p];
        while (current) {
            work_item_t* next = current->next;
            free(current);
            current = next;
        }
    }
    
    pthread_mutex_unlock(&buffer->mutex);
    
    // Destroy synchronization primitives
    for (int i = 0; i < NUM_PRIORITIES; i++) {
        pthread_cond_destroy(&buffer->priority_available[i]);
    }
    pthread_cond_destroy(&buffer->not_empty);
    pthread_cond_destroy(&buffer->not_full);
    pthread_mutex_destroy(&buffer->mutex);
    
    printf("Priority buffer destroyed\n");
}
```

### Thread Pool with Work Queue

A practical implementation of a thread pool using condition variables:

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#define MAX_THREADS 10
#define MAX_QUEUE_SIZE 100

typedef struct work_item {
    void (*function)(void* arg);
    void* argument;
    int priority;
    struct timespec submitted_time;
    struct work_item* next;
} work_item_t;

typedef struct {
    pthread_t threads[MAX_THREADS];
    int num_threads;
    
    work_item_t* work_queue_head;
    work_item_t* work_queue_tail;
    int queue_size;
    int max_queue_size;
    
    pthread_mutex_t queue_mutex;
    pthread_cond_t work_available;
    pthread_cond_t queue_not_full;
    
    int shutdown;
    int threads_working;
    
    // Statistics
    int total_tasks_submitted;
    int total_tasks_completed;
    int total_tasks_rejected;
    double total_execution_time;
    double total_wait_time;
} thread_pool_t;

// Initialize thread pool
int thread_pool_init(thread_pool_t* pool, int num_threads, int max_queue_size) {
    if (num_threads > MAX_THREADS || max_queue_size > MAX_QUEUE_SIZE) {
        return -1;
    }
    
    pool->num_threads = num_threads;
    pool->max_queue_size = max_queue_size;
    pool->work_queue_head = NULL;
    pool->work_queue_tail = NULL;
    pool->queue_size = 0;
    pool->shutdown = 0;
    pool->threads_working = 0;
    
    pool->total_tasks_submitted = 0;
    pool->total_tasks_completed = 0;
    pool->total_tasks_rejected = 0;
    pool->total_execution_time = 0.0;
    pool->total_wait_time = 0.0;
    
    // Initialize synchronization primitives
    if (pthread_mutex_init(&pool->queue_mutex, NULL) != 0) {
        return -1;
    }
    
    if (pthread_cond_init(&pool->work_available, NULL) != 0) {
        pthread_mutex_destroy(&pool->queue_mutex);
        return -1;
    }
    
    if (pthread_cond_init(&pool->queue_not_full, NULL) != 0) {
        pthread_cond_destroy(&pool->work_available);
        pthread_mutex_destroy(&pool->queue_mutex);
        return -1;
    }
    
    // Create worker threads
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL, thread_pool_worker, pool) != 0) {
            // Cleanup on failure
            pool->shutdown = 1;
            for (int j = 0; j < i; j++) {
                pthread_cancel(pool->threads[j]);
                pthread_join(pool->threads[j], NULL);
            }
            pthread_cond_destroy(&pool->queue_not_full);
            pthread_cond_destroy(&pool->work_available);
            pthread_mutex_destroy(&pool->queue_mutex);
            return -1;
        }
    }
    
    printf("Thread pool initialized with %d threads, max queue size: %d\n", 
           num_threads, max_queue_size);
    return 0;
}

// Worker thread function
void* thread_pool_worker(void* arg) {
    thread_pool_t* pool = (thread_pool_t*)arg;
    pthread_t thread_id = pthread_self();
    
    printf("Worker thread %lu started\n", (unsigned long)thread_id);
    
    while (1) {
        pthread_mutex_lock(&pool->queue_mutex);
        
        // Wait for work or shutdown
        while (pool->queue_size == 0 && !pool->shutdown) {
            printf("Worker %lu: Waiting for work...\n", (unsigned long)thread_id);
            pthread_cond_wait(&pool->work_available, &pool->queue_mutex);
        }
        
        // Check for shutdown
        if (pool->shutdown && pool->queue_size == 0) {
            pthread_mutex_unlock(&pool->queue_mutex);
            printf("Worker %lu: Shutting down\n", (unsigned long)thread_id);
            break;
        }
        
        // Get work item
        work_item_t* work = pool->work_queue_head;
        pool->work_queue_head = work->next;
        if (!pool->work_queue_head) {
            pool->work_queue_tail = NULL;
        }
        pool->queue_size--;
        pool->threads_working++;
        
        // Signal that queue is not full
        pthread_cond_signal(&pool->queue_not_full);
        
        pthread_mutex_unlock(&pool->queue_mutex);
        
        // Execute work
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        double wait_time = (start_time.tv_sec - work->submitted_time.tv_sec) +
                          (start_time.tv_nsec - work->submitted_time.tv_nsec) / 1e9;
        
        printf("Worker %lu: Executing task (waited %.3fs)\n", 
               (unsigned long)thread_id, wait_time);
        
        work->function(work->argument);
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double execution_time = (end_time.tv_sec - start_time.tv_sec) +
                               (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        
        printf("Worker %lu: Completed task in %.3fs\n", 
               (unsigned long)thread_id, execution_time);
        
        // Update statistics
        pthread_mutex_lock(&pool->queue_mutex);
        pool->threads_working--;
        pool->total_tasks_completed++;
        pool->total_execution_time += execution_time;
        pool->total_wait_time += wait_time;
        pthread_mutex_unlock(&pool->queue_mutex);
        
        free(work);
    }
    
    return NULL;
}

// Submit work to thread pool
int thread_pool_submit(thread_pool_t* pool, void (*function)(void*), void* argument, int priority) {
    if (pool->shutdown) {
        return -1;
    }
    
    work_item_t* work = malloc(sizeof(work_item_t));
    if (!work) {
        return -1;
    }
    
    work->function = function;
    work->argument = argument;
    work->priority = priority;
    clock_gettime(CLOCK_MONOTONIC, &work->submitted_time);
    work->next = NULL;
    
    pthread_mutex_lock(&pool->queue_mutex);
    
    // Check if queue is full
    if (pool->queue_size >= pool->max_queue_size) {
        pool->total_tasks_rejected++;
        pthread_mutex_unlock(&pool->queue_mutex);
        free(work);
        printf("Thread pool: Task rejected - queue full (%d/%d)\n", 
               pool->queue_size, pool->max_queue_size);
        return -1;
    }
    
    // Add to queue
    if (pool->work_queue_tail) {
        pool->work_queue_tail->next = work;
    } else {
        pool->work_queue_head = work;
    }
    pool->work_queue_tail = work;
    pool->queue_size++;
    pool->total_tasks_submitted++;
    
    printf("Thread pool: Task submitted (queue: %d/%d, total: %d)\n", 
           pool->queue_size, pool->max_queue_size, pool->total_tasks_submitted);
    
    // Signal workers
    pthread_cond_signal(&pool->work_available);
    
    pthread_mutex_unlock(&pool->queue_mutex);
    return 0;
}

// Wait for all tasks to complete
void thread_pool_wait_completion(thread_pool_t* pool) {
    pthread_mutex_lock(&pool->queue_mutex);
    
    while (pool->queue_size > 0 || pool->threads_working > 0) {
        pthread_mutex_unlock(&pool->queue_mutex);
        printf("Waiting for completion: queue=%d, working=%d\n", 
               pool->queue_size, pool->threads_working);
        usleep(100000); // 100ms
        pthread_mutex_lock(&pool->queue_mutex);
    }
    
    pthread_mutex_unlock(&pool->queue_mutex);
    printf("All tasks completed\n");
}

// Get thread pool statistics
void thread_pool_get_stats(thread_pool_t* pool) {
    pthread_mutex_lock(&pool->queue_mutex);
    
    printf("\n=== Thread Pool Statistics ===\n");
    printf("Number of threads: %d\n", pool->num_threads);
    printf("Max queue size: %d\n", pool->max_queue_size);
    printf("Current queue size: %d\n", pool->queue_size);
    printf("Threads currently working: %d\n", pool->threads_working);
    printf("Total tasks submitted: %d\n", pool->total_tasks_submitted);
    printf("Total tasks completed: %d\n", pool->total_tasks_completed);
    printf("Total tasks rejected: %d\n", pool->total_tasks_rejected);
    
    if (pool->total_tasks_completed > 0) {
        printf("Average execution time: %.3f seconds\n", 
               pool->total_execution_time / pool->total_tasks_completed);
        printf("Average wait time: %.3f seconds\n", 
               pool->total_wait_time / pool->total_tasks_completed);
    }
    
    double success_rate = pool->total_tasks_submitted > 0 ? 
        (double)pool->total_tasks_completed / pool->total_tasks_submitted * 100 : 0;
    printf("Success rate: %.1f%%\n", success_rate);
    
    pthread_mutex_unlock(&pool->queue_mutex);
}

// Shutdown thread pool
void thread_pool_shutdown(thread_pool_t* pool) {
    pthread_mutex_lock(&pool->queue_mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->work_available);
    pthread_mutex_unlock(&pool->queue_mutex);
    
    // Wait for all threads to finish
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
        printf("Joined worker thread %d\n", i);
    }
    
    // Free remaining work items
    pthread_mutex_lock(&pool->queue_mutex);
    work_item_t* current = pool->work_queue_head;
    while (current) {
        work_item_t* next = current->next;
        free(current);
        current = next;
    }
    pthread_mutex_unlock(&pool->queue_mutex);
    
    // Destroy synchronization primitives
    pthread_cond_destroy(&pool->queue_not_full);
    pthread_cond_destroy(&pool->work_available);
    pthread_mutex_destroy(&pool->queue_mutex);
    
    printf("Thread pool shutdown complete\n");
}
```

## Advanced Synchronization Patterns

### Monitor Pattern Implementation

The monitor pattern encapsulates both data and synchronization, providing a clean abstraction for thread-safe operations:

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Account monitor for thread-safe banking operations
typedef struct {
    double balance;
    char account_number[32];
    char owner_name[64];
    
    pthread_mutex_t mutex;
    pthread_cond_t balance_changed;
    pthread_cond_t sufficient_funds;
    
    // Transaction history
    struct transaction* transaction_history;
    int transaction_count;
    
    // Monitor statistics
    int deposit_count;
    int withdrawal_count;
    int failed_withdrawal_count;
    double total_deposited;
    double total_withdrawn;
} account_monitor_t;

typedef struct transaction {
    double amount;
    int transaction_type; // 0 = deposit, 1 = withdrawal
    struct timespec timestamp;
    double balance_after;
    struct transaction* next;
} transaction_t;

// Initialize account monitor
int account_monitor_init(account_monitor_t* account, const char* account_number, 
                        const char* owner_name, double initial_balance) {
    account->balance = initial_balance;
    strncpy(account->account_number, account_number, sizeof(account->account_number) - 1);
    strncpy(account->owner_name, owner_name, sizeof(account->owner_name) - 1);
    account->transaction_history = NULL;
    account->transaction_count = 0;
    account->deposit_count = 0;
    account->withdrawal_count = 0;
    account->failed_withdrawal_count = 0;
    account->total_deposited = 0.0;
    account->total_withdrawn = 0.0;
    
    if (pthread_mutex_init(&account->mutex, NULL) != 0) {
        return -1;
    }
    
    if (pthread_cond_init(&account->balance_changed, NULL) != 0) {
        pthread_mutex_destroy(&account->mutex);
        return -1;
    }
    
    if (pthread_cond_init(&account->sufficient_funds, NULL) != 0) {
        pthread_cond_destroy(&account->balance_changed);
        pthread_mutex_destroy(&account->mutex);
        return -1;
    }
    
    printf("Account monitor initialized: %s (%s) with balance $%.2f\n", 
           account_number, owner_name, initial_balance);
    return 0;
}

// Add transaction to history
void add_transaction(account_monitor_t* account, double amount, int type, double new_balance) {
    transaction_t* trans = malloc(sizeof(transaction_t));
    if (trans) {
        trans->amount = amount;
        trans->transaction_type = type;
        clock_gettime(CLOCK_REALTIME, &trans->timestamp);
        trans->balance_after = new_balance;
        trans->next = account->transaction_history;
        account->transaction_history = trans;
        account->transaction_count++;
    }
}

// Monitor method: deposit money
int account_deposit(account_monitor_t* account, double amount, int client_id) {
    if (amount <= 0) {
        printf("Client %d: Invalid deposit amount: $%.2f\n", client_id, amount);
        return -1;
    }
    
    pthread_mutex_lock(&account->mutex);
    
    double old_balance = account->balance;
    account->balance += amount;
    account->deposit_count++;
    account->total_deposited += amount;
    
    add_transaction(account, amount, 0, account->balance);
    
    printf("Client %d: Deposited $%.2f to %s. Balance: $%.2f -> $%.2f\n",
           client_id, amount, account->account_number, old_balance, account->balance);
    
    // Notify waiting withdrawals
    pthread_cond_broadcast(&account->sufficient_funds);
    pthread_cond_signal(&account->balance_changed);
    
    pthread_mutex_unlock(&account->mutex);
    return 0;
}

// Monitor method: withdraw money
int account_withdraw(account_monitor_t* account, double amount, int client_id) {
    if (amount <= 0) {
        printf("Client %d: Invalid withdrawal amount: $%.2f\n", client_id, amount);
        return -1;
    }
    
    pthread_mutex_lock(&account->mutex);
    
    // Wait for sufficient funds
    while (account->balance < amount) {
        printf("Client %d: Insufficient funds for $%.2f withdrawal (balance: $%.2f). Waiting...\n",
               client_id, amount, account->balance);
        
        pthread_cond_wait(&account->sufficient_funds, &account->mutex);
        
        printf("Client %d: Woke up, rechecking balance for $%.2f withdrawal...\n",
               client_id, amount);
    }
    
    double old_balance = account->balance;
    account->balance -= amount;
    account->withdrawal_count++;
    account->total_withdrawn += amount;
    
    add_transaction(account, amount, 1, account->balance);
    
    printf("Client %d: Withdrew $%.2f from %s. Balance: $%.2f -> $%.2f\n",
           client_id, amount, account->account_number, old_balance, account->balance);
    
    pthread_cond_signal(&account->balance_changed);
    
    pthread_mutex_unlock(&account->mutex);
    return 0;
}

// Monitor method: try withdrawal (non-blocking)
int account_try_withdraw(account_monitor_t* account, double amount, int client_id) {
    if (amount <= 0) {
        return -1;
    }
    
    pthread_mutex_lock(&account->mutex);
    
    if (account->balance >= amount) {
        double old_balance = account->balance;
        account->balance -= amount;
        account->withdrawal_count++;
        account->total_withdrawn += amount;
        
        add_transaction(account, amount, 1, account->balance);
        
        printf("Client %d: Successfully withdrew $%.2f from %s. Balance: $%.2f -> $%.2f\n",
               client_id, amount, account->account_number, old_balance, account->balance);
        
        pthread_cond_signal(&account->balance_changed);
        pthread_mutex_unlock(&account->mutex);
        return 0;
    } else {
        account->failed_withdrawal_count++;
        printf("Client %d: Failed to withdraw $%.2f from %s (insufficient funds: $%.2f)\n",
               client_id, amount, account->account_number, account->balance);
        
        pthread_mutex_unlock(&account->mutex);
        return 1; // Insufficient funds
    }
}

// Monitor method: wait for balance to reach target
int account_wait_for_balance(account_monitor_t* account, double target_balance, int client_id) {
    pthread_mutex_lock(&account->mutex);
    
    while (account->balance < target_balance) {
        printf("Client %d: Waiting for balance to reach $%.2f (current: $%.2f)\n",
               client_id, target_balance, account->balance);
        
        pthread_cond_wait(&account->balance_changed, &account->mutex);
    }
    
    printf("Client %d: Target balance $%.2f reached! (current: $%.2f)\n",
           client_id, target_balance, account->balance);
    
    pthread_mutex_unlock(&account->mutex);
    return 0;
}

// Monitor method: get account statement
void account_get_statement(account_monitor_t* account) {
    pthread_mutex_lock(&account->mutex);
    
    printf("\n=== Account Statement for %s ===\n", account->account_number);
    printf("Account holder: %s\n", account->owner_name);
    printf("Current balance: $%.2f\n", account->balance);
    printf("Total transactions: %d\n", account->transaction_count);
    printf("Deposits: %d (total: $%.2f)\n", account->deposit_count, account->total_deposited);
    printf("Withdrawals: %d (total: $%.2f)\n", account->withdrawal_count, account->total_withdrawn);
    printf("Failed withdrawals: %d\n", account->failed_withdrawal_count);
    
    if (account->transaction_history) {
        printf("\nRecent transactions:\n");
        transaction_t* trans = account->transaction_history;
        int count = 0;
        while (trans && count < 10) { // Show last 10 transactions
            const char* type = (trans->transaction_type == 0) ? "DEPOSIT" : "WITHDRAWAL";
            printf("  %s: $%.2f (Balance after: $%.2f)\n", 
                   type, trans->amount, trans->balance_after);
            trans = trans->next;
            count++;
        }
    }
    
    pthread_mutex_unlock(&account->mutex);
}

// Cleanup account monitor
void account_monitor_destroy(account_monitor_t* account) {
    pthread_mutex_lock(&account->mutex);
    
    // Free transaction history
    transaction_t* current = account->transaction_history;
    while (current) {
        transaction_t* next = current->next;
        free(current);
        current = next;
    }
    
    pthread_mutex_unlock(&account->mutex);
    
    pthread_cond_destroy(&account->sufficient_funds);
    pthread_cond_destroy(&account->balance_changed);
    pthread_mutex_destroy(&account->mutex);
}
```

### Reader-Writer Lock Implementation Using Condition Variables

A custom reader-writer lock with writer preference to prevent writer starvation:

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int active_readers;
    int active_writers;
    int waiting_readers;
    int waiting_writers;
    
    pthread_mutex_t mutex;
    pthread_cond_t readers_ok;
    pthread_cond_t writers_ok;
    
    // Statistics
    int total_read_locks;
    int total_write_locks;
    double total_read_wait_time;
    double total_write_wait_time;
    int max_concurrent_readers;
} custom_rwlock_t;

// Initialize custom reader-writer lock
int custom_rwlock_init(custom_rwlock_t* rwlock) {
    rwlock->active_readers = 0;
    rwlock->active_writers = 0;
    rwlock->waiting_readers = 0;
    rwlock->waiting_writers = 0;
    rwlock->total_read_locks = 0;
    rwlock->total_write_locks = 0;
    rwlock->total_read_wait_time = 0.0;
    rwlock->total_write_wait_time = 0.0;
    rwlock->max_concurrent_readers = 0;
    
    if (pthread_mutex_init(&rwlock->mutex, NULL) != 0) {
        return -1;
    }
    
    if (pthread_cond_init(&rwlock->readers_ok, NULL) != 0) {
        pthread_mutex_destroy(&rwlock->mutex);
        return -1;
    }
    
    if (pthread_cond_init(&rwlock->writers_ok, NULL) != 0) {
        pthread_cond_destroy(&rwlock->readers_ok);
        pthread_mutex_destroy(&rwlock->mutex);
        return -1;
    }
    
    printf("Custom reader-writer lock initialized\n");
    return 0;
}

// Acquire read lock
int custom_rwlock_rdlock(custom_rwlock_t* rwlock, int reader_id) {
    struct timespec start_time, acquire_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    pthread_mutex_lock(&rwlock->mutex);
    
    rwlock->waiting_readers++;
    
    // Wait while there are active writers or waiting writers (writer preference)
    while (rwlock->active_writers > 0 || rwlock->waiting_writers > 0) {
        printf("Reader %d: Waiting (active_writers=%d, waiting_writers=%d)\n",
               reader_id, rwlock->active_writers, rwlock->waiting_writers);
        
        pthread_cond_wait(&rwlock->readers_ok, &rwlock->mutex);
    }
    
    rwlock->waiting_readers--;
    rwlock->active_readers++;
    rwlock->total_read_locks++;
    
    if (rwlock->active_readers > rwlock->max_concurrent_readers) {
        rwlock->max_concurrent_readers = rwlock->active_readers;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &acquire_time);
    double wait_time = (acquire_time.tv_sec - start_time.tv_sec) +
                       (acquire_time.tv_nsec - start_time.tv_nsec) / 1e9;
    rwlock->total_read_wait_time += wait_time;
    
    printf("Reader %d: Acquired read lock (wait: %.3fs, active readers: %d)\n",
           reader_id, wait_time, rwlock->active_readers);
    
    pthread_mutex_unlock(&rwlock->mutex);
    return 0;
}

// Release read lock
int custom_rwlock_rdunlock(custom_rwlock_t* rwlock, int reader_id) {
    pthread_mutex_lock(&rwlock->mutex);
    
    rwlock->active_readers--;
    
    printf("Reader %d: Released read lock (remaining readers: %d)\n",
           reader_id, rwlock->active_readers);
    
    // If no more readers and there are waiting writers, wake one
    if (rwlock->active_readers == 0 && rwlock->waiting_writers > 0) {
        pthread_cond_signal(&rwlock->writers_ok);
    }
    
    pthread_mutex_unlock(&rwlock->mutex);
    return 0;
}

// Acquire write lock
int custom_rwlock_wrlock(custom_rwlock_t* rwlock, int writer_id) {
    struct timespec start_time, acquire_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    pthread_mutex_lock(&rwlock->mutex);
    
    rwlock->waiting_writers++;
    
    // Wait while there are active readers or writers
    while (rwlock->active_readers > 0 || rwlock->active_writers > 0) {
        printf("Writer %d: Waiting (active_readers=%d, active_writers=%d)\n",
               writer_id, rwlock->active_readers, rwlock->active_writers);
        
        pthread_cond_wait(&rwlock->writers_ok, &rwlock->mutex);
    }
    
    rwlock->waiting_writers--;
    rwlock->active_writers++;
    rwlock->total_write_locks++;
    
    clock_gettime(CLOCK_MONOTONIC, &acquire_time);
    double wait_time = (acquire_time.tv_sec - start_time.tv_sec) +
                       (acquire_time.tv_nsec - start_time.tv_nsec) / 1e9;
    rwlock->total_write_wait_time += wait_time;
    
    printf("Writer %d: Acquired write lock (wait: %.3fs)\n", writer_id, wait_time);
    
    pthread_mutex_unlock(&rwlock->mutex);
    return 0;
}

// Release write lock
int custom_rwlock_wrunlock(custom_rwlock_t* rwlock, int writer_id) {
    pthread_mutex_lock(&rwlock->mutex);
    
    rwlock->active_writers--;
    
    printf("Writer %d: Released write lock\n", writer_id);
    
    // Prioritize waiting writers (writer preference)
    if (rwlock->waiting_writers > 0) {
        pthread_cond_signal(&rwlock->writers_ok);
    } else if (rwlock->waiting_readers > 0) {
        // Wake all waiting readers
        pthread_cond_broadcast(&rwlock->readers_ok);
    }
    
    pthread_mutex_unlock(&rwlock->mutex);
    return 0;
}

// Get lock statistics
void custom_rwlock_get_stats(custom_rwlock_t* rwlock) {
    pthread_mutex_lock(&rwlock->mutex);
    
    printf("\n=== Reader-Writer Lock Statistics ===\n");
    printf("Currently active readers: %d\n", rwlock->active_readers);
    printf("Currently active writers: %d\n", rwlock->active_writers);
    printf("Currently waiting readers: %d\n", rwlock->waiting_readers);
    printf("Currently waiting writers: %d\n", rwlock->waiting_writers);
    printf("Total read locks acquired: %d\n", rwlock->total_read_locks);
    printf("Total write locks acquired: %d\n", rwlock->total_write_locks);
    printf("Max concurrent readers: %d\n", rwlock->max_concurrent_readers);
    
    if (rwlock->total_read_locks > 0) {
        printf("Average read lock wait time: %.3f seconds\n",
               rwlock->total_read_wait_time / rwlock->total_read_locks);
    }
    
    if (rwlock->total_write_locks > 0) {
        printf("Average write lock wait time: %.3f seconds\n",
               rwlock->total_write_wait_time / rwlock->total_write_locks);
    }
    
    double read_ratio = (rwlock->total_read_locks + rwlock->total_write_locks) > 0 ?
        (double)rwlock->total_read_locks / (rwlock->total_read_locks + rwlock->total_write_locks) * 100 : 0;
    printf("Read/Write ratio: %.1f%% reads, %.1f%% writes\n", read_ratio, 100 - read_ratio);
    
    pthread_mutex_unlock(&rwlock->mutex);
}

// Cleanup custom reader-writer lock
void custom_rwlock_destroy(custom_rwlock_t* rwlock) {
    custom_rwlock_get_stats(rwlock);
    
    pthread_cond_destroy(&rwlock->writers_ok);
    pthread_cond_destroy(&rwlock->readers_ok);
    pthread_mutex_destroy(&rwlock->mutex);
    
    printf("Custom reader-writer lock destroyed\n");
}
```

### Barrier Implementation with Condition Variables

A reusable barrier that supports multiple synchronization points:

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int total_threads;
    int arrived_threads;
    int generation;
    
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    
    // Statistics
    int total_synchronizations;
    int max_wait_time_ms;
    double avg_wait_time;
} reusable_barrier_t;

// Initialize reusable barrier
int reusable_barrier_init(reusable_barrier_t* barrier, int thread_count) {
    if (thread_count <= 0) {
        return -1;
    }
    
    barrier->total_threads = thread_count;
    barrier->arrived_threads = 0;
    barrier->generation = 0;
    barrier->total_synchronizations = 0;
    barrier->max_wait_time_ms = 0;
    barrier->avg_wait_time = 0.0;
    
    if (pthread_mutex_init(&barrier->mutex, NULL) != 0) {
        return -1;
    }
    
    if (pthread_cond_init(&barrier->condition, NULL) != 0) {
        pthread_mutex_destroy(&barrier->mutex);
        return -1;
    }
    
    printf("Reusable barrier initialized for %d threads\n", thread_count);
    return 0;
}

// Wait at barrier
int reusable_barrier_wait(reusable_barrier_t* barrier, int thread_id) {
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    pthread_mutex_lock(&barrier->mutex);
    
    int my_generation = barrier->generation;
    barrier->arrived_threads++;
    
    printf("Thread %d: Arrived at barrier (generation %d, %d/%d threads)\n",
           thread_id, my_generation, barrier->arrived_threads, barrier->total_threads);
    
    if (barrier->arrived_threads == barrier->total_threads) {
        // Last thread to arrive
        printf("Thread %d: Last thread arrived, releasing all threads\n", thread_id);
        
        barrier->arrived_threads = 0;
        barrier->generation++;
        barrier->total_synchronizations++;
        
        pthread_cond_broadcast(&barrier->condition);
        
        pthread_mutex_unlock(&barrier->mutex);
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        int wait_time_ms = ((end_time.tv_sec - start_time.tv_sec) * 1000) +
                          ((end_time.tv_nsec - start_time.tv_nsec) / 1000000);
        
        printf("Thread %d: Released from barrier after %d ms (serial thread)\n",
               thread_id, wait_time_ms);
        
        return PTHREAD_BARRIER_SERIAL_THREAD; // Special return value
    } else {
        // Wait for other threads
        while (barrier->generation == my_generation) {
            pthread_cond_wait(&barrier->condition, &barrier->mutex);
        }
        
        pthread_mutex_unlock(&barrier->mutex);
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        int wait_time_ms = ((end_time.tv_sec - start_time.tv_sec) * 1000) +
                          ((end_time.tv_nsec - start_time.tv_nsec) / 1000000);
        
        // Update statistics
        pthread_mutex_lock(&barrier->mutex);
        if (wait_time_ms > barrier->max_wait_time_ms) {
            barrier->max_wait_time_ms = wait_time_ms;
        }
        barrier->avg_wait_time = (barrier->avg_wait_time * (barrier->total_threads - 1) + wait_time_ms) / barrier->total_threads;
        pthread_mutex_unlock(&barrier->mutex);
        
        printf("Thread %d: Released from barrier after %d ms\n", thread_id, wait_time_ms);
        
        return 0;
    }
}

// Get barrier statistics
void reusable_barrier_get_stats(reusable_barrier_t* barrier) {
    pthread_mutex_lock(&barrier->mutex);
    
    printf("\n=== Barrier Statistics ===\n");
    printf("Thread count: %d\n", barrier->total_threads);
    printf("Current generation: %d\n", barrier->generation);
    printf("Currently arrived threads: %d\n", barrier->arrived_threads);
    printf("Total synchronizations: %d\n", barrier->total_synchronizations);
    printf("Max wait time: %d ms\n", barrier->max_wait_time_ms);
    printf("Average wait time: %.1f ms\n", barrier->avg_wait_time);
    
    pthread_mutex_unlock(&barrier->mutex);
}

// Cleanup reusable barrier
void reusable_barrier_destroy(reusable_barrier_t* barrier) {
    reusable_barrier_get_stats(barrier);
    
    pthread_cond_destroy(&barrier->condition);
    pthread_mutex_destroy(&barrier->mutex);
    
    printf("Reusable barrier destroyed\n");
}
```

## Performance Analysis and Optimization

### Condition Variable Performance Characteristics

Understanding the performance implications of different condition variable usage patterns:

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// Performance measurement framework
typedef struct {
    const char* test_name;
    int iterations;
    double total_time;
    double avg_time_per_operation;
    int signal_count;
    int broadcast_count;
    int wait_count;
} perf_result_t;

// Benchmark signal vs broadcast performance
void benchmark_signal_vs_broadcast() {
    const int ITERATIONS = 10000;
    const int NUM_WAITERS = 5;
    
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
    volatile int wake_count = 0;
    volatile int test_complete = 0;
    
    printf("\n=== Signal vs Broadcast Performance Test ===\n");
    
    // Test 1: pthread_cond_signal performance
    {
        wake_count = 0;
        test_complete = 0;
        
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        for (int i = 0; i < ITERATIONS; i++) {
            pthread_mutex_lock(&mutex);
            wake_count++;
            pthread_cond_signal(&condition);
            pthread_mutex_unlock(&mutex);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double signal_time = (end_time.tv_sec - start_time.tv_sec) +
                            (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        
        printf("Signal test: %d operations in %.3f seconds (%.1f ns/operation)\n",
               ITERATIONS, signal_time, (signal_time * 1e9) / ITERATIONS);
    }
    
    // Test 2: pthread_cond_broadcast performance
    {
        wake_count = 0;
        test_complete = 0;
        
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        for (int i = 0; i < ITERATIONS; i++) {
            pthread_mutex_lock(&mutex);
            wake_count++;
            pthread_cond_broadcast(&condition);
            pthread_mutex_unlock(&mutex);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double broadcast_time = (end_time.tv_sec - start_time.tv_sec) +
                               (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        
        printf("Broadcast test: %d operations in %.3f seconds (%.1f ns/operation)\n",
               ITERATIONS, broadcast_time, (broadcast_time * 1e9) / ITERATIONS);
    }
    
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&condition);
}

// Measure condition variable wait latency
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int ready;
    struct timespec signal_time;
    struct timespec wait_start_time;
    double latency;
} latency_test_t;

void* latency_waiter(void* arg) {
    latency_test_t* test = (latency_test_t*)arg;
    
    pthread_mutex_lock(&test->mutex);
    
    clock_gettime(CLOCK_MONOTONIC, &test->wait_start_time);
    
    while (!test->ready) {
        pthread_cond_wait(&test->condition, &test->mutex);
    }
    
    struct timespec wake_time;
    clock_gettime(CLOCK_MONOTONIC, &wake_time);
    
    // Calculate latency from signal to wake
    test->latency = (wake_time.tv_sec - test->signal_time.tv_sec) +
                   (wake_time.tv_nsec - test->signal_time.tv_nsec) / 1e9;
    
    pthread_mutex_unlock(&test->mutex);
    return NULL;
}

void* latency_signaler(void* arg) {
    latency_test_t* test = (latency_test_t*)arg;
    
    usleep(10000); // Give waiter time to start waiting
    
    pthread_mutex_lock(&test->mutex);
    test->ready = 1;
    clock_gettime(CLOCK_MONOTONIC, &test->signal_time);
    pthread_cond_signal(&test->condition);
    pthread_mutex_unlock(&test->mutex);
    
    return NULL;
}

void measure_condition_variable_latency() {
    const int NUM_TESTS = 1000;
    double total_latency = 0.0;
    double min_latency = 1e9;
    double max_latency = 0.0;
    
    printf("\n=== Condition Variable Latency Test ===\n");
    
    for (int i = 0; i < NUM_TESTS; i++) {
        latency_test_t test = {0};
        test.ready = 0;
        
        pthread_mutex_init(&test.mutex, NULL);
        pthread_cond_init(&test.condition, NULL);
        
        pthread_t waiter, signaler;
        pthread_create(&waiter, NULL, latency_waiter, &test);
        pthread_create(&signaler, NULL, latency_signaler, &test);
        
        pthread_join(waiter, NULL);
        pthread_join(signaler, NULL);
        
        total_latency += test.latency;
        if (test.latency < min_latency) min_latency = test.latency;
        if (test.latency > max_latency) max_latency = test.latency;
        
        pthread_cond_destroy(&test.condition);
        pthread_mutex_destroy(&test.mutex);
    }
    
    printf("Latency statistics over %d tests:\n", NUM_TESTS);
    printf("  Average: %.1f microseconds\n", (total_latency / NUM_TESTS) * 1e6);
    printf("  Minimum: %.1f microseconds\n", min_latency * 1e6);
    printf("  Maximum: %.1f microseconds\n", max_latency * 1e6);
}

// Test condition variable scalability with varying thread counts
void test_condition_variable_scalability() {
    const int thread_counts[] = {2, 4, 8, 16, 32};
    const int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    const int OPERATIONS_PER_THREAD = 1000;
    
    printf("\n=== Condition Variable Scalability Test ===\n");
    printf("Threads\tTotal Time\tOps/Second\tAvg Latency\n");
    printf("-------\t----------\t----------\t-----------\n");
    
    for (int test = 0; test < num_tests; test++) {
        int num_threads = thread_counts[test];
        pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
        
        pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
        pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
        volatile int counter = 0;
        volatile int active_threads = num_threads;
        
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        // Create threads
        for (int i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], NULL, scalability_test_thread, 
                          &(struct {
                              pthread_mutex_t* mutex;
                              pthread_cond_t* condition;
                              volatile int* counter;
                              volatile int* active_threads;
                              int operations;
                              int thread_id;
                          }){&mutex, &condition, &counter, &active_threads, 
                             OPERATIONS_PER_THREAD, i});
        }
        
        // Wait for all threads
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double total_time = (end_time.tv_sec - start_time.tv_sec) +
                           (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        
        int total_operations = num_threads * OPERATIONS_PER_THREAD;
        double ops_per_second = total_operations / total_time;
        double avg_latency = (total_time / total_operations) * 1e6; // microseconds
        
        printf("%d\t%.3f s\t%.0f\t\t%.1f μs\n", 
               num_threads, total_time, ops_per_second, avg_latency);
        
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&condition);
        free(threads);
    }
}

void* scalability_test_thread(void* arg) {
    struct {
        pthread_mutex_t* mutex;
        pthread_cond_t* condition;
        volatile int* counter;
        volatile int* active_threads;
        int operations;
        int thread_id;
    } *data = arg;
    
    for (int i = 0; i < data->operations; i++) {
        pthread_mutex_lock(data->mutex);
        
        (*data->counter)++;
        
        if (*data->counter % (*data->active_threads) == 0) {
            pthread_cond_broadcast(data->condition);
        } else {
            while (*data->counter % (*data->active_threads) != 0) {
                pthread_cond_wait(data->condition, data->mutex);
            }
        }
        
        pthread_mutex_unlock(data->mutex);
    }
    
    return NULL;
}
```

## Common Mistakes and Best Practices

### Critical Mistake #1: Not Using While Loops

```c
// WRONG: Using if statement
pthread_mutex_lock(&mutex);
if (!condition_met) {
    pthread_cond_wait(&condition, &mutex);
}
// Process data - BUG: condition might not be true due to spurious wakeup
process_data();
pthread_mutex_unlock(&mutex);

// CORRECT: Using while loop
pthread_mutex_lock(&mutex);
while (!condition_met) {
    pthread_cond_wait(&condition, &mutex);
}
// Process data - SAFE: condition is guaranteed to be true
process_data();
pthread_mutex_unlock(&mutex);
```

### Critical Mistake #2: Signaling Without Holding Mutex

```c
// WRONG: Signaling without mutex protection
condition_met = 1;
pthread_cond_signal(&condition); // Race condition possible

// CORRECT: Signal while holding mutex
pthread_mutex_lock(&mutex);
condition_met = 1;
pthread_cond_signal(&condition);
pthread_mutex_unlock(&mutex);
```

### Critical Mistake #3: Using Wrong Signal Type

```c
// WRONG: Using signal when multiple threads should wake
pthread_mutex_lock(&mutex);
all_threads_should_proceed = 1;
pthread_cond_signal(&condition); // Only one thread wakes up
pthread_mutex_unlock(&mutex);

// CORRECT: Using broadcast for multiple threads
pthread_mutex_lock(&mutex);
all_threads_should_proceed = 1;
pthread_cond_broadcast(&condition); // All waiting threads wake up
pthread_mutex_unlock(&mutex);
```

### Advanced Best Practices

#### Predicate Functions for Complex Conditions

```c
// Use predicate functions for complex conditions
typedef struct {
    int* buffer;
    int size;
    int capacity;
    int readers;
    int writers;
} shared_resource_t;

// Predicate functions make conditions clear
int can_read(shared_resource_t* resource) {
    return resource->size > 0 && resource->writers == 0;
}

int can_write(shared_resource_t* resource) {
    return resource->size < resource->capacity && 
           resource->readers == 0 && resource->writers == 0;
}

// Clean waiting pattern
void wait_for_read_access(shared_resource_t* resource, pthread_cond_t* cond, pthread_mutex_t* mutex) {
    while (!can_read(resource)) {
        pthread_cond_wait(cond, mutex);
    }
    resource->readers++;
}

void wait_for_write_access(shared_resource_t* resource, pthread_cond_t* cond, pthread_mutex_t* mutex) {
    while (!can_write(resource)) {
        pthread_cond_wait(cond, mutex);
    }
    resource->writers++;
}
```

#### Timeout Handling Best Practices

```c
// Robust timeout handling with retry logic
int wait_with_backoff(pthread_cond_t* cond, pthread_mutex_t* mutex, 
                     int* condition, int max_attempts) {
    int attempt = 0;
    int timeout_ms = 100; // Start with 100ms timeout
    
    while (attempt < max_attempts && !(*condition)) {
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += timeout_ms / 1000;
        timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
        
        int result = pthread_cond_timedwait(cond, mutex, &timeout);
        
        if (result == ETIMEDOUT) {
            attempt++;
            timeout_ms = (timeout_ms < 5000) ? timeout_ms * 2 : 5000; // Exponential backoff, max 5s
            printf("Timeout attempt %d/%d, next timeout: %d ms\n", 
                   attempt, max_attempts, timeout_ms);
        } else if (result != 0) {
            fprintf(stderr, "Condition wait error: %s\n", strerror(result));
            return -1;
        }
    }
    
    return (*condition) ? 0 : -1; // 0 = success, -1 = failed after max attempts
}
```

## Comprehensive Exercises

### Exercise 1: Multi-Level Cache System (Advanced)

**Objective**: Implement a multi-level cache system with condition variables for cache coherency.

**Requirements**:
1. Implement L1 (thread-local) and L2 (shared) cache levels
2. Use condition variables for cache invalidation notifications
3. Implement write-through and write-back policies
4. Handle cache eviction with proper synchronization
5. Provide cache statistics and hit rate monitoring
6. Support concurrent readers and exclusive writers

**Starter Framework**:
```c
typedef struct cache_entry {
    char key[64];
    void* data;
    size_t data_size;
    int dirty;
    int access_count;
    struct timespec last_access;
    struct cache_entry* next;
} cache_entry_t;

typedef struct {
    cache_entry_t** buckets;
    int bucket_count;
    int max_entries;
    int current_entries;
    
    pthread_mutex_t mutex;
    pthread_cond_t space_available;
    pthread_cond_t data_available;
    
    // Statistics
    int hits;
    int misses;
    int evictions;
} cache_level_t;

// Functions to implement:
// - cache_init()
// - cache_get()
// - cache_put()
// - cache_invalidate()
// - cache_flush()
// - cache_get_stats()
```

### Exercise 2: Event-Driven Task Scheduler (Expert)

**Objective**: Build an event-driven task scheduler using condition variables for task coordination.

**Requirements**:
1. Support different task types: periodic, one-time, conditional
2. Implement priority-based scheduling
3. Use condition variables for event notifications
4. Support task dependencies and chains
5. Handle task failures and retries
6. Provide real-time monitoring and statistics

### Exercise 3: Producer-Consumer with Backpressure (Intermediate)

**Objective**: Implement a producer-consumer system with backpressure control.

**Requirements**:
1. Dynamic buffer size adjustment based on consumer throughput
2. Producer throttling when consumers are overwhelmed
3. Multiple priority levels with fairness guarantees
4. Graceful degradation under high load
5. Comprehensive metrics and monitoring

### Exercise 4: Readers-Writers with Fairness (Advanced)

**Objective**: Implement various readers-writers solutions with different fairness policies.

**Requirements**:
1. Reader preference, writer preference, and fair scheduling
2. Support for upgradable locks (read to write)
3. Timeout support for all operations
4. Deadlock detection and prevention
5. Performance comparison between different policies

### Exercise 5: Condition Variable-Based State Machine (Expert)

**Objective**: Implement a complex state machine using condition variables for state transitions.

**Requirements**:
1. Multiple states with complex transition rules
2. Event-driven state changes
3. State transition history and rollback capability
4. Multiple threads can trigger state changes
5. Timeout-based automatic transitions
6. Comprehensive logging and debugging support

## Assessment and Self-Evaluation

### Knowledge Levels

**Level 1: Fundamental Understanding**
- [ ] Explain the purpose and mechanics of condition variables
- [ ] Understand why mutexes are always used with condition variables
- [ ] Implement basic producer-consumer pattern
- [ ] Handle spurious wakeups correctly
- [ ] Choose between signal and broadcast appropriately

**Level 2: Practical Application**
- [ ] Implement thread-safe data structures using condition variables
- [ ] Handle timeouts and error conditions gracefully
- [ ] Design complex synchronization patterns
- [ ] Optimize condition variable usage for performance
- [ ] Debug condition variable-related issues

**Level 3: Advanced Mastery**
- [ ] Implement custom synchronization primitives
- [ ] Design scalable concurrent systems
- [ ] Analyze and optimize performance characteristics
- [ ] Handle edge cases and error recovery
- [ ] Mentor others in condition variable usage

### Performance Benchmarking Tasks

1. **Latency Measurement**
   - Measure signal-to-wake latency under different loads
   - Compare with other synchronization mechanisms
   - Identify performance bottlenecks

2. **Scalability Testing**
   - Test with varying numbers of waiting threads
   - Measure throughput degradation
   - Identify optimal configuration parameters

3. **Fairness Analysis**
   - Measure wait time distribution
   - Test for thread starvation scenarios
   - Evaluate different signaling strategies

### Code Quality Checklist

**Correctness**
- [ ] Always use while loops with condition variables
- [ ] Proper mutex usage with condition variables
- [ ] Correct signal vs broadcast usage
- [ ] Handle all error conditions
- [ ] No race conditions or deadlocks

**Performance**
- [ ] Minimize critical section length
- [ ] Use appropriate signaling strategy
- [ ] Avoid unnecessary broadcasts
- [ ] Efficient condition checking
- [ ] Proper resource cleanup

**Maintainability**
- [ ] Clear and descriptive variable names
- [ ] Well-documented synchronization logic
- [ ] Modular and testable design
- [ ] Comprehensive error handling
- [ ] Consistent coding style

### Common Interview Questions

1. **Explain spurious wakeups and why they occur**
2. **When would you use broadcast instead of signal?**
3. **How do condition variables differ from semaphores?**
4. **Describe the lost wakeup problem and how to avoid it**
5. **What are the performance implications of using condition variables?**
6. **How would you debug a condition variable deadlock?**
7. **Explain the relationship between condition variables and monitors**

## Next Steps and Advanced Topics

### Recommended Learning Path

1. **Advanced Synchronization Patterns**
   - Read-Copy-Update (RCU) mechanisms
   - Lock-free programming with atomic operations
   - Software transactional memory
   - Hazard pointers for memory management

2. **System-Level Programming**
   - Kernel synchronization primitives
   - Futex-based implementations
   - Real-time scheduling considerations
   - NUMA-aware synchronization

3. **Language-Specific Features**
   - C++11/14/17/20 condition variables
   - Java concurrent utilities
   - Go channels and select statements
   - Rust async/await and futures

### Tools for Advanced Development

**Debugging and Analysis**
- ThreadSanitizer (TSan) for race condition detection
- Helgrind for synchronization error detection
- Intel Inspector for threading analysis
- Custom logging frameworks for condition variable tracing

**Performance Profiling**
- perf for Linux performance analysis
- Intel VTune for detailed profiling
- Google Benchmark for microbenchmarking
- Custom metrics collection frameworks

**Testing Frameworks**
- Stress testing with high thread counts
- Property-based testing for concurrent code
- Model checking with TLA+ or SPIN
- Chaos engineering for robustness testing

This comprehensive guide provides the foundation for mastering condition variables and advancing to expert-level concurrent programming. The key to proficiency is consistent practice with increasingly complex scenarios and thorough understanding of the underlying principles.

## Next Section
[Thread Local Storage](05_Thread_Local_Storage.md)

## Next Section
[Thread Local Storage](05_Thread_Local_Storage.md)
