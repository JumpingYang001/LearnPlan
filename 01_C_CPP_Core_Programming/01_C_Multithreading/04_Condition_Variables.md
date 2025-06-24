# Condition Variables

*Duration: 1 week*

## Overview

Condition variables allow threads to wait for specific conditions to become true and signal other threads when conditions change. They are essential for implementing efficient producer-consumer patterns and thread coordination.

## Concept and Usage Patterns

### What are Condition Variables?
- Synchronization primitives that allow threads to wait for conditions
- Always used with a mutex for thread safety
- Provide efficient blocking and signaling mechanisms

### Basic Operations

```c
#include <pthread.h>

pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Or dynamic initialization
pthread_cond_t condition;
pthread_mutex_t mutex;
pthread_cond_init(&condition, NULL);
pthread_mutex_init(&mutex, NULL);
```

## Core Functions

### `pthread_cond_wait()`

```c
int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex);
```

**Behavior:**
1. Atomically unlocks mutex and waits on condition
2. When signaled, atomically locks mutex and returns
3. Always check condition in a loop (spurious wakeups)

### `pthread_cond_signal()`

```c
int pthread_cond_signal(pthread_cond_t *cond);
```

**Purpose:**
- Wakes up at least one waiting thread
- Use when only one thread needs to be notified

### `pthread_cond_broadcast()`

```c
int pthread_cond_broadcast(pthread_cond_t *cond);
```

**Purpose:**
- Wakes up all waiting threads
- Use when multiple threads might need to proceed

### Timed Wait

```c
#include <time.h>

struct timespec timeout;
clock_gettime(CLOCK_REALTIME, &timeout);
timeout.tv_sec += 5; // 5 second timeout

int result = pthread_cond_timedwait(&condition, &mutex, &timeout);
```

## Standard Usage Pattern

```c
// Waiting thread
pthread_mutex_lock(&mutex);
while (!condition_met) {
    pthread_cond_wait(&condition, &mutex);
}
// Condition is now true and mutex is locked
// Do work
pthread_mutex_unlock(&mutex);

// Signaling thread
pthread_mutex_lock(&mutex);
// Modify shared state
condition_met = 1;
pthread_cond_signal(&condition);
pthread_mutex_unlock(&mutex);
```

## Producer-Consumer Problems

### Simple Producer-Consumer

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define BUFFER_SIZE 10

typedef struct {
    int buffer[BUFFER_SIZE];
    int count;
    int in;
    int out;
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
} Buffer;

Buffer shared_buffer = {
    .count = 0,
    .in = 0,
    .out = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .not_full = PTHREAD_COND_INITIALIZER,
    .not_empty = PTHREAD_COND_INITIALIZER
};

void* producer(void* arg) {
    int producer_id = *(int*)arg;
    
    for (int i = 0; i < 20; i++) {
        int item = producer_id * 100 + i;
        
        pthread_mutex_lock(&shared_buffer.mutex);
        
        // Wait while buffer is full
        while (shared_buffer.count == BUFFER_SIZE) {
            printf("Producer %d: Buffer full, waiting...\n", producer_id);
            pthread_cond_wait(&shared_buffer.not_full, &shared_buffer.mutex);
        }
        
        // Produce item
        shared_buffer.buffer[shared_buffer.in] = item;
        shared_buffer.in = (shared_buffer.in + 1) % BUFFER_SIZE;
        shared_buffer.count++;
        
        printf("Producer %d: Produced item %d (count: %d)\n", 
               producer_id, item, shared_buffer.count);
        
        // Signal that buffer is not empty
        pthread_cond_signal(&shared_buffer.not_empty);
        
        pthread_mutex_unlock(&shared_buffer.mutex);
        
        usleep(100000); // Simulate work
    }
    
    return NULL;
}

void* consumer(void* arg) {
    int consumer_id = *(int*)arg;
    
    for (int i = 0; i < 20; i++) {
        pthread_mutex_lock(&shared_buffer.mutex);
        
        // Wait while buffer is empty
        while (shared_buffer.count == 0) {
            printf("Consumer %d: Buffer empty, waiting...\n", consumer_id);
            pthread_cond_wait(&shared_buffer.not_empty, &shared_buffer.mutex);
        }
        
        // Consume item
        int item = shared_buffer.buffer[shared_buffer.out];
        shared_buffer.out = (shared_buffer.out + 1) % BUFFER_SIZE;
        shared_buffer.count--;
        
        printf("Consumer %d: Consumed item %d (count: %d)\n", 
               consumer_id, item, shared_buffer.count);
        
        // Signal that buffer is not full
        pthread_cond_signal(&shared_buffer.not_full);
        
        pthread_mutex_unlock(&shared_buffer.mutex);
        
        usleep(150000); // Simulate work
    }
    
    return NULL;
}

int main() {
    pthread_t producers[2], consumers[2];
    int producer_ids[2] = {1, 2};
    int consumer_ids[2] = {1, 2};
    
    // Create producers and consumers
    for (int i = 0; i < 2; i++) {
        pthread_create(&producers[i], NULL, producer, &producer_ids[i]);
        pthread_create(&consumers[i], NULL, consumer, &consumer_ids[i]);
    }
    
    // Join all threads
    for (int i = 0; i < 2; i++) {
        pthread_join(producers[i], NULL);
        pthread_join(consumers[i], NULL);
    }
    
    printf("All threads completed\n");
    return 0;
}
```

## Implementing Thread-Safe Queues

### Thread-Safe Queue Structure

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct QueueNode {
    void* data;
    struct QueueNode* next;
} QueueNode;

typedef struct {
    QueueNode* head;
    QueueNode* tail;
    size_t size;
    size_t max_size;
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
} ThreadSafeQueue;

ThreadSafeQueue* queue_create(size_t max_size) {
    ThreadSafeQueue* queue = malloc(sizeof(ThreadSafeQueue));
    if (!queue) return NULL;
    
    queue->head = NULL;
    queue->tail = NULL;
    queue->size = 0;
    queue->max_size = max_size;
    
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->not_full, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    
    return queue;
}

int queue_enqueue(ThreadSafeQueue* queue, void* data) {
    pthread_mutex_lock(&queue->mutex);
    
    // Wait while queue is full
    while (queue->size >= queue->max_size) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    // Create new node
    QueueNode* new_node = malloc(sizeof(QueueNode));
    if (!new_node) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    
    new_node->data = data;
    new_node->next = NULL;
    
    // Add to queue
    if (queue->tail) {
        queue->tail->next = new_node;
    } else {
        queue->head = new_node;
    }
    queue->tail = new_node;
    queue->size++;
    
    // Signal that queue is not empty
    pthread_cond_signal(&queue->not_empty);
    
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

void* queue_dequeue(ThreadSafeQueue* queue) {
    pthread_mutex_lock(&queue->mutex);
    
    // Wait while queue is empty
    while (queue->size == 0) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }
    
    // Remove from queue
    QueueNode* node = queue->head;
    void* data = node->data;
    
    queue->head = node->next;
    if (!queue->head) {
        queue->tail = NULL;
    }
    queue->size--;
    
    free(node);
    
    // Signal that queue is not full
    pthread_cond_signal(&queue->not_full);
    
    pthread_mutex_unlock(&queue->mutex);
    return data;
}

int queue_try_dequeue(ThreadSafeQueue* queue, void** data) {
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->size == 0) {
        pthread_mutex_unlock(&queue->mutex);
        return 0; // Empty queue
    }
    
    // Remove from queue
    QueueNode* node = queue->head;
    *data = node->data;
    
    queue->head = node->next;
    if (!queue->head) {
        queue->tail = NULL;
    }
    queue->size--;
    
    free(node);
    
    // Signal that queue is not full
    pthread_cond_signal(&queue->not_full);
    
    pthread_mutex_unlock(&queue->mutex);
    return 1; // Success
}

void queue_destroy(ThreadSafeQueue* queue) {
    pthread_mutex_lock(&queue->mutex);
    
    // Free all remaining nodes
    while (queue->head) {
        QueueNode* temp = queue->head;
        queue->head = queue->head->next;
        free(temp);
    }
    
    pthread_mutex_unlock(&queue->mutex);
    
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_full);
    pthread_cond_destroy(&queue->not_empty);
    
    free(queue);
}
```

## Advanced Patterns

### Barrier Implementation with Condition Variables

```c
typedef struct {
    int count;
    int total;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
} SimpleBarrier;

void barrier_init(SimpleBarrier* barrier, int total) {
    barrier->count = 0;
    barrier->total = total;
    pthread_mutex_init(&barrier->mutex, NULL);
    pthread_cond_init(&barrier->condition, NULL);
}

void barrier_wait(SimpleBarrier* barrier) {
    pthread_mutex_lock(&barrier->mutex);
    
    barrier->count++;
    
    if (barrier->count == barrier->total) {
        // Last thread to arrive
        barrier->count = 0; // Reset for reuse
        pthread_cond_broadcast(&barrier->condition);
    } else {
        // Wait for other threads
        while (barrier->count != 0) {
            pthread_cond_wait(&barrier->condition, &barrier->mutex);
        }
    }
    
    pthread_mutex_unlock(&barrier->mutex);
}
```

### Read-Preference Reader-Writer Lock

```c
typedef struct {
    int readers;
    int writers;
    int waiting_writers;
    pthread_mutex_t mutex;
    pthread_cond_t read_cond;
    pthread_cond_t write_cond;
} RWLock;

void rwlock_read_lock(RWLock* lock) {
    pthread_mutex_lock(&lock->mutex);
    
    while (lock->writers > 0 || lock->waiting_writers > 0) {
        pthread_cond_wait(&lock->read_cond, &lock->mutex);
    }
    
    lock->readers++;
    pthread_mutex_unlock(&lock->mutex);
}

void rwlock_read_unlock(RWLock* lock) {
    pthread_mutex_lock(&lock->mutex);
    
    lock->readers--;
    if (lock->readers == 0) {
        pthread_cond_signal(&lock->write_cond);
    }
    
    pthread_mutex_unlock(&lock->mutex);
}

void rwlock_write_lock(RWLock* lock) {
    pthread_mutex_lock(&lock->mutex);
    
    lock->waiting_writers++;
    while (lock->readers > 0 || lock->writers > 0) {
        pthread_cond_wait(&lock->write_cond, &lock->mutex);
    }
    lock->waiting_writers--;
    lock->writers++;
    
    pthread_mutex_unlock(&lock->mutex);
}

void rwlock_write_unlock(RWLock* lock) {
    pthread_mutex_lock(&lock->mutex);
    
    lock->writers--;
    if (lock->waiting_writers > 0) {
        pthread_cond_signal(&lock->write_cond);
    } else {
        pthread_cond_broadcast(&lock->read_cond);
    }
    
    pthread_mutex_unlock(&lock->mutex);
}
```

## Common Mistakes and Best Practices

### Always Use Loops for Condition Checking

```c
// WRONG
pthread_mutex_lock(&mutex);
if (!condition) {
    pthread_cond_wait(&cond, &mutex);
}
// Process...
pthread_mutex_unlock(&mutex);

// CORRECT
pthread_mutex_lock(&mutex);
while (!condition) {
    pthread_cond_wait(&cond, &mutex);
}
// Process...
pthread_mutex_unlock(&mutex);
```

### Spurious Wakeups
- `pthread_cond_wait()` can return even when not signaled
- Always check the actual condition in a loop
- This is required by POSIX standard

### Signal vs Broadcast
- Use `pthread_cond_signal()` when only one thread should proceed
- Use `pthread_cond_broadcast()` when multiple threads might proceed
- Broadcasting is safer but potentially less efficient

## Exercises

1. **Dining Philosophers**
   - Implement the classic dining philosophers problem
   - Use condition variables to avoid deadlock

2. **Work Queue**
   - Create a thread pool with a work queue
   - Use condition variables for work distribution

3. **Reader-Writer Problem**
   - Implement various reader-writer solutions
   - Compare read-preference vs write-preference

4. **Bounded Buffer**
   - Implement multiple producer-consumer scenarios
   - Handle dynamic buffer sizing

## Performance Considerations

### Condition Variable Performance
- Avoid unnecessary broadcasts (use signal when appropriate)
- Minimize time holding mutex while checking conditions
- Consider using multiple condition variables for different conditions

### Alternative Approaches
- Compare with semaphores for simple counting scenarios
- Consider lock-free approaches for high-performance needs

## Assessment

You should be able to:
- Implement producer-consumer patterns correctly
- Create thread-safe data structures using condition variables
- Understand spurious wakeups and handle them properly
- Choose between signal and broadcast appropriately
- Debug condition variable-related issues
- Implement complex synchronization patterns

## Next Section
[Thread Local Storage](05_Thread_Local_Storage.md)
