# Advanced Threading Patterns

*Duration: 2 weeks*

## Overview

Advanced threading patterns help solve complex concurrency problems efficiently. This section covers sophisticated patterns like thread pools, task-based parallelism, and advanced synchronization techniques.

## Thread Pools

### Thread Pool Design Principles

A thread pool maintains a fixed number of worker threads that process tasks from a shared queue, avoiding the overhead of creating and destroying threads frequently.

**Benefits:**
- Reduced thread creation/destruction overhead
- Controlled resource usage
- Better load balancing
- Improved system stability

### Basic Thread Pool Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>

typedef struct {
    void (*function)(void*);
    void* argument;
} Task;

typedef struct {
    Task* tasks;
    int front;
    int rear;
    int count;
    int capacity;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} TaskQueue;

typedef struct {
    pthread_t* threads;
    TaskQueue* task_queue;
    int num_threads;
    bool shutdown;
} ThreadPool;

// Task Queue Implementation
TaskQueue* task_queue_create(int capacity) {
    TaskQueue* queue = malloc(sizeof(TaskQueue));
    queue->tasks = malloc(sizeof(Task) * capacity);
    queue->front = 0;
    queue->rear = -1;
    queue->count = 0;
    queue->capacity = capacity;
    
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);
    
    return queue;
}

void task_queue_enqueue(TaskQueue* queue, Task task) {
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->count == queue->capacity) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    queue->rear = (queue->rear + 1) % queue->capacity;
    queue->tasks[queue->rear] = task;
    queue->count++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
}

Task task_queue_dequeue(TaskQueue* queue) {
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->count == 0) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }
    
    Task task = queue->tasks[queue->front];
    queue->front = (queue->front + 1) % queue->capacity;
    queue->count--;
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    return task;
}

// Worker Thread Function
void* worker_thread(void* arg) {
    ThreadPool* pool = (ThreadPool*)arg;
    
    while (!pool->shutdown) {
        Task task = task_queue_dequeue(pool->task_queue);
        
        if (task.function) {
            task.function(task.argument);
        }
    }
    
    return NULL;
}

// Thread Pool Implementation
ThreadPool* thread_pool_create(int num_threads, int queue_capacity) {
    ThreadPool* pool = malloc(sizeof(ThreadPool));
    pool->threads = malloc(sizeof(pthread_t) * num_threads);
    pool->task_queue = task_queue_create(queue_capacity);
    pool->num_threads = num_threads;
    pool->shutdown = false;
    
    // Create worker threads
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }
    
    return pool;
}

void thread_pool_add_task(ThreadPool* pool, void (*function)(void*), void* argument) {
    Task task = {function, argument};
    task_queue_enqueue(pool->task_queue, task);
}

void thread_pool_destroy(ThreadPool* pool) {
    pool->shutdown = true;
    
    // Wake up all threads
    pthread_mutex_lock(&pool->task_queue->mutex);
    pthread_cond_broadcast(&pool->task_queue->not_empty);
    pthread_mutex_unlock(&pool->task_queue->mutex);
    
    // Wait for all threads to finish
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    // Cleanup
    free(pool->threads);
    free(pool->task_queue->tasks);
    pthread_mutex_destroy(&pool->task_queue->mutex);
    pthread_cond_destroy(&pool->task_queue->not_empty);
    pthread_cond_destroy(&pool->task_queue->not_full);
    free(pool->task_queue);
    free(pool);
}

// Example usage
void sample_task(void* arg) {
    int task_id = *(int*)arg;
    printf("Executing task %d on thread %lu\n", task_id, pthread_self());
    sleep(1); // Simulate work
    printf("Task %d completed\n", task_id);
}

int main() {
    ThreadPool* pool = thread_pool_create(4, 10);
    
    int task_ids[20];
    for (int i = 0; i < 20; i++) {
        task_ids[i] = i;
        thread_pool_add_task(pool, sample_task, &task_ids[i]);
    }
    
    sleep(10); // Let tasks complete
    thread_pool_destroy(pool);
    
    return 0;
}
```

### Dynamic Thread Pool

```c
typedef struct {
    pthread_t* threads;
    TaskQueue* task_queue;
    int min_threads;
    int max_threads;
    int current_threads;
    int active_threads;
    bool shutdown;
    pthread_mutex_t pool_mutex;
    pthread_cond_t work_available;
} DynamicThreadPool;

void* dynamic_worker_thread(void* arg) {
    DynamicThreadPool* pool = (DynamicThreadPool*)arg;
    struct timespec timeout;
    
    while (!pool->shutdown) {
        pthread_mutex_lock(&pool->pool_mutex);
        
        // Wait for work with timeout
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += 30; // 30 second timeout
        
        while (pool->task_queue->count == 0 && !pool->shutdown) {
            int result = pthread_cond_timedwait(&pool->work_available, 
                                               &pool->pool_mutex, &timeout);
            
            if (result == ETIMEDOUT && pool->current_threads > pool->min_threads) {
                // Thread timeout, reduce pool size
                pool->current_threads--;
                pthread_mutex_unlock(&pool->pool_mutex);
                return NULL;
            }
        }
        
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->pool_mutex);
            break;
        }
        
        pool->active_threads++;
        pthread_mutex_unlock(&pool->pool_mutex);
        
        // Get and execute task
        Task task = task_queue_dequeue(pool->task_queue);
        if (task.function) {
            task.function(task.argument);
        }
        
        pthread_mutex_lock(&pool->pool_mutex);
        pool->active_threads--;
        
        // Check if we need more threads
        if (pool->task_queue->count > 0 && 
            pool->current_threads < pool->max_threads &&
            pool->active_threads == pool->current_threads) {
            
            // Add new thread
            pthread_t new_thread;
            if (pthread_create(&new_thread, NULL, dynamic_worker_thread, pool) == 0) {
                pool->threads[pool->current_threads] = new_thread;
                pool->current_threads++;
            }
        }
        
        pthread_mutex_unlock(&pool->pool_mutex);
    }
    
    return NULL;
}
```

## Task-Based Parallelism

### Future/Promise Pattern

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    void* result;
    bool ready;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
} Future;

typedef struct {
    void* (*function)(void*);
    void* argument;
    Future* future;
} AsyncTask;

Future* future_create() {
    Future* future = malloc(sizeof(Future));
    future->result = NULL;
    future->ready = false;
    pthread_mutex_init(&future->mutex, NULL);
    pthread_cond_init(&future->condition, NULL);
    return future;
}

void* async_executor(void* arg) {
    AsyncTask* task = (AsyncTask*)arg;
    
    // Execute the task
    void* result = task->function(task->argument);
    
    // Set the result
    pthread_mutex_lock(&task->future->mutex);
    task->future->result = result;
    task->future->ready = true;
    pthread_cond_signal(&task->future->condition);
    pthread_mutex_unlock(&task->future->mutex);
    
    free(task);
    return NULL;
}

Future* async_submit(ThreadPool* pool, void* (*function)(void*), void* argument) {
    Future* future = future_create();
    AsyncTask* task = malloc(sizeof(AsyncTask));
    task->function = function;
    task->argument = argument;
    task->future = future;
    
    pthread_t thread;
    pthread_create(&thread, NULL, async_executor, task);
    pthread_detach(thread);
    
    return future;
}

void* future_get(Future* future) {
    pthread_mutex_lock(&future->mutex);
    
    while (!future->ready) {
        pthread_cond_wait(&future->condition, &future->mutex);
    }
    
    void* result = future->result;
    pthread_mutex_unlock(&future->mutex);
    
    return result;
}

void future_destroy(Future* future) {
    pthread_mutex_destroy(&future->mutex);
    pthread_cond_destroy(&future->condition);
    free(future);
}

// Example usage
void* calculate_fibonacci(void* arg) {
    int n = *(int*)arg;
    if (n <= 1) return (void*)(long)n;
    
    int a = 0, b = 1, c;
    for (int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    
    return (void*)(long)b;
}

int main() {
    int n = 30;
    Future* future = async_submit(NULL, calculate_fibonacci, &n);
    
    printf("Calculating Fibonacci(%d) asynchronously...\n", n);
    
    // Do other work while calculation runs
    sleep(1);
    
    long result = (long)future_get(future);
    printf("Fibonacci(%d) = %ld\n", n, result);
    
    future_destroy(future);
    return 0;
}
```

### Work Stealing Queue

```c
typedef struct {
    Task* tasks;
    int head;
    int tail;
    int capacity;
    pthread_mutex_t mutex;
} WorkStealingQueue;

typedef struct {
    WorkStealingQueue* queues;
    pthread_t* threads;
    int num_threads;
    bool shutdown;
} WorkStealingPool;

WorkStealingQueue* ws_queue_create(int capacity) {
    WorkStealingQueue* queue = malloc(sizeof(WorkStealingQueue));
    queue->tasks = malloc(sizeof(Task) * capacity);
    queue->head = 0;
    queue->tail = 0;
    queue->capacity = capacity;
    pthread_mutex_init(&queue->mutex, NULL);
    return queue;
}

bool ws_queue_push(WorkStealingQueue* queue, Task task) {
    pthread_mutex_lock(&queue->mutex);
    
    int next_tail = (queue->tail + 1) % queue->capacity;
    if (next_tail == queue->head) {
        pthread_mutex_unlock(&queue->mutex);
        return false; // Queue full
    }
    
    queue->tasks[queue->tail] = task;
    queue->tail = next_tail;
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

bool ws_queue_pop(WorkStealingQueue* queue, Task* task) {
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->head == queue->tail) {
        pthread_mutex_unlock(&queue->mutex);
        return false; // Queue empty
    }
    
    queue->tail = (queue->tail - 1 + queue->capacity) % queue->capacity;
    *task = queue->tasks[queue->tail];
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

bool ws_queue_steal(WorkStealingQueue* queue, Task* task) {
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->head == queue->tail) {
        pthread_mutex_unlock(&queue->mutex);
        return false; // Queue empty
    }
    
    *task = queue->tasks[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

void* work_stealing_worker(void* arg) {
    WorkStealingPool* pool = (WorkStealingPool*)arg;
    int thread_id = (int)(uintptr_t)pthread_getspecific(thread_id_key);
    WorkStealingQueue* my_queue = &pool->queues[thread_id];
    
    while (!pool->shutdown) {
        Task task;
        bool found_work = false;
        
        // Try to get work from own queue
        if (ws_queue_pop(my_queue, &task)) {
            found_work = true;
        } else {
            // Try to steal work from other threads
            for (int i = 0; i < pool->num_threads; i++) {
                if (i != thread_id && ws_queue_steal(&pool->queues[i], &task)) {
                    found_work = true;
                    break;
                }
            }
        }
        
        if (found_work) {
            task.function(task.argument);
        } else {
            usleep(1000); // Brief sleep if no work found
        }
    }
    
    return NULL;
}
```

## Thread Cancellation and Cleanup

### Cancellation Types

```c
#include <pthread.h>

void setup_cancellation() {
    // Set cancellation state
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    
    // Set cancellation type
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);
    // or PTHREAD_CANCEL_ASYNCHRONOUS for immediate cancellation
}

void* cancellable_thread(void* arg) {
    setup_cancellation();
    
    // Install cleanup handlers
    pthread_cleanup_push(cleanup_function, cleanup_arg);
    
    while (1) {
        // Cancellation point
        pthread_testcancel();
        
        // Do work
        do_some_work();
        
        // Another cancellation point
        sleep(1);
    }
    
    pthread_cleanup_pop(1); // Execute cleanup on normal exit
    return NULL;
}

void cleanup_function(void* arg) {
    printf("Cleaning up resources\n");
    // Free resources, unlock mutexes, etc.
}

int main() {
    pthread_t thread;
    
    pthread_create(&thread, NULL, cancellable_thread, NULL);
    
    sleep(5);
    
    // Cancel the thread
    pthread_cancel(thread);
    
    void* result;
    pthread_join(thread, &result);
    
    if (result == PTHREAD_CANCELED) {
        printf("Thread was cancelled\n");
    }
    
    return 0;
}
```

### Robust Cleanup Patterns

```c
typedef struct {
    pthread_mutex_t* mutex;
    FILE* file;
    void* buffer;
} ThreadResources;

void cleanup_thread_resources(void* arg) {
    ThreadResources* resources = (ThreadResources*)arg;
    
    if (resources->mutex) {
        pthread_mutex_unlock(resources->mutex);
    }
    
    if (resources->file) {
        fclose(resources->file);
    }
    
    if (resources->buffer) {
        free(resources->buffer);
    }
    
    free(resources);
}

void* robust_thread(void* arg) {
    ThreadResources* resources = malloc(sizeof(ThreadResources));
    resources->mutex = &shared_mutex;
    resources->file = NULL;
    resources->buffer = NULL;
    
    pthread_cleanup_push(cleanup_thread_resources, resources);
    
    // Acquire resources
    pthread_mutex_lock(resources->mutex);
    resources->file = fopen("data.txt", "r");
    resources->buffer = malloc(1024);
    
    // Do work that might be cancelled
    process_file(resources->file, resources->buffer);
    
    // Normal cleanup
    pthread_mutex_unlock(resources->mutex);
    fclose(resources->file);
    free(resources->buffer);
    free(resources);
    
    pthread_cleanup_pop(0); // Don't execute cleanup on normal exit
    return NULL;
}
```

## Performance Optimization Patterns

### Lock-Free Stack

```c
#include <stdatomic.h>

typedef struct StackNode {
    void* data;
    struct StackNode* next;
} StackNode;

typedef struct {
    _Atomic(StackNode*) head;
} LockFreeStack;

void stack_init(LockFreeStack* stack) {
    atomic_store(&stack->head, NULL);
}

void stack_push(LockFreeStack* stack, void* data) {
    StackNode* new_node = malloc(sizeof(StackNode));
    new_node->data = data;
    
    StackNode* old_head;
    do {
        old_head = atomic_load(&stack->head);
        new_node->next = old_head;
    } while (!atomic_compare_exchange_weak(&stack->head, &old_head, new_node));
}

void* stack_pop(LockFreeStack* stack) {
    StackNode* old_head;
    StackNode* new_head;
    
    do {
        old_head = atomic_load(&stack->head);
        if (!old_head) {
            return NULL; // Stack empty
        }
        new_head = old_head->next;
    } while (!atomic_compare_exchange_weak(&stack->head, &old_head, new_head));
    
    void* data = old_head->data;
    free(old_head);
    return data;
}
```

### Thread-Local Memory Pool

```c
__thread struct {
    void* blocks[1024];
    int count;
    size_t block_size;
} thread_pool = {.count = 0, .block_size = 64};

void* fast_alloc(size_t size) {
    if (size <= thread_pool.block_size && thread_pool.count > 0) {
        return thread_pool.blocks[--thread_pool.count];
    }
    return malloc(size);
}

void fast_free(void* ptr, size_t size) {
    if (size <= thread_pool.block_size && 
        thread_pool.count < sizeof(thread_pool.blocks)/sizeof(void*)) {
        thread_pool.blocks[thread_pool.count++] = ptr;
    } else {
        free(ptr);
    }
}
```

## Advanced Synchronization Patterns

### Reader-Writer Lock with Priority

```c
typedef struct {
    int readers;
    int writers;
    int waiting_readers;
    int waiting_writers;
    bool writer_priority;
    pthread_mutex_t mutex;
    pthread_cond_t readers_cond;
    pthread_cond_t writers_cond;
} PriorityRWLock;

void priority_rwlock_read_lock(PriorityRWLock* lock) {
    pthread_mutex_lock(&lock->mutex);
    
    if (lock->writer_priority) {
        lock->waiting_readers++;
        while (lock->writers > 0 || lock->waiting_writers > 0) {
            pthread_cond_wait(&lock->readers_cond, &lock->mutex);
        }
        lock->waiting_readers--;
    } else {
        while (lock->writers > 0) {
            pthread_cond_wait(&lock->readers_cond, &lock->mutex);
        }
    }
    
    lock->readers++;
    pthread_mutex_unlock(&lock->mutex);
}

void priority_rwlock_write_lock(PriorityRWLock* lock) {
    pthread_mutex_lock(&lock->mutex);
    
    lock->waiting_writers++;
    while (lock->readers > 0 || lock->writers > 0) {
        pthread_cond_wait(&lock->writers_cond, &lock->mutex);
    }
    lock->waiting_writers--;
    lock->writers++;
    
    pthread_mutex_unlock(&lock->mutex);
}
```

## Exercises

1. **Enhanced Thread Pool**
   - Implement priority queue for tasks
   - Add thread pool statistics and monitoring
   - Support for task cancellation

2. **Fork-Join Framework**
   - Implement a fork-join task framework
   - Support recursive task decomposition
   - Compare performance with thread pool

3. **Actor Model Implementation**
   - Create an actor-based concurrency system
   - Each actor has its own message queue
   - Implement message passing between actors

4. **Lock-Free Data Structures**
   - Implement lock-free queue and hash table
   - Compare performance with locked versions
   - Handle ABA problem correctly

## Assessment

You should be able to:
- Design and implement efficient thread pools
- Create task-based parallel algorithms
- Implement advanced synchronization patterns
- Use lock-free programming techniques appropriately
- Handle thread cancellation and cleanup properly
- Optimize threading performance for specific use cases

## Next Section
[Performance Considerations](07_Performance_Considerations.md)
