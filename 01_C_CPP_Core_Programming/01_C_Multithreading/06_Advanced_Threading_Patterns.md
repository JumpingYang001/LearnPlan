# Advanced Threading Patterns

*Duration: 2 weeks*

# Advanced Threading Patterns: Mastering Complex Concurrency

*Duration: 2 weeks*

## Overview

Advanced threading patterns represent the pinnacle of concurrent programming, addressing complex real-world scenarios where basic synchronization primitives are insufficient. These patterns enable you to build scalable, high-performance multi-threaded systems that can handle thousands of concurrent operations while maintaining correctness and efficiency.

### The Evolution of Threading Patterns

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Threading Pattern Hierarchy                     │
├─────────────────────────────────────────────────────────────────────┤
│  Basic Patterns     │  Advanced Patterns    │  Expert Patterns     │
├─────────────────────────────────────────────────────────────────────┤
│  • Mutex/Lock      │  • Thread Pools       │  • Actor Model       │
│  • Condition Vars  │  • Producer-Consumer  │  • Lock-Free Structures │
│  • Semaphores      │  • Reader-Writer      │  • Work Stealing      │
│  • Barriers        │  • Future/Promise     │  • Transactional Memory │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Advanced Patterns Matter

**Performance Scaling**: Traditional locking approaches break down under high contention:

```c
// PROBLEM: Traditional approach doesn't scale
typedef struct {
    int data[1000000];
    pthread_mutex_t mutex;
} shared_data_t;

// High contention - all threads fight for the same lock
void traditional_approach(shared_data_t* shared) {
    pthread_mutex_lock(&shared->mutex);
    // Critical section - only one thread can work
    process_data(shared->data);
    pthread_mutex_unlock(&shared->mutex);
}

// SOLUTION: Advanced pattern - work distribution
typedef struct {
    work_stealing_queue_t* queues;
    int num_workers;
    atomic_int completed_tasks;
} advanced_system_t;

// Multiple workers, no contention
void advanced_approach(advanced_system_t* system) {
    task_t* task = steal_work(system->queues, get_worker_id());
    if (task) {
        process_task(task);
        atomic_fetch_add(&system->completed_tasks, 1);
    }
}
```

### Core Pattern Categories

#### 1. **Task Management Patterns**
- **Thread Pools**: Fixed/dynamic worker thread management
- **Work Stealing**: Load balancing across threads
- **Fork-Join**: Recursive parallel decomposition
- **Future/Promise**: Asynchronous result handling

#### 2. **Synchronization Patterns**
- **Lock-Free Programming**: Wait-free data structures
- **Transactional Memory**: Atomic multi-location updates
- **Event-Driven**: Asynchronous event processing
- **Pipeline**: Stream processing with stages

#### 3. **Communication Patterns**
- **Actor Model**: Message-passing concurrency
- **Channel-Based**: Go-style communication
- **Publish-Subscribe**: Event broadcasting
- **Request-Reply**: Synchronous message exchange

#### 4. **Resource Management Patterns**
- **Object Pools**: Resource lifecycle management
- **Copy-on-Write**: Lazy copying for performance
- **Memory Barriers**: Hardware-level synchronization
- **NUMA-Aware**: Non-uniform memory access optimization

### Performance Characteristics Comparison

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>

// Performance measurement framework
typedef struct {
    const char* pattern_name;
    double operations_per_second;
    double memory_usage_mb;
    double cpu_utilization;
    int scalability_factor; // 1-10 scale
} pattern_performance_t;

// Example measurements (typical values)
static const pattern_performance_t pattern_comparison[] = {
    {"Basic Mutex",           1000000,  0.1,  25.0, 2},
    {"Reader-Writer Lock",    3000000,  0.2,  45.0, 4},
    {"Thread Pool",           8000000,  2.0,  80.0, 7},
    {"Work Stealing",        15000000,  4.0,  95.0, 9},
    {"Lock-Free Queue",      25000000,  1.0,  98.0, 10},
    {"Actor Model",          12000000,  8.0,  85.0, 8},
};

void print_pattern_comparison() {
    printf("┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐\n");
    printf("│ Pattern             │ Ops/Sec     │ Memory (MB) │ CPU Usage   │ Scalability │\n");
    printf("├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤\n");
    
    for (int i = 0; i < 6; i++) {
        printf("│ %-19s │ %11.0f │ %11.1f │ %10.1f%% │ %11d │\n",
               pattern_comparison[i].pattern_name,
               pattern_comparison[i].operations_per_second,
               pattern_comparison[i].memory_usage_mb,
               pattern_comparison[i].cpu_utilization,
               pattern_comparison[i].scalability_factor);
    }
    
    printf("└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘\n");
}
```

### Pattern Selection Guidelines

#### **When to Use Each Pattern:**

1. **Thread Pools** → CPU-bound tasks with predictable load
2. **Work Stealing** → Irregular workloads with load imbalance
3. **Lock-Free** → High contention, low-latency requirements
4. **Actor Model** → Complex state management, fault tolerance
5. **Pipeline** → Stream processing, data transformation
6. **Future/Promise** → Asynchronous operations, composition

### Real-World Application Scenarios

```c
// High-Frequency Trading System
typedef struct {
    lock_free_queue_t* order_queue;
    thread_pool_t* execution_pool;
    atomic_uint64_t processed_orders;
    atomic_uint64_t rejected_orders;
} trading_system_t;

// Web Server with Connection Pool
typedef struct {
    work_stealing_pool_t* worker_pool;
    connection_pool_t* db_connections;
    pipeline_t* request_pipeline;
    actor_system_t* session_actors;
} web_server_t;

// Game Engine with Task System
typedef struct {
    fork_join_pool_t* physics_pool;
    pipeline_t* rendering_pipeline;
    event_system_t* input_events;
    memory_pool_t* object_pools;
} game_engine_t;
```

### Advanced Pattern Principles

#### 1. **Minimize Contention**
```c
// BAD: Single point of contention
static int global_counter = 0;
static pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void increment_slow() {
    pthread_mutex_lock(&counter_mutex);
    global_counter++;  // All threads wait here
    pthread_mutex_unlock(&counter_mutex);
}

// GOOD: Per-thread counters, periodic aggregation
__thread int thread_counter = 0;
static atomic_int aggregated_counter = 0;

void increment_fast() {
    thread_counter++;  // No contention
    
    if (thread_counter % 1000 == 0) {
        atomic_fetch_add(&aggregated_counter, 1000);
        thread_counter = 0;
    }
}
```

#### 2. **Optimize for Cache Locality**
```c
// BAD: False sharing
struct bad_design {
    int counter1;  // Different cache lines
    int counter2;  // but accessed by different threads
    int counter3;
};

// GOOD: Cache-line alignment
struct good_design {
    alignas(64) int counter1;  // Each counter on its own cache line
    alignas(64) int counter2;
    alignas(64) int counter3;
};
```

#### 3. **Design for Composability**
```c
// Composable pattern design
typedef struct operation {
    void (*execute)(struct operation* op);
    void (*cleanup)(struct operation* op);
    struct operation* next;
} operation_t;

// Operations can be chained, parallelized, or distributed
typedef struct {
    operation_t* operations;
    execution_strategy_t strategy;
} workflow_t;
```

This comprehensive overview sets the foundation for understanding why advanced threading patterns are essential for modern concurrent programming. The following sections will dive deep into implementing these patterns with production-ready code and real-world optimizations.

## Advanced Thread Pool Implementations

### Thread Pool Architecture Overview

Thread pools are the foundation of most high-performance concurrent systems. They manage a collection of worker threads that execute tasks from a shared queue, providing several key benefits:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Thread Pool Architecture                    │
├─────────────────────────────────────────────────────────────────────┤
│  Client Code                                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │    Task 1   │    │    Task 2   │    │    Task 3   │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│         │                   │                   │                  │
│         └───────────────────┼───────────────────┘                  │
│                             ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Task Queue                               │   │
│  │  [Task] → [Task] → [Task] → [Task] → [Task]                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             │                                      │
│  ┌──────────────────────────┼──────────────────────────────────┐   │
│  │                          ▼                                  │   │
│  │  Worker 1    Worker 2    Worker 3    Worker 4              │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │   │
│  │  │ Thread  │ │ Thread  │ │ Thread  │ │ Thread  │           │   │
│  │  │ Pool    │ │ Pool    │ │ Pool    │ │ Pool    │           │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Production-Grade Thread Pool Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <time.h>
#include <stdatomic.h>
#include <signal.h>

#define MAX_THREADS 64
#define MAX_QUEUE_SIZE 10000
#define THREAD_NAME_SIZE 32

// Task priority levels
typedef enum {
    PRIORITY_LOW = 0,
    PRIORITY_NORMAL = 1,
    PRIORITY_HIGH = 2,
    PRIORITY_CRITICAL = 3,
    PRIORITY_LEVELS = 4
} task_priority_t;

// Task structure with comprehensive metadata
typedef struct task {
    void (*function)(void*);
    void* argument;
    task_priority_t priority;
    
    // Timing and tracking
    struct timespec created_time;
    struct timespec start_time;
    struct timespec end_time;
    
    // Identification and debugging
    int task_id;
    const char* task_name;
    const char* source_file;
    int source_line;
    
    // Cancellation support
    atomic_bool cancelled;
    void (*cleanup_function)(void*);
    
    // Completion callback
    void (*completion_callback)(struct task*, void*);
    void* callback_data;
    
    struct task* next;
} task_t;

// Priority queue for tasks
typedef struct {
    task_t* queues[PRIORITY_LEVELS];
    task_t* tails[PRIORITY_LEVELS];
    int counts[PRIORITY_LEVELS];
    int total_count;
    int max_size;
    
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    
    // Statistics
    atomic_ulong total_enqueued;
    atomic_ulong total_dequeued;
    atomic_ulong total_dropped;
} priority_queue_t;

// Worker thread statistics
typedef struct {
    int thread_id;
    pthread_t thread_handle;
    char thread_name[THREAD_NAME_SIZE];
    
    // State tracking
    atomic_int state; // 0=idle, 1=running, 2=blocked
    atomic_ulong tasks_completed;
    atomic_ulong total_execution_time_ns;
    
    // Current task info
    task_t* current_task;
    struct timespec task_start_time;
    
    // Performance metrics
    double avg_task_time;
    double cpu_utilization;
    struct timespec last_active_time;
} worker_thread_t;

// Main thread pool structure
typedef struct {
    // Core components
    priority_queue_t* task_queue;
    worker_thread_t* workers;
    int num_threads;
    int max_threads;
    int min_threads;
    
    // State management
    atomic_bool shutdown;
    atomic_bool paused;
    atomic_int active_threads;
    
    // Dynamic scaling
    atomic_int pending_tasks;
    atomic_int idle_threads;
    int scale_up_threshold;
    int scale_down_threshold;
    
    // Statistics and monitoring
    atomic_ulong total_tasks_processed;
    atomic_ulong total_tasks_rejected;
    struct timespec creation_time;
    
    // Configuration
    int thread_stack_size;
    int thread_priority;
    bool enable_monitoring;
    
    // Synchronization
    pthread_mutex_t pool_mutex;
    pthread_cond_t scaling_condition;
    
    // Monitoring thread
    pthread_t monitor_thread;
    int monitor_interval_ms;
} thread_pool_t;

// Task creation macro for debugging
#define CREATE_TASK(pool, func, arg, prio, name) \
    create_task_detailed(pool, func, arg, prio, name, __FILE__, __LINE__)

// Initialize priority queue
priority_queue_t* priority_queue_create(int max_size) {
    priority_queue_t* queue = calloc(1, sizeof(priority_queue_t));
    if (!queue) return NULL;
    
    queue->max_size = max_size;
    queue->total_count = 0;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0 ||
        pthread_cond_init(&queue->not_empty, NULL) != 0 ||
        pthread_cond_init(&queue->not_full, NULL) != 0) {
        free(queue);
        return NULL;
    }
    
    return queue;
}

// Create task with detailed metadata
task_t* create_task_detailed(thread_pool_t* pool, void (*function)(void*), 
                           void* argument, task_priority_t priority,
                           const char* name, const char* file, int line) {
    static atomic_int task_counter = 0;
    
    task_t* task = malloc(sizeof(task_t));
    if (!task) return NULL;
    
    task->function = function;
    task->argument = argument;
    task->priority = priority;
    task->task_id = atomic_fetch_add(&task_counter, 1);
    task->task_name = name;
    task->source_file = file;
    task->source_line = line;
    task->next = NULL;
    
    clock_gettime(CLOCK_MONOTONIC, &task->created_time);
    atomic_store(&task->cancelled, false);
    
    task->cleanup_function = NULL;
    task->completion_callback = NULL;
    task->callback_data = NULL;
    
    return task;
}

// Enqueue task with priority handling
bool priority_queue_enqueue(priority_queue_t* queue, task_t* task) {
    pthread_mutex_lock(&queue->mutex);
    
    // Check if queue is full
    while (queue->total_count >= queue->max_size) {
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += 1; // 1 second timeout
        
        int result = pthread_cond_timedwait(&queue->not_full, &queue->mutex, &timeout);
        if (result == ETIMEDOUT) {
            pthread_mutex_unlock(&queue->mutex);
            atomic_fetch_add(&queue->total_dropped, 1);
            return false;
        }
    }
    
    // Add to appropriate priority queue
    int priority = task->priority;
    if (queue->queues[priority] == NULL) {
        queue->queues[priority] = task;
        queue->tails[priority] = task;
    } else {
        queue->tails[priority]->next = task;
        queue->tails[priority] = task;
    }
    
    queue->counts[priority]++;
    queue->total_count++;
    atomic_fetch_add(&queue->total_enqueued, 1);
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}

// Dequeue task with priority handling
task_t* priority_queue_dequeue(priority_queue_t* queue, int timeout_ms) {
    pthread_mutex_lock(&queue->mutex);
    
    struct timespec timeout;
    if (timeout_ms > 0) {
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += timeout_ms / 1000;
        timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
    }
    
    while (queue->total_count == 0) {
        int result;
        if (timeout_ms > 0) {
            result = pthread_cond_timedwait(&queue->not_empty, &queue->mutex, &timeout);
            if (result == ETIMEDOUT) {
                pthread_mutex_unlock(&queue->mutex);
                return NULL;
            }
        } else {
            pthread_cond_wait(&queue->not_empty, &queue->mutex);
        }
    }
    
    // Find highest priority task
    task_t* task = NULL;
    for (int priority = PRIORITY_LEVELS - 1; priority >= 0; priority--) {
        if (queue->queues[priority] != NULL) {
            task = queue->queues[priority];
            queue->queues[priority] = task->next;
            
            if (queue->queues[priority] == NULL) {
                queue->tails[priority] = NULL;
            }
            
            queue->counts[priority]--;
            queue->total_count--;
            atomic_fetch_add(&queue->total_dequeued, 1);
            break;
        }
    }
    
    if (task) {
        task->next = NULL;
        pthread_cond_signal(&queue->not_full);
    }
    
    pthread_mutex_unlock(&queue->mutex);
    return task;
}

// Worker thread main function
void* worker_thread_main(void* arg) {
    worker_thread_t* worker = (worker_thread_t*)arg;
    thread_pool_t* pool = (thread_pool_t*)worker->thread_handle; // Pass pool reference
    
    // Set thread name for debugging
    pthread_setname_np(pthread_self(), worker->thread_name);
    
    printf("Worker %d (%s) started\n", worker->thread_id, worker->thread_name);
    
    while (!atomic_load(&pool->shutdown)) {
        // Check if pool is paused
        if (atomic_load(&pool->paused)) {
            atomic_store(&worker->state, 2); // blocked
            usleep(10000); // 10ms
            continue;
        }
        
        // Try to get a task
        atomic_store(&worker->state, 0); // idle
        atomic_fetch_add(&pool->idle_threads, 1);
        
        task_t* task = priority_queue_dequeue(pool->task_queue, 1000); // 1 second timeout
        
        atomic_fetch_sub(&pool->idle_threads, 1);
        
        if (!task) {
            // No task available, check for shutdown
            continue;
        }
        
        // Check if task is cancelled
        if (atomic_load(&task->cancelled)) {
            if (task->cleanup_function) {
                task->cleanup_function(task->argument);
            }
            free(task);
            continue;
        }
        
        // Execute task
        atomic_store(&worker->state, 1); // running
        worker->current_task = task;
        clock_gettime(CLOCK_MONOTONIC, &worker->task_start_time);
        clock_gettime(CLOCK_MONOTONIC, &task->start_time);
        
        // Execute the actual task
        task->function(task->argument);
        
        // Record completion
        clock_gettime(CLOCK_MONOTONIC, &task->end_time);
        
        // Calculate execution time
        long execution_time_ns = (task->end_time.tv_sec - task->start_time.tv_sec) * 1000000000L +
                                (task->end_time.tv_nsec - task->start_time.tv_nsec);
        
        // Update worker statistics
        atomic_fetch_add(&worker->tasks_completed, 1);
        atomic_fetch_add(&worker->total_execution_time_ns, execution_time_ns);
        
        // Update pool statistics
        atomic_fetch_add(&pool->total_tasks_processed, 1);
        
        // Call completion callback if provided
        if (task->completion_callback) {
            task->completion_callback(task, task->callback_data);
        }
        
        // Cleanup
        worker->current_task = NULL;
        clock_gettime(CLOCK_MONOTONIC, &worker->last_active_time);
        
        free(task);
    }
    
    printf("Worker %d (%s) shutting down\n", worker->thread_id, worker->thread_name);
    atomic_store(&worker->state, -1); // shutdown
    
    return NULL;
}

// Create thread pool
thread_pool_t* thread_pool_create(int min_threads, int max_threads, int queue_size) {
    if (min_threads <= 0 || max_threads <= 0 || min_threads > max_threads ||
        max_threads > MAX_THREADS || queue_size <= 0) {
        return NULL;
    }
    
    thread_pool_t* pool = calloc(1, sizeof(thread_pool_t));
    if (!pool) return NULL;
    
    pool->task_queue = priority_queue_create(queue_size);
    if (!pool->task_queue) {
        free(pool);
        return NULL;
    }
    
    pool->workers = calloc(max_threads, sizeof(worker_thread_t));
    if (!pool->workers) {
        free(pool->task_queue);
        free(pool);
        return NULL;
    }
    
    // Initialize pool configuration
    pool->min_threads = min_threads;
    pool->max_threads = max_threads;
    pool->num_threads = min_threads;
    pool->scale_up_threshold = queue_size / 4;
    pool->scale_down_threshold = queue_size / 8;
    pool->monitor_interval_ms = 1000;
    pool->enable_monitoring = true;
    
    atomic_store(&pool->shutdown, false);
    atomic_store(&pool->paused, false);
    atomic_store(&pool->active_threads, 0);
    atomic_store(&pool->pending_tasks, 0);
    atomic_store(&pool->idle_threads, 0);
    
    clock_gettime(CLOCK_MONOTONIC, &pool->creation_time);
    
    // Initialize synchronization
    if (pthread_mutex_init(&pool->pool_mutex, NULL) != 0 ||
        pthread_cond_init(&pool->scaling_condition, NULL) != 0) {
        free(pool->workers);
        free(pool->task_queue);
        free(pool);
        return NULL;
    }
    
    // Create initial worker threads
    for (int i = 0; i < min_threads; i++) {
        worker_thread_t* worker = &pool->workers[i];
        worker->thread_id = i;
        snprintf(worker->thread_name, THREAD_NAME_SIZE, "Worker-%d", i);
        
        atomic_store(&worker->state, 0);
        atomic_store(&worker->tasks_completed, 0);
        atomic_store(&worker->total_execution_time_ns, 0);
        
        if (pthread_create(&worker->thread_handle, NULL, worker_thread_main, worker) != 0) {
            // Cleanup on failure
            atomic_store(&pool->shutdown, true);
            for (int j = 0; j < i; j++) {
                pthread_join(pool->workers[j].thread_handle, NULL);
            }
            pthread_mutex_destroy(&pool->pool_mutex);
            pthread_cond_destroy(&pool->scaling_condition);
            free(pool->workers);
            free(pool->task_queue);
            free(pool);
            return NULL;
        }
        
        atomic_fetch_add(&pool->active_threads, 1);
    }
    
    printf("Thread pool created with %d-%d threads\n", min_threads, max_threads);
    return pool;
}

// Add task to thread pool
bool thread_pool_submit(thread_pool_t* pool, void (*function)(void*), 
                       void* argument, task_priority_t priority, const char* name) {
    if (!pool || !function || atomic_load(&pool->shutdown)) {
        return false;
    }
    
    task_t* task = CREATE_TASK(pool, function, argument, priority, name);
    if (!task) {
        atomic_fetch_add(&pool->total_tasks_rejected, 1);
        return false;
    }
    
    if (!priority_queue_enqueue(pool->task_queue, task)) {
        free(task);
        atomic_fetch_add(&pool->total_tasks_rejected, 1);
        return false;
    }
    
    atomic_fetch_add(&pool->pending_tasks, 1);
    
    // Check if we need to scale up
    pthread_mutex_lock(&pool->pool_mutex);
    if (pool->task_queue->total_count > pool->scale_up_threshold &&
        pool->num_threads < pool->max_threads) {
        pthread_cond_signal(&pool->scaling_condition);
    }
    pthread_mutex_unlock(&pool->pool_mutex);
    
    return true;
}

// Dynamic scaling monitor thread
void* monitor_thread_main(void* arg) {
    thread_pool_t* pool = (thread_pool_t*)arg;
    
    while (!atomic_load(&pool->shutdown)) {
        usleep(pool->monitor_interval_ms * 1000);
        
        if (!pool->enable_monitoring) continue;
        
        int queue_size = pool->task_queue->total_count;
        int idle_threads = atomic_load(&pool->idle_threads);
        int active_threads = atomic_load(&pool->active_threads);
        
        pthread_mutex_lock(&pool->pool_mutex);
        
        // Scale up logic
        if (queue_size > pool->scale_up_threshold && 
            pool->num_threads < pool->max_threads &&
            idle_threads < 2) {
            
            // Create new worker thread
            int new_thread_id = pool->num_threads;
            worker_thread_t* worker = &pool->workers[new_thread_id];
            
            worker->thread_id = new_thread_id;
            snprintf(worker->thread_name, THREAD_NAME_SIZE, "Worker-%d", new_thread_id);
            atomic_store(&worker->state, 0);
            atomic_store(&worker->tasks_completed, 0);
            atomic_store(&worker->total_execution_time_ns, 0);
            
            if (pthread_create(&worker->thread_handle, NULL, worker_thread_main, worker) == 0) {
                pool->num_threads++;
                atomic_fetch_add(&pool->active_threads, 1);
                printf("Scaled up: Added worker %d (total: %d)\n", new_thread_id, pool->num_threads);
            }
        }
        
        // Scale down logic
        else if (queue_size < pool->scale_down_threshold && 
                 pool->num_threads > pool->min_threads &&
                 idle_threads > pool->num_threads / 2) {
            
            // Mark a thread for termination (implement thread-specific shutdown)
            // This is a simplified approach - in production, use more sophisticated methods
            if (pool->num_threads > pool->min_threads) {
                pool->num_threads--;
                printf("Scaled down: Removed worker (total: %d)\n", pool->num_threads);
            }
        }
        
        pthread_mutex_unlock(&pool->pool_mutex);
    }
    
    return NULL;
}

// Get comprehensive pool statistics
typedef struct {
    int active_threads;
    int idle_threads;
    int queue_size;
    unsigned long total_processed;
    unsigned long total_rejected;
    double avg_processing_time_ms;
    double cpu_utilization;
    double queue_utilization;
    
    // Per-worker stats
    struct {
        int worker_id;
        int state;
        unsigned long tasks_completed;
        double avg_task_time_ms;
        const char* current_task_name;
    } worker_stats[MAX_THREADS];
    
    int num_workers;
} pool_statistics_t;

void thread_pool_get_statistics(thread_pool_t* pool, pool_statistics_t* stats) {
    if (!pool || !stats) return;
    
    memset(stats, 0, sizeof(pool_statistics_t));
    
    stats->active_threads = atomic_load(&pool->active_threads);
    stats->idle_threads = atomic_load(&pool->idle_threads);
    stats->queue_size = pool->task_queue->total_count;
    stats->total_processed = atomic_load(&pool->total_tasks_processed);
    stats->total_rejected = atomic_load(&pool->total_tasks_rejected);
    
    // Calculate queue utilization
    stats->queue_utilization = (double)stats->queue_size / pool->task_queue->max_size * 100.0;
    
    // Collect per-worker statistics
    stats->num_workers = pool->num_threads;
    unsigned long total_execution_time = 0;
    
    for (int i = 0; i < pool->num_threads; i++) {
        worker_thread_t* worker = &pool->workers[i];
        
        stats->worker_stats[i].worker_id = worker->thread_id;
        stats->worker_stats[i].state = atomic_load(&worker->state);
        stats->worker_stats[i].tasks_completed = atomic_load(&worker->tasks_completed);
        
        unsigned long worker_time = atomic_load(&worker->total_execution_time_ns);
        total_execution_time += worker_time;
        
        if (stats->worker_stats[i].tasks_completed > 0) {
            stats->worker_stats[i].avg_task_time_ms = 
                (double)worker_time / stats->worker_stats[i].tasks_completed / 1000000.0;
        }
        
        stats->worker_stats[i].current_task_name = 
            worker->current_task ? worker->current_task->task_name : "idle";
    }
    
    // Calculate overall average processing time
    if (stats->total_processed > 0) {
        stats->avg_processing_time_ms = (double)total_execution_time / stats->total_processed / 1000000.0;
    }
    
    // Estimate CPU utilization (simplified)
    stats->cpu_utilization = (double)(stats->active_threads - stats->idle_threads) / stats->active_threads * 100.0;
}

// Print detailed pool statistics
void thread_pool_print_statistics(thread_pool_t* pool) {
    pool_statistics_t stats;
    thread_pool_get_statistics(pool, &stats);
    
    printf("\n=== Thread Pool Statistics ===\n");
    printf("Active Threads: %d\n", stats.active_threads);
    printf("Idle Threads: %d\n", stats.idle_threads);
    printf("Queue Size: %d (%.1f%% full)\n", stats.queue_size, stats.queue_utilization);
    printf("Tasks Processed: %lu\n", stats.total_processed);
    printf("Tasks Rejected: %lu\n", stats.total_rejected);
    printf("Avg Processing Time: %.3f ms\n", stats.avg_processing_time_ms);
    printf("CPU Utilization: %.1f%%\n", stats.cpu_utilization);
    
    printf("\nWorker Thread Details:\n");
    printf("┌────────┬───────────┬──────────────┬─────────────┬──────────────────┐\n");
    printf("│ Worker │   State   │    Tasks     │ Avg Time(ms)│  Current Task    │\n");
    printf("├────────┼───────────┼──────────────┼─────────────┼──────────────────┤\n");
    
    for (int i = 0; i < stats.num_workers; i++) {
        const char* state_str;
        switch (stats.worker_stats[i].state) {
            case 0: state_str = "Idle"; break;
            case 1: state_str = "Running"; break;
            case 2: state_str = "Blocked"; break;
            default: state_str = "Unknown"; break;
        }
        
        printf("│ %6d │ %-9s │ %12lu │ %11.3f │ %-16s │\n",
               stats.worker_stats[i].worker_id,
               state_str,
               stats.worker_stats[i].tasks_completed,
               stats.worker_stats[i].avg_task_time_ms,
               stats.worker_stats[i].current_task_name);
    }
    
    printf("└────────┴───────────┴──────────────┴─────────────┴──────────────────┘\n");
}

// Pause/resume pool operations
void thread_pool_pause(thread_pool_t* pool) {
    if (pool) {
        atomic_store(&pool->paused, true);
        printf("Thread pool paused\n");
    }
}

void thread_pool_resume(thread_pool_t* pool) {
    if (pool) {
        atomic_store(&pool->paused, false);
        printf("Thread pool resumed\n");
    }
}

// Graceful shutdown with timeout
bool thread_pool_shutdown(thread_pool_t* pool, int timeout_seconds) {
    if (!pool) return false;
    
    printf("Initiating thread pool shutdown...\n");
    
    // Signal shutdown
    atomic_store(&pool->shutdown, true);
    
    // Wake up all waiting threads
    pthread_mutex_lock(&pool->task_queue->mutex);
    pthread_cond_broadcast(&pool->task_queue->not_empty);
    pthread_mutex_unlock(&pool->task_queue->mutex);
    
    // Wait for workers to finish with timeout
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_seconds;
    
    bool all_joined = true;
    for (int i = 0; i < pool->num_threads; i++) {
        struct timespec remaining_timeout = timeout;
        int result = pthread_timedjoin_np(pool->workers[i].thread_handle, NULL, &remaining_timeout);
        
        if (result == ETIMEDOUT) {
            printf("Warning: Worker %d did not shutdown within timeout\n", i);
            pthread_cancel(pool->workers[i].thread_handle);
            pthread_join(pool->workers[i].thread_handle, NULL);
            all_joined = false;
        }
    }
    
    // Join monitor thread if it exists
    if (pool->enable_monitoring) {
        pthread_join(pool->monitor_thread, NULL);
    }
    
    printf("Thread pool shutdown complete\n");
    return all_joined;
}

// Complete cleanup
void thread_pool_destroy(thread_pool_t* pool) {
    if (!pool) return;
    
    // Ensure shutdown
    if (!atomic_load(&pool->shutdown)) {
        thread_pool_shutdown(pool, 5);
    }
    
    // Print final statistics
    thread_pool_print_statistics(pool);
    
    // Cleanup resources
    if (pool->task_queue) {
        pthread_mutex_destroy(&pool->task_queue->mutex);
        pthread_cond_destroy(&pool->task_queue->not_empty);
        pthread_cond_destroy(&pool->task_queue->not_full);
        free(pool->task_queue);
    }
    
    pthread_mutex_destroy(&pool->pool_mutex);
    pthread_cond_destroy(&pool->scaling_condition);
    
    free(pool->workers);
    free(pool);
    
    printf("Thread pool resources cleaned up\n");
}

### Advanced Thread Pool Usage Examples

```c
// Example 1: CPU-intensive task processing
void cpu_intensive_task(void* arg) {
    int* data = (int*)arg;
    int n = *data;
    
    // Simulate CPU-intensive work (calculating prime numbers)
    int count = 0;
    for (int i = 2; i <= n; i++) {
        bool is_prime = true;
        for (int j = 2; j * j <= i; j++) {
            if (i % j == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) count++;
    }
    
    printf("Found %d primes up to %d\n", count, n);
    free(data);
}

// Example 2: I/O intensive task with error handling
void io_intensive_task(void* arg) {
    const char* filename = (const char*)arg;
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    // Process file
    char buffer[1024];
    int line_count = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        line_count++;
        // Simulate processing time
        usleep(1000); // 1ms per line
    }
    
    fclose(file);
    printf("Processed %d lines from %s\n", line_count, filename);
}

// Example 3: Network request simulation
void network_request_task(void* arg) {
    int request_id = *(int*)arg;
    
    // Simulate network latency
    usleep((rand() % 100 + 50) * 1000); // 50-150ms
    
    // Simulate success/failure
    if (rand() % 10 == 0) { // 10% failure rate
        printf("Request %d failed\n", request_id);
    } else {
        printf("Request %d completed successfully\n", request_id);
    }
    
    free(arg);
}

// Comprehensive test function
int test_advanced_thread_pool() {
    printf("=== Advanced Thread Pool Test ===\n");
    
    // Create thread pool with dynamic scaling
    thread_pool_t* pool = thread_pool_create(2, 8, 100);
    if (!pool) {
        printf("Failed to create thread pool\n");
        return -1;
    }
    
    // Start monitoring
    if (pool->enable_monitoring) {
        pthread_create(&pool->monitor_thread, NULL, monitor_thread_main, pool);
    }
    
    // Submit various types of tasks
    printf("Submitting CPU-intensive tasks...\n");
    for (int i = 0; i < 10; i++) {
        int* data = malloc(sizeof(int));
        *data = 1000 + i * 500;
        thread_pool_submit(pool, cpu_intensive_task, data, PRIORITY_NORMAL, "cpu_task");
    }
    
    printf("Submitting high-priority tasks...\n");
    for (int i = 0; i < 5; i++) {
        int* data = malloc(sizeof(int));
        *data = i;
        thread_pool_submit(pool, network_request_task, data, PRIORITY_HIGH, "network_task");
    }
    
    printf("Submitting low-priority tasks...\n");
    for (int i = 0; i < 20; i++) {
        int* data = malloc(sizeof(int));
        *data = i + 100;
        thread_pool_submit(pool, network_request_task, data, PRIORITY_LOW, "background_task");
    }
    
    // Monitor progress
    for (int i = 0; i < 10; i++) {
        sleep(2);
        thread_pool_print_statistics(pool);
    }
    
    // Test pause/resume
    printf("\nTesting pause/resume...\n");
    thread_pool_pause(pool);
    sleep(2);
    thread_pool_resume(pool);
    
    // Wait for completion
    printf("Waiting for tasks to complete...\n");
    while (pool->task_queue->total_count > 0) {
        sleep(1);
    }
    
    sleep(2); // Let final tasks complete
    
    // Shutdown and cleanup
    thread_pool_shutdown(pool, 10);
    thread_pool_destroy(pool);
    
    return 0;
}
```

## Advanced Task-Based Parallelism

### Future/Promise Pattern with Composition

The Future/Promise pattern enables asynchronous programming by separating task submission from result retrieval. Advanced implementations support composition, chaining, and error handling.

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <errno.h>
#include <time.h>

typedef enum {
    FUTURE_PENDING,
    FUTURE_COMPLETED,
    FUTURE_FAILED,
    FUTURE_CANCELLED
} future_state_t;

typedef struct future {
    void* result;
    int error_code;
    char error_message[256];
    atomic_int state;
    
    // Synchronization
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    
    // Chaining support
    struct future* next;
    void* (*then_function)(void*, struct future*);
    void (*error_handler)(int, const char*, struct future*);
    
    // Cancellation support
    atomic_bool cancelled;
    void (*cancel_handler)(struct future*);
    
    // Metadata
    int future_id;
    struct timespec created_time;
    struct timespec completed_time;
    
    // Reference counting for cleanup
    atomic_int ref_count;
} future_t;

typedef struct {
    void* (*function)(void*);
    void* argument;
    future_t* future;
    thread_pool_t* pool;
} async_task_t;

static atomic_int future_id_counter = 0;

// Create a new future
future_t* future_create() {
    future_t* future = calloc(1, sizeof(future_t));
    if (!future) return NULL;
    
    atomic_store(&future->state, FUTURE_PENDING);
    atomic_store(&future->cancelled, false);
    atomic_store(&future->ref_count, 1);
    
    future->future_id = atomic_fetch_add(&future_id_counter, 1);
    clock_gettime(CLOCK_MONOTONIC, &future->created_time);
    
    if (pthread_mutex_init(&future->mutex, NULL) != 0 ||
        pthread_cond_init(&future->condition, NULL) != 0) {
        free(future);
        return NULL;
    }
    
    return future;
}

// Increment reference count
void future_retain(future_t* future) {
    if (future) {
        atomic_fetch_add(&future->ref_count, 1);
    }
}

// Decrement reference count and cleanup if needed
void future_release(future_t* future) {
    if (!future) return;
    
    if (atomic_fetch_sub(&future->ref_count, 1) == 1) {
        // Last reference, cleanup
        pthread_mutex_destroy(&future->mutex);
        pthread_cond_destroy(&future->condition);
        
        if (future->next) {
            future_release(future->next);
        }
        
        free(future);
    }
}

// Async task executor
void* async_executor(void* arg) {
    async_task_t* task = (async_task_t*)arg;
    future_t* future = task->future;
    
    // Check for cancellation before starting
    if (atomic_load(&future->cancelled)) {
        pthread_mutex_lock(&future->mutex);
        atomic_store(&future->state, FUTURE_CANCELLED);
        pthread_cond_broadcast(&future->condition);
        pthread_mutex_unlock(&future->mutex);
        
        if (future->cancel_handler) {
            future->cancel_handler(future);
        }
        
        free(task);
        future_release(future);
        return NULL;
    }
    
    // Execute the task
    void* result = NULL;
    int error_code = 0;
    
    try {
        result = task->function(task->argument);
    } catch (...) {
        error_code = EFAULT;
        snprintf(future->error_message, sizeof(future->error_message), 
                "Task execution failed");
    }
    
    // Set the result
    pthread_mutex_lock(&future->mutex);
    
    clock_gettime(CLOCK_MONOTONIC, &future->completed_time);
    
    if (error_code == 0) {
        future->result = result;
        atomic_store(&future->state, FUTURE_COMPLETED);
    } else {
        future->error_code = error_code;
        atomic_store(&future->state, FUTURE_FAILED);
        
        if (future->error_handler) {
            future->error_handler(error_code, future->error_message, future);
        }
    }
    
    pthread_cond_broadcast(&future->condition);
    pthread_mutex_unlock(&future->mutex);
    
    // Handle chaining
    if (future->next && future->then_function) {
        if (atomic_load(&future->state) == FUTURE_COMPLETED) {
            // Execute continuation
            void* chained_result = future->then_function(result, future);
            
            pthread_mutex_lock(&future->next->mutex);
            future->next->result = chained_result;
            atomic_store(&future->next->state, FUTURE_COMPLETED);
            clock_gettime(CLOCK_MONOTONIC, &future->next->completed_time);
            pthread_cond_broadcast(&future->next->condition);
            pthread_mutex_unlock(&future->next->mutex);
        }
    }
    
    free(task);
    future_release(future);
    return NULL;
}

// Submit async task
future_t* async_submit(thread_pool_t* pool, void* (*function)(void*), void* argument) {
    future_t* future = future_create();
    if (!future) return NULL;
    
    async_task_t* task = malloc(sizeof(async_task_t));
    if (!task) {
        future_release(future);
        return NULL;
    }
    
    task->function = function;
    task->argument = argument;
    task->future = future;
    task->pool = pool;
    
    future_retain(future); // Retain for the async task
    
    if (pool) {
        // Submit to thread pool
        if (!thread_pool_submit(pool, (void(*)(void*))async_executor, task, 
                              PRIORITY_NORMAL, "async_task")) {
            free(task);
            future_release(future);
            future_release(future);
            return NULL;
        }
    } else {
        // Create dedicated thread
        pthread_t thread;
        if (pthread_create(&thread, NULL, async_executor, task) != 0) {
            free(task);
            future_release(future);
            future_release(future);
            return NULL;
        }
        pthread_detach(thread);
    }
    
    return future;
}

// Wait for future result with timeout
void* future_get_timeout(future_t* future, int timeout_ms) {
    if (!future) return NULL;
    
    pthread_mutex_lock(&future->mutex);
    
    if (timeout_ms > 0) {
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += timeout_ms / 1000;
        timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
        
        while (atomic_load(&future->state) == FUTURE_PENDING) {
            int result = pthread_cond_timedwait(&future->condition, &future->mutex, &timeout);
            if (result == ETIMEDOUT) {
                pthread_mutex_unlock(&future->mutex);
                return NULL;
            }
        }
    } else {
        while (atomic_load(&future->state) == FUTURE_PENDING) {
            pthread_cond_wait(&future->condition, &future->mutex);
        }
    }
    
    void* result = NULL;
    future_state_t state = atomic_load(&future->state);
    
    if (state == FUTURE_COMPLETED) {
        result = future->result;
    } else if (state == FUTURE_FAILED) {
        printf("Future %d failed: %s (error %d)\n", 
               future->future_id, future->error_message, future->error_code);
    } else if (state == FUTURE_CANCELLED) {
        printf("Future %d was cancelled\n", future->future_id);
    }
    
    pthread_mutex_unlock(&future->mutex);
    return result;
}

// Wait for future result (no timeout)
void* future_get(future_t* future) {
    return future_get_timeout(future, 0);
}

// Cancel a future
bool future_cancel(future_t* future) {
    if (!future) return false;
    
    atomic_store(&future->cancelled, true);
    
    pthread_mutex_lock(&future->mutex);
    if (atomic_load(&future->state) == FUTURE_PENDING) {
        atomic_store(&future->state, FUTURE_CANCELLED);
        pthread_cond_broadcast(&future->condition);
        
        if (future->cancel_handler) {
            future->cancel_handler(future);
        }
    }
    pthread_mutex_unlock(&future->mutex);
    
    return true;
}

// Chain futures with then()
future_t* future_then(future_t* future, void* (*then_func)(void*, future_t*)) {
    if (!future || !then_func) return NULL;
    
    future_t* next_future = future_create();
    if (!next_future) return NULL;
    
    pthread_mutex_lock(&future->mutex);
    future->next = next_future;
    future->then_function = then_func;
    future_retain(next_future); // Retain for chaining
    pthread_mutex_unlock(&future->mutex);
    
    return next_future;
}

// Set error handler
void future_catch(future_t* future, void (*error_handler)(int, const char*, future_t*)) {
    if (future && error_handler) {
        pthread_mutex_lock(&future->mutex);
        future->error_handler = error_handler;
        pthread_mutex_unlock(&future->mutex);
    }
}

// Check if future is complete
bool future_is_done(future_t* future) {
    if (!future) return false;
    
    future_state_t state = atomic_load(&future->state);
    return state != FUTURE_PENDING;
}

// Get future execution time
double future_get_execution_time_ms(future_t* future) {
    if (!future || !future_is_done(future)) return -1.0;
    
    return (future->completed_time.tv_sec - future->created_time.tv_sec) * 1000.0 +
           (future->completed_time.tv_nsec - future->created_time.tv_nsec) / 1000000.0;
}
```

### Fork-Join Pattern Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <math.h>

#define MIN_TASK_SIZE 1000

typedef struct fork_join_task {
    void* (*compute)(struct fork_join_task*);
    void* (*fork)(struct fork_join_task*, int);
    void* (*join)(void**, int);
    
    void* data;
    int start_index;
    int end_index;
    int threshold;
    
    struct fork_join_task** subtasks;
    int num_subtasks;
    
    future_t* future;
    thread_pool_t* pool;
} fork_join_task_t;

// Recursive fork-join execution
void* fork_join_execute(void* arg) {
    fork_join_task_t* task = (fork_join_task_t*)arg;
    
    int task_size = task->end_index - task->start_index;
    
    // Base case: task is small enough to compute directly
    if (task_size <= task->threshold) {
        return task->compute(task);
    }
    
    // Fork phase: divide task into subtasks
    void* fork_result = task->fork(task, task_size);
    if (!fork_result) {
        return task->compute(task); // Fallback to direct computation
    }
    
    // Create futures for subtasks
    future_t** futures = malloc(task->num_subtasks * sizeof(future_t*));
    if (!futures) {
        return task->compute(task);
    }
    
    // Submit subtasks asynchronously
    for (int i = 0; i < task->num_subtasks; i++) {
        futures[i] = async_submit(task->pool, fork_join_execute, task->subtasks[i]);
        if (!futures[i]) {
            // Handle submission failure
            for (int j = 0; j < i; j++) {
                future_cancel(futures[j]);
                future_release(futures[j]);
            }
            free(futures);
            return task->compute(task);
        }
    }
    
    // Join phase: collect results from subtasks
    void** results = malloc(task->num_subtasks * sizeof(void*));
    if (!results) {
        for (int i = 0; i < task->num_subtasks; i++) {
            future_cancel(futures[i]);
            future_release(futures[i]);
        }
        free(futures);
        return task->compute(task);
    }
    
    // Wait for all subtasks to complete
    for (int i = 0; i < task->num_subtasks; i++) {
        results[i] = future_get(futures[i]);
        future_release(futures[i]);
    }
    
    // Combine results
    void* final_result = task->join(results, task->num_subtasks);
    
    // Cleanup
    free(futures);
    free(results);
    
    return final_result;
}

// Example: Parallel sum using fork-join
typedef struct {
    int* array;
    int start;
    int end;
    long result;
} sum_task_data_t;

void* sum_compute(fork_join_task_t* task) {
    sum_task_data_t* data = (sum_task_data_t*)task->data;
    
    long sum = 0;
    for (int i = data->start; i < data->end; i++) {
        sum += data->array[i];
    }
    
    data->result = sum;
    return data;
}

void* sum_fork(fork_join_task_t* task, int task_size) {
    sum_task_data_t* parent_data = (sum_task_data_t*)task->data;
    
    // Divide into two subtasks
    task->num_subtasks = 2;
    task->subtasks = malloc(2 * sizeof(fork_join_task_t*));
    
    if (!task->subtasks) return NULL;
    
    int mid = parent_data->start + task_size / 2;
    
    // Create left subtask
    task->subtasks[0] = malloc(sizeof(fork_join_task_t));
    sum_task_data_t* left_data = malloc(sizeof(sum_task_data_t));
    left_data->array = parent_data->array;
    left_data->start = parent_data->start;
    left_data->end = mid;
    
    task->subtasks[0]->compute = sum_compute;
    task->subtasks[0]->fork = sum_fork;
    task->subtasks[0]->join = task->join;
    task->subtasks[0]->data = left_data;
    task->subtasks[0]->threshold = task->threshold;
    task->subtasks[0]->pool = task->pool;
    
    // Create right subtask
    task->subtasks[1] = malloc(sizeof(fork_join_task_t));
    sum_task_data_t* right_data = malloc(sizeof(sum_task_data_t));
    right_data->array = parent_data->array;
    right_data->start = mid;
    right_data->end = parent_data->end;
    
    task->subtasks[1]->compute = sum_compute;
    task->subtasks[1]->fork = sum_fork;
    task->subtasks[1]->join = task->join;
    task->subtasks[1]->data = right_data;
    task->subtasks[1]->threshold = task->threshold;
    task->subtasks[1]->pool = task->pool;
    
    return task->subtasks;
}

void* sum_join(void** results, int num_results) {
    long total_sum = 0;
    
    for (int i = 0; i < num_results; i++) {
        sum_task_data_t* data = (sum_task_data_t*)results[i];
        total_sum += data->result;
    }
    
    sum_task_data_t* result = malloc(sizeof(sum_task_data_t));
    result->result = total_sum;
    return result;
}

// Test fork-join pattern
void test_fork_join_sum() {
    const int ARRAY_SIZE = 10000000;
    int* array = malloc(ARRAY_SIZE * sizeof(int));
    
    // Initialize array with random values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;
    }
    
    // Create thread pool
    thread_pool_t* pool = thread_pool_create(4, 8, 100);
    
    // Setup fork-join task
    fork_join_task_t* task = malloc(sizeof(fork_join_task_t));
    sum_task_data_t* data = malloc(sizeof(sum_task_data_t));
    
    data->array = array;
    data->start = 0;
    data->end = ARRAY_SIZE;
    
    task->compute = sum_compute;
    task->fork = sum_fork;
    task->join = sum_join;
    task->data = data;
    task->threshold = MIN_TASK_SIZE;
    task->pool = pool;
    
    // Execute parallel sum
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    future_t* future = async_submit(pool, fork_join_execute, task);
    sum_task_data_t* result = (sum_task_data_t*)future_get(future);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double parallel_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    printf("Parallel sum result: %ld\n", result->result);
    printf("Parallel execution time: %.3f ms\n", parallel_time);
    
    // Compare with sequential sum
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    long sequential_sum = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        sequential_sum += array[i];
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double sequential_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                            (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    printf("Sequential sum result: %ld\n", sequential_sum);
    printf("Sequential execution time: %.3f ms\n", sequential_time);
    printf("Speedup: %.2fx\n", sequential_time / parallel_time);
    
    // Cleanup
    future_release(future);
    thread_pool_shutdown(pool, 5);
    thread_pool_destroy(pool);
    free(array);
    free(result);
    free(data);
    free(task);
}
```

### Work Stealing Queue Implementation

Work stealing is an advanced load balancing technique where idle threads "steal" work from busy threads, ensuring optimal resource utilization.

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>

#define QUEUE_SIZE 1024
#define STEAL_BATCH_SIZE 32

// Lock-free work stealing deque
typedef struct {
    void** tasks;
    atomic_size_t top;    // Owner thread pushes/pops here
    atomic_size_t bottom; // Thieves steal from here
    size_t capacity;
    
    // Statistics
    atomic_ulong push_count;
    atomic_ulong pop_count;
    atomic_ulong steal_count;
    atomic_ulong steal_attempts;
} work_stealing_deque_t;

// Work stealing thread pool
typedef struct {
    work_stealing_deque_t* queues;
    pthread_t* threads;
    atomic_bool* thread_active;
    int num_threads;
    atomic_bool shutdown;
    
    // Global statistics
    atomic_ulong total_tasks_processed;
    atomic_ulong total_steals_successful;
    atomic_ulong total_steal_attempts;
    
    // Load balancing
    atomic_int* thread_loads;
    int rebalance_threshold;
} work_stealing_pool_t;

// Thread-local storage for worker identification
__thread int worker_id = -1;

// Create work stealing deque
work_stealing_deque_t* ws_deque_create(size_t capacity) {
    work_stealing_deque_t* deque = malloc(sizeof(work_stealing_deque_t));
    if (!deque) return NULL;
    
    deque->tasks = calloc(capacity, sizeof(void*));
    if (!deque->tasks) {
        free(deque);
        return NULL;
    }
    
    deque->capacity = capacity;
    atomic_store(&deque->top, 0);
    atomic_store(&deque->bottom, 0);
    atomic_store(&deque->push_count, 0);
    atomic_store(&deque->pop_count, 0);
    atomic_store(&deque->steal_count, 0);
    atomic_store(&deque->steal_attempts, 0);
    
    return deque;
}

// Push task to deque (owner thread only)
bool ws_deque_push(work_stealing_deque_t* deque, void* task) {
    if (!deque || !task) return false;
    
    size_t top = atomic_load(&deque->top);
    size_t bottom = atomic_load(&deque->bottom);
    
    // Check if deque is full
    if (top - bottom >= deque->capacity) {
        return false;
    }
    
    // Store task
    deque->tasks[top % deque->capacity] = task;
    
    // Ensure task is visible before updating top
    atomic_thread_fence(memory_order_release);
    atomic_store(&deque->top, top + 1);
    
    atomic_fetch_add(&deque->push_count, 1);
    return true;
}

// Pop task from deque (owner thread only)
void* ws_deque_pop(work_stealing_deque_t* deque) {
    if (!deque) return NULL;
    
    size_t top = atomic_load(&deque->top);
    size_t bottom = atomic_load(&deque->bottom);
    
    if (top <= bottom) {
        return NULL; // Empty deque
    }
    
    // Try to pop from top
    top--;
    atomic_store(&deque->top, top);
    
    // Ensure top update is visible
    atomic_thread_fence(memory_order_seq_cst);
    
    // Check if we raced with a steal
    if (top < atomic_load(&deque->bottom)) {
        // Restore top and fail
        atomic_store(&deque->top, top + 1);
        return NULL;
    }
    
    void* task = deque->tasks[top % deque->capacity];
    
    // If this was the last task, synchronize with potential stealers
    if (top == atomic_load(&deque->bottom)) {
        // Use CAS to ensure atomicity with steal operations
        size_t expected_bottom = bottom;
        if (!atomic_compare_exchange_strong(&deque->bottom, &expected_bottom, bottom + 1)) {
            // Someone stole the last task
            atomic_store(&deque->top, top + 1);
            return NULL;
        }
    }
    
    atomic_fetch_add(&deque->pop_count, 1);
    return task;
}

// Steal task from deque (thief threads)
void* ws_deque_steal(work_stealing_deque_t* deque) {
    if (!deque) return NULL;
    
    atomic_fetch_add(&deque->steal_attempts, 1);
    
    size_t bottom = atomic_load(&deque->bottom);
    size_t top = atomic_load(&deque->top);
    
    if (bottom >= top) {
        return NULL; // Empty deque
    }
    
    // Try to steal from bottom
    void* task = deque->tasks[bottom % deque->capacity];
    
    // Use CAS to atomically steal
    if (!atomic_compare_exchange_strong(&deque->bottom, &bottom, bottom + 1)) {
        return NULL; // Failed to steal
    }
    
    atomic_fetch_add(&deque->steal_count, 1);
    return task;
}

// Get deque statistics
void ws_deque_get_stats(work_stealing_deque_t* deque, 
                       unsigned long* pushes, unsigned long* pops,
                       unsigned long* steals, unsigned long* steal_attempts) {
    if (!deque) return;
    
    if (pushes) *pushes = atomic_load(&deque->push_count);
    if (pops) *pops = atomic_load(&deque->pop_count);
    if (steals) *steals = atomic_load(&deque->steal_count);
    if (steal_attempts) *steal_attempts = atomic_load(&deque->steal_attempts);
}

// Get current deque size
size_t ws_deque_size(work_stealing_deque_t* deque) {
    if (!deque) return 0;
    
    size_t top = atomic_load(&deque->top);
    size_t bottom = atomic_load(&deque->bottom);
    
    return (top > bottom) ? (top - bottom) : 0;
}

// Worker thread main function
void* work_stealing_worker(void* arg) {
    work_stealing_pool_t* pool = (work_stealing_pool_t*)arg;
    
    // Get worker ID (passed through thread creation)
    worker_id = (int)(intptr_t)pthread_getspecific(/* worker_id_key */);
    
    work_stealing_deque_t* my_deque = &pool->queues[worker_id];
    atomic_store(&pool->thread_active[worker_id], true);
    
    printf("Work stealing worker %d started\n", worker_id);
    
    while (!atomic_load(&pool->shutdown)) {
        void* task = NULL;
        bool found_work = false;
        
        // Try to get work from own deque
        task = ws_deque_pop(my_deque);
        if (task) {
            found_work = true;
        } else {
            // Try to steal work from other threads
            for (int attempts = 0; attempts < pool->num_threads * 2 && !task; attempts++) {
                int victim = rand() % pool->num_threads;
                if (victim != worker_id) {
                    task = ws_deque_steal(&pool->queues[victim]);
                    if (task) {
                        found_work = true;
                        atomic_fetch_add(&pool->total_steals_successful, 1);
                        break;
                    }
                }
                atomic_fetch_add(&pool->total_steal_attempts, 1);
            }
        }
        
        if (found_work && task) {
            // Execute task
            void (*task_func)(void*) = (void (*)(void*))task;
            task_func(NULL); // Simplified - in practice, tasks carry data
            
            atomic_fetch_add(&pool->total_tasks_processed, 1);
            atomic_fetch_add(&pool->thread_loads[worker_id], 1);
        } else {
            // No work found, brief sleep
            usleep(1000); // 1ms
        }
    }
    
    atomic_store(&pool->thread_active[worker_id], false);
    printf("Work stealing worker %d shutting down\n", worker_id);
    
    return NULL;
}

// Create work stealing pool
work_stealing_pool_t* ws_pool_create(int num_threads) {
    if (num_threads <= 0) return NULL;
    
    work_stealing_pool_t* pool = malloc(sizeof(work_stealing_pool_t));
    if (!pool) return NULL;
    
    // Initialize arrays
    pool->queues = malloc(num_threads * sizeof(work_stealing_deque_t));
    pool->threads = malloc(num_threads * sizeof(pthread_t));
    pool->thread_active = malloc(num_threads * sizeof(atomic_bool));
    pool->thread_loads = malloc(num_threads * sizeof(atomic_int));
    
    if (!pool->queues || !pool->threads || !pool->thread_active || !pool->thread_loads) {
        free(pool->queues);
        free(pool->threads);
        free(pool->thread_active);
        free(pool->thread_loads);
        free(pool);
        return NULL;
    }
    
    pool->num_threads = num_threads;
    atomic_store(&pool->shutdown, false);
    atomic_store(&pool->total_tasks_processed, 0);
    atomic_store(&pool->total_steals_successful, 0);
    atomic_store(&pool->total_steal_attempts, 0);
    pool->rebalance_threshold = 10;
    
    // Initialize per-thread structures
    for (int i = 0; i < num_threads; i++) {
        pool->queues[i] = *ws_deque_create(QUEUE_SIZE);
        atomic_store(&pool->thread_active[i], false);
        atomic_store(&pool->thread_loads[i], 0);
    }
    
    // Create worker threads
    for (int i = 0; i < num_threads; i++) {
        // Pass worker ID through thread creation
        if (pthread_create(&pool->threads[i], NULL, work_stealing_worker, pool) != 0) {
            // Cleanup on failure
            atomic_store(&pool->shutdown, true);
            for (int j = 0; j < i; j++) {
                pthread_join(pool->threads[j], NULL);
            }
            free(pool->queues);
            free(pool->threads);
            free(pool->thread_active);
            free(pool->thread_loads);
            free(pool);
            return NULL;
        }
    }
    
    printf("Work stealing pool created with %d threads\n", num_threads);
    return pool;
}

// Submit task to work stealing pool
bool ws_pool_submit(work_stealing_pool_t* pool, void (*task_func)(void*)) {
    if (!pool || !task_func || atomic_load(&pool->shutdown)) {
        return false;
    }
    
    // Find thread with least load
    int target_thread = 0;
    int min_load = atomic_load(&pool->thread_loads[0]);
    
    for (int i = 1; i < pool->num_threads; i++) {
        int load = atomic_load(&pool->thread_loads[i]);
        if (load < min_load) {
            min_load = load;
            target_thread = i;
        }
    }
    
    // Try to push to target thread's deque
    if (ws_deque_push(&pool->queues[target_thread], (void*)task_func)) {
        return true;
    }
    
    // If target thread's deque is full, try others
    for (int i = 0; i < pool->num_threads; i++) {
        if (i != target_thread && ws_deque_push(&pool->queues[i], (void*)task_func)) {
            return true;
        }
    }
    
    return false; // All deques are full
}

// Get work stealing pool statistics
void ws_pool_print_statistics(work_stealing_pool_t* pool) {
    if (!pool) return;
    
    printf("\n=== Work Stealing Pool Statistics ===\n");
    printf("Total tasks processed: %lu\n", atomic_load(&pool->total_tasks_processed));
    printf("Total successful steals: %lu\n", atomic_load(&pool->total_steals_successful));
    printf("Total steal attempts: %lu\n", atomic_load(&pool->total_steal_attempts));
    
    if (atomic_load(&pool->total_steal_attempts) > 0) {
        printf("Steal success rate: %.1f%%\n", 
               (double)atomic_load(&pool->total_steals_successful) / 
               atomic_load(&pool->total_steal_attempts) * 100.0);
    }
    
    printf("\nPer-thread statistics:\n");
    printf("┌────────┬────────┬─────────┬──────────┬─────────┬──────────────┐\n");
    printf("│ Thread │ Active │  Load   │  Pushes  │  Pops   │    Steals    │\n");
    printf("├────────┼────────┼─────────┼──────────┼─────────┼──────────────┤\n");
    
    for (int i = 0; i < pool->num_threads; i++) {
        unsigned long pushes, pops, steals, steal_attempts;
        ws_deque_get_stats(&pool->queues[i], &pushes, &pops, &steals, &steal_attempts);
        
        printf("│ %6d │ %6s │ %7d │ %8lu │ %7lu │ %4lu / %5lu │\n",
               i,
               atomic_load(&pool->thread_active[i]) ? "Yes" : "No",
               atomic_load(&pool->thread_loads[i]),
               pushes, pops, steals, steal_attempts);
    }
    
    printf("└────────┴────────┴─────────┴──────────┴─────────┴──────────────┘\n");
}

// Shutdown work stealing pool
void ws_pool_shutdown(work_stealing_pool_t* pool, int timeout_seconds) {
    if (!pool) return;
    
    printf("Shutting down work stealing pool...\n");
    
    atomic_store(&pool->shutdown, true);
    
    // Wait for all threads to finish
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    // Print final statistics
    ws_pool_print_statistics(pool);
    
    // Cleanup
    for (int i = 0; i < pool->num_threads; i++) {
        free(pool->queues[i].tasks);
    }
    
    free(pool->queues);
    free(pool->threads);
    free(pool->thread_active);
    free(pool->thread_loads);
    free(pool);
    
    printf("Work stealing pool shutdown complete\n");
}

// Example: CPU-intensive tasks for work stealing
void cpu_intensive_task(void* arg) {
    int work_amount = rand() % 10000 + 1000; // Variable work amount
    
    volatile long sum = 0;
    for (int i = 0; i < work_amount; i++) {
        sum += i * i;
    }
    
    printf("Worker %d completed task (work: %d)\n", worker_id, work_amount);
}

// Test work stealing pool
void test_work_stealing_pool() {
    printf("=== Work Stealing Pool Test ===\n");
    
    work_stealing_pool_t* pool = ws_pool_create(4);
    if (!pool) {
        printf("Failed to create work stealing pool\n");
        return;
    }
    
    // Submit tasks with varying complexity
    printf("Submitting 100 tasks...\n");
    for (int i = 0; i < 100; i++) {
        if (!ws_pool_submit(pool, cpu_intensive_task)) {
            printf("Failed to submit task %d\n", i);
        }
    }
    
    // Monitor progress
    for (int i = 0; i < 10; i++) {
        sleep(1);
        ws_pool_print_statistics(pool);
    }
    
    // Wait for completion
    printf("Waiting for tasks to complete...\n");
    sleep(5);
    
    ws_pool_shutdown(pool, 10);
}
```

## Advanced Thread Cancellation and Cleanup Patterns

### Comprehensive Cancellation Framework

Thread cancellation is a critical aspect of robust multi-threaded applications. Advanced patterns provide graceful shutdown mechanisms with proper resource cleanup.

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <time.h>

// Cancellation types
typedef enum {
    CANCEL_NONE = 0,
    CANCEL_DEFERRED,
    CANCEL_ASYNCHRONOUS,
    CANCEL_COOPERATIVE
} cancellation_type_t;

// Resource tracking for cleanup
typedef struct resource_node {
    void* resource;
    void (*cleanup_func)(void*);
    const char* resource_name;
    struct resource_node* next;
} resource_node_t;

// Thread cancellation context
typedef struct {
    pthread_t thread;
    atomic_bool cancel_requested;
    cancellation_type_t cancel_type;
    
    // Resource tracking
    resource_node_t* resources;
    pthread_mutex_t resource_mutex;
    
    // Cleanup handlers
    void (**cleanup_handlers)(void*);
    void** cleanup_args;
    int cleanup_count;
    int cleanup_capacity;
    
    // Statistics
    int cancellation_points_checked;
    struct timespec last_cancel_check;
    
    // Custom cancellation function
    bool (*custom_cancel_check)(void);
} cancellation_context_t;

// Thread-local cancellation context
__thread cancellation_context_t* tls_cancel_context = NULL;

// Initialize cancellation context
cancellation_context_t* cancellation_context_create() {
    cancellation_context_t* ctx = calloc(1, sizeof(cancellation_context_t));
    if (!ctx) return NULL;
    
    ctx->thread = pthread_self();
    atomic_store(&ctx->cancel_requested, false);
    ctx->cancel_type = CANCEL_DEFERRED;
    
    if (pthread_mutex_init(&ctx->resource_mutex, NULL) != 0) {
        free(ctx);
        return NULL;
    }
    
    ctx->cleanup_capacity = 10;
    ctx->cleanup_handlers = malloc(ctx->cleanup_capacity * sizeof(void (*)(void*)));
    ctx->cleanup_args = malloc(ctx->cleanup_capacity * sizeof(void*));
    
    if (!ctx->cleanup_handlers || !ctx->cleanup_args) {
        pthread_mutex_destroy(&ctx->resource_mutex);
        free(ctx->cleanup_handlers);
        free(ctx->cleanup_args);
        free(ctx);
        return NULL;
    }
    
    return ctx;
}

// Get or create thread-local cancellation context
cancellation_context_t* get_cancellation_context() {
    if (!tls_cancel_context) {
        tls_cancel_context = cancellation_context_create();
    }
    return tls_cancel_context;
}

// Register a resource for automatic cleanup
bool register_resource(void* resource, void (*cleanup_func)(void*), const char* name) {
    cancellation_context_t* ctx = get_cancellation_context();
    if (!ctx || !resource || !cleanup_func) return false;
    
    resource_node_t* node = malloc(sizeof(resource_node_t));
    if (!node) return false;
    
    node->resource = resource;
    node->cleanup_func = cleanup_func;
    node->resource_name = name;
    
    pthread_mutex_lock(&ctx->resource_mutex);
    node->next = ctx->resources;
    ctx->resources = node;
    pthread_mutex_unlock(&ctx->resource_mutex);
    
    printf("Registered resource: %s\n", name);
    return true;
}

// Unregister a resource
bool unregister_resource(void* resource) {
    cancellation_context_t* ctx = tls_cancel_context;
    if (!ctx || !resource) return false;
    
    pthread_mutex_lock(&ctx->resource_mutex);
    
    resource_node_t** current = &ctx->resources;
    while (*current) {
        if ((*current)->resource == resource) {
            resource_node_t* to_remove = *current;
            *current = (*current)->next;
            
            printf("Unregistered resource: %s\n", to_remove->resource_name);
            free(to_remove);
            
            pthread_mutex_unlock(&ctx->resource_mutex);
            return true;
        }
        current = &(*current)->next;
    }
    
    pthread_mutex_unlock(&ctx->resource_mutex);
    return false;
}

// Push cleanup handler
bool push_cleanup_handler(void (*handler)(void*), void* arg) {
    cancellation_context_t* ctx = get_cancellation_context();
    if (!ctx || !handler) return false;
    
    if (ctx->cleanup_count >= ctx->cleanup_capacity) {
        // Expand cleanup handler arrays
        int new_capacity = ctx->cleanup_capacity * 2;
        void (**new_handlers)(void*) = realloc(ctx->cleanup_handlers, 
                                              new_capacity * sizeof(void (*)(void*)));
        void** new_args = realloc(ctx->cleanup_args, new_capacity * sizeof(void*));
        
        if (!new_handlers || !new_args) {
            free(new_handlers);
            free(new_args);
            return false;
        }
        
        ctx->cleanup_handlers = new_handlers;
        ctx->cleanup_args = new_args;
        ctx->cleanup_capacity = new_capacity;
    }
    
    ctx->cleanup_handlers[ctx->cleanup_count] = handler;
    ctx->cleanup_args[ctx->cleanup_count] = arg;
    ctx->cleanup_count++;
    
    return true;
}

// Pop cleanup handler
void pop_cleanup_handler(bool execute) {
    cancellation_context_t* ctx = tls_cancel_context;
    if (!ctx || ctx->cleanup_count <= 0) return;
    
    ctx->cleanup_count--;
    
    if (execute) {
        ctx->cleanup_handlers[ctx->cleanup_count](ctx->cleanup_args[ctx->cleanup_count]);
    }
}

// Cleanup all resources
void cleanup_all_resources(cancellation_context_t* ctx) {
    if (!ctx) return;
    
    // Execute cleanup handlers in reverse order
    for (int i = ctx->cleanup_count - 1; i >= 0; i--) {
        printf("Executing cleanup handler %d\n", i);
        ctx->cleanup_handlers[i](ctx->cleanup_args[i]);
    }
    
    // Cleanup registered resources
    pthread_mutex_lock(&ctx->resource_mutex);
    resource_node_t* current = ctx->resources;
    while (current) {
        resource_node_t* next = current->next;
        
        printf("Cleaning up resource: %s\n", current->resource_name);
        current->cleanup_func(current->resource);
        free(current);
        
        current = next;
    }
    ctx->resources = NULL;
    pthread_mutex_unlock(&ctx->resource_mutex);
}

// Check for cancellation (cooperative cancellation)
bool check_cancellation() {
    cancellation_context_t* ctx = tls_cancel_context;
    if (!ctx) return false;
    
    ctx->cancellation_points_checked++;
    clock_gettime(CLOCK_MONOTONIC, &ctx->last_cancel_check);
    
    // Check if cancellation was requested
    if (atomic_load(&ctx->cancel_requested)) {
        printf("Cancellation detected, cleaning up...\n");
        cleanup_all_resources(ctx);
        pthread_exit(PTHREAD_CANCELED);
    }
    
    // Check custom cancellation function
    if (ctx->custom_cancel_check && ctx->custom_cancel_check()) {
        printf("Custom cancellation condition met\n");
        atomic_store(&ctx->cancel_requested, true);
        cleanup_all_resources(ctx);
        pthread_exit(PTHREAD_CANCELED);
    }
    
    return false;
}

// Request cancellation
void request_cancellation(pthread_t thread) {
    // This is a simplified approach - in practice, you'd maintain a registry
    // of cancellation contexts indexed by thread ID
    printf("Cancellation requested for thread %lu\n", (unsigned long)thread);
    pthread_cancel(thread);
}

// Set cancellation type
void set_cancellation_type(cancellation_type_t type) {
    cancellation_context_t* ctx = get_cancellation_context();
    if (!ctx) return;
    
    ctx->cancel_type = type;
    
    switch (type) {
        case CANCEL_DEFERRED:
            pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);
            break;
        case CANCEL_ASYNCHRONOUS:
            pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
            break;
        case CANCEL_COOPERATIVE:
            pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);
            break;
        default:
            pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
            break;
    }
}

// Robust thread with comprehensive cancellation
void* robust_thread_function(void* arg) {
    printf("Robust thread started\n");
    
    // Initialize cancellation context
    cancellation_context_t* ctx = get_cancellation_context();
    set_cancellation_type(CANCEL_COOPERATIVE);
    
    // Example: Allocate and register resources
    FILE* file = fopen("test_file.txt", "w");
    if (file) {
        register_resource(file, (void(*)(void*))fclose, "test_file");
    }
    
    void* buffer = malloc(1024);
    if (buffer) {
        register_resource(buffer, free, "buffer");
    }
    
    pthread_mutex_t* mutex = malloc(sizeof(pthread_mutex_t));
    if (mutex) {
        pthread_mutex_init(mutex, NULL);
        register_resource(mutex, (void(*)(void*))pthread_mutex_destroy, "mutex");
    }
    
    // Simulate work with regular cancellation checks
    for (int i = 0; i < 1000; i++) {
        // Check for cancellation every 10 iterations
        if (i % 10 == 0) {
            check_cancellation();
        }
        
        // Simulate work
        if (file) {
            fprintf(file, "Iteration %d\n", i);
            fflush(file);
        }
        
        if (buffer) {
            sprintf((char*)buffer, "Work item %d", i);
        }
        
        // Simulate varying work duration
        usleep((rand() % 10 + 1) * 1000); // 1-10ms
    }
    
    // Normal cleanup
    printf("Thread completing normally\n");
    
    unregister_resource(file);
    unregister_resource(buffer);
    unregister_resource(mutex);
    
    if (file) fclose(file);
    if (buffer) free(buffer);
    if (mutex) {
        pthread_mutex_destroy(mutex);
        free(mutex);
    }
    
    printf("Robust thread completed\n");
    return NULL;
}

// Test robust cancellation
void test_robust_cancellation() {
    printf("=== Robust Cancellation Test ===\n");
    
    pthread_t thread;
    if (pthread_create(&thread, NULL, robust_thread_function, NULL) != 0) {
        printf("Failed to create thread\n");
        return;
    }
    
    // Let thread run for a while
    sleep(2);
    
    // Cancel the thread
    printf("Requesting thread cancellation...\n");
    request_cancellation(thread);
    
    // Wait for thread to complete
    void* result;
    if (pthread_join(thread, &result) == 0) {
        if (result == PTHREAD_CANCELED) {
            printf("Thread was successfully cancelled\n");
        } else {
            printf("Thread completed normally\n");
        }
    }
}
```

### RAII-Style Resource Management

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// RAII-style resource wrapper
typedef struct {
    void* resource;
    void (*destructor)(void*);
    bool is_valid;
} raii_resource_t;

// RAII constructor
raii_resource_t raii_create(void* resource, void (*destructor)(void*)) {
    raii_resource_t raii = {
        .resource = resource,
        .destructor = destructor,
        .is_valid = (resource != NULL && destructor != NULL)
    };
    return raii;
}

// RAII destructor
void raii_destroy(raii_resource_t* raii) {
    if (raii && raii->is_valid && raii->resource) {
        raii->destructor(raii->resource);
        raii->resource = NULL;
        raii->is_valid = false;
    }
}

// RAII move semantics
raii_resource_t raii_move(raii_resource_t* src) {
    raii_resource_t moved = *src;
    src->resource = NULL;
    src->is_valid = false;
    return moved;
}

// RAII scope guard macro
#define RAII_SCOPE_GUARD(var, resource, destructor) \
    raii_resource_t var = raii_create(resource, destructor); \
    __attribute__((cleanup(raii_destroy))) raii_resource_t* var##_guard = &var

// Example usage with automatic cleanup
void raii_example_function() {
    // File with automatic cleanup
    FILE* file = fopen("raii_test.txt", "w");
    RAII_SCOPE_GUARD(file_guard, file, (void(*)(void*))fclose);
    
    // Memory with automatic cleanup
    void* buffer = malloc(1024);
    RAII_SCOPE_GUARD(buffer_guard, buffer, free);
    
    // Mutex with automatic cleanup
    pthread_mutex_t* mutex = malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(mutex, NULL);
    RAII_SCOPE_GUARD(mutex_guard, mutex, (void(*)(void*))pthread_mutex_destroy);
    
    // Work with resources
    if (file && buffer && mutex) {
        fprintf(file, "RAII test successful\n");
        sprintf((char*)buffer, "RAII buffer test");
        
        pthread_mutex_lock(mutex);
        // Critical section
        pthread_mutex_unlock(mutex);
    }
    
    // Resources are automatically cleaned up when function exits
    // due to RAII scope guards
}
```

## High-Performance Lock-Free Data Structures

### Lock-Free Stack with ABA Problem Prevention

Lock-free data structures eliminate the overhead of locking while maintaining thread safety through atomic operations and careful memory ordering.

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

// ABA problem prevention using tagged pointers
typedef struct {
    void* ptr;
    uintptr_t tag;
} tagged_pointer_t;

typedef struct stack_node {
    void* data;
    struct stack_node* next;
} stack_node_t;

typedef struct {
    atomic_uintptr_t head; // Tagged pointer to prevent ABA
    atomic_ulong push_count;
    atomic_ulong pop_count;
    atomic_ulong aba_prevention_count;
} lock_free_stack_t;

// Helper functions for tagged pointers
static inline tagged_pointer_t make_tagged_pointer(void* ptr, uintptr_t tag) {
    return (tagged_pointer_t){.ptr = ptr, .tag = tag};
}

static inline uintptr_t pack_tagged_pointer(tagged_pointer_t tp) {
    // Pack pointer and tag into single word (assumes 48-bit pointers)
    return ((uintptr_t)tp.ptr & 0x0000FFFFFFFFFFFF) | (tp.tag << 48);
}

static inline tagged_pointer_t unpack_tagged_pointer(uintptr_t packed) {
    return (tagged_pointer_t){
        .ptr = (void*)(packed & 0x0000FFFFFFFFFFFF),
        .tag = packed >> 48
    };
}

// Initialize lock-free stack
void lf_stack_init(lock_free_stack_t* stack) {
    if (!stack) return;
    
    atomic_store(&stack->head, 0);
    atomic_store(&stack->push_count, 0);
    atomic_store(&stack->pop_count, 0);
    atomic_store(&stack->aba_prevention_count, 0);
}

// Push item onto lock-free stack
bool lf_stack_push(lock_free_stack_t* stack, void* data) {
    if (!stack || !data) return false;
    
    stack_node_t* new_node = malloc(sizeof(stack_node_t));
    if (!new_node) return false;
    
    new_node->data = data;
    
    uintptr_t old_head_packed;
    tagged_pointer_t old_head, new_head;
    
    do {
        old_head_packed = atomic_load(&stack->head);
        old_head = unpack_tagged_pointer(old_head_packed);
        
        new_node->next = (stack_node_t*)old_head.ptr;
        new_head = make_tagged_pointer(new_node, old_head.tag + 1);
        
    } while (!atomic_compare_exchange_weak(&stack->head, &old_head_packed, 
                                          pack_tagged_pointer(new_head)));
    
    atomic_fetch_add(&stack->push_count, 1);
    return true;
}

// Pop item from lock-free stack
void* lf_stack_pop(lock_free_stack_t* stack) {
    if (!stack) return NULL;
    
    uintptr_t old_head_packed;
    tagged_pointer_t old_head, new_head;
    stack_node_t* node;
    void* data;
    
    do {
        old_head_packed = atomic_load(&stack->head);
        old_head = unpack_tagged_pointer(old_head_packed);
        node = (stack_node_t*)old_head.ptr;
        
        if (!node) {
            return NULL; // Stack is empty
        }
        
        new_head = make_tagged_pointer(node->next, old_head.tag + 1);
        
    } while (!atomic_compare_exchange_weak(&stack->head, &old_head_packed,
                                          pack_tagged_pointer(new_head)));
    
    data = node->data;
    free(node);
    
    atomic_fetch_add(&stack->pop_count, 1);
    return data;
}

// Check if stack is empty (snapshot in time)
bool lf_stack_is_empty(lock_free_stack_t* stack) {
    if (!stack) return true;
    
    uintptr_t head_packed = atomic_load(&stack->head);
    tagged_pointer_t head = unpack_tagged_pointer(head_packed);
    return head.ptr == NULL;
}

// Get stack statistics
void lf_stack_get_stats(lock_free_stack_t* stack, unsigned long* pushes, 
                       unsigned long* pops, unsigned long* aba_preventions) {
    if (!stack) return;
    
    if (pushes) *pushes = atomic_load(&stack->push_count);
    if (pops) *pops = atomic_load(&stack->pop_count);
    if (aba_preventions) *aba_preventions = atomic_load(&stack->aba_prevention_count);
}
```

### Lock-Free Queue (Michael & Scott Algorithm)

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stdbool.h>

typedef struct queue_node {
    atomic_uintptr_t data; // Tagged pointer
    atomic_uintptr_t next; // Tagged pointer
} queue_node_t;

typedef struct {
    atomic_uintptr_t head; // Tagged pointer
    atomic_uintptr_t tail; // Tagged pointer
    atomic_ulong enqueue_count;
    atomic_ulong dequeue_count;
    atomic_ulong memory_reclamation_count;
} lock_free_queue_t;

// Hazard pointer system for memory reclamation
#define MAX_THREADS 64
#define HAZARD_POINTERS_PER_THREAD 3

typedef struct {
    atomic_uintptr_t hazard_pointers[MAX_THREADS][HAZARD_POINTERS_PER_THREAD];
    atomic_uintptr_t retired_nodes[MAX_THREADS * 100]; // Retired node list
    atomic_int retired_count[MAX_THREADS];
} hazard_pointer_system_t;

static hazard_pointer_system_t hp_system = {0};
__thread int thread_id = -1;

// Initialize lock-free queue
bool lf_queue_init(lock_free_queue_t* queue) {
    if (!queue) return false;
    
    // Create dummy node
    queue_node_t* dummy = malloc(sizeof(queue_node_t));
    if (!dummy) return false;
    
    atomic_store(&dummy->data, 0);
    atomic_store(&dummy->next, 0);
    
    uintptr_t dummy_packed = pack_tagged_pointer(make_tagged_pointer(dummy, 0));
    atomic_store(&queue->head, dummy_packed);
    atomic_store(&queue->tail, dummy_packed);
    atomic_store(&queue->enqueue_count, 0);
    atomic_store(&queue->dequeue_count, 0);
    atomic_store(&queue->memory_reclamation_count, 0);
    
    return true;
}

// Set hazard pointer
void set_hazard_pointer(int thread_id, int index, void* ptr) {
    if (thread_id >= 0 && thread_id < MAX_THREADS && 
        index >= 0 && index < HAZARD_POINTERS_PER_THREAD) {
        atomic_store(&hp_system.hazard_pointers[thread_id][index], (uintptr_t)ptr);
    }
}

// Check if pointer is hazardous
bool is_hazardous(void* ptr) {
    for (int i = 0; i < MAX_THREADS; i++) {
        for (int j = 0; j < HAZARD_POINTERS_PER_THREAD; j++) {
            if (atomic_load(&hp_system.hazard_pointers[i][j]) == (uintptr_t)ptr) {
                return true;
            }
        }
    }
    return false;
}

// Retire node for later reclamation
void retire_node(queue_node_t* node) {
    if (thread_id < 0 || thread_id >= MAX_THREADS) return;
    
    int retired_index = atomic_fetch_add(&hp_system.retired_count[thread_id], 1);
    if (retired_index < MAX_THREADS * 100) {
        atomic_store(&hp_system.retired_nodes[thread_id * 100 + retired_index], (uintptr_t)node);
    }
    
    // Attempt to reclaim retired nodes
    if (retired_index > 50) { // Threshold for reclamation attempt
        for (int i = 0; i < retired_index; i++) {
            uintptr_t node_ptr = atomic_load(&hp_system.retired_nodes[thread_id * 100 + i]);
            if (node_ptr && !is_hazardous((void*)node_ptr)) {
                free((void*)node_ptr);
                atomic_store(&hp_system.retired_nodes[thread_id * 100 + i], 0);
                atomic_fetch_add(&hp_system.memory_reclamation_count, 1);
            }
        }
    }
}

// Enqueue operation
bool lf_queue_enqueue(lock_free_queue_t* queue, void* data) {
    if (!queue || !data) return false;
    
    queue_node_t* new_node = malloc(sizeof(queue_node_t));
    if (!new_node) return false;
    
    atomic_store(&new_node->data, pack_tagged_pointer(make_tagged_pointer(data, 0)));
    atomic_store(&new_node->next, 0);
    
    while (true) {
        uintptr_t tail_packed = atomic_load(&queue->tail);
        tagged_pointer_t tail_tp = unpack_tagged_pointer(tail_packed);
        queue_node_t* tail = (queue_node_t*)tail_tp.ptr;
        
        // Set hazard pointer
        set_hazard_pointer(thread_id, 0, tail);
        
        // Verify tail hasn't changed
        if (atomic_load(&queue->tail) != tail_packed) {
            continue;
        }
        
        uintptr_t next_packed = atomic_load(&tail->next);
        tagged_pointer_t next_tp = unpack_tagged_pointer(next_packed);
        
        if (next_tp.ptr == NULL) {
            // Try to link new node at end of list
            tagged_pointer_t new_next = make_tagged_pointer(new_node, next_tp.tag + 1);
            if (atomic_compare_exchange_weak(&tail->next, &next_packed, 
                                           pack_tagged_pointer(new_next))) {
                // Successfully linked new node, now move tail
                tagged_pointer_t new_tail = make_tagged_pointer(new_node, tail_tp.tag + 1);
                atomic_compare_exchange_weak(&queue->tail, &tail_packed, 
                                           pack_tagged_pointer(new_tail));
                break;
            }
        } else {
            // Help move tail forward
            tagged_pointer_t new_tail = make_tagged_pointer(next_tp.ptr, tail_tp.tag + 1);
            atomic_compare_exchange_weak(&queue->tail, &tail_packed, 
                                       pack_tagged_pointer(new_tail));
        }
    }
    
    // Clear hazard pointer
    set_hazard_pointer(thread_id, 0, NULL);
    
    atomic_fetch_add(&queue->enqueue_count, 1);
    return true;
}

// Dequeue operation
void* lf_queue_dequeue(lock_free_queue_t* queue) {
    if (!queue) return NULL;
    
    while (true) {
        uintptr_t head_packed = atomic_load(&queue->head);
        tagged_pointer_t head_tp = unpack_tagged_pointer(head_packed);
        queue_node_t* head = (queue_node_t*)head_tp.ptr;
        
        uintptr_t tail_packed = atomic_load(&queue->tail);
        tagged_pointer_t tail_tp = unpack_tagged_pointer(tail_packed);
        
        // Set hazard pointers
        set_hazard_pointer(thread_id, 0, head);
        
        // Verify head hasn't changed
        if (atomic_load(&queue->head) != head_packed) {
            continue;
        }
        
        uintptr_t next_packed = atomic_load(&head->next);
        tagged_pointer_t next_tp = unpack_tagged_pointer(next_packed);
        queue_node_t* next = (queue_node_t*)next_tp.ptr;
        
        if (head == tail_tp.ptr) {
            if (next == NULL) {
                // Queue is empty
                set_hazard_pointer(thread_id, 0, NULL);
                return NULL;
            }
            
            // Help move tail forward
            tagged_pointer_t new_tail = make_tagged_pointer(next, tail_tp.tag + 1);
            atomic_compare_exchange_weak(&queue->tail, &tail_packed, 
                                       pack_tagged_pointer(new_tail));
        } else {
            if (next == NULL) {
                continue; // Inconsistent state, retry
            }
            
            // Set hazard pointer for next node
            set_hazard_pointer(thread_id, 1, next);
            
            // Read data before moving head
            uintptr_t data_packed = atomic_load(&next->data);
            tagged_pointer_t data_tp = unpack_tagged_pointer(data_packed);
            
            // Try to move head forward
            tagged_pointer_t new_head = make_tagged_pointer(next, head_tp.tag + 1);
            if (atomic_compare_exchange_weak(&queue->head, &head_packed,
                                           pack_tagged_pointer(new_head))) {
                // Successfully dequeued
                void* data = data_tp.ptr;
                
                // Retire old head node
                retire_node(head);
                
                // Clear hazard pointers
                set_hazard_pointer(thread_id, 0, NULL);
                set_hazard_pointer(thread_id, 1, NULL);
                
                atomic_fetch_add(&queue->dequeue_count, 1);
                return data;
            }
        }
        
        // Clear hazard pointers before retry
        set_hazard_pointer(thread_id, 0, NULL);
        set_hazard_pointer(thread_id, 1, NULL);
    }
}

// Check if queue is empty (snapshot in time)
bool lf_queue_is_empty(lock_free_queue_t* queue) {
    if (!queue) return true;
    
    uintptr_t head_packed = atomic_load(&queue->head);
    uintptr_t tail_packed = atomic_load(&queue->tail);
    
    tagged_pointer_t head_tp = unpack_tagged_pointer(head_packed);
    tagged_pointer_t tail_tp = unpack_tagged_pointer(tail_packed);
    
    if (head_tp.ptr == tail_tp.ptr) {
        queue_node_t* head = (queue_node_t*)head_tp.ptr;
        uintptr_t next_packed = atomic_load(&head->next);
        tagged_pointer_t next_tp = unpack_tagged_pointer(next_packed);
        return next_tp.ptr == NULL;
    }
    
    return false;
}

// Get queue statistics
void lf_queue_get_stats(lock_free_queue_t* queue, unsigned long* enqueues,
                       unsigned long* dequeues, unsigned long* reclamations) {
    if (!queue) return;
    
    if (enqueues) *enqueues = atomic_load(&queue->enqueue_count);
    if (dequeues) *dequeues = atomic_load(&queue->dequeue_count);
    if (reclamations) *reclamations = atomic_load(&queue->memory_reclamation_count);
}
```

### Lock-Free Hash Table

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <stdbool.h>

#define HASH_TABLE_SIZE 1024
#define MAX_PROBE_DISTANCE 64

typedef struct {
    atomic_uintptr_t key;   // Tagged pointer or direct value
    atomic_uintptr_t value; // Tagged pointer or direct value
    atomic_int probe_distance;
} hash_entry_t;

typedef struct {
    hash_entry_t entries[HASH_TABLE_SIZE];
    atomic_ulong insert_count;
    atomic_ulong lookup_count;
    atomic_ulong delete_count;
    atomic_ulong collision_count;
} lock_free_hash_table_t;

// Hash function (FNV-1a)
uint32_t hash_key(const void* key, size_t len) {
    const uint8_t* data = (const uint8_t*)key;
    uint32_t hash = 2166136261u;
    
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    
    return hash;
}

// Initialize hash table
void lf_hash_init(lock_free_hash_table_t* table) {
    if (!table) return;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        atomic_store(&table->entries[i].key, 0);
        atomic_store(&table->entries[i].value, 0);
        atomic_store(&table->entries[i].probe_distance, -1);
    }
    
    atomic_store(&table->insert_count, 0);
    atomic_store(&table->lookup_count, 0);
    atomic_store(&table->delete_count, 0);
    atomic_store(&table->collision_count, 0);
}

// Insert key-value pair
bool lf_hash_insert(lock_free_hash_table_t* table, uintptr_t key, uintptr_t value) {
    if (!table || key == 0) return false;
    
    uint32_t hash = hash_key(&key, sizeof(key));
    int start_index = hash % HASH_TABLE_SIZE;
    
    for (int probe = 0; probe < MAX_PROBE_DISTANCE; probe++) {
        int index = (start_index + probe) % HASH_TABLE_SIZE;
        hash_entry_t* entry = &table->entries[index];
        
        // Try to claim empty slot
        uintptr_t expected_key = 0;
        if (atomic_compare_exchange_strong(&entry->key, &expected_key, key)) {
            // Successfully claimed slot
            atomic_store(&entry->value, value);
            atomic_store(&entry->probe_distance, probe);
            atomic_fetch_add(&table->insert_count, 1);
            
            if (probe > 0) {
                atomic_fetch_add(&table->collision_count, 1);
            }
            
            return true;
        }
        
        // Check if key already exists
        uintptr_t existing_key = atomic_load(&entry->key);
        if (existing_key == key) {
            // Update existing value
            atomic_store(&entry->value, value);
            return true;
        }
    }
    
    return false; // Table full or max probe distance reached
}

// Lookup value by key
bool lf_hash_lookup(lock_free_hash_table_t* table, uintptr_t key, uintptr_t* value) {
    if (!table || key == 0 || !value) return false;
    
    atomic_fetch_add(&table->lookup_count, 1);
    
    uint32_t hash = hash_key(&key, sizeof(key));
    int start_index = hash % HASH_TABLE_SIZE;
    
    for (int probe = 0; probe < MAX_PROBE_DISTANCE; probe++) {
        int index = (start_index + probe) % HASH_TABLE_SIZE;
        hash_entry_t* entry = &table->entries[index];
        
        uintptr_t stored_key = atomic_load(&entry->key);
        int probe_distance = atomic_load(&entry->probe_distance);
        
        if (stored_key == 0 || probe_distance < probe) {
            // Key not found
            return false;
        }
        
        if (stored_key == key) {
            // Found key
            *value = atomic_load(&entry->value);
            return true;
        }
    }
    
    return false;
}

// Delete key-value pair
bool lf_hash_delete(lock_free_hash_table_t* table, uintptr_t key) {
    if (!table || key == 0) return false;
    
    uint32_t hash = hash_key(&key, sizeof(key));
    int start_index = hash % HASH_TABLE_SIZE;
    
    for (int probe = 0; probe < MAX_PROBE_DISTANCE; probe++) {
        int index = (start_index + probe) % HASH_TABLE_SIZE;
        hash_entry_t* entry = &table->entries[index];
        
        uintptr_t stored_key = atomic_load(&entry->key);
        
        if (stored_key == 0) {
            // Key not found
            return false;
        }
        
        if (stored_key == key) {
            // Found key, mark as deleted
            atomic_store(&entry->key, 0);
            atomic_store(&entry->value, 0);
            atomic_store(&entry->probe_distance, -1);
            atomic_fetch_add(&table->delete_count, 1);
            
            // Note: In a production implementation, you'd need to handle
            // the gap left by deletion to maintain lookup correctness
            return true;
        }
    }
    
    return false;
}

// Get hash table statistics
void lf_hash_get_stats(lock_free_hash_table_t* table, unsigned long* inserts,
                      unsigned long* lookups, unsigned long* deletes,
                      unsigned long* collisions, double* load_factor) {
    if (!table) return;
    
    if (inserts) *inserts = atomic_load(&table->insert_count);
    if (lookups) *lookups = atomic_load(&table->lookup_count);
    if (deletes) *deletes = atomic_load(&table->delete_count);
    if (collisions) *collisions = atomic_load(&table->collision_count);
    
    if (load_factor) {
        unsigned long total_inserts = atomic_load(&table->insert_count);
        unsigned long total_deletes = atomic_load(&table->delete_count);
        *load_factor = (double)(total_inserts - total_deletes) / HASH_TABLE_SIZE;
    }
}
```

## Advanced Synchronization Patterns

Advanced synchronization patterns go beyond basic mutexes and condition variables to address complex scenarios like fairness, starvation prevention, hierarchical locking, and transactional memory. These patterns are essential for building high-performance concurrent systems.

### Comprehensive Reader-Writer Lock with Fairness

A sophisticated reader-writer lock that prevents both reader and writer starvation through fair queuing:

```c
#include <pthread.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <time.h>
#include <errno.h>

typedef enum {
    FAIR_RW_READER_PRIORITY,
    FAIR_RW_WRITER_PRIORITY,
    FAIR_RW_FIFO_PRIORITY
} fair_rw_policy_t;

typedef struct fair_rw_waiter {
    pthread_cond_t cond;
    bool is_writer;
    bool signaled;
    struct fair_rw_waiter* next;
    struct timespec arrival_time;
} fair_rw_waiter_t;

typedef struct {
    // Current state
    atomic_int active_readers;
    atomic_int active_writers;
    
    // Priority policy
    fair_rw_policy_t policy;
    
    // Fair queue management
    pthread_mutex_t queue_mutex;
    fair_rw_waiter_t* queue_head;
    fair_rw_waiter_t* queue_tail;
    
    // Statistics
    atomic_ulong reader_acquisitions;
    atomic_ulong writer_acquisitions;
    atomic_ulong reader_wait_time_ns;
    atomic_ulong writer_wait_time_ns;
    atomic_ulong max_queue_length;
    
    // Configuration
    int max_concurrent_readers;
    struct timespec writer_timeout;
} fair_rwlock_t;

// Initialize fair reader-writer lock
int fair_rwlock_init(fair_rwlock_t* lock, fair_rw_policy_t policy) {
    if (!lock) return EINVAL;
    
    atomic_store(&lock->active_readers, 0);
    atomic_store(&lock->active_writers, 0);
    lock->policy = policy;
    
    if (pthread_mutex_init(&lock->queue_mutex, NULL) != 0) {
        return errno;
    }
    
    lock->queue_head = NULL;
    lock->queue_tail = NULL;
    lock->max_concurrent_readers = INT_MAX;
    lock->writer_timeout.tv_sec = 5;  // 5 second timeout
    lock->writer_timeout.tv_nsec = 0;
    
    // Initialize statistics
    atomic_store(&lock->reader_acquisitions, 0);
    atomic_store(&lock->writer_acquisitions, 0);
    atomic_store(&lock->reader_wait_time_ns, 0);
    atomic_store(&lock->writer_wait_time_ns, 0);
    atomic_store(&lock->max_queue_length, 0);
    
    return 0;
}

// Add waiter to fair queue
static void enqueue_waiter(fair_rwlock_t* lock, fair_rw_waiter_t* waiter) {
    clock_gettime(CLOCK_MONOTONIC, &waiter->arrival_time);
    waiter->signaled = false;
    waiter->next = NULL;
    
    if (pthread_cond_init(&waiter->cond, NULL) != 0) {
        return; // Handle error appropriately
    }
    
    if (lock->queue_tail) {
        lock->queue_tail->next = waiter;
        lock->queue_tail = waiter;
    } else {
        lock->queue_head = lock->queue_tail = waiter;
    }
    
    // Update queue length statistics
    int queue_length = 0;
    fair_rw_waiter_t* current = lock->queue_head;
    while (current) {
        queue_length++;
        current = current->next;
    }
    
    unsigned long max_len = atomic_load(&lock->max_queue_length);
    if (queue_length > max_len) {
        atomic_store(&lock->max_queue_length, queue_length);
    }
}

// Signal next appropriate waiter
static void signal_next_waiter(fair_rwlock_t* lock) {
    if (!lock->queue_head) return;
    
    fair_rw_waiter_t* waiter = lock->queue_head;
    bool can_proceed = false;
    
    switch (lock->policy) {
        case FAIR_RW_READER_PRIORITY:
            // Readers have priority unless a writer is active
            if (!waiter->is_writer) {
                can_proceed = (atomic_load(&lock->active_writers) == 0);
            } else {
                can_proceed = (atomic_load(&lock->active_readers) == 0 && 
                              atomic_load(&lock->active_writers) == 0);
            }
            break;
            
        case FAIR_RW_WRITER_PRIORITY:
            // Writers have priority
            if (waiter->is_writer) {
                can_proceed = (atomic_load(&lock->active_readers) == 0 && 
                              atomic_load(&lock->active_writers) == 0);
            } else {
                // Check if any writers are waiting
                fair_rw_waiter_t* check = lock->queue_head;
                bool writers_waiting = false;
                while (check) {
                    if (check->is_writer) {
                        writers_waiting = true;
                        break;
                    }
                    check = check->next;
                }
                can_proceed = (!writers_waiting && atomic_load(&lock->active_writers) == 0);
            }
            break;
            
        case FAIR_RW_FIFO_PRIORITY:
            // Strict FIFO ordering
            if (!waiter->is_writer) {
                can_proceed = (atomic_load(&lock->active_writers) == 0);
            } else {
                can_proceed = (atomic_load(&lock->active_readers) == 0 && 
                              atomic_load(&lock->active_writers) == 0);
            }
            break;
    }
    
    if (can_proceed) {
        // Remove from queue
        lock->queue_head = waiter->next;
        if (!lock->queue_head) {
            lock->queue_tail = NULL;
        }
        
        waiter->signaled = true;
        pthread_cond_signal(&waiter->cond);
    }
}

// Reader lock acquisition
int fair_rwlock_rdlock(fair_rwlock_t* lock) {
    if (!lock) return EINVAL;
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    pthread_mutex_lock(&lock->queue_mutex);
    
    // Check if we can acquire immediately
    if (atomic_load(&lock->active_writers) == 0 && !lock->queue_head &&
        atomic_load(&lock->active_readers) < lock->max_concurrent_readers) {
        
        atomic_fetch_add(&lock->active_readers, 1);
        atomic_fetch_add(&lock->reader_acquisitions, 1);
        pthread_mutex_unlock(&lock->queue_mutex);
        return 0;
    }
    
    // Need to wait - enqueue ourselves
    fair_rw_waiter_t waiter = { .is_writer = false };
    enqueue_waiter(lock, &waiter);
    
    // Wait for our turn
    while (!waiter.signaled) {
        pthread_cond_wait(&waiter.cond, &lock->queue_mutex);
    }
    
    // We've been signaled - acquire the lock
    atomic_fetch_add(&lock->active_readers, 1);
    atomic_fetch_add(&lock->reader_acquisitions, 1);
    
    // Clean up condition variable
    pthread_cond_destroy(&waiter.cond);
    
    // Signal next waiter if possible
    signal_next_waiter(lock);
    
    pthread_mutex_unlock(&lock->queue_mutex);
    
    // Update wait time statistics
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    unsigned long wait_time_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000UL +
                                (end_time.tv_nsec - start_time.tv_nsec);
    atomic_fetch_add(&lock->reader_wait_time_ns, wait_time_ns);
    
    return 0;
}

// Writer lock acquisition with timeout
int fair_rwlock_wrlock_timed(fair_rwlock_t* lock, const struct timespec* timeout) {
    if (!lock) return EINVAL;
    
    struct timespec start_time, end_time, abs_timeout;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    if (timeout) {
        abs_timeout.tv_sec = start_time.tv_sec + timeout->tv_sec;
        abs_timeout.tv_nsec = start_time.tv_nsec + timeout->tv_nsec;
        if (abs_timeout.tv_nsec >= 1000000000) {
            abs_timeout.tv_sec++;
            abs_timeout.tv_nsec -= 1000000000;
        }
    }
    
    pthread_mutex_lock(&lock->queue_mutex);
    
    // Check if we can acquire immediately
    if (atomic_load(&lock->active_readers) == 0 && 
        atomic_load(&lock->active_writers) == 0 && !lock->queue_head) {
        
        atomic_fetch_add(&lock->active_writers, 1);
        atomic_fetch_add(&lock->writer_acquisitions, 1);
        pthread_mutex_unlock(&lock->queue_mutex);
        return 0;
    }
    
    // Need to wait - enqueue ourselves
    fair_rw_waiter_t waiter = { .is_writer = true };
    enqueue_waiter(lock, &waiter);
    
    int result = 0;
    
    // Wait for our turn with timeout
    while (!waiter.signaled) {
        if (timeout) {
            result = pthread_cond_timedwait(&waiter.cond, &lock->queue_mutex, &abs_timeout);
            if (result == ETIMEDOUT) {
                // Remove ourselves from queue on timeout
                fair_rw_waiter_t** current = &lock->queue_head;
                while (*current && *current != &waiter) {
                    current = &(*current)->next;
                }
                if (*current) {
                    *current = waiter.next;
                    if (lock->queue_tail == &waiter) {
                        lock->queue_tail = NULL;
                    }
                }
                pthread_cond_destroy(&waiter.cond);
                pthread_mutex_unlock(&lock->queue_mutex);
                return ETIMEDOUT;
            }
        } else {
            pthread_cond_wait(&waiter.cond, &lock->queue_mutex);
        }
    }
    
    // We've been signaled - acquire the lock
    atomic_fetch_add(&lock->active_writers, 1);
    atomic_fetch_add(&lock->writer_acquisitions, 1);
    
    // Clean up condition variable
    pthread_cond_destroy(&waiter.cond);
    
    pthread_mutex_unlock(&lock->queue_mutex);
    
    // Update wait time statistics
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    unsigned long wait_time_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000UL +
                                (end_time.tv_nsec - start_time.tv_nsec);
    atomic_fetch_add(&lock->writer_wait_time_ns, wait_time_ns);
    
    return 0;
}

// Writer lock acquisition (blocking)
int fair_rwlock_wrlock(fair_rwlock_t* lock) {
    return fair_rwlock_wrlock_timed(lock, NULL);
}

// Reader unlock
int fair_rwlock_rdunlock(fair_rwlock_t* lock) {
    if (!lock) return EINVAL;
    
    atomic_fetch_sub(&lock->active_readers, 1);
    
    pthread_mutex_lock(&lock->queue_mutex);
    signal_next_waiter(lock);
    pthread_mutex_unlock(&lock->queue_mutex);
    
    return 0;
}

// Writer unlock
int fair_rwlock_wrunlock(fair_rwlock_t* lock) {
    if (!lock) return EINVAL;
    
    atomic_fetch_sub(&lock->active_writers, 1);
    
    pthread_mutex_lock(&lock->queue_mutex);
    
    // Signal all compatible waiters (multiple readers can proceed)
    while (lock->queue_head && !lock->queue_head->is_writer && 
           atomic_load(&lock->active_writers) == 0) {
        signal_next_waiter(lock);
    }
    
    // Or signal a single writer if no readers
    if (lock->queue_head && lock->queue_head->is_writer && 
        atomic_load(&lock->active_readers) == 0) {
        signal_next_waiter(lock);
    }
    
    pthread_mutex_unlock(&lock->queue_mutex);
    
    return 0;
}

// Get lock statistics
void fair_rwlock_get_stats(fair_rwlock_t* lock, 
                          unsigned long* reader_acq, unsigned long* writer_acq,
                          double* avg_reader_wait_ms, double* avg_writer_wait_ms,
                          unsigned long* max_queue_len) {
    if (!lock) return;
    
    if (reader_acq) *reader_acq = atomic_load(&lock->reader_acquisitions);
    if (writer_acq) *writer_acq = atomic_load(&lock->writer_acquisitions);
    if (max_queue_len) *max_queue_len = atomic_load(&lock->max_queue_length);
    
    if (avg_reader_wait_ms) {
        unsigned long total_wait = atomic_load(&lock->reader_wait_time_ns);
        unsigned long acquisitions = atomic_load(&lock->reader_acquisitions);
        *avg_reader_wait_ms = acquisitions > 0 ? 
            (double)total_wait / (acquisitions * 1000000.0) : 0.0;
    }
    
    if (avg_writer_wait_ms) {
        unsigned long total_wait = atomic_load(&lock->writer_wait_time_ns);
        unsigned long acquisitions = atomic_load(&lock->writer_acquisitions);
        *avg_writer_wait_ms = acquisitions > 0 ? 
            (double)total_wait / (acquisitions * 1000000.0) : 0.0;
    }
}
```

### Hierarchical Locking with Deadlock Prevention

Advanced locking system that prevents deadlocks through lock ordering:

```c
#include <limits.h>

#define MAX_LOCK_LEVELS 16
#define INVALID_LOCK_LEVEL UINT_MAX

typedef struct hierarchical_mutex {
    pthread_mutex_t mutex;
    unsigned int level;
    atomic_bool is_locked;
    atomic_int owner_thread_id;
    
    // Statistics
    atomic_ulong acquisition_count;
    atomic_ulong contention_count;
    struct timespec last_acquired;
} hierarchical_mutex_t;

// Thread-local storage for current lock level
__thread unsigned int current_lock_level = UINT_MAX;
__thread hierarchical_mutex_t* held_locks[MAX_LOCK_LEVELS];
__thread int num_held_locks = 0;

// Initialize hierarchical mutex
int hierarchical_mutex_init(hierarchical_mutex_t* hmutex, unsigned int level) {
    if (!hmutex || level == INVALID_LOCK_LEVEL) return EINVAL;
    
    if (pthread_mutex_init(&hmutex->mutex, NULL) != 0) {
        return errno;
    }
    
    hmutex->level = level;
    atomic_store(&hmutex->is_locked, false);
    atomic_store(&hmutex->owner_thread_id, -1);
    atomic_store(&hmutex->acquisition_count, 0);
    atomic_store(&hmutex->contention_count, 0);
    
    return 0;
}

// Lock with hierarchy checking
int hierarchical_mutex_lock(hierarchical_mutex_t* hmutex) {
    if (!hmutex) return EINVAL;
    
    // Check hierarchy constraint
    if (hmutex->level >= current_lock_level) {
        fprintf(stderr, "Lock hierarchy violation: trying to acquire level %u "
                       "while holding level %u\n", hmutex->level, current_lock_level);
        return EDEADLK;
    }
    
    // Check for contention
    bool was_locked = atomic_load(&hmutex->is_locked);
    if (was_locked) {
        atomic_fetch_add(&hmutex->contention_count, 1);
    }
    
    // Acquire the lock
    int result = pthread_mutex_lock(&hmutex->mutex);
    if (result != 0) return result;
    
    // Update state
    atomic_store(&hmutex->is_locked, true);
    atomic_store(&hmutex->owner_thread_id, gettid());
    atomic_fetch_add(&hmutex->acquisition_count, 1);
    clock_gettime(CLOCK_MONOTONIC, &hmutex->last_acquired);
    
    // Update thread-local state
    current_lock_level = hmutex->level;
    held_locks[num_held_locks++] = hmutex;
    
    return 0;
}

// Unlock and restore previous level
int hierarchical_mutex_unlock(hierarchical_mutex_t* hmutex) {
    if (!hmutex) return EINVAL;
    
    // Verify we own this lock
    if (atomic_load(&hmutex->owner_thread_id) != gettid()) {
        return EPERM;
    }
    
    // Update state
    atomic_store(&hmutex->is_locked, false);
    atomic_store(&hmutex->owner_thread_id, -1);
    
    // Update thread-local state
    num_held_locks--;
    current_lock_level = (num_held_locks > 0) ? 
        held_locks[num_held_locks - 1]->level : UINT_MAX;
    
    return pthread_mutex_unlock(&hmutex->mutex);
}

// Try lock with hierarchy checking
int hierarchical_mutex_trylock(hierarchical_mutex_t* hmutex) {
    if (!hmutex) return EINVAL;
    
    // Check hierarchy constraint
    if (hmutex->level >= current_lock_level) {
        return EDEADLK;
    }
    
    // Try to acquire the lock
    int result = pthread_mutex_trylock(&hmutex->mutex);
    if (result == 0) {
        // Successfully acquired
        atomic_store(&hmutex->is_locked, true);
        atomic_store(&hmutex->owner_thread_id, gettid());
        atomic_fetch_add(&hmutex->acquisition_count, 1);
        clock_gettime(CLOCK_MONOTONIC, &hmutex->last_acquired);
        
        // Update thread-local state
        current_lock_level = hmutex->level;
        held_locks[num_held_locks++] = hmutex;
    }
    
    return result;
}
```

### Software Transactional Memory (STM) Implementation

A simple STM system for composable atomic operations:

```c
#include <setjmp.h>

#define MAX_STM_VARS 1024
#define MAX_TRANSACTION_DEPTH 16

typedef struct stm_var {
    atomic_uintptr_t value;
    atomic_ulong version;
    pthread_mutex_t write_lock;
} stm_var_t;

typedef struct stm_log_entry {
    stm_var_t* var;
    uintptr_t old_value;
    uintptr_t new_value;
    unsigned long version;
    bool is_write;
} stm_log_entry_t;

typedef struct stm_transaction {
    stm_log_entry_t read_log[MAX_STM_VARS];
    stm_log_entry_t write_log[MAX_STM_VARS];
    int read_count;
    int write_count;
    
    // Retry mechanism
    jmp_buf retry_point;
    int retry_count;
    int max_retries;
    
    // Statistics
    struct timespec start_time;
    unsigned long validation_count;
    unsigned long abort_count;
} stm_transaction_t;

// Thread-local transaction context
__thread stm_transaction_t* current_transaction = NULL;
__thread stm_transaction_t transaction_stack[MAX_TRANSACTION_DEPTH];
__thread int transaction_depth = 0;

// Global STM statistics
static atomic_ulong global_commits = 0;
static atomic_ulong global_aborts = 0;
static atomic_ulong global_retries = 0;

// Initialize STM variable
void stm_var_init(stm_var_t* var, uintptr_t initial_value) {
    if (!var) return;
    
    atomic_store(&var->value, initial_value);
    atomic_store(&var->version, 0);
    pthread_mutex_init(&var->write_lock, NULL);
}

// Start transaction
stm_transaction_t* stm_start() {
    if (transaction_depth >= MAX_TRANSACTION_DEPTH) {
        return NULL; // Transaction depth exceeded
    }
    
    stm_transaction_t* tx = &transaction_stack[transaction_depth++];
    
    tx->read_count = 0;
    tx->write_count = 0;
    tx->retry_count = 0;
    tx->max_retries = 100;
    tx->validation_count = 0;
    tx->abort_count = 0;
    
    clock_gettime(CLOCK_MONOTONIC, &tx->start_time);
    
    current_transaction = tx;
    
    // Set retry point
    if (setjmp(tx->retry_point) != 0) {
        // We've returned from a retry
        tx->retry_count++;
        tx->read_count = 0;
        tx->write_count = 0;
        
        if (tx->retry_count > tx->max_retries) {
            // Too many retries - abort transaction
            current_transaction = (transaction_depth > 1) ? 
                &transaction_stack[transaction_depth - 2] : NULL;
            transaction_depth--;
            atomic_fetch_add(&global_aborts, 1);
            return NULL;
        }
        
        // Brief backoff before retry
        usleep(tx->retry_count * 1000);
    }
    
    return tx;
}

// Read STM variable
uintptr_t stm_read(stm_var_t* var) {
    if (!var || !current_transaction) {
        return atomic_load(&var->value); // Non-transactional read
    }
    
    stm_transaction_t* tx = current_transaction;
    
    // Check if we've already read this variable
    for (int i = 0; i < tx->read_count; i++) {
        if (tx->read_log[i].var == var) {
            return tx->read_log[i].old_value;
        }
    }
    
    // Check if we've written to this variable
    for (int i = 0; i < tx->write_count; i++) {
        if (tx->write_log[i].var == var) {
            return tx->write_log[i].new_value;
        }
    }
    
    // Perform consistent read
    unsigned long version = atomic_load(&var->version);
    uintptr_t value = atomic_load(&var->value);
    
    // Verify version hasn't changed
    if (atomic_load(&var->version) != version) {
        // Conflict detected - retry transaction
        atomic_fetch_add(&global_retries, 1);
        longjmp(tx->retry_point, 1);
    }
    
    // Add to read log
    if (tx->read_count < MAX_STM_VARS) {
        tx->read_log[tx->read_count].var = var;
        tx->read_log[tx->read_count].old_value = value;
        tx->read_log[tx->read_count].version = version;
        tx->read_log[tx->read_count].is_write = false;
        tx->read_count++;
    }
    
    return value;
}

// Write STM variable
void stm_write(stm_var_t* var, uintptr_t new_value) {
    if (!var || !current_transaction) {
        atomic_store(&var->value, new_value); // Non-transactional write
        return;
    }
    
    stm_transaction_t* tx = current_transaction;
    
    // Check if we've already written to this variable
    for (int i = 0; i < tx->write_count; i++) {
        if (tx->write_log[i].var == var) {
            tx->write_log[i].new_value = new_value;
            return;
        }
    }
    
    // Read current value for validation
    uintptr_t old_value = stm_read(var);
    
    // Add to write log
    if (tx->write_count < MAX_STM_VARS) {
        tx->write_log[tx->write_count].var = var;
        tx->write_log[tx->write_count].old_value = old_value;
        tx->write_log[tx->write_count].new_value = new_value;
        tx->write_log[tx->write_count].version = atomic_load(&var->version);
        tx->write_log[tx->write_count].is_write = true;
        tx->write_count++;
    }
}

// Validate transaction
static bool stm_validate(stm_transaction_t* tx) {
    tx->validation_count++;
    
    // Check all read variables for consistency
    for (int i = 0; i < tx->read_count; i++) {
        stm_log_entry_t* entry = &tx->read_log[i];
        
        // Check if version has changed
        if (atomic_load(&entry->var->version) != entry->version) {
            return false;
        }
        
        // Check if value has changed
        if (atomic_load(&entry->var->value) != entry->old_value) {
            return false;
        }
    }
    
    return true;
}

// Commit transaction
bool stm_commit() {
    if (!current_transaction) return false;
    
    stm_transaction_t* tx = current_transaction;
    
    // Acquire write locks in sorted order to prevent deadlock
    stm_var_t* locked_vars[MAX_STM_VARS];
    int num_locked = 0;
    
    // Sort write variables by address
    for (int i = 0; i < tx->write_count; i++) {
        stm_var_t* var = tx->write_log[i].var;
        
        // Insert in sorted position
        int pos = 0;
        while (pos < num_locked && locked_vars[pos] < var) {
            pos++;
        }
        
        // Check for duplicates
        if (pos < num_locked && locked_vars[pos] == var) {
            continue;
        }
        
        // Shift and insert
        for (int j = num_locked; j > pos; j--) {
            locked_vars[j] = locked_vars[j-1];
        }
        locked_vars[pos] = var;
        num_locked++;
    }
    
    // Acquire locks
    for (int i = 0; i < num_locked; i++) {
        pthread_mutex_lock(&locked_vars[i]->write_lock);
    }
    
    // Validate again under locks
    if (!stm_validate(tx)) {
        // Release locks and retry
        for (int i = num_locked - 1; i >= 0; i--) {
            pthread_mutex_unlock(&locked_vars[i]->write_lock);
        }
        
        atomic_fetch_add(&global_retries, 1);
        longjmp(tx->retry_point, 1);
    }
    
    // Apply writes
    for (int i = 0; i < tx->write_count; i++) {
        stm_log_entry_t* entry = &tx->write_log[i];
        atomic_store(&entry->var->value, entry->new_value);
        atomic_fetch_add(&entry->var->version, 1);
    }
    
    // Release locks
    for (int i = num_locked - 1; i >= 0; i--) {
        pthread_mutex_unlock(&locked_vars[i]->write_lock);
    }
    
    // Clean up transaction
    current_transaction = (transaction_depth > 1) ? 
        &transaction_stack[transaction_depth - 2] : NULL;
    transaction_depth--;
    
    atomic_fetch_add(&global_commits, 1);
    return true;
}

// Abort transaction
void stm_abort() {
    if (!current_transaction) return;
    
    stm_transaction_t* tx = current_transaction;
    tx->abort_count++;
    
    current_transaction = (transaction_depth > 1) ? 
        &transaction_stack[transaction_depth - 2] : NULL;
    transaction_depth--;
    
    atomic_fetch_add(&global_aborts, 1);
}
```

### Example: Bank Transfer with STM

```c
// Bank account using STM
typedef struct {
    stm_var_t balance;
    int account_id;
} stm_account_t;

// Transfer money between accounts atomically
bool transfer_money(stm_account_t* from, stm_account_t* to, int amount) {
    stm_transaction_t* tx = stm_start();
    if (!tx) return false;
    
    // Read current balances
    int from_balance = (int)stm_read(&from->balance);
    int to_balance = (int)stm_read(&to->balance);
    
    // Check sufficient funds
    if (from_balance < amount) {
        stm_abort();
        return false;
    }
    
    // Perform transfer
    stm_write(&from->balance, from_balance - amount);
    stm_write(&to->balance, to_balance + amount);
    
    // Commit transaction
    return stm_commit();
}

// Example usage
void test_stm_transfer() {
    stm_account_t account1, account2;
    
    stm_var_init(&account1.balance, 1000);
    stm_var_init(&account2.balance, 500);
    account1.account_id = 1;
    account2.account_id = 2;
    
    printf("Before transfer: Account1=%d, Account2=%d\n",
           (int)atomic_load(&account1.balance.value),
           (int)atomic_load(&account2.balance.value));
    
    bool success = transfer_money(&account1, &account2, 200);
    
    printf("Transfer %s\n", success ? "succeeded" : "failed");
    printf("After transfer: Account1=%d, Account2=%d\n",
           (int)atomic_load(&account1.balance.value),
           (int)atomic_load(&account2.balance.value));
}
```
```

## Comprehensive Exercises and Real-World Challenges

These exercises are designed to test your mastery of advanced threading patterns through practical, real-world scenarios. Each exercise includes multiple difficulty levels and comprehensive evaluation criteria.

### Exercise 1: High-Performance Web Server Architecture
**Difficulty: Advanced | Time Estimate: 3-4 days**

Build a complete multi-threaded web server using advanced threading patterns:

#### **Part A: Basic Implementation (Foundation Level)**
```c
// Requirements:
// 1. Thread pool for handling connections
// 2. Work-stealing queue for load balancing
// 3. Lock-free statistics collection
// 4. RAII resource management

typedef struct {
    int socket_fd;
    struct sockaddr_in client_addr;
    struct timespec received_time;
    char* request_buffer;
    size_t buffer_size;
} http_request_t;

typedef struct {
    // Core components
    work_stealing_pool_t* worker_pool;
    lock_free_queue_t* connection_queue;
    
    // Configuration
    int port;
    int max_connections;
    int keepalive_timeout;
    
    // Statistics (lock-free)
    atomic_ulong total_requests;
    atomic_ulong active_connections;
    atomic_ulong bytes_transferred;
    atomic_ulong error_count;
    
    // Server state
    atomic_bool shutdown_requested;
    pthread_t acceptor_thread;
} web_server_t;

// Your implementation should include:
web_server_t* web_server_create(int port, int num_workers);
bool web_server_start(web_server_t* server);
void web_server_stop(web_server_t* server);
void web_server_destroy(web_server_t* server);

// Handler functions to implement:
void handle_http_request(http_request_t* request);
void send_http_response(int client_fd, int status_code, 
                       const char* content_type, const char* body);
```

#### **Part B: Advanced Features (Expert Level)**
```c
// Add these advanced features:

// 1. HTTP pipelining support
typedef struct {
    http_request_t* requests;
    int request_count;
    int capacity;
    bool keep_alive;
    atomic_int pipeline_depth;
} http_pipeline_t;

// 2. Dynamic load balancing
typedef struct {
    double cpu_usage;
    int active_requests;
    int queue_depth;
    struct timespec last_update;
} worker_metrics_t;

// 3. Adaptive thread pool sizing
void web_server_adjust_pool_size(web_server_t* server);

// 4. Connection pooling for upstream services
typedef struct {
    char* host;
    int port;
    int max_connections;
    int timeout_ms;
    connection_pool_t* pool;
} upstream_service_t;

// 5. Real-time monitoring and profiling
typedef struct {
    // Latency histograms
    atomic_ulong latency_buckets[10]; // 0-1ms, 1-5ms, 5-10ms, etc.
    
    // Throughput metrics
    atomic_ulong requests_per_second[60]; // Rolling window
    
    // Error tracking
    atomic_ulong error_types[ERROR_TYPE_COUNT];
    
    // Resource utilization
    double memory_usage_mb;
    double cpu_utilization;
    int fd_count;
} server_metrics_t;
```

#### **Evaluation Criteria:**
- **Correctness (40%)**: Thread safety, proper resource cleanup, error handling
- **Performance (30%)**: Throughput, latency, CPU/memory efficiency
- **Architecture (20%)**: Code organization, extensibility, maintainability  
- **Testing (10%)**: Unit tests, integration tests, load testing

#### **Performance Benchmarks:**
```bash
# Your server should handle:
# - 10,000+ concurrent connections
# - 50,000+ requests per second
# - <1ms average response time for static content
# - <5ms 99th percentile latency
# - Memory usage <100MB for basic server

# Test with:
wrk -t12 -c400 -d30s --latency http://localhost:8080/
ab -n 100000 -c 1000 http://localhost:8080/
```

### Exercise 2: Lock-Free Producer-Consumer with Backpressure
**Difficulty: Expert | Time Estimate: 2-3 days**

Implement a sophisticated producer-consumer system with advanced flow control:

#### **Requirements:**
```c
typedef struct {
    void* data;
    size_t size;
    int priority;            // 0 = highest, higher numbers = lower priority
    struct timespec deadline; // When this message expires
    atomic_int ref_count;    // Reference counting for zero-copy
} message_t;

typedef struct {
    // Multi-priority queues (lock-free)
    lock_free_queue_t* priority_queues[MAX_PRIORITIES];
    
    // Flow control
    atomic_int current_size;
    atomic_int max_size;
    atomic_int producer_blocked_count;
    atomic_int consumer_blocked_count;
    
    // Backpressure mechanism
    atomic_bool backpressure_active;
    double backpressure_threshold; // 0.8 = 80% full
    
    // Statistics
    atomic_ulong messages_produced;
    atomic_ulong messages_consumed;
    atomic_ulong messages_dropped_expired;
    atomic_ulong messages_dropped_full;
    
    // Wait-free notifications
    futex_t producer_futex;
    futex_t consumer_futex;
} priority_queue_system_t;

// Core API to implement:
int pqs_produce(priority_queue_system_t* pqs, message_t* msg, 
               int timeout_ms);
message_t* pqs_consume(priority_queue_system_t* pqs, 
                      int timeout_ms);
void pqs_enable_backpressure(priority_queue_system_t* pqs, 
                            double threshold);
```

#### **Advanced Features:**
```c
// 1. Zero-copy message passing
message_t* message_create_zero_copy(void* data, size_t size);
void message_addref(message_t* msg);
void message_release(message_t* msg);

// 2. Batch operations for better performance
int pqs_produce_batch(priority_queue_system_t* pqs, 
                     message_t** msgs, int count, int timeout_ms);
int pqs_consume_batch(priority_queue_system_t* pqs, 
                     message_t** msgs, int max_count, int timeout_ms);

// 3. Message expiration and cleanup
void pqs_cleanup_expired(priority_queue_system_t* pqs);

// 4. Dynamic priority adjustment
void pqs_adjust_priority(priority_queue_system_t* pqs, 
                        message_t* msg, int new_priority);

// 5. Flow control callbacks
typedef void (*backpressure_callback_t)(priority_queue_system_t* pqs, 
                                       bool active, void* user_data);
void pqs_set_backpressure_callback(priority_queue_system_t* pqs,
                                  backpressure_callback_t callback,
                                  void* user_data);
```

#### **Testing Requirements:**
```c
// Create comprehensive test suite:
void test_basic_produce_consume();
void test_priority_ordering();
void test_backpressure_activation();
void test_message_expiration();
void test_zero_copy_semantics();
void test_concurrent_stress(int num_producers, int num_consumers);
void test_memory_leaks();
void benchmark_throughput();
void benchmark_latency();
```

### Exercise 3: Distributed Actor System Implementation
**Difficulty: Expert | Time Estimate: 4-5 days**

Build a fault-tolerant actor system with supervision trees:

#### **Core Actor Framework:**
```c
typedef struct actor actor_t;
typedef struct actor_message actor_message_t;
typedef struct actor_system actor_system_t;

typedef enum {
    ACTOR_CREATED,
    ACTOR_RUNNING,
    ACTOR_SUSPENDED,
    ACTOR_TERMINATED,
    ACTOR_RESTARTING
} actor_state_t;

typedef enum {
    RESTART_ALWAYS,
    RESTART_ON_FAILURE,
    RESTART_NEVER,
    RESTART_TEMPORARY
} restart_strategy_t;

struct actor {
    // Identity
    char* name;
    int actor_id;
    
    // State management
    atomic_int state;
    void* private_data;
    
    // Message handling
    lock_free_queue_t* mailbox;
    atomic_int message_count;
    
    // Behavior
    void (*message_handler)(actor_t* self, actor_message_t* msg);
    void (*pre_start)(actor_t* self);
    void (*post_stop)(actor_t* self);
    void (*pre_restart)(actor_t* self, const char* reason);
    void (*post_restart)(actor_t* self);
    
    // Supervision
    actor_t* supervisor;
    actor_t** children;
    int child_count;
    int max_children;
    restart_strategy_t restart_strategy;
    int restart_count;
    struct timespec last_restart;
    
    // Threading
    pthread_t thread;
    atomic_bool stop_requested;
    
    // Statistics
    atomic_ulong messages_processed;
    atomic_ulong restart_count_total;
    struct timespec created_time;
    struct timespec last_message_time;
};

struct actor_message {
    char* type;
    void* payload;
    size_t payload_size;
    actor_t* sender;
    actor_t* recipient;
    struct timespec timestamp;
    int priority;
    
    // Response handling
    bool expects_reply;
    int correlation_id;
    struct timespec timeout;
};

// Core API to implement:
actor_system_t* actor_system_create(const char* name);
actor_t* actor_create(actor_system_t* system, const char* name,
                     void (*message_handler)(actor_t*, actor_message_t*));
bool actor_send(actor_t* actor, actor_message_t* msg);
actor_message_t* actor_send_and_wait(actor_t* actor, actor_message_t* msg,
                                   int timeout_ms);
void actor_stop(actor_t* actor);
void actor_system_shutdown(actor_system_t* system);
```

#### **Supervision and Fault Tolerance:**
```c
// Supervision tree implementation
typedef enum {
    SUPERVISOR_ONE_FOR_ONE,    // Restart only failed child
    SUPERVISOR_ONE_FOR_ALL,    // Restart all children if one fails
    SUPERVISOR_REST_FOR_ONE    // Restart failed child and all younger siblings
} supervisor_strategy_t;

typedef struct {
    supervisor_strategy_t strategy;
    int max_restarts;
    int time_window_seconds;
    void (*escalation_handler)(actor_t* supervisor, actor_t* failed_child);
} supervision_spec_t;

actor_t* supervisor_create(actor_system_t* system, const char* name,
                         supervision_spec_t* spec);
bool supervisor_add_child(actor_t* supervisor, actor_t* child);
void supervisor_handle_child_failure(actor_t* supervisor, actor_t* child,
                                    const char* reason);
```

#### **Remote Actor Communication:**
```c
// Network layer for distributed actors
typedef struct {
    char* host;
    int port;
    int socket_fd;
    pthread_t network_thread;
    
    // Serialization
    void* (*serialize)(actor_message_t* msg, size_t* size);
    actor_message_t* (*deserialize)(void* data, size_t size);
    
    // Connection management
    connection_pool_t* connections;
    atomic_bool connected;
} remote_actor_system_t;

bool actor_system_connect_remote(actor_system_t* local,
                               const char* remote_host, int remote_port);
actor_t* actor_lookup_remote(actor_system_t* system,
                           const char* remote_name);
```

#### **Testing and Benchmarking:**
```c
// Comprehensive test scenarios:

// 1. Basic actor lifecycle
void test_actor_creation_and_destruction();
void test_message_sending_and_receiving();

// 2. Supervision and fault tolerance
void test_child_restart_on_failure();
void test_supervision_strategies();
void test_escalation_handling();

// 3. Performance and scalability
void benchmark_message_throughput(int num_actors, int messages_per_actor);
void benchmark_actor_creation_time();
void test_memory_usage_scaling(int num_actors);

// 4. Distributed communication
void test_remote_message_sending();
void test_network_partition_handling();
void test_distributed_supervision();

// 5. Real-world scenario: Chat server
typedef struct {
    char* username;
    actor_t* connection_actor;
    char* current_room;
} chat_user_t;

actor_t* chat_room_create(const char* room_name);
actor_t* chat_user_create(const char* username);
void chat_system_benchmark(int num_users, int messages_per_user);
```

### Exercise 4: Memory-Efficient Object Pool with NUMA Awareness
**Difficulty: Advanced | Time Estimate: 2-3 days**

Design a sophisticated object pool that optimizes for NUMA architectures:

#### **NUMA-Aware Object Pool:**
```c
#include <numa.h>
#include <numaif.h>

typedef struct {
    void* objects;           // Pool of objects
    atomic_int* free_list;   // Indices of free objects
    atomic_int free_count;
    atomic_int total_count;
    
    // NUMA node information
    int numa_node;
    size_t object_size;
    size_t alignment;
    
    // Statistics
    atomic_ulong allocations;
    atomic_ulong deallocations;
    atomic_ulong allocation_failures;
    atomic_ulong cross_node_allocations;
    
    // Memory management
    void* memory_region;
    size_t memory_size;
    bool use_huge_pages;
} numa_object_pool_t;

typedef struct {
    numa_object_pool_t** node_pools;
    int num_nodes;
    
    // Global fallback pool
    numa_object_pool_t* global_pool;
    
    // Policy configuration
    bool strict_numa_policy;
    bool allow_cross_node_allocation;
    double load_balance_threshold;
    
    // Pool management
    pthread_t rebalance_thread;
    atomic_bool shutdown_requested;
} numa_pool_system_t;

// API to implement:
numa_pool_system_t* numa_pool_create(size_t object_size, 
                                    int objects_per_node,
                                    bool use_huge_pages);
void* numa_pool_alloc(numa_pool_system_t* system);
void numa_pool_free(numa_pool_system_t* system, void* obj);
void numa_pool_rebalance(numa_pool_system_t* system);
void numa_pool_get_stats(numa_pool_system_t* system, 
                        int node, pool_stats_t* stats);
```

#### **Advanced Features:**
```c
// 1. Dynamic pool resizing
typedef struct {
    double target_utilization;    // 0.7 = keep 70% utilized
    int min_objects_per_node;
    int max_objects_per_node;
    struct timespec resize_interval;
} pool_resize_policy_t;

void numa_pool_set_resize_policy(numa_pool_system_t* system,
                                pool_resize_policy_t* policy);

// 2. Object lifecycle callbacks
typedef struct {
    void (*on_construct)(void* obj);
    void (*on_destroy)(void* obj);
    void (*on_reset)(void* obj);    // Called when returned to pool
    bool (*on_validate)(void* obj); // Validate object integrity
} object_callbacks_t;

// 3. Memory prefetching and cache optimization
void numa_pool_prefetch_objects(numa_pool_system_t* system, int count);
void numa_pool_warm_cache(numa_pool_system_t* system);

// 4. Garbage collection and defragmentation
void numa_pool_compact(numa_pool_system_t* system);
void numa_pool_gc_unused_memory(numa_pool_system_t* system);
```

#### **Performance Requirements:**
```c
// Benchmarking targets:
// - Allocation time: <50ns on local NUMA node
// - Cross-node allocation penalty: <2x local allocation time
// - Memory efficiency: >90% utilization at steady state
// - Scalability: Linear performance up to 64 NUMA nodes

void benchmark_allocation_latency();
void benchmark_numa_affinity();
void benchmark_memory_fragmentation();
void test_concurrent_allocation(int num_threads);
```

### Exercise 5: Real-Time Task Scheduler with EDF Algorithm
**Difficulty: Expert | Time Estimate: 3-4 days**

Implement a real-time scheduler using Earliest Deadline First (EDF) algorithm:

#### **Real-Time Task System:**
```c
typedef enum {
    TASK_PERIODIC,
    TASK_APERIODIC,
    TASK_SPORADIC
} task_type_t;

typedef enum {
    TASK_READY,
    TASK_RUNNING,
    TASK_BLOCKED,
    TASK_DEADLINE_MISSED,
    TASK_COMPLETED
} task_state_t;

typedef struct rt_task {
    // Task identification
    int task_id;
    char* name;
    task_type_t type;
    
    // Timing constraints
    struct timespec deadline;     // Absolute deadline
    struct timespec period;       // For periodic tasks
    struct timespec wcet;         // Worst-case execution time
    struct timespec arrival_time;
    
    // Execution context
    void (*task_function)(struct rt_task* self);
    void* task_data;
    
    // Scheduling state
    task_state_t state;
    int priority;                 // Dynamic priority (deadline-based)
    struct timespec remaining_time;
    
    // Statistics
    atomic_ulong executions;
    atomic_ulong deadline_misses;
    struct timespec total_execution_time;
    struct timespec max_response_time;
    
    // Resource requirements
    int required_cpu_cores;
    size_t required_memory;
    bool requires_exclusive_access;
    
    // Linked list for ready queue
    struct rt_task* next;
} rt_task_t;

typedef struct {
    // Task queues (priority-ordered by deadline)
    rt_task_t* ready_queue;
    rt_task_t* blocked_queue;
    
    // Scheduler state
    rt_task_t* current_task;
    pthread_t scheduler_thread;
    
    // Timing
    struct timespec scheduler_tick;  // 1ms default
    timer_t system_timer;
    
    // Configuration
    bool preemptive;
    bool admit_control_enabled;
    double cpu_utilization_limit;   // 0.69 for EDF feasibility
    
    // Statistics
    atomic_ulong context_switches;
    atomic_ulong deadline_misses;
    atomic_ulong tasks_admitted;
    atomic_ulong tasks_rejected;
    
    // System resources
    int num_cpu_cores;
    size_t available_memory;
    
    // Synchronization
    pthread_mutex_t scheduler_mutex;
    pthread_cond_t scheduler_cond;
    atomic_bool shutdown_requested;
} rt_scheduler_t;

// Core scheduler API:
rt_scheduler_t* rt_scheduler_create(int num_cores);
bool rt_scheduler_admit_task(rt_scheduler_t* scheduler, rt_task_t* task);
void rt_scheduler_start(rt_scheduler_t* scheduler);
void rt_scheduler_stop(rt_scheduler_t* scheduler);
void rt_scheduler_remove_task(rt_scheduler_t* scheduler, int task_id);
```

#### **Advanced Scheduling Features:**
```c
// 1. Admission control with schedulability analysis
bool rt_admission_control_edf(rt_scheduler_t* scheduler, rt_task_t* new_task);
double rt_calculate_utilization(rt_scheduler_t* scheduler);

// 2. Deadline miss handling
typedef enum {
    DEADLINE_MISS_LOG,
    DEADLINE_MISS_RESTART,
    DEADLINE_MISS_DEGRADE,
    DEADLINE_MISS_ABORT
} deadline_miss_policy_t;

void rt_scheduler_set_deadline_policy(rt_scheduler_t* scheduler,
                                    deadline_miss_policy_t policy);

// 3. Resource sharing with Priority Inheritance Protocol
typedef struct {
    pthread_mutex_t mutex;
    rt_task_t* owner;
    rt_task_t* blocked_tasks;
    int ceiling_priority;
} rt_mutex_t;

int rt_mutex_lock(rt_mutex_t* mutex, rt_task_t* requestor);
int rt_mutex_unlock(rt_mutex_t* mutex, rt_task_t* owner);

// 4. Multicore EDF with task migration
typedef struct {
    rt_scheduler_t* schedulers[MAX_CORES];
    atomic_int load_per_core[MAX_CORES];
    bool migration_enabled;
    double migration_threshold;
} multicore_rt_system_t;

void rt_system_balance_load(multicore_rt_system_t* system);
bool rt_task_migrate(rt_task_t* task, int from_core, int to_core);
```

#### **Testing and Validation:**
```c
// Comprehensive test suite for real-time constraints:

// 1. Schedulability tests
void test_edf_schedulability();
void test_admission_control();
void test_deadline_guarantee();

// 2. Timing validation
void test_response_time_analysis();
void test_deadline_miss_detection();
void test_jitter_measurement();

// 3. Resource contention
void test_priority_inheritance();
void test_resource_sharing_protocols();

// 4. Real-world scenarios
void simulate_control_system(int num_sensors, int control_frequency);
void simulate_multimedia_streaming(int num_streams, int frame_rate);
void simulate_network_packet_processing(int packet_rate);

// 5. Performance benchmarks
void benchmark_context_switch_overhead();
void benchmark_scheduler_latency();
void measure_interrupt_response_time();
```

### Self-Assessment Rubric

For each exercise, evaluate yourself using this comprehensive rubric:

#### **Technical Mastery (40 points)**
- **Exceptional (36-40)**: Implements all requirements with advanced optimizations, handles edge cases perfectly
- **Proficient (28-35)**: Implements core requirements correctly with good error handling
- **Developing (20-27)**: Basic implementation works but has some threading issues or missing features
- **Needs Improvement (0-19)**: Significant threading bugs, memory leaks, or incorrect behavior

#### **Performance and Scalability (25 points)**
- **Exceptional (23-25)**: Exceeds performance benchmarks, scales linearly, minimal overhead
- **Proficient (18-22)**: Meets performance targets, good scalability characteristics
- **Developing (13-17)**: Acceptable performance but doesn't scale well or has overhead issues
- **Needs Improvement (0-12)**: Poor performance, doesn't meet basic throughput requirements

#### **Code Quality and Architecture (20 points)**
- **Exceptional (18-20)**: Clean, maintainable code with excellent separation of concerns
- **Proficient (14-17)**: Well-structured code with good abstractions
- **Developing (10-13)**: Code works but is difficult to understand or extend
- **Needs Improvement (0-9)**: Poor code organization, hard to read or maintain

#### **Testing and Documentation (15 points)**
- **Exceptional (14-15)**: Comprehensive test suite, excellent documentation with examples
- **Proficient (11-13)**: Good test coverage, clear documentation
- **Developing (8-10)**: Basic tests, minimal documentation
- **Needs Improvement (0-7)**: No tests or documentation, unclear how to use the code

#### **Total Score Interpretation:**
- **90-100**: Expert level - Ready for senior threading/systems programming roles
- **75-89**: Advanced level - Strong foundation with some areas for improvement
- **60-74**: Intermediate level - Understands concepts but needs more practice
- **Below 60**: Novice level - Requires significant additional study and practice

### Additional Challenge Projects

For those seeking even greater challenges:

1. **Distributed Consensus Implementation**: Build a Raft or PBFT consensus algorithm
2. **High-Frequency Trading Engine**: Sub-microsecond latency message processing
3. **GPU-CPU Hybrid Task Scheduler**: Coordinate work between CPU threads and GPU kernels
4. **Kernel-Level Thread Scheduler**: Implement scheduling policies in a custom kernel module
5. **Blockchain Validation Engine**: Parallel validation of cryptocurrency transactions

## Comprehensive Assessment Framework

This assessment framework evaluates your mastery of advanced threading patterns through both theoretical understanding and practical implementation skills. Complete all sections to demonstrate expert-level proficiency.

### Part I: Advanced Concepts Analysis (30 points)

#### **Question 1: Threading Pattern Selection (10 points)**
You are designing a high-performance financial trading system that must process 1 million orders per second with strict latency requirements (<100μs). The system has the following characteristics:
- 64-core NUMA machine with 4 NUMA nodes
- Orders arrive in bursts with high variability
- Each order requires validation, risk checking, and execution
- System must maintain FIFO ordering within each symbol
- Fault tolerance is critical - no lost orders

**Analyze and justify your choices for:**
a) Primary threading pattern (thread pool, work-stealing, actor model, etc.)
b) Synchronization mechanisms for order queues
c) Memory management strategy
d) Load balancing approach across NUMA nodes
e) Fault tolerance and recovery mechanisms

**Expected Answer Length: 800-1000 words with code examples**

#### **Question 2: Lock-Free Algorithm Design (10 points)**
Design a lock-free algorithm for a multi-producer, multi-consumer priority queue that supports the following operations:
- `enqueue(item, priority)` - Add item with given priority
- `dequeue()` - Remove highest priority item
- `peek()` - View highest priority item without removing
- `size()` - Get current queue size

**Requirements:**
- Must handle ABA problem correctly
- Should minimize memory allocation
- Must provide progress guarantees
- Handle arbitrary number of threads

**Provide:**
a) Complete data structure definition
b) Detailed algorithm for each operation
c) Analysis of time complexity
d) Discussion of memory ordering requirements
e) Proof sketch for correctness

#### **Question 3: Performance Analysis (10 points)**
Given the following threading scenarios, analyze the performance characteristics and identify bottlenecks:

```c
// Scenario A: Traditional mutex-based approach
typedef struct {
    int counter;
    pthread_mutex_t mutex;
} counter_t;

void increment_mutex(counter_t* c) {
    pthread_mutex_lock(&c->mutex);
    c->counter++;
    pthread_mutex_unlock(&c->mutex);
}

// Scenario B: Atomic operations
atomic_int atomic_counter = 0;

void increment_atomic() {
    atomic_fetch_add(&atomic_counter, 1);
}

// Scenario C: Thread-local aggregation
__thread int local_counter = 0;
atomic_int global_counter = 0;

void increment_local() {
    local_counter++;
    if (local_counter % 1000 == 0) {
        atomic_fetch_add(&global_counter, 1000);
        local_counter = 0;
    }
}
```

**Analyze:**
a) Scalability characteristics with increasing thread count (1, 2, 4, 8, 16, 32, 64 threads)
b) Cache coherency impact for each approach
c) Memory bandwidth requirements
d) When each approach would be optimal
e) Hybrid approach combining benefits of all three

### Part II: Implementation Challenges (40 points)

#### **Challenge 1: Advanced Work-Stealing Implementation (20 points)**
Implement a work-stealing thread pool with the following advanced features:

```c
// Required API:
typedef struct advanced_work_stealing_pool aws_pool_t;

aws_pool_t* aws_create(int num_threads);
bool aws_submit_task(aws_pool_t* pool, void (*task)(void*), void* arg, int priority);
bool aws_submit_batch(aws_pool_t* pool, task_batch_t* batch);
void aws_set_affinity_policy(aws_pool_t* pool, affinity_policy_t policy);
void aws_enable_load_balancing(aws_pool_t* pool, bool enabled);
aws_stats_t aws_get_statistics(aws_pool_t* pool);
void aws_shutdown(aws_pool_t* pool);
```

**Required Features:**
1. **Hierarchical Work Stealing**: Workers prefer to steal from threads on the same NUMA node
2. **Priority Support**: High-priority tasks jump to front of deque
3. **Adaptive Load Balancing**: Monitor queue lengths and redistribute work
4. **Task Batching**: Submit multiple related tasks efficiently
5. **NUMA Awareness**: Pin threads to specific NUMA nodes
6. **Comprehensive Statistics**: Track steals, load distribution, latency histograms

**Evaluation Criteria:**
- Correctness of work-stealing algorithm (5 points)
- NUMA optimization implementation (4 points)
- Priority handling mechanism (3 points)
- Load balancing effectiveness (3 points)
- Performance and scalability (3 points)
- Code quality and documentation (2 points)

#### **Challenge 2: Software Transactional Memory System (20 points)**
Implement a complete STM system supporting nested transactions:

```c
// Required API:
typedef struct stm_system stm_system_t;
typedef struct stm_transaction stm_tx_t;
typedef struct stm_var stm_var_t;

stm_system_t* stm_system_create();
stm_var_t* stm_var_create(stm_system_t* system, void* initial_value, size_t size);

stm_tx_t* stm_begin(stm_system_t* system);
void* stm_read(stm_tx_t* tx, stm_var_t* var);
void stm_write(stm_tx_t* tx, stm_var_t* var, void* value);
bool stm_commit(stm_tx_t* tx);
void stm_abort(stm_tx_t* tx);

// Nested transaction support
stm_tx_t* stm_begin_nested(stm_tx_t* parent);
bool stm_commit_nested(stm_tx_t* nested);

// Retry mechanism
void stm_retry(stm_tx_t* tx); // Block until any read variable changes
```

**Required Features:**
1. **MVCC-based Implementation**: Multi-version concurrency control
2. **Nested Transactions**: Support arbitrary nesting depth
3. **Retry Mechanism**: Compositional blocking operations
4. **Contention Management**: Handle high-contention scenarios
5. **Memory Management**: Efficient cleanup of old versions
6. **Validation Optimization**: Incremental validation during transaction

**Test Implementation:**
```c
// Implement these test scenarios:
void test_bank_transfer(); // Classic bank account transfer
void test_nested_transactions(); // Nested transaction rollback
void test_retry_mechanism(); // Producer-consumer with retry
void test_high_contention(); // Many threads, few variables
void benchmark_stm_vs_locks(); // Performance comparison
```

**Evaluation Criteria:**
- Transaction isolation correctness (5 points)
- Nested transaction implementation (4 points)
- Retry mechanism functionality (3 points)
- Contention management effectiveness (3 points)
- Performance optimization (3 points)
- Comprehensive testing (2 points)

### Part III: Real-World System Design (30 points)

#### **System Design Challenge: Distributed Message Queue**
Design and partially implement a distributed message queue system similar to Apache Kafka, focusing on the threading and concurrency aspects.

**System Requirements:**
- Handle 1M messages/second sustained throughput
- Support multiple topics with partitions
- Provide at-least-once delivery guarantees
- Scale horizontally across multiple nodes
- Handle node failures gracefully
- Support both publish/subscribe and message queuing patterns

**Your solution must address:**

#### **1. Producer Architecture (8 points)**
```c
typedef struct message_producer producer_t;

producer_t* producer_create(const char* bootstrap_servers);
int producer_send(producer_t* producer, const char* topic, 
                 const void* key, size_t key_len,
                 const void* value, size_t value_len);
int producer_send_async(producer_t* producer, const char* topic,
                       const void* key, size_t key_len,
                       const void* value, size_t value_len,
                       void (*callback)(int status, void* user_data),
                       void* user_data);
void producer_flush(producer_t* producer);
```

**Design Requirements:**
- Asynchronous batching for high throughput
- Automatic retry with exponential backoff
- Partitioning strategy (round-robin, hash-based, custom)
- Connection pooling to brokers
- Memory-efficient message buffering

#### **2. Consumer Architecture (8 points)**
```c
typedef struct message_consumer consumer_t;

consumer_t* consumer_create(const char* group_id, const char* bootstrap_servers);
int consumer_subscribe(consumer_t* consumer, const char** topics, int topic_count);
message_t* consumer_poll(consumer_t* consumer, int timeout_ms);
int consumer_commit_sync(consumer_t* consumer);
int consumer_commit_async(consumer_t* consumer, commit_callback_t callback);
```

**Design Requirements:**
- Consumer group coordination
- Automatic partition assignment and rebalancing
- Offset management and persistence
- Heartbeat mechanism for failure detection
- Message prefetching and local buffering

#### **3. Broker Threading Model (8 points)**
```c
typedef struct message_broker broker_t;

broker_t* broker_create(int port);
void broker_add_topic(broker_t* broker, const char* topic, int num_partitions);
void broker_start(broker_t* broker);
void broker_shutdown(broker_t* broker);
```

**Design Requirements:**
- Accept connections from producers and consumers
- Manage topic metadata and partition assignments
- Handle message persistence to disk
- Coordinate with other brokers for replication
- Health monitoring and metrics collection

#### **4. Fault Tolerance and Consistency (6 points)**
**Address these aspects:**
- **Replication Strategy**: How do you replicate messages across brokers?
- **Leader Election**: How is partition leadership determined?
- **Split-Brain Prevention**: How do you handle network partitions?
- **Recovery Mechanism**: How does a failed broker catch up?
- **Consistency Guarantees**: What consistency model do you provide?

### Assessment Scoring Guide

#### **Grading Scale:**
- **A+ (95-100)**: Exceptional mastery - All implementations are correct, optimized, and production-ready
- **A (90-94)**: Advanced proficiency - Minor issues in optimization or edge case handling
- **A- (85-89)**: Good understanding - Some threading bugs or missing advanced features
- **B+ (80-84)**: Satisfactory - Basic requirements met but lacks optimization
- **B (75-79)**: Needs improvement - Several threading issues or incomplete implementations
- **Below 75**: Requires significant additional study

#### **Detailed Evaluation Criteria:**

**Technical Correctness (40%)**
- Thread safety and data race prevention
- Proper synchronization primitive usage
- Memory management and leak prevention
- Error handling and edge case coverage

**Performance and Scalability (30%)**
- Achieves specified performance benchmarks
- Scales well with increasing load and core count
- Efficient resource utilization
- Minimal contention and overhead

**Code Quality (20%)**
- Clean, readable, and maintainable code
- Proper abstraction and modularity
- Comprehensive error handling
- Clear documentation and comments

**Innovation and Optimization (10%)**
- Creative solutions to complex problems
- Advanced optimization techniques
- Thoughtful design decisions
- Understanding of hardware implications

### Supplementary Assessment Activities

#### **Code Review Exercise**
Review and critique the following code snippets, identifying threading issues and suggesting improvements:

1. **Race Condition Detection**
2. **Deadlock Prevention Analysis**
3. **Performance Optimization Opportunities**
4. **Memory Ordering Correctness**

#### **Performance Benchmarking Lab**
Design and execute benchmarks to compare:
- Different synchronization mechanisms under various contention levels
- Lock-free vs. lock-based data structures
- Thread pool configurations for different workload patterns
- NUMA-aware vs. NUMA-oblivious implementations

#### **Case Study Analysis**
Analyze real-world threading issues from production systems:
- Deadlock incidents and their resolution
- Performance degradation due to false sharing
- Scalability bottlenecks in large-scale systems
- Race conditions that led to data corruption

### Self-Study Verification Checklist

Before considering yourself proficient in advanced threading patterns, ensure you can:

**Conceptual Understanding:**
- [ ] Explain the trade-offs between different threading patterns
- [ ] Design lock-free algorithms with correctness proofs
- [ ] Analyze the performance characteristics of concurrent systems
- [ ] Understand hardware implications of threading decisions

**Practical Implementation:**
- [ ] Implement production-quality thread pools and work-stealing systems
- [ ] Build lock-free data structures handling ABA problems
- [ ] Create fault-tolerant concurrent systems with proper cleanup
- [ ] Optimize threading code for specific hardware architectures

**System Design:**
- [ ] Design scalable multi-threaded architectures
- [ ] Handle threading in distributed systems
- [ ] Implement proper monitoring and debugging for concurrent systems
- [ ] Make informed decisions about threading patterns for specific use cases

**Professional Skills:**
- [ ] Debug complex threading issues in production environments
- [ ] Profile and optimize multi-threaded applications
- [ ] Mentor others on concurrent programming best practices
- [ ] Contribute to threading-related open-source projects

### Continuous Learning Path

To maintain and advance your threading expertise:

1. **Stay Current**: Follow developments in hardware (new CPU architectures, memory models)
2. **Open Source Contribution**: Contribute to high-performance threading libraries
3. **Conference Participation**: Attend conferences like CppCon, C++Now, Systems conferences
4. **Industry Collaboration**: Work on real-world high-performance systems
5. **Research Engagement**: Read current research papers on concurrency and parallelism
6. **Teaching and Mentoring**: Explain concepts to others to deepen your own understanding

## Next Section
[Performance Considerations](07_Performance_Considerations.md)
