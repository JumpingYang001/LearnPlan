# Project 4: Thread Pool Implementation

## Objective
Create a reusable thread pool with work queue and task submission API. This project demonstrates advanced threading patterns, task-based parallelism, and resource management.

## Requirements

### Basic Requirements
1. Create a fixed-size thread pool with configurable number of worker threads
2. Implement a work queue for task submission
3. Support function pointer-based tasks with void* arguments
4. Provide proper shutdown mechanism
5. Handle errors gracefully and provide resource cleanup

### Advanced Requirements
1. Dynamic thread pool with auto-scaling capabilities
2. Priority-based task scheduling
3. Task cancellation and timeout mechanisms
4. Thread pool statistics and monitoring
5. Work stealing between threads for load balancing

## Implementation Guide

### Basic Thread Pool Structure

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

// Task structure
typedef struct {
    void (*function)(void* arg);
    void* argument;
    int task_id;
    int priority;
    time_t submit_time;
} Task;

// Task queue (circular buffer)
typedef struct {
    Task* tasks;
    int capacity;
    int size;
    int head;
    int tail;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} TaskQueue;

// Thread pool structure
typedef struct {
    pthread_t* workers;
    int num_threads;
    TaskQueue* task_queue;
    bool shutdown;
    bool immediate_shutdown; // Don't process remaining tasks
    
    // Statistics
    long total_tasks_submitted;
    long total_tasks_completed;
    long total_tasks_failed;
    time_t start_time;
    
    pthread_mutex_t stats_mutex;
} ThreadPool;

// Worker thread data
typedef struct {
    ThreadPool* pool;
    int worker_id;
    long tasks_processed;
    bool is_busy;
    pthread_t thread_id;
} WorkerData;
```

### Task Queue Implementation

```c
TaskQueue* task_queue_create(int capacity) {
    TaskQueue* queue = malloc(sizeof(TaskQueue));
    if (!queue) return NULL;
    
    queue->tasks = malloc(sizeof(Task) * capacity);
    if (!queue->tasks) {
        free(queue);
        return NULL;
    }
    
    queue->capacity = capacity;
    queue->size = 0;
    queue->head = 0;
    queue->tail = 0;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue->tasks);
        free(queue);
        return NULL;
    }
    
    if (pthread_cond_init(&queue->not_empty, NULL) != 0) {
        pthread_mutex_destroy(&queue->mutex);
        free(queue->tasks);
        free(queue);
        return NULL;
    }
    
    if (pthread_cond_init(&queue->not_full, NULL) != 0) {
        pthread_cond_destroy(&queue->not_empty);
        pthread_mutex_destroy(&queue->mutex);
        free(queue->tasks);
        free(queue);
        return NULL;
    }
    
    return queue;
}

bool task_queue_enqueue(TaskQueue* queue, const Task* task) {
    if (!queue || !task) return false;
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size >= queue->capacity) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    queue->tasks[queue->tail] = *task;
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->size++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}

bool task_queue_dequeue(TaskQueue* queue, Task* task, bool blocking) {
    if (!queue || !task) return false;
    
    pthread_mutex_lock(&queue->mutex);
    
    if (blocking) {
        while (queue->size == 0) {
            pthread_cond_wait(&queue->not_empty, &queue->mutex);
        }
    } else {
        if (queue->size == 0) {
            pthread_mutex_unlock(&queue->mutex);
            return false;
        }
    }
    
    *task = queue->tasks[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}

bool task_queue_dequeue_timeout(TaskQueue* queue, Task* task, int timeout_ms) {
    if (!queue || !task) return false;
    
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_ms / 1000;
    timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
    
    if (timeout.tv_nsec >= 1000000000) {
        timeout.tv_sec++;
        timeout.tv_nsec -= 1000000000;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size == 0) {
        int result = pthread_cond_timedwait(&queue->not_empty, &queue->mutex, &timeout);
        if (result == ETIMEDOUT) {
            pthread_mutex_unlock(&queue->mutex);
            return false;
        }
    }
    
    *task = queue->tasks[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}

int task_queue_size(TaskQueue* queue) {
    if (!queue) return -1;
    
    pthread_mutex_lock(&queue->mutex);
    int size = queue->size;
    pthread_mutex_unlock(&queue->mutex);
    
    return size;
}

void task_queue_destroy(TaskQueue* queue) {
    if (!queue) return;
    
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
    free(queue->tasks);
    free(queue);
}
```

### Worker Thread Implementation

```c
void* worker_thread_function(void* arg) {
    WorkerData* worker = (WorkerData*)arg;
    ThreadPool* pool = worker->pool;
    
    printf("Worker %d started (Thread ID: %lu)\n", 
           worker->worker_id, (unsigned long)pthread_self());
    
    while (true) {
        Task task;
        
        // Try to get a task from the queue
        if (!task_queue_dequeue_timeout(pool->task_queue, &task, 1000)) {
            // Timeout occurred, check if we should shutdown
            if (pool->shutdown || pool->immediate_shutdown) {
                break;
            }
            continue;
        }
        
        // Check for shutdown after getting a task
        if (pool->immediate_shutdown) {
            break;
        }
        
        // Execute the task
        worker->is_busy = true;
        
        if (task.function) {
            struct timespec start_time, end_time;
            clock_gettime(CLOCK_MONOTONIC, &start_time);
            
            printf("Worker %d: Executing task %d\n", worker->worker_id, task.task_id);
            
            // Execute the task function
            task.function(task.argument);
            
            clock_gettime(CLOCK_MONOTONIC, &end_time);
            
            double execution_time = (end_time.tv_sec - start_time.tv_sec) +
                                   (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
            
            printf("Worker %d: Completed task %d (execution time: %.6f seconds)\n", 
                   worker->worker_id, task.task_id, execution_time);
            
            worker->tasks_processed++;
            
            // Update pool statistics
            pthread_mutex_lock(&pool->stats_mutex);
            pool->total_tasks_completed++;
            pthread_mutex_unlock(&pool->stats_mutex);
        }
        
        worker->is_busy = false;
        
        // Check for graceful shutdown (process remaining tasks)
        if (pool->shutdown && task_queue_size(pool->task_queue) == 0) {
            break;
        }
    }
    
    printf("Worker %d shutting down (processed %ld tasks)\n", 
           worker->worker_id, worker->tasks_processed);
    
    return NULL;
}
```

### Thread Pool Core Implementation

```c
ThreadPool* thread_pool_create(int num_threads, int queue_capacity) {
    if (num_threads <= 0 || queue_capacity <= 0) {
        return NULL;
    }
    
    ThreadPool* pool = malloc(sizeof(ThreadPool));
    if (!pool) return NULL;
    
    // Initialize pool structure
    pool->num_threads = num_threads;
    pool->shutdown = false;
    pool->immediate_shutdown = false;
    pool->total_tasks_submitted = 0;
    pool->total_tasks_completed = 0;
    pool->total_tasks_failed = 0;
    pool->start_time = time(NULL);
    
    if (pthread_mutex_init(&pool->stats_mutex, NULL) != 0) {
        free(pool);
        return NULL;
    }
    
    // Create task queue
    pool->task_queue = task_queue_create(queue_capacity);
    if (!pool->task_queue) {
        pthread_mutex_destroy(&pool->stats_mutex);
        free(pool);
        return NULL;
    }
    
    // Allocate worker threads and data
    pool->workers = malloc(sizeof(pthread_t) * num_threads);
    if (!pool->workers) {
        task_queue_destroy(pool->task_queue);
        pthread_mutex_destroy(&pool->stats_mutex);
        free(pool);
        return NULL;
    }
    
    WorkerData* worker_data = malloc(sizeof(WorkerData) * num_threads);
    if (!worker_data) {
        free(pool->workers);
        task_queue_destroy(pool->task_queue);
        pthread_mutex_destroy(&pool->stats_mutex);
        free(pool);
        return NULL;
    }
    
    // Create worker threads
    for (int i = 0; i < num_threads; i++) {
        worker_data[i].pool = pool;
        worker_data[i].worker_id = i;
        worker_data[i].tasks_processed = 0;
        worker_data[i].is_busy = false;
        
        if (pthread_create(&pool->workers[i], NULL, worker_thread_function, &worker_data[i]) != 0) {
            // Clean up already created threads
            pool->immediate_shutdown = true;
            for (int j = 0; j < i; j++) {
                pthread_join(pool->workers[j], NULL);
            }
            
            free(worker_data);
            free(pool->workers);
            task_queue_destroy(pool->task_queue);
            pthread_mutex_destroy(&pool->stats_mutex);
            free(pool);
            return NULL;
        }
        
        worker_data[i].thread_id = pool->workers[i];
    }
    
    printf("Thread pool created with %d worker threads\n", num_threads);
    return pool;
}

bool thread_pool_submit_task(ThreadPool* pool, void (*function)(void*), void* argument) {
    if (!pool || !function || pool->shutdown || pool->immediate_shutdown) {
        return false;
    }
    
    static int task_id_counter = 0;
    Task task;
    task.function = function;
    task.argument = argument;
    task.task_id = ++task_id_counter;
    task.priority = 0; // Default priority
    task.submit_time = time(NULL);
    
    if (task_queue_enqueue(pool->task_queue, &task)) {
        pthread_mutex_lock(&pool->stats_mutex);
        pool->total_tasks_submitted++;
        pthread_mutex_unlock(&pool->stats_mutex);
        
        printf("Submitted task %d to thread pool\n", task.task_id);
        return true;
    }
    
    pthread_mutex_lock(&pool->stats_mutex);
    pool->total_tasks_failed++;
    pthread_mutex_unlock(&pool->stats_mutex);
    
    return false;
}

// Shutdown thread pool (process remaining tasks)
void thread_pool_shutdown(ThreadPool* pool) {
    if (!pool || pool->shutdown) return;
    
    printf("Initiating thread pool shutdown...\n");
    
    pool->shutdown = true;
    
    // Wake up all worker threads
    pthread_mutex_lock(&pool->task_queue->mutex);
    pthread_cond_broadcast(&pool->task_queue->not_empty);
    pthread_mutex_unlock(&pool->task_queue->mutex);
    
    // Wait for all threads to finish
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->workers[i], NULL);
    }
    
    printf("Thread pool shutdown completed\n");
}

// Immediate shutdown (don't process remaining tasks)
void thread_pool_shutdown_immediate(ThreadPool* pool) {
    if (!pool) return;
    
    printf("Initiating immediate thread pool shutdown...\n");
    
    pool->immediate_shutdown = true;
    
    // Wake up all worker threads
    pthread_mutex_lock(&pool->task_queue->mutex);
    pthread_cond_broadcast(&pool->task_queue->not_empty);
    pthread_mutex_unlock(&pool->task_queue->mutex);
    
    // Wait for all threads to finish
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->workers[i], NULL);
    }
    
    printf("Immediate thread pool shutdown completed\n");
}

void thread_pool_destroy(ThreadPool* pool) {
    if (!pool) return;
    
    if (!pool->shutdown && !pool->immediate_shutdown) {
        thread_pool_shutdown(pool);
    }
    
    task_queue_destroy(pool->task_queue);
    pthread_mutex_destroy(&pool->stats_mutex);
    free(pool->workers);
    free(pool);
}
```

### Thread Pool Statistics and Monitoring

```c
typedef struct {
    int num_threads;
    int queue_size;
    int queue_capacity;
    long total_submitted;
    long total_completed;
    long total_failed;
    double uptime_seconds;
    double throughput_tasks_per_second;
    int busy_threads;
} ThreadPoolStats;

ThreadPoolStats thread_pool_get_stats(ThreadPool* pool) {
    ThreadPoolStats stats = {0};
    
    if (!pool) return stats;
    
    pthread_mutex_lock(&pool->stats_mutex);
    
    stats.num_threads = pool->num_threads;
    stats.queue_size = task_queue_size(pool->task_queue);
    stats.queue_capacity = pool->task_queue->capacity;
    stats.total_submitted = pool->total_tasks_submitted;
    stats.total_completed = pool->total_tasks_completed;
    stats.total_failed = pool->total_tasks_failed;
    
    time_t current_time = time(NULL);
    stats.uptime_seconds = difftime(current_time, pool->start_time);
    
    if (stats.uptime_seconds > 0) {
        stats.throughput_tasks_per_second = stats.total_completed / stats.uptime_seconds;
    }
    
    pthread_mutex_unlock(&pool->stats_mutex);
    
    return stats;
}

void thread_pool_print_stats(ThreadPool* pool) {
    ThreadPoolStats stats = thread_pool_get_stats(pool);
    
    printf("\n=== Thread Pool Statistics ===\n");
    printf("Number of threads: %d\n", stats.num_threads);
    printf("Queue size: %d/%d\n", stats.queue_size, stats.queue_capacity);
    printf("Tasks submitted: %ld\n", stats.total_submitted);
    printf("Tasks completed: %ld\n", stats.total_completed);
    printf("Tasks failed: %ld\n", stats.total_failed);
    printf("Pending tasks: %ld\n", stats.total_submitted - stats.total_completed - stats.total_failed);
    printf("Uptime: %.2f seconds\n", stats.uptime_seconds);
    printf("Throughput: %.2f tasks/second\n", stats.throughput_tasks_per_second);
    printf("Queue utilization: %.1f%%\n", 
           (stats.queue_size * 100.0) / stats.queue_capacity);
}
```

### Example Task Functions and Test Framework

```c
// Sample task functions
void cpu_intensive_task(void* arg) {
    int iterations = *(int*)arg;
    volatile long sum = 0;
    
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 100000; j++) {
            sum += j;
        }
    }
    
    printf("CPU intensive task completed (iterations: %d, sum: %ld)\n", 
           iterations, sum);
}

void io_simulation_task(void* arg) {
    int delay_ms = *(int*)arg;
    
    printf("Starting I/O simulation (delay: %dms)\n", delay_ms);
    usleep(delay_ms * 1000);
    printf("I/O simulation completed\n");
}

void memory_allocation_task(void* arg) {
    int size_mb = *(int*)arg;
    
    printf("Allocating %d MB of memory\n", size_mb);
    
    void* memory = malloc(size_mb * 1024 * 1024);
    if (memory) {
        // Touch the memory to ensure it's allocated
        memset(memory, 0, size_mb * 1024 * 1024);
        printf("Memory allocation successful\n");
        
        // Keep it for a short time
        usleep(500000); // 500ms
        
        free(memory);
        printf("Memory freed\n");
    } else {
        printf("Memory allocation failed\n");
    }
}

void simple_print_task(void* arg) {
    char* message = (char*)arg;
    printf("Task message: %s\n", message);
    usleep(100000); // 100ms delay
}

// Test framework
void run_thread_pool_test() {
    printf("=== Thread Pool Test ===\n");
    
    // Create thread pool
    ThreadPool* pool = thread_pool_create(4, 20);
    if (!pool) {
        printf("Failed to create thread pool\n");
        return;
    }
    
    // Test 1: CPU intensive tasks
    printf("\nTest 1: CPU intensive tasks\n");
    int cpu_iterations[] = {100, 200, 150, 300, 250};
    for (int i = 0; i < 5; i++) {
        thread_pool_submit_task(pool, cpu_intensive_task, &cpu_iterations[i]);
    }
    
    sleep(3);
    thread_pool_print_stats(pool);
    
    // Test 2: I/O simulation tasks
    printf("\nTest 2: I/O simulation tasks\n");
    int io_delays[] = {100, 200, 150, 300, 250, 180, 220};
    for (int i = 0; i < 7; i++) {
        thread_pool_submit_task(pool, io_simulation_task, &io_delays[i]);
    }
    
    sleep(2);
    thread_pool_print_stats(pool);
    
    // Test 3: Memory allocation tasks
    printf("\nTest 3: Memory allocation tasks\n");
    int memory_sizes[] = {10, 20, 15, 25, 30};
    for (int i = 0; i < 5; i++) {
        thread_pool_submit_task(pool, memory_allocation_task, &memory_sizes[i]);
    }
    
    sleep(3);
    thread_pool_print_stats(pool);
    
    // Test 4: Many small tasks
    printf("\nTest 4: Many small tasks\n");
    char messages[20][50];
    for (int i = 0; i < 20; i++) {
        snprintf(messages[i], sizeof(messages[i]), "Small task number %d", i + 1);
        thread_pool_submit_task(pool, simple_print_task, messages[i]);
    }
    
    sleep(4);
    thread_pool_print_stats(pool);
    
    // Shutdown and cleanup
    printf("\nShutting down thread pool...\n");
    thread_pool_shutdown(pool);
    thread_pool_print_stats(pool);
    thread_pool_destroy(pool);
}

int main() {
    run_thread_pool_test();
    return 0;
}
```

### Advanced: Dynamic Thread Pool

```c
typedef struct {
    ThreadPool* base_pool;
    int min_threads;
    int max_threads;
    int current_threads;
    double scale_up_threshold;   // Queue utilization % to add threads
    double scale_down_threshold; // Idle time to remove threads
    time_t last_scale_time;
    pthread_mutex_t scale_mutex;
    pthread_t monitor_thread;
    bool monitor_running;
} DynamicThreadPool;

void* monitor_thread_function(void* arg) {
    DynamicThreadPool* dyn_pool = (DynamicThreadPool*)arg;
    
    while (dyn_pool->monitor_running) {
        sleep(5); // Check every 5 seconds
        
        ThreadPoolStats stats = thread_pool_get_stats(dyn_pool->base_pool);
        double queue_utilization = (stats.queue_size * 100.0) / stats.queue_capacity;
        
        pthread_mutex_lock(&dyn_pool->scale_mutex);
        
        time_t now = time(NULL);
        bool can_scale = (now - dyn_pool->last_scale_time) >= 10; // Min 10 seconds between scaling
        
        if (can_scale) {
            if (queue_utilization > dyn_pool->scale_up_threshold && 
                dyn_pool->current_threads < dyn_pool->max_threads) {
                
                // Scale up
                printf("Scaling up: Queue utilization %.1f%% > %.1f%%\n", 
                       queue_utilization, dyn_pool->scale_up_threshold);
                
                // Implementation would add new worker threads here
                dyn_pool->current_threads++;
                dyn_pool->last_scale_time = now;
                
            } else if (queue_utilization < dyn_pool->scale_down_threshold && 
                      dyn_pool->current_threads > dyn_pool->min_threads) {
                
                // Scale down
                printf("Scaling down: Queue utilization %.1f%% < %.1f%%\n", 
                       queue_utilization, dyn_pool->scale_down_threshold);
                
                // Implementation would remove worker threads here
                dyn_pool->current_threads--;
                dyn_pool->last_scale_time = now;
            }
        }
        
        pthread_mutex_unlock(&dyn_pool->scale_mutex);
    }
    
    return NULL;
}
```

### Priority Task Implementation

```c
typedef struct {
    Task* tasks;
    int capacity;
    int size;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} PriorityTaskQueue;

// Max-heap implementation for priority queue
void priority_queue_heapify_up(Task* tasks, int index) {
    if (index == 0) return;
    
    int parent = (index - 1) / 2;
    if (tasks[parent].priority < tasks[index].priority) {
        Task temp = tasks[parent];
        tasks[parent] = tasks[index];
        tasks[index] = temp;
        priority_queue_heapify_up(tasks, parent);
    }
}

void priority_queue_heapify_down(Task* tasks, int size, int index) {
    int largest = index;
    int left = 2 * index + 1;
    int right = 2 * index + 2;
    
    if (left < size && tasks[left].priority > tasks[largest].priority) {
        largest = left;
    }
    
    if (right < size && tasks[right].priority > tasks[largest].priority) {
        largest = right;
    }
    
    if (largest != index) {
        Task temp = tasks[index];
        tasks[index] = tasks[largest];
        tasks[largest] = temp;
        priority_queue_heapify_down(tasks, size, largest);
    }
}

bool priority_task_queue_enqueue(PriorityTaskQueue* queue, const Task* task) {
    if (!queue || !task) return false;
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size >= queue->capacity) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    // Add task to end and heapify up
    queue->tasks[queue->size] = *task;
    priority_queue_heapify_up(queue->tasks, queue->size);
    queue->size++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}

bool priority_task_queue_dequeue(PriorityTaskQueue* queue, Task* task) {
    if (!queue || !task) return false;
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size == 0) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }
    
    // Remove highest priority task (root of heap)
    *task = queue->tasks[0];
    queue->tasks[0] = queue->tasks[queue->size - 1];
    queue->size--;
    
    if (queue->size > 0) {
        priority_queue_heapify_down(queue->tasks, queue->size, 0);
    }
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    
    return true;
}
```

## Performance Benchmarking

```c
typedef struct {
    int num_threads;
    int queue_size;
    int num_tasks;
    double total_time;
    double throughput;
    double avg_response_time;
} BenchmarkResult;

BenchmarkResult benchmark_thread_pool(int num_threads, int queue_size, int num_tasks) {
    BenchmarkResult result = {0};
    result.num_threads = num_threads;
    result.queue_size = queue_size;
    result.num_tasks = num_tasks;
    
    ThreadPool* pool = thread_pool_create(num_threads, queue_size);
    if (!pool) {
        return result;
    }
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Submit tasks
    int task_data[num_tasks];
    for (int i = 0; i < num_tasks; i++) {
        task_data[i] = 100; // Standard work amount
        thread_pool_submit_task(pool, cpu_intensive_task, &task_data[i]);
    }
    
    // Wait for completion
    while (true) {
        ThreadPoolStats stats = thread_pool_get_stats(pool);
        if (stats.total_completed >= num_tasks) {
            break;
        }
        usleep(10000); // 10ms
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    result.total_time = (end_time.tv_sec - start_time.tv_sec) +
                       (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    result.throughput = num_tasks / result.total_time;
    result.avg_response_time = result.total_time / num_tasks;
    
    thread_pool_shutdown(pool);
    thread_pool_destroy(pool);
    
    return result;
}

void run_performance_tests() {
    printf("\n=== Thread Pool Performance Tests ===\n");
    
    int thread_counts[] = {1, 2, 4, 8, 16};
    int queue_sizes[] = {10, 50, 100};
    int task_count = 1000;
    
    printf("%-8s %-10s %-10s %-12s %-15s %-15s\n", 
           "Threads", "QueueSize", "Tasks", "Time(s)", "Throughput", "AvgResponse(s)");
    printf("%.80s\n", "----------------------------------------"
                      "----------------------------------------");
    
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 3; j++) {
            BenchmarkResult result = benchmark_thread_pool(
                thread_counts[i], queue_sizes[j], task_count);
            
            printf("%-8d %-10d %-10d %-12.3f %-15.2f %-15.6f\n",
                   result.num_threads, result.queue_size, result.num_tasks,
                   result.total_time, result.throughput, result.avg_response_time);
        }
    }
}
```

## Learning Objectives

After completing this project, you should understand:
- Thread pool design patterns and implementation
- Task queue management and scheduling
- Worker thread lifecycle management
- Resource management and cleanup in multi-threaded systems
- Performance optimization for concurrent task processing
- Dynamic scaling and load balancing techniques

## Extensions

1. **Work Stealing Implementation**
   - Each thread has its own queue
   - Idle threads steal work from busy threads

2. **Future/Promise Pattern**
   - Return future objects for submitted tasks
   - Allow waiting for task completion and result retrieval

3. **Task Dependencies**
   - Support for task dependencies and execution ordering
   - DAG-based task scheduling

4. **Distributed Thread Pool**
   - Scale across multiple processes or machines
   - Network-based task distribution

## Assessment Criteria

- **Correctness (30%)**: Proper thread synchronization and task execution
- **Performance (25%)**: Efficient resource utilization and scalability
- **Features (25%)**: Implementation of advanced features
- **Code Quality (20%)**: Clean, maintainable, well-documented code

## Next Project
[Project 5: Parallel Computation](Project5_Parallel_Computation.md)
