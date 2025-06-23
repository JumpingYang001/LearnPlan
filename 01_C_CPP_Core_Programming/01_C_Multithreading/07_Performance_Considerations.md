# Performance Considerations

*Duration: 1 week*

# Performance Considerations: Mastering Multi-Threaded Performance

*Duration: 1 week*

## Overview

Performance optimization in multi-threaded applications is both an art and a science, requiring deep understanding of hardware architectures, operating system behavior, and algorithmic complexity. This comprehensive guide will transform you from a basic threading programmer into a performance-conscious systems architect capable of building high-performance concurrent applications.

### The Performance Challenge Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                Threading Performance Stack                      │
├─────────────────────────────────────────────────────────────────┤
│  Application Level    │  Algorithm Design & Data Structures    │
├─────────────────────────────────────────────────────────────────┤
│  Threading Level      │  Synchronization & Task Distribution   │
├─────────────────────────────────────────────────────────────────┤
│  Runtime Level        │  Thread Pools & Work Scheduling        │
├─────────────────────────────────────────────────────────────────┤
│  OS Level            │  Context Switching & CPU Scheduling     │
├─────────────────────────────────────────────────────────────────┤
│  Hardware Level      │  Cache Coherency & Memory Hierarchy     │
└─────────────────────────────────────────────────────────────────┘
```

Understanding performance requires mastering each level of this stack. A single misconfiguration at any level can completely negate optimizations at higher levels.

### Performance Metrics That Matter

**Throughput vs. Latency Trade-offs:**
```c
// High throughput, higher latency
typedef struct {
    task_t* tasks[BATCH_SIZE];
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
} batch_queue_t;

// Low latency, potentially lower throughput
typedef struct {
    atomic_ptr_t head;
    atomic_ptr_t tail;
} lockfree_queue_t;
```

**Key Performance Indicators (KPIs):**
1. **Throughput**: Operations per second under load
2. **Latency**: Response time for individual operations
3. **Scalability**: Performance improvement with additional cores
4. **Efficiency**: Resource utilization (CPU, memory, bandwidth)
5. **Predictability**: Variance in performance under different conditions

### The Cost of Threading: A Reality Check

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <stdatomic.h>

// Performance measurement framework
typedef struct {
    struct timespec start;
    struct timespec end;
    const char* operation_name;
    unsigned long operations;
} perf_timer_t;

void perf_start(perf_timer_t* timer, const char* name, unsigned long ops) {
    timer->operation_name = name;
    timer->operations = ops;
    clock_gettime(CLOCK_MONOTONIC, &timer->start);
}

void perf_end_and_report(perf_timer_t* timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->end);
    
    double elapsed = (timer->end.tv_sec - timer->start.tv_sec) + 
                     (timer->end.tv_nsec - timer->start.tv_nsec) / 1e9;
    
    double ops_per_sec = timer->operations / elapsed;
    double ns_per_op = (elapsed * 1e9) / timer->operations;
    
    printf("%-30s: %10lu ops in %8.3f sec = %12.0f ops/sec (%8.2f ns/op)\n",
           timer->operation_name, timer->operations, elapsed, ops_per_sec, ns_per_op);
}

// Demonstrate the shocking cost of naive threading
void demonstrate_threading_overhead() {
    const unsigned long OPERATIONS = 1000000;
    perf_timer_t timer;
    
    printf("Threading Overhead Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    // Baseline: Simple function call
    perf_start(&timer, "Function call baseline", OPERATIONS);
    volatile int result = 0;
    for (unsigned long i = 0; i < OPERATIONS; i++) {
        result += 1; // Trivial operation
    }
    perf_end_and_report(&timer);
    
    // Atomic operations
    atomic_int atomic_counter = 0;
    perf_start(&timer, "Atomic increment", OPERATIONS);
    for (unsigned long i = 0; i < OPERATIONS; i++) {
        atomic_fetch_add(&atomic_counter, 1);
    }
    perf_end_and_report(&timer);
    
    // Mutex operations
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    int counter = 0;
    perf_start(&timer, "Mutex lock/unlock", OPERATIONS);
    for (unsigned long i = 0; i < OPERATIONS; i++) {
        pthread_mutex_lock(&mutex);
        counter++;
        pthread_mutex_unlock(&mutex);
    }
    perf_end_and_report(&timer);
    
    pthread_mutex_destroy(&mutex);
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("Key Insights:\n");
    printf("• Atomic operations: ~10-50x slower than simple operations\n");
    printf("• Mutex operations: ~100-1000x slower than simple operations\n");
    printf("• Thread creation: ~10,000-100,000x slower than function calls\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
}
```

### Hardware Architecture Impact

Modern CPU architecture significantly affects threading performance:

```c
// Demonstrate NUMA effects
#ifdef __linux__
#include <numa.h>
#include <numaif.h>

void demonstrate_numa_effects() {
    if (numa_available() < 0) {
        printf("NUMA not available on this system\n");
        return;
    }
    
    int num_nodes = numa_num_configured_nodes();
    printf("NUMA Analysis (Nodes: %d)\n", num_nodes);
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    const size_t MEMORY_SIZE = 100 * 1024 * 1024; // 100MB
    const unsigned long ITERATIONS = 1000000;
    
    for (int node = 0; node < num_nodes; node++) {
        // Allocate memory on specific NUMA node
        void* memory = numa_alloc_onnode(MEMORY_SIZE, node);
        if (!memory) continue;
        
        // Bind current thread to different nodes and measure access time
        for (int cpu_node = 0; cpu_node < num_nodes; cpu_node++) {
            struct bitmask* cpu_mask = numa_allocate_cpumask();
            numa_node_to_cpus(cpu_node, cpu_mask);
            numa_sched_setaffinity(0, cpu_mask);
            
            perf_timer_t timer;
            perf_start(&timer, "", ITERATIONS);
            
            volatile char* mem_ptr = (volatile char*)memory;
            for (unsigned long i = 0; i < ITERATIONS; i++) {
                mem_ptr[i % MEMORY_SIZE] = (char)i;
            }
            
            clock_gettime(CLOCK_MONOTONIC, &timer.end);
            double elapsed = (timer.end.tv_sec - timer.start.tv_sec) + 
                            (timer.end.tv_nsec - timer.start.tv_nsec) / 1e9;
            
            printf("Memory Node %d, CPU Node %d: %8.2f ns/access %s\n",
                   node, cpu_node, (elapsed * 1e9) / ITERATIONS,
                   (node == cpu_node) ? "(LOCAL)" : "(REMOTE)");
            
            numa_free_cpumask(cpu_mask);
        }
        
        numa_free(memory, MEMORY_SIZE);
        printf("───────────────────────────────────────────────────────────────────\n");
    }
}
#endif

// Cache hierarchy demonstration
void demonstrate_cache_hierarchy() {
    printf("Cache Hierarchy Performance Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    // Test different memory access patterns
    const size_t sizes[] = {
        1024,           // L1 cache size (32KB typical)
        32 * 1024,      // L1 cache boundary
        256 * 1024,     // L2 cache size (256KB typical)
        8 * 1024 * 1024, // L3 cache size (8MB typical)
        64 * 1024 * 1024, // Beyond L3 cache
        512 * 1024 * 1024 // Main memory
    };
    
    const char* labels[] = {
        "L1 Cache", "L1 Boundary", "L2 Cache", 
        "L3 Cache", "Beyond L3", "Main Memory"
    };
    
    for (int i = 0; i < 6; i++) {
        size_t size = sizes[i];
        volatile int* array = malloc(size);
        if (!array) continue;
        
        // Initialize memory
        for (size_t j = 0; j < size / sizeof(int); j++) {
            array[j] = j;
        }
        
        perf_timer_t timer;
        const unsigned long iterations = 10000000;
        perf_start(&timer, labels[i], iterations);
        
        volatile int sum = 0;
        for (unsigned long iter = 0; iter < iterations; iter++) {
            sum += array[iter % (size / sizeof(int))];
        }
        
        perf_end_and_report(&timer);
        free((void*)array);
    }
    
    printf("═══════════════════════════════════════════════════════════════════\n\n");
}
```

### Threading Models Performance Comparison

```c
// Compare different threading models
typedef struct {
    const char* model_name;
    void (*benchmark_func)(unsigned long operations);
    double operations_per_second;
    double memory_usage_mb;
    int cpu_cores_used;
} threading_model_t;

// Single-threaded baseline
void benchmark_single_threaded(unsigned long operations) {
    volatile unsigned long result = 0;
    for (unsigned long i = 0; i < operations; i++) {
        result += i * i;
    }
}

// Multiple threads with shared data (high contention)
static pthread_mutex_t shared_mutex = PTHREAD_MUTEX_INITIALIZER;
static unsigned long shared_counter = 0;

void* high_contention_worker(void* arg) {
    unsigned long iterations = *(unsigned long*)arg;
    
    for (unsigned long i = 0; i < iterations; i++) {
        pthread_mutex_lock(&shared_mutex);
        shared_counter += i * i;
        pthread_mutex_unlock(&shared_mutex);
    }
    return NULL;
}

void benchmark_high_contention(unsigned long operations) {
    const int num_threads = 4;
    pthread_t threads[num_threads];
    unsigned long ops_per_thread = operations / num_threads;
    
    shared_counter = 0;
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, high_contention_worker, &ops_per_thread);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

// Thread-local computation (minimal contention)
void* low_contention_worker(void* arg) {
    unsigned long iterations = *(unsigned long*)arg;
    volatile unsigned long local_result = 0;
    
    for (unsigned long i = 0; i < iterations; i++) {
        local_result += i * i;
    }
    return NULL;
}

void benchmark_low_contention(unsigned long operations) {
    const int num_threads = 4;
    pthread_t threads[num_threads];
    unsigned long ops_per_thread = operations / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, low_contention_worker, &ops_per_thread);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

void compare_threading_models() {
    const unsigned long OPERATIONS = 10000000;
    
    threading_model_t models[] = {
        {"Single-threaded", benchmark_single_threaded, 0, 0, 1},
        {"High contention", benchmark_high_contention, 0, 0, 4},
        {"Low contention", benchmark_low_contention, 0, 0, 4}
    };
    
    printf("Threading Models Performance Comparison\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    
    for (int i = 0; i < 3; i++) {
        perf_timer_t timer;
        perf_start(&timer, models[i].model_name, OPERATIONS);
        
        models[i].benchmark_func(OPERATIONS);
        
        perf_end_and_report(&timer);
    }
    
    printf("═══════════════════════════════════════════════════════════════════\n\n");
}
```

Understanding performance characteristics of multi-threaded applications is crucial for building efficient systems. This section covers thread overhead, optimization techniques, and performance analysis through hands-on measurement and real-world scenarios.

## Advanced Thread Overhead Analysis

Thread overhead is far more complex than simple creation and destruction costs. Modern applications must consider memory usage, context switching, cache pollution, and synchronization overhead to build truly efficient systems.

### Comprehensive Thread Creation Analysis

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <string.h>

// Advanced timing utilities
typedef struct {
    struct timespec start;
    struct timespec end;
    const char* description;
    unsigned long operations;
    size_t memory_before;
    size_t memory_after;
} advanced_timer_t;

size_t get_memory_usage() {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;
    
    char line[256];
    size_t vm_rss = 0;
    
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %zu kB", &vm_rss);
            break;
        }
    }
    
    fclose(file);
    return vm_rss * 1024; // Convert to bytes
}

void advanced_timer_start(advanced_timer_t* timer, const char* desc, unsigned long ops) {
    timer->description = desc;
    timer->operations = ops;
    timer->memory_before = get_memory_usage();
    clock_gettime(CLOCK_MONOTONIC, &timer->start);
}

void advanced_timer_end(advanced_timer_t* timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->end);
    timer->memory_after = get_memory_usage();
    
    double elapsed = (timer->end.tv_sec - timer->start.tv_sec) + 
                     (timer->end.tv_nsec - timer->start.tv_nsec) / 1e9;
    
    double ops_per_sec = timer->operations / elapsed;
    double ns_per_op = (elapsed * 1e9) / timer->operations;
    double memory_per_op = (double)(timer->memory_after - timer->memory_before) / timer->operations;
    
    printf("%-35s: %8lu ops, %10.2f ops/sec, %8.2f ns/op, %8.2f bytes/op\n",
           timer->description, timer->operations, ops_per_sec, ns_per_op, memory_per_op);
}

// Different thread creation patterns
void* minimal_thread(void* arg) {
    return NULL;
}

void* stack_heavy_thread(void* arg) {
    // Large stack allocation
    char buffer[1024 * 1024]; // 1MB stack
    memset(buffer, 0, sizeof(buffer));
    return NULL;
}

void* cpu_bound_thread(void* arg) {
    // CPU-intensive work
    volatile long sum = 0;
    for (int i = 0; i < 1000000; i++) {
        sum += i * i;
    }
    return NULL;
}

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int value;
} sync_data_t;

void* sync_heavy_thread(void* arg) {
    sync_data_t* data = (sync_data_t*)arg;
    
    pthread_mutex_lock(&data->mutex);
    data->value++;
    pthread_cond_signal(&data->cond);
    pthread_mutex_unlock(&data->mutex);
    
    return NULL;
}

void comprehensive_thread_overhead_analysis() {
    printf("Comprehensive Thread Overhead Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    
    const int NUM_THREADS = 1000;
    pthread_t threads[NUM_THREADS];
    advanced_timer_t timer;
    
    // 1. Minimal thread overhead
    advanced_timer_start(&timer, "Minimal thread (create+join)", NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, minimal_thread, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    advanced_timer_end(&timer);
    
    // 2. Stack-heavy threads
    advanced_timer_start(&timer, "Stack-heavy thread (1MB stack)", NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, stack_heavy_thread, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    advanced_timer_end(&timer);
    
    // 3. CPU-bound threads
    advanced_timer_start(&timer, "CPU-bound thread", NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, cpu_bound_thread, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    advanced_timer_end(&timer);
    
    // 4. Synchronization-heavy threads
    sync_data_t sync_data = {
        .mutex = PTHREAD_MUTEX_INITIALIZER,
        .cond = PTHREAD_COND_INITIALIZER,
        .value = 0
    };
    
    advanced_timer_start(&timer, "Sync-heavy thread", NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, sync_heavy_thread, &sync_data);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    advanced_timer_end(&timer);
    
    pthread_mutex_destroy(&sync_data.mutex);
    pthread_cond_destroy(&sync_data.cond);
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}

// Thread pool vs. thread-per-task comparison
typedef struct {
    pthread_t* threads;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_cond;
    void (**tasks)(void*);
    void** task_args;
    int queue_size;
    int queue_head;
    int queue_tail;
    int queue_count;
    int num_threads;
    int shutdown;
} simple_thread_pool_t;

void* thread_pool_worker(void* arg) {
    simple_thread_pool_t* pool = (simple_thread_pool_t*)arg;
    
    while (1) {
        pthread_mutex_lock(&pool->queue_mutex);
        
        while (pool->queue_count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->queue_cond, &pool->queue_mutex);
        }
        
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->queue_mutex);
            break;
        }
        
        void (*task)(void*) = pool->tasks[pool->queue_head];
        void* task_arg = pool->task_args[pool->queue_head];
        
        pool->queue_head = (pool->queue_head + 1) % pool->queue_size;
        pool->queue_count--;
        
        pthread_mutex_unlock(&pool->queue_mutex);
        
        task(task_arg);
    }
    
    return NULL;
}

simple_thread_pool_t* create_thread_pool(int num_threads, int queue_size) {
    simple_thread_pool_t* pool = malloc(sizeof(simple_thread_pool_t));
    if (!pool) return NULL;
    
    pool->threads = malloc(num_threads * sizeof(pthread_t));
    pool->tasks = malloc(queue_size * sizeof(void(*)(void*)));
    pool->task_args = malloc(queue_size * sizeof(void*));
    
    if (!pool->threads || !pool->tasks || !pool->task_args) {
        free(pool->threads);
        free(pool->tasks);
        free(pool->task_args);
        free(pool);
        return NULL;
    }
    
    pthread_mutex_init(&pool->queue_mutex, NULL);
    pthread_cond_init(&pool->queue_cond, NULL);
    
    pool->queue_size = queue_size;
    pool->queue_head = 0;
    pool->queue_tail = 0;
    pool->queue_count = 0;
    pool->num_threads = num_threads;
    pool->shutdown = 0;
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, thread_pool_worker, pool);
    }
    
    return pool;
}

void thread_pool_submit(simple_thread_pool_t* pool, void (*task)(void*), void* arg) {
    pthread_mutex_lock(&pool->queue_mutex);
    
    pool->tasks[pool->queue_tail] = task;
    pool->task_args[pool->queue_tail] = arg;
    pool->queue_tail = (pool->queue_tail + 1) % pool->queue_size;
    pool->queue_count++;
    
    pthread_cond_signal(&pool->queue_cond);
    pthread_mutex_unlock(&pool->queue_mutex);
}

void destroy_thread_pool(simple_thread_pool_t* pool) {
    if (!pool) return;
    
    pthread_mutex_lock(&pool->queue_mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->queue_cond);
    pthread_mutex_unlock(&pool->queue_mutex);
    
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    pthread_mutex_destroy(&pool->queue_mutex);
    pthread_cond_destroy(&pool->queue_cond);
    
    free(pool->threads);
    free(pool->tasks);
    free(pool->task_args);
    free(pool);
}

void simple_task(void* arg) {
    volatile int* result = (volatile int*)arg;
    *result = 42;
}

void compare_thread_models() {
    printf("Thread Creation Model Comparison\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    
    const int NUM_TASKS = 10000;
    volatile int results[NUM_TASKS];
    advanced_timer_t timer;
    
    // Thread-per-task model
    advanced_timer_start(&timer, "Thread-per-task (1000 tasks)", 1000);
    pthread_t threads[1000];
    for (int i = 0; i < 1000; i++) {
        pthread_create(&threads[i], NULL, (void*(*)(void*))simple_task, (void*)&results[i]);
    }
    for (int i = 0; i < 1000; i++) {
        pthread_join(threads[i], NULL);
    }
    advanced_timer_end(&timer);
    
    // Thread pool model
    simple_thread_pool_t* pool = create_thread_pool(4, 1000);
    
    advanced_timer_start(&timer, "Thread pool (4 threads, 10k tasks)", NUM_TASKS);
    for (int i = 0; i < NUM_TASKS; i++) {
        thread_pool_submit(pool, simple_task, (void*)&results[i]);
    }
    
    // Wait for all tasks to complete
    while (1) {
        pthread_mutex_lock(&pool->queue_mutex);
        int count = pool->queue_count;
        pthread_mutex_unlock(&pool->queue_mutex);
        if (count == 0) break;
        usleep(1000);
    }
    advanced_timer_end(&timer);
    
    destroy_thread_pool(pool);
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}
```

### Advanced Memory Overhead Analysis

```c
#include <malloc.h>

// Detailed memory analysis
typedef struct {
    size_t total_memory;
    size_t heap_memory;
    size_t stack_memory;
    size_t shared_memory;
    int num_threads;
} memory_analysis_t;

void get_detailed_memory_info(memory_analysis_t* info) {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return;
    
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %zu kB", &info->total_memory);
            info->total_memory *= 1024;
        } else if (strncmp(line, "VmData:", 7) == 0) {
            sscanf(line, "VmData: %zu kB", &info->heap_memory);
            info->heap_memory *= 1024;
        } else if (strncmp(line, "VmStk:", 6) == 0) {
            sscanf(line, "VmStk: %zu kB", &info->stack_memory);
            info->stack_memory *= 1024;
        } else if (strncmp(line, "Threads:", 8) == 0) {
            sscanf(line, "Threads: %d", &info->num_threads);
        }
    }
    
    fclose(file);
}

void* memory_analyzer_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    // Allocate different amounts of memory per thread
    size_t allocation_sizes[] = {1024, 4096, 16384, 65536, 262144}; // 1KB to 256KB
    void* allocations[5];
    
    for (int i = 0; i < 5; i++) {
        allocations[i] = malloc(allocation_sizes[i]);
        if (allocations[i]) {
            memset(allocations[i], thread_id, allocation_sizes[i]);
        }
    }
    
    // Hold memory for measurement
    sleep(1);
    
    // Cleanup
    for (int i = 0; i < 5; i++) {
        free(allocations[i]);
    }
    
    return NULL;
}

void analyze_memory_scaling() {
    printf("Memory Scaling Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    printf("Threads | Total Memory | Heap Memory | Stack Memory | Memory/Thread\n");
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    
    memory_analysis_t baseline;
    get_detailed_memory_info(&baseline);
    
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    
    for (int t = 0; t < 8; t++) {
        int num_threads = thread_counts[t];
        pthread_t threads[num_threads];
        int thread_ids[num_threads];
        
        // Create threads
        for (int i = 0; i < num_threads; i++) {
            thread_ids[i] = i;
            pthread_create(&threads[i], NULL, memory_analyzer_thread, &thread_ids[i]);
        }
        
        // Measure memory usage
        sleep(2); // Let threads allocate memory
        memory_analysis_t current;
        get_detailed_memory_info(&current);
        
        size_t total_delta = current.total_memory - baseline.total_memory;
        size_t heap_delta = current.heap_memory - baseline.heap_memory;
        size_t stack_delta = current.stack_memory - baseline.stack_memory;
        double memory_per_thread = (double)total_delta / num_threads;
        
        printf("%7d | %10.2f MB | %9.2f MB | %10.2f MB | %10.2f KB\n",
               num_threads,
               total_delta / (1024.0 * 1024.0),
               heap_delta / (1024.0 * 1024.0),
               stack_delta / (1024.0 * 1024.0),
               memory_per_thread / 1024.0);
        
        // Join threads
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
        
        sleep(1); // Let memory settle
    }
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}

// Stack size optimization analysis
void analyze_stack_sizes() {
    printf("Stack Size Optimization Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    
    size_t stack_sizes[] = {
        PTHREAD_STACK_MIN,
        16 * 1024,    // 16KB
        64 * 1024,    // 64KB
        256 * 1024,   // 256KB
        1024 * 1024,  // 1MB
        2 * 1024 * 1024, // 2MB
        8 * 1024 * 1024  // 8MB (default on many systems)
    };
    
    const char* stack_labels[] = {
        "PTHREAD_STACK_MIN", "16KB", "64KB", "256KB", "1MB", "2MB", "8MB"
    };
    
    for (int i = 0; i < 7; i++) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, stack_sizes[i]);
        
        const int num_threads = 100;
        pthread_t threads[num_threads];
        
        memory_analysis_t before, after;
        get_detailed_memory_info(&before);
        
        advanced_timer_t timer;
        advanced_timer_start(&timer, stack_labels[i], num_threads);
        
        for (int j = 0; j < num_threads; j++) {
            int result = pthread_create(&threads[j], &attr, minimal_thread, NULL);
            if (result != 0) {
                printf("Failed to create thread with stack size %s\n", stack_labels[i]);
                break;
            }
        }
        
        for (int j = 0; j < num_threads; j++) {
            pthread_join(threads[j], NULL);
        }
        
        advanced_timer_end(&timer);
        
        get_detailed_memory_info(&after);
        double peak_memory = (after.total_memory - before.total_memory) / (1024.0 * 1024.0);
        printf("  Peak memory usage: %.2f MB\n", peak_memory);
        
        pthread_attr_destroy(&attr);
    }
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}

## Advanced Context Switching Analysis

Context switching is one of the most significant performance bottlenecks in multi-threaded applications. Understanding when, why, and how context switches occur is essential for building high-performance systems.

### Comprehensive Context Switch Measurement

```c
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdatomic.h>

// Get thread ID (Linux-specific)
static inline pid_t gettid() {
    return syscall(SYS_gettid);
}

// Context switch measurement framework
typedef struct {
    atomic_ulong voluntary_switches;
    atomic_ulong involuntary_switches;
    atomic_ulong total_runtime_ns;
    atomic_ulong context_switch_overhead_ns;
    int thread_id;
    int cpu_id;
} context_switch_stats_t;

// Read context switch statistics from /proc
void read_context_switches(pid_t tid, unsigned long* voluntary, unsigned long* involuntary) {
    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/status", tid);
    
    FILE* file = fopen(path, "r");
    if (!file) return;
    
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "voluntary_ctxt_switches:", 24) == 0) {
            sscanf(line, "voluntary_ctxt_switches: %lu", voluntary);
        } else if (strncmp(line, "nonvoluntary_ctxt_switches:", 27) == 0) {
            sscanf(line, "nonvoluntary_ctxt_switches: %lu", involuntary);
        }
    }
    
    fclose(file);
}

// Different workload patterns that cause context switches
volatile atomic_int shared_counter = 0;
pthread_mutex_t contention_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t contention_cond = PTHREAD_COND_INITIALIZER;

// High contention workload
void* high_contention_worker(void* arg) {
    context_switch_stats_t* stats = (context_switch_stats_t*)arg;
    pid_t tid = gettid();
    
    unsigned long vol_start, invol_start, vol_end, invol_end;
    read_context_switches(tid, &vol_start, &invol_start);
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Highly contended operations
    for (int i = 0; i < 10000; i++) {
        pthread_mutex_lock(&contention_mutex);
        shared_counter++;
        // Simulate some work while holding the lock
        for (volatile int j = 0; j < 1000; j++);
        pthread_mutex_unlock(&contention_mutex);
        
        // Force context switch opportunity
        if (i % 100 == 0) {
            sched_yield();
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    read_context_switches(tid, &vol_end, &invol_end);
    
    unsigned long runtime_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000UL +
                              (end_time.tv_nsec - start_time.tv_nsec);
    
    atomic_store(&stats->voluntary_switches, vol_end - vol_start);
    atomic_store(&stats->involuntary_switches, invol_end - invol_start);
    atomic_store(&stats->total_runtime_ns, runtime_ns);
    
    return NULL;
}

// CPU-bound workload
void* cpu_bound_worker(void* arg) {
    context_switch_stats_t* stats = (context_switch_stats_t*)arg;
    pid_t tid = gettid();
    
    unsigned long vol_start, invol_start, vol_end, invol_end;
    read_context_switches(tid, &vol_start, &invol_start);
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Pure CPU work with no system calls
    volatile long sum = 0;
    for (long i = 0; i < 100000000L; i++) {
        sum += i * i + i * 3 + 7;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    read_context_switches(tid, &vol_end, &invol_end);
    
    unsigned long runtime_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000UL +
                              (end_time.tv_nsec - start_time.tv_nsec);
    
    atomic_store(&stats->voluntary_switches, vol_end - vol_start);
    atomic_store(&stats->involuntary_switches, invol_end - invol_start);
    atomic_store(&stats->total_runtime_ns, runtime_ns);
    
    return NULL;
}

// I/O bound workload
void* io_bound_worker(void* arg) {
    context_switch_stats_t* stats = (context_switch_stats_t*)arg;
    pid_t tid = gettid();
    
    unsigned long vol_start, invol_start, vol_end, invol_end;
    read_context_switches(tid, &vol_start, &invol_start);
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // I/O operations that cause context switches
    for (int i = 0; i < 1000; i++) {
        // Simulate I/O with sleep
        struct timespec io_delay = {0, 1000000}; // 1ms
        nanosleep(&io_delay, NULL);
        
        // Some computation between I/O
        volatile int work = 0;
        for (int j = 0; j < 10000; j++) {
            work += j;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    read_context_switches(tid, &vol_end, &invol_end);
    
    unsigned long runtime_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000UL +
                              (end_time.tv_nsec - start_time.tv_nsec);
    
    atomic_store(&stats->voluntary_switches, vol_end - vol_start);
    atomic_store(&stats->involuntary_switches, invol_end - invol_start);
    atomic_store(&stats->total_runtime_ns, runtime_ns);
    
    return NULL;
}

void analyze_context_switching_patterns() {
    printf("Context Switching Pattern Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    printf("Workload Type    | Vol. Switches | Invol. Switches | Runtime (ms) | Switches/ms\n");
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    
    typedef struct {
        const char* name;
        void* (*worker_func)(void*);
    } workload_type_t;
    
    workload_type_t workloads[] = {
        {"High Contention", high_contention_worker},
        {"CPU Bound", cpu_bound_worker},
        {"I/O Bound", io_bound_worker}
    };
    
    for (int w = 0; w < 3; w++) {
        const int num_threads = 4;
        pthread_t threads[num_threads];
        context_switch_stats_t stats[num_threads];
        
        // Initialize stats
        for (int i = 0; i < num_threads; i++) {
            memset(&stats[i], 0, sizeof(context_switch_stats_t));
            stats[i].thread_id = i;
        }
        
        // Create and run threads
        for (int i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], NULL, workloads[w].worker_func, &stats[i]);
        }
        
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
        
        // Aggregate statistics
        unsigned long total_vol = 0, total_invol = 0, total_runtime = 0;
        for (int i = 0; i < num_threads; i++) {
            total_vol += atomic_load(&stats[i].voluntary_switches);
            total_invol += atomic_load(&stats[i].involuntary_switches);
            total_runtime += atomic_load(&stats[i].total_runtime_ns);
        }
        
        double avg_runtime_ms = (total_runtime / num_threads) / 1000000.0;
        double switches_per_ms = (total_vol + total_invol) / avg_runtime_ms;
        
        printf("%-16s | %13lu | %15lu | %12.2f | %11.2f\n",
               workloads[w].name, total_vol, total_invol, avg_runtime_ms, switches_per_ms);
    }
    
    printf("═══════════════════════════════════════════════════════────────────────────════════\n\n");
}

// Measure pure context switch overhead
void measure_context_switch_overhead() {
    printf("Context Switch Overhead Measurement\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    
    // Use pipe communication to force context switches
    int pipe1[2], pipe2[2];
    pipe(pipe1);
    pipe(pipe2);
    
    const int iterations = 100000;
    struct timespec start_time, end_time;
    
    if (fork() == 0) {
        // Child process
        char byte;
        for (int i = 0; i < iterations; i++) {
            read(pipe1[0], &byte, 1);  // Read from parent
            write(pipe2[1], &byte, 1); // Write back to parent
        }
        exit(0);
    } else {
        // Parent process
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        char byte = 'X';
        for (int i = 0; i < iterations; i++) {
            write(pipe1[1], &byte, 1); // Write to child
            read(pipe2[0], &byte, 1);  // Read from child
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        
        double elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000.0 +
                           (end_time.tv_nsec - start_time.tv_nsec);
        
        double context_switch_time = elapsed_ns / (iterations * 2); // 2 switches per iteration
        
        printf("Context switch overhead: %.2f nanoseconds per switch\n", context_switch_time);
        printf("Context switches per second: %.0f\n", 1000000000.0 / context_switch_time);
        
        wait(NULL); // Wait for child process
    }
    
    close(pipe1[0]); close(pipe1[1]);
    close(pipe2[0]); close(pipe2[1]);
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}
```

### Advanced CPU Affinity and NUMA Optimization

```c
#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <numa.h>

// NUMA-aware thread placement
typedef struct {
    int thread_id;
    int numa_node;
    int cpu_core;
    unsigned long operations_completed;
    double performance_score;
} numa_thread_info_t;

void* numa_aware_worker(void* arg) {
    numa_thread_info_t* info = (numa_thread_info_t*)arg;
    
    // Set CPU affinity to specific core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(info->cpu_core, &cpuset);
    
    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        printf("Failed to set CPU affinity for thread %d\n", info->thread_id);
        return NULL;
    }
    
    // Allocate memory on the same NUMA node
    const size_t memory_size = 64 * 1024 * 1024; // 64MB
    void* memory = numa_alloc_onnode(memory_size, info->numa_node);
    if (!memory) {
        printf("Failed to allocate NUMA memory for thread %d\n", info->thread_id);
        return NULL;
    }
    
    // Measure performance of memory-intensive operations
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    volatile char* mem_ptr = (volatile char*)memory;
    const unsigned long operations = 10000000;
    
    for (unsigned long i = 0; i < operations; i++) {
        mem_ptr[i % memory_size] = (char)(i & 0xFF);
        if (i % 1000000 == 0) {
            // Simulate some computation
            volatile int compute = 0;
            for (int j = 0; j < 1000; j++) {
                compute += j * j;
            }
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    info->operations_completed = operations;
    info->performance_score = operations / elapsed;
    
    numa_free(memory, memory_size);
    return NULL;
}

void analyze_numa_performance() {
    if (numa_available() < 0) {
        printf("NUMA not available on this system\n");
        return;
    }
    
    int num_nodes = numa_num_configured_nodes();
    int num_cpus = numa_num_configured_cpus();
    
    printf("NUMA Performance Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    printf("System: %d NUMA nodes, %d CPU cores\n", num_nodes, num_cpus);
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    printf("Thread | NUMA Node | CPU Core | Operations/sec | Performance Ratio\n");
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    
    const int threads_per_node = 2;
    const int total_threads = num_nodes * threads_per_node;
    
    pthread_t threads[total_threads];
    numa_thread_info_t thread_info[total_threads];
    
    int thread_idx = 0;
    double baseline_performance = 0;
    
    // Create threads distributed across NUMA nodes
    for (int node = 0; node < num_nodes; node++) {
        struct bitmask* cpu_mask = numa_allocate_cpumask();
        numa_node_to_cpus(node, cpu_mask);
        
        // Find CPUs for this NUMA node
        int node_cpus[threads_per_node];
        int cpu_count = 0;
        
        for (int cpu = 0; cpu < num_cpus && cpu_count < threads_per_node; cpu++) {
            if (numa_bitmask_isbitset(cpu_mask, cpu)) {
                node_cpus[cpu_count++] = cpu;
            }
        }
        
        // Create threads for this node
        for (int t = 0; t < threads_per_node && t < cpu_count; t++) {
            thread_info[thread_idx].thread_id = thread_idx;
            thread_info[thread_idx].numa_node = node;
            thread_info[thread_idx].cpu_core = node_cpus[t];
            
            pthread_create(&threads[thread_idx], NULL, numa_aware_worker, 
                          &thread_info[thread_idx]);
            thread_idx++;
        }
        
        numa_free_cpumask(cpu_mask);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < thread_idx; i++) {
        pthread_join(threads[i], NULL);
        
        if (baseline_performance == 0) {
            baseline_performance = thread_info[i].performance_score;
        }
        
        double ratio = thread_info[i].performance_score / baseline_performance;
        
        printf("%6d | %9d | %8d | %14.0f | %13.2f\n",
               thread_info[i].thread_id,
               thread_info[i].numa_node,
               thread_info[i].cpu_core,
               thread_info[i].performance_score,
               ratio);
    }
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}

// CPU cache warming effects
void analyze_cache_warming_effects() {
    printf("CPU Cache Warming Effects Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    
    const size_t test_sizes[] = {
        16 * 1024,        // 16KB - L1 cache
        256 * 1024,       // 256KB - L2 cache  
        8 * 1024 * 1024,  // 8MB - L3 cache
        64 * 1024 * 1024  // 64MB - Main memory
    };
    
    const char* size_names[] = {"L1 Cache", "L2 Cache", "L3 Cache", "Main Memory"};
    
    for (int s = 0; s < 4; s++) {
        size_t size = test_sizes[s];
        volatile int* array = malloc(size);
        if (!array) continue;
        
        int array_size = size / sizeof(int);
        
        // Initialize array
        for (int i = 0; i < array_size; i++) {
            array[i] = i;
        }
        
        printf("Testing %s (%zu KB):\n", size_names[s], size / 1024);
        
        // Cold cache test
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        volatile long sum = 0;
        for (int iter = 0; iter < 10; iter++) {
            for (int i = 0; i < array_size; i++) {
                sum += array[i];
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double cold_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        // Warm cache test (repeated access)
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        sum = 0;
        for (int iter = 0; iter < 100; iter++) {
            for (int i = 0; i < array_size; i++) {
                sum += array[i];
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double warm_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        double cold_ns_per_access = (cold_time * 1e9) / (10 * array_size);
        double warm_ns_per_access = (warm_time * 1e9) / (100 * array_size);
        double speedup = cold_ns_per_access / warm_ns_per_access;
        
        printf("  Cold cache: %.2f ns/access\n", cold_ns_per_access);
        printf("  Warm cache: %.2f ns/access\n", warm_ns_per_access);
        printf("  Speedup: %.2fx\n", speedup);
        printf("  ─────────────────────────────────────────────────────────\n");
        
        free((void*)array);
    }
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}

## Advanced Cache Coherency and Memory Performance

Cache coherency issues are among the most subtle and devastating performance problems in multi-threaded applications. Understanding cache behavior is essential for achieving optimal performance.

### Comprehensive False Sharing Analysis

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <stdatomic.h>

#define CACHE_LINE_SIZE 64
#define NUM_THREADS 8
#define ITERATIONS 50000000

// Performance measurement with hardware counters simulation
typedef struct {
    unsigned long cache_misses;
    unsigned long cache_references;
    unsigned long instructions;
    unsigned long cycles;
    double cache_miss_rate;
    double ipc; // Instructions per cycle
} perf_counters_t;

// False sharing demonstration structures
typedef struct {
    // BAD: All counters share cache lines
    volatile long counters[NUM_THREADS];
} false_sharing_data_t;

typedef struct {
    // GOOD: Each counter has its own cache line
    volatile long counter;
    char padding[CACHE_LINE_SIZE - sizeof(long)];
} cache_aligned_counter_t;

typedef struct {
    cache_aligned_counter_t counters[NUM_THREADS];
} cache_friendly_data_t;

// Thread-local data (best)
typedef struct {
    volatile long local_counter;
    atomic_long* global_counter;
    int thread_id;
} thread_local_data_t;

// Global data structures
static false_sharing_data_t false_sharing_data;
static cache_friendly_data_t cache_friendly_data;
static atomic_long global_atomic_counter = 0;

// False sharing workload
void* false_sharing_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < ITERATIONS; i++) {
        false_sharing_data.counters[thread_id]++;
        
        // Add some computation to make the effect more visible
        if (i % 10000 == 0) {
            volatile int work = 0;
            for (int j = 0; j < 100; j++) {
                work += j * j;
            }
        }
    }
    
    return NULL;
}

// Cache-friendly workload
void* cache_friendly_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < ITERATIONS; i++) {
        cache_friendly_data.counters[thread_id].counter++;
        
        // Same computation as false sharing version
        if (i % 10000 == 0) {
            volatile int work = 0;
            for (int j = 0; j < 100; j++) {
                work += j * j;
            }
        }
    }
    
    return NULL;
}

// Thread-local workload
void* thread_local_worker(void* arg) {
    thread_local_data_t* data = (thread_local_data_t*)arg;
    
    for (int i = 0; i < ITERATIONS; i++) {
        data->local_counter++;
        
        // Periodic sync to global counter
        if (i % 100000 == 0) {
            atomic_fetch_add(data->global_counter, 100000);
            data->local_counter = 0;
        }
        
        // Same computation
        if (i % 10000 == 0) {
            volatile int work = 0;
            for (int j = 0; j < 100; j++) {
                work += j * j;
            }
        }
    }
    
    // Final sync
    atomic_fetch_add(data->global_counter, data->local_counter);
    
    return NULL;
}

double benchmark_workload(void* (*worker_func)(void*), void* thread_data[], 
                         const char* description) {
    pthread_t threads[NUM_THREADS];
    struct timespec start_time, end_time;
    
    printf("Running %s benchmark...\n", description);
    
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker_func, thread_data[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    return elapsed;
}

void comprehensive_false_sharing_analysis() {
    printf("Comprehensive False Sharing Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    printf("Test Configuration:\n");
    printf("  Threads: %d\n", NUM_THREADS);
    printf("  Iterations per thread: %d\n", ITERATIONS);
    printf("  Cache line size: %d bytes\n", CACHE_LINE_SIZE);
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    
    // Initialize data structures
    memset(&false_sharing_data, 0, sizeof(false_sharing_data));
    memset(&cache_friendly_data, 0, sizeof(cache_friendly_data));
    atomic_store(&global_atomic_counter, 0);
    
    // Prepare thread data
    int thread_ids[NUM_THREADS];
    thread_local_data_t tl_data[NUM_THREADS];
    
    void* false_sharing_args[NUM_THREADS];
    void* cache_friendly_args[NUM_THREADS];
    void* thread_local_args[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        false_sharing_args[i] = &thread_ids[i];
        cache_friendly_args[i] = &thread_ids[i];
        
        tl_data[i].local_counter = 0;
        tl_data[i].global_counter = &global_atomic_counter;
        tl_data[i].thread_id = i;
        thread_local_args[i] = &tl_data[i];
    }
    
    // Run benchmarks
    double false_sharing_time = benchmark_workload(false_sharing_worker, 
                                                  false_sharing_args, 
                                                  "False Sharing");
    
    double cache_friendly_time = benchmark_workload(cache_friendly_worker, 
                                                   cache_friendly_args, 
                                                   "Cache Friendly");
    
    double thread_local_time = benchmark_workload(thread_local_worker, 
                                                 thread_local_args, 
                                                 "Thread Local");
    
    // Calculate performance metrics
    unsigned long total_operations = (unsigned long)NUM_THREADS * ITERATIONS;
    
    double false_sharing_ops_per_sec = total_operations / false_sharing_time;
    double cache_friendly_ops_per_sec = total_operations / cache_friendly_time;
    double thread_local_ops_per_sec = total_operations / thread_local_time;
    
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    printf("Performance Results:\n");
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    printf("Approach          | Time (sec) | Ops/sec (M) | Relative Performance\n");
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    printf("False Sharing     | %10.3f | %11.1f | %8.2fx (baseline)\n",
           false_sharing_time, false_sharing_ops_per_sec / 1000000.0, 1.0);
    printf("Cache Friendly    | %10.3f | %11.1f | %8.2fx faster\n",
           cache_friendly_time, cache_friendly_ops_per_sec / 1000000.0, 
           false_sharing_time / cache_friendly_time);
    printf("Thread Local      | %10.3f | %11.1f | %8.2fx faster\n",
           thread_local_time, thread_local_ops_per_sec / 1000000.0, 
           false_sharing_time / thread_local_time);
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}

// Memory access pattern analysis
typedef enum {
    ACCESS_SEQUENTIAL,
    ACCESS_RANDOM,
    ACCESS_STRIDED,
    ACCESS_BLOCKED
} access_pattern_t;

void* memory_access_worker(void* arg) {
    typedef struct {
        access_pattern_t pattern;
        volatile char* memory;
        size_t size;
        int thread_id;
        int num_threads;
        unsigned long* operations_completed;
    } access_worker_data_t;
    
    access_worker_data_t* data = (access_worker_data_t*)arg;
    
    const unsigned long iterations = 10000000;
    unsigned long ops = 0;
    
    switch (data->pattern) {
        case ACCESS_SEQUENTIAL:
            // Sequential access pattern
            for (unsigned long i = 0; i < iterations; i++) {
                size_t offset = (i + data->thread_id * 1024) % data->size;
                data->memory[offset] = (char)(i & 0xFF);
                ops++;
            }
            break;
            
        case ACCESS_RANDOM:
            // Random access pattern
            for (unsigned long i = 0; i < iterations; i++) {
                size_t offset = ((i * 1103515245 + 12345) + data->thread_id * 7919) % data->size;
                data->memory[offset] = (char)(i & 0xFF);
                ops++;
            }
            break;
            
        case ACCESS_STRIDED:
            // Strided access pattern
            for (unsigned long i = 0; i < iterations; i++) {
                size_t offset = ((i * 64) + data->thread_id * 128) % data->size;
                data->memory[offset] = (char)(i & 0xFF);
                ops++;
            }
            break;
            
        case ACCESS_BLOCKED:
            // Blocked access pattern (good for cache)
            size_t block_size = 1024;
            size_t thread_offset = data->thread_id * block_size;
            for (unsigned long i = 0; i < iterations; i++) {
                size_t offset = (thread_offset + (i % block_size)) % data->size;
                data->memory[offset] = (char)(i & 0xFF);
                ops++;
            }
            break;
    }
    
    *data->operations_completed = ops;
    return NULL;
}

void analyze_memory_access_patterns() {
    printf("Memory Access Pattern Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    
    const size_t memory_size = 64 * 1024 * 1024; // 64MB
    volatile char* memory = malloc(memory_size);
    if (!memory) {
        printf("Failed to allocate memory for test\n");
        return;
    }
    
    // Initialize memory
    memset((void*)memory, 0, memory_size);
    
    access_pattern_t patterns[] = {
        ACCESS_SEQUENTIAL, ACCESS_RANDOM, ACCESS_STRIDED, ACCESS_BLOCKED
    };
    
    const char* pattern_names[] = {
        "Sequential", "Random", "Strided", "Blocked"
    };
    
    printf("Memory size: %zu MB\n", memory_size / (1024 * 1024));
    printf("Number of threads: %d\n", NUM_THREADS);
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    printf("Access Pattern | Time (sec) | Ops/sec (M) | Relative Performance\n");
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    
    double baseline_time = 0;
    
    for (int p = 0; p < 4; p++) {
        pthread_t threads[NUM_THREADS];
        unsigned long operations_completed[NUM_THREADS];
        
        typedef struct {
            access_pattern_t pattern;
            volatile char* memory;
            size_t size;
            int thread_id;
            int num_threads;
            unsigned long* operations_completed;
        } access_worker_data_t;
        
        access_worker_data_t worker_data[NUM_THREADS];
        
        for (int i = 0; i < NUM_THREADS; i++) {
            worker_data[i].pattern = patterns[p];
            worker_data[i].memory = memory;
            worker_data[i].size = memory_size;
            worker_data[i].thread_id = i;
            worker_data[i].num_threads = NUM_THREADS;
            worker_data[i].operations_completed = &operations_completed[i];
        }
        
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_create(&threads[i], NULL, memory_access_worker, &worker_data[i]);
        }
        
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        
        double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                         (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        
        unsigned long total_ops = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            total_ops += operations_completed[i];
        }
        
        double ops_per_sec = total_ops / elapsed;
        
        if (p == 0) baseline_time = elapsed;
        double relative_performance = baseline_time / elapsed;
        
        printf("%-14s | %10.3f | %11.1f | %8.2fx\n",
               pattern_names[p], elapsed, ops_per_sec / 1000000.0, relative_performance);
    }
    
    free((void*)memory);
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}

// Cache line contention visualization
void visualize_cache_line_contention() {
    printf("Cache Line Contention Visualization\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    
    // Create different data layouts
    typedef struct {
        char data[CACHE_LINE_SIZE];
    } cache_line_t;
    
    // Layout 1: Multiple threads accessing same cache line
    static volatile cache_line_t shared_cache_line;
    
    // Layout 2: Each thread gets its own cache line
    static volatile cache_line_t per_thread_cache_lines[NUM_THREADS];
    
    printf("Cache Line Size: %d bytes\n", CACHE_LINE_SIZE);
    printf("Testing contention with %d threads\n", NUM_THREADS);
    printf("─────────────────────────────────────────────────────────────────────────────────\n");
    
    // Test 1: High contention (all threads access same cache line)
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    
    void* high_contention_worker(void* arg) {
        int id = *(int*)arg;
        for (int i = 0; i < 1000000; i++) {
            shared_cache_line.data[id % CACHE_LINE_SIZE] = (char)i;
        }
        return NULL;
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, high_contention_worker, &thread_ids[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double high_contention_time = (end.tv_sec - start.tv_sec) + 
                                  (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Test 2: Low contention (each thread gets its own cache line)
    void* low_contention_worker(void* arg) {
        int id = *(int*)arg;
        for (int i = 0; i < 1000000; i++) {
            per_thread_cache_lines[id].data[i % CACHE_LINE_SIZE] = (char)i;
        }
        return NULL;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, low_contention_worker, &thread_ids[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double low_contention_time = (end.tv_sec - start.tv_sec) + 
                                 (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("High contention (shared cache line): %.3f seconds\n", high_contention_time);
    printf("Low contention (separate cache lines): %.3f seconds\n", low_contention_time);
    printf("Performance improvement: %.2fx\n", high_contention_time / low_contention_time);
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("%s: %.6f seconds\n", description, elapsed);
    return elapsed;
}

void demonstrate_false_sharing() {
    printf("Demonstrating false sharing impact:\n");
    
    double false_sharing_time = measure_performance(false_sharing_thread, 
                                                   "With false sharing");
    
    double cache_friendly_time = measure_performance(cache_friendly_thread, 
                                                    "Cache friendly");
    
    printf("Performance improvement: %.2fx\n", 
           false_sharing_time / cache_friendly_time);
}
```

### Memory Access Patterns

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define ARRAY_SIZE (1024 * 1024)
#define NUM_THREADS 4

int global_array[ARRAY_SIZE];

typedef struct {
    int thread_id;
    int start_index;
    int end_index;
    long sum;
} ThreadData;

// Sequential access pattern
void* sequential_access(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    data->sum = 0;
    
    for (int i = data->start_index; i < data->end_index; i++) {
        data->sum += global_array[i];
    }
    
    return NULL;
}

// Strided access pattern (poor cache locality)
void* strided_access(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    data->sum = 0;
    
    for (int i = data->thread_id; i < ARRAY_SIZE; i += NUM_THREADS) {
        data->sum += global_array[i];
    }
    
    return NULL;
}

void compare_access_patterns() {
    // Initialize array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        global_array[i] = i % 100;
    }
    
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    struct timespec start, end;
    
    // Test sequential access
    printf("Testing sequential access pattern:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int chunk_size = ARRAY_SIZE / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start_index = i * chunk_size;
        thread_data[i].end_index = (i + 1) * chunk_size;
        pthread_create(&threads[i], NULL, sequential_access, &thread_data[i]);
    }
    
    long total_sum = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total_sum += thread_data[i].sum;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double sequential_time = (end.tv_sec - start.tv_sec) + 
                            (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Sequential access: %.6f seconds, sum = %ld\n", 
           sequential_time, total_sum);
    
    // Test strided access
    printf("\nTesting strided access pattern:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    total_sum = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        pthread_create(&threads[i], NULL, strided_access, &thread_data[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total_sum += thread_data[i].sum;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double strided_time = (end.tv_sec - start.tv_sec) + 
                         (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Strided access: %.6f seconds, sum = %ld\n", 
           strided_time, total_sum);
    
    printf("\nPerformance ratio: %.2fx\n", strided_time / sequential_time);
}
```

## Advanced Scalability Analysis and Performance Modeling

Scalability is the holy grail of parallel computing, yet achieving linear speedup is rare in practice. Understanding the fundamental limits and bottlenecks is crucial for realistic performance expectations and system design.

### Comprehensive Amdahl's Law Analysis

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <stdatomic.h>
#include <string.h>

#define WORK_SIZE 10000000

// Performance analysis framework
typedef struct {
    double serial_fraction;
    double parallel_fraction;
    double synchronization_overhead;
    const char* workload_name;
} workload_profile_t;

typedef struct {
    int thread_id;
    int num_threads;
    volatile double* data;
    atomic_double* result;
    pthread_barrier_t* barrier;
    
    // Performance tracking
    struct timespec start_time;
    struct timespec end_time;
    unsigned long operations_completed;
    
    // Workload configuration
    workload_profile_t* profile;
} thread_context_t;

// Different workload patterns with varying serial fractions
void* cpu_intensive_workload(void* arg) {
    thread_context_t* ctx = (thread_context_t*)arg;
    
    clock_gettime(CLOCK_MONOTONIC, &ctx->start_time);
    
    // Minimal serial section (1% serial, 99% parallel)
    if (ctx->thread_id == 0) {
        // Serial initialization
        for (int i = 0; i < WORK_SIZE / 100; i++) {
            ctx->data[i] = sqrt(i + 1.0);
        }
    }
    
    // Synchronization point
    pthread_barrier_wait(ctx->barrier);
    
    // Parallel computation
    int chunk_size = (WORK_SIZE * 99 / 100) / ctx->num_threads;
    int start_idx = (WORK_SIZE / 100) + ctx->thread_id * chunk_size;
    int end_idx = start_idx + chunk_size;
    
    volatile double local_sum = 0.0;
    for (int i = start_idx; i < end_idx; i++) {
        local_sum += sin(i) * cos(i) + sqrt(i);
    }
    
    // Atomic reduction (synchronization overhead)
    double current = atomic_load(ctx->result);
    while (!atomic_compare_exchange_weak(ctx->result, &current, current + local_sum)) {
        // Retry on failure
    }
    
    ctx->operations_completed = chunk_size;
    clock_gettime(CLOCK_MONOTONIC, &ctx->end_time);
    
    return NULL;
}

void* memory_intensive_workload(void* arg) {
    thread_context_t* ctx = (thread_context_t*)arg;
    
    clock_gettime(CLOCK_MONOTONIC, &ctx->start_time);
    
    // Higher serial fraction due to memory allocation/initialization (20% serial)
    if (ctx->thread_id == 0) {
        for (int i = 0; i < WORK_SIZE / 5; i++) {
            ctx->data[i] = (double)(i * 3.14159);
        }
    }
    
    pthread_barrier_wait(ctx->barrier);
    
    // Memory-bound parallel work (80% parallel)
    int chunk_size = (WORK_SIZE * 4 / 5) / ctx->num_threads;
    int start_idx = (WORK_SIZE / 5) + ctx->thread_id * chunk_size;
    int end_idx = start_idx + chunk_size;
    
    volatile double local_sum = 0.0;
    for (int i = start_idx; i < end_idx; i++) {
        // Memory-intensive operations
        for (int j = 0; j < 10; j++) {
            local_sum += ctx->data[i % (WORK_SIZE / 5)] * j;
        }
    }
    
    // Heavy synchronization (contention)
    static pthread_mutex_t reduction_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&reduction_mutex);
    double current = atomic_load(ctx->result);
    atomic_store(ctx->result, current + local_sum);
    pthread_mutex_unlock(&reduction_mutex);
    
    ctx->operations_completed = chunk_size;
    clock_gettime(CLOCK_MONOTONIC, &ctx->end_time);
    
    return NULL;
}

void* sync_intensive_workload(void* arg) {
    thread_context_t* ctx = (thread_context_t*)arg;
    
    clock_gettime(CLOCK_MONOTONIC, &ctx->start_time);
    
    // Very high serial fraction due to frequent synchronization (50% serial)
    static pthread_mutex_t work_mutex = PTHREAD_MUTEX_INITIALIZER;
    static atomic_int work_counter = 0;
    
    volatile double local_sum = 0.0;
    
    while (1) {
        // Frequent synchronization to get work
        pthread_mutex_lock(&work_mutex);
        int work_idx = atomic_fetch_add(&work_counter, 1);
        pthread_mutex_unlock(&work_mutex);
        
        if (work_idx >= WORK_SIZE) break;
        
        // Small amount of work between synchronizations
        for (int i = 0; i < 1000; i++) {
            local_sum += sin(work_idx + i) * cos(work_idx + i);
        }
        
        // Another synchronization point
        pthread_mutex_lock(&work_mutex);
        double current = atomic_load(ctx->result);
        atomic_store(ctx->result, current + local_sum);
        local_sum = 0.0;
        pthread_mutex_unlock(&work_mutex);
    }
    
    ctx->operations_completed = atomic_load(&work_counter);
    clock_gettime(CLOCK_MONOTONIC, &ctx->end_time);
    
    return NULL;
}

// Comprehensive scalability measurement
double measure_workload_scalability(void* (*workload_func)(void*), 
                                   workload_profile_t* profile,
                                   int num_threads) {
    pthread_t threads[num_threads];
    thread_context_t contexts[num_threads];
    pthread_barrier_t barrier;
    
    // Allocate shared data
    volatile double* shared_data = malloc(WORK_SIZE * sizeof(double));
    atomic_double shared_result = 0.0;
    
    // Initialize barrier
    pthread_barrier_init(&barrier, NULL, num_threads);
    
    struct timespec overall_start, overall_end;
    clock_gettime(CLOCK_MONOTONIC, &overall_start);
    
    // Create and configure threads
    for (int i = 0; i < num_threads; i++) {
        contexts[i].thread_id = i;
        contexts[i].num_threads = num_threads;
        contexts[i].data = shared_data;
        contexts[i].result = &shared_result;
        contexts[i].barrier = &barrier;
        contexts[i].profile = profile;
        contexts[i].operations_completed = 0;
        
        pthread_create(&threads[i], NULL, workload_func, &contexts[i]);
    }
    
    // Wait for completion
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &overall_end);
    
    double total_time = (overall_end.tv_sec - overall_start.tv_sec) + 
                        (overall_end.tv_nsec - overall_start.tv_nsec) / 1e9;
    
    // Calculate per-thread statistics
    unsigned long total_ops = 0;
    double max_thread_time = 0.0;
    double min_thread_time = 1000.0;
    
    for (int i = 0; i < num_threads; i++) {
        double thread_time = (contexts[i].end_time.tv_sec - contexts[i].start_time.tv_sec) + 
                            (contexts[i].end_time.tv_nsec - contexts[i].start_time.tv_nsec) / 1e9;
        
        total_ops += contexts[i].operations_completed;
        if (thread_time > max_thread_time) max_thread_time = thread_time;
        if (thread_time < min_thread_time) min_thread_time = thread_time;
    }
    
    printf("  Total time: %.6f s, Max thread: %.6f s, Load balance: %.3f\n",
           total_time, max_thread_time, min_thread_time / max_thread_time);
    
    // Cleanup
    pthread_barrier_destroy(&barrier);
    free((void*)shared_data);
    
    return total_time;
}

void comprehensive_scalability_analysis() {
    printf("Comprehensive Scalability Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    
    workload_profile_t workloads[] = {
        {0.01, 0.99, 0.001, "CPU-Intensive (1% serial)"},
        {0.20, 0.80, 0.050, "Memory-Intensive (20% serial)"},
        {0.50, 0.50, 0.200, "Sync-Intensive (50% serial)"}
    };
    
    void* (*workload_functions[])(void*) = {
        cpu_intensive_workload,
        memory_intensive_workload,
        sync_intensive_workload
    };
    
    for (int w = 0; w < 3; w++) {
        printf("\n%s Workload:\n", workloads[w].workload_name);
        printf("─────────────────────────────────────────────────────────────────────────────────\n");
        printf("Threads | Time (s) | Speedup | Efficiency | Theoretical | Scalability\n");
        printf("─────────────────────────────────────────────────────────────────────────────────\n");
        
        double baseline_time = 0;
        
        for (int threads = 1; threads <= 16; threads *= 2) {
            printf("%7d | ", threads);
            
            double time = measure_workload_scalability(workload_functions[w], 
                                                      &workloads[w], threads);
            
            if (threads == 1) {
                baseline_time = time;
                printf("%8.3f | %7.2fx | %10.1f%% | %11.2fx | %11s\n",
                       time, 1.0, 100.0, 1.0, "baseline");
            } else {
                double speedup = baseline_time / time;
                double efficiency = speedup / threads * 100.0;
                
                // Amdahl's law prediction
                double serial_frac = workloads[w].serial_fraction;
                double theoretical = 1.0 / (serial_frac + (1.0 - serial_frac) / threads);
                
                // Gustafson's law (scaled problem size)
                double gustafson = threads - serial_frac * (threads - 1);
                
                // Scalability metric (0-1, where 1 is perfect scaling)
                double scalability = speedup / threads;
                
                printf("%8.3f | %7.2fx | %10.1f%% | %11.2fx | %11.3f\n",
                       time, speedup, efficiency, theoretical, scalability);
            }
        }
    }
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
}

// Universal Scalability Law analysis
typedef struct {
    double alpha;  // Contention coefficient
    double beta;   // Coherency coefficient
    double gamma;  // Crosstalk coefficient
} usl_coefficients_t;

double usl_predict_throughput(usl_coefficients_t* coeffs, int n) {
    // USL: Throughput(N) = N / (1 + α(N-1) + βN(N-1))
    double denominator = 1.0 + coeffs->alpha * (n - 1) + coeffs->beta * n * (n - 1);
    return n / denominator;
}

void analyze_universal_scalability_law() {
    printf("\nUniversal Scalability Law Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════\n");
    
    // Example coefficients for different system types
    usl_coefficients_t systems[] = {
        {0.05, 0.01, 0.0},  // Low contention system
        {0.15, 0.05, 0.0},  // Medium contention system
        {0.30, 0.15, 0.0}   // High contention system
    };
    
    const char* system_names[] = {
        "Low Contention", "Medium Contention", "High Contention"
    };
    
    printf("CPU Cores |");
    for (int s = 0; s < 3; s++) {
        printf(" %16s |", system_names[s]);
    }
    printf("\n");
    
    printf("──────────|");
    for (int s = 0; s < 3; s++) {
        printf("─────────────────|");
    }
    printf("\n");
    
    for (int cores = 1; cores <= 64; cores *= 2) {
        printf("%9d |", cores);
        
        for (int s = 0; s < 3; s++) {
            double throughput = usl_predict_throughput(&systems[s], cores);
            printf(" %13.2fx |", throughput);
        }
        printf("\n");
    }
    
    printf("═══════════════════════════════════════════════════════════════════════════════════\n\n");
}
```

## Lock Contention Analysis

### Measuring Lock Contention

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>

#define NUM_OPERATIONS 1000000

// Shared data protected by mutex
int shared_counter = 0;
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

// Atomic counter for comparison
_Atomic int atomic_counter = 0;

// Lock contention measurement
_Atomic long lock_wait_time = 0;
_Atomic int lock_contentions = 0;

typedef struct {
    int thread_id;
    int num_threads;
    int operations_per_thread;
} ThreadInfo;

void* mutex_contention_test(void* arg) {
    ThreadInfo* info = (ThreadInfo*)arg;
    struct timespec start, end;
    
    for (int i = 0; i < info->operations_per_thread; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        int result = pthread_mutex_trylock(&counter_mutex);
        if (result == EBUSY) {
            atomic_fetch_add(&lock_contentions, 1);
            pthread_mutex_lock(&counter_mutex);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        long wait_ns = (end.tv_sec - start.tv_sec) * 1000000000L + 
                       (end.tv_nsec - start.tv_nsec);
        atomic_fetch_add(&lock_wait_time, wait_ns);
        
        shared_counter++;
        pthread_mutex_unlock(&counter_mutex);
    }
    
    return NULL;
}

void* atomic_test(void* arg) {
    ThreadInfo* info = (ThreadInfo*)arg;
    
    for (int i = 0; i < info->operations_per_thread; i++) {
        atomic_fetch_add(&atomic_counter, 1);
    }
    
    return NULL;
}

void measure_lock_contention(int num_threads) {
    printf("\nTesting with %d threads:\n", num_threads);
    
    // Reset counters
    shared_counter = 0;
    atomic_counter = 0;
    lock_wait_time = 0;
    lock_contentions = 0;
    
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadInfo* thread_info = malloc(num_threads * sizeof(ThreadInfo));
    
    int ops_per_thread = NUM_OPERATIONS / num_threads;
    
    // Test mutex performance
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < num_threads; i++) {
        thread_info[i].thread_id = i;
        thread_info[i].num_threads = num_threads;
        thread_info[i].operations_per_thread = ops_per_thread;
        pthread_create(&threads[i], NULL, mutex_contention_test, &thread_info[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double mutex_time = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Test atomic performance
    atomic_counter = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, atomic_test, &thread_info[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double atomic_time = (end.tv_sec - start.tv_sec) + 
                         (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Results
    printf("Mutex approach:\n");
    printf("  Time: %.6f seconds\n", mutex_time);
    printf("  Counter: %d\n", shared_counter);
    printf("  Lock contentions: %d (%.2f%%)\n", 
           lock_contentions, (double)lock_contentions / NUM_OPERATIONS * 100);
    printf("  Average wait time: %.0f ns\n", 
           (double)lock_wait_time / NUM_OPERATIONS);
    
    printf("Atomic approach:\n");
    printf("  Time: %.6f seconds\n", atomic_time);
    printf("  Counter: %d\n", atomic_counter);
    printf("  Speedup: %.2fx\n", mutex_time / atomic_time);
    
    free(threads);
    free(thread_info);
}

void analyze_lock_contention() {
    printf("Lock Contention Analysis:\n");
    
    for (int threads = 1; threads <= 8; threads *= 2) {
        measure_lock_contention(threads);
    }
}
```

## Performance Profiling Integration

### Custom Performance Counters

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef struct {
    char name[64];
    long count;
    double total_time;
    double min_time;
    double max_time;
    pthread_mutex_t mutex;
} PerfCounter;

#define MAX_COUNTERS 10
static PerfCounter perf_counters[MAX_COUNTERS];
static int num_counters = 0;
static pthread_mutex_t counters_mutex = PTHREAD_MUTEX_INITIALIZER;

int create_perf_counter(const char* name) {
    pthread_mutex_lock(&counters_mutex);
    
    if (num_counters >= MAX_COUNTERS) {
        pthread_mutex_unlock(&counters_mutex);
        return -1;
    }
    
    int id = num_counters++;
    strncpy(perf_counters[id].name, name, sizeof(perf_counters[id].name) - 1);
    perf_counters[id].count = 0;
    perf_counters[id].total_time = 0.0;
    perf_counters[id].min_time = INFINITY;
    perf_counters[id].max_time = 0.0;
    pthread_mutex_init(&perf_counters[id].mutex, NULL);
    
    pthread_mutex_unlock(&counters_mutex);
    return id;
}

void record_perf_event(int counter_id, double duration) {
    if (counter_id < 0 || counter_id >= num_counters) return;
    
    PerfCounter* counter = &perf_counters[counter_id];
    pthread_mutex_lock(&counter->mutex);
    
    counter->count++;
    counter->total_time += duration;
    if (duration < counter->min_time) counter->min_time = duration;
    if (duration > counter->max_time) counter->max_time = duration;
    
    pthread_mutex_unlock(&counter->mutex);
}

void print_perf_counters() {
    printf("\nPerformance Counters:\n");
    printf("%-20s %10s %12s %12s %12s %12s\n", 
           "Name", "Count", "Total(s)", "Avg(ms)", "Min(ms)", "Max(ms)");
    printf("%.80s\n", "----------------------------------------"
                      "----------------------------------------");
    
    for (int i = 0; i < num_counters; i++) {
        PerfCounter* counter = &perf_counters[i];
        pthread_mutex_lock(&counter->mutex);
        
        double avg_ms = counter->count > 0 ? 
                       (counter->total_time / counter->count) * 1000 : 0;
        
        printf("%-20s %10ld %12.6f %12.3f %12.3f %12.3f\n",
               counter->name,
               counter->count,
               counter->total_time,
               avg_ms,
               counter->min_time * 1000,
               counter->max_time * 1000);
        
        pthread_mutex_unlock(&counter->mutex);
    }
}

// Example usage
int db_query_counter;
int computation_counter;

void* database_simulation(void* arg) {
    for (int i = 0; i < 100; i++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Simulate database query
        usleep(rand() % 10000); // 0-10ms
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double duration = (end.tv_sec - start.tv_sec) + 
                         (end.tv_nsec - start.tv_nsec) / 1e9;
        
        record_perf_event(db_query_counter, duration);
    }
    
    return NULL;
}

void demonstrate_perf_profiling() {
    db_query_counter = create_perf_counter("DB_Query");
    computation_counter = create_perf_counter("Computation");
    
    pthread_t threads[4];
    
    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, database_simulation, NULL);
    }
    
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    print_perf_counters();
}
```

## Optimization Techniques

### Thread-Local Optimization

```c
__thread struct {
    int cache[1024];
    int cache_size;
    long cache_hits;
    long cache_misses;
} thread_cache = {.cache_size = 0, .cache_hits = 0, .cache_misses = 0};

int optimized_lookup(int key) {
    // Check thread-local cache first
    for (int i = 0; i < thread_cache.cache_size; i++) {
        if (thread_cache.cache[i] == key) {
            thread_cache.cache_hits++;
            return key * key; // Simulate computation result
        }
    }
    
    thread_cache.cache_misses++;
    
    // Add to cache if space available
    if (thread_cache.cache_size < 1024) {
        thread_cache.cache[thread_cache.cache_size++] = key;
    }
    
    return key * key; // Simulate expensive computation
}

void print_cache_stats() {
    printf("Thread cache stats - Hits: %ld, Misses: %ld, Hit rate: %.2f%%\n",
           thread_cache.cache_hits, thread_cache.cache_misses,
           (double)thread_cache.cache_hits / 
           (thread_cache.cache_hits + thread_cache.cache_misses) * 100);
}
```

## Comprehensive Performance Engineering Exercises

These exercises are designed to develop real-world performance analysis and optimization skills. Each exercise builds upon previous concepts while introducing new challenges.

### Exercise 1: Multi-Threaded Performance Benchmark Suite
**Difficulty: Intermediate | Duration: 2-3 days**

Create a comprehensive benchmarking framework for evaluating different threading strategies:

#### **Core Requirements:**
```c
// Performance benchmark framework API
typedef struct {
    const char* benchmark_name;
    int min_threads;
    int max_threads;
    unsigned long operations_per_thread;
    
    // Benchmark function
    void* (*worker_function)(void* args);
    
    // Setup/teardown
    void* (*setup_function)(int num_threads);
    void (*cleanup_function)(void* setup_data);
    
    // Results
    double* execution_times;
    double* throughput_results;
    double* latency_results;
    
    // Configuration
    bool measure_memory_usage;
    bool measure_cache_misses;
    bool measure_context_switches;
} performance_benchmark_t;

// Your implementation should include:
performance_benchmark_t* benchmark_create(const char* name);
void benchmark_configure(performance_benchmark_t* bench, /* parameters */);
void benchmark_run(performance_benchmark_t* bench);
void benchmark_report_results(performance_benchmark_t* bench, const char* output_file);
void benchmark_compare(performance_benchmark_t** benchmarks, int count);
```

#### **Benchmark Categories to Implement:**

1. **Synchronization Primitive Comparison**
   ```c
   // Compare: mutex, spinlock, atomic operations, read-write locks
   void* mutex_benchmark_worker(void* args);
   void* spinlock_benchmark_worker(void* args);
   void* atomic_benchmark_worker(void* args);
   void* rwlock_benchmark_worker(void* args);
   ```

2. **Memory Access Pattern Analysis**
   ```c
   // Compare: sequential, random, strided access patterns
   void* sequential_access_worker(void* args);
   void* random_access_worker(void* args);
   void* strided_access_worker(void* args);
   ```

3. **Data Structure Performance**
   ```c
   // Compare: array, linked list, tree, hash table operations
   void* array_operations_worker(void* args);
   void* list_operations_worker(void* args);
   void* tree_operations_worker(void* args);
   void* hashtable_operations_worker(void* args);
   ```

#### **Advanced Features:**
- Automatic thread scaling analysis (1 to MAX_CORES)
- Memory usage tracking per thread count
- Cache miss rate estimation
- CPU utilization monitoring
- Load balancing effectiveness measurement

#### **Deliverables:**
- Complete benchmark framework implementation
- Performance report generator (HTML/CSV output)
- Comparative analysis tool
- Documentation with usage examples

### Exercise 2: Cache-Aware Algorithm Optimization
**Difficulty: Advanced | Duration: 3-4 days**

Optimize matrix multiplication for different cache architectures:

#### **Base Implementation:**
```c
// Naive matrix multiplication (baseline)
void matrix_multiply_naive(double* A, double* B, double* C, int N);

// Your optimized implementations:
void matrix_multiply_blocked(double* A, double* B, double* C, int N, int block_size);
void matrix_multiply_cache_aware(double* A, double* B, double* C, int N);
void matrix_multiply_numa_aware(double* A, double* B, double* C, int N, int num_threads);
void matrix_multiply_vectorized(double* A, double* B, double* C, int N);
```

#### **Optimization Techniques to Implement:**
1. **Cache Blocking/Tiling:** Divide matrices into cache-friendly blocks
2. **Loop Unrolling:** Reduce loop overhead
3. **Memory Prefetching:** Hint processor about future memory accesses
4. **NUMA Optimization:** Bind threads and memory to NUMA nodes
5. **Vectorization:** Use SIMD instructions (SSE/AVX)

#### **Performance Analysis:**
```c
typedef struct {
    double gflops;          // Billion floating-point operations per second
    double cache_miss_rate; // L1, L2, L3 cache miss rates
    double memory_bandwidth; // GB/s
    double cpu_utilization; // Percentage
    double energy_consumption; // Estimated watts
} matrix_performance_metrics_t;

// Measure and compare all implementations
void analyze_matrix_performance(int* matrix_sizes, int num_sizes);
```

#### **Evaluation Criteria:**
- **Performance:** Achieve >80% of theoretical peak FLOPS
- **Scalability:** Linear speedup up to memory bandwidth limit
- **Efficiency:** >90% CPU utilization across all cores
- **Memory:** Minimize cache misses and memory traffic

### Exercise 3: Real-Time System Performance Profiler
**Difficulty: Expert | Duration: 4-5 days**

Build a production-grade performance profiler for multi-threaded applications:

#### **Core Profiler Architecture:**
```c
typedef struct thread_profiler {
    // Thread identification
    pthread_t thread_id;
    pid_t system_tid;
    char thread_name[64];
    
    // Performance counters
    atomic_ulong cpu_cycles;
    atomic_ulong instructions_executed;
    atomic_ulong cache_misses;
    atomic_ulong context_switches;
    atomic_ulong page_faults;
    
    // Timing information
    struct timespec creation_time;
    struct timespec last_sample_time;
    atomic_ulong total_cpu_time_ns;
    atomic_ulong total_wall_time_ns;
    
    // Call stack sampling
    void** call_stack_samples;
    int max_stack_depth;
    atomic_int sample_count;
    
    // Memory usage
    atomic_ulong heap_allocations;
    atomic_ulong heap_deallocations;
    atomic_ulong peak_memory_usage;
    atomic_ulong current_memory_usage;
    
    // Lock contention tracking
    atomic_ulong lock_acquisitions;
    atomic_ulong lock_contentions;
    atomic_ulong total_lock_wait_time_ns;
    
} thread_profiler_t;

typedef struct system_profiler {
    thread_profiler_t** thread_profilers;
    int max_threads;
    atomic_int active_threads;
    
    // Global system metrics
    atomic_ulong total_context_switches;
    atomic_ulong total_interrupts;
    atomic_double cpu_utilization;
    atomic_ulong memory_usage_bytes;
    atomic_double memory_bandwidth_gbps;
    
    // Sampling configuration
    int sampling_frequency_hz;
    bool profile_call_stacks;
    bool profile_memory_allocations;
    bool profile_lock_contention;
    
    // Output configuration
    char* output_directory;
    bool generate_flame_graphs;
    bool generate_timeline_view;
    
} system_profiler_t;

// Core API to implement:
system_profiler_t* profiler_create(const char* output_dir);
void profiler_start(system_profiler_t* profiler);
void profiler_stop(system_profiler_t* profiler);
void profiler_attach_thread(system_profiler_t* profiler, const char* thread_name);
void profiler_detach_thread(system_profiler_t* profiler);
void profiler_sample_all_threads(system_profiler_t* profiler);
void profiler_generate_report(system_profiler_t* profiler);
```

#### **Advanced Features:**
1. **Low-Overhead Sampling:** <1% performance impact
2. **Call Stack Unwinding:** Capture complete call stacks
3. **Symbol Resolution:** Map addresses to function names
4. **Timeline Visualization:** Generate interactive timeline
5. **Hotspot Detection:** Identify performance bottlenecks
6. **Memory Leak Detection:** Track allocation/deallocation patterns
7. **Lock Contention Analysis:** Identify synchronization bottlenecks

#### **Integration Requirements:**
```c
// Simple macros for application integration
#define PROFILER_START_THREAD(name) profiler_attach_thread(global_profiler, name)
#define PROFILER_END_THREAD() profiler_detach_thread(global_profiler)
#define PROFILER_MARK_FUNCTION() /* Automatic function entry/exit tracking */
#define PROFILER_CUSTOM_COUNTER(name, value) /* Custom performance counters */
```

### Exercise 4: Distributed Performance Monitoring System
**Difficulty: Expert | Duration: 5-6 days**

Create a system to monitor performance across multiple machines:

#### **Distributed Architecture:**
```c
// Network protocol for performance data
typedef struct {
    uint32_t magic_number;
    uint32_t version;
    uint64_t timestamp_ns;
    uint32_t data_length;
    uint32_t checksum;
} perf_packet_header_t;

typedef struct {
    char hostname[64];
    int cpu_cores;
    double cpu_utilization;
    uint64_t memory_total_bytes;
    uint64_t memory_used_bytes;
    uint32_t active_threads;
    
    // Per-thread performance data
    thread_perf_data_t* thread_data;
    uint32_t thread_count;
    
} node_performance_data_t;

// Distributed monitoring API:
typedef struct distributed_monitor distributed_monitor_t;

distributed_monitor_t* monitor_create_collector(int port);
distributed_monitor_t* monitor_create_agent(const char* collector_host, int port);
void monitor_start(distributed_monitor_t* monitor);
void monitor_register_node(distributed_monitor_t* collector, const char* hostname);
void monitor_send_performance_data(distributed_monitor_t* agent, 
                                  node_performance_data_t* data);
void monitor_aggregate_cluster_metrics(distributed_monitor_t* collector);
void monitor_generate_cluster_report(distributed_monitor_t* collector, 
                                    const char* output_file);
```

#### **Features to Implement:**
1. **Real-time Data Collection:** Gather metrics from multiple nodes
2. **Data Aggregation:** Combine and analyze cluster-wide performance
3. **Anomaly Detection:** Identify performance outliers
4. **Load Balancing Recommendations:** Suggest optimal thread/process distribution
5. **Predictive Analysis:** Forecast performance bottlenecks
6. **Visualization Dashboard:** Web-based real-time monitoring interface

### Exercise 5: Performance Regression Testing Framework
**Difficulty: Advanced | Duration: 3-4 days**

Build an automated system to detect performance regressions:

#### **Core Framework:**
```c
typedef struct {
    const char* test_name;
    double (*performance_test)(void);
    
    // Performance criteria
    double expected_performance;
    double tolerance_percentage;
    
    // Historical data
    double* historical_results;
    int num_historical_results;
    
    // Regression detection
    bool (*regression_detector)(double current, double* historical, int count);
    
} performance_regression_test_t;

// Framework API:
typedef struct regression_framework regression_framework_t;

regression_framework_t* regression_framework_create(const char* results_db_path);
void regression_framework_add_test(regression_framework_t* framework, 
                                  performance_regression_test_t* test);
bool regression_framework_run_all_tests(regression_framework_t* framework);
void regression_framework_generate_report(regression_framework_t* framework, 
                                         const char* report_path);
```

#### **Advanced Regression Detection:**
```c
// Statistical regression detection methods
bool simple_threshold_detector(double current, double* historical, int count);
bool statistical_outlier_detector(double current, double* historical, int count);
bool trend_analysis_detector(double current, double* historical, int count);
bool machine_learning_detector(double current, double* historical, int count);
```

### Assessment Rubric

For each exercise, evaluate yourself using this framework:

#### **Technical Implementation (40%)**
- **Expert (36-40)**: Code is production-ready, handles all edge cases, excellent error handling
- **Advanced (28-35)**: Solid implementation with good error handling and performance
- **Intermediate (20-27)**: Basic requirements met, some issues with robustness
- **Novice (0-19)**: Incomplete implementation with significant issues

#### **Performance Optimization (30%)**
- **Expert (27-30)**: Achieves near-optimal performance, innovative optimizations
- **Advanced (21-26)**: Good performance with several optimization techniques applied
- **Intermediate (15-20)**: Some optimizations applied, reasonable performance
- **Novice (0-14)**: Little to no optimization, poor performance

#### **Analysis and Documentation (20%)**
- **Expert (18-20)**: Thorough analysis, excellent documentation, deep insights
- **Advanced (14-17)**: Good analysis with clear documentation
- **Intermediate (10-13)**: Basic analysis, adequate documentation
- **Novice (0-9)**: Poor analysis, minimal documentation

#### **Innovation and Depth (10%)**
- **Expert (9-10)**: Creative solutions, goes beyond requirements
- **Advanced (7-8)**: Some innovative approaches
- **Intermediate (5-6)**: Meets requirements without innovation
- **Novice (0-4)**: Minimal effort, no creative thinking

### Performance Engineering Challenges

#### **Challenge 1: Zero-Latency Message Passing**
Design a message passing system with <100ns latency between threads on the same NUMA node.

#### **Challenge 2: 99.99% Uptime Performance Monitor**
Create a performance monitoring system that can run continuously for months without impacting application performance.

#### **Challenge 3: Adaptive Threading System**
Build a system that automatically adjusts thread pool sizes based on real-time performance metrics.

#### **Challenge 4: Hardware-Specific Optimization**
Optimize the same algorithm for Intel x86, ARM, and RISC-V architectures, documenting the performance differences.

#### **Challenge 5: Energy-Efficient Computing**
Design threading strategies that minimize energy consumption while maintaining performance targets.

## Comprehensive Performance Engineering Assessment

This assessment evaluates your mastery of multi-threaded performance analysis, optimization techniques, and real-world problem-solving skills. Complete all sections to demonstrate expert-level proficiency.

### Part I: Theoretical Performance Analysis (25 points)

#### **Question 1: Scalability Bottleneck Analysis (8 points)**

You are tasked with analyzing a distributed web server that shows poor scalability beyond 16 cores. Given the following performance data:

```
Cores:    1     2     4     8    16    32    64
RPS:   1000  1800  3200  5600  8000  8200  8100
CPU%:    95    90    85    75    50    25    12
```

**Analyze and provide:**
a) Calculate the scalability efficiency for each core count
b) Identify the primary bottleneck(s) limiting scalability
c) Apply Amdahl's Law to estimate the serial fraction
d) Use Universal Scalability Law to model the contention and coherency delays
e) Recommend specific optimization strategies with expected performance impact

**Expected Answer:** Detailed mathematical analysis with graphs, identification of specific bottlenecks (likely I/O or synchronization), and concrete optimization recommendations.

#### **Question 2: Cache Coherency Performance Impact (8 points)**

Given a NUMA system with 4 nodes, each containing 16 cores and 64GB RAM, analyze the following false sharing scenario:

```c
// Thread data structure
typedef struct {
    long counter;           // 8 bytes
    char status;           // 1 byte  
    long timestamp;        // 8 bytes
    double performance;    // 8 bytes
    // Total: 25 bytes per thread
} thread_data_t;

thread_data_t worker_data[64];  // 64 threads, tightly packed
```

**Calculate and explain:**
a) How many threads share each cache line (assume 64-byte cache lines)?
b) Estimate the performance impact of false sharing vs. cache-aligned layout
c) Design an optimal memory layout for this NUMA system
d) Calculate expected memory bandwidth requirements for different access patterns
e) Propose a dynamic load balancing strategy considering NUMA topology

#### **Question 3: Real-Time Performance Constraints (9 points)**

Design a performance monitoring system for a high-frequency trading application with these requirements:
- Maximum 50μs latency for order processing
- 1 million orders per second sustained throughput  
- 99.99% reliability
- Real-time risk management
- Regulatory compliance logging

**Address these aspects:**
a) **Threading Architecture:** Justify your choice of threading model
b) **Memory Management:** Strategy for avoiding GC pauses and memory fragmentation
c) **Synchronization:** Minimize lock contention while ensuring data consistency
d) **Hardware Optimization:** CPU affinity, NUMA awareness, interrupt handling
e) **Monitoring Strategy:** How to measure performance without impacting latency
f) **Failure Recovery:** Handle component failures within latency constraints

### Part II: Practical Performance Optimization (35 points)

#### **Challenge 1: Lock-Free Data Structure Performance (15 points)**

Implement and analyze a lock-free skip list with the following specifications:

```c
typedef struct skiplist_node {
    int key;
    void* value;
    atomic_ptr_t forward[MAX_LEVEL];
    int level;
} skiplist_node_t;

typedef struct {
    skiplist_node_t* header;
    atomic_int max_level;
    atomic_ulong insert_count;
    atomic_ulong delete_count;
    atomic_ulong search_count;
} lockfree_skiplist_t;

// API to implement:
lockfree_skiplist_t* skiplist_create();
bool skiplist_insert(lockfree_skiplist_t* list, int key, void* value);
bool skiplist_delete(lockfree_skiplist_t* list, int key);
void* skiplist_search(lockfree_skiplist_t* list, int key);
void skiplist_destroy(lockfree_skiplist_t* list);
```

**Implementation Requirements:**
- Handle ABA problem correctly using hazard pointers or epochs
- Achieve lock-free progress guarantee
- Support concurrent readers and writers
- Maintain probabilistic balancing properties

**Performance Analysis:**
- Benchmark against std::map and concurrent hash tables
- Measure performance with varying contention levels
- Analyze memory usage and cache behavior
- Test scalability up to 64 threads

**Evaluation Criteria:**
- Correctness under high concurrency (5 points)
- Performance optimization and scalability (5 points)
- Memory management and ABA prevention (3 points)
- Code quality and documentation (2 points)

#### **Challenge 2: CPU Cache Optimization (10 points)**

Optimize matrix-vector multiplication for modern CPU architectures:

```c
// Baseline implementation
void matvec_baseline(double* matrix, double* vector, double* result, 
                    int rows, int cols);

// Your optimized implementations:
void matvec_cache_blocked(double* matrix, double* vector, double* result, 
                         int rows, int cols, int block_size);
void matvec_simd_optimized(double* matrix, double* vector, double* result, 
                          int rows, int cols);
void matvec_numa_aware(double* matrix, double* vector, double* result, 
                      int rows, int cols, int num_threads);
```

**Optimization Targets:**
- Achieve >80% of theoretical peak FLOPS
- Minimize cache misses at all levels (L1, L2, L3)
- Scale linearly with thread count up to memory bandwidth limit
- Support matrices up to 100,000 x 100,000 elements

**Analysis Requirements:**
- Profile cache behavior using hardware performance counters
- Measure memory bandwidth utilization
- Compare against optimized BLAS implementations
- Document performance on different CPU architectures

#### **Challenge 3: Real-Time Thread Scheduler (10 points)**

Implement a custom thread scheduler for real-time applications:

```c
typedef struct {
    int thread_id;
    int priority;               // 0 = highest, higher numbers = lower priority
    struct timespec deadline;   // Absolute deadline
    struct timespec period;     // For periodic tasks
    struct timespec wcet;       // Worst-case execution time
    
    void (*task_function)(void* data);
    void* task_data;
    
    // Scheduling state
    enum { READY, RUNNING, BLOCKED, DEADLINE_MISSED } state;
    struct timespec last_execution;
    atomic_ulong executions;
    atomic_ulong deadline_misses;
} rt_task_t;

// Scheduler API:
typedef struct rt_scheduler rt_scheduler_t;

rt_scheduler_t* rt_scheduler_create(int num_cores);
bool rt_scheduler_admit_task(rt_scheduler_t* sched, rt_task_t* task);
void rt_scheduler_start(rt_scheduler_t* sched);
void rt_scheduler_stop(rt_scheduler_t* sched);
rt_task_t* rt_scheduler_get_next_task(rt_scheduler_t* sched);
void rt_scheduler_task_completed(rt_scheduler_t* sched, rt_task_t* task);
```

**Requirements:**
- Implement Earliest Deadline First (EDF) scheduling
- Support admission control with schedulability analysis
- Handle deadline misses gracefully
- Provide real-time guarantees for admitted tasks
- Support both periodic and aperiodic tasks

### Part III: System Design and Architecture (40 points)

#### **Design Challenge: High-Performance Computing Cluster Scheduler**

Design a complete scheduling system for a 1000-node HPC cluster with the following characteristics:
- 64 cores per node (64,000 total cores)
- Mixed workloads: batch jobs, interactive sessions, real-time analytics
- Heterogeneous hardware: CPU, GPU, FPGA nodes
- Network: 100Gb/s InfiniBand with 1μs latency
- Storage: Parallel file system with 100GB/s aggregate bandwidth

#### **Architecture Requirements (15 points)**

**Design and justify your architectural decisions for:**

1. **Scheduler Architecture (5 points)**
   ```c
   typedef struct {
       // Core scheduler components
       job_queue_t* pending_jobs;
       resource_manager_t* resource_mgr;
       load_balancer_t* load_balancer;
       
       // Performance optimization
       prediction_engine_t* performance_predictor;
       placement_optimizer_t* placement_opt;
       
       // Fault tolerance
       checkpoint_manager_t* checkpoint_mgr;
       failure_detector_t* failure_detect;
       
   } cluster_scheduler_t;
   ```

2. **Resource Management (5 points)**
   - CPU/GPU/memory allocation strategies
   - Network bandwidth reservation
   - Storage I/O scheduling
   - Power management and thermal constraints

3. **Performance Optimization (5 points)**
   - Job placement algorithms considering data locality
   - Dynamic load balancing across heterogeneous resources
   - Predictive scaling based on workload patterns
   - Network topology awareness for communication-heavy jobs

#### **Threading and Concurrency Design (15 points)**

**Design the internal threading architecture:**

1. **Scheduler Thread Pool (5 points)**
   ```c
   // Design considerations:
   // - How many threads for different scheduler components?
   // - How to minimize lock contention in job queues?
   // - How to handle priority inversion?
   // - How to ensure real-time response for high-priority jobs?
   ```

2. **Inter-Node Communication (5 points)**
   ```c
   // Address these aspects:
   // - Asynchronous vs. synchronous communication patterns
   // - Message serialization and network protocol design
   // - Fault tolerance and network partition handling
   // - Flow control and congestion management
   ```

3. **Data Management and Consistency (5 points)**
   ```c
   // Consider these challenges:
   // - Distributed state management across 1000 nodes
   // - Consistency guarantees for resource allocation
   // - Transaction handling for complex job submissions
   // - Metadata caching and invalidation strategies
   ```

#### **Performance and Reliability Analysis (10 points)**

**Provide detailed analysis of:**

1. **Scalability Projections (3 points)**
   - Expected performance with 10,000 concurrent jobs
   - Scheduler overhead per node
   - Network bandwidth requirements for coordination
   - Storage requirements for logging and metadata

2. **Fault Tolerance Strategy (4 points)**
   - Single points of failure and mitigation strategies
   - Recovery time objectives for different failure scenarios
   - Data consistency during partial system failures
   - Automated testing and chaos engineering approaches

3. **Performance Monitoring (3 points)**
   - Real-time metrics collection with minimal overhead
   - Anomaly detection and automated response
   - Capacity planning and trend analysis
   - Integration with external monitoring systems

### Assessment Scoring Framework

#### **Mastery Levels:**

**Expert Level (90-100 points)**
- Demonstrates deep understanding of performance engineering principles
- Provides innovative solutions to complex problems
- Shows mastery of both theoretical concepts and practical implementation
- Code is production-ready with excellent documentation

**Advanced Level (75-89 points)** 
- Strong grasp of performance concepts with good practical application
- Implements most requirements correctly with minor issues
- Shows good problem-solving skills and optimization techniques
- Code is well-structured with adequate documentation

**Intermediate Level (60-74 points)**
- Basic understanding of performance concepts
- Implements core requirements but may miss advanced optimizations
- Some gaps in theoretical knowledge or practical application
- Code works but may lack robustness or optimization

**Developing Level (45-59 points)**
- Limited understanding of performance engineering
- Struggles with complex optimization problems
- Implementation has significant issues or missing features
- Requires substantial improvement in both theory and practice

**Novice Level (Below 45 points)**
- Minimal understanding of performance concepts
- Unable to implement requirements correctly
- Lacks problem-solving skills for performance challenges
- Needs significant additional study and practice

#### **Detailed Scoring Rubric:**

**Technical Accuracy (40%)**
- Correct implementation of algorithms and data structures
- Proper use of synchronization primitives
- Memory management and resource cleanup
- Error handling and edge case coverage

**Performance Optimization (30%)**
- Achievement of performance targets
- Effective use of optimization techniques
- Scalability and efficiency considerations
- Hardware-aware programming

**System Design (20%)**
- Architectural decisions and trade-offs
- Scalability and maintainability
- Integration and modularity
- Real-world applicability

**Communication and Documentation (10%)**
- Clear explanation of design decisions
- Comprehensive documentation
- Effective presentation of results
- Professional code organization

### Continuous Performance Engineering Development

#### **Advanced Topics for Further Study:**
1. **Quantum Computing Performance Models**
2. **Machine Learning-Driven Performance Optimization**
3. **Edge Computing and IoT Performance Constraints**
4. **Blockchain and Cryptocurrency Performance Engineering**
5. **Neuromorphic Computing Architectures**

#### **Industry Certifications and Standards:**
- Intel VTune Profiler Certification
- NVIDIA CUDA Performance Analysis
- ARM Performance Analysis Certification
- HPC Performance Engineering Certification

#### **Open Source Contribution Opportunities:**
- Linux kernel scheduler improvements
- LLVM compiler optimizations
- Database engine performance enhancements
- High-performance computing libraries

#### **Professional Development Path:**
1. **Performance Engineer**: Focus on application-level optimizations
2. **Systems Architect**: Design high-performance distributed systems  
3. **Research Scientist**: Develop new performance analysis techniques
4. **Technical Lead**: Guide performance engineering across organizations
5. **Distinguished Engineer**: Drive industry-wide performance standards

## Next Section
[Debugging Threaded Applications](08_Debugging_Threaded_Applications.md)
