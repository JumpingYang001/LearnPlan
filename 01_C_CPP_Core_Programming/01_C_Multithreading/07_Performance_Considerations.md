# Performance Considerations

*Duration: 1 week*

## Overview

Understanding performance characteristics of multi-threaded applications is crucial for building efficient systems. This section covers thread overhead, optimization techniques, and performance analysis.

## Thread Overhead

### Thread Creation and Destruction Costs

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

void* minimal_thread(void* arg) {
    return NULL;
}

void measure_thread_creation_overhead() {
    const int num_threads = 1000;
    pthread_t threads[num_threads];
    double start_time, end_time;
    
    // Measure thread creation + join overhead
    start_time = get_time();
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, minimal_thread, NULL);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    end_time = get_time();
    
    printf("Created and joined %d threads in %.6f seconds\n", 
           num_threads, end_time - start_time);
    printf("Average per thread: %.6f seconds\n", 
           (end_time - start_time) / num_threads);
}

// Compare with thread pool performance
void measure_task_execution_overhead() {
    const int num_tasks = 10000;
    
    // Direct function calls
    double start_time = get_time();
    for (int i = 0; i < num_tasks; i++) {
        minimal_thread(NULL);
    }
    double direct_time = get_time() - start_time;
    
    printf("Direct function calls: %.6f seconds\n", direct_time);
    
    // Thread creation for each task
    start_time = get_time();
    for (int i = 0; i < 100; i++) { // Reduced count due to overhead
        pthread_t thread;
        pthread_create(&thread, NULL, minimal_thread, NULL);
        pthread_join(thread, NULL);
    }
    double thread_time = get_time() - start_time;
    
    printf("Thread per task (100 tasks): %.6f seconds\n", thread_time);
    printf("Overhead per thread: %.6f seconds\n", thread_time / 100);
}
```

### Memory Overhead Analysis

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/resource.h>

void print_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    
    printf("Memory usage: %ld KB\n", usage.ru_maxrss);
}

void* stack_consumer(void* arg) {
    // Allocate large stack variable to see stack usage
    char large_buffer[1024 * 1024]; // 1MB
    large_buffer[0] = 1; // Touch the memory
    sleep(1);
    return NULL;
}

void measure_memory_overhead() {
    printf("Initial memory usage:\n");
    print_memory_usage();
    
    const int num_threads = 100;
    pthread_t threads[num_threads];
    
    printf("\nCreating %d threads with large stacks:\n", num_threads);
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, stack_consumer, NULL);
    }
    
    print_memory_usage();
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("\nAfter joining all threads:\n");
    print_memory_usage();
}
```

## Context Switching Costs

### Measuring Context Switch Overhead

```c
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

volatile int switch_count = 0;
pthread_mutex_t switch_mutex = PTHREAD_MUTEX_INITIALIZER;

void* context_switch_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < 10000; i++) {
        pthread_mutex_lock(&switch_mutex);
        switch_count++;
        pthread_mutex_unlock(&switch_mutex);
        
        // Force context switch
        sched_yield();
    }
    
    return NULL;
}

void measure_context_switches() {
    const int num_threads = 4;
    pthread_t threads[num_threads];
    int thread_ids[num_threads];
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, context_switch_thread, &thread_ids[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Completed %d context switches in %.6f seconds\n", 
           switch_count, elapsed);
    printf("Average per switch: %.9f seconds\n", elapsed / switch_count);
}
```

### CPU Affinity for Performance

```c
#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
#include <stdio.h>

void set_thread_affinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        printf("Failed to set CPU affinity\n");
    } else {
        printf("Thread bound to CPU %d\n", cpu_id);
    }
}

void* cpu_bound_work(void* arg) {
    int cpu_id = *(int*)arg;
    set_thread_affinity(cpu_id);
    
    // CPU-intensive work
    volatile long sum = 0;
    for (long i = 0; i < 1000000000L; i++) {
        sum += i;
    }
    
    printf("CPU %d: Sum = %ld\n", cpu_id, sum);
    return NULL;
}

void test_cpu_affinity() {
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    printf("System has %d CPUs\n", num_cpus);
    
    pthread_t threads[num_cpus];
    int cpu_ids[num_cpus];
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < num_cpus; i++) {
        cpu_ids[i] = i;
        pthread_create(&threads[i], NULL, cpu_bound_work, &cpu_ids[i]);
    }
    
    for (int i = 0; i < num_cpus; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Completed CPU-bound work in %.6f seconds\n", elapsed);
}
```

## Cache Coherency Issues

### False Sharing Detection and Prevention

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

#define CACHE_LINE_SIZE 64
#define NUM_THREADS 4
#define ITERATIONS 10000000

// Bad: False sharing
struct bad_counters {
    volatile long counter[NUM_THREADS];
} bad_data;

// Good: Cache line aligned
struct good_counters {
    volatile long counter;
    char padding[CACHE_LINE_SIZE - sizeof(long)];
} good_data[NUM_THREADS] __attribute__((aligned(CACHE_LINE_SIZE)));

void* false_sharing_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < ITERATIONS; i++) {
        bad_data.counter[thread_id]++;
    }
    
    return NULL;
}

void* cache_friendly_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < ITERATIONS; i++) {
        good_data[thread_id].counter++;
    }
    
    return NULL;
}

double measure_performance(void* (*thread_func)(void*), const char* description) {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
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

## Scalability Analysis

### Amdahl's Law Demonstration

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

#define WORK_SIZE 1000000

typedef struct {
    int thread_id;
    int num_threads;
    double* data;
    double partial_result;
} ComputeData;

// Serial computation (cannot be parallelized)
double serial_computation(double* data, int size) {
    double result = 0.0;
    for (int i = 0; i < size / 10; i++) { // 10% of work is serial
        result += sqrt(data[i]);
    }
    return result;
}

// Parallel computation
void* parallel_computation(void* arg) {
    ComputeData* comp_data = (ComputeData*)arg;
    int start = comp_data->thread_id * (WORK_SIZE * 0.9) / comp_data->num_threads;
    int end = (comp_data->thread_id + 1) * (WORK_SIZE * 0.9) / comp_data->num_threads;
    
    comp_data->partial_result = 0.0;
    for (int i = start; i < end; i++) {
        comp_data->partial_result += sqrt(comp_data->data[WORK_SIZE / 10 + i]);
    }
    
    return NULL;
}

double measure_parallel_performance(int num_threads) {
    double* data = malloc(WORK_SIZE * sizeof(double));
    
    // Initialize data
    for (int i = 0; i < WORK_SIZE; i++) {
        data[i] = i + 1.0;
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Serial part (10% of work)
    double serial_result = serial_computation(data, WORK_SIZE);
    
    // Parallel part (90% of work)
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ComputeData* comp_data = malloc(num_threads * sizeof(ComputeData));
    
    for (int i = 0; i < num_threads; i++) {
        comp_data[i].thread_id = i;
        comp_data[i].num_threads = num_threads;
        comp_data[i].data = data;
        pthread_create(&threads[i], NULL, parallel_computation, &comp_data[i]);
    }
    
    double parallel_result = 0.0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        parallel_result += comp_data[i].partial_result;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    
    double total_result = serial_result + parallel_result;
    printf("Threads: %d, Time: %.6f s, Result: %.2f\n", 
           num_threads, elapsed, total_result);
    
    free(data);
    free(threads);
    free(comp_data);
    
    return elapsed;
}

void analyze_scalability() {
    printf("Analyzing scalability with Amdahl's Law:\n");
    printf("(10%% serial work, 90%% parallel work)\n\n");
    
    double baseline_time = measure_parallel_performance(1);
    
    printf("\nScalability analysis:\n");
    printf("Threads\tTime(s)\t\tSpeedup\t\tEfficiency\tTheoretical Max\n");
    
    for (int threads = 1; threads <= 8; threads *= 2) {
        double time = measure_parallel_performance(threads);
        double speedup = baseline_time / time;
        double efficiency = speedup / threads;
        
        // Amdahl's law: S = 1 / (F + (1-F)/N)
        // where F = 0.1 (serial fraction), N = number of threads
        double theoretical_max = 1.0 / (0.1 + 0.9 / threads);
        
        printf("%d\t%.6f\t%.2fx\t\t%.2f%%\t\t%.2fx\n", 
               threads, time, speedup, efficiency * 100, theoretical_max);
    }
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

## Exercises

1. **Performance Benchmarking Suite**
   - Create comprehensive threading performance tests
   - Measure various synchronization primitives
   - Generate performance reports

2. **Scalability Analysis Tool**
   - Implement automated scalability testing
   - Visualize performance vs thread count
   - Identify optimal thread pool sizes

3. **Cache-Aware Algorithm Design**
   - Implement cache-friendly parallel algorithms
   - Compare different memory access patterns
   - Optimize for specific CPU architectures

4. **Lock-Free Performance Comparison**
   - Compare lock-free vs locked implementations
   - Measure contention under different loads
   - Analyze performance characteristics

## Assessment

You should be able to:
- Measure and analyze threading performance overhead
- Identify and resolve cache coherency issues
- Apply Amdahl's Law to scalability analysis
- Optimize thread performance for specific workloads
- Use profiling tools effectively
- Design performance-conscious threaded applications

## Next Section
[Debugging Threaded Applications](08_Debugging_Threaded_Applications.md)
