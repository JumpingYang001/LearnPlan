# Project 5: Parallel Computation

## Objective
Implement parallel algorithms (merge sort and matrix multiplication) and compare performance with single-threaded versions. This project demonstrates divide-and-conquer parallelism and performance analysis.

## Requirements

### Basic Requirements
1. Implement parallel merge sort using recursive thread creation
2. Implement parallel matrix multiplication with thread-based decomposition
3. Compare performance with single-threaded implementations
4. Provide configurable parallelism levels
5. Include comprehensive performance analysis and benchmarking

### Advanced Requirements
1. Implement work-stealing parallel algorithms
2. Add cache-optimized versions for better performance
3. Support different data types (generic implementation)
4. Implement parallel quick sort and heap sort
5. Add NUMA-aware optimizations for multi-socket systems

## Implementation Guide

### Parallel Merge Sort

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

#define MAX_THREADS 16
#define SEQUENTIAL_THRESHOLD 1000

typedef struct {
    int* array;
    int* temp_array;
    int left;
    int right;
    int depth;
    int max_depth;
} MergeSortData;

// Sequential merge sort for small arrays
void sequential_merge_sort(int* array, int* temp, int left, int right) {
    if (left >= right) return;
    
    int mid = left + (right - left) / 2;
    
    sequential_merge_sort(array, temp, left, mid);
    sequential_merge_sort(array, temp, mid + 1, right);
    
    // Merge the two halves
    int i = left, j = mid + 1, k = left;
    
    while (i <= mid && j <= right) {
        if (array[i] <= array[j]) {
            temp[k++] = array[i++];
        } else {
            temp[k++] = array[j++];
        }
    }
    
    while (i <= mid) temp[k++] = array[i++];
    while (j <= right) temp[k++] = array[j++];
    
    // Copy back to original array
    for (i = left; i <= right; i++) {
        array[i] = temp[i];
    }
}

// Merge two sorted subarrays
void merge(int* array, int* temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    
    while (i <= mid && j <= right) {
        if (array[i] <= array[j]) {
            temp[k++] = array[i++];
        } else {
            temp[k++] = array[j++];
        }
    }
    
    while (i <= mid) temp[k++] = array[i++];
    while (j <= right) temp[k++] = array[j++];
    
    // Copy back to original array
    for (i = left; i <= right; i++) {
        array[i] = temp[i];
    }
}

void* parallel_merge_sort_thread(void* arg) {
    MergeSortData* data = (MergeSortData*)arg;
    
    if (data->left >= data->right) {
        return NULL;
    }
    
    int size = data->right - data->left + 1;
    
    // Use sequential sort for small arrays or when max depth reached
    if (size <= SEQUENTIAL_THRESHOLD || data->depth >= data->max_depth) {
        sequential_merge_sort(data->array, data->temp_array, data->left, data->right);
        return NULL;
    }
    
    int mid = data->left + (data->right - data->left) / 2;
    
    // Create data for left and right halves
    MergeSortData left_data = {
        .array = data->array,
        .temp_array = data->temp_array,
        .left = data->left,
        .right = mid,
        .depth = data->depth + 1,
        .max_depth = data->max_depth
    };
    
    MergeSortData right_data = {
        .array = data->array,
        .temp_array = data->temp_array,
        .left = mid + 1,
        .right = data->right,
        .depth = data->depth + 1,
        .max_depth = data->max_depth
    };
    
    pthread_t left_thread, right_thread;
    
    // Create threads for left and right halves
    pthread_create(&left_thread, NULL, parallel_merge_sort_thread, &left_data);
    pthread_create(&right_thread, NULL, parallel_merge_sort_thread, &right_data);
    
    // Wait for both threads to complete
    pthread_join(left_thread, NULL);
    pthread_join(right_thread, NULL);
    
    // Merge the sorted halves
    merge(data->array, data->temp_array, data->left, mid, data->right);
    
    return NULL;
}

void parallel_merge_sort(int* array, int size, int max_threads) {
    if (size <= 1) return;
    
    int* temp_array = malloc(size * sizeof(int));
    if (!temp_array) {
        printf("Memory allocation failed\n");
        return;
    }
    
    // Calculate maximum depth based on number of threads
    int max_depth = 0;
    int threads = 1;
    while (threads < max_threads && threads < size / SEQUENTIAL_THRESHOLD) {
        max_depth++;
        threads *= 2;
    }
    
    MergeSortData data = {
        .array = array,
        .temp_array = temp_array,
        .left = 0,
        .right = size - 1,
        .depth = 0,
        .max_depth = max_depth
    };
    
    parallel_merge_sort_thread(&data);
    
    free(temp_array);
}

// Single-threaded merge sort for comparison
void single_threaded_merge_sort(int* array, int size) {
    if (size <= 1) return;
    
    int* temp_array = malloc(size * sizeof(int));
    if (!temp_array) {
        printf("Memory allocation failed\n");
        return;
    }
    
    sequential_merge_sort(array, temp_array, 0, size - 1);
    
    free(temp_array);
}
```

### Parallel Matrix Multiplication

```c
#define CACHE_LINE_SIZE 64

typedef struct {
    double** matrix_a;
    double** matrix_b;
    double** result;
    int start_row;
    int end_row;
    int size;
} MatrixMultData;

// Allocate cache-aligned matrix
double** allocate_matrix(int size) {
    double** matrix = malloc(size * sizeof(double*));
    if (!matrix) return NULL;
    
    for (int i = 0; i < size; i++) {
        matrix[i] = aligned_alloc(CACHE_LINE_SIZE, size * sizeof(double));
        if (!matrix[i]) {
            // Free already allocated rows
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    
    return matrix;
}

void free_matrix(double** matrix, int size) {
    if (!matrix) return;
    
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void initialize_matrix_random(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX * 100.0;
        }
    }
}

void initialize_matrix_zero(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = 0.0;
        }
    }
}

// Single-threaded matrix multiplication
void single_threaded_matrix_multiply(double** a, double** b, double** result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = 0.0;
            for (int k = 0; k < size; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Cache-optimized single-threaded matrix multiplication (blocked)
void blocked_matrix_multiply(double** a, double** b, double** result, int size, int block_size) {
    for (int i = 0; i < size; i += block_size) {
        for (int j = 0; j < size; j += block_size) {
            for (int k = 0; k < size; k += block_size) {
                // Multiply blocks
                int max_i = (i + block_size < size) ? i + block_size : size;
                int max_j = (j + block_size < size) ? j + block_size : size;
                int max_k = (k + block_size < size) ? k + block_size : size;
                
                for (int ii = i; ii < max_i; ii++) {
                    for (int jj = j; jj < max_j; jj++) {
                        double sum = result[ii][jj];
                        for (int kk = k; kk < max_k; kk++) {
                            sum += a[ii][kk] * b[kk][jj];
                        }
                        result[ii][jj] = sum;
                    }
                }
            }
        }
    }
}

void* parallel_matrix_multiply_thread(void* arg) {
    MatrixMultData* data = (MatrixMultData*)arg;
    
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->size; j++) {
            data->result[i][j] = 0.0;
            for (int k = 0; k < data->size; k++) {
                data->result[i][j] += data->matrix_a[i][k] * data->matrix_b[k][j];
            }
        }
    }
    
    return NULL;
}

void parallel_matrix_multiply(double** a, double** b, double** result, int size, int num_threads) {
    if (num_threads > size) {
        num_threads = size;
    }
    
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    MatrixMultData* thread_data = malloc(num_threads * sizeof(MatrixMultData));
    
    int rows_per_thread = size / num_threads;
    int extra_rows = size % num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].matrix_a = a;
        thread_data[i].matrix_b = b;
        thread_data[i].result = result;
        thread_data[i].size = size;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i + 1) * rows_per_thread;
        
        // Distribute extra rows among first few threads
        if (i < extra_rows) {
            thread_data[i].end_row++;
        }
        if (i > 0 && extra_rows > 0 && i <= extra_rows) {
            thread_data[i].start_row += i;
            thread_data[i].end_row += i;
        } else if (i > extra_rows) {
            thread_data[i].start_row += extra_rows;
            thread_data[i].end_row += extra_rows;
        }
        
        pthread_create(&threads[i], NULL, parallel_matrix_multiply_thread, &thread_data[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    free(threads);
    free(thread_data);
}

// Cache-optimized parallel matrix multiplication
typedef struct {
    double** matrix_a;
    double** matrix_b;
    double** result;
    int start_row;
    int end_row;
    int size;
    int block_size;
} BlockedMatrixMultData;

void* blocked_parallel_matrix_multiply_thread(void* arg) {
    BlockedMatrixMultData* data = (BlockedMatrixMultData*)arg;
    
    for (int i = data->start_row; i < data->end_row; i += data->block_size) {
        int max_i = (i + data->block_size < data->end_row) ? i + data->block_size : data->end_row;
        
        for (int j = 0; j < data->size; j += data->block_size) {
            int max_j = (j + data->block_size < data->size) ? j + data->block_size : data->size;
            
            for (int k = 0; k < data->size; k += data->block_size) {
                int max_k = (k + data->block_size < data->size) ? k + data->block_size : data->size;
                
                // Multiply blocks
                for (int ii = i; ii < max_i; ii++) {
                    for (int jj = j; jj < max_j; jj++) {
                        double sum = data->result[ii][jj];
                        for (int kk = k; kk < max_k; kk++) {
                            sum += data->matrix_a[ii][kk] * data->matrix_b[kk][jj];
                        }
                        data->result[ii][jj] = sum;
                    }
                }
            }
        }
    }
    
    return NULL;
}

void blocked_parallel_matrix_multiply(double** a, double** b, double** result, 
                                     int size, int num_threads, int block_size) {
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    BlockedMatrixMultData* thread_data = malloc(num_threads * sizeof(BlockedMatrixMultData));
    
    int rows_per_thread = size / num_threads;
    int extra_rows = size % num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].matrix_a = a;
        thread_data[i].matrix_b = b;
        thread_data[i].result = result;
        thread_data[i].size = size;
        thread_data[i].block_size = block_size;
        thread_data[i].start_row = i * rows_per_thread + (i < extra_rows ? i : extra_rows);
        thread_data[i].end_row = thread_data[i].start_row + rows_per_thread + (i < extra_rows ? 1 : 0);
        
        pthread_create(&threads[i], NULL, blocked_parallel_matrix_multiply_thread, &thread_data[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    free(threads);
    free(thread_data);
}
```

### Performance Benchmarking Framework

```c
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

typedef struct {
    int size;
    int threads;
    double sequential_time;
    double parallel_time;
    double speedup;
    double efficiency;
    double gflops; // For matrix multiplication
} BenchmarkResult;

BenchmarkResult benchmark_merge_sort(int size, int num_threads) {
    BenchmarkResult result = {0};
    result.size = size;
    result.threads = num_threads;
    
    // Generate random data
    int* original_data = malloc(size * sizeof(int));
    int* single_data = malloc(size * sizeof(int));
    int* parallel_data = malloc(size * sizeof(int));
    
    srand(42); // Fixed seed for reproducible results
    for (int i = 0; i < size; i++) {
        original_data[i] = rand() % 10000;
    }
    
    // Single-threaded benchmark
    memcpy(single_data, original_data, size * sizeof(int));
    double start_time = get_time();
    single_threaded_merge_sort(single_data, size);
    result.sequential_time = get_time() - start_time;
    
    // Parallel benchmark
    memcpy(parallel_data, original_data, size * sizeof(int));
    start_time = get_time();
    parallel_merge_sort(parallel_data, size, num_threads);
    result.parallel_time = get_time() - start_time;
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < size; i++) {
        if (single_data[i] != parallel_data[i]) {
            correct = false;
            break;
        }
    }
    
    if (!correct) {
        printf("ERROR: Parallel sort produced incorrect results!\n");
    }
    
    // Calculate metrics
    result.speedup = result.sequential_time / result.parallel_time;
    result.efficiency = result.speedup / num_threads;
    
    free(original_data);
    free(single_data);
    free(parallel_data);
    
    return result;
}

BenchmarkResult benchmark_matrix_multiply(int size, int num_threads) {
    BenchmarkResult result = {0};
    result.size = size;
    result.threads = num_threads;
    
    // Allocate matrices
    double** a = allocate_matrix(size);
    double** b = allocate_matrix(size);
    double** result_single = allocate_matrix(size);
    double** result_parallel = allocate_matrix(size);
    
    if (!a || !b || !result_single || !result_parallel) {
        printf("Matrix allocation failed\n");
        return result;
    }
    
    // Initialize matrices
    srand(42);
    initialize_matrix_random(a, size);
    initialize_matrix_random(b, size);
    
    // Single-threaded benchmark
    initialize_matrix_zero(result_single, size);
    double start_time = get_time();
    single_threaded_matrix_multiply(a, b, result_single, size);
    result.sequential_time = get_time() - start_time;
    
    // Parallel benchmark
    initialize_matrix_zero(result_parallel, size);
    start_time = get_time();
    parallel_matrix_multiply(a, b, result_parallel, size, num_threads);
    result.parallel_time = get_time() - start_time;
    
    // Verify correctness (sample check)
    bool correct = true;
    for (int i = 0; i < size && correct; i += size / 10) {
        for (int j = 0; j < size && correct; j += size / 10) {
            if (fabs(result_single[i][j] - result_parallel[i][j]) > 1e-6) {
                correct = false;
            }
        }
    }
    
    if (!correct) {
        printf("ERROR: Parallel matrix multiplication produced incorrect results!\n");
    }
    
    // Calculate metrics
    result.speedup = result.sequential_time / result.parallel_time;
    result.efficiency = result.speedup / num_threads;
    
    // Calculate GFLOPS (2*n^3 operations for matrix multiplication)
    double operations = 2.0 * size * size * size;
    result.gflops = (operations / result.parallel_time) / 1e9;
    
    free_matrix(a, size);
    free_matrix(b, size);
    free_matrix(result_single, size);
    free_matrix(result_parallel, size);
    
    return result;
}

void run_merge_sort_benchmarks() {
    printf("\n=== Merge Sort Performance Benchmarks ===\n");
    printf("%-10s %-8s %-12s %-12s %-10s %-12s\n", 
           "Size", "Threads", "Sequential", "Parallel", "Speedup", "Efficiency");
    printf("%.70s\n", "----------------------------------------------------------------------");
    
    int sizes[] = {100000, 500000, 1000000, 2000000};
    int thread_counts[] = {1, 2, 4, 8, 16};
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            BenchmarkResult result = benchmark_merge_sort(sizes[i], thread_counts[j]);
            
            printf("%-10d %-8d %-12.6f %-12.6f %-10.2f %-12.2f\n",
                   result.size, result.threads, result.sequential_time,
                   result.parallel_time, result.speedup, result.efficiency);
        }
        printf("\n");
    }
}

void run_matrix_multiply_benchmarks() {
    printf("\n=== Matrix Multiplication Performance Benchmarks ===\n");
    printf("%-8s %-8s %-12s %-12s %-10s %-12s %-10s\n", 
           "Size", "Threads", "Sequential", "Parallel", "Speedup", "Efficiency", "GFLOPS");
    printf("%.75s\n", "---------------------------------------------------------------------------");
    
    int sizes[] = {256, 512, 1024};
    int thread_counts[] = {1, 2, 4, 8};
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            BenchmarkResult result = benchmark_matrix_multiply(sizes[i], thread_counts[j]);
            
            printf("%-8d %-8d %-12.6f %-12.6f %-10.2f %-12.2f %-10.2f\n",
                   result.size, result.threads, result.sequential_time,
                   result.parallel_time, result.speedup, result.efficiency, result.gflops);
        }
        printf("\n");
    }
}
```

### Advanced: Work-Stealing Parallel Quick Sort

```c
#include <stdatomic.h>

#define MAX_WORK_QUEUES 16

typedef struct {
    int* array;
    int left;
    int right;
} WorkItem;

typedef struct {
    WorkItem* items;
    int capacity;
    _Atomic int head;
    _Atomic int tail;
    pthread_mutex_t mutex;
} WorkStealingQueue;

typedef struct {
    WorkStealingQueue* queues;
    int num_queues;
    int* array;
    int size;
    _Atomic int active_threads;
    _Atomic bool shutdown;
} WorkStealingPool;

WorkStealingQueue* ws_queue_create(int capacity) {
    WorkStealingQueue* queue = malloc(sizeof(WorkStealingQueue));
    if (!queue) return NULL;
    
    queue->items = malloc(sizeof(WorkItem) * capacity);
    if (!queue->items) {
        free(queue);
        return NULL;
    }
    
    queue->capacity = capacity;
    atomic_store(&queue->head, 0);
    atomic_store(&queue->tail, 0);
    pthread_mutex_init(&queue->mutex, NULL);
    
    return queue;
}

bool ws_queue_push(WorkStealingQueue* queue, const WorkItem* item) {
    pthread_mutex_lock(&queue->mutex);
    
    int head = atomic_load(&queue->head);
    int tail = atomic_load(&queue->tail);
    
    if ((tail + 1) % queue->capacity == head) {
        pthread_mutex_unlock(&queue->mutex);
        return false; // Queue full
    }
    
    queue->items[tail] = *item;
    atomic_store(&queue->tail, (tail + 1) % queue->capacity);
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

bool ws_queue_pop(WorkStealingQueue* queue, WorkItem* item) {
    pthread_mutex_lock(&queue->mutex);
    
    int head = atomic_load(&queue->head);
    int tail = atomic_load(&queue->tail);
    
    if (head == tail) {
        pthread_mutex_unlock(&queue->mutex);
        return false; // Queue empty
    }
    
    tail = (tail - 1 + queue->capacity) % queue->capacity;
    *item = queue->items[tail];
    atomic_store(&queue->tail, tail);
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

bool ws_queue_steal(WorkStealingQueue* queue, WorkItem* item) {
    pthread_mutex_lock(&queue->mutex);
    
    int head = atomic_load(&queue->head);
    int tail = atomic_load(&queue->tail);
    
    if (head == tail) {
        pthread_mutex_unlock(&queue->mutex);
        return false; // Queue empty
    }
    
    *item = queue->items[head];
    atomic_store(&queue->head, (head + 1) % queue->capacity);
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

int partition(int* array, int left, int right) {
    int pivot = array[right];
    int i = left - 1;
    
    for (int j = left; j < right; j++) {
        if (array[j] <= pivot) {
            i++;
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    
    int temp = array[i + 1];
    array[i + 1] = array[right];
    array[right] = temp;
    
    return i + 1;
}

void* work_stealing_quicksort_thread(void* arg) {
    WorkStealingPool* pool = (WorkStealingPool*)arg;
    int thread_id = atomic_fetch_add(&pool->active_threads, 1);
    WorkStealingQueue* my_queue = &pool->queues[thread_id];
    
    while (!atomic_load(&pool->shutdown)) {
        WorkItem item;
        bool found_work = false;
        
        // Try to get work from own queue
        if (ws_queue_pop(my_queue, &item)) {
            found_work = true;
        } else {
            // Try to steal work from other queues
            for (int i = 0; i < pool->num_queues; i++) {
                if (i != thread_id && ws_queue_steal(&pool->queues[i], &item)) {
                    found_work = true;
                    break;
                }
            }
        }
        
        if (found_work) {
            if (item.right > item.left) {
                if (item.right - item.left < 1000) {
                    // Use sequential sort for small arrays
                    sequential_merge_sort(item.array, 
                                        malloc((item.right - item.left + 1) * sizeof(int)),
                                        item.left, item.right);
                } else {
                    // Partition and create new work items
                    int pivot_index = partition(item.array, item.left, item.right);
                    
                    if (pivot_index - 1 > item.left) {
                        WorkItem left_item = {item.array, item.left, pivot_index - 1};
                        ws_queue_push(my_queue, &left_item);
                    }
                    
                    if (pivot_index + 1 < item.right) {
                        WorkItem right_item = {item.array, pivot_index + 1, item.right};
                        ws_queue_push(my_queue, &right_item);
                    }
                }
            }
        } else {
            // No work found, check if all threads are idle
            usleep(1000); // 1ms
        }
    }
    
    return NULL;
}
```

### CPU Cache and Memory Optimization

```c
// CPU cache-aware optimizations
void cache_optimized_matrix_transpose(double** matrix, double** result, int size) {
    const int block_size = 64; // Cache-friendly block size
    
    for (int i = 0; i < size; i += block_size) {
        for (int j = 0; j < size; j += block_size) {
            int max_i = (i + block_size < size) ? i + block_size : size;
            int max_j = (j + block_size < size) ? j + block_size : size;
            
            for (int ii = i; ii < max_i; ii++) {
                for (int jj = j; jj < max_j; jj++) {
                    result[jj][ii] = matrix[ii][jj];
                }
            }
        }
    }
}

// Memory bandwidth test
void measure_memory_bandwidth() {
    const int size = 100 * 1024 * 1024; // 100MB
    char* src = malloc(size);
    char* dst = malloc(size);
    
    if (!src || !dst) {
        printf("Memory allocation failed\n");
        return;
    }
    
    // Initialize source
    for (int i = 0; i < size; i++) {
        src[i] = i % 256;
    }
    
    double start_time = get_time();
    
    // Copy memory
    memcpy(dst, src, size);
    
    double end_time = get_time();
    
    double bandwidth_gb_s = (size / (1024.0 * 1024.0 * 1024.0)) / (end_time - start_time);
    
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    
    free(src);
    free(dst);
}

// NUMA awareness
#ifdef __linux__
#include <numa.h>
#include <numaif.h>

void set_numa_policy() {
    if (numa_available() >= 0) {
        printf("NUMA available with %d nodes\n", numa_max_node() + 1);
        
        // Set memory policy to interleave across all nodes
        numa_set_interleave_mask(numa_all_nodes_ptr);
        
        printf("Set NUMA memory policy to interleave\n");
    }
}

void bind_thread_to_node(int thread_id, int num_nodes) {
    int node = thread_id % num_nodes;
    numa_run_on_node(node);
    printf("Thread %d bound to NUMA node %d\n", thread_id, node);
}
#endif
```

### Comprehensive Test Suite

```c
void verify_sort_correctness(int* array, int size) {
    for (int i = 1; i < size; i++) {
        if (array[i] < array[i-1]) {
            printf("ERROR: Array not sorted at index %d (%d < %d)\n", 
                   i, array[i], array[i-1]);
            return;
        }
    }
    printf("Sort verification: PASSED\n");
}

void verify_matrix_multiplication(double** a, double** b, double** result, int size) {
    // Verify a few random elements
    srand(12345);
    for (int test = 0; test < 10; test++) {
        int i = rand() % size;
        int j = rand() % size;
        
        double expected = 0.0;
        for (int k = 0; k < size; k++) {
            expected += a[i][k] * b[k][j];
        }
        
        if (fabs(result[i][j] - expected) > 1e-10) {
            printf("ERROR: Matrix multiplication incorrect at (%d,%d): "
                   "expected %.10f, got %.10f\n", 
                   i, j, expected, result[i][j]);
            return;
        }
    }
    printf("Matrix multiplication verification: PASSED\n");
}

void run_comprehensive_tests() {
    printf("\n=== Comprehensive Parallel Computation Tests ===\n");
    
#ifdef __linux__
    set_numa_policy();
#endif
    
    measure_memory_bandwidth();
    
    // Test different problem sizes and thread counts
    run_merge_sort_benchmarks();
    run_matrix_multiply_benchmarks();
    
    // Scalability analysis
    printf("\n=== Scalability Analysis ===\n");
    
    int sizes[] = {1000000, 2000000};
    int max_threads = sysconf(_SC_NPROCESSORS_ONLN);
    
    printf("System has %d CPU cores\n", max_threads);
    
    for (int i = 0; i < 2; i++) {
        printf("\nMerge Sort Scalability (size: %d):\n", sizes[i]);
        printf("Threads\tTime(s)\t\tSpeedup\t\tEfficiency\n");
        
        BenchmarkResult baseline = benchmark_merge_sort(sizes[i], 1);
        
        for (int threads = 1; threads <= max_threads; threads *= 2) {
            BenchmarkResult result = benchmark_merge_sort(sizes[i], threads);
            double speedup = baseline.parallel_time / result.parallel_time;
            double efficiency = speedup / threads;
            
            printf("%d\t%.6f\t%.2f\t\t%.2f\n", 
                   threads, result.parallel_time, speedup, efficiency);
        }
    }
}

int main() {
    printf("Parallel Computation Performance Analysis\n");
    printf("==========================================\n");
    
    run_comprehensive_tests();
    
    return 0;
}
```

## Learning Objectives

After completing this project, you should understand:
- Parallel algorithm design and implementation
- Performance analysis and scalability measurement
- Cache optimization techniques for parallel algorithms
- Work-stealing and load balancing strategies
- NUMA-aware programming concepts
- Amdahl's Law and its practical implications

## Extensions

1. **GPU Acceleration**
   - Implement CUDA or OpenCL versions
   - Compare CPU vs GPU performance

2. **Distributed Computing**
   - Implement MPI versions for cluster computing
   - Network-based parallel algorithms

3. **Hybrid Approaches**
   - Combine different parallelization strategies
   - Auto-tuning for optimal performance

4. **Real-world Applications**
   - Image processing algorithms
   - Scientific computing applications

## Assessment Criteria

- **Correctness (25%)**: All algorithms produce correct results
- **Performance (30%)**: Significant speedup over sequential versions
- **Analysis (25%)**: Comprehensive performance analysis and insights
- **Code Quality (20%)**: Clean, optimized, well-documented code

## Conclusion

This project demonstrates the power and complexity of parallel computing, showing how proper algorithm design and implementation can dramatically improve performance for computationally intensive tasks.
