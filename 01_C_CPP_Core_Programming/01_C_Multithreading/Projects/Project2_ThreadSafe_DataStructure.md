# Project 2: Thread-Safe Data Structure

## Objective
Implement a thread-safe linked list that supports concurrent add/remove/search operations. This project demonstrates synchronization mechanisms and thread-safe data structure design.

## Requirements

### Basic Requirements
1. Implement a thread-safe singly linked list
2. Support concurrent insert, delete, and search operations
3. Use mutex locks for synchronization
4. Handle edge cases (empty list, single element, etc.)
5. Provide proper error handling and memory management

### Advanced Requirements
1. Implement fine-grained locking (per-node locking)
2. Add performance benchmarks comparing different locking strategies
3. Implement reader-writer locks for optimized search operations
4. Add support for bulk operations
5. Implement lock-free operations using atomic operations

## Implementation Guide

### Basic Thread-Safe Linked List

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

typedef struct ListNode {
    int data;
    struct ListNode* next;
} ListNode;

typedef struct {
    ListNode* head;
    pthread_mutex_t mutex;
    size_t size;
} ThreadSafeList;

// Initialize the list
ThreadSafeList* list_create() {
    ThreadSafeList* list = malloc(sizeof(ThreadSafeList));
    if (!list) {
        return NULL;
    }
    
    list->head = NULL;
    list->size = 0;
    
    if (pthread_mutex_init(&list->mutex, NULL) != 0) {
        free(list);
        return NULL;
    }
    
    return list;
}

// Insert at the beginning of the list
bool list_insert(ThreadSafeList* list, int data) {
    if (!list) return false;
    
    ListNode* new_node = malloc(sizeof(ListNode));
    if (!new_node) return false;
    
    new_node->data = data;
    
    pthread_mutex_lock(&list->mutex);
    
    new_node->next = list->head;
    list->head = new_node;
    list->size++;
    
    pthread_mutex_unlock(&list->mutex);
    
    return true;
}

// Delete first occurrence of data
bool list_delete(ThreadSafeList* list, int data) {
    if (!list) return false;
    
    pthread_mutex_lock(&list->mutex);
    
    ListNode* current = list->head;
    ListNode* previous = NULL;
    
    while (current != NULL) {
        if (current->data == data) {
            if (previous == NULL) {
                // Deleting head node
                list->head = current->next;
            } else {
                previous->next = current->next;
            }
            
            free(current);
            list->size--;
            pthread_mutex_unlock(&list->mutex);
            return true;
        }
        
        previous = current;
        current = current->next;
    }
    
    pthread_mutex_unlock(&list->mutex);
    return false; // Not found
}

// Search for data in the list
bool list_search(ThreadSafeList* list, int data) {
    if (!list) return false;
    
    pthread_mutex_lock(&list->mutex);
    
    ListNode* current = list->head;
    while (current != NULL) {
        if (current->data == data) {
            pthread_mutex_unlock(&list->mutex);
            return true;
        }
        current = current->next;
    }
    
    pthread_mutex_unlock(&list->mutex);
    return false;
}

// Get the size of the list
size_t list_size(ThreadSafeList* list) {
    if (!list) return 0;
    
    pthread_mutex_lock(&list->mutex);
    size_t size = list->size;
    pthread_mutex_unlock(&list->mutex);
    
    return size;
}

// Print the list (for debugging)
void list_print(ThreadSafeList* list) {
    if (!list) return;
    
    pthread_mutex_lock(&list->mutex);
    
    printf("List contents: ");
    ListNode* current = list->head;
    while (current != NULL) {
        printf("%d ", current->data);
        current = current->next;
    }
    printf("(size: %zu)\n", list->size);
    
    pthread_mutex_unlock(&list->mutex);
}

// Destroy the list and free all memory
void list_destroy(ThreadSafeList* list) {
    if (!list) return;
    
    pthread_mutex_lock(&list->mutex);
    
    ListNode* current = list->head;
    while (current != NULL) {
        ListNode* temp = current;
        current = current->next;
        free(temp);
    }
    
    pthread_mutex_unlock(&list->mutex);
    pthread_mutex_destroy(&list->mutex);
    free(list);
}
```

### Test Framework

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define NUM_THREADS 8
#define OPERATIONS_PER_THREAD 1000

typedef struct {
    ThreadSafeList* list;
    int thread_id;
    int operations;
    int successful_inserts;
    int successful_deletes;
    int successful_searches;
} ThreadTestData;

void* test_thread_function(void* arg) {
    ThreadTestData* data = (ThreadTestData*)arg;
    
    srand(time(NULL) + data->thread_id);
    
    for (int i = 0; i < data->operations; i++) {
        int operation = rand() % 3; // 0: insert, 1: delete, 2: search
        int value = rand() % 100;
        
        switch (operation) {
            case 0: // Insert
                if (list_insert(data->list, value)) {
                    data->successful_inserts++;
                }
                break;
                
            case 1: // Delete
                if (list_delete(data->list, value)) {
                    data->successful_deletes++;
                }
                break;
                
            case 2: // Search
                if (list_search(data->list, value)) {
                    data->successful_searches++;
                }
                break;
        }
        
        // Small random delay to increase contention
        if (rand() % 100 < 5) {
            usleep(rand() % 1000);
        }
    }
    
    return NULL;
}

void run_concurrent_test() {
    ThreadSafeList* list = list_create();
    if (!list) {
        printf("Failed to create list\n");
        return;
    }
    
    pthread_t threads[NUM_THREADS];
    ThreadTestData thread_data[NUM_THREADS];
    struct timespec start_time, end_time;
    
    printf("Starting concurrent test with %d threads, %d operations each\n", 
           NUM_THREADS, OPERATIONS_PER_THREAD);
    
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].list = list;
        thread_data[i].thread_id = i;
        thread_data[i].operations = OPERATIONS_PER_THREAD;
        thread_data[i].successful_inserts = 0;
        thread_data[i].successful_deletes = 0;
        thread_data[i].successful_searches = 0;
        
        pthread_create(&threads[i], NULL, test_thread_function, &thread_data[i]);
    }
    
    // Join threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    // Calculate results
    int total_inserts = 0, total_deletes = 0, total_searches = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        total_inserts += thread_data[i].successful_inserts;
        total_deletes += thread_data[i].successful_deletes;
        total_searches += thread_data[i].successful_searches;
    }
    
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                         (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    printf("\nTest Results:\n");
    printf("Total successful inserts: %d\n", total_inserts);
    printf("Total successful deletes: %d\n", total_deletes);
    printf("Total successful searches: %d\n", total_searches);
    printf("Final list size: %zu\n", list_size(list));
    printf("Elapsed time: %.6f seconds\n", elapsed_time);
    printf("Operations per second: %.0f\n", 
           (NUM_THREADS * OPERATIONS_PER_THREAD) / elapsed_time);
    
    list_destroy(list);
}
```

### Advanced: Fine-Grained Locking

```c
typedef struct FineGrainedNode {
    int data;
    struct FineGrainedNode* next;
    pthread_mutex_t mutex;
} FineGrainedNode;

typedef struct {
    FineGrainedNode* head;
    pthread_mutex_t head_mutex;
    size_t size;
    pthread_mutex_t size_mutex;
} FineGrainedList;

FineGrainedList* fine_grained_list_create() {
    FineGrainedList* list = malloc(sizeof(FineGrainedList));
    if (!list) return NULL;
    
    list->head = NULL;
    list->size = 0;
    
    if (pthread_mutex_init(&list->head_mutex, NULL) != 0) {
        free(list);
        return NULL;
    }
    
    if (pthread_mutex_init(&list->size_mutex, NULL) != 0) {
        pthread_mutex_destroy(&list->head_mutex);
        free(list);
        return NULL;
    }
    
    return list;
}

bool fine_grained_list_insert(FineGrainedList* list, int data) {
    if (!list) return false;
    
    FineGrainedNode* new_node = malloc(sizeof(FineGrainedNode));
    if (!new_node) return false;
    
    new_node->data = data;
    if (pthread_mutex_init(&new_node->mutex, NULL) != 0) {
        free(new_node);
        return false;
    }
    
    pthread_mutex_lock(&list->head_mutex);
    
    new_node->next = list->head;
    list->head = new_node;
    
    pthread_mutex_unlock(&list->head_mutex);
    
    // Update size
    pthread_mutex_lock(&list->size_mutex);
    list->size++;
    pthread_mutex_unlock(&list->size_mutex);
    
    return true;
}

bool fine_grained_list_delete(FineGrainedList* list, int data) {
    if (!list) return false;
    
    pthread_mutex_lock(&list->head_mutex);
    
    if (list->head == NULL) {
        pthread_mutex_unlock(&list->head_mutex);
        return false;
    }
    
    // Check if head node should be deleted
    if (list->head->data == data) {
        FineGrainedNode* to_delete = list->head;
        pthread_mutex_lock(&to_delete->mutex);
        
        list->head = to_delete->next;
        
        pthread_mutex_unlock(&list->head_mutex);
        pthread_mutex_unlock(&to_delete->mutex);
        pthread_mutex_destroy(&to_delete->mutex);
        free(to_delete);
        
        // Update size
        pthread_mutex_lock(&list->size_mutex);
        list->size--;
        pthread_mutex_unlock(&list->size_mutex);
        
        return true;
    }
    
    // Search for node to delete
    FineGrainedNode* prev = list->head;
    pthread_mutex_lock(&prev->mutex);
    pthread_mutex_unlock(&list->head_mutex);
    
    FineGrainedNode* current = prev->next;
    
    while (current != NULL) {
        pthread_mutex_lock(&current->mutex);
        
        if (current->data == data) {
            prev->next = current->next;
            
            pthread_mutex_unlock(&prev->mutex);
            pthread_mutex_unlock(&current->mutex);
            pthread_mutex_destroy(&current->mutex);
            free(current);
            
            // Update size
            pthread_mutex_lock(&list->size_mutex);
            list->size--;
            pthread_mutex_unlock(&list->size_mutex);
            
            return true;
        }
        
        pthread_mutex_unlock(&prev->mutex);
        prev = current;
        current = current->next;
    }
    
    pthread_mutex_unlock(&prev->mutex);
    return false;
}

bool fine_grained_list_search(FineGrainedList* list, int data) {
    if (!list) return false;
    
    pthread_mutex_lock(&list->head_mutex);
    FineGrainedNode* current = list->head;
    
    if (current != NULL) {
        pthread_mutex_lock(&current->mutex);
    }
    pthread_mutex_unlock(&list->head_mutex);
    
    while (current != NULL) {
        if (current->data == data) {
            pthread_mutex_unlock(&current->mutex);
            return true;
        }
        
        FineGrainedNode* next = current->next;
        if (next != NULL) {
            pthread_mutex_lock(&next->mutex);
        }
        
        pthread_mutex_unlock(&current->mutex);
        current = next;
    }
    
    return false;
}
```

### Reader-Writer Lock Version

```c
typedef struct {
    ListNode* head;
    pthread_rwlock_t rwlock;
    size_t size;
} RWList;

RWList* rw_list_create() {
    RWList* list = malloc(sizeof(RWList));
    if (!list) return NULL;
    
    list->head = NULL;
    list->size = 0;
    
    if (pthread_rwlock_init(&list->rwlock, NULL) != 0) {
        free(list);
        return NULL;
    }
    
    return list;
}

bool rw_list_insert(RWList* list, int data) {
    if (!list) return false;
    
    ListNode* new_node = malloc(sizeof(ListNode));
    if (!new_node) return false;
    
    new_node->data = data;
    
    pthread_rwlock_wrlock(&list->rwlock);
    
    new_node->next = list->head;
    list->head = new_node;
    list->size++;
    
    pthread_rwlock_unlock(&list->rwlock);
    
    return true;
}

bool rw_list_search(RWList* list, int data) {
    if (!list) return false;
    
    pthread_rwlock_rdlock(&list->rwlock);
    
    ListNode* current = list->head;
    while (current != NULL) {
        if (current->data == data) {
            pthread_rwlock_unlock(&list->rwlock);
            return true;
        }
        current = current->next;
    }
    
    pthread_rwlock_unlock(&list->rwlock);
    return false;
}

bool rw_list_delete(RWList* list, int data) {
    if (!list) return false;
    
    pthread_rwlock_wrlock(&list->rwlock);
    
    ListNode* current = list->head;
    ListNode* previous = NULL;
    
    while (current != NULL) {
        if (current->data == data) {
            if (previous == NULL) {
                list->head = current->next;
            } else {
                previous->next = current->next;
            }
            
            free(current);
            list->size--;
            pthread_rwlock_unlock(&list->rwlock);
            return true;
        }
        
        previous = current;
        current = current->next;
    }
    
    pthread_rwlock_unlock(&list->rwlock);
    return false;
}
```

## Performance Benchmarking

```c
typedef struct {
    const char* name;
    double insert_time;
    double search_time;
    double delete_time;
    int operations;
} BenchmarkResult;

BenchmarkResult benchmark_list_implementation(
    void* list,
    bool (*insert_func)(void*, int),
    bool (*search_func)(void*, int),
    bool (*delete_func)(void*, int),
    const char* name
) {
    BenchmarkResult result = {0};
    result.name = name;
    result.operations = 10000;
    
    struct timespec start, end;
    
    // Benchmark inserts
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < result.operations; i++) {
        insert_func(list, i);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    result.insert_time = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Benchmark searches
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < result.operations; i++) {
        search_func(list, i / 2);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    result.search_time = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Benchmark deletes
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < result.operations / 2; i++) {
        delete_func(list, i);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    result.delete_time = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    
    return result;
}

void print_benchmark_results(BenchmarkResult* results, int count) {
    printf("\nBenchmark Results:\n");
    printf("%-20s %-12s %-12s %-12s\n", "Implementation", "Insert(s)", "Search(s)", "Delete(s)");
    printf("%.60s\n", "------------------------------------------------------------");
    
    for (int i = 0; i < count; i++) {
        printf("%-20s %-12.6f %-12.6f %-12.6f\n",
               results[i].name,
               results[i].insert_time,
               results[i].search_time,
               results[i].delete_time);
    }
}
```

## Testing and Validation

### Correctness Tests

```c
void test_basic_operations() {
    printf("Testing basic operations...\n");
    
    ThreadSafeList* list = list_create();
    
    // Test empty list
    assert(list_size(list) == 0);
    assert(!list_search(list, 1));
    assert(!list_delete(list, 1));
    
    // Test single element
    assert(list_insert(list, 1));
    assert(list_size(list) == 1);
    assert(list_search(list, 1));
    assert(!list_search(list, 2));
    
    // Test multiple elements
    assert(list_insert(list, 2));
    assert(list_insert(list, 3));
    assert(list_size(list) == 3);
    
    // Test deletion
    assert(list_delete(list, 2));
    assert(list_size(list) == 2);
    assert(!list_search(list, 2));
    
    list_destroy(list);
    printf("Basic operations test passed!\n");
}

void test_concurrent_operations() {
    printf("Testing concurrent operations...\n");
    
    run_concurrent_test();
    
    printf("Concurrent operations test completed!\n");
}
```

## Extensions

1. **Sorted Linked List**
   - Maintain sorted order during insertions
   - Optimize search operations

2. **Generic Data Type Support**
   - Use void* for data and function pointers for comparison
   - Implement type-safe wrappers

3. **Memory Pool**
   - Use a memory pool for node allocation
   - Reduce malloc/free overhead

4. **Lock-Free Implementation**
   - Use atomic operations and hazard pointers
   - Compare performance with locked versions

## Learning Objectives

After completing this project, you should understand:
- Thread-safe data structure design principles
- Different synchronization strategies (coarse-grained, fine-grained, reader-writer)
- Performance trade-offs between different locking strategies
- Concurrent testing and validation techniques
- Memory management in multi-threaded environments

## Common Pitfalls

1. **Deadlocks** in fine-grained locking
2. **Memory leaks** from incomplete cleanup
3. **Race conditions** in size counting
4. **ABA problems** in lock-free implementations
5. **Incorrect locking order** causing deadlocks

## Assessment Criteria

- **Correctness (30%)**: All operations work correctly under concurrent access
- **Performance (25%)**: Efficient implementation with good scalability
- **Code Quality (25%)**: Clean, well-documented, maintainable code
- **Advanced Features (20%)**: Implementation of advanced locking strategies

## Next Project
[Project 3: Producer-Consumer Queue](Project3_Producer_Consumer_Queue.md)
