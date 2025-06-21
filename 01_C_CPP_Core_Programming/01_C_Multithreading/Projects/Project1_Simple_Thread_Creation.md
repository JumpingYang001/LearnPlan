# Project 1: Simple Thread Creation

## Objective
Create multiple threads that print their IDs and join all threads before program exit. This project demonstrates basic thread creation, identification, and lifecycle management.

## Requirements

### Basic Requirements
1. Create 5 threads using `pthread_create()`
2. Each thread should print its thread ID and a custom message
3. Join all threads before program termination
4. Handle errors properly (check return values)
5. Ensure proper resource cleanup

### Advanced Requirements
1. Pass unique data to each thread
2. Collect return values from threads
3. Implement timeout for thread joins
4. Add thread attributes (detached vs joinable)
5. Measure thread creation and join overhead

## Implementation Guide

### Basic Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 5

typedef struct {
    int thread_number;
    char message[100];
} ThreadData;

void* thread_function(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    printf("Thread %d (ID: %lu): %s\n", 
           data->thread_number, 
           (unsigned long)pthread_self(), 
           data->message);
    
    // Simulate some work
    sleep(1);
    
    printf("Thread %d completed\n", data->thread_number);
    
    // Return some data
    int* result = malloc(sizeof(int));
    *result = data->thread_number * 100;
    return result;
}

int main() {
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    
    printf("Creating %d threads...\n", NUM_THREADS);
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_number = i + 1;
        snprintf(thread_data[i].message, sizeof(thread_data[i].message), 
                "Hello from thread %d", i + 1);
        
        int result = pthread_create(&threads[i], NULL, thread_function, &thread_data[i]);
        if (result != 0) {
            fprintf(stderr, "Error creating thread %d: %s\n", i, strerror(result));
            exit(1);
        }
    }
    
    printf("All threads created successfully\n");
    
    // Join threads and collect results
    for (int i = 0; i < NUM_THREADS; i++) {
        void* thread_result;
        int result = pthread_join(threads[i], &thread_result);
        
        if (result != 0) {
            fprintf(stderr, "Error joining thread %d: %s\n", i, strerror(result));
        } else {
            int* return_value = (int*)thread_result;
            printf("Thread %d returned: %d\n", i + 1, *return_value);
            free(return_value);
        }
    }
    
    printf("All threads completed\n");
    return 0;
}
```

### Advanced Implementation with Attributes

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <errno.h>

#define NUM_THREADS 5
#define STACK_SIZE (1024 * 1024) // 1MB stack

typedef struct {
    int thread_id;
    int work_amount;
    struct timespec start_time;
    struct timespec end_time;
} AdvancedThreadData;

void* advanced_thread_function(void* arg) {
    AdvancedThreadData* data = (AdvancedThreadData*)arg;
    
    // Record start time
    clock_gettime(CLOCK_MONOTONIC, &data->start_time);
    
    printf("Advanced Thread %d started (pthread_self: %lu)\n", 
           data->thread_id, (unsigned long)pthread_self());
    
    // Simulate variable work
    for (int i = 0; i < data->work_amount; i++) {
        // Some CPU work
        volatile int sum = 0;
        for (int j = 0; j < 100000; j++) {
            sum += j;
        }
    }
    
    // Record end time
    clock_gettime(CLOCK_MONOTONIC, &data->end_time);
    
    printf("Advanced Thread %d completed\n", data->thread_id);
    
    return NULL;
}

int create_thread_with_attributes(pthread_t* thread, AdvancedThreadData* data, 
                                 bool detached, size_t stack_size) {
    pthread_attr_t attr;
    int result;
    
    // Initialize attributes
    result = pthread_attr_init(&attr);
    if (result != 0) {
        fprintf(stderr, "pthread_attr_init failed: %s\n", strerror(result));
        return result;
    }
    
    // Set detach state
    if (detached) {
        result = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        if (result != 0) {
            fprintf(stderr, "pthread_attr_setdetachstate failed: %s\n", strerror(result));
            pthread_attr_destroy(&attr);
            return result;
        }
    }
    
    // Set stack size
    if (stack_size > 0) {
        result = pthread_attr_setstacksize(&attr, stack_size);
        if (result != 0) {
            fprintf(stderr, "pthread_attr_setstacksize failed: %s\n", strerror(result));
            pthread_attr_destroy(&attr);
            return result;
        }
    }
    
    // Create thread
    result = pthread_create(thread, &attr, advanced_thread_function, data);
    
    // Clean up attributes
    pthread_attr_destroy(&attr);
    
    return result;
}

int main() {
    pthread_t threads[NUM_THREADS];
    AdvancedThreadData thread_data[NUM_THREADS];
    struct timespec program_start, program_end;
    
    clock_gettime(CLOCK_MONOTONIC, &program_start);
    
    printf("Creating %d advanced threads...\n", NUM_THREADS);
    
    // Create threads with different configurations
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i + 1;
        thread_data[i].work_amount = (i + 1) * 10; // Variable work load
        
        bool detached = (i == NUM_THREADS - 1); // Last thread is detached
        size_t stack_size = (i % 2 == 0) ? STACK_SIZE : 0; // Alternate stack sizes
        
        int result = create_thread_with_attributes(&threads[i], &thread_data[i], 
                                                  detached, stack_size);
        if (result != 0) {
            fprintf(stderr, "Failed to create thread %d: %s\n", i + 1, strerror(result));
            exit(1);
        }
        
        printf("Thread %d created (detached: %s, custom stack: %s)\n", 
               i + 1, detached ? "yes" : "no", (stack_size > 0) ? "yes" : "no");
    }
    
    // Join joinable threads (all except the last one)
    for (int i = 0; i < NUM_THREADS - 1; i++) {
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_sec += 10; // 10 second timeout
        
        int result = pthread_timedjoin_np(threads[i], NULL, &timeout);
        if (result == ETIMEDOUT) {
            printf("Thread %d join timed out\n", i + 1);
        } else if (result != 0) {
            fprintf(stderr, "Error joining thread %d: %s\n", i + 1, strerror(result));
        } else {
            double execution_time = 
                (thread_data[i].end_time.tv_sec - thread_data[i].start_time.tv_sec) +
                (thread_data[i].end_time.tv_nsec - thread_data[i].start_time.tv_nsec) / 1e9;
            
            printf("Thread %d joined successfully (execution time: %.6f seconds)\n", 
                   i + 1, execution_time);
        }
    }
    
    // Wait a bit for the detached thread to complete
    printf("Waiting for detached thread to complete...\n");
    sleep(3);
    
    clock_gettime(CLOCK_MONOTONIC, &program_end);
    
    double total_time = 
        (program_end.tv_sec - program_start.tv_sec) +
        (program_end.tv_nsec - program_start.tv_nsec) / 1e9;
    
    printf("Program completed in %.6f seconds\n", total_time);
    
    return 0;
}
```

## Testing and Validation

### Test Cases

1. **Basic Functionality Test**
   ```bash
   # Compile and run
   gcc -pthread -o thread_creation project1_basic.c
   ./thread_creation
   
   # Expected: All threads create, execute, and join successfully
   ```

2. **Error Handling Test**
   ```c
   // Test with invalid thread creation
   for (int i = 0; i < 10000; i++) {
       pthread_t thread;
       int result = pthread_create(&thread, NULL, thread_function, NULL);
       if (result != 0) {
           printf("Thread creation failed at iteration %d: %s\n", i, strerror(result));
           break;
       }
       pthread_join(thread, NULL);
   }
   ```

3. **Performance Test**
   ```c
   // Measure thread creation overhead
   struct timespec start, end;
   clock_gettime(CLOCK_MONOTONIC, &start);
   
   // Create and join threads
   for (int i = 0; i < 1000; i++) {
       pthread_t thread;
       pthread_create(&thread, NULL, minimal_function, NULL);
       pthread_join(thread, NULL);
   }
   
   clock_gettime(CLOCK_MONOTONIC, &end);
   // Calculate and print overhead
   ```

## Extensions

1. **Thread Pool Version**
   - Create threads once and reuse them
   - Compare performance with creating new threads each time

2. **Different Thread Attributes**
   - Experiment with different stack sizes
   - Test scheduling policies and priorities

3. **Error Recovery**
   - Handle thread creation failures gracefully
   - Implement retry mechanisms

4. **Monitoring**
   - Add thread execution time measurement
   - Monitor resource usage (memory, CPU)

## Learning Objectives

After completing this project, you should understand:
- Basic thread creation and management
- Thread attributes and their effects
- Proper error handling in multi-threaded programs
- Resource cleanup and memory management
- Thread lifecycle and synchronization points
- Performance implications of thread creation

## Common Pitfalls to Avoid

1. **Not checking return values** from pthread functions
2. **Memory leaks** from not freeing thread return values
3. **Passing stack variables** to threads that outlive the function
4. **Not joining threads** leading to resource leaks
5. **Assuming thread execution order** without synchronization

## Grading Criteria

- **Correctness (40%)**: Program works as specified
- **Error Handling (20%)**: Proper error checking and handling
- **Code Quality (20%)**: Clean, well-documented code
- **Advanced Features (20%)**: Implementation of advanced requirements

## Next Project
[Project 2: Thread-Safe Data Structure](Project2_ThreadSafe_DataStructure.md)
