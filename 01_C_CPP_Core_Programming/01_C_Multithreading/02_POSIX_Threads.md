# POSIX Threads (pthreads)

*Duration: 2 weeks*

## Overview

POSIX Threads (pthreads) is the standard threading API for Unix-like systems. This section covers the fundamental pthread functions and concepts.

## Core pthread Functions

### Thread Creation with `pthread_create()`

```c
#include <pthread.h>

int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine) (void *), void *arg);
```

**Parameters:**
- `thread`: Pointer to pthread_t variable to store thread ID
- `attr`: Thread attributes (NULL for default)
- `start_routine`: Function to execute in the new thread
- `arg`: Argument passed to start_routine

### Thread Joining with `pthread_join()`

```c
int pthread_join(pthread_t thread, void **retval);
```

**Purpose:**
- Wait for thread termination
- Retrieve thread's return value
- Clean up thread resources

### Thread Detachment with `pthread_detach()`

```c
int pthread_detach(pthread_t thread);
```

**Purpose:**
- Mark thread as detached
- Automatic resource cleanup when thread terminates
- Cannot be joined after detachment

### Thread Identification

```c
pthread_t pthread_self(void);
int pthread_equal(pthread_t t1, pthread_t t2);
```

**Functions:**
- `pthread_self()`: Get current thread ID
- `pthread_equal()`: Compare thread IDs

### Thread Attributes

```c
int pthread_attr_init(pthread_attr_t *attr);
int pthread_attr_destroy(pthread_attr_t *attr);
int pthread_attr_setdetachstate(pthread_attr_t *attr, int detachstate);
int pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize);
```

**Common Attributes:**
- Detach state (joinable/detached)
- Stack size
- Stack address
- Scheduling policy and priority

## Example Code Templates

### Basic Thread Creation Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void* thread_function(void* arg) {
    int thread_num = *(int*)arg;
    printf("Thread %d is running\n", thread_num);
    sleep(2);
    printf("Thread %d finished\n", thread_num);
    return NULL;
}

int main() {
    pthread_t threads[3];
    int thread_args[3];
    
    // Create threads
    for (int i = 0; i < 3; i++) {
        thread_args[i] = i;
        if (pthread_create(&threads[i], NULL, thread_function, &thread_args[i]) != 0) {
            perror("pthread_create");
            exit(1);
        }
    }
    
    // Join threads
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("All threads completed\n");
    return 0;
}
```

### Thread with Return Value Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void* calculate_square(void* arg) {
    int num = *(int*)arg;
    int* result = malloc(sizeof(int));
    *result = num * num;
    return result;
}

int main() {
    pthread_t thread;
    int input = 5;
    int* result;
    
    pthread_create(&thread, NULL, calculate_square, &input);
    pthread_join(thread, (void**)&result);
    
    printf("Square of %d is %d\n", input, *result);
    free(result);
    
    return 0;
}
```

## Key Concepts

### Thread Lifecycle
1. **Creation**: `pthread_create()` starts new thread
2. **Execution**: Thread runs start_routine function
3. **Termination**: Thread returns or calls `pthread_exit()`
4. **Cleanup**: Resources freed (joinable threads need `pthread_join()`)

### Error Handling
- pthread functions return 0 on success, error code on failure
- Always check return values
- Use `perror()` or `strerror()` for error messages

### Best Practices
- Always join joinable threads or detach them
- Check return values of all pthread functions
- Pass thread arguments carefully (avoid stack variables)
- Use proper data types for thread functions

## Exercises

1. **Multi-threaded Hello World**
   - Create 5 threads that print "Hello from thread X"
   - Join all threads before program exit

2. **Thread with Parameters**
   - Pass different parameters to each thread
   - Have each thread perform different calculations

3. **Detached Threads**
   - Create detached threads that run independently
   - Compare behavior with joinable threads

4. **Thread Attributes**
   - Experiment with different stack sizes
   - Create threads with custom attributes

## Common Pitfalls

1. **Passing stack variables to threads**
   ```c
   // WRONG
   for (int i = 0; i < 5; i++) {
       pthread_create(&threads[i], NULL, func, &i);
   }
   
   // CORRECT
   int args[5];
   for (int i = 0; i < 5; i++) {
       args[i] = i;
       pthread_create(&threads[i], NULL, func, &args[i]);
   }
   ```

2. **Forgetting to join threads**
   - Leads to resource leaks
   - Program may exit before threads complete

3. **Not checking return values**
   - Silent failures can be hard to debug

## Assessment

You should be able to:
- Create and manage threads using pthread API
- Properly pass arguments to thread functions
- Handle thread termination and resource cleanup
- Use thread attributes effectively
- Debug basic threading issues

## Next Section
[Thread Synchronization Mechanisms](03_Thread_Synchronization.md)
