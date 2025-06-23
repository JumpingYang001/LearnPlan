# Debugging Threaded Applications

*Duration: 1 week*

# Debugging Threaded Applications: Mastering Concurrent Program Analysis

*Duration: 1 week*

## Overview

Debugging multi-threaded applications represents one of the most challenging aspects of systems programming. Unlike single-threaded programs where execution is deterministic and reproducible, concurrent programs exhibit non-deterministic behavior that can make bugs difficult to reproduce, isolate, and fix. This comprehensive guide provides you with the tools, techniques, and methodologies necessary to become an expert at debugging complex concurrent systems.

### The Complexity of Concurrent Debugging

Multi-threaded debugging is fundamentally different from traditional debugging because:

```c
// Traditional single-threaded bug - always reproduces the same way
int divide_by_zero_bug() {
    int x = 10;
    int y = 0;
    return x / y;  // Always crashes here
}

// Multi-threaded bug - may appear/disappear based on timing
static int shared_counter = 0;
static pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void* unreliable_increment(void* arg) {
    for (int i = 0; i < 1000000; i++) {
        // Sometimes works, sometimes doesn't - depends on scheduler timing
        if (rand() % 100 == 0) {  // 1% chance of "forgetting" to lock
            shared_counter++;     // RACE CONDITION - but only sometimes!
        } else {
            pthread_mutex_lock(&counter_mutex);
            shared_counter++;
            pthread_mutex_unlock(&counter_mutex);
        }
    }
    return NULL;
}

// The Heisenberg Effect: Observation changes behavior
void* heisenberg_thread(void* arg) {
    volatile int* flag = (volatile int*)arg;
    
    while (*flag == 0) {
        // Adding printf() for debugging might fix the race condition
        // by introducing timing delays!
        // printf("Waiting...\n");  // Commenting this might break the program
        
        // Or even worse - compiler optimizations might eliminate the loop
        // if we don't use 'volatile'
    }
    
    return NULL;
}
```

### The Debugging Challenge Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Threading Bug Complexity Pyramid                │
├─────────────────────────────────────────────────────────────────────┤
│  Expert Level    │  Distributed Race Conditions                    │
│                  │  Lock-Free Algorithm Bugs                       │
│                  │  Memory Ordering Issues                         │
├─────────────────────────────────────────────────────────────────────┤
│  Advanced Level  │  Complex Deadlocks (3+ locks)                  │
│                  │  Priority Inversion                             │
│                  │  ABA Problems                                   │
│                  │  False Sharing                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Intermediate    │  Simple Deadlocks (2 locks)                    │
│                  │  Producer-Consumer Issues                       │
│                  │  Thread Pool Problems                           │
├─────────────────────────────────────────────────────────────────────┤
│  Beginner Level  │  Basic Race Conditions                         │
│                  │  Unprotected Shared Variables                   │
│                  │  Missing Synchronization                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Essential Debugging Principles

#### 1. **Reproducibility is King**
```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

// Debugging framework to make bugs reproducible
typedef struct {
    unsigned int seed;
    int thread_count;
    int iteration_count;
    bool enable_delays;
    bool enable_logging;
    char* log_filename;
} debug_config_t;

// Global debug configuration
debug_config_t g_debug_config = {
    .seed = 12345,              // Fixed seed for reproducible randomness
    .thread_count = 4,
    .iteration_count = 100000,
    .enable_delays = true,      // Add deterministic delays
    .enable_logging = true,
    .log_filename = "debug.log"
};

// Reproducible random delays
void debug_delay(int thread_id, const char* location) {
    if (!g_debug_config.enable_delays) return;
    
    // Use thread_id and location to create deterministic delays
    unsigned int local_seed = g_debug_config.seed + thread_id + 
                             strlen(location);
    
    int delay_us = (rand_r(&local_seed) % 1000) + 1; // 1-1000 microseconds
    usleep(delay_us);
    
    if (g_debug_config.enable_logging) {
        printf("[DEBUG] Thread %d delayed %d μs at %s\n", 
               thread_id, delay_us, location);
    }
}

// Example of making a race condition more reproducible
static int shared_resource = 0;
static pthread_mutex_t resource_mutex = PTHREAD_MUTEX_INITIALIZER;

void* reproducible_race_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < g_debug_config.iteration_count; i++) {
        // Add strategic delays to expose race conditions
        debug_delay(thread_id, "before_critical_section");
        
        // Critical section - sometimes with proper locking
        if (i % 1000 != 0) {  // Introduce controlled race condition
            pthread_mutex_lock(&resource_mutex);
            int temp = shared_resource;
            debug_delay(thread_id, "inside_critical_section");
            shared_resource = temp + 1;
            pthread_mutex_unlock(&resource_mutex);
        } else {
            // Intentional race condition for debugging practice
            int temp = shared_resource;
            debug_delay(thread_id, "race_condition_window");
            shared_resource = temp + 1;
        }
        
        debug_delay(thread_id, "after_critical_section");
    }
    
    return NULL;
}
```

#### 2. **Systematic Bug Classification**
```c
#include <stdatomic.h>
#include <stdbool.h>

// Bug detection and classification system
typedef enum {
    BUG_RACE_CONDITION,
    BUG_DEADLOCK,
    BUG_LIVELOCK,
    BUG_STARVATION,
    BUG_PRIORITY_INVERSION,
    BUG_ABA_PROBLEM,
    BUG_FALSE_SHARING,
    BUG_MEMORY_ORDERING,
    BUG_RESOURCE_LEAK,
    BUG_UNKNOWN
} bug_type_t;

typedef struct {
    bug_type_t type;
    char description[256];
    char location[128];
    struct timespec timestamp;
    int thread_count;
    bool reproducible;
    int severity;  // 1-10 scale
} bug_report_t;

// Bug detection framework
typedef struct {
    atomic_int total_bugs_found;
    atomic_int bugs_by_type[BUG_UNKNOWN + 1];
    bug_report_t recent_bugs[100];
    atomic_int bug_index;
    
    // Detection flags
    atomic_bool deadlock_detected;
    atomic_int max_wait_time_ms;
    atomic_int race_condition_count;
    
    // Statistics
    atomic_ulong memory_accesses;
    atomic_ulong lock_acquisitions;
    atomic_ulong lock_contentions;
} bug_detector_t;

static bug_detector_t g_bug_detector = {0};

// Automatic bug detection
void report_bug(bug_type_t type, const char* description, 
                const char* location, int severity) {
    int index = atomic_fetch_add(&g_bug_detector.bug_index, 1) % 100;
    bug_report_t* report = &g_bug_detector.recent_bugs[index];
    
    report->type = type;
    strncpy(report->description, description, sizeof(report->description) - 1);
    strncpy(report->location, location, sizeof(report->location) - 1);
    clock_gettime(CLOCK_MONOTONIC, &report->timestamp);
    report->severity = severity;
    
    atomic_fetch_add(&g_bug_detector.total_bugs_found, 1);
    atomic_fetch_add(&g_bug_detector.bugs_by_type[type], 1);
    
    printf("[BUG DETECTED] Type: %d, Severity: %d, Location: %s\n"
           "Description: %s\n", type, severity, location, description);
}

// Smart lock wrapper with deadlock detection
typedef struct {
    pthread_mutex_t mutex;
    atomic_int owner_thread_id;
    struct timespec lock_time;
    char* lock_name;
    atomic_int contention_count;
} smart_mutex_t;

int smart_mutex_init(smart_mutex_t* smutex, const char* name) {
    if (!smutex || !name) return -1;
    
    int result = pthread_mutex_init(&smutex->mutex, NULL);
    if (result != 0) return result;
    
    atomic_store(&smutex->owner_thread_id, -1);
    smutex->lock_name = strdup(name);
    atomic_store(&smutex->contention_count, 0);
    
    return 0;
}

int smart_mutex_lock(smart_mutex_t* smutex) {
    if (!smutex) return -1;
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Try lock first to detect contention
    if (pthread_mutex_trylock(&smutex->mutex) != 0) {
        atomic_fetch_add(&smutex->contention_count, 1);
        atomic_fetch_add(&g_bug_detector.lock_contentions, 1);
        
        // Block on lock
        int result = pthread_mutex_lock(&smutex->mutex);
        if (result != 0) return result;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    // Calculate wait time
    long wait_time_ms = (end_time.tv_sec - start_time.tv_sec) * 1000 +
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000;
    
    // Detect potential deadlock
    if (wait_time_ms > 5000) {  // 5 second threshold
        char bug_desc[256];
        snprintf(bug_desc, sizeof(bug_desc), 
                "Long wait time (%ld ms) acquiring lock '%s'", 
                wait_time_ms, smutex->lock_name);
        report_bug(BUG_DEADLOCK, bug_desc, smutex->lock_name, 8);
    }
    
    // Update ownership
    atomic_store(&smutex->owner_thread_id, gettid());
    smutex->lock_time = end_time;
    atomic_fetch_add(&g_bug_detector.lock_acquisitions, 1);
    
    return 0;
}

int smart_mutex_unlock(smart_mutex_t* smutex) {
    if (!smutex) return -1;
    
    // Verify ownership
    if (atomic_load(&smutex->owner_thread_id) != gettid()) {
        report_bug(BUG_RACE_CONDITION, 
                  "Thread unlocking mutex it doesn't own",
                  smutex->lock_name, 9);
    }
    
    atomic_store(&smutex->owner_thread_id, -1);
    return pthread_mutex_unlock(&smutex->mutex);
}
```

### Modern Debugging Toolchain

#### **Static Analysis Tools**
```bash
# Clang Static Analyzer
clang --analyze -Xanalyzer -analyzer-checker=alpha.core.CallAndMessageUnInitRefArg \
      -Xanalyzer -analyzer-checker=alpha.deadcode.UnreachableCode \
      -Xanalyzer -analyzer-checker=alpha.security.taint.TaintPropagation \
      threaded_program.c

# Cppcheck with threading checks
cppcheck --enable=all --std=c11 --platform=unix64 \
         --check-config --suppress=missingIncludeSystem \
         threaded_program.c

# PC-lint Plus (commercial)
pclp64 +rw(_to_semi) -w2 threaded_program.c
```

#### **Dynamic Analysis Tools**
```bash
# ThreadSanitizer (TSan)
gcc -fsanitize=thread -fPIE -pie -g -O1 threaded_program.c -o program_tsan
./program_tsan

# AddressSanitizer with threading
gcc -fsanitize=address -fsanitize=thread -g -O1 threaded_program.c -o program_asan
./program_asan

# Valgrind with Helgrind
valgrind --tool=helgrind --read-var-info=yes ./threaded_program

# Intel Inspector (commercial)
inspxe-cl -collect ti3 -knob scope=extreme -- ./threaded_program
```

#### **Specialized Threading Tools**
```bash
# FastTrack dynamic race detector
java -jar fasttrack.jar threaded_program

# Lockdep (Linux kernel debugging)
echo 1 > /proc/sys/kernel/prove_locking

# Intel Thread Checker (legacy but powerful)
tcheck_cl -analyze threaded_program

# IBM Thread and Memory Debugger
export TMD_ENABLED=1
./threaded_program
```

### Cross-Platform Debugging Considerations

#### **Windows Threading Debugging**
```c
// Windows-specific debugging helpers
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <dbghelp.h>

// Enhanced crash dump generation
void setup_crash_handler() {
    SetUnhandledExceptionFilter(crash_handler);
    
    // Enable heap debugging
    int flags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    flags |= _CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF;
    _CrtSetDbgFlag(flags);
}

LONG WINAPI crash_handler(EXCEPTION_POINTERS* exception_info) {
    // Create detailed crash dump with thread information
    HANDLE dump_file = CreateFile(L"crash_dump.dmp", GENERIC_WRITE, 0, NULL,
                                 CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    
    if (dump_file != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION dump_info;
        dump_info.ThreadId = GetCurrentThreadId();
        dump_info.ExceptionPointers = exception_info;
        dump_info.ClientPointers = FALSE;
        
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(),
                         dump_file, MiniDumpWithFullMemory,
                         &dump_info, NULL, NULL);
        
        CloseHandle(dump_file);
    }
    
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif
```

#### **Linux Threading Debugging**
```c
// Linux-specific debugging features
#ifdef __linux__
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>

// Thread naming for easier debugging
void set_thread_name(const char* name) {
    prctl(PR_SET_NAME, name, 0, 0, 0);
}

// Core dump configuration
void setup_core_dumps() {
    // Enable core dumps
    struct rlimit core_limit;
    core_limit.rlim_cur = RLIM_INFINITY;
    core_limit.rlim_max = RLIM_INFINITY;
    setrlimit(RLIMIT_CORE, &core_limit);
    
    // Set core dump pattern
    system("echo 'core.%e.%p.%t' > /proc/sys/kernel/core_pattern");
}

// Performance monitoring
void setup_perf_monitoring() {
    // Enable performance monitoring for threads
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    
    // Set up perf events
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.size = sizeof(struct perf_event_attr);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    
    // This would require root privileges in practice
    // int fd = perf_event_open(&pe, 0, -1, -1, 0);
}
#endif
```

## Common Threading Bugs: Comprehensive Analysis and Detection

Understanding the landscape of threading bugs is crucial for effective debugging. Each bug type has unique characteristics, symptoms, and detection strategies. This section provides an in-depth analysis of the most common threading issues you'll encounter in production systems.

### Race Conditions: The Most Elusive Bugs

Race conditions are the most common and often the most difficult to debug threading issues. They occur when multiple threads access shared data without proper synchronization, leading to unpredictable results that depend on the timing of thread execution.

#### **Classification of Race Conditions**

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>

// Type 1: Read-Modify-Write Race Condition
static int global_counter = 0;
static pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment_counter_unsafe(void* arg) {
    int iterations = *(int*)arg;
    
    for (int i = 0; i < iterations; i++) {
        // CLASSIC RACE CONDITION: Non-atomic read-modify-write
        // Multiple threads can read the same value simultaneously
        int temp = global_counter;  // Read
        temp = temp + 1;            // Modify  
        global_counter = temp;      // Write
        
        // The window between read and write is where races occur
        // Even single increment (counter++) is not atomic!
    }
    
    return NULL;
}

void* increment_counter_safe(void* arg) {
    int iterations = *(int*)arg;
    
    for (int i = 0; i < iterations; i++) {
        // SOLUTION 1: Mutex protection
        pthread_mutex_lock(&counter_mutex);
        global_counter++;
        pthread_mutex_unlock(&counter_mutex);
    }
    
    return NULL;
}

// Type 2: Check-Then-Act Race Condition
typedef struct {
    int* buffer;
    int size;
    int count;
    pthread_mutex_t mutex;
} shared_buffer_t;

void* check_then_act_unsafe(void* arg) {
    shared_buffer_t* buf = (shared_buffer_t*)arg;
    
    // RACE CONDITION: State can change between check and act
    if (buf->count < buf->size) {  // Check
        // Another thread might add items here!
        buf->buffer[buf->count] = rand() % 100;  // Act
        buf->count++;
    }
    
    return NULL;
}

void* check_then_act_safe(void* arg) {
    shared_buffer_t* buf = (shared_buffer_t*)arg;
    
    // SOLUTION: Atomic check-then-act
    pthread_mutex_lock(&buf->mutex);
    if (buf->count < buf->size) {
        buf->buffer[buf->count] = rand() % 100;
        buf->count++;
    }
    pthread_mutex_unlock(&buf->mutex);
    
    return NULL;
}

// Type 3: Initialization Race Condition
static bool initialized = false;
static pthread_mutex_t init_mutex = PTHREAD_MUTEX_INITIALIZER;
static expensive_resource_t* resource = NULL;

expensive_resource_t* get_resource_unsafe() {
    // RACE CONDITION: Multiple threads might initialize simultaneously
    if (!initialized) {
        resource = create_expensive_resource();  // Expensive operation
        initialized = true;
    }
    return resource;
}

expensive_resource_t* get_resource_safe() {
    // SOLUTION: Double-checked locking (with proper memory barriers)
    if (!initialized) {
        pthread_mutex_lock(&init_mutex);
        if (!initialized) {  // Check again under lock
            resource = create_expensive_resource();
            // Memory barrier ensures resource is fully initialized
            __sync_synchronize();
            initialized = true;
        }
        pthread_mutex_unlock(&init_mutex);
    }
    return resource;
}

// Modern C11 approach with call_once
pthread_once_t once_control = PTHREAD_ONCE_INIT;

void initialize_resource() {
    resource = create_expensive_resource();
}

expensive_resource_t* get_resource_modern() {
    pthread_once(&once_control, initialize_resource);
    return resource;
}

// Type 4: Data Race in Complex Structures
typedef struct {
    char* data;
    size_t length;
    size_t capacity;
    atomic_int ref_count;
} shared_string_t;

void* modify_string_unsafe(void* arg) {
    shared_string_t* str = (shared_string_t*)arg;
    
    // MULTIPLE RACE CONDITIONS:
    // 1. Length and data can be inconsistent
    // 2. Memory reallocation can invalidate pointers
    // 3. Reference counting races
    
    if (str->length > 0) {
        str->data[0] = 'X';  // Race: data might be reallocated
        str->length++;       // Race: length not atomic with data changes
    }
    
    atomic_fetch_add(&str->ref_count, 1);  // This part is safe
    
    return NULL;
}

// Advanced Race Condition: ABA Problem
typedef struct node {
    int data;
    struct node* next;
} node_t;

typedef struct {
    atomic_uintptr_t head;
    atomic_ulong pop_count;
    atomic_ulong aba_detected;
} lock_free_stack_t;

void* stack_pop_aba_vulnerable(void* arg) {
    lock_free_stack_t* stack = (lock_free_stack_t*)arg;
    
    while (true) {
        // Load head pointer
        node_t* head = (node_t*)atomic_load(&stack->head);
        if (head == NULL) break;
        
        // ABA PROBLEM: Between these two operations, another thread might:
        // 1. Pop head (A)
        // 2. Pop next (B)  
        // 3. Push A back
        // Now head looks the same but next might be invalid!
        
        node_t* next = head->next;  // This might be invalid!
        
        // Compare-and-swap might succeed even though state changed
        if (atomic_compare_exchange_weak(&stack->head, 
                                       (uintptr_t*)&head, 
                                       (uintptr_t)next)) {
            atomic_fetch_add(&stack->pop_count, 1);
            free(head);
            break;
        }
    }
    
    return NULL;
}

// Solution: Hazard Pointers or Version Counters
typedef struct {
    atomic_uintptr_t head;
    atomic_ulong version;  // Version counter to detect ABA
} aba_safe_stack_t;

// Comprehensive Race Condition Detection
void demonstrate_race_conditions() {
    printf("=== RACE CONDITION DEMONSTRATION ===\n");
    
    const int num_threads = 8;
    const int iterations = 100000;
    
    pthread_t threads[num_threads];
    int thread_iterations = iterations;
    
    // Reset counter
    global_counter = 0;
    
    printf("Starting race condition test with %d threads, %d iterations each...\n",
           num_threads, iterations);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Create threads with race condition
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, increment_counter_unsafe, &thread_iterations);
    }
    
    // Wait for completion
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    long duration_ms = (end.tv_sec - start.tv_sec) * 1000 + 
                      (end.tv_nsec - start.tv_nsec) / 1000000;
    
    int expected = num_threads * iterations;
    int actual = global_counter;
    int lost_updates = expected - actual;
    double loss_percentage = (double)lost_updates / expected * 100.0;
    
    printf("Results:\n");
    printf("  Expected: %d\n", expected);
    printf("  Actual: %d\n", actual);
    printf("  Lost updates: %d (%.2f%%)\n", lost_updates, loss_percentage);
    printf("  Duration: %ld ms\n", duration_ms);
    printf("  Updates per second: %.0f\n", (double)actual / duration_ms * 1000);
    
    if (lost_updates > 0) {
        printf("  ⚠️  RACE CONDITION DETECTED! Lost %d updates\n", lost_updates);
    } else {
        printf("  ✅ No race condition detected (this time)\n");
    }
}

// Advanced Race Detection with Memory Fences
void demonstrate_memory_ordering_races() {
    printf("\n=== MEMORY ORDERING RACE CONDITIONS ===\n");
    
    // This demonstrates subtle races that only appear on weakly ordered architectures
    static atomic_int flag1 = 0;
    static atomic_int flag2 = 0;
    static atomic_int shared_data = 0;
    
    // Thread 1: Producer
    atomic_store_explicit(&shared_data, 42, memory_order_relaxed);
    atomic_store_explicit(&flag1, 1, memory_order_release);  // Signal data ready
    
    // Thread 2: Consumer 
    if (atomic_load_explicit(&flag1, memory_order_acquire) == 1) {
        int data = atomic_load_explicit(&shared_data, memory_order_relaxed);
        printf("Consumer read: %d\n", data);  // Should always be 42
    }
    
    // Without proper memory ordering, consumer might see flag1=1 but shared_data=0
    // This race is architecture-dependent and very hard to reproduce
}
```

#### **Race Condition Detection Strategies**

```c
// Automated race detection using compiler intrinsics
#define ENABLE_RACE_DETECTION 1

#if ENABLE_RACE_DETECTION
// Simple happens-before tracking
typedef struct {
    atomic_ulong thread_clocks[64];  // Vector clock per thread
    atomic_ulong global_clock;
} race_detector_t;

static race_detector_t g_race_detector = {0};

void rd_thread_start(int thread_id) {
    atomic_store(&g_race_detector.thread_clocks[thread_id], 
                atomic_fetch_add(&g_race_detector.global_clock, 1));
}

void rd_memory_access(void* addr, bool is_write, int thread_id) {
    // Record memory access for race detection
    // In production, this would use more sophisticated algorithms
    // like FastTrack or DJIT+
    
    uintptr_t addr_key = (uintptr_t)addr;
    uintptr_t thread_time = atomic_load(&g_race_detector.thread_clocks[thread_id]);
    
    // Store access information (simplified)
    // Real implementation would check for concurrent accesses
    static atomic_uintptr_t last_access_addr = 0;
    static atomic_uintptr_t last_access_time = 0;
    static atomic_int last_access_thread = -1;
    static atomic_bool last_was_write = false;
    
    uintptr_t prev_addr = atomic_exchange(&last_access_addr, addr_key);
    uintptr_t prev_time = atomic_exchange(&last_access_time, thread_time);
    int prev_thread = atomic_exchange(&last_access_thread, thread_id);
    bool prev_write = atomic_exchange(&last_was_write, is_write);
    
    // Detect race: same address, different threads, overlapping time, at least one write
    if (prev_addr == addr_key && prev_thread != thread_id && 
        (is_write || prev_write) && prev_time >= thread_time - 1000) {
        printf("🔴 RACE DETECTED: addr=%p, threads=%d,%d, %s/%s\n",
               addr, prev_thread, thread_id,
               prev_write ? "write" : "read",
               is_write ? "write" : "read");
    }
}

#define MEMORY_ACCESS(addr, is_write) \
    rd_memory_access(addr, is_write, gettid())
#else
#define MEMORY_ACCESS(addr, is_write) do {} while(0)
#endif

// Example usage with race detection
void* monitored_increment(void* arg) {
    int iterations = *(int*)arg;
    
    for (int i = 0; i < iterations; i++) {
        MEMORY_ACCESS(&global_counter, false);  // Read access
        int temp = global_counter;
        
        MEMORY_ACCESS(&global_counter, true);   // Write access  
        global_counter = temp + 1;
    }
    
    return NULL;
}
```
```

### Deadlocks: The Silent Killers

Deadlocks represent one of the most critical threading bugs because they can bring entire systems to a halt. Unlike race conditions that produce incorrect results, deadlocks cause threads to wait indefinitely, leading to complete system freezes.

#### **The Anatomy of Deadlocks**

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <string.h>

// Classic Two-Lock Deadlock Scenario
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

// Resource representation for better understanding
typedef struct {
    int id;
    char name[32];
    pthread_mutex_t* mutex;
    int owner_thread_id;
    struct timespec lock_time;
} resource_t;

static resource_t resource_a = {1, "Database", &mutex1, -1, {0}};
static resource_t resource_b = {2, "FileSystem", &mutex2, -1, {0}};

void acquire_resource(resource_t* res, int thread_id) {
    printf("Thread %d: Attempting to acquire %s (resource %d)\n", 
           thread_id, res->name, res->id);
    
    clock_gettime(CLOCK_MONOTONIC, &res->lock_time);
    pthread_mutex_lock(res->mutex);
    
    res->owner_thread_id = thread_id;
    printf("Thread %d: ✅ Acquired %s\n", thread_id, res->name);
}

void release_resource(resource_t* res, int thread_id) {
    printf("Thread %d: Releasing %s\n", thread_id, res->name);
    res->owner_thread_id = -1;
    pthread_mutex_unlock(res->mutex);
}

// Thread 1: Acquires A then B (normal order)
void* thread1_function(void* arg) {
    int thread_id = 1;
    printf("Thread %d: Starting execution\n", thread_id);
    
    acquire_resource(&resource_a, thread_id);
    printf("Thread %d: Working with %s...\n", thread_id, resource_a.name);
    
    sleep(2); // Simulate work and increase deadlock probability
    
    printf("Thread %d: Now need %s too...\n", thread_id, resource_b.name);
    acquire_resource(&resource_b, thread_id);  // Will block if thread2 has it
    
    printf("Thread %d: Working with both resources\n", thread_id);
    sleep(1);
    
    release_resource(&resource_b, thread_id);
    release_resource(&resource_a, thread_id);
    
    printf("Thread %d: Completed successfully\n", thread_id);
    return NULL;
}

// Thread 2: Acquires B then A (reverse order - DEADLOCK!)
void* thread2_function(void* arg) {
    int thread_id = 2;
    printf("Thread %d: Starting execution\n", thread_id);
    
    acquire_resource(&resource_b, thread_id);
    printf("Thread %d: Working with %s...\n", thread_id, resource_b.name);
    
    sleep(2); // Simulate work and increase deadlock probability
    
    printf("Thread %d: Now need %s too...\n", thread_id, resource_a.name);
    acquire_resource(&resource_a, thread_id);  // Will block if thread1 has it
    
    printf("Thread %d: Working with both resources\n", thread_id);
    sleep(1);
    
    release_resource(&resource_a, thread_id);
    release_resource(&resource_b, thread_id);
    
    printf("Thread %d: Completed successfully\n", thread_id);
    return NULL;
}

void demonstrate_classic_deadlock() {
    printf("=== CLASSIC DEADLOCK DEMONSTRATION ===\n");
    printf("Two threads acquiring the same locks in different orders\n\n");
    
    pthread_t thread1, thread2;
    
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    pthread_create(&thread1, NULL, thread1_function, NULL);
    pthread_create(&thread2, NULL, thread2_function, NULL);
    
    // Wait with timeout to detect deadlock
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += 10;  // 10 second timeout
    
    void* result1, *result2;
    int join_result1 = pthread_timedjoin_np(thread1, &result1, &timeout);
    int join_result2 = pthread_timedjoin_np(thread2, &result2, &timeout);
    
    if (join_result1 == ETIMEDOUT || join_result2 == ETIMEDOUT) {
        printf("\n🔴 DEADLOCK DETECTED! Threads have been waiting for >10 seconds\n");
        printf("Thread 1 status: %s\n", join_result1 == 0 ? "Completed" : "Deadlocked");
        printf("Thread 2 status: %s\n", join_result2 == 0 ? "Completed" : "Deadlocked");
        
        // Force termination (not recommended in production)
        pthread_cancel(thread1);
        pthread_cancel(thread2);
        
        pthread_join(thread1, NULL);
        pthread_join(thread2, NULL);
    } else {
        printf("\n✅ No deadlock occurred (lucky timing!)\n");
    }
}

// Solution: Lock Ordering
void* thread_ordered_function(void* arg) {
    int thread_id = *(int*)arg;
    
    // SOLUTION: Always acquire locks in the same order (by address or ID)
    resource_t* first_lock = (resource_a.id < resource_b.id) ? &resource_a : &resource_b;
    resource_t* second_lock = (resource_a.id < resource_b.id) ? &resource_b : &resource_a;
    
    printf("Thread %d: Acquiring locks in ordered sequence\n", thread_id);
    
    acquire_resource(first_lock, thread_id);
    acquire_resource(second_lock, thread_id);
    
    printf("Thread %d: Working with both resources safely\n", thread_id);
    sleep(1);
    
    release_resource(second_lock, thread_id);
    release_resource(first_lock, thread_id);
    
    printf("Thread %d: Completed successfully\n", thread_id);
    return NULL;
}

// Advanced Deadlock: Circular Wait with Multiple Threads
#define MAX_RESOURCES 5
#define MAX_THREADS 5

typedef struct {
    pthread_mutex_t mutexes[MAX_RESOURCES];
    char names[MAX_RESOURCES][32];
    int acquisition_order[MAX_THREADS][MAX_RESOURCES];
} circular_deadlock_system_t;

static circular_deadlock_system_t cd_system = {
    .names = {"Resource_0", "Resource_1", "Resource_2", "Resource_3", "Resource_4"},
    .acquisition_order = {
        {0, 1, 2, -1, -1},  // Thread 0: R0 -> R1 -> R2
        {1, 2, 3, -1, -1},  // Thread 1: R1 -> R2 -> R3  
        {2, 3, 4, -1, -1},  // Thread 2: R2 -> R3 -> R4
        {3, 4, 0, -1, -1},  // Thread 3: R3 -> R4 -> R0
        {4, 0, 1, -1, -1}   // Thread 4: R4 -> R0 -> R1 (creates cycle!)
    }
};

void* circular_deadlock_thread(void* arg) {
    int thread_id = *(int*)arg;
    
    printf("Thread %d: Starting circular deadlock scenario\n", thread_id);
    
    for (int i = 0; i < MAX_RESOURCES && cd_system.acquisition_order[thread_id][i] != -1; i++) {
        int resource_id = cd_system.acquisition_order[thread_id][i];
        
        printf("Thread %d: Acquiring %s\n", thread_id, cd_system.names[resource_id]);
        pthread_mutex_lock(&cd_system.mutexes[resource_id]);
        printf("Thread %d: ✅ Got %s\n", thread_id, cd_system.names[resource_id]);
        
        sleep(1);  // Hold lock for a while
    }
    
    printf("Thread %d: Releasing all resources\n", thread_id);
    
    // Release in reverse order
    for (int i = MAX_RESOURCES - 1; i >= 0; i--) {
        if (cd_system.acquisition_order[thread_id][i] != -1) {
            int resource_id = cd_system.acquisition_order[thread_id][i];
            pthread_mutex_unlock(&cd_system.mutexes[resource_id]);
            printf("Thread %d: Released %s\n", thread_id, cd_system.names[resource_id]);
        }
    }
    
    return NULL;
}

// Deadlock Detection Algorithm Implementation
typedef struct deadlock_detector {
    pthread_mutex_t* mutexes[MAX_RESOURCES];
    int owner_threads[MAX_RESOURCES];
    int waiting_threads[MAX_RESOURCES];
    pthread_mutex_t detector_mutex;
    bool detection_enabled;
} deadlock_detector_t;

static deadlock_detector_t g_deadlock_detector = {
    .detector_mutex = PTHREAD_MUTEX_INITIALIZER,
    .detection_enabled = true
};

void dd_register_wait(int thread_id, int resource_id) {
    if (!g_deadlock_detector.detection_enabled) return;
    
    pthread_mutex_lock(&g_deadlock_detector.detector_mutex);
    g_deadlock_detector.waiting_threads[resource_id] = thread_id;
    
    // Check for deadlock cycle
    bool visited[MAX_THREADS] = {false};
    bool rec_stack[MAX_THREADS] = {false};
    
    if (has_cycle(thread_id, visited, rec_stack)) {
        printf("🔴 DEADLOCK CYCLE DETECTED involving thread %d!\n", thread_id);
        print_deadlock_cycle(thread_id);
    }
    
    pthread_mutex_unlock(&g_deadlock_detector.detector_mutex);
}

void dd_register_acquisition(int thread_id, int resource_id) {
    if (!g_deadlock_detector.detection_enabled) return;
    
    pthread_mutex_lock(&g_deadlock_detector.detector_mutex);
    g_deadlock_detector.owner_threads[resource_id] = thread_id;
    g_deadlock_detector.waiting_threads[resource_id] = -1;
    pthread_mutex_unlock(&g_deadlock_detector.detector_mutex);
}

// Banker's Algorithm for Deadlock Avoidance
typedef struct {
    int available[MAX_RESOURCES];
    int max_need[MAX_THREADS][MAX_RESOURCES];
    int allocation[MAX_THREADS][MAX_RESOURCES];
    int need[MAX_THREADS][MAX_RESOURCES];
    pthread_mutex_t banker_mutex;
} banker_system_t;

static banker_system_t g_banker = {
    .banker_mutex = PTHREAD_MUTEX_INITIALIZER
};

bool is_safe_state() {
    int work[MAX_RESOURCES];
    bool finish[MAX_THREADS] = {false};
    
    // Copy available resources
    for (int i = 0; i < MAX_RESOURCES; i++) {
        work[i] = g_banker.available[i];
    }
    
    // Find a safe sequence
    bool found = true;
    while (found) {
        found = false;
        for (int i = 0; i < MAX_THREADS; i++) {
            if (!finish[i]) {
                bool can_complete = true;
                for (int j = 0; j < MAX_RESOURCES; j++) {
                    if (g_banker.need[i][j] > work[j]) {
                        can_complete = false;
                        break;
                    }
                }
                
                if (can_complete) {
                    for (int j = 0; j < MAX_RESOURCES; j++) {
                        work[j] += g_banker.allocation[i][j];
                    }
                    finish[i] = true;
                    found = true;
                }
            }
        }
    }
    
    // Check if all threads can complete
    for (int i = 0; i < MAX_THREADS; i++) {
        if (!finish[i]) return false;
    }
    return true;
}

bool request_resources(int thread_id, int request[]) {
    pthread_mutex_lock(&g_banker.banker_mutex);
    
    // Check if request exceeds need
    for (int i = 0; i < MAX_RESOURCES; i++) {
        if (request[i] > g_banker.need[thread_id][i]) {
            pthread_mutex_unlock(&g_banker.banker_mutex);
            return false;
        }
    }
    
    // Check if request exceeds available
    for (int i = 0; i < MAX_RESOURCES; i++) {
        if (request[i] > g_banker.available[i]) {
            pthread_mutex_unlock(&g_banker.banker_mutex);
            return false;  // Would have to wait
        }
    }
    
    // Tentatively allocate resources
    for (int i = 0; i < MAX_RESOURCES; i++) {
        g_banker.available[i] -= request[i];
        g_banker.allocation[thread_id][i] += request[i];
        g_banker.need[thread_id][i] -= request[i];
    }
    
    // Check if resulting state is safe
    if (is_safe_state()) {
        pthread_mutex_unlock(&g_banker.banker_mutex);
        return true;  // Grant request
    } else {
        // Rollback allocation
        for (int i = 0; i < MAX_RESOURCES; i++) {
            g_banker.available[i] += request[i];
            g_banker.allocation[thread_id][i] -= request[i];
            g_banker.need[thread_id][i] += request[i];
        }
        pthread_mutex_unlock(&g_banker.banker_mutex);
        return false;  // Deny request to avoid deadlock
    }
}

// Timeout-based Deadlock Recovery
typedef struct {
    pthread_mutex_t mutex;
    struct timespec timeout;
    int owner_thread;
    bool has_timeout;
} timeout_mutex_t;

int timeout_mutex_init(timeout_mutex_t* tmutex, int timeout_seconds) {
    int result = pthread_mutex_init(&tmutex->mutex, NULL);
    if (result != 0) return result;
    
    tmutex->timeout.tv_sec = timeout_seconds;
    tmutex->timeout.tv_nsec = 0;
    tmutex->owner_thread = -1;
    tmutex->has_timeout = true;
    
    return 0;
}

int timeout_mutex_lock(timeout_mutex_t* tmutex) {
    if (!tmutex->has_timeout) {
        return pthread_mutex_lock(&tmutex->mutex);
    }
    
    struct timespec abs_timeout;
    clock_gettime(CLOCK_REALTIME, &abs_timeout);
    abs_timeout.tv_sec += tmutex->timeout.tv_sec;
    abs_timeout.tv_nsec += tmutex->timeout.tv_nsec;
    
    if (abs_timeout.tv_nsec >= 1000000000) {
        abs_timeout.tv_sec++;
        abs_timeout.tv_nsec -= 1000000000;
    }
    
    int result = pthread_mutex_timedlock(&tmutex->mutex, &abs_timeout);
    if (result == 0) {
        tmutex->owner_thread = gettid();
    } else if (result == ETIMEDOUT) {
        printf("⚠️ Lock timeout - possible deadlock detected for thread %d\n", gettid());
    }
    
    return result;
}
```

#### **Deadlock Prevention Strategies**

```c
// Strategy 1: Lock Hierarchy (Ordered Locking)
#define LOCK_LEVEL_DATABASE    1
#define LOCK_LEVEL_FILESYSTEM  2
#define LOCK_LEVEL_NETWORK     3
#define LOCK_LEVEL_CACHE       4

typedef struct hierarchical_lock {
    pthread_mutex_t mutex;
    int level;
    char name[32];
} hierarchical_lock_t;

__thread int current_lock_level = 0;

int hierarchical_lock(hierarchical_lock_t* hlock) {
    if (hlock->level <= current_lock_level) {
        printf("🔴 Lock hierarchy violation! Current level: %d, Attempting: %d (%s)\n",
               current_lock_level, hlock->level, hlock->name);
        return EDEADLK;
    }
    
    int result = pthread_mutex_lock(&hlock->mutex);
    if (result == 0) {
        current_lock_level = hlock->level;
        printf("✅ Acquired hierarchical lock %s (level %d)\n", hlock->name, hlock->level);
    }
    
    return result;
}

int hierarchical_unlock(hierarchical_lock_t* hlock) {
    if (hlock->level != current_lock_level) {
        printf("🔴 Lock hierarchy violation during unlock!\n");
        return EPERM;
    }
    
    current_lock_level = 0;  // Reset to allow any lock next time
    printf("Released hierarchical lock %s (level %d)\n", hlock->name, hlock->level);
    return pthread_mutex_unlock(&hlock->mutex);
}

// Strategy 2: Try-Lock with Backoff
int acquire_multiple_locks_safe(pthread_mutex_t* locks[], int count, int max_attempts) {
    int attempts = 0;
    
    while (attempts < max_attempts) {
        bool all_acquired = true;
        int acquired_count = 0;
        
        // Try to acquire all locks
        for (int i = 0; i < count; i++) {
            if (pthread_mutex_trylock(locks[i]) != 0) {
                all_acquired = false;
                break;
            }
            acquired_count++;
        }
        
        if (all_acquired) {
            printf("✅ Successfully acquired all %d locks\n", count);
            return 0;  // Success
        }
        
        // Release any locks we managed to acquire
        for (int i = 0; i < acquired_count; i++) {
            pthread_mutex_unlock(locks[i]);
        }
        
        attempts++;
        
        // Exponential backoff
        int backoff_ms = (1 << attempts) * 10;  // 10ms, 20ms, 40ms, 80ms...
        printf("⚠️ Lock acquisition failed, backing off for %d ms (attempt %d/%d)\n",
               backoff_ms, attempts, max_attempts);
        
        usleep(backoff_ms * 1000);
    }
    
    printf("🔴 Failed to acquire locks after %d attempts\n", max_attempts);
    return EDEADLK;
}

// Strategy 3: Lock-Free Alternatives
typedef struct lock_free_counter {
    atomic_long value;
    atomic_long operations;
} lock_free_counter_t;

void lock_free_increment(lock_free_counter_t* counter) {
    // No locks = no deadlocks!
    atomic_fetch_add(&counter->value, 1);
    atomic_fetch_add(&counter->operations, 1);
}

long lock_free_read(lock_free_counter_t* counter) {
    atomic_fetch_add(&counter->operations, 1);
    return atomic_load(&counter->value);
}

// Example: Replace mutex-protected resource with lock-free alternative
void demonstrate_lock_free_solution() {
    printf("=== LOCK-FREE DEADLOCK PREVENTION ===\n");
    
    lock_free_counter_t counter = {0};
    
    // Multiple threads can increment without any possibility of deadlock
    #pragma omp parallel for
    for (int i = 0; i < 1000000; i++) {
        lock_free_increment(&counter);
    }
    
    printf("Final counter value: %ld\n", lock_free_read(&counter));
    printf("Total operations: %ld\n", atomic_load(&counter.operations));
    printf("✅ No deadlocks possible with lock-free design!\n");
}
```

### Livelocks and Starvation

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

// Livelock example: threads keep backing off and retrying
pthread_mutex_t resource1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t resource2 = PTHREAD_MUTEX_INITIALIZER;

void* polite_thread1(void* arg) {
    for (int i = 0; i < 10; i++) {
        printf("Thread 1: Attempting to acquire resources\n");
        
        if (pthread_mutex_trylock(&resource1) == 0) {
            printf("Thread 1: Got resource1\n");
            
            if (pthread_mutex_trylock(&resource2) == 0) {
                printf("Thread 1: Got both resources, working...\n");
                sleep(1);
                pthread_mutex_unlock(&resource2);
                pthread_mutex_unlock(&resource1);
                break;
            } else {
                printf("Thread 1: Can't get resource2, releasing resource1\n");
                pthread_mutex_unlock(&resource1);
                usleep(rand() % 1000); // Random backoff
            }
        } else {
            printf("Thread 1: Can't get resource1\n");
            usleep(rand() % 1000);
        }
    }
    
    return NULL;
}

void* polite_thread2(void* arg) {
    for (int i = 0; i < 10; i++) {
        printf("Thread 2: Attempting to acquire resources\n");
        
        if (pthread_mutex_trylock(&resource2) == 0) {
            printf("Thread 2: Got resource2\n");
            
            if (pthread_mutex_trylock(&resource1) == 0) {
                printf("Thread 2: Got both resources, working...\n");
                sleep(1);
                pthread_mutex_unlock(&resource1);
                pthread_mutex_unlock(&resource2);
                break;
            } else {
                printf("Thread 2: Can't get resource1, releasing resource2\n");
                pthread_mutex_unlock(&resource2);
                usleep(rand() % 1000); // Random backoff
            }
        } else {
            printf("Thread 2: Can't get resource2\n");
            usleep(rand() % 1000);
        }
    }
    
    return NULL;
}

void demonstrate_livelock() {
    pthread_t thread1, thread2;
    
    printf("Starting livelock demonstration...\n");
    
    pthread_create(&thread1, NULL, polite_thread1, NULL);
    pthread_create(&thread2, NULL, polite_thread2, NULL);
    
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    
    printf("Livelock demonstration completed\n");
}
```

## Advanced Thread-Aware Debugging with GDB

GDB (GNU Debugger) is the most powerful tool for debugging multi-threaded applications on Unix-like systems. However, debugging concurrent programs requires specialized knowledge of GDB's threading features and advanced techniques.

### Comprehensive GDB Threading Commands

```bash
# Essential GDB Threading Commands
info threads              # List all threads with their status
thread <n>                # Switch to thread n
thread apply all <cmd>    # Run command on all threads
thread apply 1-5 <cmd>    # Run command on threads 1-5
thread name <name>        # Set name for current thread

# Advanced thread inspection
info threads verbose      # Detailed thread information
thread apply all bt       # Backtrace for all threads
thread apply all bt full  # Full backtrace with local variables

# Thread-specific breakpoints
break function thread 3   # Break only when thread 3 hits function
break *0x400567 thread 2  # Break at address only for thread 2

# Lock and synchronization debugging
info mutex               # Show mutex information (if available)
info locks              # Display lock information
print pthread_mutex     # Examine mutex state

# Scheduler control
set scheduler-locking on    # Only current thread runs
set scheduler-locking off   # All threads run freely
set scheduler-locking step  # Only current thread runs during stepping

# Non-stop mode (advanced)
set non-stop on            # Enable non-stop debugging
set target-async on        # Enable asynchronous target operations
```

### Production-Ready GDB Debugging Session

Let's debug a complex multi-threaded application with realistic scenarios:

```c
// complex_threaded_app.c - Application with multiple threading issues
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <sys/syscall.h>

#define NUM_WORKER_THREADS 4
#define NUM_TASKS 1000
#define MAX_RETRIES 3

// Global state with various threading issues
typedef struct {
    pthread_mutex_t task_mutex;
    pthread_cond_t task_cond;
    
    int* task_queue;
    int queue_size;
    int queue_head;
    int queue_tail;
    int active_tasks;
    
    pthread_mutex_t stats_mutex;
    long total_processed;
    long total_errors;
    
    bool shutdown_requested;
    pthread_t worker_threads[NUM_WORKER_THREADS];
    pthread_t monitor_thread;
} thread_pool_t;

static thread_pool_t g_pool = {0};

// Helper function to get thread information
void print_thread_info(const char* context) {
    pid_t tid = syscall(SYS_gettid);
    pthread_t pthread_id = pthread_self();
    
    printf("[TID: %d, pthread: %lu] %s\n", tid, pthread_id, context);
}

// Problematic worker function with various bugs
void* worker_thread(void* arg) {
    int worker_id = *(int*)arg;
    char thread_name[32];
    snprintf(thread_name, sizeof(thread_name), "Worker-%d", worker_id);
    
    // Set thread name for easier debugging
    pthread_setname_np(pthread_self(), thread_name);
    
    print_thread_info("Worker thread starting");
    
    while (!g_pool.shutdown_requested) {
        pthread_mutex_lock(&g_pool.task_mutex);
        
        // Wait for tasks with proper condition variable usage
        while (g_pool.active_tasks == 0 && !g_pool.shutdown_requested) {
            print_thread_info("Waiting for tasks");
            pthread_cond_wait(&g_pool.task_cond, &g_pool.task_mutex);
        }
        
        if (g_pool.shutdown_requested) {
            pthread_mutex_unlock(&g_pool.task_mutex);
            break;
        }
        
        // Get task from queue
        int task_id = -1;
        if (g_pool.active_tasks > 0) {
            task_id = g_pool.task_queue[g_pool.queue_head];
            g_pool.queue_head = (g_pool.queue_head + 1) % g_pool.queue_size;
            g_pool.active_tasks--;
            
            printf("Worker %d: Processing task %d\n", worker_id, task_id);
        }
        
        pthread_mutex_unlock(&g_pool.task_mutex);
        
        if (task_id != -1) {
            // Simulate work with potential for various issues
            int work_time = (task_id % 5) + 1;
            sleep(work_time);
            
            // Intentional bug: Race condition in statistics update
            if (task_id % 10 == 0) {
                // Sometimes forget to lock (race condition)
                g_pool.total_processed++;
            } else {
                // Usually lock properly
                pthread_mutex_lock(&g_pool.stats_mutex);
                g_pool.total_processed++;
                pthread_mutex_unlock(&g_pool.stats_mutex);
            }
            
            // Simulate errors
            if (task_id % 50 == 0) {
                pthread_mutex_lock(&g_pool.stats_mutex);
                g_pool.total_errors++;
                pthread_mutex_unlock(&g_pool.stats_mutex);
                printf("Worker %d: Error processing task %d\n", worker_id, task_id);
            }
        }
    }
    
    print_thread_info("Worker thread exiting");
    return NULL;
}

// Monitor thread that can cause deadlocks
void* monitor_thread(void* arg) {
    pthread_setname_np(pthread_self(), "Monitor");
    print_thread_info("Monitor thread starting");
    
    while (!g_pool.shutdown_requested) {
        sleep(5);
        
        // Potential deadlock: acquire locks in different order than workers
        pthread_mutex_lock(&g_pool.stats_mutex);
        printf("Monitor: Stats - Processed: %ld, Errors: %ld\n", 
               g_pool.total_processed, g_pool.total_errors);
        
        // Try to get task mutex while holding stats mutex (potential deadlock)
        if (pthread_mutex_trylock(&g_pool.task_mutex) == 0) {
            printf("Monitor: Active tasks: %d\n", g_pool.active_tasks);
            pthread_mutex_unlock(&g_pool.task_mutex);
        } else {
            printf("Monitor: Task mutex busy\n");
        }
        
        pthread_mutex_unlock(&g_pool.stats_mutex);
    }
    
    print_thread_info("Monitor thread exiting");
    return NULL;
}

int main() {
    printf("Starting complex multi-threaded application...\n");
    printf("PID: %d\n", getpid());
    
    // Initialize thread pool
    pthread_mutex_init(&g_pool.task_mutex, NULL);
    pthread_mutex_init(&g_pool.stats_mutex, NULL);
    pthread_cond_init(&g_pool.task_cond, NULL);
    
    g_pool.queue_size = NUM_TASKS;
    g_pool.task_queue = malloc(g_pool.queue_size * sizeof(int));
    g_pool.queue_head = 0;
    g_pool.queue_tail = 0;
    g_pool.active_tasks = 0;
    
    // Add tasks to queue
    pthread_mutex_lock(&g_pool.task_mutex);
    for (int i = 0; i < NUM_TASKS; i++) {
        g_pool.task_queue[g_pool.queue_tail] = i;
        g_pool.queue_tail = (g_pool.queue_tail + 1) % g_pool.queue_size;
        g_pool.active_tasks++;
    }
    pthread_mutex_unlock(&g_pool.task_mutex);
    
    printf("Added %d tasks to queue\n", NUM_TASKS);
    
    // Create worker threads
    int worker_ids[NUM_WORKER_THREADS];
    for (int i = 0; i < NUM_WORKER_THREADS; i++) {
        worker_ids[i] = i;
        if (pthread_create(&g_pool.worker_threads[i], NULL, 
                          worker_thread, &worker_ids[i]) != 0) {
            perror("Failed to create worker thread");
            exit(1);
        }
    }
    
    // Create monitor thread
    if (pthread_create(&g_pool.monitor_thread, NULL, monitor_thread, NULL) != 0) {
        perror("Failed to create monitor thread");
        exit(1);
    }
    
    // Wake up workers
    pthread_cond_broadcast(&g_pool.task_cond);
    
    // Wait for completion
    for (int i = 0; i < NUM_WORKER_THREADS; i++) {
        pthread_join(g_pool.worker_threads[i], NULL);
    }
    
    pthread_join(g_pool.monitor_thread, NULL);
    
    printf("Final stats - Processed: %ld, Errors: %ld\n", 
           g_pool.total_processed, g_pool.total_errors);
    
    // Cleanup
    free(g_pool.task_queue);
    pthread_mutex_destroy(&g_pool.task_mutex);
    pthread_mutex_destroy(&g_pool.stats_mutex);
    pthread_cond_destroy(&g_pool.task_cond);
    
    return 0;
}
```

### Advanced GDB Debugging Session

Here's a comprehensive debugging session for the above program:

```bash
# Compile with debugging symbols and threading support
gcc -g -O0 -pthread -o complex_app complex_threaded_app.c

# Start GDB session
gdb ./complex_app

# Set up GDB for multi-threaded debugging
(gdb) set pagination off
(gdb) set print thread-events on
(gdb) set scheduler-locking off
(gdb) set non-stop off

# Set breakpoints for common threading issues
(gdb) break worker_thread
(gdb) break monitor_thread
(gdb) break pthread_mutex_lock
(gdb) break pthread_cond_wait

# Advanced breakpoint for race condition detection
(gdb) break complex_threaded_app.c:85
(gdb) condition 2 task_id % 10 == 0
(gdb) commands 2
    print "Race condition opportunity detected!"
    print task_id
    print worker_id
    continue
end

# Run the program
(gdb) run

# When stopped, examine thread state
(gdb) info threads
(gdb) thread apply all bt

# Switch to specific thread and examine state
(gdb) thread 2
(gdb) print worker_id
(gdb) print task_id
(gdb) print g_pool

# Examine mutex state (if compiled with pthread debug info)
(gdb) print g_pool.task_mutex
(gdb) print g_pool.stats_mutex

# Look for deadlock patterns
(gdb) thread apply all print $pc
(gdb) thread apply all where

# Set watchpoint for race condition detection
(gdb) watch g_pool.total_processed
(gdb) commands
    print "total_processed modified!"
    print pthread_self()
    bt 3
    continue
end

# Advanced thread analysis
(gdb) define print_all_thread_stacks
  set $i = 1
  while $i <= $_thread
    printf "Thread %d:\n", $i
    thread $i
    bt 3
    printf "\n"
    set $i = $i + 1
  end
end

(gdb) print_all_thread_stacks

# Detect potential deadlocks
(gdb) define detect_deadlock
  printf "Checking for deadlock patterns...\n"
  thread apply all print $pc
  thread apply all info registers rip
  printf "Threads waiting on mutexes:\n"
  thread apply all x/i $pc
end

(gdb) detect_deadlock

# Memory debugging
(gdb) x/10x g_pool.task_queue
(gdb) print sizeof(g_pool)
(gdb) print &g_pool
```

### GDB Scripting for Automated Debugging

```bash
# Create debugging script file: debug_threads.gdb
echo '
# Comprehensive threading debug script
set pagination off
set print thread-events on
set scheduler-locking off

# Custom commands for thread analysis
define thread_summary
  printf "=== THREAD SUMMARY ===\n"
  info threads
  printf "\n=== THREAD BACKTRACES ===\n"
  thread apply all bt 5
  printf "\n=== MUTEX STATES ===\n"
  print g_pool.task_mutex
  print g_pool.stats_mutex
end

define check_race_conditions
  printf "=== RACE CONDITION CHECK ===\n"
  print g_pool.total_processed
  print g_pool.total_errors
  print g_pool.active_tasks
  thread apply all print $pc
end

define deadlock_analysis
  printf "=== DEADLOCK ANALYSIS ===\n"
  printf "Threads and their current locations:\n"
  thread apply all printf "Thread %d: %s\n", $_thread, $pc
  thread apply all x/i $pc
end

# Set up breakpoints and watchpoints
break worker_thread
break monitor_thread

# Race condition detection
watch g_pool.total_processed
commands
  printf "Race condition detected: total_processed changed!\n"
  printf "Current thread: %d\n", $_thread
  bt 3
  continue
end

# Run analysis
run
thread_summary
check_race_conditions
deadlock_analysis
' > debug_threads.gdb

# Use the script
gdb -x debug_threads.gdb ./complex_app
```

### Remote Debugging Multi-Threaded Applications

```bash
# Server side (target machine)
gdbserver :2345 ./complex_app

# Client side (development machine)
gdb ./complex_app
(gdb) target remote target-machine:2345
(gdb) set sysroot /path/to/target/filesystem
(gdb) info threads
(gdb) thread apply all bt

# For embedded systems or containers
docker run -it --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  your-container gdbserver :2345 ./complex_app

# Connect from host
gdb -ex "target remote localhost:2345" ./complex_app
```

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t debug_mutex = PTHREAD_MUTEX_INITIALIZER;
int shared_value = 0;

void* debug_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < 5; i++) {
        pthread_mutex_lock(&debug_mutex);
        
        int old_value = shared_value;
        sleep(1); // Deliberate delay for debugging
        shared_value = old_value + 1;
        
        printf("Thread %d: Updated shared_value from %d to %d\n", 
               thread_id, old_value, shared_value);
        
        pthread_mutex_unlock(&debug_mutex);
        sleep(1);
    }
    
    return NULL;
}

int main() {
    const int num_threads = 3;
    pthread_t threads[num_threads];
    int thread_ids[num_threads];
    
    printf("Starting debug example with %d threads\n", num_threads);
    
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i + 1;
        pthread_create(&threads[i], NULL, debug_worker, &thread_ids[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Final shared_value: %d\n", shared_value);
    return 0;
}
```

## Advanced Thread Sanitizers and Analysis Tools

Thread sanitizers are automated tools that can detect threading bugs at runtime with minimal performance overhead. They're essential for catching issues that manual testing might miss.

### ThreadSanitizer (TSan): The Race Condition Hunter

ThreadSanitizer is Google's dynamic race detector that can find data races, use-after-free bugs, and other memory safety issues.

```bash
# Compile with ThreadSanitizer
gcc -fsanitize=thread -fPIE -pie -g -O1 -o program program.c -lpthread

# Additional useful flags
gcc -fsanitize=thread \
    -fsanitize-recover=thread \
    -fno-omit-frame-pointer \
    -g -O1 -o program program.c -lpthread

# Environment variables for TSan
export TSAN_OPTIONS="halt_on_error=0:history_size=7:detect_thread_leaks=true"
export TSAN_OPTIONS="$TSAN_OPTIONS:print_stacktrace=1:print_stats=1"

# Run the program
./program 2>&1 | tee tsan_output.log
```

### Comprehensive TSan Example with Multiple Bug Types

```c
// tsan_test_comprehensive.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>
#include <string.h>

// Global variables for demonstrating various race conditions
int global_counter = 0;                    // Classic data race
int* dynamic_array = NULL;                 // Heap race condition
static int static_var = 0;                 // Static variable race
__thread int thread_local_var = 0;         // Thread-local (should be safe)

// Mutex for some protections
pthread_mutex_t protection_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t allocation_mutex = PTHREAD_MUTEX_INITIALIZER;

// Atomic variable for comparison
atomic_int atomic_counter = 0;

// Structure with multiple fields for complex races
typedef struct {
    int field1;
    int field2;
    char* string_field;
    atomic_int atomic_field;
} shared_struct_t;

static shared_struct_t shared_data = {0};

// Function with classic data race
void* classic_race_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < 10000; i++) {
        // RACE CONDITION: Multiple threads incrementing without synchronization
        global_counter++;  // TSan will detect this!
        
        // Safe atomic operation for comparison
        atomic_fetch_add(&atomic_counter, 1);
        
        // Thread-local access (safe)
        thread_local_var++;
    }
    
    printf("Thread %d: global_counter contribution completed\n", thread_id);
    return NULL;
}

// Function with heap allocation races
void* heap_race_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < 100; i++) {
        // RACE CONDITION: Multiple threads accessing dynamic_array
        if (dynamic_array != NULL) {
            dynamic_array[thread_id % 1000] = thread_id;  // TSan will detect this!
        }
        
        // Sometimes reallocate (even more dangerous!)
        if (i % 20 == 0) {
            pthread_mutex_lock(&allocation_mutex);
            if (dynamic_array != NULL) {
                free(dynamic_array);
                dynamic_array = malloc(1000 * sizeof(int));
                memset(dynamic_array, 0, 1000 * sizeof(int));
            }
            pthread_mutex_unlock(&allocation_mutex);
        }
        
        usleep(1000); // Small delay to increase race probability
    }
    
    return NULL;
}

// Function with structure field races
void* struct_race_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < 1000; i++) {
        // RACE CONDITION: Multiple fields accessed without coordination
        shared_data.field1 = thread_id;      // Race on field1
        shared_data.field2 = thread_id * 2;  // Race on field2
        
        // String field race (very dangerous!)
        if (shared_data.string_field != NULL) {
            sprintf(shared_data.string_field, "Thread_%d_Iter_%d", thread_id, i);
        }
        
        // Atomic field access (safe)
        atomic_store(&shared_data.atomic_field, thread_id);
        
        usleep(100);
    }
    
    return NULL;
}

// Function demonstrating use-after-free
void* use_after_free_worker(void* arg) {
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < 50; i++) {
        // Allocate memory
        int* local_ptr = malloc(sizeof(int) * 100);
        *local_ptr = thread_id;
        
        // Use the memory
        for (int j = 0; j < 100; j++) {
            local_ptr[j] = thread_id + j;
        }
        
        // Free memory
        free(local_ptr);
        
        // POTENTIAL USE-AFTER-FREE: Access freed memory
        if (i % 10 == 0) {
            printf("Thread %d: Freed memory content: %d\n", thread_id, *local_ptr);
        }
        
        usleep(1000);
    }
    
    return NULL;
}

// Function with lock order violations (potential deadlock)
pthread_mutex_t mutex_a = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_b = PTHREAD_MUTEX_INITIALIZER;

void* lock_order_worker1(void* arg) {
    for (int i = 0; i < 100; i++) {
        // Order: A then B
        pthread_mutex_lock(&mutex_a);
        usleep(1000);
        pthread_mutex_lock(&mutex_b);
        
        // Critical section
        global_counter += 1;
        
        pthread_mutex_unlock(&mutex_b);
        pthread_mutex_unlock(&mutex_a);
        
        usleep(1000);
    }
    return NULL;
}

void* lock_order_worker2(void* arg) {
    for (int i = 0; i < 100; i++) {
        // Order: B then A (DEADLOCK POTENTIAL!)
        pthread_mutex_lock(&mutex_b);
        usleep(1000);
        pthread_mutex_lock(&mutex_a);
        
        // Critical section  
        global_counter += 1;
        
        pthread_mutex_unlock(&mutex_a);
        pthread_mutex_unlock(&mutex_b);
        
        usleep(1000);
    }
    return NULL;
}

int main() {
    printf("ThreadSanitizer Comprehensive Test\n");
    printf("==================================\n");
    
    // Initialize dynamic array
    dynamic_array = malloc(1000 * sizeof(int));
    memset(dynamic_array, 0, 1000 * sizeof(int));
    
    // Initialize shared structure
    shared_data.string_field = malloc(100);
    atomic_init(&shared_data.atomic_field, 0);
    
    const int num_threads = 4;
    pthread_t threads[num_threads];
    int thread_ids[num_threads];
    
    // Test 1: Classic race conditions
    printf("Test 1: Starting classic race condition test...\n");
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, classic_race_worker, &thread_ids[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Classic race test completed. Global counter: %d (expected: %d)\n", 
           global_counter, num_threads * 10000);
    printf("Atomic counter: %d (should be correct: %d)\n", 
           atomic_load(&atomic_counter), num_threads * 10000);
    
    // Test 2: Heap allocation races
    printf("\nTest 2: Starting heap allocation race test...\n");
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, heap_race_worker, &thread_ids[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Test 3: Structure field races
    printf("\nTest 3: Starting structure field race test...\n");
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, struct_race_worker, &thread_ids[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Structure race test completed. Final values: field1=%d, field2=%d\n",
           shared_data.field1, shared_data.field2);
    
    // Test 4: Use-after-free (commented out as it's very dangerous)
    /*
    printf("\nTest 4: Starting use-after-free test...\n");
    for (int i = 0; i < 2; i++) {
        pthread_create(&threads[i], NULL, use_after_free_worker, &thread_ids[i]);
    }
    
    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
    }
    */
    
    // Test 5: Lock ordering issues
    printf("\nTest 5: Starting lock ordering test...\n");
    pthread_t lock_thread1, lock_thread2;
    pthread_create(&lock_thread1, NULL, lock_order_worker1, NULL);
    pthread_create(&lock_thread2, NULL, lock_order_worker2, NULL);
    
    pthread_join(lock_thread1, NULL);
    pthread_join(lock_thread2, NULL);
    
    // Cleanup
    free(dynamic_array);
    free(shared_data.string_field);
    
    printf("\nAll tests completed!\n");
    printf("Check TSan output above for detected race conditions.\n");
    
    return 0;
}
```

### Understanding TSan Output

```bash
# Example TSan race condition report
==================
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 4 at 0x7fff12345678 by thread T2:
    #0 classic_race_worker tsan_test_comprehensive.c:45:23
    #1 <null> <null>

  Previous write of size 4 at 0x7fff12345678 by thread T1:
    #0 classic_race_worker tsan_test_comprehensive.c:45:23
    #1 <null> <null>

  Location is global 'global_counter' of size 4 at 0x7fff12345678 (tsan_test_comprehensive.c:12:5)

  Thread T2 (tid=12347, running) created by main thread at:
    #0 pthread_create <null>
    #1 main tsan_test_comprehensive.c:195:9

  Thread T1 (tid=12346, running) created by main thread at:
    #0 pthread_create <null>
    #1 main tsan_test_comprehensive.c:195:9

SUMMARY: ThreadSanitizer: data race tsan_test_comprehensive.c:45:23 in classic_race_worker
==================
```

### Advanced TSan Configuration

```bash
# Comprehensive TSan options
export TSAN_OPTIONS="halt_on_error=0"              # Continue after finding errors
export TSAN_OPTIONS="$TSAN_OPTIONS:history_size=7" # Larger history for better reports
export TSAN_OPTIONS="$TSAN_OPTIONS:detect_thread_leaks=true"    # Find thread leaks
export TSAN_OPTIONS="$TSAN_OPTIONS:report_thread_leaks=true"    # Report thread leaks
export TSAN_OPTIONS="$TSAN_OPTIONS:detect_signal_unsafe=true"   # Detect signal handler races
export TSAN_OPTIONS="$TSAN_OPTIONS:detect_mutex_leaks=true"     # Find mutex leaks
export TSAN_OPTIONS="$TSAN_OPTIONS:print_module_map=1"          # Print loaded modules
export TSAN_OPTIONS="$TSAN_OPTIONS:exitcode=66"                 # Custom exit code for races

# Suppression file for known false positives
cat > tsan_suppressions.txt << EOF
# Suppress known false positives
race:third_party_library_function
race:legacy_code_with_known_race

# Suppress by file
race:*/third_party/*

# Suppress specific patterns
race:*atomic*
EOF

export TSAN_OPTIONS="$TSAN_OPTIONS:suppressions=tsan_suppressions.txt"
```

### Helgrind: Valgrind's Thread Analysis Tool

```bash
# Using Helgrind for race detection and lock analysis
valgrind --tool=helgrind \
         --read-var-info=yes \
         --track-lockorders=yes \
         --check-stack-refs=yes \
         --free-is-write=yes \
         ./program

# Example Helgrind output
==12345== Helgrind, a thread error detector
==12345== Copyright (C) 2007-2017, and GNU GPL'd, by OpenWorks LLP et al.
==12345== Using Valgrind-3.13.0 and LibVEX; rerun with -h for copyright info
==12345== Command: ./program
==12345== 
==12345== Possible data race during read of size 4 at 0x30A014 by thread #2
==12345== Locks held: none
==12345==    at 0x400567: classic_race_worker (program.c:45)
==12345==    by 0x4E46FA2: start_thread (pthread_create.c:486)
==12345== 
==12345== This conflicts with a previous write of size 4 by thread #1
==12345== Locks held: none
==12345==    at 0x400567: classic_race_worker (program.c:45)
==12345==    by 0x4E46FA2: start_thread (pthread_create.c:486)
==12345==  Address 0x30A014 is 0 bytes inside data symbol "global_counter"
```

### Intel Inspector: Commercial Grade Analysis

```bash
# Intel Inspector (commercial tool with advanced features)
inspxe-cl -collect ti3 -knob scope=extreme -result-dir inspection_results -- ./program

# Generate report
inspxe-cl -report summary -result-dir inspection_results
inspxe-cl -report problems -result-dir inspection_results

# GUI mode
inspxe-gui inspection_results
```

### Custom Race Detection Framework

```c
// custom_race_detector.c - Simple race detection framework
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#define MAX_MEMORY_ACCESSES 10000
#define MAX_THREADS 64

typedef struct {
    void* address;
    int thread_id;
    bool is_write;
    uint64_t timestamp;
    const char* file;
    int line;
} memory_access_t;

typedef struct {
    memory_access_t accesses[MAX_MEMORY_ACCESSES];
    atomic_int access_count;
    atomic_uint64_t global_clock;
    uint64_t thread_clocks[MAX_THREADS];
    pthread_mutex_t detector_mutex;
} race_detector_t;

static race_detector_t g_race_detector = {
    .access_count = 0,
    .global_clock = 0,
    .detector_mutex = PTHREAD_MUTEX_INITIALIZER
};

// Initialize thread in race detector
void rd_thread_init(int thread_id) {
    pthread_mutex_lock(&g_race_detector.detector_mutex);
    g_race_detector.thread_clocks[thread_id] = 
        atomic_fetch_add(&g_race_detector.global_clock, 1);
    pthread_mutex_unlock(&g_race_detector.detector_mutex);
}

// Record memory access
void rd_record_access(void* addr, bool is_write, int thread_id, 
                     const char* file, int line) {
    uint64_t timestamp = atomic_fetch_add(&g_race_detector.global_clock, 1);
    int index = atomic_fetch_add(&g_race_detector.access_count, 1);
    
    if (index < MAX_MEMORY_ACCESSES) {
        memory_access_t* access = &g_race_detector.accesses[index];
        access->address = addr;
        access->thread_id = thread_id;
        access->is_write = is_write;
        access->timestamp = timestamp;
        access->file = file;
        access->line = line;
        
        // Check for races with previous accesses
        rd_check_race(access, index);
    }
}

// Simple race detection algorithm
void rd_check_race(memory_access_t* current, int current_index) {
    for (int i = current_index - 1; i >= 0 && 
         i >= current_index - 1000; i--) { // Check last 1000 accesses
        
        memory_access_t* prev = &g_race_detector.accesses[i];
        
        // Same address, different threads, at least one write
        if (prev->address == current->address && 
            prev->thread_id != current->thread_id &&
            (prev->is_write || current->is_write)) {
            
            // Check if accesses are concurrent (simplified)
            uint64_t time_diff = current->timestamp - prev->timestamp;
            if (time_diff < 1000) { // Within 1000 time units
                printf("🔴 RACE DETECTED!\n");
                printf("  Address: %p\n", current->address);
                printf("  Thread %d (%s) at %s:%d\n", 
                       prev->thread_id, prev->is_write ? "write" : "read",
                       prev->file, prev->line);
                printf("  Thread %d (%s) at %s:%d\n", 
                       current->thread_id, current->is_write ? "write" : "read",
                       current->file, current->line);
                printf("  Time difference: %lu\n", time_diff);
            }
        }
    }
}

// Macros for easy instrumentation
#define RD_READ(addr, tid) rd_record_access(addr, false, tid, __FILE__, __LINE__)
#define RD_WRITE(addr, tid) rd_record_access(addr, true, tid, __FILE__, __LINE__)

// Example usage
int global_var = 0;

void* instrumented_thread(void* arg) {
    int tid = *(int*)arg;
    rd_thread_init(tid);
    
    for (int i = 0; i < 1000; i++) {
        RD_READ(&global_var, tid);
        int temp = global_var;
        
        RD_WRITE(&global_var, tid);
        global_var = temp + 1;
    }
    
    return NULL;
}
```
    
    pthread_join(writer, NULL);
    pthread_join(reader, NULL);
    
    printf("Final value: %d\n", racy_variable);
    return 0;
}
```

### Helgrind (Valgrind's Thread Checker)

```bash
# Install valgrind
sudo apt-get install valgrind

# Run with Helgrind
valgrind --tool=helgrind ./program

# Run with detailed output
valgrind --tool=helgrind --read-var-info=yes ./program
```

## Professional Custom Debugging Tools

Building custom debugging tools tailored to your specific threading problems can provide insights that generic tools might miss. Here are production-ready examples of custom debugging frameworks.

### Advanced Thread State Monitor

```c
// advanced_thread_monitor.c - Production-grade thread monitoring system
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <signal.h>
#include <errno.h>

#define MAX_THREADS 256
#define MAX_THREAD_NAME 64
#define MAX_CALL_STACK 16
#define MONITOR_INTERVAL_MS 100

typedef enum {
    THREAD_STATE_CREATED,
    THREAD_STATE_RUNNING,
    THREAD_STATE_WAITING,
    THREAD_STATE_BLOCKED,
    THREAD_STATE_TERMINATED,
    THREAD_STATE_UNKNOWN
} thread_state_t;

typedef struct {
    void* address;
    uint64_t timestamp;
    const char* function;
    const char* file;
    int line;
} call_frame_t;

typedef struct {
    pthread_t pthread_id;
    pid_t tid;
    char name[MAX_THREAD_NAME];
    thread_state_t state;
    
    // Timing information
    struct timespec created_time;
    struct timespec last_activity;
    uint64_t cpu_time_ns;
    uint64_t wall_time_ns;
    
    // Resource tracking
    int locks_held;
    int locks_waiting;
    void* current_lock_address;
    
    // Call stack tracking
    call_frame_t call_stack[MAX_CALL_STACK];
    int stack_depth;
    
    // Statistics
    atomic_ulong context_switches;
    atomic_ulong memory_allocations;
    atomic_ulong system_calls;
    
    // Health indicators
    bool is_responsive;
    uint64_t max_response_time_ms;
    uint64_t total_blocked_time_ms;
    
    // Custom metrics
    void* custom_data;
    void (*custom_metrics_callback)(void* data);
} thread_info_t;

typedef struct {
    thread_info_t threads[MAX_THREADS];
    atomic_int thread_count;
    atomic_bool monitoring_active;
    
    pthread_t monitor_thread;
    pthread_mutex_t registry_mutex;
    
    // Global statistics
    atomic_ulong total_context_switches;
    atomic_ulong total_deadlocks_detected;
    atomic_ulong total_race_conditions;
    
    // Configuration
    bool enable_stack_trace;
    bool enable_performance_monitoring;
    bool enable_deadlock_detection;
    int monitoring_interval_ms;
    
    // Output configuration
    FILE* log_file;
    bool enable_console_output;
    bool enable_json_output;
} thread_monitor_t;

static thread_monitor_t g_monitor = {
    .thread_count = 0,
    .monitoring_active = false,
    .registry_mutex = PTHREAD_MUTEX_INITIALIZER,
    .enable_stack_trace = true,
    .enable_performance_monitoring = true,
    .enable_deadlock_detection = true,
    .monitoring_interval_ms = MONITOR_INTERVAL_MS,
    .log_file = NULL,
    .enable_console_output = true,
    .enable_json_output = false
};

// Initialize thread monitoring system
int tm_init(const char* log_filename) {
    if (log_filename) {
        g_monitor.log_file = fopen(log_filename, "w");
        if (!g_monitor.log_file) {
            perror("Failed to open log file");
            return -1;
        }
    }
    
    atomic_store(&g_monitor.monitoring_active, true);
    
    // Start monitor thread
    if (pthread_create(&g_monitor.monitor_thread, NULL, monitor_thread_main, NULL) != 0) {
        perror("Failed to create monitor thread");
        return -1;
    }
    
    printf("Thread monitor initialized\n");
    return 0;
}

// Register a thread for monitoring
int tm_register_thread(const char* name) {
    pthread_mutex_lock(&g_monitor.registry_mutex);
    
    int index = atomic_fetch_add(&g_monitor.thread_count, 1);
    if (index >= MAX_THREADS) {
        pthread_mutex_unlock(&g_monitor.registry_mutex);
        return -1;
    }
    
    thread_info_t* info = &g_monitor.threads[index];
    
    // Initialize thread info
    info->pthread_id = pthread_self();
    info->tid = syscall(SYS_gettid);
    strncpy(info->name, name ? name : "Unknown", MAX_THREAD_NAME - 1);
    info->state = THREAD_STATE_CREATED;
    
    clock_gettime(CLOCK_MONOTONIC, &info->created_time);
    info->last_activity = info->created_time;
    
    info->locks_held = 0;
    info->locks_waiting = 0;
    info->current_lock_address = NULL;
    info->stack_depth = 0;
    
    atomic_store(&info->context_switches, 0);
    atomic_store(&info->memory_allocations, 0);
    atomic_store(&info->system_calls, 0);
    
    info->is_responsive = true;
    info->max_response_time_ms = 0;
    info->total_blocked_time_ms = 0;
    
    pthread_mutex_unlock(&g_monitor.registry_mutex);
    
    printf("Registered thread: %s (TID: %d)\n", info->name, info->tid);
    return index;
}

// Update thread state
void tm_set_state(thread_state_t state) {
    pthread_t current = pthread_self();
    
    pthread_mutex_lock(&g_monitor.registry_mutex);
    
    for (int i = 0; i < atomic_load(&g_monitor.thread_count); i++) {
        if (pthread_equal(g_monitor.threads[i].pthread_id, current)) {
            g_monitor.threads[i].state = state;
            clock_gettime(CLOCK_MONOTONIC, &g_monitor.threads[i].last_activity);
            break;
        }
    }
    
    pthread_mutex_unlock(&g_monitor.registry_mutex);
}

// Record lock acquisition
void tm_lock_acquired(void* lock_addr, const char* function, const char* file, int line) {
    pthread_t current = pthread_self();
    
    pthread_mutex_lock(&g_monitor.registry_mutex);
    
    for (int i = 0; i < atomic_load(&g_monitor.thread_count); i++) {
        if (pthread_equal(g_monitor.threads[i].pthread_id, current)) {
            g_monitor.threads[i].locks_held++;
            g_monitor.threads[i].current_lock_address = lock_addr;
            
            // Add to call stack if enabled
            if (g_monitor.enable_stack_trace && g_monitor.threads[i].stack_depth < MAX_CALL_STACK) {
                call_frame_t* frame = &g_monitor.threads[i].call_stack[g_monitor.threads[i].stack_depth++];
                frame->address = lock_addr;
                frame->function = function;
                frame->file = file;
                frame->line = line;
                clock_gettime(CLOCK_MONOTONIC, (struct timespec*)&frame->timestamp);
            }
            
            break;
        }
    }
    
    pthread_mutex_unlock(&g_monitor.registry_mutex);
}

// Record lock release
void tm_lock_released(void* lock_addr) {
    pthread_t current = pthread_self();
    
    pthread_mutex_lock(&g_monitor.registry_mutex);
    
    for (int i = 0; i < atomic_load(&g_monitor.thread_count); i++) {
        if (pthread_equal(g_monitor.threads[i].pthread_id, current)) {
            if (g_monitor.threads[i].locks_held > 0) {
                g_monitor.threads[i].locks_held--;
            }
            
            if (g_monitor.threads[i].current_lock_address == lock_addr) {
                g_monitor.threads[i].current_lock_address = NULL;
            }
            
            // Remove from call stack
            if (g_monitor.enable_stack_trace) {
                for (int j = g_monitor.threads[i].stack_depth - 1; j >= 0; j--) {
                    if (g_monitor.threads[i].call_stack[j].address == lock_addr) {
                        // Shift stack down
                        for (int k = j; k < g_monitor.threads[i].stack_depth - 1; k++) {
                            g_monitor.threads[i].call_stack[k] = g_monitor.threads[i].call_stack[k + 1];
                        }
                        g_monitor.threads[i].stack_depth--;
                        break;
                    }
                }
            }
            
            break;
        }
    }
    
    pthread_mutex_unlock(&g_monitor.registry_mutex);
}

// Main monitoring thread
void* monitor_thread_main(void* arg) {
    printf("Thread monitor started\n");
    
    while (atomic_load(&g_monitor.monitoring_active)) {
        // Check thread health
        tm_check_thread_health();
        
        // Detect deadlocks
        if (g_monitor.enable_deadlock_detection) {
            tm_detect_deadlocks();
        }
        
        // Generate reports
        if (g_monitor.enable_console_output) {
            tm_print_status();
        }
        
        if (g_monitor.log_file) {
            tm_log_status();
        }
        
        // Sleep for monitoring interval
        usleep(g_monitor.monitoring_interval_ms * 1000);
    }
    
    printf("Thread monitor stopped\n");
    return NULL;
}

// Check thread health and responsiveness
void tm_check_thread_health() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    
    pthread_mutex_lock(&g_monitor.registry_mutex);
    
    for (int i = 0; i < atomic_load(&g_monitor.thread_count); i++) {
        thread_info_t* thread = &g_monitor.threads[i];
        
        // Calculate time since last activity
        uint64_t inactive_time_ms = 
            (now.tv_sec - thread->last_activity.tv_sec) * 1000 +
            (now.tv_nsec - thread->last_activity.tv_nsec) / 1000000;
        
        // Check if thread is unresponsive
        if (inactive_time_ms > 5000) { // 5 seconds
            if (thread->is_responsive) {
                printf("⚠️ Thread %s appears unresponsive (inactive for %lu ms)\n",
                       thread->name, inactive_time_ms);
                thread->is_responsive = false;
            }
        } else {
            thread->is_responsive = true;
        }
        
        // Update max response time
        if (inactive_time_ms > thread->max_response_time_ms) {
            thread->max_response_time_ms = inactive_time_ms;
        }
        
        // Check for threads holding locks too long
        if (thread->locks_held > 0 && inactive_time_ms > 1000) {
            printf("⚠️ Thread %s holding %d locks for %lu ms\n",
                   thread->name, thread->locks_held, inactive_time_ms);
        }
    }
    
    pthread_mutex_unlock(&g_monitor.registry_mutex);
}

// Simple deadlock detection
void tm_detect_deadlocks() {
    pthread_mutex_lock(&g_monitor.registry_mutex);
    
    // Check for circular wait conditions
    for (int i = 0; i < atomic_load(&g_monitor.thread_count); i++) {
        thread_info_t* thread1 = &g_monitor.threads[i];
        
        if (thread1->state == THREAD_STATE_BLOCKED && thread1->locks_waiting > 0) {
            // Look for another thread that might be blocking this one
            for (int j = 0; j < atomic_load(&g_monitor.thread_count); j++) {
                if (i == j) continue;
                
                thread_info_t* thread2 = &g_monitor.threads[j];
                
                // Simple heuristic: both threads blocked and holding locks
                if (thread2->state == THREAD_STATE_BLOCKED && 
                    thread2->locks_held > 0 && thread1->locks_held > 0) {
                    
                    printf("🔴 POTENTIAL DEADLOCK DETECTED!\n");
                    printf("  Thread %s (TID: %d) - Holding: %d, Waiting: %d\n",
                           thread1->name, thread1->tid, thread1->locks_held, thread1->locks_waiting);
                    printf("  Thread %s (TID: %d) - Holding: %d, Waiting: %d\n",
                           thread2->name, thread2->tid, thread2->locks_held, thread2->locks_waiting);
                    
                    atomic_fetch_add(&g_monitor.total_deadlocks_detected, 1);
                }
            }
        }
    }
    
    pthread_mutex_unlock(&g_monitor.registry_mutex);
}

// Print current status
void tm_print_status() {
    static int report_counter = 0;
    
    if (++report_counter % 10 != 0) return; // Print every 10 cycles
    
    printf("\n=== THREAD MONITOR STATUS ===\n");
    printf("Active threads: %d\n", atomic_load(&g_monitor.thread_count));
    printf("Total deadlocks detected: %lu\n", atomic_load(&g_monitor.total_deadlocks_detected));
    printf("Total context switches: %lu\n", atomic_load(&g_monitor.total_context_switches));
    
    pthread_mutex_lock(&g_monitor.registry_mutex);
    
    printf("\nThread Details:\n");
    printf("%-20s %-8s %-12s %-8s %-8s %-12s\n", 
           "NAME", "TID", "STATE", "LOCKS", "RESP", "MAX_RESP(ms)");
    printf("%-20s %-8s %-12s %-8s %-8s %-12s\n", 
           "----", "---", "-----", "-----", "----", "------------");
    
    for (int i = 0; i < atomic_load(&g_monitor.thread_count); i++) {
        thread_info_t* thread = &g_monitor.threads[i];
        
        const char* state_str;
        switch (thread->state) {
            case THREAD_STATE_CREATED: state_str = "CREATED"; break;
            case THREAD_STATE_RUNNING: state_str = "RUNNING"; break;
            case THREAD_STATE_WAITING: state_str = "WAITING"; break;
            case THREAD_STATE_BLOCKED: state_str = "BLOCKED"; break;
            case THREAD_STATE_TERMINATED: state_str = "TERMINATED"; break;
            default: state_str = "UNKNOWN"; break;
        }
        
        printf("%-20s %-8d %-12s %-8d %-8s %-12lu\n",
               thread->name, thread->tid, state_str, thread->locks_held,
               thread->is_responsive ? "YES" : "NO", thread->max_response_time_ms);
    }
    
    pthread_mutex_unlock(&g_monitor.registry_mutex);
    printf("=============================\n\n");
}

// Wrapper macros for easy integration
#define TM_LOCK_ACQUIRED(lock) tm_lock_acquired(lock, __FUNCTION__, __FILE__, __LINE__)
#define TM_LOCK_RELEASED(lock) tm_lock_released(lock)
#define TM_SET_STATE(state) tm_set_state(state)

// Signal handler for graceful shutdown
void tm_signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down thread monitor...\n", sig);
    atomic_store(&g_monitor.monitoring_active, false);
}

// Cleanup function
void tm_shutdown() {
    atomic_store(&g_monitor.monitoring_active, false);
    
    if (g_monitor.monitor_thread) {
        pthread_join(g_monitor.monitor_thread, NULL);
    }
    
    if (g_monitor.log_file) {
        fclose(g_monitor.log_file);
    }
    
    printf("Thread monitor shutdown complete\n");
}
```

### Comprehensive Deadlock Detection System

```c
// deadlock_detector.c - Advanced deadlock detection and prevention
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#define MAX_LOCKS 1024
#define MAX_THREADS 256
#define MAX_LOCK_NAME 64

typedef struct lock_info {
    void* lock_address;
    char name[MAX_LOCK_NAME];
    int owner_thread_id;
    struct timespec acquisition_time;
    atomic_int waiting_threads[MAX_THREADS];
    int waiting_count;
} lock_info_t;

typedef struct thread_lock_info {
    int thread_id;
    int held_locks[MAX_LOCKS];
    int held_count;
    int waiting_for_lock;
    struct timespec wait_start_time;
} thread_lock_info_t;

typedef struct deadlock_detector {
    lock_info_t locks[MAX_LOCKS];
    atomic_int lock_count;
    
    thread_lock_info_t threads[MAX_THREADS];
    atomic_int thread_count;
    
    pthread_mutex_t detector_mutex;
    
    // Detection algorithm state
    int dependency_graph[MAX_THREADS][MAX_THREADS];
    bool visited[MAX_THREADS];
    bool recursion_stack[MAX_THREADS];
    
    // Statistics
    atomic_ulong deadlocks_detected;
    atomic_ulong deadlocks_prevented;
    atomic_ulong false_positives;
    
    // Configuration
    bool enable_prevention;
    bool enable_recovery;
    int detection_interval_ms;
    int max_wait_time_ms;
} deadlock_detector_t;

static deadlock_detector_t g_detector = {
    .lock_count = 0,
    .thread_count = 0,
    .detector_mutex = PTHREAD_MUTEX_INITIALIZER,
    .deadlocks_detected = 0,
    .deadlocks_prevented = 0,
    .false_positives = 0,
    .enable_prevention = true,
    .enable_recovery = false,
    .detection_interval_ms = 100,
    .max_wait_time_ms = 5000
};

// Initialize deadlock detector
int dd_init() {
    pthread_mutex_init(&g_detector.detector_mutex, NULL);
    
    // Clear dependency graph
    memset(g_detector.dependency_graph, 0, sizeof(g_detector.dependency_graph));
    
    printf("Deadlock detector initialized\n");
    return 0;
}

// Register a lock for monitoring
int dd_register_lock(void* lock_addr, const char* name) {
    pthread_mutex_lock(&g_detector.detector_mutex);
    
    int index = atomic_fetch_add(&g_detector.lock_count, 1);
    if (index >= MAX_LOCKS) {
        pthread_mutex_unlock(&g_detector.detector_mutex);
        return -1;
    }
    
    lock_info_t* lock = &g_detector.locks[index];
    lock->lock_address = lock_addr;
    strncpy(lock->name, name ? name : "Unknown", MAX_LOCK_NAME - 1);
    lock->owner_thread_id = -1;
    lock->waiting_count = 0;
    
    pthread_mutex_unlock(&g_detector.detector_mutex);
    
    printf("Registered lock: %s at %p\n", lock->name, lock_addr);
    return index;
}

// Find lock index by address
int dd_find_lock_index(void* lock_addr) {
    for (int i = 0; i < atomic_load(&g_detector.lock_count); i++) {
        if (g_detector.locks[i].lock_address == lock_addr) {
            return i;
        }
    }
    return -1;
}

// Find or create thread info
int dd_get_thread_index(int thread_id) {
    // Look for existing thread
    for (int i = 0; i < atomic_load(&g_detector.thread_count); i++) {
        if (g_detector.threads[i].thread_id == thread_id) {
            return i;
        }
    }
    
    // Create new thread info
    int index = atomic_fetch_add(&g_detector.thread_count, 1);
    if (index < MAX_THREADS) {
        thread_lock_info_t* thread = &g_detector.threads[index];
        thread->thread_id = thread_id;
        thread->held_count = 0;
        thread->waiting_for_lock = -1;
        return index;
    }
    
    return -1;
}

// Build dependency graph for cycle detection
void dd_build_dependency_graph() {
    // Clear graph
    memset(g_detector.dependency_graph, 0, sizeof(g_detector.dependency_graph));
    
    // For each thread, if it's waiting for a lock held by another thread,
    // create an edge in the dependency graph
    for (int i = 0; i < atomic_load(&g_detector.thread_count); i++) {
        thread_lock_info_t* waiting_thread = &g_detector.threads[i];
        
        if (waiting_thread->waiting_for_lock != -1) {
            lock_info_t* lock = &g_detector.locks[waiting_thread->waiting_for_lock];
            
            // Find the thread that owns this lock
            for (int j = 0; j < atomic_load(&g_detector.thread_count); j++) {
                thread_lock_info_t* holding_thread = &g_detector.threads[j];
                
                if (holding_thread->thread_id == lock->owner_thread_id) {
                    // Create dependency: waiting_thread depends on holding_thread
                    g_detector.dependency_graph[i][j] = 1;
                    break;
                }
            }
        }
    }
}

// Depth-first search for cycle detection
bool dd_has_cycle_util(int node) {
    g_detector.visited[node] = true;
    g_detector.recursion_stack[node] = true;
    
    // Check all adjacent vertices
    for (int i = 0; i < atomic_load(&g_detector.thread_count); i++) {
        if (g_detector.dependency_graph[node][i]) {
            if (!g_detector.visited[i] && dd_has_cycle_util(i)) {
                return true;
            } else if (g_detector.recursion_stack[i]) {
                return true; // Back edge found - cycle detected!
            }
        }
    }
    
    g_detector.recursion_stack[node] = false;
    return false;
}

// Detect cycles in dependency graph
bool dd_detect_cycle() {
    // Initialize visited and recursion stack
    memset(g_detector.visited, false, sizeof(g_detector.visited));
    memset(g_detector.recursion_stack, false, sizeof(g_detector.recursion_stack));
    
    // Check for cycles starting from each unvisited node
    for (int i = 0; i < atomic_load(&g_detector.thread_count); i++) {
        if (!g_detector.visited[i]) {
            if (dd_has_cycle_util(i)) {
                return true;
            }
        }
    }
    
    return false;
}

// Record lock acquisition attempt
bool dd_before_lock(void* lock_addr, int thread_id) {
    if (!g_detector.enable_prevention) return true;
    
    pthread_mutex_lock(&g_detector.detector_mutex);
    
    int lock_index = dd_find_lock_index(lock_addr);
    int thread_index = dd_get_thread_index(thread_id);
    
    if (lock_index == -1 || thread_index == -1) {
        pthread_mutex_unlock(&g_detector.detector_mutex);
        return true; // Unknown lock/thread - allow
    }
    
    lock_info_t* lock = &g_detector.locks[lock_index];
    thread_lock_info_t* thread = &g_detector.threads[thread_index];
    
    // If lock is available, allow acquisition
    if (lock->owner_thread_id == -1) {
        pthread_mutex_unlock(&g_detector.detector_mutex);
        return true;
    }
    
    // If lock is held by same thread, allow (recursive lock)
    if (lock->owner_thread_id == thread_id) {
        pthread_mutex_unlock(&g_detector.detector_mutex);
        return true;
    }
    
    // Record waiting state
    thread->waiting_for_lock = lock_index;
    clock_gettime(CLOCK_MONOTONIC, &thread->wait_start_time);
    
    // Add to lock's waiting list
    atomic_store(&lock->waiting_threads[lock->waiting_count++], thread_id);
    
    // Build dependency graph and check for cycles
    dd_build_dependency_graph();
    
    if (dd_detect_cycle()) {
        printf("🔴 DEADLOCK PREVENTED!\n");
        printf("  Thread %d attempting to acquire lock %s would create a cycle\n",
               thread_id, lock->name);
        
        // Remove from waiting list
        thread->waiting_for_lock = -1;
        lock->waiting_count--;
        
        atomic_fetch_add(&g_detector.deadlocks_prevented, 1);
        
        pthread_mutex_unlock(&g_detector.detector_mutex);
        return false; // Prevent acquisition
    }
    
    pthread_mutex_unlock(&g_detector.detector_mutex);
    return true; // Allow acquisition
}

// Record successful lock acquisition
void dd_after_lock(void* lock_addr, int thread_id) {
    pthread_mutex_lock(&g_detector.detector_mutex);
    
    int lock_index = dd_find_lock_index(lock_addr);
    int thread_index = dd_get_thread_index(thread_id);
    
    if (lock_index != -1 && thread_index != -1) {
        lock_info_t* lock = &g_detector.locks[lock_index];
        thread_lock_info_t* thread = &g_detector.threads[thread_index];
        
        // Record ownership
        lock->owner_thread_id = thread_id;
        clock_gettime(CLOCK_MONOTONIC, &lock->acquisition_time);
        
        // Add to thread's held locks
        if (thread->held_count < MAX_LOCKS) {
            thread->held_locks[thread->held_count++] = lock_index;
        }
        
        // Clear waiting state
        thread->waiting_for_lock = -1;
        
        printf("Lock %s acquired by thread %d\n", lock->name, thread_id);
    }
    
    pthread_mutex_unlock(&g_detector.detector_mutex);
}

// Record lock release
void dd_release_lock(void* lock_addr, int thread_id) {
    pthread_mutex_lock(&g_detector.detector_mutex);
    
    int lock_index = dd_find_lock_index(lock_addr);
    int thread_index = dd_get_thread_index(thread_id);
    
    if (lock_index != -1 && thread_index != -1) {
        lock_info_t* lock = &g_detector.locks[lock_index];
        thread_lock_info_t* thread = &g_detector.threads[thread_index];
        
        // Verify ownership
        if (lock->owner_thread_id == thread_id) {
            lock->owner_thread_id = -1;
            
            // Remove from thread's held locks
            for (int i = 0; i < thread->held_count; i++) {
                if (thread->held_locks[i] == lock_index) {
                    // Shift array
                    for (int j = i; j < thread->held_count - 1; j++) {
                        thread->held_locks[j] = thread->held_locks[j + 1];
                    }
                    thread->held_count--;
                    break;
                }
            }
            
            printf("Lock %s released by thread %d\n", lock->name, thread_id);
        }
    }
    
    pthread_mutex_unlock(&g_detector.detector_mutex);
}

// Print deadlock detector status
void dd_print_status() {
    pthread_mutex_lock(&g_detector.detector_mutex);
    
    printf("\n=== DEADLOCK DETECTOR STATUS ===\n");
    printf("Locks monitored: %d\n", atomic_load(&g_detector.lock_count));
    printf("Threads tracked: %d\n", atomic_load(&g_detector.thread_count));
    printf("Deadlocks detected: %lu\n", atomic_load(&g_detector.deadlocks_detected));
    printf("Deadlocks prevented: %lu\n", atomic_load(&g_detector.deadlocks_prevented));
    
    printf("\nLock Status:\n");
    for (int i = 0; i < atomic_load(&g_detector.lock_count); i++) {
        lock_info_t* lock = &g_detector.locks[i];
        printf("  %s: %s (Owner: %d, Waiting: %d)\n",
               lock->name,
               lock->owner_thread_id == -1 ? "FREE" : "HELD",
               lock->owner_thread_id,
               lock->waiting_count);
    }
    
    pthread_mutex_unlock(&g_detector.detector_mutex);
    printf("===============================\n\n");
}

// Wrapper macros for easy integration
#define DD_BEFORE_LOCK(lock, tid) dd_before_lock(lock, tid)
#define DD_AFTER_LOCK(lock, tid) dd_after_lock(lock, tid)
#define DD_RELEASE_LOCK(lock, tid) dd_release_lock(lock, tid)
```

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_THREADS 10
#define MAX_NAME_LEN 32

typedef enum {
    THREAD_CREATED,
    THREAD_RUNNING,
    THREAD_WAITING,
    THREAD_FINISHED
} ThreadState;

typedef struct {
    pthread_t thread_id;
    char name[MAX_NAME_LEN];
    ThreadState state;
    time_t last_update;
    void* current_function;
} ThreadInfo;

static ThreadInfo thread_registry[MAX_THREADS];
static int thread_count = 0;
static pthread_mutex_t registry_mutex = PTHREAD_MUTEX_INITIALIZER;

int register_debug_thread(const char* name) {
    pthread_mutex_lock(&registry_mutex);
    
    if (thread_count >= MAX_THREADS) {
        pthread_mutex_unlock(&registry_mutex);
        return -1;
    }
    
    int id = thread_count++;
    thread_registry[id].thread_id = pthread_self();
    strncpy(thread_registry[id].name, name, MAX_NAME_LEN - 1);
    thread_registry[id].state = THREAD_CREATED;
    thread_registry[id].last_update = time(NULL);
    
    pthread_mutex_unlock(&registry_mutex);
    return id;
}

void update_thread_state(int thread_id, ThreadState state) {
    if (thread_id < 0 || thread_id >= thread_count) return;
    
    pthread_mutex_lock(&registry_mutex);
    thread_registry[thread_id].state = state;
    thread_registry[thread_id].last_update = time(NULL);
    pthread_mutex_unlock(&registry_mutex);
}

void print_thread_status() {
    pthread_mutex_lock(&registry_mutex);
    
    printf("\n=== Thread Status ===\n");
    printf("%-15s %-12s %-10s %s\n", "Name", "Thread ID", "State", "Last Update");
    printf("%.60s\n", "------------------------------------------------------------");
    
    for (int i = 0; i < thread_count; i++) {
        const char* state_str;
        switch (thread_registry[i].state) {
            case THREAD_CREATED:  state_str = "Created"; break;
            case THREAD_RUNNING:  state_str = "Running"; break;
            case THREAD_WAITING:  state_str = "Waiting"; break;
            case THREAD_FINISHED: state_str = "Finished"; break;
            default: state_str = "Unknown"; break;
        }
        
        printf("%-15s %-12lu %-10s %ld\n",
               thread_registry[i].name,
               (unsigned long)thread_registry[i].thread_id,
               state_str,
               thread_registry[i].last_update);
    }
    
    pthread_mutex_unlock(&registry_mutex);
}

// Example worker function with debug support
void* debug_monitored_worker(void* arg) {
    char* name = (char*)arg;
    int thread_id = register_debug_thread(name);
    
    update_thread_state(thread_id, THREAD_RUNNING);
    
    for (int i = 0; i < 5; i++) {
        printf("%s: Working iteration %d\n", name, i + 1);
        sleep(2);
        
        if (i == 2) {
            update_thread_state(thread_id, THREAD_WAITING);
            printf("%s: Simulating wait condition\n", name);
            sleep(3);
            update_thread_state(thread_id, THREAD_RUNNING);
        }
    }
    
    update_thread_state(thread_id, THREAD_FINISHED);
    return NULL;
}

void demonstrate_thread_monitoring() {
    pthread_t threads[3];
    char* names[] = {"Worker-1", "Worker-2", "Worker-3"};
    
    // Create monitoring threads
    for (int i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, debug_monitored_worker, names[i]);
    }
    
    // Monitor thread status
    for (int i = 0; i < 10; i++) {
        sleep(2);
        print_thread_status();
    }
    
    // Wait for completion
    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }
    
    print_thread_status();
}
```

### Deadlock Detection Tool

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAX_LOCKS 10
#define MAX_THREADS 10

typedef struct {
    pthread_mutex_t* mutex;
    pthread_t owner;
    time_t acquired_time;
    char name[32];
} LockInfo;

typedef struct {
    pthread_t thread_id;
    pthread_mutex_t* waiting_for;
    pthread_mutex_t* owned_locks[MAX_LOCKS];
    int owned_count;
} ThreadLockInfo;

static LockInfo lock_registry[MAX_LOCKS];
static ThreadLockInfo thread_locks[MAX_THREADS];
static int lock_count = 0;
static int thread_lock_count = 0;
static pthread_mutex_t detector_mutex = PTHREAD_MUTEX_INITIALIZER;

void register_lock(pthread_mutex_t* mutex, const char* name) {
    pthread_mutex_lock(&detector_mutex);
    
    if (lock_count < MAX_LOCKS) {
        lock_registry[lock_count].mutex = mutex;
        lock_registry[lock_count].owner = 0;
        lock_registry[lock_count].acquired_time = 0;
        strncpy(lock_registry[lock_count].name, name, 31);
        lock_count++;
    }
    
    pthread_mutex_unlock(&detector_mutex);
}

void record_lock_acquisition(pthread_mutex_t* mutex) {
    pthread_mutex_lock(&detector_mutex);
    
    pthread_t current_thread = pthread_self();
    
    // Update lock registry
    for (int i = 0; i < lock_count; i++) {
        if (lock_registry[i].mutex == mutex) {
            lock_registry[i].owner = current_thread;
            lock_registry[i].acquired_time = time(NULL);
            break;
        }
    }
    
    // Update thread lock info
    for (int i = 0; i < thread_lock_count; i++) {
        if (thread_locks[i].thread_id == current_thread) {
            if (thread_locks[i].owned_count < MAX_LOCKS) {
                thread_locks[i].owned_locks[thread_locks[i].owned_count++] = mutex;
            }
            thread_locks[i].waiting_for = NULL;
            break;
        }
    }
    
    pthread_mutex_unlock(&detector_mutex);
}

void record_lock_wait(pthread_mutex_t* mutex) {
    pthread_mutex_lock(&detector_mutex);
    
    pthread_t current_thread = pthread_self();
    
    // Find or create thread entry
    int thread_index = -1;
    for (int i = 0; i < thread_lock_count; i++) {
        if (thread_locks[i].thread_id == current_thread) {
            thread_index = i;
            break;
        }
    }
    
    if (thread_index == -1 && thread_lock_count < MAX_THREADS) {
        thread_index = thread_lock_count++;
        thread_locks[thread_index].thread_id = current_thread;
        thread_locks[thread_index].owned_count = 0;
    }
    
    if (thread_index != -1) {
        thread_locks[thread_index].waiting_for = mutex;
    }
    
    pthread_mutex_unlock(&detector_mutex);
}

void detect_potential_deadlock() {
    pthread_mutex_lock(&detector_mutex);
    
    printf("\n=== Potential Deadlock Detection ===\n");
    
    for (int i = 0; i < thread_lock_count; i++) {
        if (thread_locks[i].waiting_for == NULL) continue;
        
        pthread_t waiting_thread = thread_locks[i].thread_id;
        pthread_mutex_t* waiting_mutex = thread_locks[i].waiting_for;
        
        // Find who owns the mutex this thread is waiting for
        pthread_t blocking_thread = 0;
        for (int j = 0; j < lock_count; j++) {
            if (lock_registry[j].mutex == waiting_mutex) {
                blocking_thread = lock_registry[j].owner;
                break;
            }
        }
        
        if (blocking_thread == 0) continue;
        
        // Check if blocking thread is waiting for any mutex owned by waiting thread
        for (int j = 0; j < thread_lock_count; j++) {
            if (thread_locks[j].thread_id != blocking_thread) continue;
            
            if (thread_locks[j].waiting_for != NULL) {
                // Check if blocking thread is waiting for a mutex owned by waiting thread
                for (int k = 0; k < thread_locks[i].owned_count; k++) {
                    if (thread_locks[i].owned_locks[k] == thread_locks[j].waiting_for) {
                        printf("POTENTIAL DEADLOCK DETECTED:\n");
                        printf("  Thread %lu waiting for mutex owned by Thread %lu\n",
                               (unsigned long)waiting_thread, (unsigned long)blocking_thread);
                        printf("  Thread %lu waiting for mutex owned by Thread %lu\n",
                               (unsigned long)blocking_thread, (unsigned long)waiting_thread);
                    }
                }
            }
        }
    }
    
    pthread_mutex_unlock(&detector_mutex);
}

// Wrapper functions for instrumented locking
int debug_mutex_lock(pthread_mutex_t* mutex) {
    record_lock_wait(mutex);
    int result = pthread_mutex_lock(mutex);
    if (result == 0) {
        record_lock_acquisition(mutex);
    }
    return result;
}

int debug_mutex_unlock(pthread_mutex_t* mutex) {
    // Remove from owned locks
    pthread_mutex_lock(&detector_mutex);
    pthread_t current_thread = pthread_self();
    
    for (int i = 0; i < thread_lock_count; i++) {
        if (thread_locks[i].thread_id == current_thread) {
            for (int j = 0; j < thread_locks[i].owned_count; j++) {
                if (thread_locks[i].owned_locks[j] == mutex) {
                    // Shift remaining locks
                    for (int k = j; k < thread_locks[i].owned_count - 1; k++) {
                        thread_locks[i].owned_locks[k] = thread_locks[i].owned_locks[k + 1];
                    }
                    thread_locks[i].owned_count--;
                    break;
                }
            }
            break;
        }
    }
    pthread_mutex_unlock(&detector_mutex);
    
    return pthread_mutex_unlock(mutex);
}
```

## Logging and Trace Analysis

Effective logging and trace analysis are crucial for debugging multi-threaded applications. Unlike single-threaded programs, concurrent applications require sophisticated logging mechanisms that can capture thread interactions, timing relationships, and race conditions without significantly impacting performance or altering the program's behavior.

### Advanced Thread-Safe Logging Framework

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <stdatomic.h>
#include <stdbool.h>

// Advanced logging levels with granular control
typedef enum {
    LOG_TRACE = 0,      // Finest level - function entry/exit
    LOG_DEBUG = 1,      // Detailed debugging information
    LOG_INFO = 2,       // General information
    LOG_WARN = 3,       // Warning conditions
    LOG_ERROR = 4,      // Error conditions
    LOG_FATAL = 5,      // Fatal errors
    LOG_CONCURRENCY = 6, // Special level for threading issues
    LOG_PERFORMANCE = 7  // Performance-related logs
} log_level_t;

// Thread context information
typedef struct {
    pthread_t thread_id;
    pid_t system_tid;
    char thread_name[64];
    int thread_priority;
    size_t stack_size;
    void* stack_addr;
} thread_context_t;

// High-performance ring buffer for lock-free logging
#define RING_BUFFER_SIZE 65536
#define MAX_LOG_MESSAGE_SIZE 512

typedef struct {
    char message[MAX_LOG_MESSAGE_SIZE];
    log_level_t level;
    struct timespec timestamp;
    thread_context_t thread_ctx;
    const char* file;
    int line;
    const char* function;
    uint64_t sequence_number;
} log_entry_t;

typedef struct {
    log_entry_t entries[RING_BUFFER_SIZE];
    atomic_size_t write_index;
    atomic_size_t read_index;
    atomic_bool full;
} ring_buffer_t;

// Advanced logger configuration
typedef struct {
    FILE* log_file;
    log_level_t min_level;
    bool enable_colors;
    bool enable_timestamps;
    bool enable_thread_info;
    bool enable_source_info;
    bool enable_performance_timing;
    bool enable_binary_format;
    
    // Lock-free ring buffer for high-performance logging
    ring_buffer_t ring_buffer;
    pthread_t background_writer;
    atomic_bool shutdown_requested;
    
    // Statistics
    atomic_uint64_t total_messages;
    atomic_uint64_t dropped_messages;
    atomic_uint64_t messages_by_level[8];
    
    // Filtering
    char include_filter[256];
    char exclude_filter[256];
    
    // Performance metrics
    struct timespec start_time;
    atomic_uint64_t total_write_time_ns;
    atomic_uint64_t max_write_time_ns;
    
    // Thread safety for file operations
    pthread_mutex_t file_mutex;
    pthread_cond_t writer_cond;
} advanced_logger_t;

static advanced_logger_t g_logger = {
    .log_file = NULL,
    .min_level = LOG_INFO,
    .enable_colors = true,
    .enable_timestamps = true,
    .enable_thread_info = true,
    .enable_source_info = true,
    .enable_performance_timing = false,
    .enable_binary_format = false,
    .shutdown_requested = ATOMIC_VAR_INIT(false),
    .total_messages = ATOMIC_VAR_INIT(0),
    .dropped_messages = ATOMIC_VAR_INIT(0),
    .total_write_time_ns = ATOMIC_VAR_INIT(0),
    .max_write_time_ns = ATOMIC_VAR_INIT(0),
    .file_mutex = PTHREAD_MUTEX_INITIALIZER,
    .writer_cond = PTHREAD_COND_INITIALIZER
};

// Get system thread ID (Linux-specific)
static pid_t get_system_tid(void) {
#ifdef __linux__
    return syscall(SYS_gettid);
#else
    return getpid(); // Fallback for non-Linux systems
#endif
}

// Get thread context information
static thread_context_t get_thread_context(void) {
    thread_context_t ctx = {0};
    ctx.thread_id = pthread_self();
    ctx.system_tid = get_system_tid();
    
    // Get thread name if available
    pthread_getname_np(ctx.thread_id, ctx.thread_name, sizeof(ctx.thread_name));
    if (strlen(ctx.thread_name) == 0) {
        snprintf(ctx.thread_name, sizeof(ctx.thread_name), "Thread-%ld", 
                (long)ctx.thread_id);
    }
    
    // Get thread attributes
    pthread_attr_t attr;
    if (pthread_getattr_np(ctx.thread_id, &attr) == 0) {
        pthread_attr_getstacksize(&attr, &ctx.stack_size);
        pthread_attr_getstackaddr(&attr, &ctx.stack_addr);
        pthread_attr_destroy(&attr);
    }
    
    return ctx;
}

// High-resolution timestamp
static struct timespec get_timestamp(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts;
}

// Background writer thread for ring buffer
static void* background_writer_thread(void* arg) {
    advanced_logger_t* logger = (advanced_logger_t*)arg;
    
    pthread_setname_np(pthread_self(), "LogWriter");
    
    while (!atomic_load(&logger->shutdown_requested)) {
        size_t read_idx = atomic_load(&logger->ring_buffer.read_index);
        size_t write_idx = atomic_load(&logger->ring_buffer.write_index);
        
        if (read_idx == write_idx && !atomic_load(&logger->ring_buffer.full)) {
            // Buffer is empty, wait for data
            pthread_mutex_lock(&logger->file_mutex);
            pthread_cond_wait(&logger->writer_cond, &logger->file_mutex);
            pthread_mutex_unlock(&logger->file_mutex);
            continue;
        }
        
        // Read and write log entries
        while (read_idx != write_idx || atomic_load(&logger->ring_buffer.full)) {
            log_entry_t* entry = &logger->ring_buffer.entries[read_idx];
            
            // Format and write the log entry
            pthread_mutex_lock(&logger->file_mutex);
            
            // Color codes for terminal output
            const char* color_codes[] = {
                "\033[0;37m", // TRACE - Light Gray
                "\033[0;36m", // DEBUG - Cyan
                "\033[0;32m", // INFO - Green
                "\033[0;33m", // WARN - Yellow
                "\033[0;31m", // ERROR - Red
                "\033[1;31m", // FATAL - Bold Red
                "\033[1;35m", // CONCURRENCY - Bold Magenta
                "\033[0;35m"  // PERFORMANCE - Magenta
            };
            const char* reset_color = "\033[0m";
            
            // Write timestamp
            if (logger->enable_timestamps) {
                fprintf(logger->log_file, "[%010ld.%09ld] ", 
                       entry->timestamp.tv_sec, entry->timestamp.tv_nsec);
            }
            
            // Write log level with color
            const char* level_names[] = {
                "TRACE", "DEBUG", "INFO", "WARN", 
                "ERROR", "FATAL", "CONCUR", "PERF"
            };
            
            if (logger->enable_colors && logger->log_file == stdout) {
                fprintf(logger->log_file, "%s[%s]%s ", 
                       color_codes[entry->level], 
                       level_names[entry->level], 
                       reset_color);
            } else {
                fprintf(logger->log_file, "[%s] ", level_names[entry->level]);
            }
            
            // Write thread information
            if (logger->enable_thread_info) {
                fprintf(logger->log_file, "[%s:%d] ", 
                       entry->thread_ctx.thread_name, 
                       entry->thread_ctx.system_tid);
            }
            
            // Write source location
            if (logger->enable_source_info) {
                fprintf(logger->log_file, "[%s:%d:%s] ", 
                       entry->file, entry->line, entry->function);
            }
            
            // Write sequence number for ordering verification
            fprintf(logger->log_file, "[#%lu] ", entry->sequence_number);
            
            // Write the actual message
            fprintf(logger->log_file, "%s\n", entry->message);
            
            fflush(logger->log_file);
            pthread_mutex_unlock(&logger->file_mutex);
            
            // Move to next entry
            read_idx = (read_idx + 1) % RING_BUFFER_SIZE;
            atomic_store(&logger->ring_buffer.read_index, read_idx);
            atomic_store(&logger->ring_buffer.full, false);
        }
    }
    
    return NULL;
}

// Initialize the advanced logger
int init_advanced_logger(const char* filename, log_level_t min_level) {
    // Initialize ring buffer
    atomic_store(&g_logger.ring_buffer.write_index, 0);
    atomic_store(&g_logger.ring_buffer.read_index, 0);
    atomic_store(&g_logger.ring_buffer.full, false);
    
    // Open log file
    if (filename) {
        g_logger.log_file = fopen(filename, "a");
        if (!g_logger.log_file) {
            perror("Failed to open log file");
            return -1;
        }
    } else {
        g_logger.log_file = stdout;
    }
    
    g_logger.min_level = min_level;
    clock_gettime(CLOCK_MONOTONIC_RAW, &g_logger.start_time);
    
    // Start background writer thread
    if (pthread_create(&g_logger.background_writer, NULL, 
                      background_writer_thread, &g_logger) != 0) {
        perror("Failed to create background writer thread");
        return -1;
    }
    
    return 0;
}

// Advanced logging function with lock-free ring buffer
void advanced_log(log_level_t level, const char* file, int line, 
                 const char* function, const char* format, ...) {
    if (level < g_logger.min_level) return;
    
    struct timespec start_time = get_timestamp();
    
    // Get next write position
    size_t write_idx = atomic_load(&g_logger.ring_buffer.write_index);
    size_t next_write_idx = (write_idx + 1) % RING_BUFFER_SIZE;
    
    if (next_write_idx == atomic_load(&g_logger.ring_buffer.read_index)) {
        // Buffer is full, drop the message
        atomic_fetch_add(&g_logger.dropped_messages, 1);
        return;
    }
    
    // Prepare log entry
    log_entry_t* entry = &g_logger.ring_buffer.entries[write_idx];
    entry->level = level;
    entry->timestamp = start_time;
    entry->thread_ctx = get_thread_context();
    entry->file = file;
    entry->line = line;
    entry->function = function;
    entry->sequence_number = atomic_fetch_add(&g_logger.total_messages, 1);
    
    // Format message
    va_list args;
    va_start(args, format);
    vsnprintf(entry->message, sizeof(entry->message), format, args);
    va_end(args);
    
    // Update statistics
    atomic_fetch_add(&g_logger.messages_by_level[level], 1);
    
    // Commit the write
    atomic_store(&g_logger.ring_buffer.write_index, next_write_idx);
    
    // Check if buffer is now full
    if (next_write_idx == atomic_load(&g_logger.ring_buffer.read_index)) {
        atomic_store(&g_logger.ring_buffer.full, true);
    }
    
    // Wake up background writer
    pthread_cond_signal(&g_logger.writer_cond);
    
    // Performance timing
    if (g_logger.enable_performance_timing) {
        struct timespec end_time = get_timestamp();
        uint64_t elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000UL +
                             (end_time.tv_nsec - start_time.tv_nsec);
        
        atomic_fetch_add(&g_logger.total_write_time_ns, elapsed_ns);
        
        uint64_t current_max = atomic_load(&g_logger.max_write_time_ns);
        while (elapsed_ns > current_max) {
            if (atomic_compare_exchange_weak(&g_logger.max_write_time_ns, 
                                           &current_max, elapsed_ns)) {
                break;
            }
        }
    }
}

// Convenience macros
#define LOG_TRACE(...) advanced_log(LOG_TRACE, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_DEBUG(...) advanced_log(LOG_DEBUG, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_INFO(...) advanced_log(LOG_INFO, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_WARN(...) advanced_log(LOG_WARN, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_ERROR(...) advanced_log(LOG_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_FATAL(...) advanced_log(LOG_FATAL, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_CONCURRENCY(...) advanced_log(LOG_CONCURRENCY, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_PERFORMANCE(...) advanced_log(LOG_PERFORMANCE, __FILE__, __LINE__, __func__, __VA_ARGS__)

// Specialized macros for threading analysis
#define LOG_THREAD_ENTER(func_name) \
    LOG_TRACE("ENTER: %s", func_name)

#define LOG_THREAD_EXIT(func_name) \
    LOG_TRACE("EXIT: %s", func_name)

#define LOG_LOCK_ACQUIRE(lock_name) \
    LOG_CONCURRENCY("ACQUIRING LOCK: %s", lock_name)

#define LOG_LOCK_ACQUIRED(lock_name) \
    LOG_CONCURRENCY("ACQUIRED LOCK: %s", lock_name)

#define LOG_LOCK_RELEASE(lock_name) \
    LOG_CONCURRENCY("RELEASING LOCK: %s", lock_name)

#define LOG_RACE_CONDITION(description) \
    LOG_CONCURRENCY("POTENTIAL RACE CONDITION: %s", description)

#define LOG_DEADLOCK_RISK(description) \
    LOG_CONCURRENCY("DEADLOCK RISK: %s", description)
```

### Production-Grade Trace Analysis System

```c
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

// Binary trace format for high-performance logging
typedef struct __attribute__((packed)) {
    uint64_t timestamp_ns;
    uint32_t thread_id;
    uint16_t event_type;
    uint16_t data_length;
    char data[0]; // Variable-length data
} binary_trace_entry_t;

// Trace event types
typedef enum {
    TRACE_FUNCTION_ENTER = 1,
    TRACE_FUNCTION_EXIT = 2,
    TRACE_LOCK_ACQUIRE = 3,
    TRACE_LOCK_ACQUIRED = 4,
    TRACE_LOCK_RELEASE = 5,
    TRACE_THREAD_CREATE = 6,
    TRACE_THREAD_JOIN = 7,
    TRACE_CONDITION_WAIT = 8,
    TRACE_CONDITION_SIGNAL = 9,
    TRACE_MEMORY_ALLOCATION = 10,
    TRACE_MEMORY_FREE = 11,
    TRACE_RACE_CONDITION = 12,
    TRACE_DEADLOCK_DETECTED = 13
} trace_event_type_t;

// Memory-mapped trace buffer for extremely high performance
typedef struct {
    void* memory_region;
    size_t total_size;
    size_t current_offset;
    int fd;
    pthread_mutex_t write_mutex;
    char filename[256];
} mmap_trace_buffer_t;

static mmap_trace_buffer_t g_trace_buffer = {0};

// Initialize memory-mapped trace buffer
int init_mmap_trace_buffer(const char* filename, size_t size_mb) {
    g_trace_buffer.total_size = size_mb * 1024 * 1024;
    strncpy(g_trace_buffer.filename, filename, sizeof(g_trace_buffer.filename) - 1);
    
    // Create and open file
    g_trace_buffer.fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (g_trace_buffer.fd == -1) {
        perror("Failed to create trace file");
        return -1;
    }
    
    // Extend file to desired size
    if (ftruncate(g_trace_buffer.fd, g_trace_buffer.total_size) == -1) {
        perror("Failed to extend trace file");
        close(g_trace_buffer.fd);
        return -1;
    }
    
    // Memory map the file
    g_trace_buffer.memory_region = mmap(NULL, g_trace_buffer.total_size,
                                       PROT_READ | PROT_WRITE, MAP_SHARED,
                                       g_trace_buffer.fd, 0);
    
    if (g_trace_buffer.memory_region == MAP_FAILED) {
        perror("Failed to memory map trace file");
        close(g_trace_buffer.fd);
        return -1;
    }
    
    pthread_mutex_init(&g_trace_buffer.write_mutex, NULL);
    g_trace_buffer.current_offset = 0;
    
    return 0;
}

// Write binary trace entry
void write_binary_trace(trace_event_type_t event_type, const void* data, size_t data_len) {
    if (!g_trace_buffer.memory_region) return;
    
    size_t entry_size = sizeof(binary_trace_entry_t) + data_len;
    
    pthread_mutex_lock(&g_trace_buffer.write_mutex);
    
    if (g_trace_buffer.current_offset + entry_size > g_trace_buffer.total_size) {
        // Buffer full - could implement circular buffer here
        pthread_mutex_unlock(&g_trace_buffer.write_mutex);
        return;
    }
    
    binary_trace_entry_t* entry = (binary_trace_entry_t*)
        ((char*)g_trace_buffer.memory_region + g_trace_buffer.current_offset);
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    entry->timestamp_ns = ts.tv_sec * 1000000000UL + ts.tv_nsec;
    entry->thread_id = get_system_tid();
    entry->event_type = event_type;
    entry->data_length = data_len;
    
    if (data && data_len > 0) {
        memcpy(entry->data, data, data_len);
    }
    
    g_trace_buffer.current_offset += entry_size;
    
    pthread_mutex_unlock(&g_trace_buffer.write_mutex);
}

// Trace analysis functions
typedef struct {
    uint64_t min_timestamp;
    uint64_t max_timestamp;
    uint32_t total_events;
    uint32_t events_by_type[32];
    uint32_t unique_threads;
    uint64_t total_lock_time;
    uint32_t potential_races;
    uint32_t deadlock_incidents;
} trace_analysis_t;

// Analyze binary trace file
int analyze_trace_file(const char* filename, trace_analysis_t* analysis) {
    memset(analysis, 0, sizeof(trace_analysis_t));
    
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Failed to open trace file for analysis");
        return -1;
    }
    
    struct stat file_stat;
    if (fstat(fd, &file_stat) == -1) {
        perror("Failed to get file stats");
        close(fd);
        return -1;
    }
    
    void* mapped_data = mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data == MAP_FAILED) {
        perror("Failed to map trace file");
        close(fd);
        return -1;
    }
    
    char* data_ptr = (char*)mapped_data;
    char* end_ptr = data_ptr + file_stat.st_size;
    
    // Track thread states for deadlock detection
    typedef struct {
        uint32_t thread_id;
        uint64_t last_lock_time;
        char held_locks[64][32]; // Assuming max 64 locks, 32 chars each
        int lock_count;
    } thread_state_t;
    
    thread_state_t thread_states[256] = {0}; // Support up to 256 threads
    int thread_count = 0;
    
    analysis->min_timestamp = UINT64_MAX;
    analysis->max_timestamp = 0;
    
    while (data_ptr < end_ptr) {
        binary_trace_entry_t* entry = (binary_trace_entry_t*)data_ptr;
        
        // Validate entry
        if (data_ptr + sizeof(binary_trace_entry_t) + entry->data_length > end_ptr) {
            break; // Incomplete entry
        }
        
        analysis->total_events++;
        analysis->events_by_type[entry->event_type]++;
        
        if (entry->timestamp_ns < analysis->min_timestamp) {
            analysis->min_timestamp = entry->timestamp_ns;
        }
        if (entry->timestamp_ns > analysis->max_timestamp) {
            analysis->max_timestamp = entry->timestamp_ns;
        }
        
        // Track unique threads
        bool thread_found = false;
        for (int i = 0; i < thread_count; i++) {
            if (thread_states[i].thread_id == entry->thread_id) {
                thread_found = true;
                break;
            }
        }
        if (!thread_found && thread_count < 256) {
            thread_states[thread_count].thread_id = entry->thread_id;
            thread_count++;
            analysis->unique_threads++;
        }
        
        // Analyze specific event types
        switch (entry->event_type) {
            case TRACE_LOCK_ACQUIRE:
                // Track lock acquisition timing
                break;
            case TRACE_LOCK_ACQUIRED:
                // Calculate lock wait time
                break;
            case TRACE_RACE_CONDITION:
                analysis->potential_races++;
                break;
            case TRACE_DEADLOCK_DETECTED:
                analysis->deadlock_incidents++;
                break;
        }
        
        // Move to next entry
        data_ptr += sizeof(binary_trace_entry_t) + entry->data_length;
    }
    
    munmap(mapped_data, file_stat.st_size);
    close(fd);
    
    return 0;
}

// Print comprehensive trace analysis report
void print_trace_analysis(const trace_analysis_t* analysis) {
    printf("=== TRACE ANALYSIS REPORT ===\n");
    printf("Total Events: %u\n", analysis->total_events);
    printf("Unique Threads: %u\n", analysis->unique_threads);
    printf("Time Span: %.3f seconds\n", 
           (analysis->max_timestamp - analysis->min_timestamp) / 1e9);
    printf("Potential Race Conditions: %u\n", analysis->potential_races);
    printf("Deadlock Incidents: %u\n", analysis->deadlock_incidents);
    
    printf("\nEvents by Type:\n");
    const char* event_names[] = {
        "UNKNOWN", "FUNCTION_ENTER", "FUNCTION_EXIT", "LOCK_ACQUIRE",
        "LOCK_ACQUIRED", "LOCK_RELEASE", "THREAD_CREATE", "THREAD_JOIN",
        "CONDITION_WAIT", "CONDITION_SIGNAL", "MEMORY_ALLOCATION",
        "MEMORY_FREE", "RACE_CONDITION", "DEADLOCK_DETECTED"
    };
    
    for (int i = 1; i < 14; i++) {
        if (analysis->events_by_type[i] > 0) {
            printf("  %s: %u\n", event_names[i], analysis->events_by_type[i]);
        }
    }
}
```

### Visualization and Timing Analysis Tools

```c
// Generate timing diagram data for visualization
typedef struct {
    char thread_name[64];
    uint64_t start_time;
    uint64_t end_time;
    char operation[128];
    int lock_level; // For nested lock visualization
} timeline_event_t;

// Export timeline data to CSV for external visualization tools
int export_timeline_csv(const char* trace_file, const char* csv_file) {
    FILE* csv = fopen(csv_file, "w");
    if (!csv) {
        perror("Failed to create CSV file");
        return -1;
    }
    
    fprintf(csv, "Thread,StartTime,EndTime,Operation,LockLevel\n");
    
    // Parse trace file and generate timeline events
    // Implementation would read binary trace and convert to CSV format
    
    fclose(csv);
    return 0;
}

// Generate Graphviz DOT file for thread interaction diagram
int generate_thread_interaction_graph(const char* trace_file, const char* dot_file) {
    FILE* dot = fopen(dot_file, "w");
    if (!dot) {
        perror("Failed to create DOT file");
        return -1;
    }
    
    fprintf(dot, "digraph ThreadInteractions {\n");
    fprintf(dot, "  rankdir=TB;\n");
    fprintf(dot, "  node [shape=box];\n");
    
    // Parse trace file and generate thread interaction graph
    // This would analyze synchronization patterns and create a visual graph
    
    fprintf(dot, "}\n");
    fclose(dot);
    
    printf("Generated thread interaction graph: %s\n", dot_file);
    printf("Convert to image with: dot -Tpng %s -o thread_interactions.png\n", dot_file);
    
    return 0;
}
```

## Debugging Exercises

These exercises are designed to test your ability to identify, debug, and fix complex threading issues in real-world scenarios. Each exercise builds upon previous concepts and introduces new challenges.

### Exercise 1: Multi-Level Race Condition Detective

**Difficulty: Intermediate**
**Estimated Time: 45 minutes**

Find and fix all race conditions in this banking system simulation. The code contains multiple types of race conditions at different levels.

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <stdatomic.h>

#define MAX_ACCOUNTS 1000
#define MAX_TRANSACTIONS 10000
#define NUM_THREADS 8

typedef struct {
    int account_id;
    volatile double balance;        // ISSUE 1: What's wrong here?
    pthread_mutex_t account_mutex;
    int transaction_count;
    double* transaction_history;    // ISSUE 2: Unprotected access
    int history_size;
    int history_capacity;
} bank_account_t;

typedef struct {
    bank_account_t accounts[MAX_ACCOUNTS];
    int total_accounts;
    volatile double total_bank_balance;  // ISSUE 3: Multiple race conditions
    pthread_mutex_t bank_mutex;
    
    // Statistics - multiple race conditions here
    int successful_transactions;
    int failed_transactions;
    double largest_transaction;
    int most_active_account;
} bank_t;

static bank_t global_bank = {0};

// Initialize bank with accounts
void init_bank(int num_accounts) {
    global_bank.total_accounts = num_accounts;
    global_bank.total_bank_balance = 0.0;
    pthread_mutex_init(&global_bank.bank_mutex, NULL);
    
    for (int i = 0; i < num_accounts; i++) {
        global_bank.accounts[i].account_id = i;
        global_bank.accounts[i].balance = 1000.0; // Initial balance
        pthread_mutex_init(&global_bank.accounts[i].account_mutex, NULL);
        global_bank.accounts[i].transaction_count = 0;
        global_bank.accounts[i].history_capacity = 100;
        global_bank.accounts[i].transaction_history = 
            malloc(100 * sizeof(double));
        global_bank.accounts[i].history_size = 0;
        
        global_bank.total_bank_balance += 1000.0; // RACE CONDITION!
    }
}

// Record transaction in account history
void record_transaction(bank_account_t* account, double amount) {
    // RACE CONDITION: Multiple threads can access history simultaneously
    if (account->history_size >= account->history_capacity) {
        account->history_capacity *= 2;
        account->transaction_history = realloc(account->transaction_history,
            account->history_capacity * sizeof(double));
    }
    
    account->transaction_history[account->history_size] = amount;
    account->history_size++;
    account->transaction_count++; // RACE CONDITION!
    
    // Update statistics - MULTIPLE RACE CONDITIONS!
    if (amount > global_bank.largest_transaction) {
        global_bank.largest_transaction = amount;
    }
    
    if (account->transaction_count > 
        global_bank.accounts[global_bank.most_active_account].transaction_count) {
        global_bank.most_active_account = account->account_id;
    }
}

// Transfer money between accounts
int transfer_money(int from_account, int to_account, double amount) {
    if (from_account == to_account) return 0;
    if (from_account >= global_bank.total_accounts || 
        to_account >= global_bank.total_accounts) return 0;
    if (amount <= 0) return 0;
    
    bank_account_t* from = &global_bank.accounts[from_account];
    bank_account_t* to = &global_bank.accounts[to_account];
    
    // DEADLOCK RISK: Lock ordering not consistent
    pthread_mutex_lock(&from->account_mutex);
    usleep(rand() % 1000); // Simulate processing time - increases deadlock risk
    pthread_mutex_lock(&to->account_mutex);
    
    if (from->balance >= amount) {
        from->balance -= amount;
        to->balance += amount;
        
        // Record transactions
        record_transaction(from, -amount);
        record_transaction(to, amount);
        
        pthread_mutex_unlock(&to->account_mutex);
        pthread_mutex_unlock(&from->account_mutex);
        
        global_bank.successful_transactions++; // RACE CONDITION!
        return 1;
    } else {
        pthread_mutex_unlock(&to->account_mutex);
        pthread_mutex_unlock(&from->account_mutex);
        
        global_bank.failed_transactions++; // RACE CONDITION!
        return 0;
    }
}

// Get account balance (seems simple, but has issues)
double get_account_balance(int account_id) {
    if (account_id >= global_bank.total_accounts) return -1.0;
    
    // RACE CONDITION: Reading without proper synchronization
    return global_bank.accounts[account_id].balance;
}

// Calculate total bank balance
double calculate_total_balance() {
    double total = 0.0;
    
    // RACE CONDITION: Not locking all accounts atomically
    for (int i = 0; i < global_bank.total_accounts; i++) {
        total += global_bank.accounts[i].balance;
    }
    
    return total;
}

// Worker thread function
void* transaction_worker(void* arg) {
    int thread_id = *(int*)arg;
    srand(time(NULL) + thread_id); // Seed random number generator
    
    for (int i = 0; i < MAX_TRANSACTIONS / NUM_THREADS; i++) {
        int from = rand() % global_bank.total_accounts;
        int to = rand() % global_bank.total_accounts;
        double amount = (rand() % 100) + 1; // $1-$100
        
        if (transfer_money(from, to, amount)) {
            // Success - maybe check balance?
            double balance = get_account_balance(from);
            if (balance < 0) {
                printf("ERROR: Negative balance detected! Account %d: $%.2f\n", 
                       from, balance);
            }
        }
        
        // Occasionally check total balance
        if (i % 1000 == 0) {
            double total = calculate_total_balance();
            printf("Thread %d: Total bank balance: $%.2f\n", thread_id, total);
        }
    }
    
    return NULL;
}

// Print final statistics
void print_statistics() {
    printf("\n=== FINAL BANK STATISTICS ===\n");
    printf("Successful transactions: %d\n", global_bank.successful_transactions);
    printf("Failed transactions: %d\n", global_bank.failed_transactions);
    printf("Largest transaction: $%.2f\n", global_bank.largest_transaction);
    printf("Most active account: %d\n", global_bank.most_active_account);
    printf("Total bank balance: $%.2f\n", calculate_total_balance());
    
    // Print individual account balances
    for (int i = 0; i < global_bank.total_accounts && i < 10; i++) {
        printf("Account %d: $%.2f (%d transactions)\n", 
               i, global_bank.accounts[i].balance, 
               global_bank.accounts[i].transaction_count);
    }
}

int main() {
    printf("Starting banking system simulation...\n");
    
    init_bank(100);
    
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, transaction_worker, &thread_ids[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    print_statistics();
    
    return 0;
}
```

**Your Task:**
1. Identify all race conditions in the code (there are at least 8)
2. Identify the deadlock risk and fix it
3. Fix all synchronization issues while maintaining performance
4. Add proper error handling and input validation
5. Implement a deadlock detection mechanism
6. Add comprehensive logging to trace the issues

**Expected Issues to Find:**
- Unprotected shared variables
- Inconsistent lock ordering (deadlock risk)
- Race conditions in statistics tracking
- Memory access race conditions
- Double-checked locking issues
- Time-of-check to time-of-use bugs

### Exercise 2: Producer-Consumer Deadlock Nightmare

**Difficulty: Advanced**
**Estimated Time: 60 minutes**

This exercise simulates a complex producer-consumer system with multiple types of resources and priorities. The code contains several subtle deadlock scenarios.

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>

#define MAX_PRODUCERS 5
#define MAX_CONSUMERS 5
#define MAX_RESOURCES 3
#define BUFFER_SIZE 10
#define MAX_ITEMS 100

typedef enum {
    RESOURCE_TYPE_A,
    RESOURCE_TYPE_B,
    RESOURCE_TYPE_C
} resource_type_t;

typedef struct {
    int item_id;
    resource_type_t required_resources[MAX_RESOURCES];
    int resource_count;
    int priority; // 1-10, higher is more important
    time_t creation_time;
    int producer_id;
} work_item_t;

typedef struct {
    work_item_t buffer[BUFFER_SIZE];
    int head;
    int tail;
    int count;
    pthread_mutex_t buffer_mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
} circular_buffer_t;

typedef struct {
    int available_count[MAX_RESOURCES];
    pthread_mutex_t resource_mutex[MAX_RESOURCES];
    pthread_cond_t resource_available[MAX_RESOURCES];
    
    // Resource allocation tracking
    int allocated_to_consumer[MAX_CONSUMERS][MAX_RESOURCES];
    bool consumer_waiting[MAX_CONSUMERS];
} resource_manager_t;

// Global state
static circular_buffer_t g_work_queue = {0};
static resource_manager_t g_resources = {0};
static pthread_mutex_t g_stats_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_items_produced = 0;
static int g_items_consumed = 0;
static bool g_shutdown_requested = false;

// Initialize systems
void init_systems() {
    // Initialize work queue
    g_work_queue.head = 0;
    g_work_queue.tail = 0;
    g_work_queue.count = 0;
    pthread_mutex_init(&g_work_queue.buffer_mutex, NULL);
    pthread_cond_init(&g_work_queue.not_full, NULL);
    pthread_cond_init(&g_work_queue.not_empty, NULL);
    
    // Initialize resources
    for (int i = 0; i < MAX_RESOURCES; i++) {
        g_resources.available_count[i] = 3; // 3 of each resource type
        pthread_mutex_init(&g_resources.resource_mutex[i], NULL);
        pthread_cond_init(&g_resources.resource_available[i], NULL);
    }
    
    for (int i = 0; i < MAX_CONSUMERS; i++) {
        for (int j = 0; j < MAX_RESOURCES; j++) {
            g_resources.allocated_to_consumer[i][j] = 0;
        }
        g_resources.consumer_waiting[i] = false;
    }
}

// DEADLOCK RISK: Acquire multiple resources without proper ordering
int acquire_resources(int consumer_id, resource_type_t* resources, int count) {
    // PROBLEM: Lock acquisition order varies based on resource types
    for (int i = 0; i < count; i++) {
        pthread_mutex_lock(&g_resources.resource_mutex[resources[i]]);
        
        while (g_resources.available_count[resources[i]] == 0) {
            g_resources.consumer_waiting[consumer_id] = true;
            
            // DEADLOCK RISK: Waiting while holding other locks
            pthread_cond_wait(&g_resources.resource_available[resources[i]], 
                            &g_resources.resource_mutex[resources[i]]);
        }
        
        g_resources.available_count[resources[i]]--;
        g_resources.allocated_to_consumer[consumer_id][resources[i]]++;
        g_resources.consumer_waiting[consumer_id] = false;
        
        pthread_mutex_unlock(&g_resources.resource_mutex[resources[i]]);
    }
    
    return 1;
}

// Release resources
void release_resources(int consumer_id, resource_type_t* resources, int count) {
    // PROBLEM: Release order might not match acquisition order
    for (int i = count - 1; i >= 0; i--) {
        pthread_mutex_lock(&g_resources.resource_mutex[resources[i]]);
        
        g_resources.available_count[resources[i]]++;
        g_resources.allocated_to_consumer[consumer_id][resources[i]]--;
        
        pthread_cond_signal(&g_resources.resource_available[resources[i]]);
        pthread_mutex_unlock(&g_resources.resource_mutex[resources[i]]);
    }
}

// Add work item to queue
int add_work_item(work_item_t* item) {
    pthread_mutex_lock(&g_work_queue.buffer_mutex);
    
    while (g_work_queue.count == BUFFER_SIZE && !g_shutdown_requested) {
        pthread_cond_wait(&g_work_queue.not_full, &g_work_queue.buffer_mutex);
    }
    
    if (g_shutdown_requested) {
        pthread_mutex_unlock(&g_work_queue.buffer_mutex);
        return 0;
    }
    
    // POTENTIAL ISSUE: Priority queue not properly implemented
    g_work_queue.buffer[g_work_queue.tail] = *item;
    g_work_queue.tail = (g_work_queue.tail + 1) % BUFFER_SIZE;
    g_work_queue.count++;
    
    pthread_cond_signal(&g_work_queue.not_empty);
    pthread_mutex_unlock(&g_work_queue.buffer_mutex);
    
    return 1;
}

// Get work item from queue
int get_work_item(work_item_t* item) {
    pthread_mutex_lock(&g_work_queue.buffer_mutex);
    
    while (g_work_queue.count == 0 && !g_shutdown_requested) {
        pthread_cond_wait(&g_work_queue.not_empty, &g_work_queue.buffer_mutex);
    }
    
    if (g_work_queue.count == 0 && g_shutdown_requested) {
        pthread_mutex_unlock(&g_work_queue.buffer_mutex);
        return 0;
    }
    
    *item = g_work_queue.buffer[g_work_queue.head];
    g_work_queue.head = (g_work_queue.head + 1) % BUFFER_SIZE;
    g_work_queue.count--;
    
    pthread_cond_signal(&g_work_queue.not_full);
    pthread_mutex_unlock(&g_work_queue.buffer_mutex);
    
    return 1;
}

// Producer thread
void* producer_thread(void* arg) {
    int producer_id = *(int*)arg;
    
    for (int i = 0; i < MAX_ITEMS / MAX_PRODUCERS; i++) {
        work_item_t item;
        item.item_id = producer_id * 1000 + i;
        item.producer_id = producer_id;
        item.priority = (rand() % 10) + 1;
        item.creation_time = time(NULL);
        
        // Randomly assign required resources
        item.resource_count = (rand() % MAX_RESOURCES) + 1;
        for (int j = 0; j < item.resource_count; j++) {
            item.required_resources[j] = rand() % MAX_RESOURCES;
        }
        
        if (!add_work_item(&item)) {
            break; // Shutdown requested
        }
        
        pthread_mutex_lock(&g_stats_mutex);
        g_items_produced++;
        pthread_mutex_unlock(&g_stats_mutex);
        
        usleep((rand() % 1000) + 500); // Simulate work
    }
    
    return NULL;
}

// Consumer thread
void* consumer_thread(void* arg) {
    int consumer_id = *(int*)arg;
    
    while (true) {
        work_item_t item;
        if (!get_work_item(&item)) {
            break; // Shutdown or no more items
        }
        
        printf("Consumer %d: Processing item %d (priority %d, resources: %d)\n",
               consumer_id, item.item_id, item.priority, item.resource_count);
        
        // Acquire required resources
        if (!acquire_resources(consumer_id, item.required_resources, 
                              item.resource_count)) {
            printf("Consumer %d: Failed to acquire resources for item %d\n",
                   consumer_id, item.item_id);
            continue;
        }
        
        // Simulate processing
        usleep((rand() % 2000) + 1000);
        
        // Release resources
        release_resources(consumer_id, item.required_resources, 
                         item.resource_count);
        
        pthread_mutex_lock(&g_stats_mutex);
        g_items_consumed++;
        pthread_mutex_unlock(&g_stats_mutex);
        
        printf("Consumer %d: Completed item %d\n", consumer_id, item.item_id);
    }
    
    return NULL;
}

// Deadlock detection thread
void* deadlock_detector(void* arg) {
    while (!g_shutdown_requested) {
        sleep(5); // Check every 5 seconds
        
        // IMPLEMENT: Deadlock detection algorithm
        // Check for circular wait conditions
        
        printf("Deadlock detector: Checking for deadlocks...\n");
        
        // Check if any consumers are waiting too long
        for (int i = 0; i < MAX_CONSUMERS; i++) {
            if (g_resources.consumer_waiting[i]) {
                printf("WARNING: Consumer %d has been waiting for resources\n", i);
            }
        }
    }
    
    return NULL;
}

int main() {
    srand(time(NULL));
    
    init_systems();
    
    pthread_t producers[MAX_PRODUCERS];
    pthread_t consumers[MAX_CONSUMERS];
    pthread_t detector_thread;
    int producer_ids[MAX_PRODUCERS];
    int consumer_ids[MAX_CONSUMERS];
    
    // Start deadlock detector
    pthread_create(&detector_thread, NULL, deadlock_detector, NULL);
    
    // Start producers
    for (int i = 0; i < MAX_PRODUCERS; i++) {
        producer_ids[i] = i;
        pthread_create(&producers[i], NULL, producer_thread, &producer_ids[i]);
    }
    
    // Start consumers
    for (int i = 0; i < MAX_CONSUMERS; i++) {
        consumer_ids[i] = i;
        pthread_create(&consumers[i], NULL, consumer_thread, &consumer_ids[i]);
    }
    
    // Wait for producers to finish
    for (int i = 0; i < MAX_PRODUCERS; i++) {
        pthread_join(producers[i], NULL);
    }
    
    // Give consumers time to finish remaining work
    sleep(5);
    
    g_shutdown_requested = true;
    
    // Wake up waiting threads
    pthread_cond_broadcast(&g_work_queue.not_empty);
    pthread_cond_broadcast(&g_work_queue.not_full);
    
    // Wait for consumers to finish
    for (int i = 0; i < MAX_CONSUMERS; i++) {
        pthread_join(consumers[i], NULL);
    }
    
    pthread_join(detector_thread, NULL);
    
    printf("\n=== FINAL STATISTICS ===\n");
    printf("Items produced: %d\n", g_items_produced);
    printf("Items consumed: %d\n", g_items_consumed);
    
    return 0;
}
```

**Your Task:**
1. Identify and fix the deadlock scenarios in resource acquisition
2. Implement proper lock ordering to prevent deadlocks
3. Complete the deadlock detection algorithm
4. Add timeout mechanisms for resource acquisition
5. Implement priority-based work item processing
6. Add comprehensive error handling and recovery
7. Implement a resource allocation graph for visualization

### Exercise 3: Lock-Free Algorithm Bug Hunt

**Difficulty: Expert**
**Estimated Time: 90 minutes**

This exercise involves debugging a lock-free queue implementation with subtle memory ordering and ABA problem issues.

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>

// Lock-free queue node
typedef struct queue_node {
    atomic_int value;
    atomic(struct queue_node*) next;
} queue_node_t;

// Lock-free queue
typedef struct {
    atomic(queue_node_t*) head;
    atomic(queue_node_t*) tail;
    atomic_uint size;
    atomic_uint enqueue_count;
    atomic_uint dequeue_count;
} lockfree_queue_t;

// Memory reclamation - hazard pointer implementation
#define MAX_THREADS 16
#define MAX_HAZARD_POINTERS 2

typedef struct {
    atomic(void*) hazard_ptrs[MAX_HAZARD_POINTERS];
    queue_node_t* retired_list;
    int retired_count;
} hazard_pointer_record_t;

static hazard_pointer_record_t hazard_records[MAX_THREADS];
static atomic_int thread_counter = ATOMIC_VAR_INIT(0);
static __thread int thread_id = -1;

// Get thread-local hazard pointer record
hazard_pointer_record_t* get_hazard_record() {
    if (thread_id == -1) {
        thread_id = atomic_fetch_add(&thread_counter, 1);
        assert(thread_id < MAX_THREADS);
        
        // Initialize hazard pointers
        for (int i = 0; i < MAX_HAZARD_POINTERS; i++) {
            atomic_store(&hazard_records[thread_id].hazard_ptrs[i], NULL);
        }
        hazard_records[thread_id].retired_list = NULL;
        hazard_records[thread_id].retired_count = 0;
    }
    
    return &hazard_records[thread_id];
}

// Set hazard pointer
void set_hazard_pointer(int index, void* ptr) {
    hazard_pointer_record_t* record = get_hazard_record();
    atomic_store(&record->hazard_ptrs[index], ptr);
}

// Check if pointer is protected by any hazard pointer
bool is_protected(void* ptr) {
    for (int i = 0; i < MAX_THREADS; i++) {
        for (int j = 0; j < MAX_HAZARD_POINTERS; j++) {
            if (atomic_load(&hazard_records[i].hazard_ptrs[j]) == ptr) {
                return true;
            }
        }
    }
    return false;
}

// Retire a pointer for later deletion
void retire_pointer(queue_node_t* ptr) {
    hazard_pointer_record_t* record = get_hazard_record();
    
    // Add to retired list
    ptr->next = (atomic(struct queue_node*))(record->retired_list);
    record->retired_list = ptr;
    record->retired_count++;
    
    // Clean up retired pointers periodically
    if (record->retired_count >= 10) {
        queue_node_t* current = record->retired_list;
        queue_node_t* prev = NULL;
        
        while (current) {
            queue_node_t* next = (queue_node_t*)atomic_load(&current->next);
            
            if (!is_protected(current)) {
                // Safe to delete
                if (prev) {
                    prev->next = (atomic(struct queue_node*))next;
                } else {
                    record->retired_list = next;
                }
                
                free(current);
                record->retired_count--;
                current = next;
            } else {
                prev = current;
                current = next;
            }
        }
    }
}

// Initialize lock-free queue
void init_lockfree_queue(lockfree_queue_t* queue) {
    queue_node_t* dummy = malloc(sizeof(queue_node_t));
    atomic_store(&dummy->value, 0);
    atomic_store(&dummy->next, NULL);
    
    atomic_store(&queue->head, dummy);
    atomic_store(&queue->tail, dummy);
    atomic_store(&queue->size, 0);
    atomic_store(&queue->enqueue_count, 0);
    atomic_store(&queue->dequeue_count, 0);
}

// BUGGY: Lock-free enqueue operation
bool enqueue(lockfree_queue_t* queue, int value) {
    queue_node_t* new_node = malloc(sizeof(queue_node_t));
    if (!new_node) return false;
    
    atomic_store(&new_node->value, value);
    atomic_store(&new_node->next, NULL);
    
    while (true) {
        queue_node_t* tail = atomic_load(&queue->tail);
        
        // ISSUE 1: Missing hazard pointer protection
        queue_node_t* next = atomic_load(&tail->next);
        
        // ISSUE 2: ABA problem - tail might have changed
        if (tail == atomic_load(&queue->tail)) {
            if (next == NULL) {
                // ISSUE 3: Memory ordering - might not be strong enough
                if (atomic_compare_exchange_weak(&tail->next, &next, new_node)) {
                    // ISSUE 4: Tail update might fail, causing inconsistent state
                    atomic_compare_exchange_weak(&queue->tail, &tail, new_node);
                    break;
                }
            } else {
                // Help move tail forward
                atomic_compare_exchange_weak(&queue->tail, &tail, next);
            }
        }
    }
    
    atomic_fetch_add(&queue->size, 1);
    atomic_fetch_add(&queue->enqueue_count, 1);
    return true;
}

// BUGGY: Lock-free dequeue operation
bool dequeue(lockfree_queue_t* queue, int* value) {
    while (true) {
        queue_node_t* head = atomic_load(&queue->head);
        queue_node_t* tail = atomic_load(&queue->tail);
        
        // ISSUE 5: Missing hazard pointer protection
        queue_node_t* next = atomic_load(&head->next);
        
        // ISSUE 6: Inconsistent snapshot - head/tail/next might not be consistent
        if (head == atomic_load(&queue->head)) {
            if (head == tail) {
                if (next == NULL) {
                    // Queue is empty
                    return false;
                }
                
                // Help move tail forward
                atomic_compare_exchange_weak(&queue->tail, &tail, next);
            } else {
                if (next == NULL) {
                    // Inconsistent state - should not happen
                    continue;
                }
                
                // ISSUE 7: Reading value before ensuring node won't be freed
                *value = atomic_load(&next->value);
                
                // ISSUE 8: Memory ordering issues
                if (atomic_compare_exchange_weak(&queue->head, &head, next)) {
                    // ISSUE 9: Immediate free without hazard pointer cleanup
                    free(head);
                    break;
                }
            }
        }
    }
    
    atomic_fetch_sub(&queue->size, 1);
    atomic_fetch_add(&queue->dequeue_count, 1);
    return true;
}

// Queue size (unreliable due to race conditions)
int queue_size(lockfree_queue_t* queue) {
    // ISSUE 10: Size counter is not atomic with operations
    return atomic_load(&queue->size);
}

// Print queue statistics
void print_queue_stats(lockfree_queue_t* queue) {
    printf("Queue size: %d\n", atomic_load(&queue->size));
    printf("Enqueue count: %d\n", atomic_load(&queue->enqueue_count));
    printf("Dequeue count: %d\n", atomic_load(&queue->dequeue_count));
}

// Test thread function
void* test_thread(void* arg) {
    lockfree_queue_t* queue = (lockfree_queue_t*)arg;
    int thread_num = thread_id;
    
    // Mix of enqueue and dequeue operations
    for (int i = 0; i < 10000; i++) {
        if (i % 3 == 0) {
            // Enqueue
            int value = thread_num * 10000 + i;
            if (!enqueue(queue, value)) {
                printf("Thread %d: Enqueue failed for value %d\n", thread_num, value);
            }
        } else {
            // Dequeue
            int value;
            if (dequeue(queue, &value)) {
                // Verify value makes sense
                if (value < 0) {
                    printf("Thread %d: Got negative value %d\n", thread_num, value);
                }
            }
        }
        
        // Occasional size check
        if (i % 1000 == 0) {
            int size = queue_size(queue);
            printf("Thread %d: Queue size at iteration %d: %d\n", 
                   thread_num, i, size);
        }
    }
    
    return NULL;
}

int main() {
    lockfree_queue_t queue;
    init_lockfree_queue(&queue);
    
    const int num_threads = 8;
    pthread_t threads[num_threads];
    
    printf("Starting lock-free queue test with %d threads...\n", num_threads);
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, test_thread, &queue);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    print_queue_stats(&queue);
    
    return 0;
}
```

**Your Task:**
1. Fix the ABA problems in both enqueue and dequeue operations
2. Implement proper hazard pointer protection
3. Fix memory ordering issues (choose appropriate memory_order parameters)
4. Implement safe memory reclamation
5. Fix the queue size tracking issues
6. Add comprehensive validation and testing
7. Implement a stress test that can reliably expose the bugs

**Expected Issues to Fix:**
- ABA problems in pointer manipulation
- Missing hazard pointer protection
- Insufficient memory ordering
- Use-after-free vulnerabilities
- Inconsistent queue size tracking
- Race conditions in memory reclamation

### Exercise 4: Real-World Debugging Scenario

**Difficulty: Expert**
**Estimated Time: 120 minutes**

You've been called in to debug a production web server that's experiencing intermittent crashes, deadlocks, and data corruption. The server handles HTTP requests using a thread pool and maintains various caches and statistics.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <time.h>
#include <signal.h>
#include <errno.h>
#include <stdatomic.h>

#define MAX_THREADS 50
#define MAX_CLIENTS 1000
#define BUFFER_SIZE 4096
#define CACHE_SIZE 1000
#define MAX_REQUESTS 10000

// HTTP request structure
typedef struct {
    int client_fd;
    char* request_data;
    size_t request_size;
    time_t arrival_time;
    int request_id;
} http_request_t;

// Cache entry
typedef struct cache_entry {
    char* key;
    char* data;
    size_t data_size;
    time_t expiry_time;
    int access_count;
    struct cache_entry* next;
    pthread_mutex_t entry_mutex; // Per-entry locking
} cache_entry_t;

// Thread pool worker
typedef struct {
    pthread_t thread;
    int worker_id;
    bool is_busy;
    http_request_t* current_request;
    pthread_mutex_t worker_mutex;
} thread_worker_t;

// Server statistics
typedef struct {
    atomic_uint total_requests;
    atomic_uint active_connections;
    atomic_uint cache_hits;
    atomic_uint cache_misses;
    atomic_uint errors;
    
    // ISSUE: Non-atomic operations on shared data
    double average_response_time;
    double total_response_time;
    int response_count;
    
    pthread_mutex_t stats_mutex; // Inconsistent usage
} server_stats_t;

// Global server state
typedef struct {
    // Thread pool
    thread_worker_t workers[MAX_THREADS];
    int active_workers;
    pthread_mutex_t pool_mutex;
    pthread_cond_t work_available;
    
    // Request queue
    http_request_t* request_queue[MAX_REQUESTS];
    int queue_head;
    int queue_tail;
    int queue_size;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_not_empty;
    pthread_cond_t queue_not_full;
    
    // Cache system
    cache_entry_t* cache_buckets[CACHE_SIZE];
    pthread_mutex_t cache_mutex; // Global cache lock - scalability issue
    
    // Server socket
    int server_socket;
    bool shutdown_requested;
    
    // Statistics
    server_stats_t stats;
    
} web_server_t;

static web_server_t g_server = {0};

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    printf("Shutdown signal received, shutting down gracefully...\n");
    g_server.shutdown_requested = true;
}

// Hash function for cache
unsigned int hash_key(const char* key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % CACHE_SIZE;
}

// BUGGY: Cache operations with multiple race conditions
cache_entry_t* cache_get(const char* key) {
    unsigned int bucket = hash_key(key);
    
    // ISSUE 1: Inconsistent locking - sometimes locks, sometimes doesn't
    if (rand() % 10 > 7) { // Simulate inconsistent locking
        pthread_mutex_lock(&g_server.cache_mutex);
    }
    
    cache_entry_t* entry = g_server.cache_buckets[bucket];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            if (time(NULL) < entry->expiry_time) {
                // ISSUE 2: Race condition - entry might be freed by another thread
                entry->access_count++; // Unprotected increment
                
                atomic_fetch_add(&g_server.stats.cache_hits, 1);
                
                // ISSUE 3: Inconsistent unlocking
                if (rand() % 10 > 7) {
                    pthread_mutex_unlock(&g_server.cache_mutex);
                }
                return entry;
            } else {
                // Expired entry - should remove it
                // ISSUE 4: Not removing expired entries properly
                break;
            }
        }
        entry = entry->next;
    }
    
    atomic_fetch_add(&g_server.stats.cache_misses, 1);
    
    if (rand() % 10 > 7) {
        pthread_mutex_unlock(&g_server.cache_mutex);
    }
    return NULL;
}

void cache_put(const char* key, const char* data, size_t data_size, int ttl) {
    unsigned int bucket = hash_key(key);
    
    pthread_mutex_lock(&g_server.cache_mutex);
    
    // ISSUE 5: Memory leaks - not checking for existing entries
    cache_entry_t* new_entry = malloc(sizeof(cache_entry_t));
    new_entry->key = strdup(key);
    new_entry->data = malloc(data_size);
    memcpy(new_entry->data, data, data_size);
    new_entry->data_size = data_size;
    new_entry->expiry_time = time(NULL) + ttl;
    new_entry->access_count = 0;
    pthread_mutex_init(&new_entry->entry_mutex, NULL);
    
    // ISSUE 6: Race condition in linked list manipulation
    new_entry->next = g_server.cache_buckets[bucket];
    g_server.cache_buckets[bucket] = new_entry;
    
    pthread_mutex_unlock(&g_server.cache_mutex);
}

// BUGGY: Request queue operations
int enqueue_request(http_request_t* request) {
    pthread_mutex_lock(&g_server.queue_mutex);
    
    while (g_server.queue_size >= MAX_REQUESTS) {
        // ISSUE 7: Potential deadlock - waiting while holding lock
        pthread_cond_wait(&g_server.queue_not_full, &g_server.queue_mutex);
    }
    
    // ISSUE 8: Buffer overflow risk
    g_server.request_queue[g_server.queue_tail] = request;
    g_server.queue_tail = (g_server.queue_tail + 1) % MAX_REQUESTS;
    g_server.queue_size++;
    
    pthread_cond_signal(&g_server.queue_not_empty);
    pthread_mutex_unlock(&g_server.queue_mutex);
    
    return 1;
}

http_request_t* dequeue_request() {
    pthread_mutex_lock(&g_server.queue_mutex);
    
    while (g_server.queue_size == 0 && !g_server.shutdown_requested) {
        pthread_cond_wait(&g_server.queue_not_empty, &g_server.queue_mutex);
    }
    
    if (g_server.queue_size == 0) {
        pthread_mutex_unlock(&g_server.queue_mutex);
        return NULL;
    }
    
    http_request_t* request = g_server.request_queue[g_server.queue_head];
    g_server.queue_head = (g_server.queue_head + 1) % MAX_REQUESTS;
    g_server.queue_size--;
    
    pthread_cond_signal(&g_server.queue_not_full);
    pthread_mutex_unlock(&g_server.queue_mutex);
    
    return request;
}

// BUGGY: Process HTTP request
void process_request(http_request_t* request) {
    time_t start_time = time(NULL);
    
    // Simulate request processing
    char response_buffer[BUFFER_SIZE];
    
    // Try cache first
    cache_entry_t* cached = cache_get(request->request_data);
    if (cached) {
        // Use cached data
        snprintf(response_buffer, sizeof(response_buffer),
                "HTTP/1.1 200 OK\r\nContent-Length: %zu\r\n\r\n%s",
                cached->data_size, cached->data);
    } else {
        // Generate response (simulate expensive operation)
        usleep((rand() % 100) * 1000); // 0-100ms delay
        
        char* response_data = "Hello World - Generated Response";
        snprintf(response_buffer, sizeof(response_buffer),
                "HTTP/1.1 200 OK\r\nContent-Length: %zu\r\n\r\n%s",
                strlen(response_data), response_data);
        
        // Cache the response
        cache_put(request->request_data, response_data, strlen(response_data), 300);
    }
    
    // Send response
    // ISSUE 9: Not checking write() return value
    write(request->client_fd, response_buffer, strlen(response_buffer));
    close(request->client_fd);
    
    // Update statistics
    time_t end_time = time(NULL);
    double response_time = difftime(end_time, start_time);
    
    // ISSUE 10: Race conditions in statistics calculation
    pthread_mutex_lock(&g_server.stats.stats_mutex);
    g_server.stats.total_response_time += response_time;
    g_server.stats.response_count++;
    g_server.stats.average_response_time = 
        g_server.stats.total_response_time / g_server.stats.response_count;
    pthread_mutex_unlock(&g_server.stats.stats_mutex);
    
    atomic_fetch_add(&g_server.stats.total_requests, 1);
    atomic_fetch_sub(&g_server.stats.active_connections, 1);
    
    // Free request
    free(request->request_data);
    free(request);
}

// Worker thread function
void* worker_thread(void* arg) {
    thread_worker_t* worker = (thread_worker_t*)arg;
    
    printf("Worker %d started\n", worker->worker_id);
    
    while (!g_server.shutdown_requested) {
        http_request_t* request = dequeue_request();
        if (!request) continue;
        
        // ISSUE 11: Race condition in worker state management
        worker->is_busy = true;
        worker->current_request = request;
        
        process_request(request);
        
        worker->is_busy = false;
        worker->current_request = NULL;
    }
    
    printf("Worker %d shutting down\n", worker->worker_id);
    return NULL;
}

// Initialize server
int init_server(int port) {
    memset(&g_server, 0, sizeof(g_server));
    
    // Initialize mutexes and conditions
    pthread_mutex_init(&g_server.pool_mutex, NULL);
    pthread_cond_init(&g_server.work_available, NULL);
    pthread_mutex_init(&g_server.queue_mutex, NULL);
    pthread_cond_init(&g_server.queue_not_empty, NULL);
    pthread_cond_init(&g_server.queue_not_full, NULL);
    pthread_mutex_init(&g_server.cache_mutex, NULL);
    pthread_mutex_init(&g_server.stats.stats_mutex, NULL);
    
    // Initialize workers
    for (int i = 0; i < MAX_THREADS; i++) {
        g_server.workers[i].worker_id = i;
        g_server.workers[i].is_busy = false;
        g_server.workers[i].current_request = NULL;
        pthread_mutex_init(&g_server.workers[i].worker_mutex, NULL);
        
        if (pthread_create(&g_server.workers[i].thread, NULL, 
                          worker_thread, &g_server.workers[i]) != 0) {
            fprintf(stderr, "Failed to create worker thread %d\n", i);
            return -1;
        }
    }
    
    // Setup server socket
    g_server.server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server.server_socket < 0) {
        perror("socket");
        return -1;
    }
    
    int opt = 1;
    setsockopt(g_server.server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    if (bind(g_server.server_socket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return -1;
    }
    
    if (listen(g_server.server_socket, 128) < 0) {
        perror("listen");
        return -1;
    }
    
    return 0;
}

// Print server statistics
void print_stats() {
    printf("\n=== SERVER STATISTICS ===\n");
    printf("Total requests: %u\n", atomic_load(&g_server.stats.total_requests));
    printf("Active connections: %u\n", atomic_load(&g_server.stats.active_connections));
    printf("Cache hits: %u\n", atomic_load(&g_server.stats.cache_hits));
    printf("Cache misses: %u\n", atomic_load(&g_server.stats.cache_misses));
    printf("Errors: %u\n", atomic_load(&g_server.stats.errors));
    
    pthread_mutex_lock(&g_server.stats.stats_mutex);
    printf("Average response time: %.3f seconds\n", g_server.stats.average_response_time);
    pthread_mutex_unlock(&g_server.stats.stats_mutex);
    
    // Worker status
    printf("\nWorker Status:\n");
    for (int i = 0; i < MAX_THREADS; i++) {
        printf("  Worker %d: %s\n", i, 
               g_server.workers[i].is_busy ? "BUSY" : "IDLE");
    }
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    int port = argc > 1 ? atoi(argv[1]) : 8080;
    
    if (init_server(port) < 0) {
        fprintf(stderr, "Failed to initialize server\n");
        return 1;
    }
    
    printf("Server listening on port %d\n", port);
    
    // Main accept loop
    while (!g_server.shutdown_requested) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_fd = accept(g_server.server_socket, 
                              (struct sockaddr*)&client_addr, &client_len);
        
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            atomic_fetch_add(&g_server.stats.errors, 1);
            continue;
        }
        
        // Create request
        http_request_t* request = malloc(sizeof(http_request_t));
        request->client_fd = client_fd;
        request->request_data = malloc(BUFFER_SIZE);
        
        // Read request data
        ssize_t bytes_read = read(client_fd, request->request_data, BUFFER_SIZE - 1);
        if (bytes_read <= 0) {
            free(request->request_data);
            free(request);
            close(client_fd);
            continue;
        }
        
        request->request_data[bytes_read] = '\0';
        request->request_size = bytes_read;
        request->arrival_time = time(NULL);
        request->request_id = atomic_fetch_add(&g_server.stats.total_requests, 1);
        
        atomic_fetch_add(&g_server.stats.active_connections, 1);
        
        // Enqueue request
        if (!enqueue_request(request)) {
            free(request->request_data);
            free(request);
            close(client_fd);
            atomic_fetch_add(&g_server.stats.errors, 1);
        }
        
        // Print stats periodically
        if (request->request_id % 100 == 0) {
            print_stats();
        }
    }
    
    // Cleanup
    close(g_server.server_socket);
    
    // Wait for workers to finish
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(g_server.workers[i].thread, NULL);
    }
    
    print_stats();
    printf("Server shutdown complete\n");
    
    return 0;
}
```

**Your Task:**
This is a comprehensive debugging challenge that requires you to:

1. **Identify and fix all race conditions** (there are over 15)
2. **Fix memory leaks and use-after-free vulnerabilities**
3. **Resolve deadlock scenarios**
4. **Implement proper error handling**
5. **Fix cache coherency issues**
6. **Optimize locking granularity**
7. **Add comprehensive logging and monitoring**
8. **Implement proper shutdown procedures**
9. **Add stress testing capabilities**
10. **Create a debugging and profiling framework**

**Areas to Focus On:**
- Cache system thread safety
- Request queue management
- Statistics calculation accuracy
- Worker thread coordination
- Memory management
- Signal handling
- Resource cleanup
- Performance bottlenecks

**Deliverables:**
1. Fixed source code with detailed comments explaining each fix
2. Test suite that can reproduce the original bugs
3. Performance benchmarking results before/after fixes
4. Documentation of debugging methodology used
5. Monitoring and alerting system for production deployment

### Self-Evaluation Rubric

After completing each exercise, evaluate your performance:

**Race Condition Detection (25 points)**
- Expert (23-25): Found all race conditions and understood root causes
- Advanced (18-22): Found most race conditions with minor oversights
- Intermediate (13-17): Found basic race conditions but missed subtle ones
- Beginner (0-12): Missed major race conditions or misidentified issues

**Deadlock Analysis (25 points)**
- Expert (23-25): Identified all deadlock scenarios and implemented proper prevention
- Advanced (18-22): Found most deadlocks with effective solutions
- Intermediate (13-17): Basic deadlock detection with some prevention measures
- Beginner (0-12): Missed deadlocks or implemented incorrect solutions

**Memory Safety (20 points)**
- Expert (18-20): Fixed all memory leaks, use-after-free, and buffer overflows
- Advanced (14-17): Fixed most memory issues with minor oversights
- Intermediate (10-13): Fixed basic memory problems
- Beginner (0-9): Significant memory safety issues remain

**Code Quality and Best Practices (15 points)**
- Expert (14-15): Clean, well-documented code following best practices
- Advanced (11-13): Good code quality with minor style issues
- Intermediate (8-10): Functional code with some quality issues
- Beginner (0-7): Poor code quality or incomplete solutions

**Testing and Validation (15 points)**
- Expert (14-15): Comprehensive test suite with edge cases
- Advanced (11-13): Good test coverage with basic validation
- Intermediate (8-10): Basic testing with limited scenarios
- Beginner (0-7): Minimal or incorrect testing

**Total Score: ___/100**

- **90-100**: Expert-level debugging skills
- **80-89**: Advanced debugging capabilities
- **70-79**: Intermediate debugging skills
- **60-69**: Basic debugging competency
- **Below 60**: Needs significant improvement

## Assessment

This comprehensive assessment evaluates your mastery of debugging multi-threaded applications across multiple dimensions. The assessment combines theoretical knowledge, practical skills, and real-world problem-solving abilities.

### Practical Debugging Assessment

#### Part A: Bug Identification and Classification (30 points)

**Time Limit: 60 minutes**

You will be presented with 5 different code samples, each containing multiple threading bugs. For each sample:

1. **Identify all bugs** (2 points per bug found)
2. **Classify each bug** by type (race condition, deadlock, etc.) (1 point per correct classification)
3. **Assess severity** (critical, major, minor) (1 point per correct assessment)
4. **Estimate impact** on system behavior (1 point per accurate assessment)

**Sample Assessment Question:**

```c
// Analyze this thread pool implementation
#include <pthread.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    void (*function)(void*);
    void* argument;
} task_t;

typedef struct {
    task_t* tasks;
    int head, tail, count, size;
    pthread_mutex_t mutex;
    pthread_cond_t notify;
    pthread_t* threads;
    int thread_count;
    bool shutdown;
} thread_pool_t;

thread_pool_t* pool_create(int thread_count, int queue_size) {
    thread_pool_t* pool = malloc(sizeof(thread_pool_t));
    
    pool->tasks = malloc(queue_size * sizeof(task_t));
    pool->thread_count = thread_count;
    pool->head = pool->tail = pool->count = 0;
    pool->size = queue_size;
    pool->shutdown = false;
    
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->notify, NULL);
    
    pool->threads = malloc(thread_count * sizeof(pthread_t));
    
    for (int i = 0; i < thread_count; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }
    
    return pool;
}

void* worker_thread(void* arg) {
    thread_pool_t* pool = (thread_pool_t*)arg;
    
    while (true) {
        pthread_mutex_lock(&pool->mutex);
        
        while (pool->count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->notify, &pool->mutex);
        }
        
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        
        task_t task = pool->tasks[pool->head];
        pool->head = (pool->head + 1) % pool->size;
        pool->count--;
        
        pthread_mutex_unlock(&pool->mutex);
        
        task.function(task.argument);
    }
    
    return NULL;
}

int pool_add_task(thread_pool_t* pool, void (*function)(void*), void* argument) {
    pthread_mutex_lock(&pool->mutex);
    
    if (pool->count == pool->size) {
        pthread_mutex_unlock(&pool->mutex);
        return -1; // Queue full
    }
    
    pool->tasks[pool->tail].function = function;
    pool->tasks[pool->tail].argument = argument;
    pool->tail = (pool->tail + 1) % pool->size;
    pool->count++;
    
    pthread_cond_signal(&pool->notify);
    pthread_mutex_unlock(&pool->mutex);
    
    return 0;
}

void pool_destroy(thread_pool_t* pool) {
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->notify);
    pthread_mutex_unlock(&pool->mutex);
    
    for (int i = 0; i < pool->thread_count; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    free(pool->tasks);
    free(pool->threads);
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->notify);
    free(pool);
}
```

**Required Analysis Format:**
```
Bug #1:
- Location: [file:line or function name]
- Type: [race condition/deadlock/resource leak/etc.]
- Severity: [critical/major/minor]
- Description: [detailed explanation]
- Impact: [potential consequences]
- Fix: [proposed solution]

Bug #2:
...
```

#### Part B: Debugging Tool Proficiency (25 points)

**Time Limit: 45 minutes**

Demonstrate proficiency with debugging tools by completing the following tasks:

1. **GDB Multi-threaded Debugging** (10 points)
   - Set up GDB for a multi-threaded program
   - Use thread-specific breakpoints
   - Analyze thread stack traces
   - Examine shared memory states

2. **ThreadSanitizer Usage** (8 points)
   - Compile a program with TSan
   - Interpret TSan output
   - Identify false positives
   - Configure suppression files

3. **Custom Tool Implementation** (7 points)
   - Implement a simple deadlock detector
   - Create a race condition monitor
   - Design a thread interaction tracer

**Sample Task:**
```bash
# Given this GDB session output, analyze the deadlock:

(gdb) info threads
  Id   Target Id         Frame
* 1    Thread 0x7ffff7fc4740 (LWP 12345) 0x7ffff7bc4ed6 pthread_mutex_lock
  2    Thread 0x7ffff6fc3700 (LWP 12346) 0x7ffff7bc4ed6 pthread_mutex_lock
  3    Thread 0x7ffff67c2700 (LWP 12347) 0x7ffff7bc4ed6 pthread_mutex_lock

(gdb) thread 1
(gdb) bt
#0  0x7ffff7bc4ed6 in pthread_mutex_lock () from /lib64/libpthread.so.0
#1  0x0000000000400b23 in transfer_funds (from=1, to=2, amount=100) at bank.c:45
#2  0x0000000000400c45 in worker_thread (arg=0x7fffffffe4ac) at bank.c:78

(gdb) thread 2
(gdb) bt
#0  0x7ffff7bc4ed6 in pthread_mutex_lock () from /lib64/libpthread.so.0
#1  0x0000000000400b23 in transfer_funds (from=2, to=1, amount=50) at bank.c:45
#2  0x0000000000400c45 in worker_thread (arg=0x7fffffffe4b0) at bank.c:78

# Questions:
# 1. What type of deadlock is this?
# 2. What is the root cause?
# 3. How would you fix it?
# 4. How could you have prevented it?
```

#### Part C: Real-World Problem Solving (25 points)

**Time Limit: 90 minutes**

You're given a production system experiencing the following symptoms:

- Intermittent application freezes
- Memory usage steadily increasing
- Occasional data corruption
- Performance degradation under load

**Your Task:**
1. **Develop a debugging strategy** (5 points)
2. **Identify likely root causes** (5 points)
3. **Design experiments to isolate issues** (5 points)
4. **Implement monitoring solutions** (5 points)
5. **Create a fix and validation plan** (5 points)

**Deliverables:**
- Debugging methodology document
- List of tools and techniques to use
- Step-by-step investigation plan
- Code for monitoring/debugging tools
- Test plan for validating fixes

#### Part D: Advanced Concepts (20 points)

**Time Limit: 30 minutes**

Answer questions on advanced debugging concepts:

1. **Memory Ordering and Race Conditions** (5 points)
   ```c
   // Explain why this code might fail and how to fix it
   static int flag = 0;
   static int data = 0;
   
   void writer() {
       data = 42;
       flag = 1;
   }
   
   void reader() {
       if (flag) {
           printf("Data: %d\n", data);
       }
   }
   ```

2. **Lock-Free Algorithm Debugging** (5 points)
   - Explain the ABA problem
   - Describe hazard pointer implementation
   - Discuss memory reclamation strategies

3. **Distributed System Debugging** (5 points)
   - Identify race conditions across network boundaries
   - Explain vector clocks for causality tracking
   - Describe distributed deadlock detection

4. **Performance Analysis** (5 points)
   - Analyze lock contention patterns
   - Identify false sharing scenarios
   - Optimize synchronization overhead

### Comprehensive Final Project

**Time Limit: 4 hours**

Build a complete debugging framework for a multi-threaded application. Your framework must include:

#### Requirements (100 points total)

1. **Thread Monitoring System** (25 points)
   - Real-time thread state tracking
   - Lock acquisition/release monitoring
   - Performance metrics collection
   - Deadlock detection algorithms

2. **Race Condition Detection** (25 points)
   - Memory access pattern analysis
   - Data race identification
   - Lockset algorithm implementation
   - Happens-before relationship tracking

3. **Logging and Visualization** (20 points)
   - High-performance thread-safe logging
   - Timeline visualization tools
   - Thread interaction graphs
   - Performance dashboard

4. **Automated Testing Framework** (15 points)
   - Stress testing capabilities
   - Bug reproduction tools
   - Regression testing suite
   - Coverage analysis

5. **Documentation and Usability** (15 points)
   - API documentation
   - User guides and tutorials
   - Example implementations
   - Best practices documentation

#### Sample Framework Architecture

```c
// Debugging Framework API
typedef struct debug_framework {
    // Thread monitoring
    thread_monitor_t* thread_monitor;
    lock_monitor_t* lock_monitor;
    
    // Race detection
    race_detector_t* race_detector;
    memory_tracker_t* memory_tracker;
    
    // Logging system
    advanced_logger_t* logger;
    trace_buffer_t* trace_buffer;
    
    // Visualization
    visualization_engine_t* viz_engine;
    
    // Configuration
    debug_config_t config;
} debug_framework_t;

// Initialize debugging framework
debug_framework_t* debug_init(const debug_config_t* config);

// Hook into application
void debug_register_thread(pthread_t thread, const char* name);
void debug_register_mutex(pthread_mutex_t* mutex, const char* name);
void debug_register_memory_region(void* ptr, size_t size, const char* name);

// Monitoring macros
#define DEBUG_LOCK(mutex) debug_lock_acquire(mutex, __FILE__, __LINE__)
#define DEBUG_UNLOCK(mutex) debug_lock_release(mutex, __FILE__, __LINE__)
#define DEBUG_MEMORY_ACCESS(ptr, size, type) debug_memory_access(ptr, size, type, __FILE__, __LINE__)

// Analysis functions
debug_report_t* debug_generate_report(debug_framework_t* framework);
void debug_export_timeline(debug_framework_t* framework, const char* filename);
void debug_visualize_interactions(debug_framework_t* framework);

// Cleanup
void debug_shutdown(debug_framework_t* framework);
```

### Evaluation Criteria

#### Technical Excellence (40%)
- **Correctness**: All bugs identified and fixed correctly
- **Completeness**: Comprehensive coverage of threading issues
- **Efficiency**: Solutions don't introduce performance penalties
- **Robustness**: Error handling and edge case coverage

#### Problem-Solving Approach (30%)
- **Methodology**: Systematic approach to debugging
- **Tool Usage**: Effective use of debugging tools
- **Root Cause Analysis**: Deep understanding of underlying issues
- **Prevention**: Proactive measures to prevent future bugs

#### Code Quality (20%)
- **Readability**: Clean, well-documented code
- **Maintainability**: Modular, extensible design
- **Best Practices**: Following industry standards
- **Testing**: Comprehensive test coverage

#### Communication (10%)
- **Documentation**: Clear explanations of solutions
- **Presentation**: Well-organized deliverables
- **Knowledge Transfer**: Ability to teach others
- **Collaboration**: Working effectively with team members

### Performance Standards

| Grade | Score Range | Description |
|-------|-------------|-------------|
| **A+** | 95-100 | Expert-level debugging skills, innovative solutions, exceptional documentation |
| **A** | 90-94 | Advanced debugging capabilities, comprehensive solutions, good documentation |
| **A-** | 85-89 | Strong debugging skills, complete solutions, adequate documentation |
| **B+** | 80-84 | Good debugging abilities, mostly complete solutions, basic documentation |
| **B** | 75-79 | Adequate debugging skills, partial solutions, minimal documentation |
| **B-** | 70-74 | Basic debugging competency, incomplete solutions, poor documentation |
| **C** | 60-69 | Minimal debugging skills, significant gaps in understanding |
| **F** | Below 60 | Inadequate debugging abilities, major deficiencies |

### Industry Readiness Indicators

After completing this assessment, you should be able to:

#### Entry Level (70-79%)
- Identify basic race conditions and deadlocks
- Use GDB for multi-threaded debugging
- Apply ThreadSanitizer effectively
- Implement simple synchronization solutions
- Write thread-safe code with guidance

#### Mid Level (80-89%)
- Debug complex multi-threaded systems
- Design custom debugging tools
- Analyze performance bottlenecks
- Implement lock-free algorithms
- Lead debugging efforts on small teams

#### Senior Level (90-94%)
- Architect debugging frameworks
- Debug distributed systems
- Optimize high-performance concurrent systems
- Mentor junior developers
- Drive debugging best practices across organization

#### Expert Level (95-100%)
- Research and develop new debugging techniques
- Design debugging tools for entire industry
- Solve previously unsolvable concurrency problems
- Publish papers on debugging methodologies
- Establish new industry standards

### Continuous Learning Path

1. **Advanced Topics to Explore:**
   - Formal verification of concurrent algorithms
   - Model checking for deadlock detection
   - Distributed systems debugging
   - GPU/CUDA debugging techniques
   - Real-time systems debugging

2. **Industry Certifications:**
   - Intel Threading Building Blocks certification
   - CUDA programming certification
   - Linux kernel development certification
   - Embedded systems debugging certification

3. **Open Source Contributions:**
   - Contribute to debugging tools (GDB, Valgrind, etc.)
   - Develop threading libraries
   - Create educational resources
   - Participate in standards committees

4. **Professional Development:**
   - Attend debugging and concurrency conferences
   - Join professional debugging communities
   - Mentor others in debugging techniques
   - Publish debugging case studies

### Resources for Further Study

#### Books
- "The Art of Multiprocessor Programming" by Herlihy & Shavit
- "Java Concurrency in Practice" by Goetz et al. (concepts apply to C/C++)
- "Programming with POSIX Threads" by Butenhof
- "Debugging Applications" by Robbins

#### Tools and Frameworks
- Intel Inspector for threading error detection
- IBM Thread and Memory Debugger
- TotalView for HPC debugging
- ARM Development Studio for embedded debugging

#### Online Resources
- Intel Threading Building Blocks documentation
- LLVM ThreadSanitizer documentation
- Linux kernel debugging guides
- Real-Time Systems debugging resources

This comprehensive assessment ensures that you have mastered not just the technical aspects of debugging multi-threaded applications, but also the problem-solving methodologies, tool usage, and professional practices that are essential for success in industry-level concurrent programming projects.

## Next Section
[Project Examples](Projects/)
