# Project 3: Producer-Consumer Queue

## Objective
Implement a producer-consumer queue using condition variables to handle multiple producers and consumers efficiently. This project demonstrates advanced synchronization techniques and inter-thread communication.

## Requirements

### Basic Requirements
1. Implement a bounded buffer (circular queue) with fixed capacity
2. Support multiple producer and consumer threads
3. Use condition variables for efficient blocking/signaling
4. Handle queue full and empty conditions gracefully
5. Provide proper error handling and resource cleanup

### Advanced Requirements
1. Add priority queue functionality
2. Implement work stealing between multiple queues
3. Add statistics collection and monitoring
4. Support different queue policies (FIFO, LIFO, Priority)
5. Implement timeout mechanisms for enqueue/dequeue operations

## Implementation Guide

### Basic Producer-Consumer Queue

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#define DEFAULT_QUEUE_SIZE 10

typedef struct {
    int item_id;
    char data[64];
    time_t timestamp;
    int priority; // For advanced version
} WorkItem;

typedef struct {
    WorkItem* buffer;
    int capacity;
    int size;
    int head;  // Next position to read from
    int tail;  // Next position to write to
    
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    
    // Statistics
    long total_produced;
    long total_consumed;
    long max_size_reached;
    bool shutdown;
} ProducerConsumerQueue;

// Create a new queue
ProducerConsumerQueue* pc_queue_create(int capacity) {
    if (capacity <= 0) {
        capacity = DEFAULT_QUEUE_SIZE;
    }
    
    ProducerConsumerQueue* queue = malloc(sizeof(ProducerConsumerQueue));
    if (!queue) {
        return NULL;
    }
    
    queue->buffer = malloc(sizeof(WorkItem) * capacity);
    if (!queue->buffer) {
        free(queue);
        return NULL;
    }
    
    queue->capacity = capacity;
    queue->size = 0;
    queue->head = 0;
    queue->tail = 0;
    queue->total_produced = 0;
    queue->total_consumed = 0;
    queue->max_size_reached = 0;
    queue->shutdown = false;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue->buffer);
        free(queue);
        return NULL;
    }
    
    if (pthread_cond_init(&queue->not_empty, NULL) != 0) {
        pthread_mutex_destroy(&queue->mutex);
        free(queue->buffer);
        free(queue);
        return NULL;
    }
    
    if (pthread_cond_init(&queue->not_full, NULL) != 0) {
        pthread_cond_destroy(&queue->not_empty);
        pthread_mutex_destroy(&queue->mutex);
        free(queue->buffer);
        free(queue);
        return NULL;
    }
    
    return queue;
}

// Enqueue an item (blocking)
bool pc_queue_enqueue(ProducerConsumerQueue* queue, const WorkItem* item) {
    if (!queue || !item) {
        return false;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    // Wait while queue is full and not shutting down
    while (queue->size >= queue->capacity && !queue->shutdown) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    if (queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    // Add item to queue
    queue->buffer[queue->tail] = *item;
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->size++;
    queue->total_produced++;
    
    if (queue->size > queue->max_size_reached) {
        queue->max_size_reached = queue->size;
    }
    
    // Signal waiting consumers
    pthread_cond_signal(&queue->not_empty);
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

// Dequeue an item (blocking)
bool pc_queue_dequeue(ProducerConsumerQueue* queue, WorkItem* item) {
    if (!queue || !item) {
        return false;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    // Wait while queue is empty and not shutting down
    while (queue->size == 0 && !queue->shutdown) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }
    
    if (queue->size == 0 && queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    // Remove item from queue
    *item = queue->buffer[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    queue->total_consumed++;
    
    // Signal waiting producers
    pthread_cond_signal(&queue->not_full);
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

// Try to enqueue with timeout
bool pc_queue_enqueue_timeout(ProducerConsumerQueue* queue, const WorkItem* item, int timeout_ms) {
    if (!queue || !item) {
        return false;
    }
    
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_ms / 1000;
    timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
    
    // Handle nanosecond overflow
    if (timeout.tv_nsec >= 1000000000) {
        timeout.tv_sec++;
        timeout.tv_nsec -= 1000000000;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size >= queue->capacity && !queue->shutdown) {
        int result = pthread_cond_timedwait(&queue->not_full, &queue->mutex, &timeout);
        if (result == ETIMEDOUT) {
            pthread_mutex_unlock(&queue->mutex);
            return false;
        }
    }
    
    if (queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    // Add item to queue
    queue->buffer[queue->tail] = *item;
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->size++;
    queue->total_produced++;
    
    if (queue->size > queue->max_size_reached) {
        queue->max_size_reached = queue->size;
    }
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

// Try to dequeue with timeout
bool pc_queue_dequeue_timeout(ProducerConsumerQueue* queue, WorkItem* item, int timeout_ms) {
    if (!queue || !item) {
        return false;
    }
    
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_ms / 1000;
    timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
    
    if (timeout.tv_nsec >= 1000000000) {
        timeout.tv_sec++;
        timeout.tv_nsec -= 1000000000;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size == 0 && !queue->shutdown) {
        int result = pthread_cond_timedwait(&queue->not_empty, &queue->mutex, &timeout);
        if (result == ETIMEDOUT) {
            pthread_mutex_unlock(&queue->mutex);
            return false;
        }
    }
    
    if (queue->size == 0 && queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    *item = queue->buffer[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->size--;
    queue->total_consumed++;
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

// Get queue statistics
typedef struct {
    int current_size;
    int capacity;
    long total_produced;
    long total_consumed;
    long max_size_reached;
    bool is_shutdown;
} QueueStats;

QueueStats pc_queue_get_stats(ProducerConsumerQueue* queue) {
    QueueStats stats = {0};
    
    if (!queue) {
        return stats;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    stats.current_size = queue->size;
    stats.capacity = queue->capacity;
    stats.total_produced = queue->total_produced;
    stats.total_consumed = queue->total_consumed;
    stats.max_size_reached = queue->max_size_reached;
    stats.is_shutdown = queue->shutdown;
    
    pthread_mutex_unlock(&queue->mutex);
    
    return stats;
}

// Shutdown the queue
void pc_queue_shutdown(ProducerConsumerQueue* queue) {
    if (!queue) return;
    
    pthread_mutex_lock(&queue->mutex);
    queue->shutdown = true;
    pthread_cond_broadcast(&queue->not_empty);
    pthread_cond_broadcast(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
}

// Destroy the queue
void pc_queue_destroy(ProducerConsumerQueue* queue) {
    if (!queue) return;
    
    pc_queue_shutdown(queue);
    
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
    
    free(queue->buffer);
    free(queue);
}

// Print queue statistics
void pc_queue_print_stats(ProducerConsumerQueue* queue) {
    QueueStats stats = pc_queue_get_stats(queue);
    
    printf("Queue Statistics:\n");
    printf("  Current Size: %d/%d\n", stats.current_size, stats.capacity);
    printf("  Total Produced: %ld\n", stats.total_produced);
    printf("  Total Consumed: %ld\n", stats.total_consumed);
    printf("  Max Size Reached: %ld\n", stats.max_size_reached);
    printf("  Pending Items: %ld\n", stats.total_produced - stats.total_consumed);
    printf("  Shutdown: %s\n", stats.is_shutdown ? "Yes" : "No");
}
```

### Producer and Consumer Threads

```c
typedef struct {
    ProducerConsumerQueue* queue;
    int producer_id;
    int items_to_produce;
    int production_delay_ms;
    int items_produced;
} ProducerData;

typedef struct {
    ProducerConsumerQueue* queue;
    int consumer_id;
    int consumption_delay_ms;
    int items_consumed;
    bool should_stop;
} ConsumerData;

void* producer_thread(void* arg) {
    ProducerData* data = (ProducerData*)arg;
    
    printf("Producer %d starting (will produce %d items)\n", 
           data->producer_id, data->items_to_produce);
    
    for (int i = 0; i < data->items_to_produce; i++) {
        WorkItem item;
        item.item_id = data->producer_id * 1000 + i;
        item.timestamp = time(NULL);
        item.priority = rand() % 10; // Random priority 0-9
        
        snprintf(item.data, sizeof(item.data), 
                "Item from Producer %d, seq %d", data->producer_id, i);
        
        if (pc_queue_enqueue(data->queue, &item)) {
            data->items_produced++;
            printf("Producer %d: Produced item %d\n", data->producer_id, item.item_id);
        } else {
            printf("Producer %d: Failed to produce item %d (queue shutdown?)\n", 
                   data->producer_id, item.item_id);
            break;
        }
        
        if (data->production_delay_ms > 0) {
            usleep(data->production_delay_ms * 1000);
        }
    }
    
    printf("Producer %d finished (produced %d items)\n", 
           data->producer_id, data->items_produced);
    
    return NULL;
}

void* consumer_thread(void* arg) {
    ConsumerData* data = (ConsumerData*)arg;
    
    printf("Consumer %d starting\n", data->consumer_id);
    
    while (!data->should_stop) {
        WorkItem item;
        
        if (pc_queue_dequeue_timeout(data->queue, &item, 1000)) { // 1 second timeout
            data->items_consumed++;
            
            time_t now = time(NULL);
            int age_seconds = (int)(now - item.timestamp);
            
            printf("Consumer %d: Consumed item %d (age: %ds, priority: %d) - %s\n", 
                   data->consumer_id, item.item_id, age_seconds, 
                   item.priority, item.data);
            
            if (data->consumption_delay_ms > 0) {
                usleep(data->consumption_delay_ms * 1000);
            }
        } else {
            // Timeout occurred, check if we should continue
            QueueStats stats = pc_queue_get_stats(data->queue);
            if (stats.is_shutdown && stats.current_size == 0) {
                break;
            }
        }
    }
    
    printf("Consumer %d finished (consumed %d items)\n", 
           data->consumer_id, data->items_consumed);
    
    return NULL;
}
```

### Test Framework

```c
#define MAX_PRODUCERS 5
#define MAX_CONSUMERS 5

typedef struct {
    int num_producers;
    int num_consumers;
    int items_per_producer;
    int queue_capacity;
    int production_delay_ms;
    int consumption_delay_ms;
    int test_duration_seconds;
} TestConfig;

void run_producer_consumer_test(TestConfig* config) {
    printf("\n=== Producer-Consumer Test ===\n");
    printf("Producers: %d, Consumers: %d\n", config->num_producers, config->num_consumers);
    printf("Items per producer: %d\n", config->items_per_producer);
    printf("Queue capacity: %d\n", config->queue_capacity);
    printf("Production delay: %dms, Consumption delay: %dms\n", 
           config->production_delay_ms, config->consumption_delay_ms);
    
    // Create queue
    ProducerConsumerQueue* queue = pc_queue_create(config->queue_capacity);
    if (!queue) {
        printf("Failed to create queue\n");
        return;
    }
    
    // Create producer and consumer data
    pthread_t producer_threads[MAX_PRODUCERS];
    pthread_t consumer_threads[MAX_CONSUMERS];
    ProducerData producer_data[MAX_PRODUCERS];
    ConsumerData consumer_data[MAX_CONSUMERS];
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Start consumers first
    for (int i = 0; i < config->num_consumers; i++) {
        consumer_data[i].queue = queue;
        consumer_data[i].consumer_id = i + 1;
        consumer_data[i].consumption_delay_ms = config->consumption_delay_ms;
        consumer_data[i].items_consumed = 0;
        consumer_data[i].should_stop = false;
        
        pthread_create(&consumer_threads[i], NULL, consumer_thread, &consumer_data[i]);
    }
    
    // Start producers
    for (int i = 0; i < config->num_producers; i++) {
        producer_data[i].queue = queue;
        producer_data[i].producer_id = i + 1;
        producer_data[i].items_to_produce = config->items_per_producer;
        producer_data[i].production_delay_ms = config->production_delay_ms;
        producer_data[i].items_produced = 0;
        
        pthread_create(&producer_threads[i], NULL, producer_thread, &producer_data[i]);
    }
    
    // Monitor progress
    for (int i = 0; i < config->test_duration_seconds; i++) {
        sleep(1);
        printf("\n--- Progress after %d seconds ---\n", i + 1);
        pc_queue_print_stats(queue);
    }
    
    // Wait for all producers to finish
    for (int i = 0; i < config->num_producers; i++) {
        pthread_join(producer_threads[i], NULL);
    }
    
    printf("\nAll producers finished. Waiting for consumers to drain queue...\n");
    
    // Wait a bit for consumers to finish processing
    sleep(2);
    
    // Shutdown queue and stop consumers
    pc_queue_shutdown(queue);
    
    for (int i = 0; i < config->num_consumers; i++) {
        consumer_data[i].should_stop = true;
        pthread_join(consumer_threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    // Calculate results
    int total_produced = 0, total_consumed = 0;
    for (int i = 0; i < config->num_producers; i++) {
        total_produced += producer_data[i].items_produced;
    }
    for (int i = 0; i < config->num_consumers; i++) {
        total_consumed += consumer_data[i].items_consumed;
    }
    
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                         (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    printf("\n=== Test Results ===\n");
    printf("Total items produced: %d\n", total_produced);
    printf("Total items consumed: %d\n", total_consumed);
    printf("Items remaining in queue: %d\n", total_produced - total_consumed);
    printf("Total test time: %.2f seconds\n", elapsed_time);
    printf("Production rate: %.2f items/sec\n", total_produced / elapsed_time);
    printf("Consumption rate: %.2f items/sec\n", total_consumed / elapsed_time);
    
    pc_queue_print_stats(queue);
    pc_queue_destroy(queue);
}

int main() {
    srand(time(NULL));
    
    // Test 1: Balanced producers and consumers
    TestConfig config1 = {
        .num_producers = 3,
        .num_consumers = 3,
        .items_per_producer = 50,
        .queue_capacity = 10,
        .production_delay_ms = 100,
        .consumption_delay_ms = 150,
        .test_duration_seconds = 10
    };
    
    run_producer_consumer_test(&config1);
    
    // Test 2: More producers than consumers (backpressure test)
    TestConfig config2 = {
        .num_producers = 5,
        .num_consumers = 2,
        .items_per_producer = 30,
        .queue_capacity = 5,
        .production_delay_ms = 50,
        .consumption_delay_ms = 200,
        .test_duration_seconds = 8
    };
    
    run_producer_consumer_test(&config2);
    
    // Test 3: More consumers than producers (starvation test)
    TestConfig config3 = {
        .num_producers = 2,
        .num_consumers = 5,
        .items_per_producer = 40,
        .queue_capacity = 15,
        .production_delay_ms = 200,
        .consumption_delay_ms = 50,
        .test_duration_seconds = 8
    };
    
    run_producer_consumer_test(&config3);
    
    return 0;
}
```

### Advanced: Priority Queue Implementation

```c
typedef struct {
    WorkItem* buffer;
    int capacity;
    int size;
    
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    
    long total_produced;
    long total_consumed;
    bool shutdown;
} PriorityQueue;

// Helper function to maintain heap property (max-heap based on priority)
void heapify_up(WorkItem* buffer, int index) {
    if (index == 0) return;
    
    int parent = (index - 1) / 2;
    if (buffer[parent].priority < buffer[index].priority) {
        WorkItem temp = buffer[parent];
        buffer[parent] = buffer[index];
        buffer[index] = temp;
        heapify_up(buffer, parent);
    }
}

void heapify_down(WorkItem* buffer, int size, int index) {
    int largest = index;
    int left = 2 * index + 1;
    int right = 2 * index + 2;
    
    if (left < size && buffer[left].priority > buffer[largest].priority) {
        largest = left;
    }
    
    if (right < size && buffer[right].priority > buffer[largest].priority) {
        largest = right;
    }
    
    if (largest != index) {
        WorkItem temp = buffer[index];
        buffer[index] = buffer[largest];
        buffer[largest] = temp;
        heapify_down(buffer, size, largest);
    }
}

bool priority_queue_enqueue(PriorityQueue* queue, const WorkItem* item) {
    if (!queue || !item) return false;
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size >= queue->capacity && !queue->shutdown) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }
    
    if (queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    // Add item to end and heapify up
    queue->buffer[queue->size] = *item;
    heapify_up(queue->buffer, queue->size);
    queue->size++;
    queue->total_produced++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

bool priority_queue_dequeue(PriorityQueue* queue, WorkItem* item) {
    if (!queue || !item) return false;
    
    pthread_mutex_lock(&queue->mutex);
    
    while (queue->size == 0 && !queue->shutdown) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }
    
    if (queue->size == 0 && queue->shutdown) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    // Remove highest priority item (root of heap)
    *item = queue->buffer[0];
    queue->buffer[0] = queue->buffer[queue->size - 1];
    queue->size--;
    queue->total_consumed++;
    
    if (queue->size > 0) {
        heapify_down(queue->buffer, queue->size, 0);
    }
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);
    return true;
}
```

## Performance Analysis

```c
typedef struct {
    double avg_enqueue_time;
    double avg_dequeue_time;
    double throughput_items_per_sec;
    int queue_utilization_percent;
    int max_queue_size;
} PerformanceMetrics;

PerformanceMetrics analyze_performance(ProducerConsumerQueue* queue, int test_duration_ms) {
    PerformanceMetrics metrics = {0};
    
    const int num_operations = 10000;
    struct timespec start, end;
    
    // Test enqueue performance
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < num_operations / 2; i++) {
        WorkItem item = {.item_id = i, .priority = i % 10};
        pc_queue_enqueue_timeout(queue, &item, 100);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    metrics.avg_enqueue_time = ((end.tv_sec - start.tv_sec) + 
                               (end.tv_nsec - start.tv_nsec) / 1e9) / (num_operations / 2);
    
    // Test dequeue performance
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < num_operations / 2; i++) {
        WorkItem item;
        pc_queue_dequeue_timeout(queue, &item, 100);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    metrics.avg_dequeue_time = ((end.tv_sec - start.tv_sec) + 
                               (end.tv_nsec - start.tv_nsec) / 1e9) / (num_operations / 2);
    
    QueueStats stats = pc_queue_get_stats(queue);
    metrics.throughput_items_per_sec = stats.total_consumed / (test_duration_ms / 1000.0);
    metrics.queue_utilization_percent = (stats.max_size_reached * 100) / stats.capacity;
    metrics.max_queue_size = stats.max_size_reached;
    
    return metrics;
}
```

## Learning Objectives

After completing this project, you should understand:
- Producer-consumer pattern implementation
- Condition variables for efficient thread synchronization
- Bounded buffer management
- Timeout mechanisms in multi-threaded applications
- Performance analysis of concurrent systems
- Priority-based queue implementations

## Extensions

1. **Work Stealing Queues**
   - Multiple queues with work stealing between them
   - Load balancing across consumer threads

2. **Batch Operations**
   - Enqueue/dequeue multiple items at once
   - Optimize for bulk transfers

3. **Queue Persistence**
   - Save queue state to disk
   - Recovery after crashes

4. **Network Distribution**
   - Distribute queue across multiple processes/machines
   - Remote procedure calls for queue operations

## Assessment Criteria

- **Correctness (35%)**: Proper synchronization without race conditions
- **Performance (30%)**: Efficient implementation with good throughput
- **Features (20%)**: Implementation of advanced features
- **Code Quality (15%)**: Clean, maintainable, well-documented code

## Next Project
[Project 4: Thread Pool Implementation](Project4_Thread_Pool.md)
