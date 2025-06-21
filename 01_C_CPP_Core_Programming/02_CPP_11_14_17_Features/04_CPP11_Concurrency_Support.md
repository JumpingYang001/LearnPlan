# C++11 Concurrency Support

## Overview

C++11 introduced built-in concurrency support to the standard library, providing thread management, synchronization primitives, and asynchronous execution capabilities. This marked a major step forward in making concurrent programming more accessible and standardized in C++.

## Key Components

### 1. std::thread

The foundation of C++11 concurrency - represents a single thread of execution.

#### Basic Thread Creation and Management

```cpp
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

// Simple function to run in a thread
void simple_task(int id, const std::string& message) {
    for (int i = 0; i < 5; ++i) {
        std::cout << "Thread " << id << ": " << message << " (" << i << ")" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Callable object (functor)
class ThreadTask {
private:
    int thread_id;
    
public:
    ThreadTask(int id) : thread_id(id) {}
    
    void operator()() {
        for (int i = 0; i < 3; ++i) {
            std::cout << "Functor thread " << thread_id << " iteration " << i << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
    }
};

void demonstrate_basic_threads() {
    std::cout << "\n=== Basic Thread Usage ===" << std::endl;
    
    // 1. Thread with function pointer
    std::thread t1(simple_task, 1, "Hello from thread");
    
    // 2. Thread with lambda
    std::thread t2([](int id) {
        for (int i = 0; i < 3; ++i) {
            std::cout << "Lambda thread " << id << " iteration " << i << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }
    }, 2);
    
    // 3. Thread with functor
    ThreadTask task(3);
    std::thread t3(task);
    
    // 4. Thread with member function
    class Worker {
    public:
        void work(int iterations) {
            for (int i = 0; i < iterations; ++i) {
                std::cout << "Member function thread iteration " << i << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(110));
            }
        }
    };
    
    Worker worker;
    std::thread t4(&Worker::work, &worker, 3);
    
    // Wait for all threads to complete
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    
    std::cout << "All threads completed" << std::endl;
}

// Thread management examples
void demonstrate_thread_management() {
    std::cout << "\n=== Thread Management ===" << std::endl;
    
    // Get hardware concurrency
    unsigned int cores = std::thread::hardware_concurrency();
    std::cout << "Hardware concurrency: " << cores << " cores" << std::endl;
    
    // Detached threads
    std::thread detached_thread([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout << "Detached thread completed" << std::endl;
    });
    
    detached_thread.detach();  // Thread runs independently
    
    // Thread IDs
    std::thread id_thread([]() {
        std::cout << "Thread ID: " << std::this_thread::get_id() << std::endl;
    });
    
    std::cout << "Main thread ID: " << std::this_thread::get_id() << std::endl;
    std::cout << "Created thread ID: " << id_thread.get_id() << std::endl;
    
    id_thread.join();
    
    // Give detached thread time to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}
```

#### Thread Pool Implementation

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class SimpleThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    
public:
    SimpleThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        
                        if (stop && tasks.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        auto result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            
            tasks.emplace([task]() { (*task)(); });
        }
        
        condition.notify_one();
        return result;
    }
    
    ~SimpleThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        
        condition.notify_all();
        
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
};

void demonstrate_thread_pool() {
    std::cout << "\n=== Thread Pool Example ===" << std::endl;
    
    SimpleThreadPool pool(4);
    
    // Submit various tasks
    auto result1 = pool.enqueue([](int x) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return x * x;
    }, 5);
    
    auto result2 = pool.enqueue([](const std::string& str) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        return str + " processed";
    }, std::string("Task"));
    
    auto result3 = pool.enqueue([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return 42;
    });
    
    // Get results
    std::cout << "Result 1: " << result1.get() << std::endl;
    std::cout << "Result 2: " << result2.get() << std::endl;
    std::cout << "Result 3: " << result3.get() << std::endl;
}
```

### 2. Mutexes and Locks

Synchronization primitives for protecting shared data.

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <shared_mutex>  // C++14

// Shared resource
class Counter {
private:
    int count = 0;
    mutable std::mutex mtx;
    
public:
    void increment() {
        std::lock_guard<std::mutex> lock(mtx);
        ++count;
    }
    
    void decrement() {
        std::lock_guard<std::mutex> lock(mtx);
        --count;
    }
    
    int get_count() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count;
    }
    
    // Batch operations
    void add_multiple(const std::vector<int>& values) {
        std::lock_guard<std::mutex> lock(mtx);
        for (int val : values) {
            count += val;
        }
    }
};

void demonstrate_basic_mutex() {
    std::cout << "\n=== Basic Mutex Usage ===" << std::endl;
    
    Counter counter;
    std::vector<std::thread> threads;
    
    // Create threads that increment the counter
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&counter, i]() {
            for (int j = 0; j < 1000; ++j) {
                counter.increment();
            }
            std::cout << "Thread " << i << " completed increments" << std::endl;
        });
    }
    
    // Create threads that decrement the counter
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([&counter, i]() {
            for (int j = 0; j < 500; ++j) {
                counter.decrement();
            }
            std::cout << "Thread " << (i + 5) << " completed decrements" << std::endl;
        });
    }
    
    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Final counter value: " << counter.get_count() << std::endl;
    std::cout << "Expected: " << (5 * 1000 - 3 * 500) << std::endl;
}

// Different types of locks
void demonstrate_lock_types() {
    std::cout << "\n=== Different Lock Types ===" << std::endl;
    
    std::mutex mtx;
    int shared_data = 0;
    
    // std::lock_guard - RAII lock (most common)
    auto task1 = [&]() {
        std::lock_guard<std::mutex> lock(mtx);
        shared_data += 10;
        std::cout << "Task1: " << shared_data << std::endl;
    };
    
    // std::unique_lock - more flexible
    auto task2 = [&]() {
        std::unique_lock<std::mutex> lock(mtx);
        shared_data += 20;
        
        // Can unlock early
        lock.unlock();
        
        // Do some work without holding the lock
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // Relock if needed
        lock.lock();
        shared_data += 5;
        std::cout << "Task2: " << shared_data << std::endl;
    };
    
    // Deferred locking
    auto task3 = [&]() {
        std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
        
        // Do some work without the lock
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        // Now acquire the lock
        lock.lock();
        shared_data += 15;
        std::cout << "Task3: " << shared_data << std::endl;
    };
    
    std::thread t1(task1);
    std::thread t2(task2);
    std::thread t3(task3);
    
    t1.join();
    t2.join();
    t3.join();
    
    std::cout << "Final shared_data: " << shared_data << std::endl;
}

// Avoiding deadlock with std::lock
void demonstrate_deadlock_avoidance() {
    std::cout << "\n=== Deadlock Avoidance ===" << std::endl;
    
    std::mutex mtx1, mtx2;
    int resource1 = 0, resource2 = 0;
    
    auto task_a = [&]() {
        for (int i = 0; i < 5; ++i) {
            // Use std::lock to acquire multiple locks atomically
            std::unique_lock<std::mutex> lock1(mtx1, std::defer_lock);
            std::unique_lock<std::mutex> lock2(mtx2, std::defer_lock);
            
            std::lock(lock1, lock2);  // Deadlock-free acquisition
            
            resource1 += 1;
            resource2 += 2;
            
            std::cout << "Task A: res1=" << resource1 << ", res2=" << resource2 << std::endl;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };
    
    auto task_b = [&]() {
        for (int i = 0; i < 5; ++i) {
            // Different order but std::lock handles it
            std::unique_lock<std::mutex> lock2(mtx2, std::defer_lock);
            std::unique_lock<std::mutex> lock1(mtx1, std::defer_lock);
            
            std::lock(lock2, lock1);  // Still deadlock-free
            
            resource1 += 3;
            resource2 += 1;
            
            std::cout << "Task B: res1=" << resource1 << ", res2=" << resource2 << std::endl;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };
    
    std::thread ta(task_a);
    std::thread tb(task_b);
    
    ta.join();
    tb.join();
    
    std::cout << "Final: res1=" << resource1 << ", res2=" << resource2 << std::endl;
}
```

### 3. std::future and std::promise

Asynchronous communication between threads.

```cpp
#include <iostream>
#include <thread>
#include <future>
#include <chrono>
#include <vector>
#include <numeric>
#include <exception>

// Basic promise/future usage
void demonstrate_promise_future() {
    std::cout << "\n=== Promise/Future Basics ===" << std::endl;
    
    // Create promise/future pair
    std::promise<int> promise;
    std::future<int> future = promise.get_future();
    
    // Start a thread that will fulfill the promise
    std::thread producer([&promise]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Perform some computation
        int result = 42 * 2;
        
        std::cout << "Producer calculated: " << result << std::endl;
        promise.set_value(result);
    });
    
    // Main thread waits for the result
    std::cout << "Waiting for result..." << std::endl;
    int result = future.get();
    std::cout << "Received result: " << result << std::endl;
    
    producer.join();
}

// Exception handling with promise/future
void demonstrate_promise_future_exceptions() {
    std::cout << "\n=== Promise/Future Exception Handling ===" << std::endl;
    
    std::promise<double> promise;
    std::future<double> future = promise.get_future();
    
    std::thread worker([&promise]() {
        try {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            // Simulate an error condition
            bool error_occurred = true;
            if (error_occurred) {
                throw std::runtime_error("Computation failed!");
            }
            
            promise.set_value(3.14);
        } catch (...) {
            // Forward exception to future
            promise.set_exception(std::current_exception());
        }
    });
    
    try {
        double result = future.get();
        std::cout << "Result: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
    
    worker.join();
}

// Shared future - multiple threads can wait for the same result
void demonstrate_shared_future() {
    std::cout << "\n=== Shared Future ===" << std::endl;
    
    std::promise<std::string> promise;
    std::shared_future<std::string> shared_future = promise.get_future().share();
    
    // Multiple threads waiting for the same result
    std::vector<std::thread> waiters;
    
    for (int i = 0; i < 3; ++i) {
        waiters.emplace_back([shared_future, i]() {
            std::cout << "Waiter " << i << " waiting..." << std::endl;
            std::string result = shared_future.get();
            std::cout << "Waiter " << i << " got: " << result << std::endl;
        });
    }
    
    // Producer thread
    std::thread producer([&promise]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        promise.set_value("Shared result!");
    });
    
    // Wait for all threads
    for (auto& t : waiters) {
        t.join();
    }
    producer.join();
}
```

### 4. std::async

High-level interface for asynchronous execution.

```cpp
#include <iostream>
#include <future>
#include <vector>
#include <numeric>
#include <chrono>
#include <algorithm>

// CPU-intensive task for demonstration
long fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Parallel computation example
std::vector<int> parallel_transform(const std::vector<int>& input, 
                                   std::function<int(int)> func) {
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = input.size() / num_threads;
    
    std::vector<std::future<std::vector<int>>> futures;
    
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? input.size() : start + chunk_size;
        
        futures.push_back(std::async(std::launch::async, [&input, func, start, end]() {
            std::vector<int> result;
            result.reserve(end - start);
            
            for (size_t j = start; j < end; ++j) {
                result.push_back(func(input[j]));
            }
            
            return result;
        }));
    }
    
    // Collect results
    std::vector<int> final_result;
    for (auto& future : futures) {
        auto partial_result = future.get();
        final_result.insert(final_result.end(), 
                           partial_result.begin(), 
                           partial_result.end());
    }
    
    return final_result;
}

void demonstrate_async() {
    std::cout << "\n=== std::async Usage ===" << std::endl;
    
    // Basic async usage
    auto future1 = std::async(std::launch::async, fibonacci, 35);
    auto future2 = std::async(std::launch::async, fibonacci, 36);
    auto future3 = std::async(std::launch::deferred, fibonacci, 37);  // Lazy evaluation
    
    std::cout << "Started async computations..." << std::endl;
    
    // Do other work while computations run
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Doing other work..." << std::endl;
    
    // Get results
    std::cout << "fib(35) = " << future1.get() << std::endl;
    std::cout << "fib(36) = " << future2.get() << std::endl;
    std::cout << "fib(37) = " << future3.get() << std::endl;  // Computed on demand
    
    // Parallel processing example
    std::cout << "\n--- Parallel Processing ---" << std::endl;
    
    std::vector<int> input(1000);
    std::iota(input.begin(), input.end(), 1);  // Fill with 1, 2, 3, ...
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Parallel square computation
    auto squared = parallel_transform(input, [](int x) { 
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        return x * x; 
    });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Processed " << input.size() << " elements in " 
              << duration.count() << " ms" << std::endl;
    std::cout << "First few squared values: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << squared[i] << " ";
    }
    std::cout << std::endl;
}

// Timeout and status checking
void demonstrate_future_status() {
    std::cout << "\n=== Future Status and Timeout ===" << std::endl;
    
    auto slow_task = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        return 42;
    });
    
    // Check status with timeout
    while (true) {
        auto status = slow_task.wait_for(std::chrono::milliseconds(500));
        
        switch (status) {
            case std::future_status::ready:
                std::cout << "Task completed! Result: " << slow_task.get() << std::endl;
                return;
                
            case std::future_status::timeout:
                std::cout << "Still waiting..." << std::endl;
                break;
                
            case std::future_status::deferred:
                std::cout << "Task is deferred" << std::endl;
                break;
        }
    }
}
```

### 5. Atomic Operations

Lock-free synchronization for simple data types.

```cpp
#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
#include <chrono>

void demonstrate_atomic_operations() {
    std::cout << "\n=== Atomic Operations ===" << std::endl;
    
    // Basic atomic types
    std::atomic<int> atomic_counter(0);
    std::atomic<bool> done(false);
    
    const int num_threads = 4;
    const int increments_per_thread = 10000;
    
    std::vector<std::thread> threads;
    
    // Producer threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&atomic_counter, increments_per_thread, i]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                atomic_counter.fetch_add(1);  // Atomic increment
            }
            std::cout << "Thread " << i << " completed increments" << std::endl;
        });
    }
    
    // Monitor thread
    std::thread monitor([&atomic_counter, &done]() {
        while (!done.load()) {
            std::cout << "Current count: " << atomic_counter.load() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    
    // Wait for producer threads
    for (auto& t : threads) {
        t.join();
    }
    
    done.store(true);
    monitor.join();
    
    std::cout << "Final count: " << atomic_counter.load() << std::endl;
    std::cout << "Expected: " << num_threads * increments_per_thread << std::endl;
}

// Lock-free stack example
template<typename T>
class LockFreeStack {
private:
    struct Node {
        T data;
        Node* next;
        Node(T item) : data(std::move(item)), next(nullptr) {}
    };
    
    std::atomic<Node*> head;
    
public:
    LockFreeStack() : head(nullptr) {}
    
    void push(T item) {
        Node* new_node = new Node(std::move(item));
        new_node->next = head.load();
        
        while (!head.compare_exchange_weak(new_node->next, new_node)) {
            // Keep trying until successful
        }
    }
    
    bool pop(T& result) {
        Node* old_head = head.load();
        
        while (old_head && !head.compare_exchange_weak(old_head, old_head->next)) {
            // Keep trying until successful
        }
        
        if (old_head) {
            result = std::move(old_head->data);
            delete old_head;
            return true;
        }
        
        return false;
    }
    
    bool empty() const {
        return head.load() == nullptr;
    }
    
    ~LockFreeStack() {
        T dummy;
        while (pop(dummy)) {
            // Clear all remaining items
        }
    }
};

void demonstrate_lock_free_stack() {
    std::cout << "\n=== Lock-Free Stack ===" << std::endl;
    
    LockFreeStack<int> stack;
    std::atomic<bool> finished(false);
    
    // Producer thread
    std::thread producer([&stack, &finished]() {
        for (int i = 0; i < 100; ++i) {
            stack.push(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        finished.store(true);
        std::cout << "Producer finished" << std::endl;
    });
    
    // Consumer thread
    std::thread consumer([&stack, &finished]() {
        int value;
        int count = 0;
        
        while (!finished.load() || !stack.empty()) {
            if (stack.pop(value)) {
                count++;
                if (count % 20 == 0) {
                    std::cout << "Consumed " << count << " items, last: " << value << std::endl;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        std::cout << "Consumer finished, total consumed: " << count << std::endl;
    });
    
    producer.join();
    consumer.join();
}
```

## Advanced Concurrency Patterns

### Producer-Consumer with Condition Variables

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>

template<typename T>
class BoundedQueue {
private:
    std::queue<T> queue;
    size_t max_size;
    std::mutex mtx;
    std::condition_variable not_full;
    std::condition_variable not_empty;
    
public:
    BoundedQueue(size_t max_sz) : max_size(max_sz) {}
    
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        not_full.wait(lock, [this] { return queue.size() < max_size; });
        
        queue.push(item);
        not_empty.notify_one();
    }
    
    T pop() {
        std::unique_lock<std::mutex> lock(mtx);
        not_empty.wait(lock, [this] { return !queue.empty(); });
        
        T item = queue.front();
        queue.pop();
        not_full.notify_one();
        
        return item;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }
};

void demonstrate_producer_consumer() {
    std::cout << "\n=== Producer-Consumer Pattern ===" << std::endl;
    
    BoundedQueue<int> queue(10);
    std::atomic<bool> stop_production(false);
    
    // Producer
    std::thread producer([&queue, &stop_production]() {
        for (int i = 0; i < 50; ++i) {
            queue.push(i);
            std::cout << "Produced: " << i << " (queue size: " << queue.size() << ")" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        stop_production.store(true);
        std::cout << "Producer finished" << std::endl;
    });
    
    // Consumer
    std::thread consumer([&queue, &stop_production]() {
        int consumed_count = 0;
        
        while (!stop_production.load() || queue.size() > 0) {
            try {
                int item = queue.pop();
                consumed_count++;
                std::cout << "Consumed: " << item << " (total consumed: " << consumed_count << ")" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            } catch (...) {
                break;
            }
        }
        
        std::cout << "Consumer finished, total consumed: " << consumed_count << std::endl;
    });
    
    producer.join();
    consumer.join();
}
```

## Performance Considerations and Best Practices

```cpp
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>

void demonstrate_performance_considerations() {
    std::cout << "\n=== Performance Considerations ===" << std::endl;
    
    const int num_operations = 1000000;
    
    // Compare atomic vs mutex performance
    std::atomic<int> atomic_counter(0);
    int mutex_counter = 0;
    std::mutex counter_mutex;
    
    // Atomic performance
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> atomic_threads;
    for (int i = 0; i < 4; ++i) {
        atomic_threads.emplace_back([&atomic_counter, num_operations]() {
            for (int j = 0; j < num_operations / 4; ++j) {
                atomic_counter.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    
    for (auto& t : atomic_threads) {
        t.join();
    }
    
    auto atomic_end = std::chrono::high_resolution_clock::now();
    auto atomic_duration = std::chrono::duration_cast<std::chrono::milliseconds>(atomic_end - start);
    
    // Mutex performance
    start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> mutex_threads;
    for (int i = 0; i < 4; ++i) {
        mutex_threads.emplace_back([&mutex_counter, &counter_mutex, num_operations]() {
            for (int j = 0; j < num_operations / 4; ++j) {
                std::lock_guard<std::mutex> lock(counter_mutex);
                ++mutex_counter;
            }
        });
    }
    
    for (auto& t : mutex_threads) {
        t.join();
    }
    
    auto mutex_end = std::chrono::high_resolution_clock::now();
    auto mutex_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mutex_end - start);
    
    std::cout << "Atomic operations: " << atomic_duration.count() << " ms" << std::endl;
    std::cout << "Mutex operations: " << mutex_duration.count() << " ms" << std::endl;
    std::cout << "Atomic counter: " << atomic_counter.load() << std::endl;
    std::cout << "Mutex counter: " << mutex_counter << std::endl;
}

int main() {
    demonstrate_basic_threads();
    demonstrate_thread_management();
    demonstrate_thread_pool();
    
    demonstrate_basic_mutex();
    demonstrate_lock_types();
    demonstrate_deadlock_avoidance();
    
    demonstrate_promise_future();
    demonstrate_promise_future_exceptions();
    demonstrate_shared_future();
    
    demonstrate_async();
    demonstrate_future_status();
    
    demonstrate_atomic_operations();
    demonstrate_lock_free_stack();
    
    demonstrate_producer_consumer();
    demonstrate_performance_considerations();
    
    return 0;
}
```

## Summary

C++11 concurrency support provides:

- **std::thread**: Basic thread management and execution
- **Mutexes and locks**: Synchronization primitives for protecting shared data
- **std::future/std::promise**: Asynchronous communication between threads
- **std::async**: High-level interface for asynchronous execution
- **Atomic operations**: Lock-free synchronization for simple operations

Key benefits:
- Standardized concurrency across platforms
- Type-safe synchronization primitives
- RAII-based lock management
- High-level abstractions for common patterns
- Performance optimizations through atomic operations

Best practices:
- Use RAII locks (lock_guard, unique_lock) instead of manual locking
- Prefer std::async for simple asynchronous tasks
- Use atomic operations for simple shared data
- Avoid deadlocks with std::lock for multiple mutexes
- Consider lock-free programming for high-performance scenarios
