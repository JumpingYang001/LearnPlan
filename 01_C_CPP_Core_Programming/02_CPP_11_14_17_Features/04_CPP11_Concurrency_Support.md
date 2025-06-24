# C++11 Concurrency Support

*Duration: 2 weeks*

## Overview

C++11 introduced built-in concurrency support to the standard library, providing thread management, synchronization primitives, and asynchronous execution capabilities. This marked a major step forward in making concurrent programming more accessible and standardized in C++.

## Learning Objectives

By the end of this section, you should be able to:

### Core Competencies
- **Create and manage threads** using `std::thread` with various callable objects
- **Implement thread-safe data structures** using mutexes and atomic operations
- **Handle asynchronous operations** with `std::future`, `std::promise`, and `std::async`
- **Apply memory ordering** concepts for atomic operations
- **Design lock-free algorithms** for high-performance scenarios
- **Debug and profile** multi-threaded C++ applications

### Advanced Skills
- **Implement common concurrency patterns** (producer-consumer, reader-writer, etc.)
- **Choose appropriate synchronization primitives** based on use case
- **Optimize performance** by reducing contention and lock overhead
- **Handle exceptions** in multi-threaded environments
- **Design thread-safe APIs** following C++ best practices

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Explain the differences between `std::thread` and platform-specific threading  
□ Create threads using functions, lambdas, and member functions  
□ Implement proper exception handling in multi-threaded code  
□ Choose between `std::mutex`, `std::recursive_mutex`, and `std::shared_mutex`  
□ Use `std::condition_variable` for thread coordination  
□ Implement basic lock-free data structures  
□ Handle thread-safe singleton patterns  
□ Profile and debug multi-threaded applications  
□ Apply memory ordering constraints appropriately

### Why C++11 Concurrency Matters

Before C++11, developers had to rely on platform-specific threading libraries (like pthreads on Unix/Linux or Windows threads on Windows). C++11 standardized concurrency, providing:

- **Platform Independence**: Write once, compile anywhere
- **Type Safety**: Template-based design with compile-time checks
- **RAII Integration**: Automatic resource management
- **Modern C++ Features**: Support for lambdas, move semantics, etc.
- **Performance**: Optimized implementations by compiler vendors

### Comparison: C++11 vs POSIX Threads

| Feature | POSIX Threads | C++11 std::thread |
|---------|---------------|-------------------|
| **Platform** | Unix/Linux primarily | Cross-platform |
| **Type Safety** | C-style, void* parameters | Type-safe templates |
| **Error Handling** | Return codes | Exceptions |
| **Resource Management** | Manual cleanup | RAII automatic cleanup |
| **Lambda Support** | No | Full support |
| **Move Semantics** | No | Full support |

```cpp
// POSIX Threads approach
void* pthread_function(void* arg) {
    int* data = static_cast<int*>(arg);
    // Work with data
    return nullptr;
}

pthread_t thread;
int data = 42;
pthread_create(&thread, nullptr, pthread_function, &data);
pthread_join(thread, nullptr);

// C++11 approach
std::thread cpp_thread([](int data) {
    // Work with data
}, 42);
cpp_thread.join();
```

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

#### Thread Exception Handling

One critical aspect often overlooked is proper exception handling in multi-threaded environments:

```cpp
#include <iostream>
#include <thread>
#include <exception>
#include <vector>

class ThreadSafeExceptionDemo {
private:
    std::vector<std::exception_ptr> thread_exceptions;
    std::mutex exception_mutex;
    
public:
    void add_exception(std::exception_ptr eptr) {
        std::lock_guard<std::mutex> lock(exception_mutex);
        thread_exceptions.push_back(eptr);
    }
    
    void check_exceptions() {
        std::lock_guard<std::mutex> lock(exception_mutex);
        if (!thread_exceptions.empty()) {
            std::rethrow_exception(thread_exceptions.front());
        }
    }
    
    void risky_thread_function(int thread_id) {
        try {
            if (thread_id == 2) {
                throw std::runtime_error("Thread " + std::to_string(thread_id) + " failed!");
            }
            
            // Simulate work
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "Thread " << thread_id << " completed successfully" << std::endl;
            
        } catch (...) {
            // Capture exception for main thread
            add_exception(std::current_exception());
        }
    }
};

void demonstrate_thread_exceptions() {
    std::cout << "\n=== Thread Exception Handling ===" << std::endl;
    
    ThreadSafeExceptionDemo demo;
    std::vector<std::thread> threads;
    
    // Create threads, one will throw an exception
    for (int i = 1; i <= 4; ++i) {
        threads.emplace_back(&ThreadSafeExceptionDemo::risky_thread_function, &demo, i);
    }
    
    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }
    
    // Check for exceptions in main thread
    try {
        demo.check_exceptions();
        std::cout << "All threads completed without exceptions" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught thread exception: " << e.what() << std::endl;
    }
}
```

#### Thread-Local Storage

```cpp
#include <iostream>
#include <thread>
#include <vector>

// Thread-local storage example
thread_local int tls_counter = 0;
thread_local std::string tls_thread_name;

class ThreadLocalDemo {
private:
    static thread_local int instance_count;
    int thread_id;
    
public:
    ThreadLocalDemo(int id) : thread_id(id) {
        ++instance_count;
        tls_thread_name = "Thread-" + std::to_string(id);
    }
    
    void show_thread_local_data() {
        std::cout << tls_thread_name << ": instance_count = " << instance_count 
                  << ", tls_counter = " << tls_counter << std::endl;
    }
    
    void increment_counters() {
        ++tls_counter;
        ++instance_count;
    }
};

// Define the thread_local static member
thread_local int ThreadLocalDemo::instance_count = 0;

void demonstrate_thread_local_storage() {
    std::cout << "\n=== Thread-Local Storage ===" << std::endl;
    
    std::vector<std::thread> threads;
    
    for (int i = 1; i <= 3; ++i) {
        threads.emplace_back([i]() {
            ThreadLocalDemo demo(i);
            
            // Each thread has its own copy of thread-local variables
            for (int j = 0; j < 5; ++j) {
                demo.increment_counters();
                demo.show_thread_local_data();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Main thread has its own thread-local storage
    std::cout << "Main thread tls_counter: " << tls_counter << std::endl;
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

#### Comprehensive Mutex Types Comparison

C++11 provides several mutex types, each optimized for different use cases:

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <shared_mutex>  // C++14
#include <chrono>
#include <vector>

class MutexComparisonDemo {
private:
    // Different mutex types
    std::mutex basic_mutex;
    std::recursive_mutex recursive_mutex;
    std::timed_mutex timed_mutex;
    std::shared_mutex shared_mutex;  // C++14
    
    int shared_data = 0;
    
public:
    // Basic mutex - simple exclusive access
    void demonstrate_basic_mutex() {
        std::lock_guard<std::mutex> lock(basic_mutex);
        shared_data++;
        std::cout << "Basic mutex: " << shared_data << std::endl;
    }
    
    // Recursive mutex - same thread can lock multiple times
    void demonstrate_recursive_mutex(int depth = 0) {
        std::lock_guard<std::recursive_mutex> lock(recursive_mutex);
        
        if (depth < 3) {
            std::cout << "Recursive lock depth: " << depth << std::endl;
            demonstrate_recursive_mutex(depth + 1);  // Same thread locks again
        }
    }
    
    // Timed mutex - attempt lock with timeout
    bool demonstrate_timed_mutex() {
        if (timed_mutex.try_lock_for(std::chrono::milliseconds(100))) {
            std::cout << "Timed mutex acquired" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            timed_mutex.unlock();
            return true;
        } else {
            std::cout << "Timed mutex timeout" << std::endl;
            return false;
        }
    }
    
    // Shared mutex - multiple readers, single writer (C++14)
    void read_with_shared_mutex(int reader_id) {
        std::shared_lock<std::shared_mutex> lock(shared_mutex);
        std::cout << "Reader " << reader_id << " reading: " << shared_data << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    void write_with_shared_mutex(int writer_id) {
        std::unique_lock<std::shared_mutex> lock(shared_mutex);
        shared_data += 10;
        std::cout << "Writer " << writer_id << " writing: " << shared_data << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
};

void demonstrate_mutex_types() {
    std::cout << "\n=== Mutex Types Comparison ===" << std::endl;
    
    MutexComparisonDemo demo;
    std::vector<std::thread> threads;
    
    // Basic mutex test
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(&MutexComparisonDemo::demonstrate_basic_mutex, &demo);
    }
    
    // Recursive mutex test
    threads.emplace_back(&MutexComparisonDemo::demonstrate_recursive_mutex, &demo, 0);
    
    // Shared mutex test - multiple readers and writers
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(&MutexComparisonDemo::read_with_shared_mutex, &demo, i);
    }
    threads.emplace_back(&MutexComparisonDemo::write_with_shared_mutex, &demo, 1);
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Timed mutex test
    std::cout << "\nTimed mutex test:" << std::endl;
    demo.demonstrate_timed_mutex();
}
```

#### Lock Hierarchies and Deadlock Prevention

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

// Hierarchical mutex for deadlock prevention
class HierarchicalMutex {
private:
    std::mutex internal_mutex;
    unsigned long const hierarchy_value;
    unsigned long previous_hierarchy_value;
    static thread_local unsigned long this_thread_hierarchy_value;
    
    void check_for_hierarchy_violation() {
        if (this_thread_hierarchy_value <= hierarchy_value) {
            throw std::logic_error("mutex hierarchy violated");
        }
    }
    
    void update_hierarchy_value() {
        previous_hierarchy_value = this_thread_hierarchy_value;
        this_thread_hierarchy_value = hierarchy_value;
    }
    
public:
    explicit HierarchicalMutex(unsigned long value) 
        : hierarchy_value(value), previous_hierarchy_value(0) {}
    
    void lock() {
        check_for_hierarchy_violation();
        internal_mutex.lock();
        update_hierarchy_value();
    }
    
    void unlock() {
        if (this_thread_hierarchy_value != hierarchy_value) {
            throw std::logic_error("mutex hierarchy violated");
        }
        this_thread_hierarchy_value = previous_hierarchy_value;
        internal_mutex.unlock();
    }
    
    bool try_lock() {
        check_for_hierarchy_violation();
        if (!internal_mutex.try_lock()) {
            return false;
        }
        update_hierarchy_value();
        return true;
    }
};

thread_local unsigned long HierarchicalMutex::this_thread_hierarchy_value(ULONG_MAX);

// Example usage
HierarchicalMutex high_level_mutex(10000);
HierarchicalMutex low_level_mutex(5000);

void demonstrate_hierarchical_locking() {
    std::cout << "\n=== Hierarchical Locking ===" << std::endl;
    
    auto correct_order = []() {
        std::lock_guard<HierarchicalMutex> lk1(high_level_mutex);  // Higher value first
        std::lock_guard<HierarchicalMutex> lk2(low_level_mutex);   // Lower value second
        std::cout << "Correct hierarchy order - success!" << std::endl;
    };
    
    auto incorrect_order = []() {
        try {
            std::lock_guard<HierarchicalMutex> lk1(low_level_mutex);   // Lower value first
            std::lock_guard<HierarchicalMutex> lk2(high_level_mutex);  // Higher value second - ERROR!
            std::cout << "Incorrect hierarchy order - should not reach here!" << std::endl;
        } catch (const std::logic_error& e) {
            std::cout << "Caught hierarchy violation: " << e.what() << std::endl;
        }
    };
    
    std::thread t1(correct_order);
    std::thread t2(incorrect_order);
    
    t1.join();
    t2.join();
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

#### Memory Ordering and Advanced Atomic Operations

Understanding memory ordering is crucial for writing correct lock-free code:

```cpp
#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
#include <chrono>

class MemoryOrderingDemo {
private:
    std::atomic<int> data{0};
    std::atomic<bool> flag{false};
    
public:
    // Sequential consistency (default, strongest)
    void sequential_consistency_example() {
        std::cout << "\n--- Sequential Consistency ---" << std::endl;
        
        auto writer = [this]() {
            data.store(42);  // Default: memory_order_seq_cst
            flag.store(true);  // Default: memory_order_seq_cst
        };
        
        auto reader = [this]() {
            while (!flag.load()) {  // Default: memory_order_seq_cst
                std::this_thread::yield();
            }
            std::cout << "Sequential: data = " << data.load() << std::endl;
        };
        
        std::thread t1(writer);
        std::thread t2(reader);
        t1.join();
        t2.join();
        
        // Reset
        data.store(0);
        flag.store(false);
    }
    
    // Acquire-Release semantics (weaker but often sufficient)
    void acquire_release_example() {
        std::cout << "\n--- Acquire-Release ---" << std::endl;
        
        auto writer = [this]() {
            data.store(84, std::memory_order_relaxed);  // Can be reordered
            flag.store(true, std::memory_order_release);  // Release barrier
        };
        
        auto reader = [this]() {
            while (!flag.load(std::memory_order_acquire)) {  // Acquire barrier
                std::this_thread::yield();
            }
            // All writes before release are visible after acquire
            std::cout << "Acquire-Release: data = " << data.load(std::memory_order_relaxed) << std::endl;
        };
        
        std::thread t1(writer);
        std::thread t2(reader);
        t1.join();
        t2.join();
        
        // Reset
        data.store(0);
        flag.store(false);
    }
    
    // Relaxed ordering (weakest, best performance)
    void relaxed_ordering_example() {
        std::cout << "\n--- Relaxed Ordering ---" << std::endl;
        
        std::atomic<int> counter{0};
        const int num_threads = 4;
        const int increments_per_thread = 100000;
        
        std::vector<std::thread> threads;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&counter, increments_per_thread]() {
                for (int j = 0; j < increments_per_thread; ++j) {
                    counter.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Relaxed ordering result: " << counter.load() 
                  << " (expected: " << num_threads * increments_per_thread << ")" << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    }
};

void demonstrate_memory_ordering() {
    std::cout << "\n=== Memory Ordering Examples ===" << std::endl;
    
    MemoryOrderingDemo demo;
    demo.sequential_consistency_example();
    demo.acquire_release_example();
    demo.relaxed_ordering_example();
}
```

#### Compare-and-Swap (CAS) Operations

```cpp
#include <iostream>
#include <atomic>
#include <thread>
#include <vector>

// Lock-free linked list using CAS
template<typename T>
class LockFreeLinkedList {
private:
    struct Node {
        T data;
        std::atomic<Node*> next;
        
        Node(const T& item) : data(item), next(nullptr) {}
    };
    
    std::atomic<Node*> head;
    
public:
    LockFreeLinkedList() : head(nullptr) {}
    
    void push_front(const T& item) {
        Node* new_node = new Node(item);
        Node* current_head = head.load();
        
        do {
            new_node->next.store(current_head);
        } while (!head.compare_exchange_weak(current_head, new_node));
    }
    
    bool pop_front(T& result) {
        Node* current_head = head.load();
        
        while (current_head != nullptr) {
            if (head.compare_exchange_weak(current_head, current_head->next.load())) {
                result = current_head->data;
                delete current_head;
                return true;
            }
        }
        
        return false;
    }
    
    void print_list() {
        Node* current = head.load();
        std::cout << "List contents: ";
        while (current != nullptr) {
            std::cout << current->data << " ";
            current = current->next.load();
        }
        std::cout << std::endl;
    }
    
    ~LockFreeLinkedList() {
        T dummy;
        while (pop_front(dummy)) {
            // Clear all remaining nodes
        }
    }
};

void demonstrate_compare_and_swap() {
    std::cout << "\n=== Compare-and-Swap Operations ===" << std::endl;
    
    LockFreeLinkedList<int> list;
    std::atomic<bool> stop_flag{false};
    
    // Producer threads
    std::vector<std::thread> producers;
    for (int i = 0; i < 3; ++i) {
        producers.emplace_back([&list, i, &stop_flag]() {
            for (int j = 0; j < 10; ++j) {
                list.push_front(i * 100 + j);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    
    // Consumer thread
    std::thread consumer([&list, &stop_flag]() {
        int consumed_count = 0;
        int value;
        
        while (consumed_count < 30) {  // Expect 30 items total
            if (list.pop_front(value)) {
                consumed_count++;
                if (consumed_count % 10 == 0) {
                    std::cout << "Consumed " << consumed_count << " items" << std::endl;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        std::cout << "Consumer finished, total consumed: " << consumed_count << std::endl;
    });
    
    // Wait for all producers
    for (auto& p : producers) {
        p.join();
    }
    
    consumer.join();
    list.print_list();
}
```

#### Atomic Smart Pointers (C++20 Preview)

```cpp
#include <iostream>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

// Demonstration of atomic operations with shared_ptr
class AtomicSmartPointerDemo {
private:
    std::shared_ptr<int> shared_data;
    
public:
    AtomicSmartPointerDemo() : shared_data(std::make_shared<int>(0)) {}
    
    void update_data(int new_value) {
        // Create new shared_ptr with updated data
        auto new_data = std::make_shared<int>(new_value);
        
        // Atomic exchange
        auto old_data = std::atomic_exchange(&shared_data, new_data);
        
        std::cout << "Updated data from " << *old_data << " to " << *new_data << std::endl;
    }
    
    int read_data() const {
        // Atomic load
        auto current_data = std::atomic_load(&shared_data);
        return *current_data;
    }
    
    bool compare_and_swap_data(int expected, int new_value) {
        auto expected_ptr = std::make_shared<int>(expected);
        auto new_ptr = std::make_shared<int>(new_value);
        
        // Note: This is a simplified example. Real atomic operations with 
        // shared_ptr are more complex and require careful consideration
        return std::atomic_compare_exchange_strong(&shared_data, &expected_ptr, new_ptr);
    }
};

void demonstrate_atomic_smart_pointers() {
    std::cout << "\n=== Atomic Smart Pointers ===" << std::endl;
    
    AtomicSmartPointerDemo demo;
    std::vector<std::thread> threads;
    
    // Multiple threads updating data
    for (int i = 1; i <= 5; ++i) {
        threads.emplace_back([&demo, i]() {
            demo.update_data(i * 10);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
    }
    
    // Reader threads
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([&demo, i]() {
            for (int j = 0; j < 5; ++j) {
                int value = demo.read_data();
                std::cout << "Reader " << i << " read: " << value << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Final data value: " << demo.read_data() << std::endl;
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

## Summary and Next Steps

C++11 concurrency support provides a comprehensive foundation for modern multi-threaded programming:

### Core Components Mastered
- **std::thread**: Cross-platform thread management with type safety and RAII
- **Synchronization primitives**: Mutexes, locks, and condition variables for safe data sharing
- **Asynchronous programming**: futures, promises, and std::async for clean async code
- **Atomic operations**: Lock-free programming with memory ordering control
- **Advanced patterns**: Thread pools, lock-free data structures, and work distribution

### Key Benefits Achieved
✅ **Platform Independence**: Write once, compile anywhere  
✅ **Type Safety**: Template-based design prevents common C-style errors  
✅ **Resource Management**: RAII ensures proper cleanup and exception safety  
✅ **Performance**: Optimized implementations with fine-grained control  
✅ **Modern Integration**: Full support for lambdas, move semantics, and C++11 features  

### Best Practices Applied
- Use RAII locks (`lock_guard`, `unique_lock`) instead of manual locking
- Prefer `std::async` for simple asynchronous tasks
- Use atomic operations for simple shared data access
- Avoid deadlocks with `std::lock` for multiple mutexes
- Apply memory ordering constraints appropriately for lock-free code
- Design exception-safe multi-threaded APIs

### Performance Considerations
- **Contention reduction**: Minimize lock duration and scope
- **Cache efficiency**: Align data structures and avoid false sharing
- **Scalability**: Design algorithms that scale with available cores
- **Lock-free optimization**: Use atomic operations where appropriate

### Debugging and Testing Strategy
- Use thread sanitizers during development
- Apply static analysis tools for concurrency issues
- Implement comprehensive unit tests for race conditions
- Profile performance under realistic loads
- Test on multiple platforms and architectures

### Migration Path from Legacy Code
```cpp
// Legacy pthread code
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_lock(&mtx);
// critical section
pthread_mutex_unlock(&mtx);

// Modern C++11 equivalent
std::mutex mtx;
{
    std::lock_guard<std::mutex> lock(mtx);
    // critical section - automatic unlock
}
```

### Next Learning Steps

**Immediate Next Topics:**
1. **C++14/17 Concurrency Enhancements**: `std::shared_mutex`, parallel algorithms
2. **Memory Models**: Deep dive into acquire-release semantics and consistency
3. **Lock-Free Data Structures**: Advanced algorithms and correctness proofs
4. **Parallel Algorithms**: STL parallel execution policies (C++17)

**Advanced Specializations:**
- **High-Performance Computing**: NUMA awareness, CPU affinity
- **Real-Time Systems**: Priority inheritance, deadline scheduling
- **Distributed Systems**: Network concurrency, actor models
- **GPU Programming**: CUDA, OpenCL integration with CPU concurrency

**Recommended Project:**
Build a multi-threaded application that demonstrates all learned concepts:
- Thread pool with work stealing
- Lock-free producer-consumer queues  
- Async I/O with futures
- Performance monitoring and profiling
- Comprehensive error handling and testing

### Industry Applications
- **Game Engines**: Parallel scene processing, physics simulation
- **Database Systems**: Concurrent transaction processing
- **Web Servers**: Request handling and connection management
- **Financial Systems**: High-frequency trading and risk calculation
- **Scientific Computing**: Parallel algorithms and data processing

The foundation built here prepares you for advanced concurrency topics and real-world multi-threaded system design. Continue practicing with increasingly complex scenarios and explore platform-specific optimizations as needed for your target applications.

## Practical Exercises and Assignments

### Exercise 1: Thread-Safe Singleton Pattern

Implement a thread-safe singleton using C++11 concurrency features:

```cpp
// TODO: Implement a thread-safe singleton class
template<typename T>
class ThreadSafeSingleton {
private:
    static std::once_flag initialized;
    static std::unique_ptr<T> instance;
    
public:
    static T& get_instance() {
        // Your implementation here
        // Use std::call_once for thread-safe initialization
    }
    
    // Prevent copying and assignment
    ThreadSafeSingleton(const ThreadSafeSingleton&) = delete;
    ThreadSafeSingleton& operator=(const ThreadSafeSingleton&) = delete;
};
```

### Exercise 2: Producer-Consumer with Multiple Queues

Create a system where multiple producers write to different queues and consumers can read from any available queue:

```cpp
// TODO: Implement multi-queue producer-consumer system
template<typename T>
class MultiQueueSystem {
private:
    std::vector<std::queue<T>> queues;
    std::vector<std::mutex> queue_mutexes;
    std::condition_variable consumer_cv;
    
public:
    MultiQueueSystem(size_t num_queues);
    void produce(const T& item, size_t queue_id);
    bool consume(T& item);  // Consume from any available queue
    void shutdown();
};
```

### Exercise 3: Parallel Algorithm Implementation

Implement a parallel merge sort using std::async:

```cpp
// TODO: Implement parallel merge sort
template<typename Iterator>
void parallel_merge_sort(Iterator first, Iterator last) {
    // Use std::async for recursive parallelization
    // Consider thread pool limitations
}
```

### Exercise 4: Lock-Free Ring Buffer

Implement a single-producer, single-consumer lock-free ring buffer:

```cpp
// TODO: Implement SPSC ring buffer
template<typename T, size_t Size>
class SPSCRingBuffer {
private:
    alignas(64) std::atomic<size_t> write_pos{0};
    alignas(64) std::atomic<size_t> read_pos{0};
    std::array<T, Size> buffer;
    
public:
    bool push(const T& item);
    bool pop(T& item);
    bool empty() const;
    bool full() const;
};
```

### Exercise 5: Thread Pool with Priority Queue

Enhance the basic thread pool to support task priorities:

```cpp
// TODO: Implement priority-based thread pool
class PriorityThreadPool {
public:
    enum class Priority { LOW, NORMAL, HIGH, CRITICAL };
    
    template<typename F, typename... Args>
    auto enqueue(Priority priority, F&& f, Args&&... args);
    
private:
    // Use priority queue for task scheduling
    // Implement fair scheduling to avoid starvation
};
```

## Advanced Topics and Patterns

### 1. Work-Stealing Queue

```cpp
// Advanced pattern for high-performance task distribution
template<typename T>
class WorkStealingQueue {
private:
    std::deque<T> queue;
    mutable std::mutex mtx;
    
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push_front(std::move(item));
    }
    
    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        
        item = std::move(queue.front());
        queue.pop_front();
        return true;
    }
    
    bool try_steal(T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        
        item = std::move(queue.back());
        queue.pop_back();
        return true;
    }
};
```

### 2. Reader-Writer Lock Implementation

```cpp
// Custom reader-writer lock using C++11 primitives
class ReaderWriterLock {
private:
    std::shared_mutex rw_mutex;  // C++14
    
public:
    class ReadLock {
        std::shared_lock<std::shared_mutex> lock;
    public:
        ReadLock(ReaderWriterLock& rwl) : lock(rwl.rw_mutex) {}
    };
    
    class WriteLock {
        std::unique_lock<std::shared_mutex> lock;
    public:
        WriteLock(ReaderWriterLock& rwl) : lock(rwl.rw_mutex) {}
    };
};
```

### 3. Atomic Reference Counting

```cpp
// Lock-free reference counting for custom smart pointers
template<typename T>
class AtomicRefPtr {
private:
    struct ControlBlock {
        std::atomic<int> ref_count{1};
        T* ptr;
        
        ControlBlock(T* p) : ptr(p) {}
    };
    
    ControlBlock* control_block;
    
public:
    explicit AtomicRefPtr(T* ptr) : control_block(new ControlBlock(ptr)) {}
    
    AtomicRefPtr(const AtomicRefPtr& other) : control_block(other.control_block) {
        if (control_block) {
            control_block->ref_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    ~AtomicRefPtr() {
        if (control_block && control_block->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete control_block->ptr;
            delete control_block;
        }
    }
    
    T* get() const { return control_block ? control_block->ptr : nullptr; }
    T& operator*() const { return *get(); }
    T* operator->() const { return get(); }
};
```

## Study Materials and Resources

### Essential Reading

**Books:**
- "C++ Concurrency in Action" by Anthony Williams (2nd Edition)
- "The Art of Multiprocessor Programming" by Maurice Herlihy and Nir Shavit
- "Effective Modern C++" by Scott Meyers (Items 34-42 on concurrency)
- "C++11 Concurrency Cookbook" by Miodrag Bolic

**Online Resources:**
- [cppreference.com - Thread support library](https://en.cppreference.com/w/cpp/thread)
- [Intel Threading Building Blocks (TBB) Documentation](https://oneapi-src.github.io/oneTBB/)
- [Herb Sutter's "atomic<> Weapons" talks](https://herbsutter.com/2013/02/11/atomic-weapons-the-c-memory-model-and-modern-hardware/)
- [C++ Standards Committee Papers on Concurrency](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/)

**Video Lectures:**
- CppCon talks on concurrency and parallelism
- "Lock-Free Programming" by Fedor Pikus (CppCon 2017)
- "The Speed of Concurrency" by Fedor Pikus (CppCon 2016)
- "Better Code: Concurrency" by Sean Parent (NDC 2017)

### Development Tools and Debugging

**Compilers and Flags:**
```bash
# GCC with thread sanitizer
g++ -std=c++11 -pthread -fsanitize=thread -g -O1 program.cpp

# Clang with memory sanitizer
clang++ -std=c++11 -pthread -fsanitize=memory -g -O1 program.cpp

# MSVC with analysis
cl /std:c++11 /analyze program.cpp
```

**Profiling Tools:**
- **Intel VTune Profiler**: Comprehensive performance analysis
- **Google Performance Tools (gperftools)**: CPU and heap profiling
- **Valgrind**: Memory error detection and profiling
  ```bash
  valgrind --tool=helgrind ./program    # Race condition detection
  valgrind --tool=drd ./program         # Alternative race detector
  ```

**Static Analysis:**
- **Clang Static Analyzer**: Built-in concurrency checks
- **PVS-Studio**: Commercial static analyzer with concurrency rules
- **Cppcheck**: Open-source static analyzer

### Practice Problems and Challenges

**Beginner Level:**
1. Implement a thread-safe counter with atomic operations
2. Create a simple message passing system using std::future
3. Build a basic producer-consumer with condition variables
4. Write a parallel file processor using std::async

**Intermediate Level:**
5. Implement a thread-safe object pool
6. Create a work-stealing thread pool
7. Build a lock-free stack with ABA problem handling
8. Design a reader-writer lock with writer preference

**Advanced Level:**
9. Implement a scalable concurrent hash map
10. Create a lock-free multi-producer, multi-consumer queue
11. Build a software transactional memory system
12. Design a distributed consensus algorithm

### Benchmarking and Performance Testing

```cpp
// Performance testing framework
class ConcurrencyBenchmark {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    template<typename Duration = std::chrono::milliseconds>
    long long stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<Duration>(end_time - start_time).count();
    }
    
    template<typename Func>
    long long measure(Func&& func) {
        start();
        func();
        return stop();
    }
};

// Usage example
void benchmark_mutex_vs_atomic() {
    ConcurrencyBenchmark bench;
    
    // Test mutex performance
    auto mutex_time = bench.measure([]() {
        // Your mutex-based code here
    });
    
    // Test atomic performance  
    auto atomic_time = bench.measure([]() {
        // Your atomic-based code here
    });
    
    std::cout << "Mutex time: " << mutex_time << " ms" << std::endl;
    std::cout << "Atomic time: " << atomic_time << " ms" << std::endl;
}
```

### Common Pitfalls and How to Avoid Them

**1. ABA Problem in Lock-Free Programming**
```cpp
// Problem: Pointer might change to same value after being different
// Solution: Use generation counters or hazard pointers
struct Tagged_Pointer {
    std::atomic<Node*> ptr;
    std::atomic<int> tag;
};
```

**2. False Sharing**
```cpp
// Problem: Multiple threads accessing different variables on same cache line
struct BadAlignment {
    alignas(64) std::atomic<int> counter1;  // Good: cache line aligned
    std::atomic<int> counter2;              // Bad: might share cache line
};
```

**3. Memory Ordering Mistakes**  
```cpp
// Always specify memory ordering explicitly for clarity
atomic_var.store(value, std::memory_order_release);  // Good
atomic_var.store(value);  // Unclear - uses seq_cst by default
```

**4. Exception Safety in Multi-threaded Code**
```cpp
// Always use RAII and exception-safe designs
class SafeResource {
    std::lock_guard<std::mutex> lock;  // RAII ensures unlock
    // Resource operations here
};
```
