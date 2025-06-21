# Thread Pool with Modern C++ Project

## Project Overview

Build a high-performance thread pool implementation using C++11 threading features, including std::thread, std::mutex, std::condition_variable, std::future, and std::packaged_task.

## Learning Objectives

- Understand C++11 threading primitives
- Implement thread-safe data structures
- Use futures and promises for asynchronous programming
- Handle thread synchronization and coordination
- Build a reusable, scalable thread pool architecture

## Project Structure

```
thread_pool_project/
├── src/
│   ├── main.cpp
│   ├── thread_pool.cpp
│   ├── thread_safe_queue.cpp
│   ├── worker_thread.cpp
│   └── task_manager.cpp
├── include/
│   ├── thread_pool.h
│   ├── thread_safe_queue.h
│   ├── worker_thread.h
│   ├── task_manager.h
│   └── task_types.h
├── examples/
│   ├── basic_usage.cpp
│   ├── performance_test.cpp
│   └── real_world_scenarios.cpp
├── tests/
│   ├── test_thread_pool.cpp
│   ├── test_thread_safe_queue.cpp
│   └── test_concurrency.cpp
└── CMakeLists.txt
```

## Core Components

### 1. Thread-Safe Queue

```cpp
// include/thread_safe_queue.h
#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

template<typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;
    bool shutdown_;
    
public:
    ThreadSafeQueue() : shutdown_(false) {}
    
    // Disable copy constructor and assignment
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
    
    // Add an item to the queue
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
            throw std::runtime_error("Queue is shut down");
        }
        queue_.push(std::move(item));
        condition_.notify_one();
    }
    
    // Try to pop an item from the queue (non-blocking)
    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    // Pop an item from the queue (blocking)
    bool wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        
        if (shutdown_ && queue_.empty()) {
            return false;  // Queue is shut down and empty
        }
        
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    // Pop an item with timeout
    template<typename Rep, typename Period>
    bool wait_for_pop(T& item, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (condition_.wait_for(lock, timeout, [this] { return !queue_.empty() || shutdown_; })) {
            if (shutdown_ && queue_.empty()) {
                return false;
            }
            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        return false;  // Timeout
    }
    
    // Get the current size of the queue
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    // Check if the queue is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    // Shutdown the queue
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        condition_.notify_all();
    }
    
    // Check if the queue is shut down
    bool is_shutdown() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return shutdown_;
    }
    
    // Clear all items from the queue
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        queue_.swap(empty);
    }
};
```

### 2. Task Types and Wrappers

```cpp
// include/task_types.h
#pragma once
#include <functional>
#include <future>
#include <memory>
#include <type_traits>

// Base task interface
class TaskBase {
public:
    virtual ~TaskBase() = default;
    virtual void execute() = 0;
    virtual bool is_valid() const = 0;
};

// Concrete task implementation
template<typename F>
class Task : public TaskBase {
private:
    F function_;
    
public:
    template<typename Func>
    Task(Func&& func) : function_(std::forward<Func>(func)) {}
    
    void execute() override {
        function_();
    }
    
    bool is_valid() const override {
        return true;
    }
};

// Task with return value
template<typename F, typename R = typename std::result_of<F()>::type>
class FutureTask : public TaskBase {
private:
    F function_;
    std::promise<R> promise_;
    
public:
    template<typename Func>
    FutureTask(Func&& func) : function_(std::forward<Func>(func)) {}
    
    std::future<R> get_future() {
        return promise_.get_future();
    }
    
    void execute() override {
        try {
            if constexpr (std::is_void_v<R>) {
                function_();
                promise_.set_value();
            } else {
                promise_.set_value(function_());
            }
        } catch (...) {
            promise_.set_exception(std::current_exception());
        }
    }
    
    bool is_valid() const override {
        return true;
    }
};

// Task priority
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

// Prioritized task wrapper
struct PrioritizedTask {
    std::unique_ptr<TaskBase> task;
    TaskPriority priority;
    std::chrono::steady_clock::time_point submit_time;
    
    PrioritizedTask(std::unique_ptr<TaskBase> t, TaskPriority p = TaskPriority::NORMAL)
        : task(std::move(t)), priority(p), submit_time(std::chrono::steady_clock::now()) {}
    
    bool operator<(const PrioritizedTask& other) const {
        // Higher priority tasks come first
        if (priority != other.priority) {
            return priority < other.priority;
        }
        // If same priority, earlier tasks come first
        return submit_time > other.submit_time;
    }
};

// Task factory functions
template<typename F>
std::unique_ptr<TaskBase> make_task(F&& func) {
    return std::make_unique<Task<std::decay_t<F>>>(std::forward<F>(func));
}

template<typename F>
auto make_future_task(F&& func) {
    using ReturnType = typename std::result_of<F()>::type;
    auto task = std::make_unique<FutureTask<std::decay_t<F>, ReturnType>>(std::forward<F>(func));
    auto future = task->get_future();
    return std::make_pair(std::move(task), std::move(future));
}
```

### 3. Thread Pool Implementation

```cpp
// include/thread_pool.h
#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <future>
#include <functional>
#include <type_traits>
#include <chrono>
#include "thread_safe_queue.h"
#include "task_types.h"

class ThreadPool {
private:
    std::vector<std::thread> workers_;
    ThreadSafeQueue<std::unique_ptr<TaskBase>> task_queue_;
    std::atomic<bool> shutdown_;
    std::atomic<size_t> active_threads_;
    std::atomic<size_t> total_tasks_executed_;
    std::atomic<size_t> total_tasks_submitted_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    std::chrono::steady_clock::time_point start_time_;
    std::vector<std::chrono::milliseconds> task_execution_times_;
    
    // Configuration
    size_t max_queue_size_;
    std::chrono::milliseconds thread_idle_timeout_;
    
    void worker_thread();
    void shutdown_workers();
    
public:
    // Configuration structure
    struct Config {
        size_t num_threads = std::thread::hardware_concurrency();
        size_t max_queue_size = 1000;
        std::chrono::milliseconds thread_idle_timeout{5000};
        bool collect_statistics = true;
    };
    
    // Constructor
    explicit ThreadPool(const Config& config = Config{});
    
    // Destructor
    ~ThreadPool();
    
    // Disable copy and move
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    
    // Submit a task without return value
    template<typename F, typename... Args>
    void submit(F&& func, Args&&... args) {
        auto task = make_task([func = std::forward<F>(func), args...]() mutable {
            func(args...);
        });
        
        if (!task_queue_.is_shutdown()) {
            task_queue_.push(std::move(task));
            ++total_tasks_submitted_;
        } else {
            throw std::runtime_error("Thread pool is shut down");
        }
    }
    
    // Submit a task with return value
    template<typename F, typename... Args>
    auto submit_with_result(F&& func, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using ReturnType = typename std::result_of<F(Args...)>::type;
        
        auto task_func = [func = std::forward<F>(func), args...]() mutable -> ReturnType {
            return func(args...);
        };
        
        auto [task, future] = make_future_task(std::move(task_func));
        
        if (!task_queue_.is_shutdown()) {
            task_queue_.push(std::move(task));
            ++total_tasks_submitted_;
            return future;
        } else {
            throw std::runtime_error("Thread pool is shut down");
        }
    }
    
    // Submit a packaged task
    template<typename F, typename... Args>
    auto submit_packaged(F&& func, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using ReturnType = typename std::result_of<F(Args...)>::type;
        
        auto packaged_task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(func), std::forward<Args>(args)...)
        );
        
        auto future = packaged_task->get_future();
        
        auto task = make_task([packaged_task]() {
            (*packaged_task)();
        });
        
        if (!task_queue_.is_shutdown()) {
            task_queue_.push(std::move(task));
            ++total_tasks_submitted_;
            return future;
        } else {
            throw std::runtime_error("Thread pool is shut down");
        }
    }
    
    // Wait for all tasks to complete
    void wait_for_all_tasks();
    
    // Shutdown the thread pool
    void shutdown();
    
    // Get current status
    struct Status {
        size_t num_threads;
        size_t active_threads;
        size_t queue_size;
        size_t total_tasks_submitted;
        size_t total_tasks_executed;
        bool is_shutdown;
        std::chrono::milliseconds uptime;
        double average_task_time_ms;
    };
    
    Status get_status() const;
    
    // Resize the thread pool
    void resize(size_t new_size);
    
    // Get queue size
    size_t get_queue_size() const { return task_queue_.size(); }
    
    // Check if shutdown
    bool is_shutdown() const { return shutdown_.load(); }
};
```

```cpp
// src/thread_pool.cpp
#include "thread_pool.h"
#include <iostream>
#include <algorithm>

ThreadPool::ThreadPool(const Config& config)
    : shutdown_(false)
    , active_threads_(0)
    , total_tasks_executed_(0)
    , total_tasks_submitted_(0)
    , max_queue_size_(config.max_queue_size)
    , thread_idle_timeout_(config.thread_idle_timeout)
    , start_time_(std::chrono::steady_clock::now()) {
    
    // Create worker threads
    workers_.reserve(config.num_threads);
    for (size_t i = 0; i < config.num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_thread, this);
    }
    
    std::cout << "ThreadPool created with " << config.num_threads << " threads" << std::endl;
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::worker_thread() {
    std::unique_ptr<TaskBase> task;
    
    while (!shutdown_.load()) {
        if (task_queue_.wait_for_pop(task, thread_idle_timeout_)) {
            if (task && task->is_valid()) {
                ++active_threads_;
                
                auto start_time = std::chrono::steady_clock::now();
                
                try {
                    task->execute();
                } catch (const std::exception& e) {
                    std::cerr << "Task execution failed: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Task execution failed with unknown exception" << std::endl;
                }
                
                auto end_time = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    task_execution_times_.push_back(duration);
                }
                
                ++total_tasks_executed_;
                --active_threads_;
                
                task.reset();  // Ensure task is destroyed
            }
        }
    }
}

void ThreadPool::shutdown() {
    if (shutdown_.exchange(true)) {
        return;  // Already shut down
    }
    
    std::cout << "Shutting down thread pool..." << std::endl;
    
    task_queue_.shutdown();
    
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    workers_.clear();
    std::cout << "Thread pool shut down complete" << std::endl;
}

void ThreadPool::wait_for_all_tasks() {
    // Wait until all submitted tasks are executed
    while (total_tasks_submitted_.load() > total_tasks_executed_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Wait until all active threads finish
    while (active_threads_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

ThreadPool::Status ThreadPool::get_status() const {
    auto now = std::chrono::steady_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    
    double avg_task_time = 0.0;
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        if (!task_execution_times_.empty()) {
            auto total_time = std::accumulate(task_execution_times_.begin(), 
                                            task_execution_times_.end(), 
                                            std::chrono::milliseconds{0});
            avg_task_time = static_cast<double>(total_time.count()) / task_execution_times_.size();
        }
    }
    
    return Status{
        workers_.size(),
        active_threads_.load(),
        task_queue_.size(),
        total_tasks_submitted_.load(),
        total_tasks_executed_.load(),
        shutdown_.load(),
        uptime,
        avg_task_time
    };
}

void ThreadPool::resize(size_t new_size) {
    if (shutdown_.load()) {
        throw std::runtime_error("Cannot resize shut down thread pool");
    }
    
    size_t current_size = workers_.size();
    
    if (new_size > current_size) {
        // Add more threads
        workers_.reserve(new_size);
        for (size_t i = current_size; i < new_size; ++i) {
            workers_.emplace_back(&ThreadPool::worker_thread, this);
        }
    } else if (new_size < current_size) {
        // Remove threads (simplified approach - just mark for shutdown)
        // In a production implementation, you'd want more sophisticated thread management
        std::cout << "Thread pool downsizing not fully implemented in this example" << std::endl;
    }
}
```

### 4. Usage Examples

```cpp
// examples/basic_usage.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "thread_pool.h"

void cpu_intensive_task(int id, int duration_ms) {
    std::cout << "Task " << id << " started" << std::endl;
    
    // Simulate CPU-intensive work
    auto start = std::chrono::steady_clock::now();
    auto end = start + std::chrono::milliseconds(duration_ms);
    
    while (std::chrono::steady_clock::now() < end) {
        // Busy wait to simulate CPU work
        for (volatile int i = 0; i < 1000; ++i) {}
    }
    
    std::cout << "Task " << id << " completed" << std::endl;
}

int computation_task(int x, int y) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return x * y + x + y;
}

void demonstrate_basic_usage() {
    std::cout << "\n=== Basic Thread Pool Usage ===" << std::endl;
    
    ThreadPool::Config config;
    config.num_threads = 4;
    config.max_queue_size = 100;
    
    ThreadPool pool(config);
    
    // Submit tasks without return values
    std::cout << "\n1. Submitting tasks without return values:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        pool.submit(cpu_intensive_task, i, 200);
    }
    
    pool.wait_for_all_tasks();
    
    // Submit tasks with return values
    std::cout << "\n2. Submitting tasks with return values:" << std::endl;
    std::vector<std::future<int>> futures;
    
    for (int i = 0; i < 5; ++i) {
        auto future = pool.submit_with_result(computation_task, i, i + 1);
        futures.push_back(std::move(future));
    }
    
    // Collect results
    for (size_t i = 0; i < futures.size(); ++i) {
        int result = futures[i].get();
        std::cout << "Result " << i << ": " << result << std::endl;
    }
    
    // Print pool status
    auto status = pool.get_status();
    std::cout << "\nPool Status:" << std::endl;
    std::cout << "  Threads: " << status.num_threads << std::endl;
    std::cout << "  Active: " << status.active_threads << std::endl;
    std::cout << "  Queue size: " << status.queue_size << std::endl;
    std::cout << "  Tasks submitted: " << status.total_tasks_submitted << std::endl;
    std::cout << "  Tasks executed: " << status.total_tasks_executed << std::endl;
    std::cout << "  Average task time: " << status.average_task_time_ms << " ms" << std::endl;
}

void demonstrate_exception_handling() {
    std::cout << "\n=== Exception Handling ===" << std::endl;
    
    ThreadPool pool;
    
    // Task that throws an exception
    auto future = pool.submit_with_result([]() -> int {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        throw std::runtime_error("Task failed!");
        return 42;
    });
    
    try {
        int result = future.get();
        std::cout << "Result: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
}

void demonstrate_packaged_tasks() {
    std::cout << "\n=== Packaged Tasks ===" << std::endl;
    
    ThreadPool pool;
    
    // Submit packaged tasks
    auto future1 = pool.submit_packaged([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return "Hello from packaged task!";
    });
    
    auto future2 = pool.submit_packaged([](int x, int y) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        return x + y;
    }, 10, 20);
    
    std::cout << "Packaged task 1 result: " << future1.get() << std::endl;
    std::cout << "Packaged task 2 result: " << future2.get() << std::endl;
}

int main() {
    try {
        demonstrate_basic_usage();
        demonstrate_exception_handling();
        demonstrate_packaged_tasks();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### 5. Performance Testing

```cpp
// examples/performance_test.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include "thread_pool.h"

class PerformanceTester {
private:
    std::mt19937 rng_;
    
public:
    PerformanceTester() : rng_(std::random_device{}()) {}
    
    // CPU-intensive task
    double cpu_task(int iterations) {
        double result = 0.0;
        for (int i = 0; i < iterations; ++i) {
            result += std::sin(i) * std::cos(i);
        }
        return result;
    }
    
    // I/O simulation task
    void io_task(int duration_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    }
    
    // Mixed workload task
    int mixed_task(int cpu_work, int io_delay) {
        // CPU work
        volatile int sum = 0;
        for (int i = 0; i < cpu_work; ++i) {
            sum += i;
        }
        
        // I/O delay
        std::this_thread::sleep_for(std::chrono::milliseconds(io_delay));
        
        return sum;
    }
    
    void test_cpu_intensive_workload(ThreadPool& pool, int num_tasks, int iterations_per_task) {
        std::cout << "\n=== CPU Intensive Workload Test ===" << std::endl;
        std::cout << "Tasks: " << num_tasks << ", Iterations per task: " << iterations_per_task << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::future<double>> futures;
        futures.reserve(num_tasks);
        
        for (int i = 0; i < num_tasks; ++i) {
            auto future = pool.submit_with_result([this, iterations_per_task]() {
                return cpu_task(iterations_per_task);
            });
            futures.push_back(std::move(future));
        }
        
        // Wait for all tasks to complete
        double total_result = 0.0;
        for (auto& future : futures) {
            total_result += future.get();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Total result: " << total_result << std::endl;
        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
        std::cout << "Tasks per second: " << (num_tasks * 1000.0) / duration.count() << std::endl;
    }
    
    void test_io_intensive_workload(ThreadPool& pool, int num_tasks, int delay_per_task) {
        std::cout << "\n=== I/O Intensive Workload Test ===" << std::endl;
        std::cout << "Tasks: " << num_tasks << ", Delay per task: " << delay_per_task << " ms" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::future<void>> futures;
        futures.reserve(num_tasks);
        
        for (int i = 0; i < num_tasks; ++i) {
            auto future = pool.submit_with_result([this, delay_per_task]() {
                io_task(delay_per_task);
            });
            futures.push_back(std::move(future));
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.get();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
        std::cout << "Expected sequential time: " << (num_tasks * delay_per_task) << " ms" << std::endl;
        std::cout << "Speedup: " << (double)(num_tasks * delay_per_task) / duration.count() << "x" << std::endl;
    }
    
    void test_mixed_workload(ThreadPool& pool, int num_tasks) {
        std::cout << "\n=== Mixed Workload Test ===" << std::endl;
        std::cout << "Tasks: " << num_tasks << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::future<int>> futures;
        futures.reserve(num_tasks);
        
        std::uniform_int_distribution<int> cpu_dist(1000, 10000);
        std::uniform_int_distribution<int> io_dist(10, 100);
        
        for (int i = 0; i < num_tasks; ++i) {
            int cpu_work = cpu_dist(rng_);
            int io_delay = io_dist(rng_);
            
            auto future = pool.submit_with_result([this, cpu_work, io_delay]() {
                return mixed_task(cpu_work, io_delay);
            });
            futures.push_back(std::move(future));
        }
        
        // Wait for all tasks to complete
        std::vector<int> results;
        results.reserve(num_tasks);
        
        for (auto& future : futures) {
            results.push_back(future.get());
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        int total_result = std::accumulate(results.begin(), results.end(), 0);
        
        std::cout << "Total result: " << total_result << std::endl;
        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
        std::cout << "Average time per task: " << (double)duration.count() / num_tasks << " ms" << std::endl;
    }
    
    void compare_thread_counts() {
        std::cout << "\n=== Thread Count Comparison ===" << std::endl;
        
        std::vector<size_t> thread_counts = {1, 2, 4, 8, std::thread::hardware_concurrency()};
        const int num_tasks = 100;
        const int iterations_per_task = 50000;
        
        for (size_t thread_count : thread_counts) {
            std::cout << "\nTesting with " << thread_count << " threads:" << std::endl;
            
            ThreadPool::Config config;
            config.num_threads = thread_count;
            ThreadPool pool(config);
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            std::vector<std::future<double>> futures;
            futures.reserve(num_tasks);
            
            for (int i = 0; i < num_tasks; ++i) {
                auto future = pool.submit_with_result([this, iterations_per_task]() {
                    return cpu_task(iterations_per_task);
                });
                futures.push_back(std::move(future));
            }
            
            // Wait for completion
            for (auto& future : futures) {
                future.get();
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "  Execution time: " << duration.count() << " ms" << std::endl;
            std::cout << "  Throughput: " << (num_tasks * 1000.0) / duration.count() << " tasks/sec" << std::endl;
        }
    }
};

int main() {
    try {
        PerformanceTester tester;
        
        // Test with different thread pool configurations
        ThreadPool::Config config;
        config.num_threads = std::thread::hardware_concurrency();
        config.max_queue_size = 1000;
        ThreadPool pool(config);
        
        tester.test_cpu_intensive_workload(pool, 50, 100000);
        tester.test_io_intensive_workload(pool, 20, 100);
        tester.test_mixed_workload(pool, 30);
        
        pool.wait_for_all_tasks();
        
        auto status = pool.get_status();
        std::cout << "\nFinal Pool Statistics:" << std::endl;
        std::cout << "  Total tasks submitted: " << status.total_tasks_submitted << std::endl;
        std::cout << "  Total tasks executed: " << status.total_tasks_executed << std::endl;
        std::cout << "  Average task execution time: " << status.average_task_time_ms << " ms" << std::endl;
        std::cout << "  Pool uptime: " << status.uptime.count() << " ms" << std::endl;
        
        // Compare different thread counts
        tester.compare_thread_counts();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Build Configuration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(ThreadPool)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Threads package
find_package(Threads REQUIRED)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/thread_pool.cpp
)

# Create library
add_library(thread_pool_lib ${SOURCES})
target_link_libraries(thread_pool_lib Threads::Threads)

# Examples
add_executable(basic_usage examples/basic_usage.cpp)
target_link_libraries(basic_usage thread_pool_lib)

add_executable(performance_test examples/performance_test.cpp)
target_link_libraries(performance_test thread_pool_lib)

# Tests (if GTest is available)
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(thread_pool_tests
        tests/test_thread_pool.cpp
        tests/test_thread_safe_queue.cpp
    )
    target_link_libraries(thread_pool_tests GTest::gtest_main thread_pool_lib)
    
    enable_testing()
    add_test(NAME ThreadPool_Tests COMMAND thread_pool_tests)
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(thread_pool_lib PRIVATE /W4)
else()
    target_compile_options(thread_pool_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()
```

## Expected Learning Outcomes

After completing this project, you should understand:

1. **Threading Fundamentals**
   - std::thread lifecycle and management
   - Thread synchronization with mutexes and condition variables
   - Thread-safe data structures

2. **Asynchronous Programming**
   - std::future and std::promise
   - std::packaged_task for function wrapping
   - Exception handling in asynchronous contexts

3. **Concurrency Patterns**
   - Producer-consumer pattern with thread-safe queues
   - Worker thread pool architecture
   - Task scheduling and load balancing

4. **Performance Optimization**
   - Measuring concurrent performance
   - Thread pool sizing strategies
   - Resource utilization optimization

## Extensions and Improvements

1. **Advanced Features**
   - Priority-based task scheduling
   - Thread pool auto-scaling
   - Task cancellation support
   - Work-stealing algorithms

2. **Monitoring and Debugging**
   - Thread pool metrics and monitoring
   - Deadlock detection
   - Performance profiling integration

3. **Real-world Applications**
   - Web server request processing
   - Image/video processing pipelines
   - Parallel algorithm implementations

This project provides a solid foundation for understanding modern C++ concurrency and building scalable multithreaded applications.
