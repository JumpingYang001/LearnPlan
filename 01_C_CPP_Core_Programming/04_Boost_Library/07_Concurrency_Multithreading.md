# Concurrency and Multithreading

*Duration: 2 weeks*

# Boost Concurrency and Multithreading

*Duration: 2 weeks*

## Overview

This comprehensive section covers Boost's powerful concurrency libraries, providing high-level abstractions for threading, asynchronous I/O, and fiber-based concurrency. You'll learn how Boost enhances standard C++ threading capabilities and provides portable, efficient solutions for modern concurrent programming.

### Why Boost for Concurrency?

**Advantages over Standard C++ Threading:**
- **Higher-level abstractions** - Simplified APIs for complex operations
- **Cross-platform compatibility** - Works consistently across different OS
- **Advanced synchronization primitives** - Beyond basic mutexes and condition variables
- **Asynchronous I/O framework** - Built-in support for network programming
- **Fiber support** - Lightweight cooperative multitasking
- **Proven stability** - Battle-tested in production environments

**Key Libraries Covered:**
1. **Boost.Thread** - Enhanced threading primitives
2. **Boost.Asio** - Asynchronous I/O and networking
3. **Boost.Fiber** - User-space cooperative threading

## Prerequisites

Before starting this section, ensure you understand:
- ✅ Basic C++ multithreading concepts (mutexes, condition variables)
- ✅ RAII and smart pointers
- ✅ Lambda functions and std::function
- ✅ Exception handling
- ✅ Basic networking concepts (sockets, TCP/UDP)

## Learning Topics

### 1. Boost.Thread - Enhanced Threading Capabilities

#### Core Concepts
- **Thread Management**: Creation, lifecycle, and cleanup
- **Advanced Synchronization**: Mutexes, condition variables, barriers
- **Thread-local Storage**: Per-thread data management
- **Thread Groups**: Managing multiple threads as a unit
- **Thread Interruption**: Cooperative cancellation mechanism
- **Futures and Promises**: Asynchronous result handling

#### Why Choose Boost.Thread over std::thread?

| Feature | std::thread | Boost.Thread |
|---------|-------------|--------------|
| **Interruption Support** | No | Yes - cooperative cancellation |
| **Thread Groups** | No | Yes - manage multiple threads |
| **Timed Operations** | Limited | Extensive timeout support |
| **Cross-platform** | C++11+ only | Works with older compilers |
| **Advanced Synchronization** | Basic | Rich set of primitives |

### 2. Boost.Asio - Asynchronous I/O Framework

#### Core Concepts
- **I/O Context**: Central event loop for asynchronous operations
- **Asynchronous Operations**: Non-blocking I/O with callbacks
- **Networking**: TCP/UDP sockets, acceptors, resolvers
- **Timers**: Scheduled and repeating operations
- **Coroutines**: Structured asynchronous programming
- **SSL/TLS Support**: Secure communications

#### Asynchronous Programming Model
```
Traditional Blocking I/O:
Thread 1: [Read File] ──────────────> [Process Data]
Thread 2: [Network Call] ──────────> [Handle Response]
Thread 3: [Database Query] ────────> [Update UI]

Boost.Asio Asynchronous I/O:
Single Thread: [Start Read] [Start Network] [Start DB Query]
                     ↓             ↓              ↓
               [File Done] [Network Done] [DB Done]
                     ↓             ↓              ↓
              [Process Data] [Handle Response] [Update UI]
```

### 3. Boost.Fiber - Cooperative Multitasking

#### Core Concepts
- **User-space Threads**: Lightweight threads managed by your application
- **Cooperative Scheduling**: Fibers yield control voluntarily
- **Fiber Synchronization**: Mutexes, condition variables for fibers
- **Channels**: Communication between fibers
- **Schedulers**: Different scheduling algorithms
- **Integration**: Working with asynchronous I/O

#### Threads vs Fibers Comparison

| Aspect | OS Threads | Boost.Fibers |
|--------|------------|--------------|
| **Creation Cost** | High (kernel call) | Very Low (user-space) |
| **Memory Usage** | ~8MB stack | ~64KB stack (configurable) |
| **Context Switch** | Expensive (kernel) | Cheap (user-space) |
| **Scheduling** | Preemptive | Cooperative |
| **Scalability** | Limited (~1000s) | High (~100,000s) |
| **Debugging** | Complex | Easier (single-threaded) |

## Code Examples and Detailed Explanations

### Part 1: Boost.Thread Fundamentals

#### Understanding Boost.Thread Advantages

**Enhanced Thread Creation and Management**
```cpp
#include <boost/thread.hpp>
#include <boost/thread/thread_group.hpp>
#include <iostream>
#include <vector>

// Traditional std::thread approach
void std_thread_example() {
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([i]() {
            std::cout << "std::thread " << i << " running\n";
        });
    }
    
    // Must manually join each thread
    for (auto& t : threads) {
        t.join();
    }
}

// Boost.Thread with thread groups - much cleaner!
void boost_thread_group_example() {
    boost::thread_group threads;
    
    for (int i = 0; i < 5; ++i) {
        threads.create_thread([i]() {
            std::cout << "boost::thread " << i << " running\n";
        });
    }
    
    // Join all threads with one call
    threads.join_all();
    std::cout << "All threads completed\n";
}
```

#### Thread-Safe Data Structures with Boost.Thread

The `ThreadSafeQueue` below demonstrates several key Boost.Thread features:
- **boost::mutex** - More portable than std::mutex
- **boost::condition_variable** - Enhanced condition variables
- **boost::unique_lock** - RAII lock management
- **Timed operations** - `wait_for` with timeout support

**Key Learning Points:**
1. **Thread Safety**: All operations are protected by mutex
2. **Blocking vs Non-blocking**: `pop()` blocks, `try_pop()` times out
3. **RAII Pattern**: Locks automatically released on scope exit
4. **Condition Variables**: Efficient waiting for state changes

```cpp
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <iostream>
#include <queue>

class ThreadSafeQueue {
public:
    void push(int value) {
        boost::unique_lock<boost::mutex> lock(mutex_);
        queue_.push(value);
        condition_.notify_one();
    }
    
    int pop() {
        boost::unique_lock<boost::mutex> lock(mutex_);
        while (queue_.empty()) {
            condition_.wait(lock);
        }
        int value = queue_.front();
        queue_.pop();
        return value;
    }
    
    bool try_pop(int& value, const boost::chrono::milliseconds& timeout) {
        boost::unique_lock<boost::mutex> lock(mutex_);
        if (condition_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            value = queue_.front();
            queue_.pop();
            return true;
        }
        return false;
    }
    
    size_t size() const {
        boost::lock_guard<boost::mutex> lock(mutex_);
        return queue_.size();
    }
    
private:
    mutable boost::mutex mutex_;
    boost::condition_variable condition_;
    std::queue<int> queue_;
};

void producer(ThreadSafeQueue& queue, int id, int count) {
    for (int i = 0; i < count; ++i) {
        int value = id * 1000 + i;
        queue.push(value);
        std::cout << "Producer " << id << " pushed: " << value << "\n";
        boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    }
}

void consumer(ThreadSafeQueue& queue, int id) {
    for (int i = 0; i < 5; ++i) {
        int value;
        if (queue.try_pop(value, boost::chrono::milliseconds(500))) {
            std::cout << "Consumer " << id << " popped: " << value << "\n";
        } else {
            std::cout << "Consumer " << id << " timed out\n";
        }
    }
}

void demonstrate_thread_basics() {
    ThreadSafeQueue queue;
    
    // Create producer threads
    boost::thread_group producers;
    for (int i = 1; i <= 2; ++i) {
        producers.create_thread([&queue, i] { producer(queue, i, 3); });
    }
    
    // Create consumer threads
    boost::thread_group consumers;
    for (int i = 1; i <= 2; ++i) {
        consumers.create_thread([&queue, i] { consumer(queue, i); });
    }
    
    // Wait for all threads
    producers.join_all();
    consumers.join_all();
    
    std::cout << "Remaining items in queue: " << queue.size() << "\n";
}
```

#### Understanding Thread Interruption - A Boost.Thread Exclusive Feature

**What is Thread Interruption?**
Thread interruption is a cooperative cancellation mechanism unique to Boost.Thread. Unlike forceful thread termination (which can cause resource leaks), interruption allows threads to cleanly exit at safe points.

**How It Works:**
1. Thread calls `boost::this_thread::interruption_point()`
2. If interrupted, throws `boost::thread_interrupted` exception
3. Thread can catch exception and perform cleanup
4. Much safer than killing threads forcefully

**Interruption Points:**
- `boost::this_thread::sleep_for()`
- `boost::this_thread::interruption_point()`
- Many Boost.Thread blocking operations
- Custom interruption points you define

**Real-world Use Case:** Background tasks that need graceful shutdown

### Thread Interruption and Cancellation
```cpp
#include <boost/thread.hpp>
#include <iostream>

class InterruptibleWorker {
public:
    void operator()() {
        try {
            for (int i = 0; i < 100; ++i) {
                // Check for interruption
                boost::this_thread::interruption_point();
                
                // Simulate work
                boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
                
                std::cout << "Working... step " << i << "\n";
                
                // Another interruption point
                boost::this_thread::interruption_point();
            }
            std::cout << "Work completed normally\n";
        } catch (const boost::thread_interrupted&) {
            std::cout << "Thread was interrupted\n";
            // Perform cleanup here
        }
    }
};

void demonstrate_thread_interruption() {
    std::cout << "Starting interruptible worker...\n";
    
    InterruptibleWorker worker;
    boost::thread worker_thread(worker);
    
    // Let it work for a while
    boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
    
    // Interrupt the thread
    std::cout << "Interrupting worker thread...\n";
    worker_thread.interrupt();
    
    // Wait for thread to finish
    worker_thread.join();
    std::cout << "Worker thread finished\n";
}

// Disable interruption for critical sections
void critical_work_with_interruption_disabled() {
    boost::this_thread::disable_interruption di;
    
    // Critical work that shouldn't be interrupted
    std::cout << "Performing critical work (interruption disabled)\n";
    boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
    
    {
        // Re-enable interruption temporarily
        boost::this_thread::restore_interruption ri(di);
        boost::this_thread::interruption_point();
    }
    
    std::cout << "Critical work completed\n";
}
```

**Key Interruption Concepts Demonstrated:**

1. **Interruption Points**: Specific locations where thread checks for interruption
2. **Exception Handling**: `boost::thread_interrupted` provides clean cleanup
3. **Disable/Restore**: Control interruption for critical sections
4. **Cooperative Nature**: Thread must actively check for interruption

**When to Use Thread Interruption:**
- ✅ Background processing tasks
- ✅ Network operations with timeouts
- ✅ File processing with cancellation support
- ❌ Real-time systems (exception overhead)
- ❌ Simple short-lived tasks

---

### Part 2: Boost.Asio - Asynchronous I/O Mastery

#### Understanding Asynchronous Programming Model

**The Problem with Blocking I/O:**
```cpp
// Blocking approach - inefficient for servers
void handle_client(tcp::socket socket) {
    char buffer[1024];
    socket.read_some(boost::asio::buffer(buffer));  // BLOCKS here
    // While blocked, server can't handle other clients
    socket.write_some(boost::asio::buffer("Response"));
}
```

**The Asynchronous Solution:**
```cpp
// Non-blocking approach - efficient for servers
void handle_client_async(tcp::socket socket) {
    auto buffer = std::make_shared<std::array<char, 1024>>();
    socket.async_read_some(boost::asio::buffer(*buffer),
        [buffer](boost::system::error_code ec, std::size_t length) {
            // This callback runs when data arrives
            // Server can handle other clients while waiting
        });
}
```

**Key Benefits of Asynchronous I/O:**
- **Scalability**: Handle thousands of connections with few threads
- **Responsiveness**: Never block waiting for slow operations
- **Resource Efficiency**: No thread per connection needed
- **Performance**: Eliminate context switching overhead

### Boost.Asio - Networking Fundamentals
```cpp
#include <boost/asio.hpp>
#include <iostream>
#include <string>
#include <memory>

using boost::asio::ip::tcp;

class EchoServer {
public:
    EchoServer(boost::asio::io_context& io_context, short port)
        : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {
        start_accept();
    }
    
private:
    class Session : public std::enable_shared_from_this<Session> {
    public:
        Session(tcp::socket socket) : socket_(std::move(socket)) {}
        
        void start() {
            do_read();
        }
        
    private:
        void do_read() {
            auto self = shared_from_this();
            socket_.async_read_some(
                boost::asio::buffer(data_, max_length),
                [this, self](boost::system::error_code ec, std::size_t length) {
                    if (!ec) {
                        do_write(length);
                    }
                });
        }
        
        void do_write(std::size_t length) {
            auto self = shared_from_this();
            boost::asio::async_write(
                socket_, 
                boost::asio::buffer(data_, length),
                [this, self](boost::system::error_code ec, std::size_t /*length*/) {
                    if (!ec) {
                        do_read();
                    }
                });
        }
        
        tcp::socket socket_;
        enum { max_length = 1024 };
        char data_[max_length];
    };
    
    void start_accept() {
        acceptor_.async_accept(
            [this](boost::system::error_code ec, tcp::socket socket) {
                if (!ec) {
                    std::cout << "New client connected\n";
                    std::make_shared<Session>(std::move(socket))->start();
                }
                start_accept();
            });    }
    
    tcp::acceptor acceptor_;
};

**Understanding the Echo Server Architecture:**

**Key Components:**
1. **io_context**: The heart of Asio - manages the event loop
2. **acceptor**: Listens for incoming connections
3. **Session**: Handles each client connection independently
4. **async_* functions**: Non-blocking operations with callbacks

**Asynchronous Flow:**
```
1. acceptor_.async_accept() → Wait for connection (non-blocking)
2. New client connects → Callback creates Session
3. Session::do_read() → Wait for data (non-blocking)
4. Data arrives → Callback triggers do_write()
5. Write completes → Callback triggers next do_read()
6. Cycle continues until client disconnects
```

**Why shared_ptr and enable_shared_from_this?**
- **Problem**: Callback might execute after Session object destroyed
- **Solution**: shared_ptr keeps Session alive during async operations
- **Pattern**: Always capture `auto self = shared_from_this()` in lambdas

void demonstrate_asio_server() {
    try {
        boost::asio::io_context io_context;
        
        EchoServer server(io_context, 12345);
        std::cout << "Echo server started on port 12345\n";
        
        // Run for a limited time for demonstration
        io_context.run_for(boost::chrono::seconds(30));
        
    } catch (std::exception& e) {
        std::cerr << "Server error: " << e.what() << "\n";
    }
}

**Testing the Server:**
```bash
# Terminal 1: Start server
./asio_server

# Terminal 2: Test with telnet
telnet localhost 12345
# Type messages - they'll be echoed back
```

// Simple client
void demonstrate_asio_client() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        
        tcp::resolver resolver(io_context);
        auto endpoints = resolver.resolve("localhost", "12345");
        
        boost::asio::connect(socket, endpoints);
        
        std::string message = "Hello, Echo Server!";
        boost::asio::write(socket, boost::asio::buffer(message));
        
        char reply[1024];
        size_t reply_length = socket.read_some(boost::asio::buffer(reply));
        
        std::cout << "Server replied: ";
        std::cout.write(reply, reply_length);
        std::cout << "\n";
        
    } catch (std::exception& e) {
        std::cerr << "Client error: " << e.what() << "\n";
    }
}
```

---

### Part 3: Boost.Asio Timers - Scheduling Without Threads

**Why Asynchronous Timers?**
Traditional timing approaches have limitations:
- `sleep()` blocks the entire thread
- Thread-based timers consume OS resources
- Hard to cancel or reschedule

**Boost.Asio Timers Benefits:**
- ✅ Non-blocking - other operations continue
- ✅ Efficient - no threads needed
- ✅ Cancellable - easy to stop/reschedule
- ✅ Precise - high-resolution timing
- ✅ Integrated - works with other Asio operations

### Asynchronous Timers and Scheduling
```cpp
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <iostream>
#include <functional>

class TimerManager {
public:
    TimerManager(boost::asio::io_context& io_context) 
        : io_context_(io_context) {}
    
    void schedule_once(int delay_ms, std::function<void()> callback) {
        auto timer = std::make_shared<boost::asio::steady_timer>(
            io_context_, boost::chrono::milliseconds(delay_ms));
        
        timer->async_wait([timer, callback](const boost::system::error_code& ec) {
            if (!ec) {
                callback();
            }
        });
    }
    
    void schedule_repeating(int interval_ms, std::function<void()> callback) {
        auto timer = std::make_shared<boost::asio::steady_timer>(
            io_context_, boost::chrono::milliseconds(interval_ms));
        
        start_repeating_timer(timer, interval_ms, callback);
    }
    
private:
    void start_repeating_timer(
        std::shared_ptr<boost::asio::steady_timer> timer,
        int interval_ms,
        std::function<void()> callback) {
        
        timer->async_wait([this, timer, interval_ms, callback](
            const boost::system::error_code& ec) {
            if (!ec) {
                callback();
                
                // Reschedule
                timer->expires_after(boost::chrono::milliseconds(interval_ms));
                start_repeating_timer(timer, interval_ms, callback);
            }
        });
    }
    
    boost::asio::io_context& io_context_;
};

void demonstrate_timers() {
    boost::asio::io_context io_context;
    TimerManager timer_manager(io_context);
    
    // Schedule one-time events
    timer_manager.schedule_once(1000, []() {
        std::cout << "One-time event after 1 second\n";
    });
    
    timer_manager.schedule_once(2500, []() {
        std::cout << "One-time event after 2.5 seconds\n";
    });
    
    // Schedule repeating event
    int counter = 0;
    timer_manager.schedule_repeating(500, [&counter]() {
        std::cout << "Repeating event #" << ++counter << "\n";
    });
    
    // Run for 5 seconds
    std::cout << "Starting timer demonstration...\n";
    io_context.run_for(boost::chrono::seconds(5));
    std::cout << "Timer demonstration completed\n";
}
```

---

### Part 4: Boost.Fiber - Lightweight Cooperative Threading

**What are Fibers?**
Fibers are user-space threads that run cooperatively within a single OS thread. Think of them as "lightweight threads" that:
- Are scheduled by your application, not the OS
- Voluntarily yield control to other fibers
- Have much smaller memory footprint than OS threads
- Allow creating hundreds of thousands of concurrent tasks

**When to Use Fibers:**
- ✅ **High concurrency needs** (10,000+ concurrent tasks)
- ✅ **I/O-heavy applications** (web servers, databases)
- ✅ **Cooperative workflows** (pipelines, state machines)
- ✅ **Memory-constrained environments**
- ❌ CPU-intensive tasks (no parallelism within single thread)
- ❌ Existing threaded code (requires rewriting)

**Fiber vs Thread Mental Model:**
```
OS Threads (Preemptive):
Thread 1: [Work] ──OS interrupts──> [Suspended]
Thread 2: [Work] ──OS interrupts──> [Suspended]
Thread 3: [Work] ──OS interrupts──> [Suspended]

Fibers (Cooperative):
Fiber 1: [Work] ──yield()──> [Suspended]
Fiber 2: [Work] ──yield()──> [Suspended]  
Fiber 3: [Work] ──yield()──> [Suspended]
```

### Boost.Fiber - Cooperative Multitasking
```cpp
#include <boost/fiber/all.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace fibers = boost::fibers;

class FiberSafeQueue {
public:
    void push(int value) {
        std::unique_lock<fibers::mutex> lock(mutex_);
        queue_.push_back(value);
        condition_.notify_one();
    }
    
    int pop() {
        std::unique_lock<fibers::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        int value = queue_.front();
        queue_.erase(queue_.begin());
        return value;
    }
    
    bool try_pop(int& value, const std::chrono::milliseconds& timeout) {
        std::unique_lock<fibers::mutex> lock(mutex_);
        if (condition_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            value = queue_.front();
            queue_.erase(queue_.begin());
            return true;
        }
        return false;
    }
    
private:
    fibers::mutex mutex_;
    fibers::condition_variable condition_;
    std::vector<int> queue_;
};

void fiber_producer(FiberSafeQueue& queue, int id, int count) {
    for (int i = 0; i < count; ++i) {
        int value = id * 100 + i;
        queue.push(value);
        std::cout << "Fiber producer " << id << " pushed: " << value << "\n";
        
        // Yield to other fibers
        boost::this_fiber::sleep_for(std::chrono::milliseconds(50));
    }
}

void fiber_consumer(FiberSafeQueue& queue, int id, int count) {
    for (int i = 0; i < count; ++i) {
        int value;
        if (queue.try_pop(value, std::chrono::milliseconds(200))) {
            std::cout << "Fiber consumer " << id << " popped: " << value << "\n";
        } else {
            std::cout << "Fiber consumer " << id << " timed out\n";
        }
        
        // Yield to other fibers
        boost::this_fiber::yield();
    }
}

void demonstrate_fibers() {
    FiberSafeQueue queue;
    
    std::vector<fibers::fiber> fiber_group;
    
    // Create producer fibers
    for (int i = 1; i <= 3; ++i) {
        fiber_group.emplace_back([&queue, i] {
            fiber_producer(queue, i, 2);
        });
    }
    
    // Create consumer fibers
    for (int i = 1; i <= 2; ++i) {
        fiber_group.emplace_back([&queue, i] {
            fiber_consumer(queue, i, 3);
        });
    }
    
    // Wait for all fibers to complete
    for (auto& f : fiber_group) {
        f.join();
    }
    
    std::cout << "All fibers completed\n";
}
```

### Producer-Consumer with Channels
```cpp
#include <boost/fiber/all.hpp>
#include <iostream>
#include <string>

namespace fibers = boost::fibers;

void demonstrate_fiber_channels() {
    // Buffered channel
    fibers::buffered_channel<std::string> channel(5);
    
    // Producer fiber
    fibers::fiber producer([&channel] {
        std::vector<std::string> messages = {
            "Hello", "World", "From", "Fiber", "Producer"
        };
        
        for (const auto& msg : messages) {
            channel.push(msg);
            std::cout << "Produced: " << msg << "\n";
            boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
        }
        
        channel.close();
        std::cout << "Producer finished\n";
    });
    
    // Consumer fiber
    fibers::fiber consumer([&channel] {
        std::string message;
        while (fibers::channel_op_status::success == channel.pop(message)) {
            std::cout << "Consumed: " << message << "\n";
            boost::this_fiber::sleep_for(std::chrono::milliseconds(150));
        }
        std::cout << "Consumer finished\n";
    });
    
    producer.join();
    consumer.join();
}
```

### Asynchronous HTTP Client
```cpp
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <iostream>
#include <string>

using boost::asio::ip::tcp;

class HttpClient {
public:
    HttpClient(boost::asio::io_context& io_context)
        : io_context_(io_context), socket_(io_context) {}
    
    void get(const std::string& host, const std::string& path,
             std::function<void(const std::string&)> callback) {
        
        tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(host, "80");
        
        boost::asio::async_connect(socket_, endpoints,
            [this, host, path, callback](boost::system::error_code ec, tcp::endpoint) {
                if (!ec) {
                    send_request(host, path, callback);
                } else {
                    callback("Connection error: " + ec.message());
                }
            });
    }
    
private:
    void send_request(const std::string& host, const std::string& path,
                     std::function<void(const std::string&)> callback) {
        
        std::string request = 
            "GET " + path + " HTTP/1.1\r\n"
            "Host: " + host + "\r\n"
            "Connection: close\r\n\r\n";
        
        boost::asio::async_write(socket_, boost::asio::buffer(request),
            [this, callback](boost::system::error_code ec, std::size_t) {
                if (!ec) {
                    read_response(callback);
                } else {
                    callback("Write error: " + ec.message());
                }
            });
    }
    
    void read_response(std::function<void(const std::string&)> callback) {
        auto buffer = std::make_shared<boost::asio::streambuf>();
        
        boost::asio::async_read_until(socket_, *buffer, "\r\n\r\n",
            [this, buffer, callback](boost::system::error_code ec, std::size_t) {
                if (!ec) {
                    read_content(buffer, callback);
                } else {
                    callback("Read error: " + ec.message());
                }
            });
    }
    
    void read_content(std::shared_ptr<boost::asio::streambuf> buffer,
                     std::function<void(const std::string&)> callback) {
        
        boost::asio::async_read(socket_, *buffer,
            [buffer, callback](boost::system::error_code ec, std::size_t) {
                std::string response;
                if (ec == boost::asio::error::eof) {
                    // Expected end of connection
                    std::istream stream(buffer.get());
                    std::string line;
                    while (std::getline(stream, line)) {
                        response += line + "\n";
                    }
                }
                callback(response);
            });
    }
    
    boost::asio::io_context& io_context_;
    tcp::socket socket_;
};

void demonstrate_http_client() {
    boost::asio::io_context io_context;
    
    HttpClient client(io_context);
    
    client.get("httpbin.org", "/json", [](const std::string& response) {
        std::cout << "HTTP Response received:\n";
        std::cout << response.substr(0, 500) << "...\n";
    });
    
    std::cout << "Starting HTTP request...\n";
    io_context.run();
    std::cout << "HTTP request completed\n";
}
```

## Practical Exercises and Hands-on Projects

### Beginner Level (Week 1)

#### Exercise 1: Thread-Safe Counter with Statistics
**Goal**: Implement a thread-safe counter that tracks statistics
```cpp
class StatsCounter {
    // TODO: Implement thread-safe counter with:
    // - increment() and decrement() methods
    // - get_count() and get_stats() methods
    // - Track: total operations, average ops/second
    // - Use Boost.Thread synchronization
};

// Requirements:
// 1. Multiple threads incrementing/decrementing simultaneously
// 2. One thread periodically printing statistics
// 3. Graceful shutdown with final statistics
```

**Learning Objectives:**
- Master Boost.Thread mutex and condition variables
- Understand thread-safe design patterns
- Practice RAII with lock guards

#### Exercise 2: Basic Asio Echo Server Enhancement
**Goal**: Extend the echo server with additional features
```cpp
// TODO: Enhance the EchoServer to support:
// 1. Connection limits (max N concurrent clients)
// 2. Logging of all connections and messages
// 3. Graceful shutdown on SIGINT
// 4. Basic statistics (bytes transferred, connections served)
```

**Learning Objectives:**
- Understand Asio's asynchronous patterns
- Handle server lifecycle management
- Practice error handling in async code

### Advanced Level (Week 2)

#### Exercise 3: Multi-threaded Work Queue with Fibers
**Goal**: Create a hybrid threading model using both threads and fibers
```cpp
// TODO: Implement WorkQueue that:
// 1. Uses multiple OS threads as workers
// 2. Each thread runs multiple fibers
// 3. Tasks can be CPU-bound or I/O-bound
// 4. Dynamic load balancing between threads
// 5. Priority-based task scheduling

class HybridWorkQueue {
    // Your implementation here
};
```

**Learning Objectives:**
- Combine different concurrency models
- Understand when to use threads vs fibers
- Implement sophisticated scheduling algorithms

#### Exercise 4: Asynchronous HTTP Server
**Goal**: Build a production-ready HTTP server using Boost.Asio
```cpp
// TODO: Create HttpServer that supports:
// 1. HTTP/1.1 with keep-alive connections
// 2. Static file serving from disk
// 3. REST API endpoints (GET/POST/PUT/DELETE)
// 4. Request/response logging
// 5. Configuration via JSON file
// 6. Graceful shutdown and reload

class HttpServer {
    // Your implementation here
};
```

**Learning Objectives:**
- Build real-world asynchronous applications
- Handle complex protocol implementation
- Practice production-ready error handling

### Expert Level Challenges

#### Challenge 1: Distributed Task Scheduler
Build a distributed system using Boost.Asio for networking:
- Master node distributes tasks to workers
- Worker nodes execute tasks and report results
- Fault tolerance (worker failures, network partitions)
- Load balancing and task reassignment
- Web interface for monitoring

#### Challenge 2: High-Performance Game Server
Create a real-time multiplayer game server:
- Handle 1000+ concurrent connections
- Sub-100ms response times
- Custom binary protocol
- State synchronization between clients
- Anti-cheat mechanisms

#### Challenge 3: Database Connection Pool
Implement a sophisticated connection pool:
- Multiple database backends
- Connection health monitoring
- Load balancing across replicas
- Query caching and optimization
- Metrics and monitoring integration

## Performance Considerations and Optimization

### Threading Performance Deep Dive

#### Thread Creation and Management Costs
```cpp
#include <chrono>
#include <boost/thread.hpp>

void benchmark_thread_creation() {
    const int num_threads = 1000;
    
    // Measure thread creation overhead
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<boost::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([]() {
            // Minimal work
            boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Created and joined " << num_threads 
              << " threads in " << duration.count() << "ms\n";
    std::cout << "Average per thread: " << duration.count() / double(num_threads) << "ms\n";
}
```

**Key Insights:**
- **Thread creation cost**: ~1-2ms per thread on typical systems
- **Memory overhead**: ~8MB stack per thread by default  
- **Context switch cost**: ~5-15 microseconds
- **Recommendation**: Use thread pools to amortize creation costs

#### Synchronization Overhead Analysis
```cpp
#include <boost/thread.hpp>
#include <atomic>

void benchmark_synchronization() {
    const int iterations = 10000000;
    boost::mutex mutex;
    std::atomic<int> atomic_counter{0};
    int unsync_counter = 0;
    
    // Benchmark 1: No synchronization (baseline)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ++unsync_counter;
    }
    auto baseline = std::chrono::high_resolution_clock::now() - start;
    
    // Benchmark 2: Mutex synchronization
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        boost::lock_guard<boost::mutex> lock(mutex);
        ++unsync_counter;
    }
    auto mutex_time = std::chrono::high_resolution_clock::now() - start;
    
    // Benchmark 3: Atomic operations
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        atomic_counter.fetch_add(1);
    }
    auto atomic_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "Synchronization Overhead Analysis:\n";
    std::cout << "Baseline (no sync): " << baseline.count() << "ns\n";
    std::cout << "Mutex overhead: " << (mutex_time - baseline).count() << "ns\n";
    std::cout << "Atomic overhead: " << (atomic_time - baseline).count() << "ns\n";
}
```

**Performance Guidelines:**
- **Atomic operations**: ~5-20x faster than mutexes for simple operations
- **Lock-free algorithms**: Consider for high-contention scenarios
- **Critical section size**: Keep as small as possible
- **False sharing**: Avoid accessing adjacent cache lines from different threads

### Asynchronous I/O Performance Optimization

#### Memory Management in Async Operations
```cpp
// BAD: Frequent allocations in hot path
void inefficient_async_handler() {
    auto buffer = std::make_shared<std::vector<char>>(1024);  // Allocation!
    socket.async_read_some(boost::asio::buffer(*buffer),
        [buffer](boost::system::error_code ec, std::size_t length) {
            // Process data
        });
}

// GOOD: Buffer pooling
class BufferPool {
    std::queue<std::shared_ptr<std::vector<char>>> available_buffers_;
    boost::mutex mutex_;
    
public:
    std::shared_ptr<std::vector<char>> get_buffer() {
        boost::lock_guard<boost::mutex> lock(mutex_);
        if (!available_buffers_.empty()) {
            auto buffer = available_buffers_.front();
            available_buffers_.pop();
            return buffer;
        }
        return std::make_shared<std::vector<char>>(1024);
    }
    
    void return_buffer(std::shared_ptr<std::vector<char>> buffer) {
        buffer->clear();
        boost::lock_guard<boost::mutex> lock(mutex_);
        available_buffers_.push(buffer);
    }
};
```

#### Connection Pooling Strategies
```cpp
class ConnectionPool {
    std::queue<tcp::socket> available_connections_;
    std::atomic<int> active_connections_{0};
    const int max_connections_;
    
public:
    // Reuse existing connections instead of creating new ones
    tcp::socket get_connection() {
        if (!available_connections_.empty()) {
            auto conn = std::move(available_connections_.front());
            available_connections_.pop();
            return conn;
        }
        
        if (active_connections_ < max_connections_) {
            ++active_connections_;
            return tcp::socket(io_context_);
        }
        
        throw std::runtime_error("Connection pool exhausted");
    }
};
```

### Fiber Performance Characteristics

#### Memory Usage Comparison
```cpp
void analyze_fiber_memory_usage() {
    const int num_tasks = 100000;
    
    // Measure thread memory usage
    std::cout << "Creating " << num_tasks << " threads...\n";
    // This would typically fail due to memory exhaustion
    // Each thread: ~8MB stack = 800GB total!
    
    // Measure fiber memory usage  
    std::cout << "Creating " << num_tasks << " fibers...\n";
    std::vector<boost::fibers::fiber> fibers;
    fibers.reserve(num_tasks);
    
    for (int i = 0; i < num_tasks; ++i) {
        fibers.emplace_back([i]() {
            // Simulate I/O wait
            boost::this_fiber::sleep_for(std::chrono::seconds(1));
        });
    }
    
    std::cout << "All fibers created successfully!\n";
    // Each fiber: ~64KB stack = 6.4GB total (manageable)
    
    for (auto& f : fibers) {
        f.join();
    }
}
```

**Performance Trade-offs:**

| Metric | OS Threads | Boost.Fibers |
|--------|------------|--------------|
| **Creation Time** | ~1-2ms | ~1-10μs |
| **Memory per Task** | 8MB | 64KB |
| **Context Switch** | 5-15μs | 50-200ns |
| **Max Concurrent** | ~1,000s | ~100,000s |
| **CPU Utilization** | Full parallelism | Single core only |

### Profiling and Optimization Tools

#### Using Boost.Asio with Profilers
```cpp
// Enable Asio handler tracking for profiling
#define BOOST_ASIO_ENABLE_HANDLER_TRACKING
#include <boost/asio.hpp>

void profile_async_operations() {
    boost::asio::io_context io_context;
    
    // Handler tracking will log:
    // - Handler creation and destruction
    // - Handler execution times
    // - Async operation completion times
    
    timer.async_wait([](boost::system::error_code ec) {
        // This handler's performance will be tracked
    });
}
```

#### Memory Profiling Tips
```bash
# Use Valgrind to detect memory issues
valgrind --tool=memcheck --leak-check=full ./your_boost_app

# Use perf to profile performance
perf record ./your_boost_app
perf report

# Monitor thread activity
htop -H  # Show threads separately
```

**Optimization Checklist:**
- ✅ Use buffer pools for frequent allocations
- ✅ Implement connection pooling for network apps
- ✅ Choose appropriate thread pool sizes
- ✅ Profile memory usage and CPU hotspots
- ✅ Consider lock-free algorithms for high contention
- ✅ Use fibers for I/O-heavy workloads with high concurrency
- ✅ Monitor and tune garbage collection (if applicable)

## Best Practices

1. **Thread Safety**
   - Use RAII for lock management
   - Minimize critical section duration
   - Avoid nested locks when possible
   - Use lock-free algorithms where appropriate

2. **Asynchronous Programming**
   - Handle all error conditions in callbacks
   - Avoid blocking operations in async contexts
   - Use shared_ptr for object lifetime management
   - Implement proper cancellation support

3. **Resource Management**
   - Limit thread pool sizes appropriately
   - Monitor and manage connection limits
   - Implement graceful shutdown procedures
   - Use memory pools for frequent allocations

## Learning Assessment and Mastery Evaluation

### Knowledge Assessment Checklist

By the end of this section, you should be able to demonstrate:

#### Boost.Thread Mastery
□ **Create and manage thread groups** for coordinated multi-threading  
□ **Implement thread interruption** for graceful task cancellation  
□ **Use advanced synchronization** (barriers, shared_mutex, etc.)  
□ **Handle thread-local storage** for per-thread data  
□ **Debug multi-threaded applications** using appropriate tools  

#### Boost.Asio Proficiency  
□ **Design asynchronous servers** that handle thousands of connections  
□ **Implement custom protocols** using Asio's networking primitives  
□ **Use timers effectively** for scheduling and timeouts  
□ **Handle errors gracefully** in asynchronous operations  
□ **Optimize memory usage** in async applications  

#### Boost.Fiber Competency
□ **Choose between threads and fibers** based on requirements  
□ **Implement fiber-based pipelines** for data processing  
□ **Use fiber channels** for inter-fiber communication  
□ **Design hybrid systems** combining threads and fibers  
□ **Understand scheduling implications** of cooperative multitasking  

### Practical Assessment Projects

#### Project 1: Multi-protocol Network Server (Intermediate)
Build a server that handles multiple protocols simultaneously:
- HTTP on port 80
- Custom binary protocol on port 8080  
- WebSocket on port 8181
- All using single-threaded asynchronous I/O
- Support 1000+ concurrent connections
- Implement graceful shutdown and statistics

**Evaluation Criteria:**
- Code organization and error handling
- Performance under load testing
- Resource usage optimization
- Protocol correctness

#### Project 2: Distributed Computing Framework (Advanced)
Create a framework for distributed task execution:
- Master-worker architecture using Boost.Asio
- Task serialization and distribution
- Worker fault tolerance and recovery
- Load balancing and task scheduling
- Web-based monitoring interface

**Evaluation Criteria:**
- System architecture design
- Fault tolerance implementation
- Scalability and performance
- Code quality and documentation

### Common Pitfalls and How to Avoid Them

#### Thread-related Issues
❌ **Deadlocks from lock ordering**
```cpp
// BAD: Inconsistent lock ordering
void thread1() {
    boost::lock_guard<boost::mutex> lock1(mutex1);
    boost::lock_guard<boost::mutex> lock2(mutex2);  // Can deadlock
}
void thread2() {
    boost::lock_guard<boost::mutex> lock2(mutex2);
    boost::lock_guard<boost::mutex> lock1(mutex1);  // Can deadlock
}
```

✅ **Solution: Always acquire locks in same order**
```cpp
// GOOD: Consistent lock ordering
void safe_function() {
    boost::lock(mutex1, mutex2);  // Acquire both atomically
    boost::lock_guard<boost::mutex> lock1(mutex1, boost::adopt_lock);
    boost::lock_guard<boost::mutex> lock2(mutex2, boost::adopt_lock);
}
```

#### Asio-related Issues
❌ **Forgetting to keep objects alive during async operations**
```cpp
void dangerous_async() {
    std::string data = "Hello";
    socket.async_write_some(boost::asio::buffer(data),  // DANGER!
        [](boost::system::error_code ec, std::size_t) {
            // 'data' may be destroyed before this callback runs
        });
}  // 'data' goes out of scope here
```

✅ **Solution: Capture data in callback or use shared_ptr**
```cpp
void safe_async() {
    auto data = std::make_shared<std::string>("Hello");
    socket.async_write_some(boost::asio::buffer(*data),
        [data](boost::system::error_code ec, std::size_t) {  // Capture shared_ptr
            // 'data' guaranteed to be alive
        });
}
```

#### Fiber-related Issues
❌ **Forgetting to yield in long-running fibers**
```cpp
void cpu_intensive_fiber() {
    for (long i = 0; i < 1000000000; ++i) {
        // Long computation without yielding
        compute_something();
    }
    // Other fibers starved during this computation
}
```

✅ **Solution: Periodically yield control**
```cpp
void cooperative_fiber() {
    for (long i = 0; i < 1000000000; ++i) {
        compute_something();
        if (i % 10000 == 0) {
            boost::this_fiber::yield();  // Give other fibers a chance
        }
    }
}
```

### Study Resources and Next Steps

#### Recommended Reading
- **"Boost.Asio C++ Network Programming" by John Torjo** - Comprehensive Asio guide
- **"C++ Concurrency in Action" by Anthony Williams** - Modern C++ concurrency
- **"The Art of Multiprocessor Programming" by Herlihy & Shavit** - Theoretical foundations

#### Online Resources
- [Boost.Asio Documentation](https://www.boost.org/doc/libs/1_82_0/doc/html/boost_asio.html)
- [Boost.Thread Documentation](https://www.boost.org/doc/libs/1_82_0/doc/html/thread.html)
- [Boost.Fiber Documentation](https://www.boost.org/doc/libs/1_82_0/libs/fiber/doc/html/index.html)

#### Practice Platforms
- **LeetCode Concurrency Problems** - Algorithm-focused threading challenges
- **Project Euler** - Mathematical problems suitable for parallel processing
- **GitHub Open Source Projects** - Contribute to real-world concurrent systems

## Final Assessment

- ✅ Can implement thread-safe data structures correctly using Boost.Thread
- ✅ Understands asynchronous I/O patterns and error handling in Boost.Asio
- ✅ Can design scalable concurrent systems using appropriate concurrency models
- ✅ Knows when to use threads vs fibers vs async I/O based on requirements
- ✅ Can profile and optimize concurrent applications for performance
- ✅ Understands common concurrency pitfalls and how to avoid them
- ✅ Can build production-ready networked applications using Boost libraries

### Certification Preparation
If pursuing formal certification, focus on:
- Implementing all practical exercises
- Building at least one complete project from each category
- Understanding performance characteristics of different approaches
- Practicing debugging concurrent applications
- Reading and understanding real-world concurrent codebases

## Next Steps

Move on to [Functional Programming](08_Functional_Programming.md) to explore Boost's functional programming utilities.
