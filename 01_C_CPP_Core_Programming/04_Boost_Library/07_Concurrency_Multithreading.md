# Concurrency and Multithreading

*Duration: 2 weeks*

## Overview

This section covers Boost's concurrency libraries, including threading primitives, asynchronous I/O, and fiber-based concurrency models.

## Learning Topics

### Boost.Thread
- Thread management and lifecycle
- Synchronization primitives (mutex, condition variables)
- Thread-local storage
- Thread groups and interruption

### Boost.Asio
- Asynchronous I/O operations
- Networking and socket programming
- Timers and time management
- Coroutines and completion handlers

### Boost.Fiber
- User-space threads (fibers)
- Fiber synchronization primitives
- Scheduling algorithms
- Integration with asynchronous operations

## Code Examples

### Basic Threading with Boost.Thread
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

### Boost.Asio - Basic Networking
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
            });
    }
    
    tcp::acceptor acceptor_;
};

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

## Practical Exercises

1. **Thread Pool Implementation**
   - Create a thread pool with work queues
   - Implement task scheduling and load balancing
   - Add thread-safe statistics and monitoring

2. **Asynchronous Web Server**
   - Build a multi-threaded web server using Boost.Asio
   - Support HTTP/1.1 with keep-alive connections
   - Implement request routing and static file serving

3. **Fiber-based Task Scheduler**
   - Create a cooperative task scheduler using fibers
   - Implement priority-based scheduling
   - Add support for async I/O integration

4. **Distributed Work Queue**
   - Build a networked work distribution system
   - Implement worker registration and heartbeats
   - Add fault tolerance and job persistence

## Performance Considerations

### Threading Performance
- Thread creation and destruction overhead
- Context switching costs
- Memory synchronization overhead
- Cache line false sharing

### Asynchronous I/O Performance
- Event loop efficiency
- Memory allocations in callbacks
- Network buffer management
- Connection pooling strategies

### Fiber Performance
- Fiber switching overhead vs thread switching
- Memory usage per fiber
- Integration with blocking operations
- Scheduler algorithm selection

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

## Assessment

- Can implement thread-safe data structures correctly
- Understands asynchronous I/O patterns and error handling
- Can design scalable concurrent systems
- Knows when to use threads vs fibers vs async I/O

## Next Steps

Move on to [Functional Programming](08_Functional_Programming.md) to explore Boost's functional programming utilities.
