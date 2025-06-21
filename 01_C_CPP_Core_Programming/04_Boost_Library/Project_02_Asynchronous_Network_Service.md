# Project 2: Asynchronous Network Service

*Estimated Duration: 3-4 weeks*
*Difficulty: Advanced*

## Project Overview

Build a high-performance, asynchronous network service using Boost.Asio that can handle multiple concurrent connections efficiently. This project demonstrates advanced networking concepts, asynchronous programming patterns, and server architecture design.

## Learning Objectives

- Master Boost.Asio for asynchronous I/O operations
- Understand event-driven programming and reactor patterns
- Implement scalable network server architectures
- Handle connection management and resource cleanup
- Design protocol handling and message processing systems

## Project Requirements

### Core Features

1. **HTTP-like Protocol Server**
   - Support basic HTTP GET/POST requests
   - Handle multiple concurrent connections
   - Implement proper HTTP response formatting
   - Support persistent connections (keep-alive)

2. **Asynchronous Operations**
   - Non-blocking accept operations
   - Asynchronous read/write operations
   - Timer-based operations for timeouts
   - Graceful shutdown handling

3. **Connection Management**
   - Connection pooling and lifecycle management
   - Request/response correlation
   - Error handling and recovery
   - Resource cleanup on disconnection

### Advanced Features

4. **Load Balancing and Scaling**
   - Worker thread pool for CPU-intensive tasks
   - Connection load balancing
   - Rate limiting and throttling
   - Health monitoring and statistics

5. **Security and Validation**
   - Input validation and sanitization
   - Basic authentication support
   - SSL/TLS support (optional)
   - Request size limits and timeout handling

6. **Monitoring and Logging**
   - Comprehensive logging system
   - Performance metrics collection
   - Connection statistics and reporting
   - Administrative interface

## Implementation Guide

### Step 1: Basic Server Architecture

```cpp
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

using boost::asio::ip::tcp;

class HttpServer;

// Forward declarations
class HttpConnection;
class HttpRequest;
class HttpResponse;
class RequestHandler;

typedef boost::shared_ptr<HttpConnection> connection_ptr;
typedef boost::shared_ptr<HttpRequest> request_ptr;
typedef boost::shared_ptr<HttpResponse> response_ptr;
```

### Step 2: HTTP Message Classes

```cpp
class HttpRequest {
public:
    enum Method { GET, POST, PUT, DELETE, UNKNOWN };
    
    HttpRequest() : method_(UNKNOWN), version_major_(1), version_minor_(1) {}
    
    Method method() const { return method_; }
    const std::string& uri() const { return uri_; }
    const std::string& body() const { return body_; }
    const std::map<std::string, std::string>& headers() const { return headers_; }
    
    void set_method(Method method) { method_ = method; }
    void set_uri(const std::string& uri) { uri_ = uri; }
    void set_body(const std::string& body) { body_ = body; }
    void add_header(const std::string& name, const std::string& value) {
        headers_[name] = value;
    }
    
    std::string get_header(const std::string& name) const {
        auto it = headers_.find(name);
        return (it != headers_.end()) ? it->second : "";
    }
    
    bool keep_alive() const {
        std::string connection = get_header("Connection");
        return (connection == "keep-alive") || 
               (version_major_ == 1 && version_minor_ == 1 && connection != "close");
    }
    
    static Method parse_method(const std::string& method_str) {
        if (method_str == "GET") return GET;
        if (method_str == "POST") return POST;
        if (method_str == "PUT") return PUT;
        if (method_str == "DELETE") return DELETE;
        return UNKNOWN;
    }
    
private:
    Method method_;
    std::string uri_;
    std::string body_;
    std::map<std::string, std::string> headers_;
    int version_major_;
    int version_minor_;
    
    friend class HttpParser;
};

class HttpResponse {
public:
    HttpResponse() : status_code_(200), version_major_(1), version_minor_(1) {}
    
    void set_status(int code, const std::string& message = "") {
        status_code_ = code;
        status_message_ = message.empty() ? default_status_message(code) : message;
    }
    
    void set_body(const std::string& body) {
        body_ = body;
        add_header("Content-Length", std::to_string(body.length()));
    }
    
    void add_header(const std::string& name, const std::string& value) {
        headers_[name] = value;
    }
    
    std::string to_string() const {
        std::ostringstream response;
        response << "HTTP/" << version_major_ << "." << version_minor_ 
                 << " " << status_code_ << " " << status_message_ << "\r\n";
        
        for (const auto& header : headers_) {
            response << header.first << ": " << header.second << "\r\n";
        }
        
        response << "\r\n" << body_;
        return response.str();
    }
    
private:
    int status_code_;
    std::string status_message_;
    std::string body_;
    std::map<std::string, std::string> headers_;
    int version_major_;
    int version_minor_;
    
    static std::string default_status_message(int code) {
        switch (code) {
            case 200: return "OK";
            case 400: return "Bad Request";
            case 404: return "Not Found";
            case 500: return "Internal Server Error";
            default: return "Unknown";
        }
    }
};
```

### Step 3: HTTP Parser

```cpp
class HttpParser {
public:
    enum ParseResult { PARSE_SUCCESS, PARSE_INCOMPLETE, PARSE_ERROR };
    
    ParseResult parse_request(const std::string& data, HttpRequest& request) {
        std::istringstream stream(data);
        std::string line;
        
        // Parse request line
        if (!std::getline(stream, line) || line.empty()) {
            return PARSE_INCOMPLETE;
        }
        
        // Remove \r if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        if (!parse_request_line(line, request)) {
            return PARSE_ERROR;
        }
        
        // Parse headers
        std::string headers_section;
        bool headers_complete = false;
        
        while (std::getline(stream, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            
            if (line.empty()) {
                headers_complete = true;
                break;
            }
            
            if (!parse_header_line(line, request)) {
                return PARSE_ERROR;
            }
        }
        
        if (!headers_complete) {
            return PARSE_INCOMPLETE;
        }
        
        // Parse body if present
        std::string content_length_str = request.get_header("Content-Length");
        if (!content_length_str.empty()) {
            size_t content_length = std::stoul(content_length_str);
            std::string body;
            char ch;
            
            while (body.length() < content_length && stream.get(ch)) {
                body += ch;
            }
            
            if (body.length() < content_length) {
                return PARSE_INCOMPLETE;
            }
            
            request.set_body(body);
        }
        
        return PARSE_SUCCESS;
    }
    
private:
    bool parse_request_line(const std::string& line, HttpRequest& request) {
        std::istringstream iss(line);
        std::string method_str, uri, version;
        
        if (!(iss >> method_str >> uri >> version)) {
            return false;
        }
        
        request.set_method(HttpRequest::parse_method(method_str));
        request.set_uri(uri);
        
        return true;
    }
    
    bool parse_header_line(const std::string& line, HttpRequest& request) {
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) {
            return false;
        }
        
        std::string name = line.substr(0, colon_pos);
        std::string value = line.substr(colon_pos + 1);
        
        // Trim whitespace
        name.erase(name.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        request.add_header(name, value);
        return true;
    }
};
```

### Step 4: Connection Management

```cpp
class HttpConnection : public boost::enable_shared_from_this<HttpConnection> {
public:
    HttpConnection(boost::asio::io_context& io_context, HttpServer& server)
        : socket_(io_context), server_(server), timer_(io_context) {}
    
    tcp::socket& socket() { return socket_; }
    
    void start() {
        // Set socket options
        socket_.set_option(tcp::no_delay(true));
        
        // Start timeout timer
        start_timeout_timer();
        
        // Begin reading request
        start_read();
    }
    
    void close() {
        boost::system::error_code ec;
        timer_.cancel(ec);
        socket_.close(ec);
    }
    
private:
    tcp::socket socket_;
    HttpServer& server_;
    boost::asio::steady_timer timer_;
    std::string read_buffer_;
    HttpParser parser_;
    static const size_t MAX_REQUEST_SIZE = 8192;
    static const int TIMEOUT_SECONDS = 30;
    
    void start_read() {
        auto self = shared_from_this();
        
        socket_.async_read_some(
            boost::asio::buffer(read_buffer_tmp_, sizeof(read_buffer_tmp_)),
            [this, self](boost::system::error_code ec, std::size_t bytes_transferred) {
                if (!ec) {
                    handle_read(bytes_transferred);
                } else {
                    handle_error(ec);
                }
            }
        );
    }
    
    void handle_read(std::size_t bytes_transferred) {
        read_buffer_.append(read_buffer_tmp_, bytes_transferred);
        
        // Check for complete request
        HttpRequest request;
        HttpParser::ParseResult result = parser_.parse_request(read_buffer_, request);
        
        switch (result) {
            case HttpParser::PARSE_SUCCESS:
                process_request(request);
                break;
                
            case HttpParser::PARSE_INCOMPLETE:
                if (read_buffer_.size() > MAX_REQUEST_SIZE) {
                    send_error_response(400, "Request too large");
                } else {
                    start_read(); // Continue reading
                }
                break;
                
            case HttpParser::PARSE_ERROR:
                send_error_response(400, "Bad request");
                break;
        }
    }
    
    void process_request(const HttpRequest& request) {
        // Cancel timeout timer during processing
        timer_.cancel();
        
        // Create response
        HttpResponse response;
        
        // Route request to appropriate handler
        handle_request(request, response);
        
        // Send response
        send_response(response, request.keep_alive());
    }
    
    void send_response(const HttpResponse& response, bool keep_alive) {
        auto response_str = boost::make_shared<std::string>(response.to_string());
        auto self = shared_from_this();
        
        boost::asio::async_write(
            socket_,
            boost::asio::buffer(*response_str),
            [this, self, response_str, keep_alive](boost::system::error_code ec, std::size_t) {
                if (!ec) {
                    if (keep_alive) {
                        // Reset for next request
                        read_buffer_.clear();
                        start_timeout_timer();
                        start_read();
                    } else {
                        close();
                    }
                } else {
                    handle_error(ec);
                }
            }
        );
    }
    
    void send_error_response(int status_code, const std::string& message) {
        HttpResponse response;
        response.set_status(status_code, message);
        response.set_body("<html><body><h1>" + message + "</h1></body></html>");
        response.add_header("Content-Type", "text/html");
        response.add_header("Connection", "close");
        
        send_response(response, false);
    }
    
    void start_timeout_timer() {
        timer_.expires_after(std::chrono::seconds(TIMEOUT_SECONDS));
        auto self = shared_from_this();
        
        timer_.async_wait([this, self](boost::system::error_code ec) {
            if (!ec) {
                // Timeout occurred
                close();
            }
        });
    }
    
    void handle_request(const HttpRequest& request, HttpResponse& response);
    void handle_error(const boost::system::error_code& ec);
    
    char read_buffer_tmp_[1024];
};
```

### Step 5: Request Handling Framework

```cpp
class RequestHandler {
public:
    virtual ~RequestHandler() = default;
    virtual void handle_request(const HttpRequest& request, HttpResponse& response) = 0;
    virtual bool can_handle(const HttpRequest& request) const = 0;
};

class StaticFileHandler : public RequestHandler {
public:
    StaticFileHandler(const std::string& document_root) 
        : document_root_(document_root) {}
    
    bool can_handle(const HttpRequest& request) const override {
        return request.method() == HttpRequest::GET;
    }
    
    void handle_request(const HttpRequest& request, HttpResponse& response) override {
        std::string file_path = document_root_ + request.uri();
        
        // Security check - prevent directory traversal
        if (file_path.find("..") != std::string::npos) {
            response.set_status(403, "Forbidden");
            return;
        }
        
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            response.set_status(404, "Not Found");
            response.set_body("<html><body><h1>File Not Found</h1></body></html>");
            response.add_header("Content-Type", "text/html");
            return;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        response.set_status(200, "OK");
        response.set_body(content);
        response.add_header("Content-Type", get_mime_type(file_path));
    }
    
private:
    std::string document_root_;
    
    std::string get_mime_type(const std::string& file_path) const {
        size_t dot_pos = file_path.find_last_of('.');
        if (dot_pos == std::string::npos) {
            return "application/octet-stream";
        }
        
        std::string extension = file_path.substr(dot_pos + 1);
        
        if (extension == "html" || extension == "htm") return "text/html";
        if (extension == "css") return "text/css";
        if (extension == "js") return "application/javascript";
        if (extension == "json") return "application/json";
        if (extension == "png") return "image/png";
        if (extension == "jpg" || extension == "jpeg") return "image/jpeg";
        if (extension == "gif") return "image/gif";
        
        return "application/octet-stream";
    }
};

class ApiHandler : public RequestHandler {
public:
    bool can_handle(const HttpRequest& request) const override {
        return request.uri().substr(0, 5) == "/api/";
    }
    
    void handle_request(const HttpRequest& request, HttpResponse& response) override {
        std::string endpoint = request.uri().substr(5); // Remove "/api/"
        
        if (endpoint == "status") {
            handle_status_request(request, response);
        } else if (endpoint == "echo") {
            handle_echo_request(request, response);
        } else {
            response.set_status(404, "API endpoint not found");
            response.set_body("{\"error\": \"Endpoint not found\"}");
            response.add_header("Content-Type", "application/json");
        }
    }
    
private:
    void handle_status_request(const HttpRequest& request, HttpResponse& response) {
        std::string json_response = R"({
            "status": "ok",
            "timestamp": ")" + get_current_timestamp() + R"(",
            "server": "Boost.Asio HTTP Server"
        })";
        
        response.set_status(200, "OK");
        response.set_body(json_response);
        response.add_header("Content-Type", "application/json");
    }
    
    void handle_echo_request(const HttpRequest& request, HttpResponse& response) {
        if (request.method() == HttpRequest::POST) {
            response.set_status(200, "OK");
            response.set_body(request.body());
            response.add_header("Content-Type", "application/json");
        } else {
            response.set_status(405, "Method Not Allowed");
            response.add_header("Allow", "POST");
        }
    }
    
    std::string get_current_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    }
};
```

### Step 6: Main Server Class

```cpp
class HttpServer {
public:
    HttpServer(boost::asio::io_context& io_context, short port)
        : io_context_(io_context), 
          acceptor_(io_context, tcp::endpoint(tcp::v4(), port)),
          connection_count_(0) {
        
        // Add default handlers
        add_handler(std::make_shared<ApiHandler>());
        add_handler(std::make_shared<StaticFileHandler>("./public"));
        
        start_accept();
    }
    
    void add_handler(std::shared_ptr<RequestHandler> handler) {
        std::lock_guard<std::mutex> lock(handlers_mutex_);
        handlers_.push_back(handler);
    }
    
    void handle_request(const HttpRequest& request, HttpResponse& response) {
        std::lock_guard<std::mutex> lock(handlers_mutex_);
        
        for (auto& handler : handlers_) {
            if (handler->can_handle(request)) {
                handler->handle_request(request, response);
                return;
            }
        }
        
        // No handler found
        response.set_status(404, "Not Found");
        response.set_body("<html><body><h1>Not Found</h1></body></html>");
        response.add_header("Content-Type", "text/html");
    }
    
    void connection_started() {
        ++connection_count_;
        std::cout << "Connection started. Active connections: " 
                  << connection_count_ << std::endl;
    }
    
    void connection_ended() {
        --connection_count_;
        std::cout << "Connection ended. Active connections: " 
                  << connection_count_ << std::endl;
    }
    
private:
    boost::asio::io_context& io_context_;
    tcp::acceptor acceptor_;
    std::vector<std::shared_ptr<RequestHandler>> handlers_;
    std::mutex handlers_mutex_;
    std::atomic<int> connection_count_;
    
    void start_accept() {
        connection_ptr new_connection = 
            boost::make_shared<HttpConnection>(io_context_, *this);
        
        acceptor_.async_accept(
            new_connection->socket(),
            boost::bind(&HttpServer::handle_accept, this, new_connection,
                       boost::asio::placeholders::error)
        );
    }
    
    void handle_accept(connection_ptr new_connection,
                      const boost::system::error_code& error) {
        if (!error) {
            connection_started();
            new_connection->start();
        }
        
        start_accept();
    }
};

// Implementation of HttpConnection::handle_request
void HttpConnection::handle_request(const HttpRequest& request, HttpResponse& response) {
    server_.handle_request(request, response);
}

void HttpConnection::handle_error(const boost::system::error_code& ec) {
    if (ec != boost::asio::error::operation_aborted) {
        std::cout << "Connection error: " << ec.message() << std::endl;
    }
    server_.connection_ended();
    close();
}
```

### Step 7: Server Application

```cpp
class ServerApplication {
public:
    ServerApplication(int port, int thread_count = std::thread::hardware_concurrency())
        : port_(port), thread_count_(thread_count), server_(io_context_, port) {}
    
    void run() {
        std::cout << "Starting HTTP server on port " << port_ 
                  << " with " << thread_count_ << " threads..." << std::endl;
        
        // Create thread pool
        std::vector<std::thread> threads;
        for (int i = 0; i < thread_count_; ++i) {
            threads.emplace_back([this]() {
                io_context_.run();
            });
        }
        
        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    void stop() {
        io_context_.stop();
    }
    
private:
    int port_;
    int thread_count_;
    boost::asio::io_context io_context_;
    HttpServer server_;
};

int main(int argc, char* argv[]) {
    try {
        int port = (argc > 1) ? std::atoi(argv[1]) : 8080;
        int threads = (argc > 2) ? std::atoi(argv[2]) : std::thread::hardware_concurrency();
        
        ServerApplication app(port, threads);
        
        // Set up signal handling for graceful shutdown
        boost::asio::io_context signal_io;
        boost::asio::signal_set signals(signal_io, SIGINT, SIGTERM);
        
        signals.async_wait([&app](boost::system::error_code ec, int signal) {
            if (!ec) {
                std::cout << "\nReceived signal " << signal << ". Shutting down..." << std::endl;
                app.stop();
            }
        });
        
        std::thread signal_thread([&signal_io]() {
            signal_io.run();
        });
        
        app.run();
        
        signal_io.stop();
        signal_thread.join();
        
    } catch (std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Advanced Features Implementation

### Performance Monitoring

```cpp
class ServerMetrics {
public:
    void record_request(const std::string& method, const std::string& uri, 
                       int status_code, std::chrono::milliseconds duration) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        total_requests_++;
        request_durations_.push_back(duration.count());
        
        status_codes_[status_code]++;
        methods_[method]++;
        
        if (request_durations_.size() > 1000) {
            request_durations_.erase(request_durations_.begin());
        }
    }
    
    std::string get_metrics_json() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        double avg_duration = 0.0;
        if (!request_durations_.empty()) {
            auto sum = std::accumulate(request_durations_.begin(), 
                                     request_durations_.end(), 0.0);
            avg_duration = sum / request_durations_.size();
        }
        
        std::ostringstream json;
        json << "{\n";
        json << "  \"total_requests\": " << total_requests_ << ",\n";
        json << "  \"average_response_time_ms\": " << avg_duration << ",\n";
        json << "  \"status_codes\": {\n";
        
        bool first = true;
        for (const auto& pair : status_codes_) {
            if (!first) json << ",\n";
            json << "    \"" << pair.first << "\": " << pair.second;
            first = false;
        }
        
        json << "\n  }\n}";
        return json.str();
    }
    
private:
    mutable std::mutex mutex_;
    std::atomic<long> total_requests_{0};
    std::vector<double> request_durations_;
    std::map<int, long> status_codes_;
    std::map<std::string, long> methods_;
};
```

### Rate Limiting

```cpp
class RateLimiter {
public:
    RateLimiter(int max_requests_per_minute = 60) 
        : max_requests_(max_requests_per_minute) {}
    
    bool allow_request(const std::string& client_ip) {
        auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto& client_data = clients_[client_ip];
        
        // Remove old timestamps
        client_data.erase(
            std::remove_if(client_data.begin(), client_data.end(),
                [now](const auto& timestamp) {
                    return now - timestamp > std::chrono::minutes(1);
                }),
            client_data.end()
        );
        
        if (client_data.size() >= max_requests_) {
            return false;
        }
        
        client_data.push_back(now);
        return true;
    }
    
private:
    int max_requests_;
    std::mutex mutex_;
    std::map<std::string, std::vector<std::chrono::steady_clock::time_point>> clients_;
};
```

## Testing and Benchmarking

### Load Testing Script

```python
#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import statistics

async def make_request(session, url):
    start_time = time.time()
    try:
        async with session.get(url) as response:
            await response.text()
            return time.time() - start_time, response.status
    except Exception as e:
        return time.time() - start_time, 0

async def load_test(url, concurrent_requests, total_requests):
    connector = aiohttp.TCPConnector(limit=concurrent_requests)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        
        for _ in range(total_requests):
            task = asyncio.create_task(make_request(session, url))
            tasks.append(task)
            
            if len(tasks) >= concurrent_requests:
                results = await asyncio.gather(*tasks)
                tasks = []
                yield results

if __name__ == "__main__":
    import sys
    
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/api/status"
    concurrent = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    print(f"Load testing {url}")
    print(f"Concurrent requests: {concurrent}")
    print(f"Total requests: {total}")
    
    asyncio.run(load_test(url, concurrent, total))
```

## Performance Considerations

1. **Connection Pooling**
   - Reuse connections for keep-alive requests
   - Implement connection limits per client
   - Monitor connection lifecycle

2. **Memory Management**
   - Use object pooling for frequent allocations
   - Implement buffer recycling
   - Monitor memory usage patterns

3. **Threading Strategy**
   - Use thread pool for CPU-intensive operations
   - Keep I/O operations on the main event loop
   - Implement work stealing for load balancing

## Security Considerations

1. **Input Validation**
   - Validate all HTTP headers and body content
   - Implement request size limits
   - Prevent buffer overflow attacks

2. **Rate Limiting**
   - Implement per-IP rate limiting
   - Add exponential backoff for repeated violations
   - Monitor for DDoS patterns

3. **Error Handling**
   - Never expose internal error details
   - Log security events for monitoring
   - Implement graceful degradation

## Assessment Criteria

- [ ] Implements complete HTTP protocol handling
- [ ] Demonstrates proper asynchronous programming patterns
- [ ] Handles multiple concurrent connections efficiently
- [ ] Includes comprehensive error handling and recovery
- [ ] Implements performance monitoring and metrics
- [ ] Provides security features and validation
- [ ] Achieves target performance benchmarks
- [ ] Includes thorough testing and documentation

## Deliverables

1. Complete server implementation with full HTTP support
2. Performance testing suite and benchmarks
3. Security analysis and hardening recommendations
4. Deployment guide and operational documentation
5. Load testing results and optimization report
