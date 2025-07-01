# Custom Tracing and Integration

*Duration: 2-3 weeks*

## Overview

Custom tracing with Perfetto allows you to instrument your applications with precise performance monitoring capabilities. This advanced topic covers integrating the Perfetto SDK into C++, Python, and JavaScript applications, creating custom trace events, and building comprehensive performance monitoring systems.

## Learning Objectives

By the end of this section, you should be able to:
- **Integrate Perfetto SDK** into various application types
- **Create custom trace events** with different categories and metadata
- **Implement performance counters** and custom metrics
- **Build trace collection systems** for production applications
- **Optimize trace overhead** and minimize performance impact
- **Analyze custom traces** using Perfetto UI and programmatic tools

## Table of Contents

1. [Perfetto SDK Integration](#perfetto-sdk-integration)
2. [Custom Trace Events](#custom-trace-events)
3. [Advanced Tracing Patterns](#advanced-tracing-patterns)
4. [Performance Monitoring Integration](#performance-monitoring-integration)
5. [Production Deployment](#production-deployment)
6. [Cross-Platform Considerations](#cross-platform-considerations)
7. [Best Practices and Optimization](#best-practices-and-optimization)

---

## Perfetto SDK Integration

### Setting Up the Development Environment

#### 1. Installing Perfetto SDK

**For C++ Projects:**
```bash
# Clone Perfetto repository
git clone https://android.googlesource.com/platform/external/perfetto/
cd perfetto

# Build the SDK
tools/install-build-deps
tools/gn gen --args='is_debug=false' out/release
tools/ninja -C out/release perfetto_unittests

# For easier integration, use vcpkg
vcpkg install perfetto
```

**For Python Projects:**
```bash
pip install perfetto
```

**For Node.js Projects:**
```bash
npm install perfetto
```

#### 2. CMake Integration Example

```cmake
# CMakeLists.txt for C++ project with Perfetto
cmake_minimum_required(VERSION 3.15)
project(PerfettoExample)

set(CMAKE_CXX_STANDARD 17)

# Find Perfetto package
find_package(PkgConfig REQUIRED)
pkg_check_modules(PERFETTO REQUIRED perfetto)

# Add executable
add_executable(perfetto_example main.cpp)

# Link Perfetto libraries
target_link_libraries(perfetto_example
    ${PERFETTO_LIBRARIES}
    pthread
)

target_include_directories(perfetto_example PRIVATE
    ${PERFETTO_INCLUDE_DIRS}
)

target_compile_options(perfetto_example PRIVATE
    ${PERFETTO_CFLAGS_OTHER}
)
```

### Basic SDK Integration

#### C++ Integration

```cpp
#include <perfetto.h>
#include <iostream>
#include <thread>
#include <chrono>

// Define trace categories
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("app")
        .SetDescription("Application-level events"),
    perfetto::Category("network")
        .SetDescription("Network operations"),
    perfetto::Category("database")
        .SetDescription("Database operations"),
    perfetto::Category("ui")
        .SetDescription("User interface events")
);

// Global trace event storage
PERFETTO_TRACK_EVENT_STATIC_STORAGE();

class PerformanceTracer {
private:
    bool initialized_ = false;

public:
    bool Initialize() {
        if (initialized_) return true;

        // Initialize Perfetto tracing
        perfetto::TrackEvent::Register();
        
        // Configure tracing session
        perfetto::TraceConfig cfg;
        cfg.add_buffers()->set_size_kb(1024);
        
        auto* ds_cfg = cfg.add_data_sources()->mutable_config();
        ds_cfg->set_name("track_event");
        
        // Start tracing session
        auto tracing_session = perfetto::Tracing::NewTrace();
        tracing_session->Setup(cfg);
        tracing_session->StartBlocking();
        
        initialized_ = true;
        std::cout << "Perfetto tracing initialized successfully\n";
        return true;
    }

    void Shutdown() {
        if (!initialized_) return;
        
        std::cout << "Shutting down Perfetto tracing\n";
        initialized_ = false;
    }

    ~PerformanceTracer() {
        Shutdown();
    }
};

// Global tracer instance
PerformanceTracer g_tracer;

int main() {
    // Initialize tracing
    if (!g_tracer.Initialize()) {
        std::cerr << "Failed to initialize tracing\n";
        return 1;
    }

    // Basic trace event
    TRACE_EVENT("app", "Application Started");
    
    std::cout << "Application running with Perfetto tracing...\n";
    
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    return 0;
}
```

#### Python Integration

```python
import perfetto
import time
import threading
from contextlib import contextmanager

class PythonTracer:
    def __init__(self):
        self.initialized = False
        self.session = None
    
    def initialize(self):
        """Initialize Perfetto tracing for Python application"""
        if self.initialized:
            return True
        
        try:
            # Configure trace config
            config = perfetto.TraceConfig()
            
            # Add buffer configuration
            buffer_config = config.buffers.add()
            buffer_config.size_kb = 1024
            buffer_config.fill_policy = perfetto.TraceConfig.BufferConfig.DISCARD
            
            # Add data source
            ds_config = config.data_sources.add()
            ds_config.config.name = "track_event"
            
            # Create and start tracing session
            self.session = perfetto.Tracing.new_trace()
            self.session.setup(config)
            self.session.start()
            
            self.initialized = True
            print("Python Perfetto tracing initialized")
            return True
            
        except Exception as e:
            print(f"Failed to initialize tracing: {e}")
            return False
    
    def shutdown(self):
        """Shutdown tracing and save trace"""
        if not self.initialized or not self.session:
            return
        
        # Stop tracing
        self.session.stop()
        
        # Read trace data
        trace_data = self.session.read_trace()
        
        # Save to file
        with open('python_trace.perfetto-trace', 'wb') as f:
            f.write(trace_data)
        
        print("Trace saved to python_trace.perfetto-trace")
        self.initialized = False
    
    @contextmanager
    def trace_function(self, category, name, **kwargs):
        """Context manager for tracing function execution"""
        start_time = time.time()
        
        # Create trace event
        event_data = {
            'category': category,
            'name': name,
            'timestamp': int(start_time * 1_000_000),  # microseconds
            'thread_id': threading.get_ident(),
            'process_id': os.getpid(),
            **kwargs
        }
        
        try:
            yield event_data
        finally:
            end_time = time.time()
            duration = int((end_time - start_time) * 1_000_000)  # microseconds
            
            # Log completion
            print(f"Traced {category}::{name} - Duration: {duration}μs")

# Example usage
tracer = PythonTracer()

def example_application():
    tracer.initialize()
    
    try:
        # Trace a function
        with tracer.trace_function("app", "data_processing", user_id=123):
            time.sleep(0.1)  # Simulate work
            
            with tracer.trace_function("database", "query_users"):
                time.sleep(0.05)  # Simulate DB query
        
        with tracer.trace_function("network", "api_call", endpoint="/users"):
            time.sleep(0.2)  # Simulate network request
            
    finally:
        tracer.shutdown()

if __name__ == "__main__":
    example_application()
```

#### JavaScript/Node.js Integration

```javascript
const perfetto = require('perfetto');
const fs = require('fs').promises;

class NodeTracer {
    constructor() {
        this.initialized = false;
        this.session = null;
        this.events = [];
    }
    
    async initialize() {
        if (this.initialized) return true;
        
        try {
            // Configure tracing
            const config = {
                buffers: [{
                    size_kb: 1024,
                    fill_policy: 'DISCARD'
                }],
                data_sources: [{
                    config: {
                        name: 'track_event'
                    }
                }]
            };
            
            // Initialize tracing session
            this.session = await perfetto.newTrace();
            await this.session.setup(config);
            await this.session.start();
            
            this.initialized = true;
            console.log('Node.js Perfetto tracing initialized');
            return true;
            
        } catch (error) {
            console.error('Failed to initialize tracing:', error);
            return false;
        }
    }
    
    traceEvent(category, name, args = {}) {
        if (!this.initialized) return;
        
        const event = {
            cat: category,
            name: name,
            ph: 'B', // Begin phase
            ts: process.hrtime.bigint() / 1000n, // Convert to microseconds
            pid: process.pid,
            tid: 0,
            args: args
        };
        
        this.events.push(event);
        
        return () => {
            // End event
            const endEvent = {
                ...event,
                ph: 'E', // End phase
                ts: process.hrtime.bigint() / 1000n
            };
            this.events.push(endEvent);
        };
    }
    
    async traceAsyncFunction(category, name, asyncFn, args = {}) {
        const endTrace = this.traceEvent(category, name, args);
        
        try {
            const result = await asyncFn();
            return result;
        } finally {
            endTrace();
        }
    }
    
    async shutdown() {
        if (!this.initialized || !this.session) return;
        
        try {
            await this.session.stop();
            const traceData = await this.session.readTrace();
            
            // Save trace file
            await fs.writeFile('node_trace.perfetto-trace', traceData);
            console.log('Trace saved to node_trace.perfetto-trace');
            
        } catch (error) {
            console.error('Error saving trace:', error);
        } finally {
            this.initialized = false;
        }
    }
}

// Example usage
async function exampleApplication() {
    const tracer = new NodeTracer();
    
    await tracer.initialize();
    
    try {
        // Trace synchronous operation
        const endSync = tracer.traceEvent('app', 'sync_operation', {type: 'calculation'});
        
        // Simulate synchronous work
        let sum = 0;
        for (let i = 0; i < 1000000; i++) {
            sum += i;
        }
        
        endSync();
        
        // Trace asynchronous operation
        await tracer.traceAsyncFunction('network', 'fetch_data', async () => {
            return new Promise(resolve => {
                setTimeout(() => resolve('data'), 100);
            });
        }, {url: 'https://api.example.com/data'});
        
        // Trace file I/O
        await tracer.traceAsyncFunction('fs', 'read_config', async () => {
            return fs.readFile('package.json', 'utf8');
        });
        
    } finally {
        await tracer.shutdown();
    }
}

// Run example
if (require.main === module) {
    exampleApplication().catch(console.error);
}

module.exports = NodeTracer;
```

## Custom Trace Events

### Understanding Trace Event Types

Perfetto supports various types of trace events, each serving different performance analysis purposes:

#### 1. Instant Events
Events that occur at a specific point in time without duration.

```cpp
#include <perfetto.h>

void LogInstantEvents() {
    // Simple instant event
    TRACE_EVENT_INSTANT("app", "UserLogin");
    
    // Instant event with metadata
    TRACE_EVENT_INSTANT("app", "UserAction", 
        "action", "button_click",
        "button_id", "submit_form",
        "user_id", 12345);
    
    // Instant event with custom scope
    TRACE_EVENT_INSTANT("ui", "FrameDropped", 
        TRACE_EVENT_SCOPE_THREAD,
        "frame_number", 1024,
        "expected_fps", 60);
}
```

#### 2. Duration Events (Scoped Events)
Events that have a start and end time, measuring duration of operations.

```cpp
#include <perfetto.h>
#include <chrono>
#include <thread>

class DatabaseManager {
public:
    void ExecuteQuery(const std::string& query) {
        // Automatic duration tracking
        TRACE_EVENT("database", "ExecuteQuery", 
            "query", query,
            "table", "users");
        
        // Simulate database work
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Event automatically ends when scope exits
    }
    
    void ProcessTransaction() {
        TRACE_EVENT("database", "ProcessTransaction");
        
        {
            TRACE_EVENT("database", "BeginTransaction");
            // Transaction setup
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        {
            TRACE_EVENT("database", "ExecuteStatements");
            // Execute multiple statements
            for (int i = 0; i < 3; i++) {
                TRACE_EVENT("database", "Statement", "statement_id", i);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        {
            TRACE_EVENT("database", "CommitTransaction");
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
    }
};
```

#### 3. Async Events
Events that span across multiple threads or have non-continuous execution.

```cpp
#include <perfetto.h>
#include <future>
#include <random>

class AsyncOperationManager {
private:
    std::random_device rd_;
    std::mt19937 gen_{rd_()};
    
public:
    std::future<std::string> StartAsyncDownload(const std::string& url) {
        // Start async event
        auto async_id = GenerateAsyncId();
        TRACE_EVENT_NESTABLE_ASYNC_BEGIN("network", "AsyncDownload", 
            async_id, "url", url);
        
        return std::async(std::launch::async, [this, url, async_id]() {
            return PerformDownload(url, async_id);
        });
    }
    
private:
    uint64_t GenerateAsyncId() {
        std::uniform_int_distribution<uint64_t> dis;
        return dis(gen_);
    }
    
    std::string PerformDownload(const std::string& url, uint64_t async_id) {
        // Intermediate async events
        TRACE_EVENT_NESTABLE_ASYNC_INSTANT("network", "ConnectingToServer", 
            async_id, "server", ExtractServerFromUrl(url));
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        TRACE_EVENT_NESTABLE_ASYNC_INSTANT("network", "DownloadProgress", 
            async_id, "bytes_downloaded", 1024, "total_bytes", 4096);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        
        TRACE_EVENT_NESTABLE_ASYNC_INSTANT("network", "DownloadProgress", 
            async_id, "bytes_downloaded", 4096, "total_bytes", 4096);
        
        // End async event
        TRACE_EVENT_NESTABLE_ASYNC_END("network", "AsyncDownload", 
            async_id, "status", "success", "final_size", 4096);
        
        return "Downloaded content from " + url;
    }
    
    std::string ExtractServerFromUrl(const std::string& url) {
        // Simple URL parsing for demo
        size_t start = url.find("://") + 3;
        size_t end = url.find("/", start);
        return url.substr(start, end - start);
    }
};
```

#### 4. Counter Events
Events that track numeric values over time (memory usage, FPS, etc.).

```cpp
#include <perfetto.h>
#include <atomic>
#include <thread>
#include <vector>

class PerformanceMonitor {
private:
    std::atomic<size_t> memory_usage_{0};
    std::atomic<int> active_connections_{0};
    std::atomic<double> cpu_usage_{0.0};
    bool monitoring_ = false;
    std::thread monitor_thread_;
    
public:
    void StartMonitoring() {
        monitoring_ = true;
        monitor_thread_ = std::thread([this]() {
            MonitoringLoop();
        });
    }
    
    void StopMonitoring() {
        monitoring_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
    
    void AllocateMemory(size_t bytes) {
        memory_usage_ += bytes;
        
        // Track memory allocation
        TRACE_COUNTER("memory", "HeapUsage", memory_usage_.load());
        TRACE_EVENT_INSTANT("memory", "Allocation", 
            "bytes", bytes, "total_usage", memory_usage_.load());
    }
    
    void FreeMemory(size_t bytes) {
        memory_usage_ -= bytes;
        TRACE_COUNTER("memory", "HeapUsage", memory_usage_.load());
        TRACE_EVENT_INSTANT("memory", "Deallocation", 
            "bytes", bytes, "total_usage", memory_usage_.load());
    }
    
    void AddConnection() {
        int new_count = ++active_connections_;
        TRACE_COUNTER("network", "ActiveConnections", new_count);
    }
    
    void RemoveConnection() {
        int new_count = --active_connections_;
        TRACE_COUNTER("network", "ActiveConnections", new_count);
    }
    
private:
    void MonitoringLoop() {
        while (monitoring_) {
            // Simulate CPU usage calculation
            cpu_usage_ = CalculateCpuUsage();
            TRACE_COUNTER("system", "CpuUsage", cpu_usage_.load());
            
            // Update counters every 100ms
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    double CalculateCpuUsage() {
        // Simplified CPU usage calculation
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 100.0);
        return dis(gen);
    }
};
```

### Custom Metadata and Arguments

#### Adding Rich Metadata to Events

```cpp
#include <perfetto.h>
#include <string>
#include <map>
#include <vector>

class RichTracingExample {
public:
    void ProcessUserRequest(int user_id, const std::string& request_type) {
        TRACE_EVENT("api", "ProcessUserRequest",
            // Basic metadata
            "user_id", user_id,
            "request_type", request_type,
            
            // Timing information
            "timestamp", GetCurrentTimestamp(),
            
            // Context information
            "session_id", GetSessionId(),
            "client_version", "1.2.3",
            
            // Performance hints
            "expected_duration_ms", 200,
            "priority", "high"
        );
        
        // Simulate request processing
        ProcessRequest(user_id, request_type);
    }
    
    void ProcessBatchOperation(const std::vector<int>& item_ids) {
        TRACE_EVENT("batch", "ProcessBatch",
            "batch_size", item_ids.size(),
            "memory_estimate_mb", EstimateMemoryUsage(item_ids),
            "parallel_workers", GetOptimalWorkerCount(item_ids.size())
        );
        
        for (size_t i = 0; i < item_ids.size(); ++i) {
            TRACE_EVENT("batch", "ProcessItem",
                "item_id", item_ids[i],
                "batch_index", i,
                "progress_percent", (i * 100) / item_ids.size()
            );
            
            ProcessSingleItem(item_ids[i]);
        }
    }
    
    void HandleError(const std::string& error_msg, int error_code) {
        TRACE_EVENT("error", "HandleError",
            "error_message", error_msg,
            "error_code", error_code,
            "stack_trace", GetStackTrace(),
            "recovery_attempted", true
        );
    }
    
private:
    uint64_t GetCurrentTimestamp() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }
    
    std::string GetSessionId() { return "session_12345"; }
    
    double EstimateMemoryUsage(const std::vector<int>& items) {
        return items.size() * 0.1; // 0.1 MB per item estimate
    }
    
    int GetOptimalWorkerCount(size_t batch_size) {
        return std::min(static_cast<int>(batch_size), 8);
    }
    
    std::string GetStackTrace() { return "main()->ProcessUserRequest()->HandleError()"; }
    
    void ProcessRequest(int user_id, const std::string& request_type) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    void ProcessSingleItem(int item_id) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
};
```

### Custom Track Events

#### Creating Custom Tracks for Better Organization

```cpp
#include <perfetto.h>

// Define custom tracks
class CustomTracks {
public:
    static perfetto::Track GetUITrack() {
        static perfetto::Track ui_track = perfetto::Track::CreateThread();
        return ui_track;
    }
    
    static perfetto::Track GetNetworkTrack() {
        static perfetto::Track network_track = perfetto::Track::CreateProcess();
        return network_track;
    }
    
    static perfetto::Track GetDatabaseTrack() {
        static perfetto::Track db_track = perfetto::Track::CreateGlobal();
        return db_track;
    }
};

class MultiTrackApplication {
public:
    void HandleUIEvent(const std::string& event_type) {
        // Trace on UI track
        TRACE_EVENT("ui", "HandleUIEvent", CustomTracks::GetUITrack(),
            "event_type", event_type,
            "thread_id", std::this_thread::get_id()
        );
        
        // Simulate UI processing
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }
    
    void MakeNetworkRequest(const std::string& endpoint) {
        // Trace on network track
        TRACE_EVENT("network", "HTTPRequest", CustomTracks::GetNetworkTrack(),
            "endpoint", endpoint,
            "method", "GET"
        );
        
        // Simulate network latency
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    
    void QueryDatabase(const std::string& query) {
        // Trace on database track
        TRACE_EVENT("database", "SQLQuery", CustomTracks::GetDatabaseTrack(),
            "query", query,
            "estimated_rows", 1000
        );
        
        // Simulate database query
        std::this_thread::sleep_for(std::chrono::milliseconds(75));
    }
    
    void ProcessComplexOperation() {
        TRACE_EVENT("app", "ComplexOperation");
        
        // These will appear on separate tracks
        HandleUIEvent("button_click");
        MakeNetworkRequest("/api/users");
        QueryDatabase("SELECT * FROM users WHERE active = 1");
    }
};
```

### Flow Events (Connecting Related Events)

```cpp
#include <perfetto.h>

class FlowEventExample {
private:
    uint64_t GenerateFlowId() {
        static std::atomic<uint64_t> flow_counter{1};
        return flow_counter++;
    }
    
public:
    void StartRequestFlow(const std::string& request_id) {
        auto flow_id = GenerateFlowId();
        
        // Start of flow
        TRACE_EVENT_BEGIN("api", "RequestFlow",
            "request_id", request_id,
            "flow_id", flow_id
        );
        
        ProcessInStages(request_id, flow_id);
        
        // End of flow
        TRACE_EVENT_END("api", "RequestFlow");
    }
    
private:
    void ProcessInStages(const std::string& request_id, uint64_t flow_id) {
        // Stage 1: Validation
        {
            TRACE_EVENT("validation", "ValidateRequest",
                "request_id", request_id
            );
            // Connect with flow
            TRACE_EVENT_FLOW_STEP("api", "RequestFlow", flow_id, "validation");
            
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        
        // Stage 2: Processing
        {
            TRACE_EVENT("processing", "ProcessRequest",
                "request_id", request_id
            );
            TRACE_EVENT_FLOW_STEP("api", "RequestFlow", flow_id, "processing");
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Stage 3: Response
        {
            TRACE_EVENT("response", "GenerateResponse",
                "request_id", request_id
            );
            TRACE_EVENT_FLOW_END("api", "RequestFlow", flow_id, "response");
            
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }
};
```

## Advanced Tracing Patterns

### Performance Profiling Integration

#### CPU Sampling Integration

```cpp
#include <perfetto.h>
#include <chrono>
#include <vector>
#include <algorithm>

class CPUProfiler {
private:
    struct SampleData {
        std::chrono::high_resolution_clock::time_point timestamp;
        std::string function_name;
        double cpu_usage;
        size_t memory_usage;
    };
    
    std::vector<SampleData> samples_;
    bool profiling_active_ = false;
    
public:
    void StartCPUProfiling() {
        TRACE_EVENT("profiler", "StartCPUProfiling");
        
        profiling_active_ = true;
        samples_.clear();
        
        // Configure CPU sampling
        perfetto::TraceConfig config;
        auto* cpu_config = config.add_data_sources()->mutable_config();
        cpu_config->set_name("linux.perf");
        
        // Set sampling frequency
        auto* perf_config = cpu_config->mutable_perf_event_config();
        perf_config->set_frequency(1000); // 1000 Hz sampling
        
        TRACE_COUNTER("profiler", "SamplingFrequency", 1000);
    }
    
    void SampleFunction(const std::string& function_name) {
        if (!profiling_active_) return;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        TRACE_EVENT("profiler", "CPUSample",
            "function", function_name,
            "sample_count", samples_.size()
        );
        
        // Collect sample data
        SampleData sample;
        sample.timestamp = start_time;
        sample.function_name = function_name;
        sample.cpu_usage = GetCurrentCPUUsage();
        sample.memory_usage = GetCurrentMemoryUsage();
        
        samples_.push_back(sample);
        
        // Update counters
        TRACE_COUNTER("profiler", "CPUUsage", sample.cpu_usage);
        TRACE_COUNTER("profiler", "MemoryUsage", sample.memory_usage);
    }
    
    void StopCPUProfiling() {
        TRACE_EVENT("profiler", "StopCPUProfiling",
            "total_samples", samples_.size()
        );
        
        profiling_active_ = false;
        
        // Analyze samples
        AnalyzeSamples();
    }
    
private:
    double GetCurrentCPUUsage() {
        // Platform-specific CPU usage calculation
        // This is a simplified version
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 100.0);
        return dis(gen);
    }
    
    size_t GetCurrentMemoryUsage() {
        // Platform-specific memory usage calculation
        // This is a simplified version
        static size_t base_memory = 1024 * 1024; // 1MB base
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, 1024 * 1024); // +/- 1MB
        return base_memory + dis(gen);
    }
    
    void AnalyzeSamples() {
        if (samples_.empty()) return;
        
        TRACE_EVENT("profiler", "AnalyzeSamples");
        
        // Calculate statistics
        double avg_cpu = 0.0;
        size_t avg_memory = 0;
        double max_cpu = 0.0;
        size_t max_memory = 0;
        
        for (const auto& sample : samples_) {
            avg_cpu += sample.cpu_usage;
            avg_memory += sample.memory_usage;
            max_cpu = std::max(max_cpu, sample.cpu_usage);
            max_memory = std::max(max_memory, sample.memory_usage);
        }
        
        avg_cpu /= samples_.size();
        avg_memory /= samples_.size();
        
        // Log analysis results
        TRACE_EVENT("profiler", "ProfilingResults",
            "average_cpu_usage", avg_cpu,
            "max_cpu_usage", max_cpu,
            "average_memory_usage", avg_memory,
            "max_memory_usage", max_memory,
            "sample_count", samples_.size()
        );
    }
};
```

#### Memory Allocation Tracking

```cpp
#include <perfetto.h>
#include <unordered_map>
#include <mutex>

class MemoryTracker {
private:
    struct AllocationInfo {
        size_t size;
        std::string location;
        std::chrono::high_resolution_clock::time_point timestamp;
    };
    
    std::unordered_map<void*, AllocationInfo> allocations_;
    mutable std::mutex allocations_mutex_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_usage_{0};
    
public:
    void* TrackedAlloc(size_t size, const std::string& location = "") {
        void* ptr = malloc(size);
        if (!ptr) return nullptr;
        
        // Record allocation
        {
            std::lock_guard<std::mutex> lock(allocations_mutex_);
            allocations_[ptr] = {
                size, 
                location.empty() ? GetCurrentLocation() : location,
                std::chrono::high_resolution_clock::now()
            };
        }
        
        // Update totals
        size_t new_total = total_allocated_.fetch_add(size) + size;
        size_t current_peak = peak_usage_.load();
        while (new_total > current_peak && 
               !peak_usage_.compare_exchange_weak(current_peak, new_total)) {
            // CAS loop to update peak
        }
        
        // Trace allocation
        TRACE_EVENT("memory", "Allocation",
            "size", size,
            "address", reinterpret_cast<uintptr_t>(ptr),
            "location", location,
            "total_allocated", new_total
        );
        
        TRACE_COUNTER("memory", "TotalAllocated", new_total);
        TRACE_COUNTER("memory", "PeakUsage", peak_usage_.load());
        
        return ptr;
    }
    
    void TrackedFree(void* ptr) {
        if (!ptr) return;
        
        size_t freed_size = 0;
        std::string location;
        
        // Find and remove allocation record
        {
            std::lock_guard<std::mutex> lock(allocations_mutex_);
            auto it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                freed_size = it->second.size;
                location = it->second.location;
                allocations_.erase(it);
            }
        }
        
        if (freed_size > 0) {
            size_t new_total = total_allocated_.fetch_sub(freed_size) - freed_size;
            
            TRACE_EVENT("memory", "Deallocation",
                "size", freed_size,
                "address", reinterpret_cast<uintptr_t>(ptr),
                "location", location,
                "total_allocated", new_total
            );
            
            TRACE_COUNTER("memory", "TotalAllocated", new_total);
        }
        
        free(ptr);
    }
    
    void GenerateMemoryReport() {
        TRACE_EVENT("memory", "GenerateMemoryReport");
        
        std::lock_guard<std::mutex> lock(allocations_mutex_);
        
        // Group allocations by location
        std::unordered_map<std::string, size_t> location_sizes;
        size_t total_size = 0;
        
        for (const auto& [ptr, info] : allocations_) {
            location_sizes[info.location] += info.size;
            total_size += info.size;
        }
        
        // Log top allocators
        std::vector<std::pair<std::string, size_t>> sorted_locations(
            location_sizes.begin(), location_sizes.end()
        );
        
        std::sort(sorted_locations.begin(), sorted_locations.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (size_t i = 0; i < std::min(size_t(10), sorted_locations.size()); ++i) {
            TRACE_EVENT("memory", "TopAllocator",
                "rank", i + 1,
                "location", sorted_locations[i].first,
                "total_size", sorted_locations[i].second,
                "percentage", (sorted_locations[i].second * 100.0) / total_size
            );
        }
        
        TRACE_EVENT("memory", "MemoryReportSummary",
            "total_allocations", allocations_.size(),
            "total_size", total_size,
            "peak_usage", peak_usage_.load()
        );
    }
    
private:
    std::string GetCurrentLocation() {
        // In a real implementation, you'd use stack unwinding
        // or compiler intrinsics to get the actual call location
        return "unknown_location";
    }
};

// Convenience macros for tracked allocation
#define TRACKED_NEW(tracker, size) (tracker).TrackedAlloc(size, __FILE__ ":" + std::to_string(__LINE__))
#define TRACKED_DELETE(tracker, ptr) (tracker).TrackedFree(ptr)
```

### Thread-Safe Tracing Patterns

#### Multi-threaded Application Tracing

```cpp
#include <perfetto.h>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>

class ThreadSafeTracer {
private:
    struct TraceEvent {
        std::string category;
        std::string name;
        std::chrono::high_resolution_clock::time_point timestamp;
        std::thread::id thread_id;
        std::unordered_map<std::string, std::string> metadata;
    };
    
    std::queue<TraceEvent> event_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_{false};
    std::thread background_thread_;
    
public:
    ThreadSafeTracer() {
        background_thread_ = std::thread([this]() {
            ProcessEventQueue();
        });
    }
    
    ~ThreadSafeTracer() {
        shutdown_ = true;
        queue_cv_.notify_all();
        if (background_thread_.joinable()) {
            background_thread_.join();
        }
    }
    
    void LogEvent(const std::string& category, const std::string& name,
                  const std::unordered_map<std::string, std::string>& metadata = {}) {
        
        TraceEvent event{
            category,
            name,
            std::chrono::high_resolution_clock::now(),
            std::this_thread::get_id(),
            metadata
        };
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            event_queue_.push(std::move(event));
        }
        
        queue_cv_.notify_one();
    }
    
    template<typename Func>
    auto TraceFunction(const std::string& category, const std::string& name, Func&& func) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        LogEvent(category, name + "_start", {
            {"thread_id", std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()))}
        });
        
        try {
            auto result = func();
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time
            ).count();
            
            LogEvent(category, name + "_end", {
                {"duration_us", std::to_string(duration)},
                {"status", "success"}
            });
            
            return result;
        }
        catch (...) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time
            ).count();
            
            LogEvent(category, name + "_end", {
                {"duration_us", std::to_string(duration)},
                {"status", "error"}
            });
            
            throw;
        }
    }
    
private:
    void ProcessEventQueue() {
        while (!shutdown_) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() { 
                return !event_queue_.empty() || shutdown_; 
            });
            
            while (!event_queue_.empty()) {
                auto event = std::move(event_queue_.front());
                event_queue_.pop();
                lock.unlock();
                
                // Process event (write to Perfetto)
                ProcessSingleEvent(event);
                
                lock.lock();
            }
        }
    }
    
    void ProcessSingleEvent(const TraceEvent& event) {
        // Convert to Perfetto trace event
        TRACE_EVENT("app", event.name.c_str(),
            "category", event.category,
            "thread_id", std::hash<std::thread::id>{}(event.thread_id),
            "timestamp", std::chrono::duration_cast<std::chrono::microseconds>(
                event.timestamp.time_since_epoch()
            ).count()
        );
        
        // Add metadata
        for (const auto& [key, value] : event.metadata) {
            // Note: In real implementation, you'd need to handle
            // metadata differently as TRACE_EVENT has limited parameters
        }
    }
};

// Example usage with worker threads
class MultiThreadedApplication {
private:
    ThreadSafeTracer tracer_;
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{true};
    
public:
    void StartWorkers(int num_workers) {
        tracer_.LogEvent("app", "StartWorkers", {
            {"worker_count", std::to_string(num_workers)}
        });
        
        for (int i = 0; i < num_workers; ++i) {
            workers_.emplace_back([this, i]() {
                WorkerThread(i);
            });
        }
    }
    
    void StopWorkers() {
        tracer_.LogEvent("app", "StopWorkers");
        
        running_ = false;
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        workers_.clear();
    }
    
private:
    void WorkerThread(int worker_id) {
        tracer_.LogEvent("worker", "WorkerStarted", {
            {"worker_id", std::to_string(worker_id)}
        });
        
        while (running_) {
            // Simulate work with tracing
            tracer_.TraceFunction("worker", "ProcessTask_" + std::to_string(worker_id), [this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                return true;
            });
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        tracer_.LogEvent("worker", "WorkerStopped", {
            {"worker_id", std::to_string(worker_id)}
        });
    }
};
```

## Performance Monitoring Integration

### Real-time Performance Dashboard

#### Web-based Performance Monitor

```cpp
// performance_monitor.h
#pragma once
#include <perfetto.h>
#include <string>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <chrono>

class WebPerformanceMonitor {
private:
    struct MetricData {
        std::atomic<double> current_value{0.0};
        std::atomic<double> min_value{std::numeric_limits<double>::max()};
        std::atomic<double> max_value{std::numeric_limits<double>::lowest()};
        std::atomic<double> avg_value{0.0};
        std::atomic<size_t> sample_count{0};
    };
    
    std::unordered_map<std::string, std::unique_ptr<MetricData>> metrics_;
    std::atomic<bool> monitoring_active_{false};
    std::thread monitoring_thread_;
    std::thread web_server_thread_;
    
public:
    bool StartMonitoring(int web_port = 8080);
    void StopMonitoring();
    void UpdateMetric(const std::string& name, double value);
    std::string GenerateJSONReport() const;
    
private:
    void MonitoringLoop();
    void WebServerLoop(int port);
    void CollectSystemMetrics();
};

// performance_monitor.cpp
#include "performance_monitor.h"
#include <sstream>
#include <fstream>
#include <iostream>

bool WebPerformanceMonitor::StartMonitoring(int web_port) {
    if (monitoring_active_) return true;
    
    TRACE_EVENT("monitor", "StartPerformanceMonitoring", "web_port", web_port);
    
    monitoring_active_ = true;
    
    // Start monitoring thread
    monitoring_thread_ = std::thread([this]() {
        MonitoringLoop();
    });
    
    // Start web server thread
    web_server_thread_ = std::thread([this, web_port]() {
        WebServerLoop(web_port);
    });
    
    std::cout << "Performance monitoring started on port " << web_port << std::endl;
    return true;
}

void WebPerformanceMonitor::StopMonitoring() {
    if (!monitoring_active_) return;
    
    TRACE_EVENT("monitor", "StopPerformanceMonitoring");
    
    monitoring_active_ = false;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    if (web_server_thread_.joinable()) {
        web_server_thread_.join();
    }
    
    std::cout << "Performance monitoring stopped" << std::endl;
}

void WebPerformanceMonitor::UpdateMetric(const std::string& name, double value) {
    TRACE_COUNTER("metrics", name.c_str(), value);
    
    auto it = metrics_.find(name);
    if (it == metrics_.end()) {
        metrics_[name] = std::make_unique<MetricData>();
        it = metrics_.find(name);
    }
    
    auto& metric = *it->second;
    
    // Update current value
    metric.current_value = value;
    
    // Update min/max
    double current_min = metric.min_value.load();
    while (value < current_min && 
           !metric.min_value.compare_exchange_weak(current_min, value)) {
        // CAS loop
    }
    
    double current_max = metric.max_value.load();
    while (value > current_max && 
           !metric.max_value.compare_exchange_weak(current_max, value)) {
        // CAS loop  
    }
    
    // Update average (simplified running average)
    size_t count = metric.sample_count.fetch_add(1) + 1;
    double current_avg = metric.avg_value.load();
    double new_avg = ((current_avg * (count - 1)) + value) / count;
    metric.avg_value = new_avg;
}

void WebPerformanceMonitor::MonitoringLoop() {
    while (monitoring_active_) {
        TRACE_EVENT("monitor", "CollectMetrics");
        
        CollectSystemMetrics();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void WebPerformanceMonitor::CollectSystemMetrics() {
    // CPU Usage
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> cpu_dis(0.0, 100.0);
    UpdateMetric("cpu_usage_percent", cpu_dis(gen));
    
    // Memory Usage (simulated)
    static std::uniform_int_distribution<> mem_dis(1024, 4096);
    UpdateMetric("memory_usage_mb", mem_dis(gen));
    
    // Network throughput (simulated)
    static std::uniform_real_distribution<> net_dis(0.0, 1000.0);
    UpdateMetric("network_throughput_mbps", net_dis(gen));
    
    // Application-specific metrics
    UpdateMetric("active_connections", 50 + (rand() % 100));
    UpdateMetric("request_latency_ms", 10.0 + (rand() % 200));
    UpdateMetric("error_rate_percent", (rand() % 10) / 10.0);
}

std::string WebPerformanceMonitor::GenerateJSONReport() const {
    std::ostringstream json;
    json << "{\n";
    json << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count() << ",\n";
    json << "  \"metrics\": {\n";
    
    bool first = true;
    for (const auto& [name, metric] : metrics_) {
        if (!first) json << ",\n";
        first = false;
        
        json << "    \"" << name << "\": {\n";
        json << "      \"current\": " << metric->current_value.load() << ",\n";
        json << "      \"min\": " << metric->min_value.load() << ",\n";
        json << "      \"max\": " << metric->max_value.load() << ",\n";
        json << "      \"average\": " << metric->avg_value.load() << ",\n";
        json << "      \"samples\": " << metric->sample_count.load() << "\n";
        json << "    }";
    }
    
    json << "\n  }\n";
    json << "}";
    
    return json.str();
}

void WebPerformanceMonitor::WebServerLoop(int port) {
    // Simplified HTTP server implementation
    // In a real implementation, you'd use a proper HTTP library
    
    while (monitoring_active_) {
        try {
            // Generate HTML dashboard
            std::string html_content = R"(
<!DOCTYPE html>
<html>
<head>
    <title>Performance Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-card { 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            padding: 15px; 
            margin: 10px; 
            display: inline-block; 
            width: 300px;
        }
        .metric-title { font-weight: bold; color: #333; }
        .metric-value { font-size: 24px; color: #007bff; }
        .metric-stats { font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <h1>Real-time Performance Dashboard</h1>
    <div id="metrics-container"></div>
    
    <script>
        function updateMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('metrics-container');
                    container.innerHTML = '';
                    
                    for (const [name, metric] of Object.entries(data.metrics)) {
                        const card = document.createElement('div');
                        card.className = 'metric-card';
                        card.innerHTML = `
                            <div class="metric-title">${name}</div>
                            <div class="metric-value">${metric.current.toFixed(2)}</div>
                            <div class="metric-stats">
                                Min: ${metric.min.toFixed(2)} | 
                                Max: ${metric.max.toFixed(2)} | 
                                Avg: ${metric.average.toFixed(2)}
                            </div>
                        `;
                        container.appendChild(card);
                    }
                });
        }
        
        // Update every second
        setInterval(updateMetrics, 1000);
        updateMetrics(); // Initial load
    </script>
</body>
</html>
)";
            
            // In a real implementation, serve this HTML and handle /api/metrics endpoint
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
        } catch (const std::exception& e) {
            TRACE_EVENT("monitor", "WebServerError", "error", e.what());
        }
    }
}
```

### Application Performance Instrumentation

#### Automatic Performance Instrumentation

```cpp
#include <perfetto.h>
#include <chrono>
#include <string>
#include <unordered_map>

// RAII performance tracer
class ScopedPerformanceTracer {
private:
    std::string category_;
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
    bool ended_ = false;
    
public:
    ScopedPerformanceTracer(const std::string& category, const std::string& name)
        : category_(category), name_(name), start_time_(std::chrono::high_resolution_clock::now()) {
        
        TRACE_EVENT_BEGIN(category_.c_str(), name_.c_str());
    }
    
    ~ScopedPerformanceTracer() {
        if (!ended_) {
            End();
        }
    }
    
    void End() {
        if (ended_) return;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_
        ).count();
        
        TRACE_EVENT_END(category_.c_str(), name_.c_str(),
            "duration_us", duration
        );
        
        ended_ = true;
    }
    
    void AddMetadata(const std::string& key, const std::string& value) {
        TRACE_EVENT_INSTANT("metadata", "AddMetadata",
            "key", key, "value", value
        );
    }
};

// Macro for easy function tracing
#define TRACE_FUNCTION() ScopedPerformanceTracer __tracer__("function", __FUNCTION__)
#define TRACE_SCOPE(category, name) ScopedPerformanceTracer __tracer__(category, name)

// Performance statistics collector
class PerformanceStatistics {
private:
    struct FunctionStats {
        std::atomic<size_t> call_count{0};
        std::atomic<uint64_t> total_time_us{0};
        std::atomic<uint64_t> min_time_us{UINT64_MAX};
        std::atomic<uint64_t> max_time_us{0};
    };
    
    std::unordered_map<std::string, std::unique_ptr<FunctionStats>> function_stats_;
    mutable std::mutex stats_mutex_;
    
public:
    void RecordFunctionCall(const std::string& function_name, uint64_t duration_us) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        
        auto it = function_stats_.find(function_name);
        if (it == function_stats_.end()) {
            function_stats_[function_name] = std::make_unique<FunctionStats>();
            it = function_stats_.find(function_name);
        }
        
        auto& stats = *it->second;
        
        stats.call_count.fetch_add(1);
        stats.total_time_us.fetch_add(duration_us);
        
        // Update min time
        uint64_t current_min = stats.min_time_us.load();
        while (duration_us < current_min && 
               !stats.min_time_us.compare_exchange_weak(current_min, duration_us)) {
        }
        
        // Update max time
        uint64_t current_max = stats.max_time_us.load();
        while (duration_us > current_max && 
               !stats.max_time_us.compare_exchange_weak(current_max, duration_us)) {
        }
        
        TRACE_EVENT("stats", "FunctionStatistics",
            "function", function_name,
            "duration_us", duration_us,
            "total_calls", stats.call_count.load(),
            "avg_time_us", stats.total_time_us.load() / stats.call_count.load()
        );
    }
    
    void GenerateReport() {
        TRACE_EVENT("stats", "GeneratePerformanceReport");
        
        std::lock_guard<std::mutex> lock(stats_mutex_);
        
        for (const auto& [function_name, stats] : function_stats_) {
            uint64_t call_count = stats->call_count.load();
            uint64_t total_time = stats->total_time_us.load();
            uint64_t avg_time = call_count > 0 ? total_time / call_count : 0;
            
            TRACE_EVENT("report", "FunctionReport",
                "function", function_name,
                "total_calls", call_count,
                "total_time_us", total_time,
                "average_time_us", avg_time,
                "min_time_us", stats->min_time_us.load(),
                "max_time_us", stats->max_time_us.load()
            );
        }
    }
};

// Global performance statistics instance
PerformanceStatistics g_perf_stats;

// Enhanced performance tracer with statistics
class StatisticsPerformanceTracer {
private:
    std::string function_name_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
public:
    StatisticsPerformanceTracer(const std::string& function_name)
        : function_name_(function_name), start_time_(std::chrono::high_resolution_clock::now()) {
        
        TRACE_EVENT_BEGIN("perf", function_name_.c_str());
    }
    
    ~StatisticsPerformanceTracer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_
        ).count();
        
        TRACE_EVENT_END("perf", function_name_.c_str());
        
        g_perf_stats.RecordFunctionCall(function_name_, duration_us);
    }
};

#define TRACE_FUNCTION_STATS() StatisticsPerformanceTracer __stats_tracer__(__FUNCTION__)

// Example instrumented application
class InstrumentedApplication {
public:
    void ProcessData() {
        TRACE_FUNCTION_STATS();
        
        // Simulate data processing
        LoadData();
        TransformData();
        SaveData();
    }
    
private:
    void LoadData() {
        TRACE_FUNCTION_STATS();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    void TransformData() {
        TRACE_FUNCTION_STATS();
        
        // Nested operation tracing
        for (int i = 0; i < 10; ++i) {
            TRACE_SCOPE("transform", "ProcessBatch");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    
    void SaveData() {
        TRACE_FUNCTION_STATS();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
public:
    void RunBenchmark(int iterations) {
        TRACE_FUNCTION_STATS();
        
        for (int i = 0; i < iterations; ++i) {
            TRACE_SCOPE("benchmark", "Iteration");
            ProcessData();
        }
        
        // Generate performance report
        g_perf_stats.GenerateReport();
    }
};
```

### Integration with Existing Monitoring Systems

#### Prometheus Integration

```cpp
#include <perfetto.h>
#include <string>
#include <sstream>
#include <unordered_map>
#include <atomic>

class PrometheusIntegration {
private:
    struct MetricInfo {
        std::atomic<double> value{0.0};
        std::string help_text;
        std::string metric_type; // "counter", "gauge", "histogram"
    };
    
    std::unordered_map<std::string, std::unique_ptr<MetricInfo>> metrics_;
    mutable std::mutex metrics_mutex_;
    
public:
    void RegisterMetric(const std::string& name, const std::string& help, 
                       const std::string& type = "gauge") {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        metrics_[name] = std::make_unique<MetricInfo>();
        metrics_[name]->help_text = help;
        metrics_[name]->metric_type = type;
        
        TRACE_EVENT("prometheus", "RegisterMetric",
            "name", name, "type", type
        );
    }
    
    void UpdateMetric(const std::string& name, double value) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            it->second->value = value;
            
            // Also trace to Perfetto
            TRACE_COUNTER("prometheus", name.c_str(), value);
        }
    }
    
    void IncrementCounter(const std::string& name, double increment = 1.0) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            double old_value = it->second->value.load();
            it->second->value = old_value + increment;
            
            TRACE_COUNTER("prometheus", name.c_str(), it->second->value.load());
        }
    }
    
    std::string GeneratePrometheusFormat() const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        std::ostringstream output;
        
        for (const auto& [name, metric] : metrics_) {
            // Add help text
            output << "# HELP " << name << " " << metric->help_text << "\n";
            output << "# TYPE " << name << " " << metric->metric_type << "\n";
            output << name << " " << metric->value.load() << "\n";
        }
        
        TRACE_EVENT("prometheus", "GenerateFormat",
            "metrics_count", metrics_.size()
        );
        
        return output.str();
    }
};

// Grafana Dashboard Integration
class GrafanaIntegration {
public:
    static std::string GenerateDashboardJSON(const std::vector<std::string>& metrics) {
        TRACE_EVENT("grafana", "GenerateDashboard", 
            "metrics_count", metrics.size()
        );
        
        std::ostringstream dashboard;
        dashboard << R"({
  "dashboard": {
    "id": null,
    "title": "Perfetto Application Metrics",
    "tags": ["perfetto", "performance"],
    "timezone": "browser",
    "panels": [
)";
        
        for (size_t i = 0; i < metrics.size(); ++i) {
            if (i > 0) dashboard << ",\n";
            
            dashboard << R"(      {
        "id": )" << (i + 1) << R"(,
        "title": ")" << metrics[i] << R"(",
        "type": "graph",
        "targets": [
          {
            "expr": ")" << metrics[i] << R"(",
            "legendFormat": ")" << metrics[i] << R"("
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": )" << ((i % 2) * 12) << R"(,
          "y": )" << ((i / 2) * 9) << R"(
        }
      })";
        }
        
        dashboard << R"(
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
})";
        
        return dashboard.str();
    }
};
```

## Production Deployment

### Optimizing Trace Overhead

#### Low-Overhead Tracing Strategies

```cpp
#include <perfetto.h>
#include <atomic>
#include <thread>
#include <chrono>

class ProductionTracer {
private:
    // Configuration for production tracing
    struct TracingConfig {
        bool enabled = true;
        double sampling_rate = 0.01; // 1% sampling
        size_t buffer_size_kb = 512;
        size_t max_events_per_second = 1000;
        bool compress_traces = true;
    };
    
    TracingConfig config_;
    std::atomic<size_t> events_this_second_{0};
    std::atomic<uint64_t> last_second_timestamp_{0};
    std::atomic<bool> rate_limited_{false};
    
public:
    explicit ProductionTracer(const TracingConfig& config) : config_(config) {
        InitializeTracing();
    }
    
    // Fast path for high-frequency events
    template<typename... Args>
    void TraceIfSampled(const char* category, const char* name, Args&&... args) {
        // Quick sampling check
        if (!ShouldTrace()) return;
        
        // Rate limiting check
        if (IsRateLimited()) return;
        
        TRACE_EVENT(category, name, std::forward<Args>(args)...);
    }
    
    // Always trace critical events
    template<typename... Args>
    void TraceCritical(const char* category, const char* name, Args&&... args) {
        if (!config_.enabled) return;
        
        TRACE_EVENT(category, name, std::forward<Args>(args)...);
    }
    
    // Conditional tracing based on context
    template<typename Condition, typename... Args>
    void TraceIf(Condition&& condition, const char* category, const char* name, Args&&... args) {
        if (!config_.enabled || !condition()) return;
        
        TRACE_EVENT(category, name, std::forward<Args>(args)...);
    }
    
private:
    bool ShouldTrace() {
        // Fast rejection if disabled
        if (!config_.enabled) return false;
        
        // Sampling logic
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        static thread_local std::uniform_real_distribution<double> dis(0.0, 1.0);
        
        return dis(gen) < config_.sampling_rate;
    }
    
    bool IsRateLimited() {
        auto now_seconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
        
        uint64_t last_second = last_second_timestamp_.load();
        
        if (now_seconds != last_second) {
            // New second, reset counter
            if (last_second_timestamp_.compare_exchange_strong(last_second, now_seconds)) {
                events_this_second_ = 0;
                rate_limited_ = false;
            }
        }
        
        size_t current_events = events_this_second_.fetch_add(1);
        
        if (current_events >= config_.max_events_per_second) {
            rate_limited_ = true;
            return true;
        }
        
        return false;
    }
    
    void InitializeTracing() {
        perfetto::TraceConfig cfg;
        
        // Configure buffer
        auto* buffer = cfg.add_buffers();
        buffer->set_size_kb(config_.buffer_size_kb);
        
        auto* ds_cfg = cfg.add_data_sources()->mutable_config();
        ds_cfg->set_name("track_event");
        
        // Performance optimizations
        auto* track_event_cfg = ds_cfg->mutable_track_event_config();
        track_event_cfg->set_disabled_categories("debug"); // Disable debug traces in production
        
        TRACE_EVENT("production", "TracingInitialized",
            "sampling_rate", config_.sampling_rate,
            "buffer_size_kb", config_.buffer_size_kb,
            "max_events_per_second", config_.max_events_per_second
        );
    }
};

// Performance-critical application example
class HighPerformanceApp {
private:
    ProductionTracer tracer_;
    std::atomic<size_t> processed_requests_{0};
    
public:
    HighPerformanceApp() : tracer_({
        .enabled = true,
        .sampling_rate = 0.005, // 0.5% sampling for high-throughput app
        .buffer_size_kb = 1024,
        .max_events_per_second = 5000,
        .compress_traces = true
    }) {}
    
    void ProcessRequest(const std::string& request_id) {
        // Always trace request start/end for monitoring
        tracer_.TraceCritical("app", "RequestStart", "id", request_id);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Sample internal operations
            tracer_.TraceIfSampled("app", "ValidateRequest", "id", request_id);
            ValidateRequest(request_id);
            
            tracer_.TraceIfSampled("app", "ProcessBusinessLogic", "id", request_id);
            ProcessBusinessLogic(request_id);
            
            tracer_.TraceIfSampled("app", "GenerateResponse", "id", request_id);
            GenerateResponse(request_id);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time
            ).count();
            
            tracer_.TraceCritical("app", "RequestCompleted", 
                "id", request_id,
                "duration_ms", duration_ms,
                "status", "success"
            );
            
            processed_requests_.fetch_add(1);
            
        } catch (const std::exception& e) {
            tracer_.TraceCritical("app", "RequestError", 
                "id", request_id,
                "error", e.what()
            );
        }
    }
    
private:
    void ValidateRequest(const std::string& request_id) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    void ProcessBusinessLogic(const std::string& request_id) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    void GenerateResponse(const std::string& request_id) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
};
```

## Cross-Platform Considerations

### Platform-Specific Implementations

#### Windows-Specific Tracing Integration

```cpp
#include <perfetto.h>
#include <windows.h>
#include <evntprov.h> // ETW integration
#include <pdh.h>      // Performance counters

class WindowsPerfettoIntegration {
private:
    REGHANDLE etw_provider_handle_ = 0;
    PDH_HQUERY pdh_query_ = nullptr;
    std::unordered_map<std::string, PDH_HCOUNTER> performance_counters_;
    
public:
    bool Initialize() {
        TRACE_EVENT("windows", "InitializePerfettoIntegration");
        
        // Register ETW provider
        GUID provider_guid = {0x12345678, 0x1234, 0x1234, {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0}};
        ULONG result = EventRegister(&provider_guid, nullptr, nullptr, &etw_provider_handle_);
        
        if (result != ERROR_SUCCESS) {
            TRACE_EVENT("windows", "ETWRegistrationFailed", "error_code", result);
            return false;
        }
        
        // Initialize PDH for performance counters
        result = PdhOpenQuery(nullptr, 0, &pdh_query_);
        if (result != ERROR_SUCCESS) {
            TRACE_EVENT("windows", "PDHInitializationFailed", "error_code", result);
            return false;
        }
        
        // Add common performance counters
        AddPerformanceCounter("CPU", "\\Processor(_Total)\\% Processor Time");
        AddPerformanceCounter("Memory", "\\Memory\\Available MBytes");
        AddPerformanceCounter("Disk", "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec");
        
        return true;
    }
    
    void Shutdown() {
        TRACE_EVENT("windows", "ShutdownPerfettoIntegration");
        
        if (etw_provider_handle_) {
            EventUnregister(etw_provider_handle_);
            etw_provider_handle_ = 0;
        }
        
        if (pdh_query_) {
            PdhCloseQuery(pdh_query_);
            pdh_query_ = nullptr;
        }
    }
    
    void LogToETW(const std::string& message, UCHAR level = TRACE_LEVEL_INFORMATION) {
        if (etw_provider_handle_ == 0) return;
        
        EVENT_DESCRIPTOR event_desc;
        EventDescCreate(&event_desc, 1, 0, 0, level, 0, 0, 0);
        
        EVENT_DATA_DESCRIPTOR data_desc;
        EventDataDescCreate(&data_desc, message.c_str(), 
                           static_cast<ULONG>(message.length() + 1));
        
        EventWrite(etw_provider_handle_, &event_desc, 1, &data_desc);
        
        // Also trace to Perfetto
        TRACE_EVENT("etw", "ETWEvent", 
            "message", message,
            "level", level
        );
    }
    
    void CollectWindowsMetrics() {
        TRACE_EVENT("windows", "CollectSystemMetrics");
        
        // Collect PDH counters
        PDH_STATUS status = PdhCollectQueryData(pdh_query_);
        if (status == ERROR_SUCCESS) {
            for (const auto& [name, counter] : performance_counters_) {
                PDH_FMT_COUNTERVALUE counter_value;
                status = PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, 
                                                   nullptr, &counter_value);
                
                if (status == ERROR_SUCCESS) {
                    TRACE_COUNTER("windows", name.c_str(), counter_value.doubleValue);
                }
            }
        }
        
        // Collect additional Windows-specific metrics
        CollectProcessInformation();
        CollectMemoryInformation();
        CollectHandleInformation();
    }
    
private:
    bool AddPerformanceCounter(const std::string& name, const std::string& path) {
        PDH_HCOUNTER counter;
        PDH_STATUS status = PdhAddEnglishCounter(pdh_query_, path.c_str(), 0, &counter);
        
        if (status == ERROR_SUCCESS) {
            performance_counters_[name] = counter;
            TRACE_EVENT("windows", "PerformanceCounterAdded", 
                "name", name, "path", path);
            return true;
        } else {
            TRACE_EVENT("windows", "PerformanceCounterFailed", 
                "name", name, "path", path, "error", status);
            return false;
        }
    }
    
    void CollectProcessInformation() {
        PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), 
                               reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), 
                               sizeof(pmc))) {
            
            TRACE_COUNTER("process", "WorkingSetSize", pmc.WorkingSetSize);
            TRACE_COUNTER("process", "PeakWorkingSetSize", pmc.PeakWorkingSetSize);
            TRACE_COUNTER("process", "PrivateUsage", pmc.PrivateUsage);
        }
        
        // Get handle count
        DWORD handle_count = 0;
        if (GetProcessHandleCount(GetCurrentProcess(), &handle_count)) {
            TRACE_COUNTER("process", "HandleCount", handle_count);
        }
    }
    
    void CollectMemoryInformation() {
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        
        if (GlobalMemoryStatusEx(&mem_status)) {
            TRACE_COUNTER("system", "TotalPhysicalMemory", mem_status.ullTotalPhys);
            TRACE_COUNTER("system", "AvailablePhysicalMemory", mem_status.ullAvailPhys);
            TRACE_COUNTER("system", "MemoryLoad", mem_status.dwMemoryLoad);
        }
    }
    
    void CollectHandleInformation() {
        // Get thread count for current process
        DWORD process_id = GetCurrentProcessId();
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        
        if (snapshot != INVALID_HANDLE_VALUE) {
            THREADENTRY32 thread_entry;
            thread_entry.dwSize = sizeof(thread_entry);
            
            DWORD thread_count = 0;
            if (Thread32First(snapshot, &thread_entry)) {
                do {
                    if (thread_entry.th32OwnerProcessID == process_id) {
                        thread_count++;
                    }
                } while (Thread32Next(snapshot, &thread_entry));
            }
            
            CloseHandle(snapshot);
            TRACE_COUNTER("process", "ThreadCount", thread_count);
        }
    }
};
```

#### Linux-Specific Integration

```cpp
#include <perfetto.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <thread>

class LinuxPerfettoIntegration {
private:
    std::thread perf_monitor_thread_;
    std::atomic<bool> monitoring_active_{false};
    
public:
    bool Initialize() {
        TRACE_EVENT("linux", "InitializePerfettoIntegration");
        
        // Check if running with appropriate permissions
        if (geteuid() != 0) {
            TRACE_EVENT("linux", "Warning", "message", "Not running as root - some metrics may be unavailable");
        }
        
        // Start performance monitoring
        StartPerformanceMonitoring();
        
        return true;
    }
    
    void Shutdown() {
        TRACE_EVENT("linux", "ShutdownPerfettoIntegration");
        
        StopPerformanceMonitoring();
    }
    
    void CollectLinuxMetrics() {
        TRACE_EVENT("linux", "CollectSystemMetrics");
        
        CollectCPUStatistics();
        CollectMemoryStatistics();
        CollectNetworkStatistics();
        CollectProcessStatistics();
        CollectFileSystemStatistics();
    }
    
private:
    void StartPerformanceMonitoring() {
        monitoring_active_ = true;
        perf_monitor_thread_ = std::thread([this]() {
            while (monitoring_active_) {
                CollectLinuxMetrics();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
    }
    
    void StopPerformanceMonitoring() {
        monitoring_active_ = false;
        if (perf_monitor_thread_.joinable()) {
            perf_monitor_thread_.join();
        }
    }
    
    void CollectCPUStatistics() {
        std::ifstream stat_file("/proc/stat");
        if (!stat_file.is_open()) return;
        
        std::string line;
        if (std::getline(stat_file, line)) {
            std::istringstream iss(line);
            std::string cpu_label;
            long long user, nice, system, idle, iowait, irq, softirq, steal;
            
            iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
            
            long long total = user + nice + system + idle + iowait + irq + softirq + steal;
            long long active = total - idle - iowait;
            
            double cpu_usage = (active * 100.0) / total;
            
            TRACE_COUNTER("linux", "CPUUsagePercent", cpu_usage);
            TRACE_COUNTER("linux", "CPUUserTime", user);
            TRACE_COUNTER("linux", "CPUSystemTime", system);
            TRACE_COUNTER("linux", "CPUIdleTime", idle);
            TRACE_COUNTER("linux", "CPUIOwaitTime", iowait);
        }
    }
    
    void CollectMemoryStatistics() {
        std::ifstream meminfo("/proc/meminfo");
        if (!meminfo.is_open()) return;
        
        std::string line;
        std::unordered_map<std::string, long long> memory_values;
        
        while (std::getline(meminfo, line)) {
            std::istringstream iss(line);
            std::string key, value_str, unit;
            
            if (iss >> key >> value_str >> unit) {
                key.pop_back(); // Remove ':')
                long long value = std::stoll(value_str);
                
                if (unit == "kB") {
                    value *= 1024; // Convert to bytes
                }
                
                memory_values[key] = value;
            }
        }
        
        // Trace important memory metrics
        if (memory_values.count("MemTotal")) {
            TRACE_COUNTER("linux", "MemoryTotal", memory_values["MemTotal"]);
        }
        if (memory_values.count("MemAvailable")) {
            TRACE_COUNTER("linux", "MemoryAvailable", memory_values["MemAvailable"]);
        }
        if (memory_values.count("MemFree")) {
            TRACE_COUNTER("linux", "MemoryFree", memory_values["MemFree"]);
        }
        if (memory_values.count("Buffers")) {
            TRACE_COUNTER("linux", "MemoryBuffers", memory_values["Buffers"]);
        }
        if (memory_values.count("Cached")) {
            TRACE_COUNTER("linux", "MemoryCached", memory_values["Cached"]);
        }
        if (memory_values.count("SwapTotal")) {
            TRACE_COUNTER("linux", "SwapTotal", memory_values["SwapTotal"]);
        }
        if (memory_values.count("SwapFree")) {
            TRACE_COUNTER("linux", "SwapFree", memory_values["SwapFree"]);
        }
    }
    
    void CollectNetworkStatistics() {
        std::ifstream netdev("/proc/net/dev");
        if (!netdev.is_open()) return;
        
        std::string line;
        // Skip first two header lines
        std::getline(netdev, line);
        std::getline(netdev, line);
        
        long long total_rx_bytes = 0, total_tx_bytes = 0;
        long long total_rx_packets = 0, total_tx_packets = 0;
        
        while (std::getline(netdev, line)) {
            std::istringstream iss(line);
            std::string interface;
            long long rx_bytes, rx_packets, rx_errs, rx_drop, rx_fifo, rx_frame, rx_compressed, rx_multicast;
            long long tx_bytes, tx_packets, tx_errs, tx_drop, tx_fifo, tx_colls, tx_carrier, tx_compressed;
            
            if (iss >> interface >> rx_bytes >> rx_packets >> rx_errs >> rx_drop >> rx_fifo >> rx_frame >> rx_compressed >> rx_multicast
                    >> tx_bytes >> tx_packets >> tx_errs >> tx_drop >> tx_fifo >> tx_colls >> tx_carrier >> tx_compressed) {
                
                // Skip loopback interface
                if (interface.find("lo:") == 0) continue;
                
                total_rx_bytes += rx_bytes;
                total_tx_bytes += tx_bytes;
                total_rx_packets += rx_packets;
                total_tx_packets += tx_packets;
            }
        }
        
        TRACE_COUNTER("linux", "NetworkRxBytes", total_rx_bytes);
        TRACE_COUNTER("linux", "NetworkTxBytes", total_tx_bytes);
        TRACE_COUNTER("linux", "NetworkRxPackets", total_rx_packets);
        TRACE_COUNTER("linux", "NetworkTxPackets", total_tx_packets);
    }
    
    void CollectProcessStatistics() {
        // Get process statistics from /proc/self/status
        std::ifstream status("/proc/self/status");
        if (!status.is_open()) return;
        
        std::string line;
        while (std::getline(status, line)) {
            std::istringstream iss(line);
            std::string key, value_str, unit;
            
            if (iss >> key >> value_str >> unit) {
                if (key == "VmRSS:") {
                    long long rss = std::stoll(value_str) * 1024; // Convert kB to bytes
                    TRACE_COUNTER("process", "RSS", rss);
                } else if (key == "VmSize:") {
                    long long vsize = std::stoll(value_str) * 1024;
                    TRACE_COUNTER("process", "VirtualSize", vsize);
                } else if (key == "Threads:") {
                    TRACE_COUNTER("process", "ThreadCount", std::stoll(value_str));
                } else if (key == "FDSize:") {
                    TRACE_COUNTER("process", "FileDescriptorCount", std::stoll(value_str));
                }
            }
        }
        
        // Get resource usage
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            TRACE_COUNTER("process", "UserCPUTime", usage.ru_utime.tv_sec * 1000000 + usage.ru_utime.tv_usec);
            TRACE_COUNTER("process", "SystemCPUTime", usage.ru_stime.tv_sec * 1000000 + usage.ru_stime.tv_usec);
            TRACE_COUNTER("process", "MaxRSS", usage.ru_maxrss * 1024); // Convert kB to bytes
            TRACE_COUNTER("process", "PageFaults", usage.ru_majflt);
            TRACE_COUNTER("process", "ContextSwitches", usage.ru_nvcsw + usage.ru_nivcsw);
        }
    }
    
    void CollectFileSystemStatistics() {
        std::ifstream diskstats("/proc/diskstats");
        if (!diskstats.is_open()) return;
        
        std::string line;
        long long total_read_sectors = 0, total_write_sectors = 0;
        
        while (std::getline(diskstats, line)) {
            std::istringstream iss(line);
            int major, minor;
            std::string device;
            long long reads_completed, reads_merged, sectors_read, time_reading;
            long long writes_completed, writes_merged, sectors_written, time_writing;
            long long ios_in_progress, time_ios, weighted_time_ios;
            
            if (iss >> major >> minor >> device >> reads_completed >> reads_merged >> sectors_read >> time_reading
                    >> writes_completed >> writes_merged >> sectors_written >> time_writing
                    >> ios_in_progress >> time_ios >> weighted_time_ios) {
                
                // Only include physical devices (not partitions)
                if (device.find_first_of("0123456789") == std::string::npos) {
                    total_read_sectors += sectors_read;
                    total_write_sectors += sectors_written;
                }
            }
        }
        
        // Convert sectors to bytes (assuming 512 bytes per sector)
        TRACE_COUNTER("linux", "DiskReadBytes", total_read_sectors * 512);
        TRACE_COUNTER("linux", "DiskWriteBytes", total_write_sectors * 512);
    }
};
```

#### macOS-Specific Integration

```cpp
#include <perfetto.h>
#include <mach/mach.h>
#include <mach/processor_info.h>
#include <mach/mach_host.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <libproc.h>

class MacOSPerfettoIntegration {
private:
    mach_port_t host_port_;
    
public:
    bool Initialize() {
        TRACE_EVENT("macos", "InitializePerfettoIntegration");
        
        host_port_ = mach_host_self();
        if (host_port_ == MACH_PORT_NULL) {
            TRACE_EVENT("macos", "InitializationFailed", "error", "Failed to get host port");
            return false;
        }
        
        return true;
    }
    
    void Shutdown() {
        TRACE_EVENT("macos", "ShutdownPerfettoIntegration");
        
        if (host_port_ != MACH_PORT_NULL) {
            mach_port_deallocate(mach_task_self(), host_port_);
            host_port_ = MACH_PORT_NULL;
        }
    }
    
    void CollectMacOSMetrics() {
        TRACE_EVENT("macos", "CollectSystemMetrics");
        
        CollectCPUStatistics();
        CollectMemoryStatistics();
        CollectProcessStatistics();
        CollectSystemInformation();
    }
    
private:
    void CollectCPUStatistics() {
        processor_info_array_t cpu_info;
        mach_msg_type_number_t num_cpu_info;
        natural_t num_cpus;
        
        kern_return_t kr = host_processor_info(host_port_, PROCESSOR_CPU_LOAD_INFO,
                                             &num_cpus, &cpu_info, &num_cpu_info);
        
        if (kr == KERN_SUCCESS) {
            processor_cpu_load_info_t cpu_load = (processor_cpu_load_info_t)cpu_info;
            
            uint64_t total_user = 0, total_system = 0, total_idle = 0, total_nice = 0;
            
            for (natural_t i = 0; i < num_cpus; i++) {
                total_user += cpu_load[i].cpu_ticks[CPU_STATE_USER];
                total_system += cpu_load[i].cpu_ticks[CPU_STATE_SYSTEM];
                total_idle += cpu_load[i].cpu_ticks[CPU_STATE_IDLE];
                total_nice += cpu_load[i].cpu_ticks[CPU_STATE_NICE];
            }
            
            uint64_t total_ticks = total_user + total_system + total_idle + total_nice;
            
            if (total_ticks > 0) {
                double user_percent = (total_user * 100.0) / total_ticks;
                double system_percent = (total_system * 100.0) / total_ticks;
                double idle_percent = (total_idle * 100.0) / total_ticks;
                
                TRACE_COUNTER("macos", "CPUUserPercent", user_percent);
                TRACE_COUNTER("macos", "CPUSystemPercent", system_percent);
                TRACE_COUNTER("macos", "CPUIdlePercent", idle_percent);
                TRACE_COUNTER("macos", "CPUUsagePercent", 100.0 - idle_percent);
            }
            
            vm_deallocate(mach_task_self(), (vm_address_t)cpu_info, 
                         num_cpu_info * sizeof(*cpu_info));
        }
    }
    
    void CollectMemoryStatistics() {
        vm_statistics64_data_t vm_stats;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        
        kern_return_t kr = host_statistics64(host_port_, HOST_VM_INFO64,
                                           (host_info64_t)&vm_stats, &count);
        
        if (kr == KERN_SUCCESS) {
            // Get page size
            vm_size_t page_size;
            host_page_size(host_port_, &page_size);
            
            uint64_t free_memory = vm_stats.free_count * page_size;
            uint64_t active_memory = vm_stats.active_count * page_size;
            uint64_t inactive_memory = vm_stats.inactive_count * page_size;
            uint64_t wired_memory = vm_stats.wire_count * page_size;
            uint64_t compressed_memory = vm_stats.compressor_page_count * page_size;
            
            TRACE_COUNTER("macos", "MemoryFree", free_memory);
            TRACE_COUNTER("macos", "MemoryActive", active_memory);
            TRACE_COUNTER("macos", "MemoryInactive", inactive_memory);
            TRACE_COUNTER("macos", "MemoryWired", wired_memory);
            TRACE_COUNTER("macos", "MemoryCompressed", compressed_memory);
            
            // Get total memory
            int64_t total_memory;
            size_t size = sizeof(total_memory);
            if (sysctlbyname("hw.memsize", &total_memory, &size, nullptr, 0) == 0) {
                TRACE_COUNTER("macos", "MemoryTotal", total_memory);
                
                uint64_t used_memory = total_memory - free_memory;
                double memory_pressure = (used_memory * 100.0) / total_memory;
                TRACE_COUNTER("macos", "MemoryUsagePercent", memory_pressure);
            }
        }
    }
    
    void CollectProcessStatistics() {
        mach_task_basic_info_data_t task_info;
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        
        kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                   (task_info_t)&task_info, &count);
        
        if (kr == KERN_SUCCESS) {
            TRACE_COUNTER("process", "ResidentSize", task_info.resident_size);
            TRACE_COUNTER("process", "VirtualSize", task_info.virtual_size);
        }
        
        // Get process info using libproc
        struct proc_taskinfo task_info_proc;
        int ret = proc_pidinfo(getpid(), PROC_PIDTASKINFO, 0, &task_info_proc, sizeof(task_info_proc));
        
        if (ret == sizeof(task_info_proc)) {
            TRACE_COUNTER("process", "UserTime", task_info_proc.pti_total_user);
            TRACE_COUNTER("process", "SystemTime", task_info_proc.pti_total_system);
            TRACE_COUNTER("process", "ThreadCount", task_info_proc.pti_threadnum);
            TRACE_COUNTER("process", "PageFaults", task_info_proc.pti_faults);
            TRACE_COUNTER("process", "PageReclaims", task_info_proc.pti_pageins);
        }
    }
    
    void CollectSystemInformation() {
        // Get system load averages
        double load_averages[3];
        if (getloadavg(load_averages, 3) != -1) {
            TRACE_COUNTER("macos", "LoadAverage1Min", load_averages[0]);
            TRACE_COUNTER("macos", "LoadAverage5Min", load_averages[1]);
            TRACE_COUNTER("macos", "LoadAverage15Min", load_averages[2]);
        }
        
        // Get system uptime
        struct timeval boot_time;
        size_t size = sizeof(boot_time);
        if (sysctlbyname("kern.boottime", &boot_time, &size, nullptr, 0) == 0) {
            struct timeval current_time;
            gettimeofday(&current_time, nullptr);
            
            uint64_t uptime = (current_time.tv_sec - boot_time.tv_sec) * 1000000 +
                             (current_time.tv_usec - boot_time.tv_usec);
            
            TRACE_COUNTER("macos", "SystemUptimeMicroseconds", uptime);
        }
    }
};
```

### Unified Cross-Platform Interface

```cpp
#include <perfetto.h>
#include <memory>

class PlatformTracer {
private:
    std::unique_ptr<void, void(*)(void*)> platform_impl_;
    
public:
    PlatformTracer() : platform_impl_(nullptr, [](void*){}) {
        Initialize();
    }
    
    void Initialize() {
        TRACE_EVENT("platform", "InitializePlatformTracer");
        
        #ifdef _WIN32
            auto* windows_impl = new WindowsPerfettoIntegration();
            if (windows_impl->Initialize()) {
                platform_impl_ = std::unique_ptr<void, void(*)(void*)>(
                    windows_impl, 
                    [](void* ptr) { 
                        static_cast<WindowsPerfettoIntegration*>(ptr)->Shutdown();
                        delete static_cast<WindowsPerfettoIntegration*>(ptr); 
                    }
                );
            } else {
                delete windows_impl;
            }
        #elif defined(__APPLE__)
            auto* macos_impl = new MacOSPerfettoIntegration();
            if (macos_impl->Initialize()) {
                platform_impl_ = std::unique_ptr<void, void(*)(void*)>(
                    macos_impl,
                    [](void* ptr) { 
                        static_cast<MacOSPerfettoIntegration*>(ptr)->Shutdown();
                        delete static_cast<MacOSPerfettoIntegration*>(ptr); 
                    }
                );
            } else {
                delete macos_impl;
            }
        #elif defined(__linux__)
            auto* linux_impl = new LinuxPerfettoIntegration();
            if (linux_impl->Initialize()) {
                platform_impl_ = std::unique_ptr<void, void(*)(void*)>(
                    linux_impl,
                    [](void* ptr) { 
                        static_cast<LinuxPerfettoIntegration*>(ptr)->Shutdown();
                        delete static_cast<LinuxPerfettoIntegration*>(ptr); 
                    }
                );
            } else {
                delete linux_impl;
            }
        #endif
    }
    
    void CollectMetrics() {
        if (!platform_impl_) return;
        
        #ifdef _WIN32
            static_cast<WindowsPerfettoIntegration*>(platform_impl_.get())->CollectWindowsMetrics();
        #elif defined(__APPLE__)
            static_cast<MacOSPerfettoIntegration*>(platform_impl_.get())->CollectMacOSMetrics();
        #elif defined(__linux__)
            static_cast<LinuxPerfettoIntegration*>(platform_impl_.get())->CollectLinuxMetrics();
        #endif
    }
    
    bool IsInitialized() const {
        return platform_impl_ != nullptr;
    }
};

// Global platform tracer instance
PlatformTracer g_platform_tracer;

// Convenient macros that work across platforms
#define TRACE_PLATFORM_METRICS() g_platform_tracer.CollectMetrics()
#define TRACE_CROSS_PLATFORM(category, name, ...) \
    do { \
        TRACE_EVENT(category, name, ##__VA_ARGS__); \
        if (g_platform_tracer.IsInitialized()) { \
            TRACE_PLATFORM_METRICS(); \
        } \
    } while(0)
```

## Best Practices and Optimization

### Performance Impact Minimization

#### Zero-Cost Abstractions for Tracing

```cpp
#include <perfetto.h>
#include <type_traits>
#include <chrono>

// Compile-time tracing configuration
template<bool Enabled>
class ConditionalTracer {
public:
    template<typename... Args>
    static constexpr void TraceEvent(Args&&...) {
        // No-op when disabled - completely optimized away
    }
    
    template<typename... Args>
    static constexpr void TraceCounter(Args&&...) {
        // No-op when disabled
    }
};

// Specialization for enabled tracing
template<>
class ConditionalTracer<true> {
public:
    template<typename... Args>
    static void TraceEvent(const char* category, const char* name, Args&&... args) {
        TRACE_EVENT(category, name, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void TraceCounter(const char* category, const char* name, Args&&... args) {
        TRACE_COUNTER(category, name, std::forward<Args>(args)...);
    }
};

// Configuration flags
constexpr bool TRACING_ENABLED = 
    #ifdef NDEBUG
        false  // Disabled in release builds
    #else
        true   // Enabled in debug builds
    #endif
;

constexpr bool PERFORMANCE_TRACING_ENABLED = true;
constexpr bool VERBOSE_TRACING_ENABLED = false;

// Type aliases for different tracing levels
using GeneralTracer = ConditionalTracer<TRACING_ENABLED>;
using PerformanceTracer = ConditionalTracer<PERFORMANCE_TRACING_ENABLED>;
using VerboseTracer = ConditionalTracer<VERBOSE_TRACING_ENABLED>;

// Lightweight RAII tracer that optimizes away when disabled
template<bool Enabled>
class ScopedTracer {
public:
    template<typename... Args>
    ScopedTracer(Args&&...) {} // No-op constructor when disabled
    
    ~ScopedTracer() = default;
};

// Specialization for enabled tracing
template<>
class ScopedTracer<true> {
private:
    const char* category_;
    const char* name_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
public:
    ScopedTracer(const char* category, const char* name) 
        : category_(category), name_(name), start_time_(std::chrono::high_resolution_clock::now()) {
        TRACE_EVENT_BEGIN(category_, name_);
    }
    
    ~ScopedTracer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_
        ).count();
        
        TRACE_EVENT_END(category_, name_, "duration_us", duration);
    }
};

// Convenient macros that disappear in optimized builds
#define TRACE_FUNCTION_PERF() ScopedTracer<PERFORMANCE_TRACING_ENABLED> __tracer__("perf", __FUNCTION__)
#define TRACE_FUNCTION_VERBOSE() ScopedTracer<VERBOSE_TRACING_ENABLED> __tracer__("verbose", __FUNCTION__)
#define TRACE_FUNCTION_GENERAL() ScopedTracer<TRACING_ENABLED> __tracer__("general", __FUNCTION__)

// Fast path macros for high-frequency operations
#define TRACE_FAST_PATH(category, name) \
    do { \
        if constexpr (PERFORMANCE_TRACING_ENABLED) { \
            TRACE_EVENT_INSTANT(category, name); \
        } \
    } while(0)

#define TRACE_COUNTER_FAST(category, name, value) \
    do { \
        if constexpr (PERFORMANCE_TRACING_ENABLED) { \
            TRACE_COUNTER(category, name, value); \
        } \
    } while(0)
```

#### Memory-Efficient Tracing

```cpp
#include <perfetto.h>
#include <string_view>
#include <array>

class MemoryOptimizedTracer {
private:
    // String interning for reducing memory usage
    static constexpr size_t MAX_INTERNED_STRINGS = 1000;
    std::array<std::string_view, MAX_INTERNED_STRINGS> interned_strings_;
    std::atomic<size_t> string_count_{0};
    
    // Circular buffer for trace data
    static constexpr size_t BUFFER_SIZE = 10000;
    struct TraceEntry {
        std::string_view category;
        std::string_view name;
        uint64_t timestamp;
        uint32_t thread_id;
    };
    
    std::array<TraceEntry, BUFFER_SIZE> trace_buffer_;
    std::atomic<size_t> buffer_index_{0};
    
public:
    // Intern strings to reduce memory usage
    std::string_view InternString(const std::string& str) {
        size_t index = string_count_.load();
        if (index >= MAX_INTERNED_STRINGS) {
            return str; // Fallback to original string
        }
        
        if (string_count_.compare_exchange_strong(index, index + 1)) {
            interned_strings_[index] = str;
            return interned_strings_[index];
        }
        
        return str; // Fallback if CAS failed
    }
    
    // Lightweight trace event recording
    void RecordEvent(std::string_view category, std::string_view name) {
        size_t index = buffer_index_.fetch_add(1) % BUFFER_SIZE;
        
        trace_buffer_[index] = {
            category,
            name,
            GetTimestamp(),
            GetCurrentThreadId()
        };
        
        // Also record to Perfetto if enabled
        if constexpr (PERFORMANCE_TRACING_ENABLED) {
            TRACE_EVENT_INSTANT(category.data(), name.data());
        }
    }
    
    // Batch flush to Perfetto
    void FlushToPerfetto() {
        TRACE_EVENT("tracer", "FlushToPerfetto");
        
        size_t current_index = buffer_index_.load();
        size_t start_index = current_index >= BUFFER_SIZE ? current_index - BUFFER_SIZE : 0;
        
        for (size_t i = start_index; i < current_index; ++i) {
            const auto& entry = trace_buffer_[i % BUFFER_SIZE];
            
            TRACE_EVENT("batched", entry.name.data(),
                "original_category", entry.category.data(),
                "timestamp", entry.timestamp,
                "thread_id", entry.thread_id
            );
        }
    }
    
private:
    uint64_t GetTimestamp() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }
    
    uint32_t GetCurrentThreadId() {
        return std::hash<std::thread::id>{}(std::this_thread::get_id());
    }
};

// Global optimized tracer
MemoryOptimizedTracer g_optimized_tracer;

// Memory-efficient macros
#define TRACE_OPTIMIZED(category, name) \
    g_optimized_tracer.RecordEvent( \
        g_optimized_tracer.InternString(category), \
        g_optimized_tracer.InternString(name) \
    )
```

### Advanced Integration Patterns

#### Microservice Tracing Architecture

```cpp
#include <perfetto.h>
#include <string>
#include <unordered_map>
#include <queue>
#include <future>

class MicroserviceTracer {
private:
    struct DistributedTrace {
        std::string trace_id;
        std::string span_id;
        std::string parent_span_id;
        std::string service_name;
        std::unordered_map<std::string, std::string> baggage;
    };
    
    thread_local DistributedTrace current_trace_;
    std::string service_name_;
    std::string service_version_;
    
public:
    explicit MicroserviceTracer(const std::string& service_name, const std::string& version)
        : service_name_(service_name), service_version_(version) {
        
        TRACE_EVENT("microservice", "ServiceInitialized",
            "service_name", service_name_,
            "service_version", service_version_
        );
    }
    
    // Start a new distributed trace
    std::string StartTrace(const std::string& operation_name) {
        current_trace_.trace_id = GenerateTraceId();
        current_trace_.span_id = GenerateSpanId();
        current_trace_.parent_span_id = "";
        current_trace_.service_name = service_name_;
        
        TRACE_EVENT_BEGIN("distributed", operation_name.c_str(),
            "trace_id", current_trace_.trace_id,
            "span_id", current_trace_.span_id,
            "service_name", service_name_,
            "service_version", service_version_
        );
        
        return current_trace_.trace_id;
    }
    
    // Continue a distributed trace from upstream service
    void ContinueTrace(const std::string& trace_id, const std::string& parent_span_id,
                      const std::string& operation_name) {
        current_trace_.trace_id = trace_id;
        current_trace_.span_id = GenerateSpanId();
        current_trace_.parent_span_id = parent_span_id;
        current_trace_.service_name = service_name_;
        
        TRACE_EVENT_BEGIN("distributed", operation_name.c_str(),
            "trace_id", current_trace_.trace_id,
            "span_id", current_trace_.span_id,
            "parent_span_id", current_trace_.parent_span_id,
            "service_name", service_name_,
            "service_version", service_version_
        );
    }
    
    // Add baggage (cross-service metadata)
    void AddBaggage(const std::string& key, const std::string& value) {
        current_trace_.baggage[key] = value;
        
        TRACE_EVENT("distributed", "BaggageAdded",
            "key", key,
            "value", value,
            "trace_id", current_trace_.trace_id
        );
    }
    
    // Create child span for downstream service call
    std::pair<std::string, std::string> CreateChildSpan(const std::string& downstream_service) {
        std::string child_span_id = GenerateSpanId();
        
        TRACE_EVENT_NESTABLE_ASYNC_BEGIN("distributed", "DownstreamCall",
            TRACE_ID_GLOBAL(std::hash<std::string>{}(current_trace_.trace_id)),
            "downstream_service", downstream_service,
            "parent_span_id", current_trace_.span_id,
            "child_span_id", child_span_id
        );
        
        return {current_trace_.trace_id, child_span_id};
    }
    
    // End current span
    void EndSpan(const std::string& status = "success") {
        TRACE_EVENT_END("distributed", 
            "trace_id", current_trace_.trace_id,
            "span_id", current_trace_.span_id,
            "status", status
        );
        
        // Add baggage to trace
        for (const auto& [key, value] : current_trace_.baggage) {
            TRACE_EVENT("distributed", "Baggage",
                "key", key, "value", value
            );
        }
    }
    
    // Get current trace context for propagation
    std::unordered_map<std::string, std::string> GetTraceContext() const {
        return {
            {"trace-id", current_trace_.trace_id},
            {"span-id", current_trace_.span_id},
            {"service-name", service_name_}
        };
    }
    
private:
    std::string GenerateTraceId() {
        // Generate unique trace ID
        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        
        std::ostringstream oss;
        oss << std::hex << gen() << gen();
        return oss.str();
    }
    
    std::string GenerateSpanId() {
        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        
        std::ostringstream oss;
        oss << std::hex << gen();
        return oss.str();
    }
};

// HTTP client with distributed tracing
class TracingHTTPClient {
private:
    MicroserviceTracer& tracer_;
    
public:
    explicit TracingHTTPClient(MicroserviceTracer& tracer) : tracer_(tracer) {}
    
    std::future<std::string> MakeRequest(const std::string& url, const std::string& method = "GET") {
        auto [trace_id, span_id] = tracer_.CreateChildSpan("http_client");
        
        return std::async(std::launch::async, [=, this]() {
            TRACE_EVENT("http", "ClientRequest",
                "url", url,
                "method", method,
                "trace_id", trace_id,
                "span_id", span_id
            );
            
            // Add trace headers to HTTP request
            auto headers = tracer_.GetTraceContext();
            
            // Simulate HTTP request
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            TRACE_EVENT_NESTABLE_ASYNC_END("distributed", "DownstreamCall",
                TRACE_ID_GLOBAL(std::hash<std::string>{}(trace_id)),
                "status", "success",
                "response_size", 1024
            );
            
            return "Response data";
        });
    }
};
```

#### Real-time Analytics Integration

```cpp
#include <perfetto.h>
#include <vector>
#include <algorithm>
#include <numeric>

class RealTimeAnalytics {
private:
    struct MetricWindow {
        std::vector<double> values;
        std::chrono::system_clock::time_point window_start;
        std::chrono::milliseconds window_duration;
    };
    
    std::unordered_map<std::string, MetricWindow> metric_windows_;
    std::mutex metrics_mutex_;
    std::thread analytics_thread_;
    std::atomic<bool> running_{false};
    
public:
    void StartAnalytics() {
        running_ = true;
        analytics_thread_ = std::thread([this]() {
            AnalyticsLoop();
        });
        
        TRACE_EVENT("analytics", "StartRealTimeAnalytics");
    }
    
    void StopAnalytics() {
        running_ = false;
        if (analytics_thread_.joinable()) {
            analytics_thread_.join();
        }
        
        TRACE_EVENT("analytics", "StopRealTimeAnalytics");
    }
    
    void RecordMetric(const std::string& name, double value) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto& window = metric_windows_[name];
        
        // Initialize window if needed
        if (window.values.empty()) {
            window.window_start = now;
            window.window_duration = std::chrono::minutes(1);
        }
        
        // Check if we need to rotate window
        if (now - window.window_start > window.window_duration) {
            ProcessWindow(name, window);
            window.values.clear();
            window.window_start = now;
        }
        
        window.values.push_back(value);
        
        // Also trace individual metric
        TRACE_COUNTER("metrics", name.c_str(), value);
    }
    
private:
    void AnalyticsLoop() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            
            TRACE_EVENT("analytics", "ProcessAnalytics");
            ProcessCurrentMetrics();
        }
    }
    
    void ProcessCurrentMetrics() {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        for (auto& [name, window] : metric_windows_) {
            if (!window.values.empty()) {
                auto stats = CalculateStatistics(window.values);
                
                TRACE_EVENT("analytics", "MetricStatistics",
                    "metric_name", name,
                    "count", window.values.size(),
                    "mean", stats.mean,
                    "median", stats.median,
                    "std_dev", stats.std_dev,
                    "min", stats.min,
                    "max", stats.max,
                    "p95", stats.p95,
                    "p99", stats.p99
                );
                
                // Detect anomalies
                DetectAnomalies(name, stats);
            }
        }
    }
    
    void ProcessWindow(const std::string& name, const MetricWindow& window) {
        if (window.values.empty()) return;
        
        auto stats = CalculateStatistics(window.values);
        
        TRACE_EVENT("analytics", "WindowProcessed",
            "metric_name", name,
            "window_duration_ms", window.window_duration.count(),
            "sample_count", window.values.size(),
            "mean", stats.mean,
            "p95", stats.p95,
            "p99", stats.p99
        );
    }
    
    struct Statistics {
        double mean = 0.0;
        double median = 0.0;
        double std_dev = 0.0;
        double min = 0.0;
        double max = 0.0;
        double p95 = 0.0;
        double p99 = 0.0;
    };
    
    Statistics CalculateStatistics(std::vector<double> values) {
        if (values.empty()) return {};
        
        std::sort(values.begin(), values.end());
        
        Statistics stats;
        stats.min = values.front();
        stats.max = values.back();
        
        // Mean
        stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        
        // Median
        size_t mid = values.size() / 2;
        if (values.size() % 2 == 0) {
            stats.median = (values[mid - 1] + values[mid]) / 2.0;
        } else {
            stats.median = values[mid];
        }
        
        // Standard deviation
        double sq_sum = 0.0;
        for (double val : values) {
            sq_sum += (val - stats.mean) * (val - stats.mean);
        }
        stats.std_dev = std::sqrt(sq_sum / values.size());
        
        // Percentiles
        stats.p95 = values[static_cast<size_t>(values.size() * 0.95)];
        stats.p99 = values[static_cast<size_t>(values.size() * 0.99)];
        
        return stats;
    }
    
    void DetectAnomalies(const std::string& metric_name, const Statistics& stats) {
        // Simple anomaly detection: values more than 3 standard deviations from mean
        double threshold = 3.0 * stats.std_dev;
        
        if (stats.max - stats.mean > threshold) {
            TRACE_EVENT("analytics", "AnomalyDetected",
                "metric_name", metric_name,
                "anomaly_type", "high_value",
                "value", stats.max,
                "mean", stats.mean,
                "threshold", threshold
            );
        }
        
        if (stats.mean - stats.min > threshold) {
            TRACE_EVENT("analytics", "AnomalyDetected",
                "metric_name", metric_name,
                "anomaly_type", "low_value",
                "value", stats.min,
                "mean", stats.mean,
                "threshold", threshold
            );
        }
        
        // High variance detection
        if (stats.std_dev > stats.mean * 0.5) { // Coefficient of variation > 0.5
            TRACE_EVENT("analytics", "AnomalyDetected",
                "metric_name", metric_name,
                "anomaly_type", "high_variance",
                "std_dev", stats.std_dev,
                "mean", stats.mean,
                "coefficient_of_variation", stats.std_dev / stats.mean
            );
        }
    }
};
```

### Complete Integration Example

```cpp
// Complete example bringing everything together
#include <perfetto.h>
#include <memory>

class ProductionReadyApplication {
private:
    std::unique_ptr<DynamicTracingConfig> config_;
    std::unique_ptr<DistributedTraceCollector> collector_;
    std::unique_ptr<MicroserviceTracer> tracer_;
    std::unique_ptr<RealTimeAnalytics> analytics_;
    std::unique_ptr<PlatformTracer> platform_tracer_;
    
public:
    bool Initialize(const std::string& service_name, const std::string& version) {
        TRACE_EVENT("app", "Initialize", "service", service_name, "version", version);
        
        try {
            // Initialize configuration
            config_ = std::make_unique<DynamicTracingConfig>("./config/tracing.json");
            
            // Initialize trace collection
            collector_ = std::make_unique<DistributedTraceCollector>();
            if (!collector_->StartCollection()) {
                return false;
            }
            
            // Initialize microservice tracer
            tracer_ = std::make_unique<MicroserviceTracer>(service_name, version);
            
            // Initialize analytics
            analytics_ = std::make_unique<RealTimeAnalytics>();
            analytics_->StartAnalytics();
            
            // Initialize platform-specific tracing
            platform_tracer_ = std::make_unique<PlatformTracer>();
            
            TRACE_EVENT("app", "InitializationComplete");
            return true;
            
        } catch (const std::exception& e) {
            TRACE_EVENT("app", "InitializationError", "error", e.what());
            return false;
        }
    }
    
    void ProcessRequest(const std::string& request_id, const std::string& operation) {
        // Start distributed trace
        auto trace_id = tracer_->StartTrace(operation);
        
        TRACE_FUNCTION_PERF();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Add request context
            tracer_->AddBaggage("request_id", request_id);
            tracer_->AddBaggage("user_agent", "example-client/1.0");
            
            // Simulate request processing
            ProcessBusinessLogic(request_id);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time
            ).count();
            
            // Record metrics
            analytics_->RecordMetric("request_duration_ms", duration_ms);
            analytics_->RecordMetric("requests_processed", 1);
            
            tracer_->EndSpan("success");
            
            TRACE_EVENT("app", "RequestCompleted",
                "request_id", request_id,
                "trace_id", trace_id,
                "duration_ms", duration_ms
            );
            
        } catch (const std::exception& e) {
            analytics_->RecordMetric("request_errors", 1);
            tracer_->EndSpan("error");
            
            TRACE_EVENT("app", "RequestError",
                "request_id", request_id,
                "trace_id", trace_id,
                "error", e.what()
            );
        }
    }
    
    void Shutdown() {
        TRACE_EVENT("app", "Shutdown");
        
        if (analytics_) {
            analytics_->StopAnalytics();
        }
        
        if (collector_) {
            collector_->StopCollection();
        }
        
        // Final trace flush
        collector_->RotateTraceFile();
    }
    
private:
    void ProcessBusinessLogic(const std::string& request_id) {
        TRACE_FUNCTION_PERF();
        
        // Simulate various operations
        {
            TRACE_SCOPE("business", "ValidateInput");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        {
            TRACE_SCOPE("business", "DatabaseQuery");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            analytics_->RecordMetric("db_query_duration_ms", 50);
        }
        
        {
            TRACE_SCOPE("business", "ProcessData");
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        
        // Collect platform metrics periodically
        platform_tracer_->CollectMetrics();
    }
};

// Usage example
int main() {
    // Initialize Perfetto SDK
    perfetto::TrackEvent::Register();
    
    ProductionReadyApplication app;
    
    if (!app.Initialize("example-service", "1.0.0")) {
        std::cerr << "Failed to initialize application" << std::endl;
        return 1;
    }
    
    // Simulate request processing
    for (int i = 0; i < 100; ++i) {
        std::string request_id = "req_" + std::to_string(i);
        app.ProcessRequest(request_id, "process_user_data");
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    app.Shutdown();
    return 0;
}
```

### Study Materials and Exercises

#### Practical Exercises

**Exercise 1: Basic SDK Integration**
```cpp
// TODO: Complete this basic Perfetto integration
#include <perfetto.h>

class MyApplication {
public:
    void Initialize() {
        // Your code here: Initialize Perfetto SDK
    }
    
    void ProcessData() {
        // Your code here: Add trace events for this function
    }
};
```

**Exercise 2: Custom Data Source**
```cpp
// TODO: Implement a custom data source for application metrics
class ApplicationMetricsDataSource : public perfetto::DataSource<ApplicationMetricsDataSource> {
public:
    // Implement required methods
};
```

**Exercise 3: Production Configuration**
```cpp
// TODO: Create a production-ready tracing configuration with:
// - Dynamic sampling rates
// - Category filtering
// - Performance optimization
// - Error handling
```

#### Advanced Projects

1. **Distributed Tracing System**: Build a complete microservice tracing system
2. **Performance Monitoring Dashboard**: Create a real-time performance dashboard
3. **Custom Trace Analysis Tool**: Develop tools for analyzing custom trace data
4. **Cross-Platform Metrics Collector**: Implement platform-specific performance collection

#### Resources

- **Documentation**: [Perfetto Tracing SDK Documentation](https://perfetto.dev/docs/instrumentation/tracing-sdk)
- **Examples**: [Perfetto SDK Examples](https://github.com/google/perfetto/tree/master/examples)
- **Performance**: [Perfetto Performance Best Practices](https://perfetto.dev/docs/concepts/clock-sync)
- **Integration**: [Platform Integration Guides](https://perfetto.dev/docs/instrumentation/platform-integration)

---

## Summary

This comprehensive guide covers custom tracing and Perfetto SDK integration from basic concepts to production deployment. Key takeaways:

1. **Start Simple**: Begin with basic trace events and gradually add complexity
2. **Optimize Early**: Consider performance impact from the beginning
3. **Plan for Production**: Implement sampling, rate limiting, and configuration management
4. **Cross-Platform Awareness**: Design for multiple operating systems
5. **Monitor and Analyze**: Build analytics and alerting into your tracing system

The examples provided offer a solid foundation for implementing production-ready tracing solutions in modern applications.
