# Performance Optimization with Perfetto

*Duration: 2 weeks*

## Overview

Perfetto is a powerful, open-source system tracing and performance analysis framework developed by Google. It provides comprehensive tools for identifying and resolving performance bottlenecks across CPU, GPU, memory, and I/O operations. This module will teach you how to effectively use Perfetto for performance optimization in production systems.

### What is Perfetto?

Perfetto is a production-grade system performance tracing platform that offers:
- **System-wide tracing**: Captures events from kernel, userspace, and hardware
- **Low overhead**: Designed for production use with minimal performance impact
- **Rich visualization**: Web-based UI for trace analysis and visualization
- **Powerful query engine**: SQL-based analysis of trace data
- **Cross-platform support**: Works on Android, Linux, ChromeOS, and Windows
- **Real-time analysis**: Both recording and live analysis capabilities

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Perfetto Ecosystem                       │
├─────────────────────────────────────────────────────────────┤
│  Perfetto UI (Web Interface)                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Trace Viewer  │  │  Query Interface │  │   Metrics   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Trace Processor (Analysis Engine)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   SQL Engine    │  │   Data Parser   │  │  Metrics    │ │
│  │                 │  │                 │  │  Engine     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Data Collection (Traced)                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Kernel Tracing  │  │ User App Tracing│  │ System Info │ │
│  │ (ftrace, etc.)  │  │ (Custom events) │  │ Collection  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Setting Up Perfetto

### Installation and Setup

**1. Download Perfetto Tools**
```bash
# Download the latest release
curl -LO https://get.perfetto.dev/perfetto
chmod +x perfetto

# Or build from source
git clone https://android.googlesource.com/platform/external/perfetto/
cd perfetto
tools/install-build-deps
tools/gn gen out/linux
tools/ninja -C out/linux
```

**2. Basic Configuration**
```bash
# Create a basic trace config
cat > trace_config.pbtx << EOF
buffers: {
    size_kb: 65536
    fill_policy: DISCARD
}

data_sources: {
    config {
        name: "linux.ftrace"
        ftrace_config {
            ftrace_events: "sched/sched_switch"
            ftrace_events: "sched/sched_wakeup"
            ftrace_events: "power/cpu_frequency"
            ftrace_events: "power/cpu_idle"
        }
    }
}

duration_ms: 10000
EOF
```

**3. Verify Installation**
```bash
# Test basic functionality
./perfetto --help
./perfetto --query-raw "SELECT name FROM sqlite_master WHERE type='table';"
```

### Web UI Setup

**Local Installation:**
```bash
# Download and serve Perfetto UI locally
curl -LO https://get.perfetto.dev/perfetto_ui.html
python3 -m http.server 8000
# Open http://localhost:8000/perfetto_ui.html
```

**Online Version:**
```
https://ui.perfetto.dev
```

## Core Concepts and Data Models

### Understanding Perfetto's Data Model

Perfetto organizes trace data into several key tables:

#### 1. **Process and Thread Tables**
```sql
-- View all processes in the trace
SELECT pid, name, start_ts, end_ts 
FROM process 
ORDER BY start_ts;

-- View threads for a specific process
SELECT tid, name, start_ts, end_ts, upid
FROM thread 
WHERE upid = (SELECT upid FROM process WHERE name = 'my_app');
```

#### 2. **Scheduling Events**
```sql
-- CPU scheduling information
SELECT ts, dur, cpu, utid, end_state, priority
FROM sched 
WHERE utid = 123
ORDER BY ts;
```

#### 3. **Slice Tables (Trace Events)**
```sql
-- Application trace events
SELECT ts, dur, name, category, thread_name
FROM slice s
JOIN thread_track t ON s.track_id = t.id
JOIN thread th ON t.utid = th.utid
WHERE th.name = 'MainThread'
ORDER BY ts;
```

### Trace Event Types

**1. Duration Events (Begin/End)**
```cpp
// C++ example using Perfetto SDK
#include "perfetto.h"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("myapp").SetDescription("My application events"));

void ExpensiveFunction() {
    TRACE_EVENT("myapp", "ExpensiveFunction");
    
    // Your expensive code here
    ProcessData();
    
    TRACE_EVENT_BEGIN("myapp", "DatabaseQuery");
    QueryDatabase();
    TRACE_EVENT_END("myapp");
}
```

**2. Instant Events**
```cpp
void LogImportantEvent() {
    TRACE_EVENT_INSTANT("myapp", "ImportantEvent", 
                        "user_id", user_id, 
                        "action", "button_click");
}
```

**3. Counter Events**
```cpp
void UpdateMemoryUsage(size_t bytes) {
    TRACE_COUNTER("myapp", "MemoryUsage", bytes);
}
```

## CPU Performance Analysis

### Analyzing CPU Usage Patterns

**1. Overall CPU Utilization**
```sql
-- CPU utilization over time
SELECT 
    ts,
    cpu,
    CASE 
        WHEN utid != 0 THEN 'busy'
        ELSE 'idle'
    END as state,
    dur
FROM sched
ORDER BY ts;
```

**2. Per-Process CPU Usage**
```sql
-- CPU time per process
SELECT 
    p.name as process_name,
    SUM(s.dur) / 1e9 as cpu_time_seconds,
    COUNT(*) as schedule_count
FROM sched s
JOIN thread t ON s.utid = t.utid
JOIN process p ON t.upid = p.upid
WHERE s.utid != 0  -- Exclude idle
GROUP BY p.name
ORDER BY cpu_time_seconds DESC;
```

**3. CPU Frequency Analysis**
```sql
-- CPU frequency changes
SELECT 
    ts,
    CAST(value as INT) as frequency_khz,
    cpu
FROM counter c
JOIN cpu_counter_track cct ON c.track_id = cct.id
WHERE cct.name = 'cpufreq'
ORDER BY ts;
```

### Practical CPU Optimization Example

**Problem**: High CPU usage in image processing application

**1. Initial Investigation**
```sql
-- Find CPU-intensive functions
SELECT 
    name,
    SUM(dur) / 1e6 as total_ms,
    COUNT(*) as call_count,
    AVG(dur) / 1e6 as avg_ms
FROM slice
WHERE category = 'imageprocessing'
GROUP BY name
ORDER BY total_ms DESC
LIMIT 10;
```

**2. Detailed Analysis**
```sql
-- Analyze specific function performance
SELECT 
    ts / 1e9 as timestamp_seconds,
    dur / 1e6 as duration_ms,
    name,
    EXTRACT_ARG(arg_set_id, 'image_size') as image_size
FROM slice
WHERE name = 'ProcessImage'
    AND dur > 50 * 1e6  -- Focus on slow calls (>50ms)
ORDER BY dur DESC;
```

**3. Optimization Code**
```cpp
// Before optimization
void ProcessImage(const Image& img) {
    TRACE_EVENT("imageprocessing", "ProcessImage", 
                "image_size", img.width * img.height);
    
    // Single-threaded processing
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            ProcessPixel(img, x, y);
        }
    }
}

// After optimization
void ProcessImageOptimized(const Image& img) {
    TRACE_EVENT("imageprocessing", "ProcessImageOptimized",
                "image_size", img.width * img.height);
    
    const int num_threads = std::thread::hardware_concurrency();
    const int rows_per_thread = img.height / num_threads;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&img, t, rows_per_thread, num_threads]() {
            TRACE_EVENT("imageprocessing", "ProcessImageThread", 
                       "thread_id", t);
            
            int start_row = t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? 
                         img.height : (t + 1) * rows_per_thread;
            
            for (int y = start_row; y < end_row; ++y) {
                for (int x = 0; x < img.width; ++x) {
                    ProcessPixel(img, x, y);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}
```

## Memory Performance Analysis

### Memory Usage Tracking

**1. Heap Memory Analysis**
```sql
-- Memory allocations over time
SELECT 
    ts / 1e9 as timestamp_seconds,
    SUM(size) OVER (ORDER BY ts) / 1024 / 1024 as heap_mb
FROM heap_graph_object
WHERE type_name = 'malloc'
ORDER BY ts;
```

**2. Memory Leaks Detection**
```sql
-- Find potential memory leaks
SELECT 
    type_name,
    COUNT(*) as object_count,
    SUM(self_size) / 1024 / 1024 as total_mb
FROM heap_graph_object
WHERE reachable = 0  -- Unreachable objects
GROUP BY type_name
ORDER BY total_mb DESC;
```

**3. Peak Memory Usage**
```sql
-- Find peak memory usage periods
WITH memory_timeline AS (
    SELECT 
        ts,
        value / 1024 / 1024 as memory_mb
    FROM counter c
    JOIN process_counter_track pct ON c.track_id = pct.id
    WHERE pct.name = 'mem.rss'
)
SELECT 
    ts / 1e9 as timestamp_seconds,
    memory_mb,
    LAG(memory_mb) OVER (ORDER BY ts) as prev_memory_mb,
    memory_mb - LAG(memory_mb) OVER (ORDER BY ts) as delta_mb
FROM memory_timeline
ORDER BY memory_mb DESC
LIMIT 20;
```

### Memory Optimization Example

**Problem**: Memory leak in web server application

**1. Investigation Code**
```cpp
// Add memory tracking
class ConnectionHandler {
private:
    std::unique_ptr<char[]> buffer_;
    
public:
    ConnectionHandler(size_t buffer_size) {
        TRACE_EVENT("webserver", "ConnectionHandler::Constructor",
                   "buffer_size", buffer_size);
        buffer_ = std::make_unique<char[]>(buffer_size);
        
        // Track memory allocation
        TRACE_COUNTER("webserver", "ActiveConnections", ++active_connections_);
        TRACE_COUNTER("webserver", "TotalMemory", 
                     active_connections_ * buffer_size);
    }
    
    ~ConnectionHandler() {
        TRACE_EVENT("webserver", "ConnectionHandler::Destructor");
        TRACE_COUNTER("webserver", "ActiveConnections", --active_connections_);
    }
};
```

**2. Analysis Query**
```sql
-- Track connection lifecycle
SELECT 
    ts / 1e9 as timestamp_seconds,
    value as active_connections
FROM counter c
JOIN track t ON c.track_id = t.id
WHERE t.name = 'ActiveConnections'
ORDER BY ts;
```

## GPU Performance Analysis

### GPU Tracing Setup

**1. Enable GPU Tracing**
```bash
# Android GPU tracing
adb shell setprop debug.egl.trace perfetto

# Linux with Mesa
export MESA_TRACE=perfetto
```

**2. GPU Metrics Queries**
```sql
-- GPU utilization
SELECT 
    ts / 1e9 as timestamp_seconds,
    value as gpu_utilization_percent
FROM counter c
JOIN gpu_counter_track gct ON c.track_id = gct.id
WHERE gct.name = 'GPU Utilization'
ORDER BY ts;

-- GPU memory usage
SELECT 
    ts / 1e9 as timestamp_seconds,
    value / 1024 / 1024 as gpu_memory_mb
FROM counter c
JOIN gpu_counter_track gct ON c.track_id = gct.id
WHERE gct.name = 'GPU Memory Usage'
ORDER BY ts;
```

### Render Performance Analysis

**1. Frame Timing Analysis**
```sql
-- Frame render times
SELECT 
    ts / 1e9 as timestamp_seconds,
    dur / 1e6 as frame_time_ms,
    CASE 
        WHEN dur > 16.67 * 1e6 THEN 'dropped'
        ELSE 'ok'
    END as frame_status
FROM slice
WHERE name = 'RenderFrame'
ORDER BY ts;
```

**2. GPU Pipeline Optimization**
```cpp
// Optimize GPU pipeline
class Renderer {
public:
    void RenderFrame() {
        TRACE_EVENT("rendering", "RenderFrame");
        
        {
            TRACE_EVENT("rendering", "UpdateUniforms");
            UpdateShaderUniforms();
        }
        
        {
            TRACE_EVENT("rendering", "DrawCalls");
            for (const auto& object : render_objects_) {
                TRACE_EVENT("rendering", "DrawObject", 
                           "vertices", object.vertex_count);
                DrawObject(object);
            }
        }
        
        {
            TRACE_EVENT("rendering", "SwapBuffers");
            SwapBuffers();
        }
    }
};
```

## I/O Performance Analysis

### File System I/O Tracking

**1. I/O Operations Analysis**
```sql
-- File I/O operations
SELECT 
    ts / 1e9 as timestamp_seconds,
    dur / 1e6 as duration_ms,
    EXTRACT_ARG(arg_set_id, 'fd') as file_descriptor,
    EXTRACT_ARG(arg_set_id, 'bytes') as bytes,
    name as operation
FROM slice
WHERE category = 'file_io'
ORDER BY dur DESC;
```

**2. I/O Bottleneck Detection**
```sql
-- Find slow I/O operations
WITH slow_io AS (
    SELECT 
        ts,
        dur,
        EXTRACT_ARG(arg_set_id, 'filename') as filename,
        name as operation
    FROM slice
    WHERE category = 'file_io' 
        AND dur > 10 * 1e6  -- > 10ms
)
SELECT 
    filename,
    operation,
    COUNT(*) as slow_count,
    AVG(dur) / 1e6 as avg_duration_ms,
    MAX(dur) / 1e6 as max_duration_ms
FROM slow_io
GROUP BY filename, operation
ORDER BY avg_duration_ms DESC;
```

### Network I/O Optimization

**Example: Optimizing HTTP client**
```cpp
class HttpClient {
public:
    std::future<Response> AsyncGet(const std::string& url) {
        TRACE_EVENT("network", "HttpClient::AsyncGet", "url", url);
        
        return std::async(std::launch::async, [this, url]() {
            TRACE_EVENT("network", "HttpRequest");
            
            auto start_time = std::chrono::steady_clock::now();
            
            {
                TRACE_EVENT("network", "DNS_Lookup");
                ResolveHostname(url);
            }
            
            {
                TRACE_EVENT("network", "TCP_Connect");
                EstablishConnection();
            }
            
            {
                TRACE_EVENT("network", "HTTP_Transaction");
                auto response = SendRequest(url);
                
                auto duration = std::chrono::steady_clock::now() - start_time;
                TRACE_COUNTER("network", "RequestDuration", 
                             std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
                
                return response;
            }
        });
    }
};
```

## Advanced Analysis Techniques

### Custom Metrics and Aggregations

**1. Creating Custom Performance Metrics**
```sql
-- Create a custom metric for app responsiveness
CREATE VIEW app_responsiveness AS
SELECT 
    ts,
    dur,
    name,
    CASE 
        WHEN dur > 100 * 1e6 THEN 'poor'     -- > 100ms
        WHEN dur > 50 * 1e6 THEN 'fair'      -- 50-100ms
        WHEN dur > 16 * 1e6 THEN 'good'      -- 16-50ms
        ELSE 'excellent'                      -- < 16ms
    END as responsiveness_rating
FROM slice
WHERE category = 'ui' AND name LIKE '%Event%';

-- Use the custom metric
SELECT 
    responsiveness_rating,
    COUNT(*) as event_count,
    AVG(dur) / 1e6 as avg_duration_ms
FROM app_responsiveness
GROUP BY responsiveness_rating;
```

**2. Time-Series Analysis**
```sql
-- Analyze performance trends over time
WITH time_buckets AS (
    SELECT 
        (ts - (SELECT MIN(ts) FROM slice)) / 1e9 / 60 as minute_bucket,
        dur
    FROM slice
    WHERE name = 'MainLoop'
),
bucketed_stats AS (
    SELECT 
        CAST(minute_bucket as INT) as minute,
        COUNT(*) as iteration_count,
        AVG(dur) / 1e6 as avg_duration_ms,
        MAX(dur) / 1e6 as max_duration_ms,
        MIN(dur) / 1e6 as min_duration_ms
    FROM time_buckets
    GROUP BY CAST(minute_bucket as INT)
)
SELECT 
    minute,
    iteration_count,
    avg_duration_ms,
    max_duration_ms,
    min_duration_ms,
    -- Calculate performance degradation
    avg_duration_ms - FIRST_VALUE(avg_duration_ms) OVER (ORDER BY minute) as degradation_ms
FROM bucketed_stats
ORDER BY minute;
```

**3. Correlation Analysis**
```sql
-- Correlate CPU usage with memory pressure
WITH cpu_memory_data AS (
    SELECT 
        ts,
        (SELECT value FROM counter WHERE track_id = cpu_track.id AND counter.ts <= slice.ts ORDER BY counter.ts DESC LIMIT 1) as cpu_freq,
        (SELECT value FROM counter WHERE track_id = mem_track.id AND counter.ts <= slice.ts ORDER BY counter.ts DESC LIMIT 1) as memory_usage
    FROM slice
    JOIN cpu_counter_track cpu_track ON cpu_track.name = 'cpufreq'
    JOIN process_counter_track mem_track ON mem_track.name = 'mem.rss'
    WHERE slice.name = 'ProcessingLoop'
)
SELECT 
    CASE 
        WHEN cpu_freq > 2000000 THEN 'high_cpu'
        WHEN cpu_freq > 1000000 THEN 'medium_cpu'
        ELSE 'low_cpu'
    END as cpu_category,
    CASE 
        WHEN memory_usage > 1000000000 THEN 'high_mem'
        WHEN memory_usage > 500000000 THEN 'medium_mem'
        ELSE 'low_mem'
    END as memory_category,
    COUNT(*) as sample_count
FROM cpu_memory_data
GROUP BY cpu_category, memory_category;
```

### Performance Regression Detection

**1. Automated Performance Regression Detection**
```python
import sqlite3
import numpy as np
from scipy import stats

def detect_performance_regression(trace_file_before, trace_file_after, function_name):
    """
    Compare performance between two traces to detect regressions
    """
    def get_function_durations(trace_file, func_name):
        conn = sqlite3.connect(trace_file)
        cursor = conn.cursor()
        
        query = """
        SELECT dur / 1e6 as duration_ms 
        FROM slice 
        WHERE name = ?
        """
        
        cursor.execute(query, (func_name,))
        durations = [row[0] for row in cursor.fetchall()]
        conn.close()
        return durations
    
    before_durations = get_function_durations(trace_file_before, function_name)
    after_durations = get_function_durations(trace_file_after, function_name)
    
    # Statistical analysis
    before_mean = np.mean(before_durations)
    after_mean = np.mean(after_durations)
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(before_durations, after_durations)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(before_durations) - 1) * np.var(before_durations) + 
                         (len(after_durations) - 1) * np.var(after_durations)) / 
                        (len(before_durations) + len(after_durations) - 2))
    cohens_d = (after_mean - before_mean) / pooled_std
    
    # Determine if regression is significant
    regression_threshold = 0.1  # 10% increase
    is_regression = (after_mean > before_mean * (1 + regression_threshold) and 
                    p_value < 0.05 and cohens_d > 0.2)
    
    return {
        'function': function_name,
        'before_mean_ms': before_mean,
        'after_mean_ms': after_mean,
        'percent_change': ((after_mean - before_mean) / before_mean) * 100,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'is_significant_regression': is_regression
    }

# Usage example
result = detect_performance_regression(
    'baseline_trace.perfetto-trace',
    'current_trace.perfetto-trace',
    'CriticalFunction'
)
print(f"Performance change: {result['percent_change']:.2f}%")
print(f"Significant regression: {result['is_significant_regression']}")
```

## Real-World Case Studies

### Case Study 1: Android App Performance Optimization

**Problem**: Android app experiencing janky UI and slow startup times

**Investigation Process**:

1. **Capture System Trace**
```bash
# Capture comprehensive Android trace
adb shell perfetto \
  -c - --txt \
  -o /data/misc/perfetto-traces/trace.perfetto-trace << EOF

buffers: {
    size_kb: 65536
    fill_policy: DISCARD
}

data_sources: {
    config {
        name: "android.surfaceflinger"
    }
}

data_sources: {
    config {
        name: "android.graphics.frame"
    }
}

data_sources: {
    config {
        name: "linux.process_stats"
        process_stats_config {
            scan_all_processes_on_start: true
        }
    }
}

duration_ms: 30000
EOF
```

2. **Analyze UI Performance**
```sql
-- Find dropped frames
SELECT 
    ts / 1e9 as timestamp_seconds,
    dur / 1e6 as frame_duration_ms,
    present_type,
    on_time_finish,
    gpu_composition
FROM expected_frame_timeline_slice
WHERE dur > 16.67 * 1e6  -- Frames longer than 16.67ms (60 FPS)
ORDER BY dur DESC
LIMIT 20;

-- Analyze main thread blocking
SELECT 
    ts / 1e9 as timestamp_seconds,
    dur / 1e6 as duration_ms,
    name,
    EXTRACT_ARG(arg_set_id, 'method_name') as method
FROM slice s
JOIN thread_track tt ON s.track_id = tt.id
JOIN thread t ON tt.utid = t.utid
WHERE t.name = 'main' 
    AND dur > 5 * 1e6  -- Blocks longer than 5ms
ORDER BY dur DESC;
```

3. **Optimization Solutions**
```kotlin
// Before: Blocking main thread
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // DON'T: Heavy operation on main thread
        val data = loadLargeDataSet()  // Blocks UI for 500ms
        updateUI(data)
    }
}

// After: Async loading with progress indication
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Show loading indicator
        showLoadingIndicator()
        
        // Load data asynchronously
        lifecycleScope.launch {
            val data = withContext(Dispatchers.IO) {
                Trace.beginSection("LoadLargeDataSet")
                try {
                    loadLargeDataSet()
                } finally {
                    Trace.endSection()
                }
            }
            
            // Update UI on main thread
            Trace.beginSection("UpdateUI")
            updateUI(data)
            hideLoadingIndicator()
            Trace.endSection()
        }
    }
}
```

### Case Study 2: Web Application Memory Leak

**Problem**: Node.js web application with increasing memory usage over time

**Investigation**:

1. **Memory Tracking Setup**
```javascript
// Add memory tracking to Node.js app
const v8 = require('v8');
const { performance, PerformanceObserver } = require('perf_hooks');

class MemoryTracker {
    constructor() {
        this.startTracking();
    }
    
    startTracking() {
        // Track garbage collection
        const obs = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                console.log(`GC ${entry.kind}: ${entry.duration}ms`);
                this.logMemoryUsage(`After GC ${entry.kind}`);
            });
        });
        obs.observe({ entryTypes: ['gc'] });
        
        // Periodic memory logging
        setInterval(() => {
            this.logMemoryUsage('Periodic check');
        }, 60000); // Every minute
    }
    
    logMemoryUsage(context) {
        const memUsage = process.memoryUsage();
        const heapStats = v8.getHeapStatistics();
        
        console.log(`[${context}] Memory Usage:`, {
            rss: `${Math.round(memUsage.rss / 1024 / 1024)}MB`,
            heapUsed: `${Math.round(memUsage.heapUsed / 1024 / 1024)}MB`,
            heapTotal: `${Math.round(memUsage.heapTotal / 1024 / 1024)}MB`,
            external: `${Math.round(memUsage.external / 1024 / 1024)}MB`,
            heapLimit: `${Math.round(heapStats.heap_size_limit / 1024 / 1024)}MB`
        });
    }
}

const memTracker = new MemoryTracker();
```

2. **Leak Detection Analysis**
```sql
-- Find objects that are growing over time
WITH memory_snapshots AS (
    SELECT 
        ts,
        type_name,
        COUNT(*) as object_count,
        SUM(self_size) as total_size
    FROM heap_graph_object
    GROUP BY ts, type_name
),
growth_analysis AS (
    SELECT 
        type_name,
        ts,
        object_count,
        total_size,
        object_count - LAG(object_count) OVER (PARTITION BY type_name ORDER BY ts) as count_delta,
        total_size - LAG(total_size) OVER (PARTITION BY type_name ORDER BY ts) as size_delta
    FROM memory_snapshots
)
SELECT 
    type_name,
    SUM(count_delta) as total_count_growth,
    SUM(size_delta) / 1024 / 1024 as total_size_growth_mb
FROM growth_analysis
WHERE count_delta > 0
GROUP BY type_name
ORDER BY total_size_growth_mb DESC;
```

3. **Fix Implementation**
```javascript
// Before: Memory leak with event listeners
class DataProcessor {
    constructor() {
        this.cache = new Map();
        
        // LEAK: Event listener not removed
        process.on('SIGUSR1', this.handleSignal.bind(this));
    }
    
    processData(data) {
        // LEAK: Cache grows indefinitely
        this.cache.set(data.id, data);
        return this.transformData(data);
    }
    
    handleSignal() {
        console.log('Signal received');
    }
}

// After: Proper cleanup and cache management
class DataProcessor {
    constructor() {
        this.cache = new LRU({ max: 1000 }); // Limited cache size
        this.signalHandler = this.handleSignal.bind(this);
        
        process.on('SIGUSR1', this.signalHandler);
    }
    
    processData(data) {
        this.cache.set(data.id, data);
        return this.transformData(data);
    }
    
    handleSignal() {
        console.log('Signal received');
    }
    
    cleanup() {
        // Proper cleanup
        process.removeListener('SIGUSR1', this.signalHandler);
        this.cache.clear();
    }
}
```

### Case Study 3: Database Query Optimization

**Problem**: High latency in database operations affecting application performance

**Investigation**:

1. **Database Tracing Setup**
```cpp
// Add database query tracing
class DatabaseConnection {
private:
    std::chrono::steady_clock::time_point query_start_;
    
public:
    ResultSet ExecuteQuery(const std::string& query) {
        TRACE_EVENT("database", "ExecuteQuery", "query", query);
        
        auto start = std::chrono::steady_clock::now();
        
        {
            TRACE_EVENT("database", "QueryParsing");
            PrepareStatement(query);
        }
        
        {
            TRACE_EVENT("database", "QueryExecution");
            auto result = ExecutePreparedStatement();
            
            auto duration = std::chrono::steady_clock::now() - start;
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            
            TRACE_COUNTER("database", "QueryDuration", duration_ms);
            
            // Log slow queries
            if (duration_ms > 100) {
                TRACE_EVENT_INSTANT("database", "SlowQuery", 
                                   "duration_ms", duration_ms,
                                   "query", query);
            }
            
            return result;
        }
    }
};
```

2. **Query Performance Analysis**
```sql
-- Analyze slow queries
SELECT 
    EXTRACT_ARG(arg_set_id, 'query') as sql_query,
    COUNT(*) as execution_count,
    AVG(dur) / 1e6 as avg_duration_ms,
    MAX(dur) / 1e6 as max_duration_ms,
    SUM(dur) / 1e6 as total_duration_ms
FROM slice
WHERE name = 'ExecuteQuery' 
    AND dur > 10 * 1e6  -- Queries longer than 10ms
GROUP BY sql_query
ORDER BY total_duration_ms DESC;

-- Find query patterns
SELECT 
    CASE 
        WHEN EXTRACT_ARG(arg_set_id, 'query') LIKE 'SELECT%' THEN 'SELECT'
        WHEN EXTRACT_ARG(arg_set_id, 'query') LIKE 'INSERT%' THEN 'INSERT'
        WHEN EXTRACT_ARG(arg_set_id, 'query') LIKE 'UPDATE%' THEN 'UPDATE'
        WHEN EXTRACT_ARG(arg_set_id, 'query') LIKE 'DELETE%' THEN 'DELETE'
        ELSE 'OTHER'
    END as query_type,
    COUNT(*) as count,
    AVG(dur) / 1e6 as avg_duration_ms
FROM slice
WHERE name = 'ExecuteQuery'
GROUP BY query_type
ORDER BY avg_duration_ms DESC;
```

## Best Practices and Performance Tips

### Tracing Best Practices

**1. Minimize Trace Overhead**
```cpp
// Use conditional tracing for hot paths
#ifdef PERFETTO_TRACE_ENABLED
#define TRACE_HOT_PATH(category, name) TRACE_EVENT(category, name)
#else
#define TRACE_HOT_PATH(category, name) do {} while(0)
#endif

void HotFunction() {
    TRACE_HOT_PATH("performance", "HotFunction");
    // Critical path code
}
```

**2. Strategic Trace Point Placement**
```cpp
class VideoEncoder {
public:
    void EncodeFrame(const Frame& frame) {
        TRACE_EVENT("video", "EncodeFrame", 
                   "frame_id", frame.id,
                   "width", frame.width,
                   "height", frame.height);
        
        // Trace major phases, not every small operation
        {
            TRACE_EVENT("video", "PreprocessFrame");
            PreprocessFrame(frame);
        }
        
        {
            TRACE_EVENT("video", "CompressFrame");
            CompressFrame(frame);
        }
        
        {
            TRACE_EVENT("video", "OutputFrame");
            OutputFrame(frame);
        }
    }
};
```

### Performance Optimization Workflow

**1. Performance Testing Pipeline**
```python
#!/usr/bin/env python3
"""
Automated performance testing with Perfetto
"""

import subprocess
import json
import sqlite3
from pathlib import Path

class PerformanceTestSuite:
    def __init__(self, app_binary, trace_config):
        self.app_binary = app_binary
        self.trace_config = trace_config
        self.baseline_results = None
    
    def run_performance_test(self, test_name, duration_ms=30000):
        """Run a performance test and return metrics"""
        trace_file = f"{test_name}.perfetto-trace"
        
        # Start tracing
        trace_process = subprocess.Popen([
            'perfetto', '-c', self.trace_config, 
            '-o', trace_file, '--background'
        ])
        
        # Run the application
        app_process = subprocess.run([
            self.app_binary, '--test-mode'
        ], timeout=duration_ms/1000)
        
        # Stop tracing
        trace_process.terminate()
        trace_process.wait()
        
        # Analyze results
        return self.analyze_trace(trace_file)
    
    def analyze_trace(self, trace_file):
        """Extract key performance metrics from trace"""
        conn = sqlite3.connect(trace_file)
        
        # CPU usage
        cpu_query = """
        SELECT AVG(CASE WHEN utid != 0 THEN 100.0 ELSE 0.0 END) as avg_cpu_usage
        FROM sched
        """
        
        # Memory usage
        memory_query = """
        SELECT MAX(value) / 1024 / 1024 as peak_memory_mb
        FROM counter c
        JOIN process_counter_track pct ON c.track_id = pct.id
        WHERE pct.name = 'mem.rss'
        """
        
        # Frame timing (if applicable)
        frame_query = """
        SELECT 
            AVG(dur) / 1e6 as avg_frame_time_ms,
            COUNT(CASE WHEN dur > 16.67 * 1e6 THEN 1 END) as dropped_frames
        FROM slice
        WHERE name = 'RenderFrame'
        """
        
        results = {
            'cpu_usage': conn.execute(cpu_query).fetchone()[0],
            'peak_memory_mb': conn.execute(memory_query).fetchone()[0],
        }
        
        frame_result = conn.execute(frame_query).fetchone();
        if frame_result[0]:  # If frame data exists
            results.update({
                'avg_frame_time_ms': frame_result[0],
                'dropped_frames': frame_result[1]
            })
        
        conn.close()
        return results
    
    def compare_with_baseline(self, current_results):
        """Compare current results with baseline"""
        if not self.baseline_results:
            print("No baseline set. Current results become baseline.")
            self.baseline_results = current_results
            return
        
        comparison = {}
        for metric, current_value in current_results.items():
            baseline_value = self.baseline_results.get(metric, 0)
            if baseline_value > 0:
                change_percent = ((current_value - baseline_value) / baseline_value) * 100
                comparison[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_percent': change_percent
                }
        
        return comparison

# Usage
if __name__ == "__main__":
    test_suite = PerformanceTestSuite('./myapp', 'trace_config.pbtx')
    
    # Run baseline test
    baseline = test_suite.run_performance_test('baseline')
    test_suite.baseline_results = baseline
    
    # Run current test
    current = test_suite.run_performance_test('current')
    comparison = test_suite.compare_with_baseline(current)
    
    print(json.dumps(comparison, indent=2))
```

## Learning Objectives

By the end of this section, you should be able to:

- **Set up and configure Perfetto** for comprehensive system tracing
- **Capture meaningful traces** for CPU, memory, GPU, and I/O analysis
- **Write effective SQL queries** to extract performance insights from trace data
- **Identify performance bottlenecks** using Perfetto's visualization and analysis tools
- **Implement targeted optimizations** based on Perfetto findings
- **Detect performance regressions** using automated analysis techniques
- **Integrate Perfetto** into continuous performance testing workflows
- **Apply advanced analysis techniques** for complex performance investigations

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Set up Perfetto tracing for your target platform (Android, Linux, Windows)  
□ Configure trace collection for specific performance scenarios  
□ Navigate the Perfetto UI effectively  
□ Write SQL queries to analyze trace data  
□ Identify common performance anti-patterns in traces  
□ Correlate trace events with application performance issues  
□ Implement effective tracing in your own applications  
□ Set up automated performance regression detection  
□ Optimize based on Perfetto findings and verify improvements  

### Practical Exercises

**Exercise 1: Basic Trace Analysis**
```sql
-- TODO: Write queries to find the top 10 longest-running functions
-- and identify which threads are consuming the most CPU time

-- Your query here:
```

**Exercise 2: Memory Leak Detection**
```sql
-- TODO: Create a query to identify objects that are growing in count
-- over time and might indicate a memory leak

-- Your query here:
```

**Exercise 3: Custom Application Tracing**
```cpp
// TODO: Add comprehensive tracing to this image processing pipeline
class ImageProcessor {
public:
    std::vector<ProcessedImage> ProcessBatch(const std::vector<Image>& images) {
        // Add tracing here
        std::vector<ProcessedImage> results;
        
        for (const auto& image : images) {
            // Add tracing for each processing step
            auto processed = ProcessSingleImage(image);
            results.push_back(processed);
        }
        
        return results;
    }
};
```

## Study Materials

### Official Documentation
- **Primary:** [Perfetto Documentation](https://perfetto.dev/docs/) - Complete reference
- **SQL Reference:** [Trace Processor SQL](https://perfetto.dev/docs/analysis/sql-tables) - Table schemas and query examples
- **API Reference:** [Perfetto SDK](https://perfetto.dev/docs/instrumentation/tracing-sdk) - Integration guide

### Advanced Resources
- **Research Papers:** "Perfetto: A Production-Grade Tracing System" - Google Research
- **Video Tutorials:** Google I/O Perfetto sessions and Android performance talks
- **Case Studies:** Android performance optimization examples using Perfetto

### Tools and Utilities
- **Perfetto UI:** https://ui.perfetto.dev - Web-based trace analysis
- **Command Line Tools:** perfetto, trace_processor_shell
- **Integration Libraries:** perfetto-sdk for C++, Python bindings

### Practice Datasets
- **Android Traces:** Example traces from AOSP performance tests
- **Chrome Traces:** Browser performance analysis examples
- **Custom Applications:** Build your own traced applications for practice

### Recommended Reading
- "Systems Performance" by Brendan Gregg - General performance analysis
- "The Art of Application Performance Testing" by Ian Molyneaux
- Android performance optimization guides and best practices

### Community Resources
- **Stack Overflow:** perfetto tag for Q&A
- **GitHub Issues:** Perfetto project for advanced discussions
- **Performance Communities:** Android Performance, Chrome Dev Tools communities

## Next Steps

After mastering Perfetto performance optimization:

1. **Advanced Profiling:** Explore Intel VTune, AMD CodeXL, or NVIDIA Nsight
2. **Continuous Performance:** Implement performance monitoring in CI/CD
3. **Platform-Specific Tools:** Deep dive into platform-specific profilers
4. **Custom Metrics:** Develop domain-specific performance metrics
5. **Performance Culture:** Build performance-conscious development practices

