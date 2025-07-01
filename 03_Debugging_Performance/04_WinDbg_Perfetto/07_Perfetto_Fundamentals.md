# Perfetto Fundamentals

*Duration: 1-2 weeks*

## Overview

**Perfetto** is a production-grade, open-source performance tracing platform developed by Google for Android and Linux systems. It provides comprehensive system-wide tracing capabilities that help developers and performance engineers understand application behavior, system bottlenecks, and resource utilization patterns.

### What Makes Perfetto Special?

- **Production-ready**: Used in Android devices worldwide
- **High performance**: Minimal overhead while tracing
- **Comprehensive**: Traces CPU, memory, I/O, GPU, and custom events
- **Scalable**: Handles large trace files (GBs) efficiently
- **Web-based UI**: Modern interface for trace analysis
- **Extensible**: Support for custom data sources

### When to Use Perfetto

✅ **Use Perfetto for:**
- Android app performance optimization
- Linux system performance analysis
- Understanding CPU scheduling behavior
- Memory allocation patterns analysis
- I/O bottleneck identification
- Frame rate and jank analysis
- Power consumption profiling
- Custom performance metrics

❌ **Consider alternatives for:**
- Windows-specific debugging (use WinDbg)
- Memory leak detection (use AddressSanitizer)
- Code coverage analysis (use gcov/llvm-cov)
- Simple function profiling (use gprof)

## Perfetto Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Perfetto Ecosystem                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Perfetto  │  │   Perfetto  │  │    Third-party      │ │
│  │     UI      │  │    SDK      │  │      Tools          │ │
│  │ (Web-based) │  │ (C++/Java)  │  │  (Chrome DevTools)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      Trace Format                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Protobuf-based Trace                      │ │
│  │         (.pb files, efficient storage)                 │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Perfetto Service                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  ┌────────────┐ ┌─────────────┐ ┌─────────────────────┐│ │
│  │  │   Traced   │ │   Probes    │ │      Consumers      ││ │
│  │  │ (Service)  │ │(Data Source)│ │   (Client Apps)     ││ │
│  │  └────────────┘ └─────────────┘ └─────────────────────┘│ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Data Sources                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ ftrace │ atrace │ heapprofd │ CPU │ GPU │ Memory │Custom│ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                  Operating System                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         Linux Kernel / Android Runtime                 │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Traced (The Service)
The central daemon that coordinates tracing activities:

```cpp
// Example: Starting the traced service
// On Linux:
sudo traced &
traced_probes &

// The service handles:
// - Session management
// - Buffer allocation
// - Data source coordination
// - Access control
```

#### 2. Data Sources (Probes)
Components that collect specific types of data:

**System Data Sources:**
- **ftrace**: Linux kernel tracing (syscalls, scheduling, I/O)
- **atrace**: Android framework events
- **heapprofd**: Memory allocation profiling
- **CPU counters**: Hardware performance counters
- **GPU events**: Graphics pipeline tracing

**Custom Data Sources:**
```cpp
// Example: Custom data source registration
#include "perfetto/tracing/track_event.h"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("myapp")
        .SetDescription("My application events"),
    perfetto::Category("myapp.network")
        .SetDescription("Network operations")
);

// Usage in code:
TRACE_EVENT("myapp", "UserLogin", "user_id", user_id);
TRACE_EVENT_BEGIN("myapp.network", "APICall");
// ... network operation ...
TRACE_EVENT_END("myapp.network");
```

#### 3. Consumers (Clients)
Applications that configure and control tracing sessions:

```cpp
// Example: Consumer configuration
#include "perfetto/tracing/tracing.h"

perfetto::TracingInitArgs args;
args.backends = perfetto::kSystemBackend;
perfetto::Tracing::Initialize(args);

// Create trace config
perfetto::TraceConfig trace_config;
trace_config.add_buffers()->set_size_kb(10240);

auto* ds_config = trace_config.add_data_sources()->mutable_config();
ds_config->set_name("linux.ftrace");
```

## Trace Configuration Deep Dive

### Basic Trace Configuration Structure

```json
{
  "traceConfig": {
    "buffers": [
      {
        "size_kb": 10240,
        "fill_policy": "ring"
      }
    ],
    "data_sources": [
      {
        "config": {
          "name": "linux.ftrace",
          "ftrace_config": {
            "ftrace_events": ["sched/sched_switch", "sched/sched_wakeup"]
          }
        }
      }
    ],
    "duration_ms": 30000,
    "flush_period_ms": 1000
  }
}
```

### Configuration Components Explained

#### Buffer Configuration
```json
{
  "buffers": [
    {
      "size_kb": 10240,           // Buffer size (10MB)
      "fill_policy": "ring",      // Overwrite old data when full
      "transfer_period_ms": 1000  // Transfer frequency
    }
  ]
}
```

**Fill Policies:**
- `"ring"`: Circular buffer, overwrites oldest data
- `"discard"`: Stops collecting when buffer is full

#### Data Source Configuration Examples

**1. Linux ftrace (Kernel Tracing)**
```json
{
  "config": {
    "name": "linux.ftrace",
    "ftrace_config": {
      "ftrace_events": [
        "sched/sched_switch",      // CPU scheduling
        "sched/sched_wakeup",      // Thread wakeups
        "block/block_rq_issue",    // Block I/O requests
        "block/block_rq_complete", // Block I/O completions
        "syscalls/sys_enter_read", // Read system calls
        "syscalls/sys_exit_read"   // Read syscall returns
      ],
      "buffer_size_kb": 2048,
      "drain_period_ms": 250
    }
  }
}
```

**2. Android Framework Tracing (atrace)**
```json
{
  "config": {
    "name": "android.packages_list"
  }
},
{
  "config": {
    "name": "android.surfaceflinger",
    "android_game_intervention_list_config": {}
  }
}
```

**3. Memory Profiling (heapprofd)**
```json
{
  "config": {
    "name": "android.heapprofd",
    "heapprofd_config": {
      "sampling_interval_bytes": 4096,
      "process_cmdline": ["com.example.myapp"],
      "shmem_size_bytes": 8388608,
      "block_client": true
    }
  }
}
```

**4. Custom Application Tracing**
```json
{
  "config": {
    "name": "track_event",
    "track_event_config": {
      "enabled_categories": ["myapp", "myapp.network", "rendering"]
    }
  }
}
```

### Advanced Configuration Example

```json
{
  "traceConfig": {
    "buffers": [
      {
        "size_kb": 65536,          // 64MB buffer
        "fill_policy": "ring"
      }
    ],
    "data_sources": [
      {
        "config": {
          "name": "linux.ftrace",
          "ftrace_config": {
            "ftrace_events": [
              "sched/sched_switch",
              "sched/sched_wakeup",
              "power/cpu_frequency",
              "power/cpu_idle",
              "mm/rss_stat",
              "oom/mark_victim"
            ],
            "symbolize_ksyms": true,
            "buffer_size_kb": 16384
          }
        }
      },
      {
        "config": {
          "name": "linux.process_stats",
          "process_stats_config": {
            "scan_all_processes_on_start": true,
            "proc_stats_poll_ms": 1000
          }
        }
      },
      {
        "config": {
          "name": "track_event",
          "track_event_config": {
            "enabled_categories": ["*"]
          }
        }
      }
    ],
    "duration_ms": 60000,         // 60 seconds
    "flush_period_ms": 5000,      // Flush every 5 seconds
    "write_into_file": true,
    "file_write_period_ms": 2500,
    "max_file_size_bytes": 1073741824  // 1GB max file size
  }
}
```

## Getting Started with Perfetto

### Installation and Setup

#### Linux Installation
```bash
# Method 1: Using prebuilt binaries
curl -LO https://get.perfetto.dev/perfetto
chmod +x perfetto
sudo mv perfetto /usr/local/bin/

# Method 2: Building from source
git clone https://android.googlesource.com/platform/external/perfetto/
cd perfetto
tools/install-build-deps
tools/gn gen --args='is_debug=false' out/linux
tools/ninja -C out/linux
```

#### Android Setup
```bash
# Enable Perfetto on Android device
adb shell setprop persist.traced.enable 1
adb shell setprop persist.traced_probes.enable 1

# Start Perfetto services
adb shell perfetto --background
```

#### Dependencies Check
```bash
# Check if ftrace is available
ls /sys/kernel/debug/tracing/events/

# Check Perfetto services
ps aux | grep traced

# Verify permissions
groups | grep -E "(root|wheel|adm)"
```

### Your First Perfetto Trace

#### Step 1: Create a Simple Configuration
```bash
# Create config.json
cat > config.json << 'EOF'
{
  "traceConfig": {
    "buffers": [{"size_kb": 8192, "fill_policy": "ring"}],
    "data_sources": [{
      "config": {
        "name": "linux.ftrace",
        "ftrace_config": {
          "ftrace_events": ["sched/sched_switch", "sched/sched_wakeup"]
        }
      }
    }],
    "duration_ms": 10000
  }
}
EOF
```

#### Step 2: Start Tracing
```bash
# Start a 10-second trace
perfetto -c config.json -o trace.perfetto-trace

# Alternative: Command-line configuration
perfetto -o trace.perfetto-trace -t 10s \
  --txt -c - <<EOF
duration_ms: 10000
buffers {
  size_kb: 8192
  fill_policy: RING_BUFFER
}
data_sources {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "sched/sched_switch"
      ftrace_events: "sched/sched_wakeup"
    }
  }
}
EOF
```

#### Step 3: Generate Load for Interesting Traces
```bash
# In another terminal, create some activity
stress-ng --cpu 2 --timeout 5s &
dd if=/dev/zero of=/tmp/testfile bs=1M count=100
find /usr -name "*.so" > /dev/null &
```

#### Step 4: Analyze the Trace
```bash
# Option 1: Open in Perfetto UI (recommended)
# Upload trace.perfetto-trace to https://ui.perfetto.dev

# Option 2: Command-line analysis
perfetto query --query "
SELECT ts, dur, name, tid 
FROM slice 
WHERE name LIKE '%sched%' 
LIMIT 10
" trace.perfetto-trace
```

## Practical Examples and Use Cases

### Example 1: CPU Scheduling Analysis

**Goal**: Understand which processes are consuming CPU time and identify scheduling bottlenecks.

```json
{
  "traceConfig": {
    "buffers": [{"size_kb": 32768, "fill_policy": "ring"}],
    "data_sources": [
      {
        "config": {
          "name": "linux.ftrace",
          "ftrace_config": {
            "ftrace_events": [
              "sched/sched_switch",
              "sched/sched_wakeup",
              "sched/sched_wakeup_new",
              "power/cpu_frequency",
              "power/cpu_idle"
            ]
          }
        }
      },
      {
        "config": {
          "name": "linux.process_stats",
          "process_stats_config": {
            "scan_all_processes_on_start": true,
            "proc_stats_poll_ms": 100
          }
        }
      }
    ],
    "duration_ms": 30000
  }
}
```

**Analysis Commands:**
```bash
# Capture CPU-intensive workload
perfetto -c cpu_analysis.json -o cpu_trace.perfetto-trace &
TRACE_PID=$!

# Generate CPU load
stress-ng --cpu 4 --cpu-load 75 --timeout 20s

# Wait for trace to complete
wait $TRACE_PID
```

**What to Look For:**
- Thread state transitions in the timeline
- CPU utilization patterns
- Context switch frequency
- Idle time analysis

### Example 2: Memory Allocation Profiling

**Goal**: Track memory allocations and identify memory hotspots.

```json
{
  "traceConfig": {
    "buffers": [{"size_kb": 65536, "fill_policy": "ring"}],
    "data_sources": [
      {
        "config": {
          "name": "android.heapprofd",
          "heapprofd_config": {
            "sampling_interval_bytes": 1024,
            "process_cmdline": ["stress-ng"],
            "continuous_dump_config": {
              "dump_phase_ms": 1000,
              "dump_interval_ms": 5000
            }
          }
        }
      }
    ],
    "duration_ms": 30000
  }
}
```

**Linux Alternative (using malloc hooks):**
```cpp
// memory_tracer.cpp - Custom memory tracking
#include "perfetto/tracing/track_event.h"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("memory")
        .SetDescription("Memory allocation events")
);

class MemoryTracker {
public:
    static void* tracked_malloc(size_t size) {
        void* ptr = malloc(size);
        TRACE_EVENT("memory", "malloc", "size", size, "ptr", ptr);
        return ptr;
    }
    
    static void tracked_free(void* ptr) {
        TRACE_EVENT("memory", "free", "ptr", ptr);
        free(ptr);
    }
};

// Replace malloc/free calls or use LD_PRELOAD
```

### Example 3: I/O Performance Analysis

**Goal**: Analyze disk I/O patterns and identify bottlenecks.

```json
{
  "traceConfig": {
    "buffers": [{"size_kb": 16384, "fill_policy": "ring"}],
    "data_sources": [
      {
        "config": {
          "name": "linux.ftrace",
          "ftrace_config": {
            "ftrace_events": [
              "block/block_rq_issue",
              "block/block_rq_complete",
              "block/block_bio_queue",
              "block/block_bio_complete",
              "syscalls/sys_enter_read",
              "syscalls/sys_exit_read",
              "syscalls/sys_enter_write",
              "syscalls/sys_exit_write",
              "syscalls/sys_enter_fsync",
              "syscalls/sys_exit_fsync"
            ]
          }
        }
      }
    ],
    "duration_ms": 20000
  }
}
```

**Test I/O Load:**
```bash
# Start trace
perfetto -c io_analysis.json -o io_trace.perfetto-trace &

# Generate I/O activity
dd if=/dev/zero of=/tmp/test_write bs=1M count=1000 &
dd if=/tmp/test_write of=/dev/null bs=1M &
find /usr -name "*.txt" -exec cat {} \; > /dev/null &

# Wait for completion
wait
```

### Example 4: Custom Application Tracing

**Goal**: Add custom tracing to your application for performance insights.

```cpp
// app_tracer.cpp - Custom application tracing
#include "perfetto/tracing/track_event.h"
#include <chrono>
#include <thread>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("myapp")
        .SetDescription("My application events"),
    perfetto::Category("myapp.database")
        .SetDescription("Database operations"),
    perfetto::Category("myapp.network")
        .SetDescription("Network operations"),
    perfetto::Category("myapp.ui")
        .SetDescription("User interface events")
);

class DatabaseManager {
public:
    void executeQuery(const std::string& query) {
        TRACE_EVENT("myapp.database", "ExecuteQuery", "query", query);
        
        // Simulate database work
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        TRACE_COUNTER("myapp.database", "ActiveConnections", ++active_connections_);
    }
    
private:
    int active_connections_ = 0;
};

class NetworkClient {
public:
    void makeRequest(const std::string& url) {
        TRACE_EVENT_BEGIN("myapp.network", "HTTPRequest", "url", url);
        
        // Simulate network request
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        TRACE_EVENT_END("myapp.network");
    }
};

class UIRenderer {
public:
    void renderFrame() {
        TRACE_EVENT("myapp.ui", "RenderFrame");
        
        {
            TRACE_EVENT("myapp.ui", "UpdateLayout");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        {
            TRACE_EVENT("myapp.ui", "DrawElements");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        frame_count_++;
        TRACE_COUNTER("myapp.ui", "FrameCount", frame_count_);
    }
    
private:
    int frame_count_ = 0;
};

int main() {
    // Initialize Perfetto
    perfetto::TracingInitArgs args;
    args.backends = perfetto::kSystemBackend;
    perfetto::Tracing::Initialize(args);
    
    DatabaseManager db;
    NetworkClient client;
    UIRenderer renderer;
    
    // Simulate application workflow
    for (int i = 0; i < 100; ++i) {
        TRACE_EVENT("myapp", "MainLoop", "iteration", i);
        
        db.executeQuery("SELECT * FROM users");
        client.makeRequest("https://api.example.com/data");
        renderer.renderFrame();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }
    
    return 0;
}
```

**Configuration for Custom Tracing:**
```json
{
  "traceConfig": {
    "buffers": [{"size_kb": 8192, "fill_policy": "ring"}],
    "data_sources": [
      {
        "config": {
          "name": "track_event",
          "track_event_config": {
            "enabled_categories": ["myapp*"]
          }
        }
      }
    ],
    "duration_ms": 10000
  }
}
```

**Compilation and Execution:**
```bash
# Compile with Perfetto SDK
g++ -std=c++17 app_tracer.cpp -lperfetto -o app_tracer

# Run with tracing
perfetto -c custom_app.json -o app_trace.perfetto-trace &
./app_tracer
```

## Analyzing Traces with Perfetto UI

### Opening Traces in Perfetto UI

1. **Web Interface**: Navigate to https://ui.perfetto.dev
2. **Upload Trace**: Click "Open trace file" and select your `.perfetto-trace` file
3. **Alternative**: Drag and drop the trace file onto the webpage

### Understanding the Perfetto UI

#### Timeline View
```
┌─────────────────────────────────────────────────────────────┐
│ [Process/Thread] ████████████████████████████████████████   │ CPU 0
│ [Process/Thread] ████████░░░░░░░░░░░░░░░░████████████████   │ CPU 1
│ [Process/Thread] ░░░░░░░░░░░░░░░░░░░░░░░░████████████████   │ CPU 2
│ [Process/Thread] ████████████████████████████████████████   │ CPU 3
├─────────────────────────────────────────────────────────────┤
│ stress-ng       ████████████████████████████████████████   │
│ dd              ░░░░░░░░░░░░░░░░░░░░░░░░████████████████     │
│ find            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████     │
└─────────────────────────────────────────────────────────────┘
  0ms    2s     4s     6s     8s    10s    12s   14s   16s
```

#### Key UI Components

**1. Track Area**: Shows processes, threads, and their activities
**2. Timeline**: Horizontal axis showing time progression
**3. Details Panel**: Shows information about selected events
**4. Query Interface**: SQL-like queries for trace analysis

### Common Analysis Patterns

#### CPU Utilization Analysis
```sql
-- Find top CPU consumers
SELECT 
  process.name,
  thread.name,
  SUM(slice.dur) / 1e9 as cpu_time_seconds
FROM slice
JOIN thread USING(utid)
JOIN process USING(upid)
WHERE slice.name = 'Running'
GROUP BY process.name, thread.name
ORDER BY cpu_time_seconds DESC
LIMIT 10;
```

#### Memory Allocation Analysis
```sql
-- Find memory allocation hotspots
SELECT 
  callsite_id,
  SUM(size) as total_allocated,
  COUNT(*) as allocation_count
FROM heap_profile_allocation
GROUP BY callsite_id
ORDER BY total_allocated DESC
LIMIT 20;
```

#### I/O Performance Analysis
```sql
-- Analyze I/O request latencies
SELECT 
  AVG(dur) / 1e6 as avg_latency_ms,
  MAX(dur) / 1e6 as max_latency_ms,
  COUNT(*) as request_count
FROM slice
WHERE name LIKE '%block_rq%'
AND dur > 0;
```

## Performance Optimization Workflows

### Workflow 1: Identifying CPU Bottlenecks

1. **Capture CPU trace** with scheduling events
2. **Identify high CPU threads** in timeline view
3. **Analyze context switches** - too many may indicate thrashing
4. **Check CPU frequency scaling** - may indicate thermal throttling
5. **Examine thread states** - threads blocked on I/O vs CPU-bound

**Example Analysis:**
```bash
# Capture comprehensive CPU trace
perfetto -o cpu_analysis.perfetto-trace -t 30s \
  -c - <<EOF
buffers {
  size_kb: 32768
  fill_policy: RING_BUFFER
}
data_sources {
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
EOF
```

### Workflow 2: Memory Leak Detection

1. **Enable heap profiling** for target process
2. **Take periodic snapshots** during application lifecycle
3. **Compare allocations** between snapshots
4. **Identify growing allocation sites**
5. **Correlate with application events**

### Workflow 3: I/O Bottleneck Analysis

1. **Capture block and syscall events**
2. **Identify slow I/O operations** in timeline
3. **Analyze queue depths** and wait times
4. **Correlate with filesystem types** and device characteristics
5. **Optimize based on patterns** (sequential vs random access)

## Best Practices and Tips

### Configuration Best Practices

✅ **DO:**
- Start with small buffer sizes and increase as needed
- Use ring buffers for continuous monitoring
- Enable only necessary data sources to minimize overhead
- Set appropriate trace duration for your use case
- Use categories to filter custom events

❌ **DON'T:**
- Use extremely large buffers unnecessarily (memory overhead)
- Enable all data sources at once (performance impact)
- Run production traces for extended periods
- Forget to specify output file location
- Ignore trace file size limits

### Performance Considerations

**Overhead Guidelines:**
- **CPU tracing**: 1-5% overhead
- **Memory profiling**: 5-15% overhead (depending on sampling rate)
- **Custom events**: <1% overhead (moderate usage)
- **Combined tracing**: Can accumulate, test in staging environment

**Buffer Sizing:**
```bash
# Calculate buffer size needs
# Rule of thumb: 1KB per ms of trace duration per active data source
# For 30s trace with 3 data sources: 30000 * 3 * 1KB = ~90MB

# Conservative sizing
buffer_size_kb = duration_ms * active_data_sources * 0.5

# Aggressive sizing (for detailed traces)
buffer_size_kb = duration_ms * active_data_sources * 2.0
```

### Troubleshooting Common Issues

**Issue 1: Permission Denied**
```bash
# Solution: Check user permissions
groups $USER | grep -E "(root|wheel|adm)"

# Or run with sudo (not recommended for production)
sudo perfetto -c config.json -o trace.perfetto-trace
```

**Issue 2: Empty or Truncated Traces**
```bash
# Check buffer size
# Increase buffer_size_kb in configuration

# Check trace duration
# Ensure duration_ms is sufficient for your workload

# Verify data sources are available
ls /sys/kernel/debug/tracing/events/
```

**Issue 3: High Overhead**
```bash
# Reduce sampling frequency
# Lower proc_stats_poll_ms
# Increase sampling_interval_bytes for heap profiling
# Use fewer ftrace events
```

**Issue 4: Missing Custom Events**
```cpp
// Ensure Perfetto is properly initialized
perfetto::TracingInitArgs args;
args.backends = perfetto::kSystemBackend;
perfetto::Tracing::Initialize(args);

// Check if categories are enabled
TRACE_EVENT_CATEGORY_GROUP_ENABLED("myapp", &enabled);
if (!enabled) {
    // Category not enabled in trace config
}
```

## Learning Objectives

By the end of this section, you should be able to:

- **Understand Perfetto architecture** and its core components (traced, probes, consumers)
- **Configure trace sessions** using JSON and protobuf configurations
- **Capture system-wide traces** including CPU, memory, and I/O events
- **Analyze performance traces** using the Perfetto UI effectively
- **Implement custom tracing** in your applications using the Perfetto SDK
- **Identify performance bottlenecks** through systematic trace analysis
- **Apply tracing best practices** for production environments
- **Troubleshoot common tracing issues** and optimize trace configurations

### Self-Assessment Checklist

Before proceeding to advanced topics, ensure you can:

□ Install and set up Perfetto on Linux/Android systems  
□ Create basic trace configurations for different use cases  
□ Capture a 30-second system trace with CPU and memory data  
□ Navigate the Perfetto UI timeline and analyze thread activity  
□ Write SQL queries to extract specific performance metrics  
□ Add custom trace events to a C++ application  
□ Identify CPU scheduling bottlenecks in trace data  
□ Configure memory allocation profiling for heap analysis  
□ Troubleshoot permission and configuration issues  
□ Explain the trade-offs between trace detail and overhead  

### Practical Exercises

**Exercise 1: Basic System Tracing**
```bash
# TODO: Create a configuration that captures:
# - CPU scheduling events
# - Process statistics
# - Block I/O events
# - Duration: 20 seconds
# - Buffer size: 16MB

# Write the configuration and capture a trace while running:
stress-ng --cpu 2 --io 1 --timeout 15s
```

**Exercise 2: Custom Application Tracing**
```cpp
// TODO: Complete this program to add tracing:
#include "perfetto/tracing/track_event.h"

class Calculator {
public:
    int add(int a, int b) {
        // Add trace event here
        return a + b;
    }
    
    int multiply(int a, int b) {
        // Add trace event with parameters
        return a * b;
    }
};

// Add category definitions and initialization
```

**Exercise 3: Memory Profiling**
```bash
# TODO: Configure heap profiling for a memory-intensive program
# - Track allocations > 1KB
# - Sample every 4KB
# - Generate dumps every 5 seconds
# - Target process: your own memory test program
```

**Exercise 4: Trace Analysis**
```sql
-- TODO: Write SQL queries to find:
-- 1. Top 10 processes by CPU time
-- 2. Average I/O latency by device
-- 3. Memory allocation patterns over time
-- 4. Context switch frequency per thread
```

## Study Materials

### Recommended Reading

**Primary Sources:**
- **Perfetto Documentation**: https://perfetto.dev/docs/
- **Getting Started Guide**: https://perfetto.dev/docs/quickstart/
- **Trace Config Reference**: https://perfetto.dev/docs/concepts/config
- **Data Sources**: https://perfetto.dev/docs/data-sources/

**Books:**
- "Systems Performance" by Brendan Gregg (Chapter 4: Profiling)
- "Linux Performance Tools" by Brendan Gregg
- "Android Performance Tuning" by Hernan Gonzalez

**Research Papers:**
- "Perfetto: A Modern Tracing Ecosystem" (Google Research)
- "Efficient System-Wide Performance Analysis" (USENIX)

### Video Resources

- **Google I/O Sessions**: "Perfetto: Next-gen System Profiling"
- **Android Dev Summit**: "Performance Profiling with Perfetto"
- **Linux Foundation**: "System Tracing with Perfetto"
- **Performance Engineering**: YouTube playlist on system profiling

### Hands-on Labs

**Lab 1: System Performance Analysis**
- Set up Perfetto tracing environment
- Capture traces during different workloads
- Analyze CPU utilization patterns
- Identify scheduling anomalies

**Lab 2: Android App Profiling**
- Configure Perfetto for Android development
- Trace app startup and navigation
- Analyze frame timing and jank
- Optimize based on findings

**Lab 3: Custom Data Source Development**
- Implement custom performance counters
- Integrate with existing applications
- Create specialized trace visualizations
- Deploy in production environment

**Lab 4: Memory Leak Investigation**
- Set up heap profiling for long-running processes
- Generate memory pressure scenarios
- Analyze allocation patterns and leaks
- Implement fixes and verify improvements

### Practice Questions

**Conceptual Questions:**
1. What are the main advantages of Perfetto over traditional profiling tools like `perf` or `gprof`?
2. How does Perfetto's architecture enable low-overhead production tracing?
3. What is the difference between ring and discard buffer policies? When would you use each?
4. Explain the role of the `traced` service in the Perfetto ecosystem.
5. How do data sources communicate with the central tracing service?

**Configuration Questions:**
6. How would you configure Perfetto to trace only specific processes?
7. What buffer size would you recommend for a 5-minute trace with moderate activity?
8. How can you enable custom trace categories for your application?
9. What ftrace events are most useful for diagnosing I/O performance issues?
10. How do you configure heap profiling to minimize overhead while maintaining useful data?

**Analysis Questions:**
11. How would you identify which thread is causing high CPU usage?
12. What SQL query would find the longest-running I/O operations?
13. How can you correlate memory allocations with specific application features?
14. What patterns in the timeline view indicate thread synchronization issues?
15. How do you distinguish between CPU-bound and I/O-bound performance problems?

**Troubleshooting Questions:**
16. What steps would you take if Perfetto traces are empty or truncated?
17. How do you resolve permission denied errors when starting traces?
18. What could cause high overhead during tracing, and how would you fix it?
19. How do you verify that custom trace events are being captured correctly?
20. What are common reasons for missing data in heap profiling traces?

### Development Environment Setup

**System Requirements:**
```bash
# Linux requirements
Linux kernel 4.4+ with CONFIG_FTRACE=y
CONFIG_DEBUG_FS=y (for debugfs access)
Root or CAP_SYS_ADMIN capabilities

# Android requirements
Android 9+ (API level 28+)
Developer options enabled
USB debugging enabled
```

**Installation Script:**
```bash
#!/bin/bash
# install_perfetto.sh

set -e

echo "Installing Perfetto..."

# Download latest release
PERFETTO_VERSION="latest"
curl -LO "https://get.perfetto.dev/perfetto"
chmod +x perfetto

# Install to system path
sudo mv perfetto /usr/local/bin/

# Verify installation
perfetto --version

# Install additional tools
sudo apt-get update
sudo apt-get install -y stress-ng

echo "Perfetto installation complete!"
echo "Try: perfetto --help"
```

**Development Tools:**
```bash
# Performance testing tools
sudo apt-get install stress-ng sysbench

# Trace analysis tools
pip3 install perfetto-trace-analyzer

# Build tools (if building from source)
sudo apt-get install ninja-build python3
```

**IDE Configuration:**
```cpp
// .vscode/c_cpp_properties.json (for VS Code)
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/include/perfetto"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```

**Compilation Examples:**
```bash
# Compile with Perfetto SDK
g++ -std=c++17 -I/usr/local/include/perfetto \
    app.cpp -lperfetto -o app

# Debug build with symbols
g++ -g -std=c++17 -I/usr/local/include/perfetto \
    app.cpp -lperfetto -o app_debug

# Android NDK build
$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ \
    -std=c++17 app.cpp -lperfetto -o app_android
```

## Integration Examples

### Web Application Performance Monitoring

```javascript
// perfetto-web-integration.js
class PerfettoWebTracer {
    constructor() {
        this.events = [];
        this.startTime = performance.now();
    }
    
    traceEvent(name, category, duration, args = {}) {
        this.events.push({
            name: name,
            cat: category,
            ph: 'X', // Complete event
            ts: (performance.now() - this.startTime) * 1000, // microseconds
            dur: duration * 1000,
            pid: 1,
            tid: 1,
            args: args
        });
    }
    
    exportTrace() {
        return JSON.stringify({
            traceEvents: this.events,
            displayTimeUnit: 'ms'
        });
    }
}

// Usage example
const tracer = new PerfettoWebTracer();

function fetchData(url) {
    const start = performance.now();
    
    return fetch(url)
        .then(response => {
            const duration = performance.now() - start;
            tracer.traceEvent('fetchData', 'network', duration, {
                url: url,
                status: response.status
            });
            return response.json();
        });
}
```

### Database Query Profiling

```python
# perfetto_db_profiler.py
import time
import json
from contextlib import contextmanager

class DatabaseProfiler:
    def __init__(self):
        self.events = []
        self.start_time = time.time()
    
    @contextmanager
    def trace_query(self, query, table=None):
        start = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start) * 1000  # ms
            self.events.append({
                'name': 'Database Query',
                'cat': 'database',
                'ph': 'X',
                'ts': (start - self.start_time) * 1000000,  # microseconds
                'dur': duration * 1000,  # microseconds
                'pid': 1,
                'tid': 1,
                'args': {
                    'query': query[:100],  # Truncate long queries
                    'table': table,
                    'duration_ms': duration
                }
            })
    
    def export_trace(self):
        return json.dumps({
            'traceEvents': self.events,
            'displayTimeUnit': 'ms'
        })

# Usage
profiler = DatabaseProfiler()

def execute_query(query, table=None):
    with profiler.trace_query(query, table):
        # Execute actual database query
        result = database.execute(query)
        return result
```

## Next Steps and Advanced Topics

After mastering the fundamentals, consider exploring:

1. **Advanced Data Sources**: Custom kernel modules, eBPF integration
2. **Distributed Tracing**: Cross-service performance analysis
3. **Real-time Monitoring**: Live trace streaming and alerting
4. **Performance Regression Detection**: Automated performance testing
5. **Mobile Performance**: Battery, thermal, and network optimization
6. **Production Deployment**: Monitoring infrastructure and alerting
