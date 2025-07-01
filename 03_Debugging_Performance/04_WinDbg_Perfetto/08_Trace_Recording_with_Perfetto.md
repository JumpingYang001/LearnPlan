# Trace Recording with Perfetto

*Duration: 2-3 days*

## Overview
Master trace recording methods and protocols for system-wide and app-specific tracing using Perfetto. This section covers comprehensive trace collection techniques, from basic command-line usage to advanced programmatic recording with custom configurations.

## What is Perfetto Trace Recording?

Perfetto is Google's production-grade system tracing framework that enables:
- **System-wide tracing** across kernel, userspace, and applications
- **High-performance data collection** with minimal overhead
- **Flexible configuration** for different tracing scenarios
- **Cross-platform support** (Android, Linux, Chrome OS, Windows)
- **Rich data sources** including CPU, memory, I/O, graphics, and custom events

### Architecture Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Applications  │    │   System Daemons│    │   Kernel Space  │
│                 │    │                 │    │                 │
│ App Traces      │    │ Service Traces  │    │ Kernel Events   │
│ Custom Events   │    │ System Metrics  │    │ Hardware Events │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     Perfetto Daemon      │
                    │   (Data Collection &     │
                    │    Buffering Service)    │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Trace File Output     │
                    │  (.perfetto-trace)       │
                    └───────────────────────────┘
```

## Recording Methods

### 1. Command Line Interface (CLI)

#### Basic Recording
```bash
# Simple system trace for 10 seconds
perfetto -o trace.perfetto-trace -t 10s

# Record with specific data sources
perfetto -o trace.perfetto-trace -t 10s \
  --buffer-size 64mb \
  --data-source linux.ftrace \
  --data-source linux.process_stats
```

#### Advanced CLI Options
```bash
# Comprehensive system trace
perfetto \
  --out system_trace.perfetto-trace \
  --time 30s \
  --buffer-size 128mb \
  --config - <<EOF
buffers: {
  size_kb: 131072
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

data_sources: {
  config {
    name: "linux.process_stats"
    process_stats_config {
      scan_all_processes_on_start: true
      proc_stats_poll_ms: 1000
    }
  }
}
EOF
```

### 2. Configuration File Method

#### Creating a Trace Configuration

**myconfig.pbtxt:**
```protobuf
# Perfetto trace configuration
# Duration: 60 seconds
duration_ms: 60000

# Buffer configuration
buffers: {
  size_kb: 65536        # 64MB buffer
  fill_policy: DISCARD  # Drop old data when buffer is full
}

# CPU scheduling events
data_sources: {
  config {
    name: "linux.ftrace"
    target_buffer: 0
    ftrace_config {
      # Scheduler events
      ftrace_events: "sched/sched_switch"
      ftrace_events: "sched/sched_wakeup"
      ftrace_events: "sched/sched_wakeup_new"
      ftrace_events: "sched/sched_process_fork"
      ftrace_events: "sched/sched_process_exec"
      ftrace_events: "sched/sched_process_exit"
      
      # Power management
      ftrace_events: "power/cpu_frequency"
      ftrace_events: "power/cpu_idle"
      ftrace_events: "power/suspend_resume"
      
      # Memory management
      ftrace_events: "mm/mm_page_alloc"
      ftrace_events: "mm/mm_page_free"
      ftrace_events: "vmscan/mm_vmscan_direct_reclaim_begin"
      ftrace_events: "vmscan/mm_vmscan_direct_reclaim_end"
      
      # I/O events
      ftrace_events: "block/block_rq_issue"
      ftrace_events: "block/block_rq_complete"
      
      # Network events
      ftrace_events: "net/netif_rx"
      ftrace_events: "net/net_dev_xmit"
    }
  }
}

# Process and memory statistics
data_sources: {
  config {
    name: "linux.process_stats"
    target_buffer: 0
    process_stats_config {
      scan_all_processes_on_start: true
      proc_stats_poll_ms: 5000
      proc_stats_cache_ttl_ms: 30000
    }
  }
}

# System information
data_sources: {
  config {
    name: "linux.sys_stats"
    target_buffer: 0
    sys_stats_config {
      stat_period_ms: 1000
      stat_counters: STAT_CPU_TIMES
      stat_counters: STAT_FORK_COUNT
      stat_counters: STAT_MEMINFO
    }
  }
}
```

**Running with Configuration:**
```bash
# Record trace using configuration file
perfetto --config myconfig.pbtxt --out trace_file.perfetto-trace

# Record with additional runtime options
perfetto \
  --config myconfig.pbtxt \
  --out detailed_trace.perfetto-trace \
  --upload  # Upload to trace processor
```

### 3. Programmatic Recording (C++)

#### Using Perfetto SDK in Your Application

**trace_recorder.cpp:**
```cpp
#include <perfetto.h>
#include <thread>
#include <chrono>

PERFETTO_DEFINE_CATEGORIES(
  perfetto::Category("app")
    .SetDescription("Application events"),
  perfetto::Category("performance")
    .SetDescription("Performance measurements")
);

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

class TraceRecorder {
public:
    static void Initialize() {
        perfetto::TracingInitArgs args;
        
        // Configure the trace config
        args.backends = perfetto::kInProcessBackend;
        perfetto::Tracing::Initialize(args);
        
        // Register track event data source
        perfetto::TrackEvent::Register();
    }
    
    static void StartTracing(const std::string& output_file) {
        // Create trace config
        perfetto::TraceConfig cfg;
        cfg.add_buffers()->set_size_kb(1024);  // 1MB buffer
        
        auto* ds_cfg = cfg.add_data_sources()->mutable_config();
        ds_cfg->set_name("track_event");
        
        // Start tracing session
        auto tracing_session = perfetto::Tracing::NewTrace();
        tracing_session->Setup(cfg);
        tracing_session->StartBlocking();
        
        // Record events
        RecordApplicationEvents();
        
        // Stop and save trace
        tracing_session->StopBlocking();
        
        std::vector<char> trace_data(tracing_session->ReadTraceBlocking());
        
        // Write to file
        std::ofstream output(output_file, std::ios::binary);
        output.write(trace_data.data(), trace_data.size());
    }
    
private:
    static void RecordApplicationEvents() {
        // Record custom application events
        TRACE_EVENT_BEGIN("app", "ApplicationStartup");
        
        // Simulate application work
        for (int i = 0; i < 10; ++i) {
            TRACE_EVENT("performance", "ProcessData", "iteration", i);
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Simulate processing
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            TRACE_COUNTER("performance", "ProcessingTime", duration.count());
        }
        
        TRACE_EVENT_END("app");
    }
};

int main() {
    TraceRecorder::Initialize();
    TraceRecorder::StartTracing("app_trace.perfetto-trace");
    return 0;
}
```

**Compilation:**
```bash
# Install Perfetto SDK first
git clone https://android.googlesource.com/platform/external/perfetto/
cd perfetto
tools/install-build-deps
tools/gn gen --args='is_debug=false' out/release
tools/ninja -C out/release

# Compile your application
g++ -std=c++17 -I./include \
    trace_recorder.cpp \
    -L./out/release \
    -lperfetto \
    -o trace_recorder
```

### 4. Android-Specific Recording

#### Using ADB for Android Tracing
```bash
# Record system trace on Android device
adb shell perfetto \
  -o /data/misc/perfetto-traces/trace \
  --txt \
  -c - <<EOF

duration_ms: 10000

buffers: {
  size_kb: 8192
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
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "sched/sched_switch"
      ftrace_events: "gfx/mali_job_slots_event"
    }
  }
}
EOF

# Pull trace file from device
adb pull /data/misc/perfetto-traces/trace ./android_trace.perfetto-trace
```

#### Systrace Integration
```bash
# Using systrace wrapper (deprecated but still useful)
python systrace.py -t 10 -o systrace.html \
  gfx input view webview wm am sm audio video camera hal \
  res dalvik rs bionic power pm ss database network adb \
  vibrator aidl nnapi rro

# Convert systrace to perfetto format
trace_to_text systrace systrace.html --output=systrace.perfetto-trace
```

## Advanced Recording Techniques

### 1. Multi-Buffer Configuration

```protobuf
# Configuration for different event types in separate buffers
buffers: {
  size_kb: 32768     # 32MB for high-frequency events
  fill_policy: DISCARD
}

buffers: {
  size_kb: 8192      # 8MB for low-frequency events
  fill_policy: RING_BUFFER
}

# High-frequency scheduler events -> Buffer 0
data_sources: {
  config {
    name: "linux.ftrace"
    target_buffer: 0
    ftrace_config {
      ftrace_events: "sched/sched_switch"
      ftrace_events: "sched/sched_wakeup"
    }
  }
}

# Low-frequency process stats -> Buffer 1
data_sources: {
  config {
    name: "linux.process_stats"
    target_buffer: 1
    process_stats_config {
      scan_all_processes_on_start: true
      proc_stats_poll_ms: 10000
    }
  }
}
```

### 2. Conditional Triggering

```protobuf
# Trigger-based recording
duration_ms: 120000  # 2 minutes max

# Define triggers
triggers: {
  trigger_name: "high_cpu_usage"
  producer_name_regex: ".*"
  stop_delay_ms: 5000
}

triggers: {
  trigger_name: "memory_pressure"
  producer_name_regex: ".*"
  stop_delay_ms: 10000
}

# Configure trigger conditions in your application
data_sources: {
  config {
    name: "track_event"
    track_event_config {
      enabled_categories: "performance"
      enabled_categories: "memory"
    }
  }
}
```

### 3. Custom Data Sources

**custom_data_source.h:**
```cpp
#include <perfetto.h>

class CustomDataSource : public perfetto::DataSource<CustomDataSource> {
public:
    void OnSetup(const SetupArgs&) override;
    void OnStart(const StartArgs&) override;
    void OnStop(const StopArgs&) override;
    
    static void RecordMetric(const std::string& name, double value);
    static void RecordEvent(const std::string& category, const std::string& name);
};

PERFETTO_DECLARE_DATA_SOURCE_STATIC_MEMBERS(CustomDataSource);
```

**custom_data_source.cpp:**
```cpp
#include "custom_data_source.h"

PERFETTO_DEFINE_DATA_SOURCE_STATIC_MEMBERS(CustomDataSource);

void CustomDataSource::OnSetup(const SetupArgs&) {
    // Initialize data source
}

void CustomDataSource::OnStart(const StartArgs&) {
    // Start collecting data
}

void CustomDataSource::OnStop(const StopArgs&) {
    // Clean up
}

void CustomDataSource::RecordMetric(const std::string& name, double value) {
    Trace([&](TraceContext ctx) {
        auto packet = ctx.NewTracePacket();
        packet->set_timestamp(perfetto::TrackEvent::GetTraceTimeNs());
        
        auto* track_event = packet->set_track_event();
        track_event->set_name(name);
        track_event->set_type(perfetto::protos::pbzero::TrackEvent::TYPE_COUNTER);
        
        auto* counter = track_event->set_counter_value();
        counter->set_double_value(value);
    });
}
```

## Practical Recording Scenarios

### 1. Performance Analysis Recording

```bash
#!/bin/bash
# performance_trace.sh - Record trace for performance analysis

echo "Starting performance trace recording..."

perfetto \
  --out performance_trace_$(date +%Y%m%d_%H%M%S).perfetto-trace \
  --time 60s \
  --buffer-size 256mb \
  --config - <<EOF
buffers: {
  size_kb: 262144
  fill_policy: DISCARD
}

# CPU and scheduling
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

# Process statistics
data_sources: {
  config {
    name: "linux.process_stats"
    process_stats_config {
      scan_all_processes_on_start: true
      proc_stats_poll_ms: 1000
    }
  }
}

# Memory events
data_sources: {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "mm/mm_page_alloc"
      ftrace_events: "mm/mm_page_free"
      ftrace_events: "oom/oom_score_adj_update"
    }
  }
}
EOF

echo "Performance trace recording completed!"
```

### 2. I/O Analysis Recording

```bash
#!/bin/bash
# io_trace.sh - Record trace for I/O analysis

perfetto \
  --out io_trace_$(date +%Y%m%d_%H%M%S).perfetto-trace \
  --time 30s \
  --config - <<EOF
duration_ms: 30000

buffers: {
  size_kb: 65536
  fill_policy: RING_BUFFER
}

# Block I/O events
data_sources: {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "block/block_rq_insert"
      ftrace_events: "block/block_rq_issue"
      ftrace_events: "block/block_rq_complete"
      ftrace_events: "block/block_bio_queue"
      ftrace_events: "block/block_bio_complete"
    }
  }
}

# File system events
data_sources: {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "ext4/ext4_da_write_begin"
      ftrace_events: "ext4/ext4_da_write_end"
      ftrace_events: "ext4/ext4_sync_file_enter"
      ftrace_events: "ext4/ext4_sync_file_exit"
    }
  }
}
EOF
```

### 3. Network Analysis Recording

```bash
# network_trace.sh - Record trace for network analysis
perfetto \
  --out network_trace.perfetto-trace \
  --time 45s \
  --config - <<EOF
duration_ms: 45000

buffers: {
  size_kb: 32768
  fill_policy: DISCARD
}

# Network events
data_sources: {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "net/netif_rx"
      ftrace_events: "net/net_dev_xmit"
      ftrace_events: "net/net_dev_queue"
      ftrace_events: "tcp/tcp_probe"
      ftrace_events: "sock/inet_sock_set_state"
    }
  }
}

# Network statistics
data_sources: {
  config {
    name: "linux.sys_stats"
    sys_stats_config {
      stat_period_ms: 2000
      stat_counters: STAT_NETDEV
    }
  }
}
EOF
```

## Troubleshooting Common Issues

### 1. Permission Issues

**Problem:** `Permission denied` when recording traces
```bash
perfetto: Permission denied opening /sys/kernel/debug/tracing/trace_marker
```

**Solutions:**
```bash
# Run with sudo (temporary solution)
sudo perfetto --config myconfig.pbtxt --out trace.perfetto-trace

# Add user to tracing group (permanent solution)
sudo usermod -a -G tracing $USER
sudo chmod 664 /sys/kernel/debug/tracing/trace_marker

# For Android development
adb root
adb shell setenforce 0  # Disable SELinux temporarily
```

### 2. Buffer Overflow Issues

**Problem:** Lost events due to buffer overflow
```
[perfetto] Buffer overflow detected, events may be lost
```

**Solutions:**
```protobuf
# Increase buffer size
buffers: {
  size_kb: 262144  # Increase from default
  fill_policy: DISCARD  # or RING_BUFFER
}

# Use multiple buffers for different event types
buffers: {
  size_kb: 131072  # High-frequency events
  fill_policy: DISCARD
}
buffers: {
  size_kb: 65536   # Low-frequency events
  fill_policy: RING_BUFFER
}
```

### 3. High Overhead Issues

**Problem:** Tracing causes significant performance impact

**Optimization Strategies:**
```protobuf
# Reduce event frequency
data_sources: {
  config {
    name: "linux.process_stats"
    process_stats_config {
      proc_stats_poll_ms: 10000  # Increase from 1000ms
      proc_stats_cache_ttl_ms: 60000
    }
  }
}

# Filter specific processes
data_sources: {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "sched/sched_switch"
      # Add process filters
      atrace_apps: "com.yourapp.package"
    }
  }
}
```

### 4. Missing Events

**Problem:** Expected events not appearing in trace

**Debugging Steps:**
```bash
# Check available events
cat /sys/kernel/debug/tracing/available_events | grep sched

# Verify event is enabled
cat /sys/kernel/debug/tracing/events/sched/sched_switch/enable

# Test with minimal config
perfetto --config - <<EOF
duration_ms: 5000
buffers: { size_kb: 1024 }
data_sources: {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "sched/sched_switch"
    }
  }
}
EOF
```

## Performance Optimization

### 1. Buffer Management

```protobuf
# Optimized buffer configuration
buffers: {
  size_kb: 65536
  fill_policy: RING_BUFFER  # Better for continuous monitoring
}

# For short-term analysis
buffers: {
  size_kb: 32768
  fill_policy: DISCARD     # Better for burst analysis
}
```

### 2. Selective Event Recording

```bash
# Record only critical events for production
perfetto --config - <<EOF
duration_ms: 300000  # 5 minutes

buffers: {
  size_kb: 16384
  fill_policy: RING_BUFFER
}

# Only essential scheduler events
data_sources: {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "sched/sched_switch"
      # Skip wakeup events to reduce overhead
    }
  }
}

# Reduce polling frequency
data_sources: {
  config {
    name: "linux.process_stats"
    process_stats_config {
      proc_stats_poll_ms: 30000  # 30 seconds
    }
  }
}
EOF
```

### 3. Compression and Storage

```bash
# Compress trace files
gzip large_trace.perfetto-trace

# Stream directly to compressed file
perfetto --config myconfig.pbtxt --out /dev/stdout | gzip > trace.perfetto-trace.gz

# Use trace_processor for analysis without decompression
trace_processor --httpd trace.perfetto-trace.gz
```

## Best Practices

### 1. Configuration Management

```bash
# Create reusable configuration templates
mkdir -p ~/.perfetto/configs

# Performance analysis template
cat > ~/.perfetto/configs/performance.pbtxt <<EOF
duration_ms: 60000
buffers: { size_kb: 131072, fill_policy: DISCARD }
data_sources: {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "sched/sched_switch"
      ftrace_events: "sched/sched_wakeup"
      ftrace_events: "power/cpu_frequency"
    }
  }
}
data_sources: {
  config {
    name: "linux.process_stats"
    process_stats_config {
      scan_all_processes_on_start: true
      proc_stats_poll_ms: 5000
    }
  }
}
EOF

# Use template
perfetto --config ~/.perfetto/configs/performance.pbtxt --out trace.perfetto-trace
```

### 2. Automated Recording Scripts

```bash
#!/bin/bash
# automated_trace.sh - Automated trace recording with analysis

set -e

TRACE_DIR="./traces"
CONFIG_FILE="$1"
DURATION="$2"
OUTPUT_PREFIX="$3"

# Validate inputs
if [[ -z "$CONFIG_FILE" || -z "$DURATION" || -z "$OUTPUT_PREFIX" ]]; then
    echo "Usage: $0 <config_file> <duration_seconds> <output_prefix>"
    exit 1
fi

# Create trace directory
mkdir -p "$TRACE_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRACE_FILE="$TRACE_DIR/${OUTPUT_PREFIX}_${TIMESTAMP}.perfetto-trace"

echo "Starting trace recording..."
echo "Config: $CONFIG_FILE"
echo "Duration: ${DURATION}s"
echo "Output: $TRACE_FILE"

# Record trace
perfetto \
    --config "$CONFIG_FILE" \
    --out "$TRACE_FILE" \
    --time "${DURATION}s"

# Verify trace file
if [[ -f "$TRACE_FILE" ]]; then
    FILE_SIZE=$(stat -c%s "$TRACE_FILE")
    echo "Trace recorded successfully!"
    echo "File: $TRACE_FILE"
    echo "Size: $((FILE_SIZE / 1024 / 1024)) MB"
    
    # Optional: Generate quick report
    if command -v trace_processor >/dev/null 2>&1; then
        echo "Generating quick analysis..."
        trace_processor --run-metrics android_cpu "$TRACE_FILE" > "${TRACE_FILE%.perfetto-trace}_report.txt"
    fi
else
    echo "Error: Trace file not created!"
    exit 1
fi
```

### 3. Integration with CI/CD

```yaml
# .github/workflows/performance-trace.yml
name: Performance Trace Analysis

on:
  pull_request:
    paths:
      - 'src/**'
      - 'performance/**'

jobs:
  trace-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Perfetto
        run: |
          curl -LO https://get.perfetto.dev/perfetto
          chmod +x perfetto
          sudo mv perfetto /usr/local/bin/
      
      - name: Record Performance Trace
        run: |
          # Run application with tracing
          ./scripts/run_with_trace.sh
          
      - name: Analyze Trace
        run: |
          # Generate performance metrics
          trace_processor --run-metrics android_cpu trace.perfetto-trace > metrics.json
          
      - name: Upload Trace Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: performance-traces
          path: |
            *.perfetto-trace
            metrics.json
```

## Learning Objectives

By the end of this section, you should be able to:

### Technical Skills
- **Configure and execute** Perfetto trace recordings using CLI, configuration files, and programmatic methods
- **Design optimal trace configurations** for different analysis scenarios (performance, I/O, memory, network)
- **Implement custom data sources** and integrate Perfetto SDK into applications
- **Troubleshoot common recording issues** including permissions, buffer overflows, and missing events
- **Optimize trace recording** for minimal performance impact in production environments

### Practical Applications
- **Record system-wide traces** for comprehensive performance analysis
- **Capture application-specific events** using custom instrumentation
- **Monitor Android applications** using ADB and platform-specific data sources
- **Automate trace collection** in CI/CD pipelines and production monitoring

### Analysis Preparation
- **Select appropriate data sources** based on analysis requirements
- **Configure buffer management** for different recording scenarios
- **Prepare traces for analysis** using trace_processor and other tools

## Self-Assessment Checklist

Before proceeding to trace analysis, ensure you can:

□ Record a basic system trace using Perfetto CLI  
□ Create and use configuration files for different tracing scenarios  
□ Troubleshoot permission and buffer overflow issues  
□ Implement custom trace events in a C++ application  
□ Configure multi-buffer setups for high-frequency events  
□ Use conditional triggers for selective recording  
□ Record Android application traces using ADB  
□ Optimize trace recording for production use  
□ Automate trace collection with scripts  
□ Integrate tracing into development workflows  

## Practical Exercises

### Exercise 1: Basic Trace Recording
```bash
# TODO: Create a trace recording configuration that captures:
# - CPU scheduling events for 30 seconds
# - Process statistics every 5 seconds
# - Memory allocation events
# - Save to a file named "basic_trace_YYYYMMDD.perfetto-trace"

# Your configuration here:
perfetto --config - <<EOF
# Complete this configuration
EOF
```

### Exercise 2: Custom Application Tracing
```cpp
// TODO: Implement a C++ application that:
// 1. Records custom performance events
// 2. Measures function execution times
// 3. Tracks memory allocations
// 4. Exports trace to Perfetto format

#include <perfetto.h>

class PerformanceTracker {
public:
    // TODO: Implement initialization
    static void Initialize();
    
    // TODO: Implement event recording
    static void RecordEvent(const std::string& name, int64_t duration_us);
    
    // TODO: Implement trace export
    static void ExportTrace(const std::string& filename);
};

// Your implementation here
```

### Exercise 3: Android Performance Analysis
```bash
# TODO: Create an Android trace recording script that:
# 1. Records UI performance events
# 2. Captures graphics pipeline events
# 3. Monitors memory usage
# 4. Runs for 60 seconds during app usage

#!/bin/bash
# Your script here
```

### Exercise 4: Production Monitoring Setup
```bash
# TODO: Design a production-safe trace recording configuration:
# 1. Minimal performance overhead (< 5%)
# 2. Continuous recording with ring buffer
# 3. Automatic file rotation
# 4. Error handling and recovery

# Your solution here
```

## Study Materials

### Official Documentation
- **Perfetto Documentation**: https://perfetto.dev/docs/
- **Trace Configuration Reference**: https://perfetto.dev/docs/concepts/config
- **Data Sources Guide**: https://perfetto.dev/docs/data-sources/
- **SDK Integration Guide**: https://perfetto.dev/docs/instrumentation/

### Video Resources
- "Perfetto: System-wide Tracing" - Google I/O sessions
- "Android Performance Tracing" - Android Dev Summit
- "Linux Kernel Tracing with Perfetto" - Linux Plumbers Conference

### Hands-on Labs
- **Lab 1:** Set up Perfetto on different platforms (Linux, Android, Chrome OS)
- **Lab 2:** Create comprehensive system performance traces
- **Lab 3:** Implement custom tracing in a real application
- **Lab 4:** Build automated trace collection pipeline

### Practice Scenarios

**Scenario 1: Web Application Performance**
```bash
# Record Chrome browser performance during web app usage
perfetto --config chrome_performance.pbtxt --out webapp_trace.perfetto-trace

# Configuration should include:
# - Chrome tracing events
# - System CPU/memory events
# - Network I/O events
```

**Scenario 2: Mobile Game Performance**
```bash
# Record Android game performance trace
adb shell perfetto --config game_performance.pbtxt --out /sdcard/game_trace.perfetto-trace

# Configuration should include:
# - Graphics pipeline events
# - Touch input events
# - Audio subsystem events
# - Thermal management events
```

**Scenario 3: Server Performance Monitoring**
```bash
# Continuous server monitoring with Perfetto
perfetto --config server_monitoring.pbtxt --out /var/log/perfetto/server_trace.perfetto-trace

# Configuration should include:
# - Process scheduling
# - Network connections
# - Disk I/O
# - Memory pressure events
```

### Reference Commands

```bash
# Essential Perfetto commands for quick reference

# Basic recording
perfetto -o trace.perfetto-trace -t 10s

# List available data sources
perfetto --query-raw

# Record with specific data source
perfetto -o trace.perfetto-trace -t 10s -c linux.ftrace

# View trace in browser
trace_processor --httpd trace.perfetto-trace

# Extract metrics
trace_processor --run-metrics android_cpu trace.perfetto-trace

# Convert other formats
trace_to_text systrace trace.perfetto-trace --output=trace.html
```

## Next Steps

After mastering trace recording, proceed to:
- **[Trace Analysis and Visualization](09_Trace_Analysis_Visualization.md)** - Learn to analyze recorded traces
- **[Performance Metrics Extraction](10_Performance_Metrics.md)** - Extract quantitative insights
- **[Advanced Perfetto Features](11_Advanced_Perfetto.md)** - Master advanced tracing techniques

---

*Remember: Effective trace recording is the foundation of performance analysis. Focus on understanding your analysis goals before configuring traces.*
