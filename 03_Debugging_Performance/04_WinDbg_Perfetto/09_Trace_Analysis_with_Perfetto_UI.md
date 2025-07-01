# Trace Analysis with Perfetto UI

*Duration: 2-3 days*

## Overview

Perfetto is Google's production-grade system tracing framework that provides powerful trace analysis capabilities through its web-based UI. It allows you to analyze system-level performance traces, understand application behavior, and identify performance bottlenecks through interactive visualizations and SQL-based querying.

### What is Perfetto?

Perfetto consists of:
- **Tracing SDK**: For instrumenting applications
- **Trace Processor**: SQL-based trace analysis engine  
- **UI**: Web-based interface for visualization and analysis
- **System Backend**: Low-level tracing infrastructure

### Key Capabilities
- ✅ **System-wide tracing** (CPU, GPU, memory, I/O)
- ✅ **Application-level tracing** with custom events
- ✅ **SQL-based analysis** with powerful query engine
- ✅ **Interactive visualization** with timeline views
- ✅ **Performance metrics** and flame graphs
- ✅ **Custom dashboards** and analysis workflows

## Getting Started with Perfetto UI

### Accessing Perfetto UI

**Option 1: Web Interface (Recommended)**
```bash
# Open in browser
https://ui.perfetto.dev/
```

**Option 2: Local Instance**
```bash
# Install and run locally
git clone https://github.com/google/perfetto.git
cd perfetto
./tools/install-build-deps
./tools/build_ui
# Open http://localhost:10000
```

### Loading Trace Files

**Supported Formats:**
- `.perfetto-trace` (native format)
- `.pb` (protobuf traces)
- `.json` (Chrome trace format)
- `.ctrace` (compressed traces)

**Loading Methods:**
1. **Drag & Drop**: Drag trace file to UI
2. **File Menu**: File → Open trace file
3. **URL Loading**: Load from remote URL
4. **Live Tracing**: Connect to running trace session

### UI Layout Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Header: File controls, Search, Settings                     │
├─────────────────────────────────────────────────────────────┤
│ Left Panel: Track selection, Process tree                   │
├─────────────────────────────────────────────────────────────┤
│ Main Timeline: Interactive trace visualization              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Timeline ruler with time markers                        │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ Process/Thread tracks                                   │ │
│ │ ├─ CPU tracks                                           │ │
│ │ ├─ GPU tracks                                           │ │
│ │ ├─ Memory tracks                                        │ │
│ │ └─ Custom tracks                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Bottom Panel: Details, SQL Console, Metrics                 │
└─────────────────────────────────────────────────────────────┘
```

## SQL-Based Trace Analysis

Perfetto's trace processor provides a powerful SQL interface for querying trace data. The underlying database contains tables representing different aspects of the trace.

### Core Database Schema

**Key Tables:**
- `slice`: Function calls, events with duration
- `sched`: CPU scheduling events
- `process`: Process information
- `thread`: Thread information  
- `track`: Timeline tracks
- `counter`: Numeric time series data
- `flow`: Event connections/dependencies
- `args`: Additional event arguments

### Essential SQL Queries

#### 1. Basic Process and Thread Analysis

**List all processes:**
```sql
SELECT pid, name, start_ts, end_ts 
FROM process 
ORDER BY start_ts;
```

**Find threads for a specific process:**
```sql
SELECT t.tid, t.name as thread_name, p.name as process_name
FROM thread t
JOIN process p ON t.upid = p.upid
WHERE p.name = 'your_app_name';
```

**Get thread activity summary:**
```sql
SELECT 
    t.name as thread_name,
    COUNT(*) as event_count,
    MIN(s.ts) as start_time,
    MAX(s.ts + s.dur) as end_time,
    (MAX(s.ts + s.dur) - MIN(s.ts)) / 1e9 as duration_seconds
FROM slice s
JOIN thread_track tt ON s.track_id = tt.id
JOIN thread t ON tt.utid = t.utid
GROUP BY t.utid
ORDER BY event_count DESC;
```

#### 2. CPU Scheduling Analysis

**CPU utilization per core:**
```sql
SELECT 
    cpu,
    COUNT(*) as sched_events,
    SUM(dur) / 1e9 as total_runtime_seconds
FROM sched
GROUP BY cpu
ORDER BY cpu;
```

**Thread scheduling patterns:**
```sql
SELECT 
    t.name as thread_name,
    COUNT(*) as times_scheduled,
    SUM(s.dur) / 1e6 as total_runtime_ms,
    AVG(s.dur) / 1e6 as avg_runtime_ms
FROM sched s
JOIN thread t ON s.utid = t.utid
WHERE t.name IS NOT NULL
GROUP BY s.utid
ORDER BY total_runtime_ms DESC
LIMIT 20;
```

**Context switches analysis:**
```sql
SELECT 
    prev_comm as previous_thread,
    next_comm as next_thread,
    COUNT(*) as switch_count
FROM raw
WHERE name = 'sched_switch'
GROUP BY prev_comm, next_comm
ORDER BY switch_count DESC
LIMIT 10;
```

#### 3. Function Call Analysis

**Most expensive function calls:**
```sql
SELECT 
    name,
    COUNT(*) as call_count,
    SUM(dur) / 1e6 as total_duration_ms,
    AVG(dur) / 1e6 as avg_duration_ms,
    MAX(dur) / 1e6 as max_duration_ms
FROM slice
WHERE dur > 0
GROUP BY name
ORDER BY total_duration_ms DESC
LIMIT 20;
```

**Function call hierarchy:**
```sql
WITH RECURSIVE call_stack AS (
    SELECT 
        id, parent_id, name, dur, depth, ts,
        CAST(name AS TEXT) as call_path
    FROM slice 
    WHERE parent_id IS NULL
    
    UNION ALL
    
    SELECT 
        s.id, s.parent_id, s.name, s.dur, s.depth, s.ts,
        CAST(cs.call_path || ' → ' || s.name AS TEXT)
    FROM slice s
    JOIN call_stack cs ON s.parent_id = cs.id
)
SELECT call_path, dur / 1e6 as duration_ms
FROM call_stack
WHERE depth <= 5  -- Limit depth to avoid huge results
ORDER BY dur DESC
LIMIT 50;
```

#### 4. Memory Analysis

**Memory allocations:**
```sql
SELECT 
    process.name as process_name,
    SUM(CASE WHEN counter.name = 'mem.rss' THEN value ELSE 0 END) as rss_kb,
    SUM(CASE WHEN counter.name = 'mem.virt' THEN value ELSE 0 END) as virt_kb
FROM counter_track
JOIN counter ON counter_track.id = counter.track_id
JOIN process_counter_track ON counter_track.id = process_counter_track.id
JOIN process ON process_counter_track.upid = process.upid
WHERE counter.name IN ('mem.rss', 'mem.virt')
GROUP BY process.upid
ORDER BY rss_kb DESC;
```

**Memory usage over time:**
```sql
SELECT 
    ts / 1e9 as time_seconds,
    value / 1024 as memory_mb,
    p.name as process_name
FROM counter c
JOIN counter_track ct ON c.track_id = ct.id
JOIN process_counter_track pct ON ct.id = pct.id
JOIN process p ON pct.upid = p.upid
WHERE ct.name = 'mem.rss'
ORDER BY ts;
```

#### 5. Performance Bottleneck Detection

**Long-running operations:**
```sql
SELECT 
    name,
    dur / 1e6 as duration_ms,
    ts / 1e9 as start_time_seconds,
    thread.name as thread_name
FROM slice
JOIN thread_track ON slice.track_id = thread_track.id
JOIN thread ON thread_track.utid = thread.utid
WHERE dur > 100e6  -- Operations longer than 100ms
ORDER BY dur DESC
LIMIT 50;
```

**Blocking operations analysis:**
```sql
SELECT 
    s.name as operation,
    COUNT(*) as occurrence_count,
    AVG(s.dur) / 1e6 as avg_duration_ms,
    SUM(s.dur) / 1e6 as total_duration_ms
FROM slice s
JOIN thread_track tt ON s.track_id = tt.id
WHERE s.name LIKE '%lock%' 
   OR s.name LIKE '%wait%' 
   OR s.name LIKE '%block%'
   OR s.name LIKE '%sleep%'
GROUP BY s.name
ORDER BY total_duration_ms DESC;
```

#### 6. Frame Rate Analysis (for graphics applications)

**Frame timing analysis:**
```sql
SELECT 
    AVG(dur) / 1e6 as avg_frame_time_ms,
    1000.0 / (AVG(dur) / 1e6) as avg_fps,
    MIN(dur) / 1e6 as min_frame_time_ms,
    MAX(dur) / 1e6 as max_frame_time_ms,
    COUNT(*) as frame_count
FROM slice
WHERE name LIKE '%frame%' OR name LIKE '%vsync%'
   OR name LIKE '%draw%' OR name LIKE '%render%';
```

### Advanced Query Techniques

#### Using Window Functions
```sql
-- Calculate moving average of CPU usage
SELECT 
    ts / 1e9 as time_seconds,
    cpu,
    dur / 1e6 as duration_ms,
    AVG(dur) OVER (
        PARTITION BY cpu 
        ORDER BY ts 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) / 1e6 as moving_avg_ms
FROM sched
WHERE dur > 0
ORDER BY cpu, ts;
```

#### Correlation Analysis
```sql
-- Find correlation between memory usage and CPU activity
WITH cpu_activity AS (
    SELECT 
        ts / 1e9 as time_seconds,
        SUM(dur) / 1e6 as cpu_usage_ms
    FROM sched
    WHERE ts % 1e9 = 0  -- Sample every second
    GROUP BY ts / 1e9
),
memory_usage AS (
    SELECT 
        ts / 1e9 as time_seconds,
        value as memory_kb
    FROM counter c
    JOIN counter_track ct ON c.track_id = ct.id
    WHERE ct.name = 'mem.rss'
      AND ts % 1e9 = 0  -- Sample every second
)
SELECT 
    c.time_seconds,
    c.cpu_usage_ms,
    m.memory_kb
FROM cpu_activity c
JOIN memory_usage m ON c.time_seconds = m.time_seconds
ORDER BY c.time_seconds;
```

## Interactive UI Analysis

### Timeline Navigation

#### Basic Controls
- **Zoom**: Mouse wheel or `Ctrl` + scroll
- **Pan**: Click and drag on timeline
- **Select**: Click and drag to select time range
- **Reset**: Double-click to reset zoom
- **Keyboard shortcuts**:
  - `W`/`S`: Zoom in/out
  - `A`/`D`: Pan left/right
  - `G`: Go to timestamp
  - `M`: Measure time between events

#### Time Range Selection
```
Timeline Selection Example:
|-------|==========|-------|
0ms    100ms      500ms   1000ms
       ^selected range^

Selected: 400ms duration
Events in range: 1,247
```

### Track Analysis

#### CPU Tracks
- **Scheduling visualization**: Shows when threads run on each CPU core
- **Color coding**: Different colors for different processes/threads
- **Gaps**: Indicate idle time or time spent in kernel
- **Overlays**: CPU frequency, thermal throttling events

#### Thread Tracks
- **Function calls**: Nested rectangles showing call hierarchy
- **Duration**: Rectangle width represents execution time
- **Depth**: Vertical position shows call stack depth
- **Annotations**: Custom events and markers

#### Memory Tracks
- **RSS/VSS**: Physical and virtual memory usage
- **Heap allocations**: Individual allocation events
- **Page faults**: Memory access patterns
- **GC events**: Garbage collection activity

### Event Detail Analysis

#### Slice Properties Panel
When clicking on an event slice, the detail panel shows:

```
Event Details:
├─ Name: "MyFunction"
├─ Duration: 15.2ms
├─ Start Time: 1,245.8ms
├─ End Time: 1,261.0ms
├─ Thread: "MainThread" (TID: 1234)
├─ Process: "MyApp" (PID: 5678)
├─ Category: "blink"
├─ Arguments:
│  ├─ "url": "https://example.com"
│  ├─ "size": 1024
│  └─ "cached": true
└─ Call Stack: [if available]
```

#### Flow Events
Flow events show relationships between events:
- **Producer-Consumer**: Data flow between threads
- **Request-Response**: Network or IPC patterns
- **Dependencies**: Causal relationships

### Visualization Techniques

#### 1. Flame Graphs
Access via: Tracks → CPU Profile → Show flame graph

```
Flame Graph Structure:
┌─────────────────────────────────────────┐ ← Root (100%)
│ main()                                  │
├─────────────────┬───────────────────────┤
│ parseConfig()   │ processData()         │ ← Level 1
│ (15%)           │ (85%)                 │
├────────┬────────┼─────────┬─────────────┤
│readFile│parseXML│loadDB() │computeStats │ ← Level 2
│(5%)    │(10%)   │(45%)    │(40%)        │
└────────┴────────┴─────────┴─────────────┘
```

**Reading Flame Graphs:**
- Width = Time spent in function
- Height = Call stack depth
- Colors = Different functions or modules

#### 2. Aggregated Views

**Time-based Aggregation:**
```sql
-- Create custom aggregated view
SELECT 
    ts - (ts % 10e6) as time_bucket_10ms,
    COUNT(*) as event_count,
    SUM(dur) / 1e6 as total_duration_ms
FROM slice
WHERE name = 'target_function'
GROUP BY time_bucket_10ms
ORDER BY time_bucket_10ms;
```

#### 3. Heatmaps
Create heatmaps for:
- CPU utilization across cores over time
- Memory allocation patterns
- Function call frequency

### Performance Metrics Dashboard

#### Key Performance Indicators (KPIs)

**System Level:**
```sql
-- Overall system utilization
SELECT 
    'CPU Utilization' as metric,
    ROUND(100.0 * SUM(dur) / (MAX(ts) - MIN(ts)), 2) || '%' as value
FROM sched
UNION ALL
SELECT 
    'Context Switches',
    COUNT(*) as value
FROM raw WHERE name = 'sched_switch'
UNION ALL
SELECT 
    'Process Count',
    COUNT(DISTINCT pid)
FROM process;
```

**Application Level:**
```sql
-- Application-specific metrics
SELECT 
    'Frame Rate' as metric,
    ROUND(1000.0 / (AVG(dur) / 1e6), 1) || ' FPS' as value
FROM slice WHERE name LIKE '%frame%'
UNION ALL
SELECT 
    'Average Response Time',
    ROUND(AVG(dur) / 1e6, 2) || ' ms'
FROM slice WHERE name LIKE '%request%'
UNION ALL  
SELECT 
    'Peak Memory Usage',
    ROUND(MAX(value) / 1024, 1) || ' MB'
FROM counter c
JOIN counter_track ct ON c.track_id = ct.id
WHERE ct.name = 'mem.rss';
```

### Custom Analysis Workflows

#### 1. Performance Regression Analysis

**Step 1: Identify regression timeframe**
```sql
-- Find performance degradation point
SELECT 
    ts / 1e9 as time_seconds,
    AVG(dur) / 1e6 as avg_duration_ms
FROM slice
WHERE name = 'critical_function'
GROUP BY ts / 1e9
ORDER BY ts / 1e9;
```

**Step 2: Compare before/after periods**
```sql
-- Compare performance before and after regression
WITH periods AS (
    SELECT 
        CASE 
            WHEN ts < 10e9 THEN 'before'
            ELSE 'after'
        END as period,
        dur
    FROM slice
    WHERE name = 'critical_function'
)
SELECT 
    period,
    COUNT(*) as call_count,
    AVG(dur) / 1e6 as avg_ms,
    MIN(dur) / 1e6 as min_ms,
    MAX(dur) / 1e6 as max_ms,
    PERCENTILE(dur, 0.95) / 1e6 as p95_ms
FROM periods
GROUP BY period;
```

#### 2. Bottleneck Analysis Workflow

**Step 1: Find slowest operations**
```sql
-- Top 10 slowest operations
SELECT 
    name,
    MAX(dur) / 1e6 as max_duration_ms,
    ts / 1e9 as occurrence_time
FROM slice
WHERE dur > 0
ORDER BY dur DESC
LIMIT 10;
```

**Step 2: Analyze call patterns**
```sql
-- Analyze what leads to slow operations
SELECT 
    parent.name as parent_function,
    child.name as slow_function,
    COUNT(*) as occurrence_count,
    AVG(child.dur) / 1e6 as avg_duration_ms
FROM slice parent
JOIN slice child ON parent.id = child.parent_id
WHERE child.dur > 100e6  -- > 100ms
GROUP BY parent.name, child.name
ORDER BY avg_duration_ms DESC;
```

## Practical Examples and Use Cases

### Example 1: Android App Performance Analysis

#### Scenario: Investigating Frame Drops
You have an Android app with stuttering animations and need to identify the cause.

**Step 1: Load the trace**
```bash
# Capture trace on Android device
adb shell perfetto --txt --config - --out /data/misc/perfetto-traces/trace.pb <<EOF
buffers: {
    size_kb: 102400
    fill_policy: RING_BUFFER
}
data_sources: {
    config: {
        name: "linux.ftrace"
        ftrace_config: {
            ftrace_events: "sched/sched_switch"
            ftrace_events: "graphics/gpu_freq"
            ftrace_events: "power/cpu_frequency"
        }
    }
}
duration_ms: 30000
EOF

# Pull trace file
adb pull /data/misc/perfetto-traces/trace.pb ./app_trace.pb
```

**Step 2: Analyze frame timing**
```sql
-- Find frame drops (>16.67ms for 60fps)
SELECT 
    slice.name,
    slice.ts / 1e6 as start_time_ms,
    slice.dur / 1e6 as duration_ms,
    thread.name as thread_name
FROM slice
JOIN thread_track ON slice.track_id = thread_track.id  
JOIN thread ON thread_track.utid = t.utid
WHERE slice.name LIKE '%doFrame%' 
   OR slice.name LIKE '%Choreographer%'
   AND slice.dur > 16.67e6  -- Frame drops
ORDER BY slice.dur DESC;
```

**Step 3: Identify root cause**
```sql
-- Find what's blocking the UI thread during frame drops
WITH long_frames AS (
    SELECT slice.id, slice.ts, slice.ts + slice.dur as end_ts
    FROM slice
    JOIN thread_track ON slice.track_id = thread_track.id
    JOIN thread ON thread_track.utid = t.utid
    WHERE slice.name LIKE '%doFrame%' 
      AND slice.dur > 16.67e6
)
SELECT 
    blocking.name as blocking_operation,
    blocking.dur / 1e6 as duration_ms,
    COUNT(*) as occurrence_count
FROM long_frames
JOIN slice blocking ON (
    blocking.ts >= long_frames.ts 
    AND blocking.ts < long_frames.end_ts
)
JOIN thread_track bt ON blocking.track_id = bt.id
JOIN thread bt_thread ON bt.utid = bt_thread.utid
WHERE bt_thread.name = 'main' 
  AND blocking.dur > 5e6  -- Operations > 5ms
GROUP BY blocking.name
ORDER BY occurrence_count DESC;
```

### Example 2: Web Application Memory Leak Detection

#### Scenario: Browser tab consuming increasing memory

**Step 1: Memory usage trend analysis**
```sql
-- Track memory growth over time
SELECT 
    ts / 1e9 as time_seconds,
    value / 1024 / 1024 as memory_mb,
    LAG(value) OVER (ORDER BY ts) as prev_value,
    (value - LAG(value) OVER (ORDER BY ts)) / 1024 as delta_kb
FROM counter c
JOIN counter_track ct ON c.track_id = ct.id
WHERE ct.name = 'mem.rss'
ORDER BY ts;
```

**Step 2: Identify allocation patterns**
```sql
-- Find frequent allocation sources
SELECT 
    slice.name as allocator,
    COUNT(*) as allocation_count,
    SUM(CAST(arg.value AS INTEGER)) as total_bytes_allocated
FROM slice
JOIN args arg ON slice.arg_set_id = arg.arg_set_id
WHERE slice.name LIKE '%alloc%' 
  AND arg.key = 'size'
GROUP BY slice.name
ORDER BY total_bytes_allocated DESC;
```

### Example 3: CPU Performance Optimization

#### Scenario: High CPU usage investigation

**Step 1: CPU utilization by process**
```sql
-- CPU time per process
SELECT 
    process.name,
    SUM(sched.dur) / 1e9 as cpu_seconds,
    100.0 * SUM(sched.dur) / (
        SELECT MAX(ts) - MIN(ts) FROM sched
    ) as cpu_percent
FROM sched
JOIN thread ON sched.utid = thread.utid
JOIN process ON thread.upid = process.upid
GROUP BY process.upid
ORDER BY cpu_seconds DESC;
```

**Step 2: Hot functions analysis**
```sql
-- Most CPU-intensive functions
SELECT 
    slice.name,
    COUNT(*) as call_count,
    SUM(slice.dur) / 1e6 as total_ms,
    AVG(slice.dur) / 1e6 as avg_ms,
    MAX(slice.dur) / 1e6 as max_ms
FROM slice
JOIN thread_track ON slice.track_id = thread_track.id
JOIN thread ON thread_track.utid = t.utid
JOIN process ON thread.upid = process.upid
WHERE process.name = 'target_process'
  AND slice.dur > 0
GROUP BY slice.name
ORDER BY total_ms DESC
LIMIT 20;
```

### Example 4: I/O Performance Analysis

#### Scenario: Slow file operations

**Step 1: File I/O timing**
```sql
-- File operation analysis
SELECT 
    CASE 
        WHEN slice.name LIKE '%read%' THEN 'Read'
        WHEN slice.name LIKE '%write%' THEN 'Write' 
        WHEN slice.name LIKE '%open%' THEN 'Open'
        WHEN slice.name LIKE '%close%' THEN 'Close'
        ELSE 'Other'
    END as operation_type,
    COUNT(*) as count,
    AVG(slice.dur) / 1e6 as avg_duration_ms,
    SUM(slice.dur) / 1e6 as total_duration_ms
FROM slice
WHERE slice.name LIKE '%file%' 
   OR slice.name LIKE '%read%'
   OR slice.name LIKE '%write%'
GROUP BY operation_type
ORDER BY total_duration_ms DESC;
```

**Step 2: I/O wait time analysis**
```sql
-- Find blocking I/O operations
SELECT 
    slice.name,
    slice.dur / 1e6 as duration_ms,
    args.value as filename
FROM slice
JOIN args ON slice.arg_set_id = args.arg_set_id
WHERE slice.dur > 10e6  -- > 10ms
  AND (slice.name LIKE '%read%' OR slice.name LIKE '%write%')
  AND args.key = 'filename'
ORDER BY slice.dur DESC;
```

## Learning Objectives

By the end of this section, you should be able to:

### Core Competencies
- **Navigate the Perfetto UI** efficiently with keyboard shortcuts and timeline controls
- **Write advanced SQL queries** to extract meaningful insights from trace data
- **Identify performance bottlenecks** using flame graphs, timeline analysis, and metrics
- **Analyze system behavior** across CPU, memory, I/O, and application-level events
- **Create custom dashboards** and derived metrics for specific analysis needs
- **Troubleshoot common issues** with trace analysis and query optimization

### Practical Skills
- **Load and visualize** different trace formats (Android, Chrome, custom)
- **Perform root cause analysis** for performance regressions and anomalies
- **Generate automated reports** and performance summaries
- **Integrate trace analysis** into development and CI/CD workflows
- **Share findings** effectively using permalinks and exported visualizations

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ **Basic Navigation**
  - Load a trace file and navigate the timeline
  - Use keyboard shortcuts for efficient navigation
  - Select time ranges and examine event details

□ **SQL Querying** 
  - Write queries to find processes, threads, and function calls
  - Analyze CPU scheduling and memory usage patterns
  - Create aggregated views and statistical summaries

□ **Performance Analysis**
  - Identify the top CPU-consuming functions
  - Detect memory leaks and allocation patterns
  - Find I/O bottlenecks and blocking operations
  - Analyze frame rate and responsiveness issues

□ **Advanced Techniques**
  - Create custom metrics and derived measurements
  - Perform correlation analysis between different metrics
  - Use window functions for time-series analysis
  - Implement anomaly detection queries

□ **Troubleshooting**
  - Optimize queries for large trace files
  - Handle missing events and incomplete traces
  - Export and share analysis results

### Hands-on Exercises

#### Exercise 1: Basic Trace Analysis
**Objective**: Analyze a sample Android app trace
```sql
-- TODO: Complete these queries
-- 1. Find the top 5 most expensive functions
SELECT ___

-- 2. Calculate average frame time
SELECT ___

-- 3. Identify memory allocation hotspots
SELECT ___
```

#### Exercise 2: Performance Regression Detection
**Scenario**: Compare two traces (before/after a code change)
```sql
-- TODO: Write queries to:
-- 1. Compare function performance between traces
-- 2. Identify new performance bottlenecks
-- 3. Generate a performance regression report
```

#### Exercise 3: Custom Dashboard Creation
**Task**: Create a performance dashboard with:
- CPU utilization summary
- Memory usage trends
- Frame rate analysis
- Top bottlenecks list

#### Exercise 4: Real-world Investigation
**Scenario**: Your app has suddenly become slow
**Steps**:
1. Capture a trace during the slow period
2. Identify the root cause using SQL analysis
3. Create a timeline view showing the problem
4. Generate a report with findings and recommendations

## Study Materials and Resources

### Essential Documentation
- **Perfetto Documentation**: https://perfetto.dev/docs/
- **SQL Reference**: https://perfetto.dev/docs/analysis/sql-tables
- **Trace Analysis Guide**: https://perfetto.dev/docs/analysis/trace-analysis

### Interactive Tutorials
- **Perfetto Codelabs**: https://perfetto.dev/docs/quickstart/
- **Android Performance**: https://developer.android.com/topic/performance/tracing
- **Chrome Tracing**: https://www.chromium.org/developers/how-tos/trace-event-profiling-tool

### Video Resources
- "Advanced Trace Analysis with Perfetto" - Google I/O
- "Performance Debugging with Perfetto" - Android Dev Summit  
- "Perfetto UI Deep Dive" - Chrome Dev Summit

### Sample Traces for Practice
```bash
# Download sample traces
wget https://storage.googleapis.com/perfetto-misc/example_android_trace.pb
wget https://storage.googleapis.com/perfetto-misc/example_chrome_trace.json
wget https://storage.googleapis.com/perfetto-misc/example_system_trace.perfetto
```

### Recommended Reading
- **"Systems Performance" by Brendan Gregg** - Chapter 12 (Tracing)
- **"High Performance Browser Networking"** - Performance analysis techniques
- **Android Performance Patterns** - Google's performance best practices

### Development Environment Setup

#### Local Perfetto Installation
```bash
# Clone and build Perfetto
git clone https://github.com/google/perfetto.git
cd perfetto
./tools/install-build-deps
./tools/build_ui

# Start local server
./out/ui/ui --http-port 10000

# Build trace processor
ninja -C out trace_processor
```

#### Browser Setup
```javascript
// Perfetto UI bookmarklet for quick access
javascript:(function(){window.open('https://ui.perfetto.dev/','perfetto','width=1200,height=800')})();
```

#### VS Code Extensions
- **Perfetto Trace Viewer**: View traces directly in VS Code
- **SQL Tools**: For writing and testing SQL queries

### Practice Scenarios

#### Scenario 1: Mobile App Optimization
**Context**: Android app with stuttering animations
**Goal**: Identify and fix frame drops
**Skills**: Frame analysis, UI thread investigation, GPU profiling

#### Scenario 2: Web Application Performance
**Context**: React app with slow page loads
**Goal**: Optimize loading performance
**Skills**: Network analysis, JavaScript profiling, resource loading

#### Scenario 3: System Service Analysis  
**Context**: Background service consuming CPU
**Goal**: Reduce system resource usage
**Skills**: Process analysis, system call tracing, scheduling

#### Scenario 4: Memory Leak Investigation
**Context**: Long-running application with growing memory
**Goal**: Find and fix memory leaks
**Skills**: Memory profiling, allocation tracking, GC analysis

### Assessment Questions

#### Knowledge Check
1. **What's the difference between slice and sched tables in Perfetto?**
2. **How do you identify the critical path in a complex operation?**
3. **What SQL techniques help analyze time-series data?**
4. **How can you correlate events across different processes?**

#### Practical Challenges
5. **Write a query to find all functions that take longer than their historical average**
6. **Create a visualization showing CPU usage correlation with memory allocations**
7. **Implement an automated performance regression detector**
8. **Design a real-time performance monitoring dashboard**

### Performance Analysis Patterns

#### Common Query Patterns
```sql
-- Template: Top-N analysis
SELECT name, COUNT(*), SUM(dur), AVG(dur)
FROM slice
WHERE [conditions]
GROUP BY name
ORDER BY [metric] DESC
LIMIT N;

-- Template: Time-series analysis
SELECT 
    ts - (ts % [bucket_size]) as time_bucket,
    [aggregation_function]([metric])
FROM [table]
GROUP BY time_bucket
ORDER BY time_bucket;

-- Template: Correlation analysis
WITH metric1 AS (...), metric2 AS (...)
SELECT m1.time, m1.value, m2.value
FROM metric1 m1
JOIN metric2 m2 ON m1.time = m2.time;
```

#### Analysis Workflows
1. **Initial Overview**: System stats, process list, timeline overview
2. **Hotspot Identification**: CPU/memory/I/O bottlenecks
3. **Deep Dive**: Function-level analysis, call patterns
4. **Root Cause**: Dependencies, blocking operations, resource contention
5. **Validation**: Before/after comparison, regression testing

## Next Steps

### Advanced Topics to Explore
- **Custom Tracing**: Instrumenting your own applications
- **Trace Processor Metrics**: Creating reusable analysis modules
- **Machine Learning**: Automated anomaly detection in traces
- **Real-time Analysis**: Live performance monitoring
