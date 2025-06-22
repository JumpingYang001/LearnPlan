# Performance Optimization with Perfetto

## Overview
Identify and resolve performance bottlenecks using Perfetto's analysis tools for CPU, GPU, and memory.

## Example: Analyzing CPU Usage (pseudo-code)
```sql
SELECT ts, dur, cpu FROM sched WHERE utid = 1;
```

*Use this query to analyze CPU usage over time in Perfetto.*
