# Perfetto Fundamentals

## Overview
Learn the architecture and core concepts of Perfetto, a performance tracing platform for Android and Linux.

## Example: Simple Trace Config (pseudo-code)
```json
{
  "traceConfig": {
    "buffers": [{"size_kb": 10240, "fill_policy": "ring"}],
    "data_sources": [{"config": {"name": "linux.ftrace"}}]
  }
}
```

*Use this config to start a basic Perfetto trace on Linux.*
