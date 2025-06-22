# Bazel for Web Development

## Overview
This section covers Bazel's support for JavaScript/TypeScript, npm/yarn integration, and web bundling.

## Example: TypeScript Binary with Bazel
```python
# BUILD file
ts_library(
    name = "app",
    srcs = ["app.ts"],
)
```
```typescript
// app.ts
console.log("Hello from Bazel TypeScript!");
```
