# Bazel Query and Debugging

## Overview
This section introduces Bazel's query language, build analysis, and performance profiling tools.

## Example: Querying Dependencies
```sh
bazel query 'deps(//main:all)'
```

This command lists all dependencies of the //main:all target.
