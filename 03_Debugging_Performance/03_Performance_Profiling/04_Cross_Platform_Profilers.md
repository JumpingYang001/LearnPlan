# Cross-Platform Profiling Tools

Overview of Intel VTune, AMD μProf, Google Performance Tools, and Tracy Profiler.

## Intel VTune Example
```sh
# Collect hotspots
tune -collect hotspots -result-dir r001 ./myprog
```

## Google gperftools Example
```c
#include <gperftools/profiler.h>
int main() {
    ProfilerStart("cpu.prof");
    // ... code to profile ...
    ProfilerStop();
    return 0;
}
```

## Tracy Profiler Example
- Integrate Tracy client library in your C++ project
- Use `ZoneScoped` macros to mark code regions
