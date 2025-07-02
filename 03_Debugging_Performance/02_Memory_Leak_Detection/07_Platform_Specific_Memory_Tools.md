# Platform-Specific Memory Tools

*Duration: 2-3 days*

Memory debugging and analysis require different tools and approaches depending on the operating system. This guide covers the most effective platform-specific tools for memory leak detection, performance analysis, and debugging.

## Overview of Platform Differences

| Feature | Linux | Windows | macOS |
|---------|-------|---------|-------|
| **Built-in Tools** | /proc filesystem, valgrind | Task Manager, PerfView | Activity Monitor, leaks |
| **Advanced Tools** | AddressSanitizer, Massif | VMMap, Application Verifier | Instruments, vmmap |
| **Kernel Integration** | Deep /proc integration | ETW (Event Tracing) | DTrace integration |
| **IDE Support** | GDB, extensive CLI tools | Visual Studio Debugger | Xcode Instruments |

---

## Linux Memory Analysis Tools

Linux provides the most comprehensive set of built-in and third-party memory analysis tools.

### Built-in System Tools

#### 1. /proc Filesystem
The `/proc` filesystem provides real-time access to kernel data structures and process information.

**System-wide Memory Information:**
```bash
# Get overall system memory statistics
cat /proc/meminfo

# Sample output explanation:
# MemTotal:       16384000 kB    # Total physical memory
# MemFree:         2048000 kB    # Free memory
# MemAvailable:   12288000 kB    # Available for new processes
# Buffers:          512000 kB    # Temporary storage for raw disk blocks
# Cached:          8192000 kB    # Page cache and slabs
# SwapTotal:       8192000 kB    # Total swap space
# SwapFree:        8000000 kB    # Free swap space

# Monitor memory usage in real-time
watch -n 1 'cat /proc/meminfo | head -10'
```

**Process-specific Memory Analysis:**
```bash
# Get memory map for a specific process
pmap <pid>

# Detailed memory mapping with permissions
pmap -x <pid>

# Example output interpretation:
# Address   Kbytes RSS   Dirty Mode  Mapping
# 00400000    1024 1024    0   r-x-- /usr/bin/myprogram
# 00600000      16   16   16   rw--- /usr/bin/myprogram
# 01000000   65536 32768 32768 rw---   [ heap ]

# Process memory status
cat /proc/<pid>/status | grep -E "(VmSize|VmRSS|VmHWM|VmData|VmStk)"

# Memory maps with detailed information
cat /proc/<pid>/maps
```

#### 2. Advanced Process Memory Analysis
```bash
# Real-time memory monitoring
top -p <pid>                    # Basic monitoring
htop -p <pid>                   # Enhanced monitoring

# Memory usage by memory type
cat /proc/<pid>/smaps           # Detailed memory segments
cat /proc/<pid>/smaps_rollup    # Summary of smaps

# Monitor memory allocations
strace -e trace=memory ./program 2>&1 | grep -E "(mmap|brk|munmap)"
```

### Valgrind Memory Tools

#### Memcheck - Memory Error Detector
```bash
# Basic memory leak detection
valgrind --tool=memcheck --leak-check=full ./program

# Comprehensive memory analysis
valgrind --tool=memcheck \
         --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         --log-file=memcheck.log \
         ./program

# Example C program for testing:
cat > test_leaks.c << 'EOF'
#include <stdlib.h>
#include <string.h>

int main() {
    // Memory leak - never freed
    char *leak = malloc(100);
    strcpy(leak, "This memory will leak");
    
    // Use after free
    char *ptr = malloc(50);
    free(ptr);
    strcpy(ptr, "Use after free");  // Error!
    
    // Double free
    char *double_free = malloc(30);
    free(double_free);
    free(double_free);  // Error!
    
    return 0;
}
EOF

gcc -g -o test_leaks test_leaks.c
valgrind --tool=memcheck --leak-check=full ./test_leaks
```

#### Massif - Heap Profiler
```bash
# Profile heap usage over time
valgrind --tool=massif ./program

# Analyze massif output
ms_print massif.out.<pid> > heap_profile.txt

# Generate graphical representation
massif-visualizer massif.out.<pid>

# Example with detailed heap snapshots
valgrind --tool=massif \
         --detailed-freq=1 \
         --max-snapshots=200 \
         --time-unit=B \
         ./program
```

### AddressSanitizer (ASan)
```bash
# Compile with AddressSanitizer
gcc -fsanitize=address -g -o program program.c

# Advanced ASan options
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:fast_unwind_on_malloc=0"
./program

# Example ASan-enabled program:
cat > asan_test.c << 'EOF'
#include <stdlib.h>
#include <string.h>

int main() {
    char *buffer = malloc(10);
    
    // Buffer overflow - ASan will detect this
    strcpy(buffer, "This string is too long for buffer");
    
    free(buffer);
    return 0;
}
EOF

gcc -fsanitize=address -g -o asan_test asan_test.c
./asan_test
```

### Linux Memory Profiling Scripts
```bash
#!/bin/bash
# memory_monitor.sh - Comprehensive memory monitoring script

PID=$1
if [ -z "$PID" ]; then
    echo "Usage: $0 <pid>"
    exit 1
fi

echo "Memory monitoring for PID: $PID"
echo "================================"

while kill -0 $PID 2>/dev/null; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Get memory stats
    VMSIZE=$(grep VmSize /proc/$PID/status | awk '{print $2}')
    VMRSS=$(grep VmRSS /proc/$PID/status | awk '{print $2}')
    VMHWM=$(grep VmHWM /proc/$PID/status | awk '{print $2}')
    
    echo "$TIMESTAMP VSZ:${VMSIZE}kB RSS:${VMRSS}kB Peak:${VMHWM}kB"
    
    sleep 1
done
```

---

## Windows Memory Analysis Tools

Windows provides both built-in tools and advanced debugging utilities for memory analysis.

### Built-in Windows Tools

#### 1. Task Manager and Resource Monitor
```powershell
# PowerShell memory analysis
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10 Name, WorkingSet, VirtualMemorySize

# Get detailed process memory information
Get-Process "notepad" | Select-Object *memory*

# Monitor memory usage for specific process
while ($true) {
    $proc = Get-Process "myapp" -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "$(Get-Date): Working Set: $($proc.WorkingSet64/1MB) MB, Virtual: $($proc.VirtualMemorySize64/1MB) MB"
    }
    Start-Sleep 1
}
```

#### 2. Performance Toolkit (WPT)
```batch
rem Install Windows Performance Toolkit
rem Available as part of Windows SDK

rem Capture ETW trace for memory analysis
wpr -start CPU -start Heap

rem Run your application
rem ...

rem Stop tracing
wpr -stop memory_trace.etl

rem Analyze with Windows Performance Analyzer
wpa memory_trace.etl
```

### VMMap - Virtual Memory Usage Analysis

**Download from Microsoft Sysinternals**

```batch
rem Command line usage
vmmap.exe -p <pid>

rem Export to file
vmmap.exe -p <pid> -o output.txt

rem Real-time monitoring
vmmap.exe <executable>
```

**Example VMMap Analysis:**
```cpp
// test_vmmap.cpp - Program to analyze with VMMap
#include <windows.h>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Process ID: " << GetCurrentProcessId() << std::endl;
    std::cout << "Press Enter to start allocating memory..." << std::endl;
    std::cin.get();
    
    std::vector<void*> allocations;
    
    // Allocate memory in chunks
    for (int i = 0; i < 100; i++) {
        void* ptr = VirtualAlloc(NULL, 1024 * 1024, // 1MB
                               MEM_COMMIT | MEM_RESERVE, 
                               PAGE_READWRITE);
        if (ptr) {
            allocations.push_back(ptr);
            std::cout << "Allocated 1MB at address: " << ptr << std::endl;
        }
        Sleep(1000); // Wait 1 second between allocations
    }
    
    std::cout << "Press Enter to free memory..." << std::endl;
    std::cin.get();
    
    // Free all allocations
    for (void* ptr : allocations) {
        VirtualFree(ptr, 0, MEM_RELEASE);
    }
    
    std::cout << "Memory freed. Press Enter to exit..." << std::endl;
    std::cin.get();
    
    return 0;
}
```

### Application Verifier
```batch
rem Enable Application Verifier for heap checking
appverif.exe

rem Command line setup
appverif -enable Heaps -for myapp.exe

rem Run your application - App Verifier will detect heap corruption
myapp.exe

rem View logs in Event Viewer or debugger
```

### Windows CRT Debug Heap
```cpp
// debug_heap.cpp - Using Windows CRT debug heap
#ifdef _DEBUG
#include <crtdbg.h>
#endif

#include <iostream>
#include <cstdlib>

int main() {
#ifdef _DEBUG
    // Enable debug heap
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    
    // Set breakpoint on specific allocation number
    // _CrtSetBreakAlloc(123);
#endif
    
    // Intentional memory leak for demonstration
    char* leak = (char*)malloc(100);
    strcpy_s(leak, 100, "This will leak");
    
    // Don't free 'leak' - debug heap will report this at exit
    
    std::cout << "Check debug output for memory leak report" << std::endl;
    
#ifdef _DEBUG
    // Generate memory leak report
    _CrtDumpMemoryLeaks();
#endif
    
    return 0;
}

// Compile with: cl /MDd /Zi debug_heap.cpp
```

### PerfView - .NET Memory Analysis
```batch
rem Download PerfView from Microsoft
rem Capture .NET heap trace
PerfView.exe /DataFile:heap_trace.etl /Zip:false collect

rem Analyze managed heap
PerfView.exe heap_trace.etl
```

### Advanced Windows Memory Debugging
```cpp
// windows_heap_debug.cpp - Advanced heap debugging
#include <windows.h>
#include <iostream>
#include <psapi.h>

void PrintMemoryInfo() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), 
                        (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    
    std::cout << "Working Set Size: " << pmc.WorkingSetSize / 1024 << " KB" << std::endl;
    std::cout << "Peak Working Set: " << pmc.PeakWorkingSetSize / 1024 << " KB" << std::endl;
    std::cout << "Private Usage: " << pmc.PrivateUsage / 1024 << " KB" << std::endl;
    std::cout << "Pagefile Usage: " << pmc.PagefileUsage / 1024 << " KB" << std::endl;
}

int main() {
    std::cout << "Initial memory state:" << std::endl;
    PrintMemoryInfo();
    
    // Allocate large chunk of memory
    const size_t size = 100 * 1024 * 1024; // 100MB
    void* ptr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    
    if (ptr) {
        std::cout << "\nAfter allocating 100MB:" << std::endl;
        PrintMemoryInfo();
        
        // Touch the memory to ensure it's committed
        memset(ptr, 0x42, size);
        
        std::cout << "\nAfter touching memory:" << std::endl;
        PrintMemoryInfo();
        
        VirtualFree(ptr, 0, MEM_RELEASE);
        
        std::cout << "\nAfter freeing memory:" << std::endl;
        PrintMemoryInfo();
    }
    
    return 0;
}
```

---

## macOS Memory Analysis Tools

macOS provides sophisticated memory analysis tools through both command-line utilities and Xcode Instruments.

### Built-in macOS Tools

#### 1. leaks - Memory Leak Detection
```bash
# Basic leak detection for running process
leaks <pid>

# Continuous monitoring
leaks <pid> -atExit

# Export results to file
leaks <pid> > leak_report.txt

# Example with detailed output
leaks -outputGraph leak_graph.dot <pid>

# Monitor specific malloc zones
leaks -exclude <zone_name> <pid>

# Example program to test leaks:
cat > test_macos_leaks.c << 'EOF'
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main() {
    printf("Process ID: %d\n", getpid());
    printf("Press Enter to create memory leaks...\n");
    getchar();
    
    // Create intentional memory leaks
    for (int i = 0; i < 10; i++) {
        char *leak = malloc(1024);
        sprintf(leak, "Leak #%d", i);
        // Intentionally not freeing
    }
    
    printf("Leaks created. Run: leaks %d\n", getpid());
    printf("Press Enter to exit...\n");
    getchar();
    
    return 0;
}
EOF

gcc -g -o test_macos_leaks test_macos_leaks.c
./test_macos_leaks &
leaks $!
```

#### 2. vmmap - Virtual Memory Analysis
```bash
# Analyze virtual memory layout
vmmap <pid>

# Detailed heap analysis
vmmap -verbose <pid>

# Filter by memory type
vmmap <pid> | grep MALLOC

# Export detailed report
vmmap -interleaved <pid> > vmmap_report.txt

# Monitor memory regions
vmmap -pages <pid>

# Example interpretation:
# REGION TYPE                      VIRTUAL
# ===========                      =======
# STACK GUARD                        56.0M
# Stack                               8192K
# __DATA                              1024K
# __TEXT                               512K
# MALLOC_LARGE                       100.0M
# MALLOC_SMALL                        16.0M
```

#### 3. Activity Monitor via Command Line
```bash
# Get process memory information
ps -o pid,rss,vsz,command -p <pid>

# Monitor memory usage continuously
while true; do
    ps -o pid,rss,vsz,command -p <pid>
    sleep 1
done

# System memory overview
vm_stat

# Sample vm_stat output:
# Pages free:                    1000000
# Pages active:                  2000000
# Pages inactive:                 500000
# Pages speculative:              100000
# Pages wired down:               800000
```

### Xcode Instruments

#### 1. Allocations Instrument
```objc
// test_instruments.m - Program for Instruments analysis
#import <Foundation/Foundation.h>

@interface MemoryTestClass : NSObject
@property (strong, nonatomic) NSMutableArray *dataArray;
@end

@implementation MemoryTestClass

- (instancetype)init {
    self = [super init];
    if (self) {
        _dataArray = [[NSMutableArray alloc] init];
    }
    return self;
}

- (void)addLargeObjects {
    for (int i = 0; i < 1000; i++) {
        NSData *data = [NSData dataWithLength:1024 * 1024]; // 1MB each
        [self.dataArray addObject:data];
    }
}

@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"Starting memory allocation test...");
        
        MemoryTestClass *tester = [[MemoryTestClass alloc] init];
        
        // Create memory pressure
        [tester addLargeObjects];
        
        NSLog(@"Allocated large objects. Check Instruments.");
        
        // Keep running for analysis
        [[NSRunLoop currentRunLoop] run];
    }
    return 0;
}

// Compile with: clang -framework Foundation test_instruments.m -o test_instruments
// Run with Instruments: instruments -t Allocations ./test_instruments
```

#### 2. Leaks Instrument
```bash
# Run leaks instrument from command line
instruments -t Leaks ./your_application

# Generate report
instruments -t Leaks -D leak_trace.trace ./your_application

# Analyze trace file
instruments -s symbols leak_trace.trace
```

### Advanced macOS Memory Debugging

#### 1. MallocDebug (Legacy) and Modern Alternatives
```bash
# Set malloc debugging environment variables
export MallocScribble=1          # Fill freed memory with 0x55
export MallocPreScribble=1       # Fill allocated memory with 0xAA
export MallocGuardEdges=1        # Add guard pages around allocations
export MallocStackLogging=1      # Enable stack logging

# Run your application with malloc debugging
./your_application

# Check for heap corruption
heap <pid>

# Sample heap output analysis:
# malloc[12345]: *** error for object 0x7f8b8c000000: incorrect checksum
# malloc[12345]: *** set a breakpoint in malloc_error_break to debug
```

#### 2. DTrace Memory Probes
```bash
# Monitor malloc/free operations
sudo dtrace -n 'pid<pid>::malloc:entry { printf("malloc(%d)", arg0); }'

# Track memory allocations by size
sudo dtrace -n '
    pid<pid>::malloc:entry 
    { 
        @allocs[arg0] = count(); 
    } 
    END 
    { 
        printa(@allocs); 
    }'

# Monitor for large allocations
sudo dtrace -n '
    pid<pid>::malloc:entry 
    /arg0 > 1024*1024/ 
    { 
        printf("Large allocation: %d bytes\n", arg0); 
        ustack(10); 
    }'
```

#### 3. Sample Tools
```bash
# Profile memory allocations
sample <pid> -file memory_sample.txt

# Generate call graph
sample <pid> -mayDie -file sample.txt
```

### macOS Memory Monitoring Scripts
```bash
#!/bin/bash
# macos_memory_monitor.sh - Comprehensive memory monitoring

PID=$1
if [ -z "$PID" ]; then
    echo "Usage: $0 <pid>"
    exit 1
fi

echo "macOS Memory Monitor for PID: $PID"
echo "=================================="

# Create monitoring loop
while kill -0 $PID 2>/dev/null; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Get process memory info
    MEMORY_INFO=$(ps -o rss,vsz -p $PID | tail -n 1)
    RSS=$(echo $MEMORY_INFO | awk '{print $1}')
    VSZ=$(echo $MEMORY_INFO | awk '{print $2}')
    
    # Get system memory info
    VM_STAT_OUTPUT=$(vm_stat)
    FREE_PAGES=$(echo "$VM_STAT_OUTPUT" | grep "Pages free" | awk '{print $3}' | tr -d '.')
    ACTIVE_PAGES=$(echo "$VM_STAT_OUTPUT" | grep "Pages active" | awk '{print $3}' | tr -d '.')
    
    # Calculate memory in MB (assuming 4KB pages)
    RSS_MB=$((RSS / 1024))
    VSZ_MB=$((VSZ / 1024))
    FREE_MB=$((FREE_PAGES * 4 / 1024))
    ACTIVE_MB=$((ACTIVE_PAGES * 4 / 1024))
    
    echo "$TIMESTAMP Process RSS:${RSS_MB}MB VSZ:${VSZ_MB}MB System Free:${FREE_MB}MB Active:${ACTIVE_MB}MB"
    
    sleep 2
done

echo "Process $PID has terminated."
```

---

## Cross-Platform Memory Analysis

### Universal Tools and Techniques

#### 1. AddressSanitizer (All Platforms)
```cpp
// cross_platform_asan.cpp - Works on Linux, Windows, macOS
#include <cstdlib>
#include <cstring>
#include <iostream>

int main() {
    std::cout << "Testing AddressSanitizer across platforms..." << std::endl;
    
    // Test 1: Heap buffer overflow
    char* buffer = (char*)malloc(10);
    strcpy(buffer, "This string is way too long!"); // Buffer overflow
    
    // Test 2: Use after free
    free(buffer);
    buffer[0] = 'X'; // Use after free
    
    return 0;
}

// Compile on any platform:
// Linux:   g++ -fsanitize=address -g cross_platform_asan.cpp
// Windows: clang++ -fsanitize=address -g cross_platform_asan.cpp
// macOS:   clang++ -fsanitize=address -g cross_platform_asan.cpp
```

#### 2. Portable Memory Tracking Class
```cpp
// memory_tracker.hpp - Cross-platform memory tracking
#pragma once
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <cstdlib>

class MemoryTracker {
private:
    struct AllocationInfo {
        size_t size;
        const char* file;
        int line;
        const char* function;
    };
    
    std::unordered_map<void*, AllocationInfo> allocations_;
    std::mutex mutex_;
    size_t total_allocated_ = 0;
    size_t peak_allocated_ = 0;
    
public:
    static MemoryTracker& instance() {
        static MemoryTracker tracker;
        return tracker;
    }
    
    void* allocate(size_t size, const char* file, int line, const char* function) {
        void* ptr = std::malloc(size);
        if (ptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            allocations_[ptr] = {size, file, line, function};
            total_allocated_ += size;
            if (total_allocated_ > peak_allocated_) {
                peak_allocated_ = total_allocated_;
            }
        }
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (ptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                total_allocated_ -= it->second.size;
                allocations_.erase(it);
            }
            std::free(ptr);
        }
    }
    
    void report_leaks() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!allocations_.empty()) {
            std::cout << "MEMORY LEAKS DETECTED:" << std::endl;
            for (const auto& [ptr, info] : allocations_) {
                std::cout << "  " << ptr << " (" << info.size << " bytes) "
                         << "allocated at " << info.file << ":" << info.line 
                         << " in " << info.function << std::endl;
            }
        }
        std::cout << "Peak memory usage: " << peak_allocated_ << " bytes" << std::endl;
    }
};

// Convenience macros
#define TRACKED_MALLOC(size) MemoryTracker::instance().allocate(size, __FILE__, __LINE__, __FUNCTION__)
#define TRACKED_FREE(ptr) MemoryTracker::instance().deallocate(ptr)

// Example usage:
#include "memory_tracker.hpp"

int main() {
    // Use tracked allocation
    char* ptr1 = (char*)TRACKED_MALLOC(100);
    char* ptr2 = (char*)TRACKED_MALLOC(200);
    
    TRACKED_FREE(ptr1);
    // ptr2 intentionally not freed to demonstrate leak detection
    
    MemoryTracker::instance().report_leaks();
    return 0;
}
```

#### 3. CMake Cross-Platform Memory Debugging Setup
```cmake
# CMakeLists.txt - Cross-platform memory debugging setup
cmake_minimum_required(VERSION 3.16)
project(MemoryDebugging)

set(CMAKE_CXX_STANDARD 17)

# Platform-specific memory debugging options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(UNIX AND NOT APPLE)
        # Linux: AddressSanitizer + Valgrind support
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")
        
        # Find Valgrind
        find_program(VALGRIND_EXECUTABLE valgrind)
        if(VALGRIND_EXECUTABLE)
            message(STATUS "Valgrind found: ${VALGRIND_EXECUTABLE}")
        endif()
        
    elseif(APPLE)
        # macOS: AddressSanitizer + Instruments support
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")
        
        # Enable malloc debugging
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMALLOC_DEBUG")
        
    elseif(WIN32)
        # Windows: CRT Debug Heap + AddressSanitizer (if available)
        if(MSVC)
            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
            add_definitions(-D_CRTDBG_MAP_ALLOC)
        else()
            # MinGW or Clang on Windows
            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
        endif()
    endif()
endif()

# Add custom targets for memory testing
add_custom_target(memcheck
    COMMAND ${CMAKE_COMMAND} -E echo "Running memory checks..."
    DEPENDS your_executable
)

if(UNIX AND NOT APPLE AND VALGRIND_EXECUTABLE)
    add_custom_target(valgrind
        COMMAND ${VALGRIND_EXECUTABLE} --tool=memcheck --leak-check=full $<TARGET_FILE:your_executable>
        DEPENDS your_executable
        COMMENT "Running Valgrind memory check"
    )
endif()

if(APPLE)
    add_custom_target(leaks
        COMMAND leaks --atExit -- $<TARGET_FILE:your_executable>
        DEPENDS your_executable
        COMMENT "Running macOS leaks check"
    )
endif()
```

---

## Platform Comparison and Best Practices

### Tool Selection Matrix

| Use Case | Linux | Windows | macOS |
|----------|-------|---------|-------|
| **Basic Leak Detection** | Valgrind Memcheck | CRT Debug Heap | leaks command |
| **Heap Profiling** | Massif, Heaptrack | VMMap, PerfView | Instruments Allocations |
| **Real-time Monitoring** | htop, /proc | Resource Monitor | Activity Monitor |
| **Advanced Analysis** | AddressSanitizer | Application Verifier | Instruments Suite |
| **Production Debugging** | perf, SystemTap | ETW, WPA | DTrace |

### Cross-Platform Best Practices

#### 1. Code Instrumentation
```cpp
// platform_memory_utils.hpp
#pragma once

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
    #ifdef _DEBUG
        #include <crtdbg.h>
    #endif
#elif __APPLE__
    #include <mach/mach.h>
    #include <malloc/malloc.h>
#elif __linux__
    #include <unistd.h>
    #include <sys/resource.h>
    #include <malloc.h>
#endif

class PlatformMemoryUtils {
public:
    static size_t getCurrentRSS() {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return pmc.WorkingSetSize;
        }
        return 0;
#elif __APPLE__
        struct mach_task_basic_info info;
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                     (task_info_t)&info, &count) == KERN_SUCCESS) {
            return info.resident_size;
        }
        return 0;
#elif __linux__
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        FILE* file = fopen("/proc/self/statm", "r");
        if (file) {
            long rss_pages;
            fscanf(file, "%*ld %ld", &rss_pages);
            fclose(file);
            return rss_pages * page_size;
        }
        return 0;
#endif
    }
    
    static void enableMemoryDebugging() {
#ifdef _WIN32
    #ifdef _DEBUG
        _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    #endif
#elif __APPLE__
        // Set malloc debugging environment variables programmatically
        setenv("MallocScribble", "1", 1);
        setenv("MallocPreScribble", "1", 1);
        setenv("MallocGuardEdges", "1", 1);
        setenv("MallocStackLogging", "1", 1);
#elif __linux__
        // Enable malloc debugging
        setenv("MALLOC_CHECK_", "2", 1);
#endif
    }
};
```

#### 2. Automated Testing Scripts
```bash
#!/bin/bash
# cross_platform_memory_test.sh

PLATFORM=$(uname -s)
EXECUTABLE=$1

if [ -z "$EXECUTABLE" ]; then
    echo "Usage: $0 <executable>"
    exit 1
fi

echo "Running memory tests on $PLATFORM for $EXECUTABLE"

case $PLATFORM in
    "Linux")
        echo "Running Valgrind..."
        valgrind --tool=memcheck --leak-check=full --error-exitcode=1 ./$EXECUTABLE
        
        echo "Running with AddressSanitizer..."
        if [ -f "${EXECUTABLE}_asan" ]; then
            ./${EXECUTABLE}_asan
        fi
        ;;
        
    "Darwin")
        echo "Running macOS leaks check..."
        leaks --atExit -- ./$EXECUTABLE
        
        echo "Running with malloc debugging..."
        env MallocScribble=1 MallocPreScribble=1 ./$EXECUTABLE
        ;;
        
    "MINGW"*|"MSYS"*|"CYGWIN"*)
        echo "Running Windows memory checks..."
        ./$EXECUTABLE.exe
        
        # Check for CRT debug output
        echo "Check debug output for memory leak reports"
        ;;
        
    *)
        echo "Unknown platform: $PLATFORM"
        exit 1
        ;;
esac

echo "Memory testing completed"
```

### Learning Objectives

By the end of this section, you should be able to:

- **Choose appropriate memory tools** for each platform (Linux, Windows, macOS)
- **Interpret memory analysis output** from platform-specific tools
- **Set up automated memory testing** in your development workflow
- **Use cross-platform memory debugging techniques**
- **Integrate memory analysis into CI/CD pipelines**
- **Debug platform-specific memory issues** effectively

### Practice Exercises

1. **Multi-platform Memory Leak Creation**: Write a program with intentional memory leaks and detect them using platform-specific tools

2. **Performance Comparison**: Compare memory profiling results across different platforms for the same application

3. **Automated Testing Setup**: Create a build system that automatically runs appropriate memory tests based on the target platform

4. **Memory Visualization**: Use graphical tools (Instruments, VMMap, Massif-visualizer) to analyze memory patterns

5. **Production Debugging**: Simulate debugging memory issues in production environments using platform-appropriate tools

---

