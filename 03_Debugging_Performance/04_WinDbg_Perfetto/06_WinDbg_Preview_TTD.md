# WinDbg Preview and Time Travel Debugging (TTD)

*Duration: 2-3 weeks*

## Overview

WinDbg Preview is Microsoft's modern, redesigned version of the classic WinDbg debugger, featuring an enhanced user interface, improved workflows, and powerful Time Travel Debugging (TTD) capabilities. Time Travel Debugging allows you to record a trace of your program's execution and then replay it forwards and backwards, making it incredibly powerful for debugging complex issues, especially those that are hard to reproduce.

### Key Features of WinDbg Preview
- **Modern Interface**: Ribbon-based UI with improved usability
- **Time Travel Debugging**: Record and replay program execution
- **Enhanced Scripting**: JavaScript and Python support
- **Better Visualization**: Memory windows, disassembly views, and call stacks
- **Integrated Help**: Built-in documentation and tutorials
- **Cloud Integration**: Symbol server integration and collaborative debugging

### What is Time Travel Debugging?
Time Travel Debugging captures a complete trace of your program's execution, including:
- All memory reads and writes
- Register states
- Function calls and returns
- Exception handling
- Thread scheduling events

This allows you to:
- Step backwards through execution
- Set breakpoints in the past
- Query the entire execution history
- Analyze how data changed over time

## Getting Started with WinDbg Preview

### Installation and Setup

**Prerequisites:**
- Windows 10 version 1903 or later
- Administrator privileges for some debugging scenarios
- Visual Studio or Windows SDK for symbol support

**Installation Steps:**
1. **From Microsoft Store** (Recommended):
   - Open Microsoft Store
   - Search for "WinDbg Preview"
   - Click "Install"

2. **From Windows SDK**:
   - Download Windows SDK
   - Select "Debugging Tools for Windows" during installation
   - WinDbg Preview will be installed alongside classic WinDbg

3. **Standalone Installation**:
   - Download from Microsoft Developer Downloads
   - Run the installer with administrator privileges

### Initial Configuration

**Setting up Symbol Paths:**
```
# In WinDbg Preview Command Window
.sympath srv*C:\Symbols*https://msdl.microsoft.com/download/symbols

# Or set environment variable
set _NT_SYMBOL_PATH=srv*C:\Symbols*https://msdl.microsoft.com/download/symbols
```

**Basic Settings:**
- **File → Settings**: Configure preferences
- **Symbol Settings**: Set up Microsoft Symbol Server
- **Source Path**: Configure source code locations
- **Workspace**: Save debugging sessions

## Time Travel Debugging (TTD) Fundamentals

### Understanding TTD Concepts

**TTD Trace File Structure:**
```
trace_file.run        # Main trace file
trace_file.idx        # Index file for quick navigation
trace_file.err        # Error log (if any)
```

**TTD Timeline Concepts:**
- **Position**: Unique identifier for each point in execution
- **Sequence**: Ordered execution steps
- **Events**: Memory accesses, function calls, exceptions
- **Threads**: Multiple execution contexts within the trace

### TTD vs Traditional Debugging

| Aspect | Traditional Debugging | Time Travel Debugging |
|--------|----------------------|----------------------|
| **Direction** | Forward only | Forward and backward |
| **Reproducibility** | May vary between runs | Identical replay every time |
| **Breakpoint Setting** | Only for future execution | Can set in past or future |
| **Memory Analysis** | Current state only | Historical memory states |
| **Performance Impact** | Minimal | High during recording |
| **Trace Persistence** | Lost after session | Saved for later analysis |

## Recording TTD Traces

### Method 1: Launch and Record

**Starting a new process with TTD:**
```cmd
# Command line approach
ttd.exe -launch "C:\path\to\your\program.exe" -args "arg1 arg2"

# Or from WinDbg Preview:
# File → Start debugging → Launch executable → Enable TTD recording
```

**WinDbg Preview UI Steps:**
1. **File → Start debugging → Launch executable**
2. **Check "Record with Time Travel Debugging"**
3. **Browse to executable**: Select your program
4. **Set arguments**: If needed
5. **Set working directory**: Usually same as executable
6. **Click "Launch"**

### Method 2: Attach to Running Process

**Attaching to existing process:**
```cmd
# Command line
ttd.exe -attach <process_id>

# WinDbg Preview UI:
# File → Attach to process → Select process → Enable TTD
```

**Important Considerations:**
- TTD recording starts from the point of attachment
- Some state information before attachment is lost
- Process must be attachable (not protected)

### Example: Recording a Complex Program

Let's create a more comprehensive example program to demonstrate TTD capabilities:

```c
// complex_example.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int id;
    char name[50];
    double value;
} DataItem;

// Function that might have bugs
DataItem* process_data(int count) {
    DataItem* items = malloc(count * sizeof(DataItem));
    
    for (int i = 0; i < count; i++) {
        items[i].id = i;
        sprintf(items[i].name, "Item_%d", i);
        items[i].value = (double)rand() / RAND_MAX * 100.0;
        
        // Intentional bug: buffer overflow every 10th item
        if (i % 10 == 9) {
            sprintf(items[i].name, "Very_Long_Item_Name_That_Causes_Buffer_Overflow_%d", i);
        }
    }
    
    return items;
}

// Function with potential access violation
void access_data(DataItem* items, int count, int index) {
    if (index >= 0 && index < count) {
        printf("Item %d: %s = %.2f\n", items[index].id, items[index].name, items[index].value);
    } else {
        // Intentional bug: out of bounds access
        printf("Invalid access: %s\n", items[index].name);  // Potential crash
    }
}

int main(int argc, char* argv[]) {
    printf("Starting complex TTD example\n");
    
    srand((unsigned int)time(NULL));
    
    // Create some data
    DataItem* data = process_data(20);
    
    // Normal access
    for (int i = 0; i < 5; i++) {
        access_data(data, 20, i);
    }
    
    // Problematic access
    printf("Attempting problematic access...\n");
    access_data(data, 20, 25);  // Out of bounds - might crash
    
    free(data);
    printf("Program completed\n");
    return 0;
}
```

**Compilation for TTD:**
```cmd
# Compile with debug information
cl /Zi /Od complex_example.c /Fe:complex_example.exe

# Or with GCC on Windows
gcc -g -O0 complex_example.c -o complex_example.exe
```

**Recording the trace:**
```cmd
# Method 1: Direct launch
ttd.exe -launch complex_example.exe -out trace_output

# Method 2: Through WinDbg Preview
# File → Start debugging → Launch executable
# Enable "Record with Time Travel Debugging"
# Select complex_example.exe
```

## Analyzing TTD Traces

### Basic Navigation Commands

**Timeline Navigation:**
```windbg
# Go to beginning of trace
!tt 0

# Go to end of trace
!tt 100

# Go to specific position
!tt 50

# Step forward/backward
!tt +1    # Forward one step
!tt -1    # Backward one step

# Jump by percentage
!tt 25    # Go to 25% through the trace
```

**Position Information:**
```windbg
# Current position
!position

# Show timeline information
!tt

# List all positions with events
!index
```

### Memory Analysis Over Time

**Tracking Memory Changes:**
```windbg
# Watch memory range over time
ba w4 0x00401000    # Break on write to address

# Query memory at different times
!tt 0               # Go to start
dd 0x00401000       # Display memory
!tt 50              # Go to middle
dd 0x00401000       # Display same memory
!tt 100             # Go to end
dd 0x00401000       # Display memory again
```

**Memory Timeline Queries:**
```windbg
# Find all writes to a memory range
dx @$cursession.TTD.Memory(0x00401000, 0x00401100, "w")

# Find reads from specific address
dx @$cursession.TTD.Memory(0x00401020, 0x00401024, "r")

# Query memory access patterns
dx @$cursession.TTD.Memory(0x00401000, 0x00401100, "rw")
```

### Function Call Analysis

**Tracking Function Calls:**
```windbg
# Set breakpoint on function
bp complex_example!process_data

# Find all calls to function
dx @$cursession.TTD.Calls("complex_example!process_data")

# Analyze function parameters over time
dx @$cursession.TTD.Calls("complex_example!access_data").Select(c => new { Position = c.TimeStart, Parameters = c.Parameters })
```

**Call Stack Analysis:**
```windbg
# View call stack at current position
k

# Navigate to function entry/exit
!tt 0
bp complex_example!process_data
g                   # Go to first call
k                   # View stack
!tt +100           # Step forward
k                   # View stack changes
```

### Exception and Crash Analysis

**Finding Exceptions:**
```windbg
# Find all exceptions in trace
dx @$cursession.TTD.Events.Where(e => e.Type == "Exception")

# Navigate to exception
!tt <exception_position>

# Analyze exception details
.exr -1             # Display exception record
.cxr -1             # Display context record
k                   # Call stack at exception
```

**Crash Investigation Workflow:**
```windbg
# 1. Load the trace file
.opendump C:\path\to\trace.run

# 2. Go to the crash point (usually at the end)
!tt 100

# 3. Examine the crash
.exr -1
.cxr -1
k

# 4. Work backwards to find root cause
!tt -100           # Go back 100 steps
k                  # Check call stack
!tt -200           # Go back more
# Continue analyzing...
```

## Advanced TTD Features

### JavaScript Debugging Extensions

**Custom TTD Queries:**
```javascript
// Find all malloc calls
function findMallocCalls() {
    var calls = host.currentSession.TTD.Calls("ntdll!RtlAllocateHeap");
    for (var call of calls) {
        host.diagnostics.debugLog("Malloc at ", call.TimeStart.toString(), 
                                  " Size: ", call.Parameters[2].toString(16), "\n");
    }
}

// Track variable changes
function trackVariableChanges(address, size) {
    var memoryAccesses = host.currentSession.TTD.Memory(address, address + size, "w");
    for (var access of memoryAccesses) {
        host.diagnostics.debugLog("Memory write at ", access.TimeStart.toString(),
                                  " Address: ", access.Address.toString(16), "\n");
    }
}
```

**Loading and Using Scripts:**
```windbg
# Load JavaScript file
.scriptload C:\path\to\debug_script.js

# Run custom function
dx @$scriptContents.findMallocCalls()

# Create object for repeated use
dx @$myDebugger = @$scriptContents
dx @$myDebugger.trackVariableChanges(0x401000, 0x100)
```

### Data Model Queries

**Complex Queries:**
```windbg
# Find all function calls that returned specific values
dx @$cursession.TTD.Calls("*").Where(c => c.ReturnValue == 0)

# Find memory writes in specific time range
dx @$cursession.TTD.Memory(0x0, 0xFFFFFFFF, "w").Where(m => m.TimeStart > 0x1000 && m.TimeStart < 0x2000)

# Analyze thread switching patterns
dx @$cursession.TTD.Events.Where(e => e.Type == "ThreadCreated" || e.Type == "ThreadTerminated")
```

### Debugging Multi-threaded Applications

**Thread-specific Analysis:**
```windbg
# List all threads in trace
~

# Switch to specific thread
~0s    # Switch to thread 0
~1s    # Switch to thread 1

# Find thread-specific events
dx @$cursession.TTD.Events.Where(e => e.ThreadId == 0x1234)

# Analyze synchronization events
dx @$cursession.TTD.Events.Where(e => e.Type == "SynchronizationEvent")
```

**Race Condition Detection:**
```windbg
# Find concurrent memory accesses
# This requires custom scripting to correlate timing

# Example: Find overlapping memory operations
dx @$cursession.TTD.Memory(0x401000, 0x401004, "rw").OrderBy(m => m.TimeStart)
```

## Practical Debugging Scenarios

### Scenario 1: Buffer Overflow Investigation

**Problem:** Program crashes with access violation
```windbg
# 1. Load trace and go to crash
.opendump buffer_overflow_trace.run
!tt 100

# 2. Examine crash details
.exr -1
.cxr -1
k

# 3. Find the problematic memory write
# Look for writes to the crashed address
dx @$cursession.TTD.Memory(0x<crash_address>, 0x<crash_address>+4, "w")

# 4. Navigate to the write that caused corruption
!tt <position_of_write>
k

# 5. Examine the source of the write
u .          # Disassemble current instruction
dv           # Display local variables
```

### Scenario 2: Memory Leak Detection

**Problem:** Program uses increasing amounts of memory
```windbg
# 1. Find all allocation calls
dx @$allocCalls = @$cursession.TTD.Calls("ntdll!RtlAllocateHeap")

# 2. Find all deallocation calls
dx @$freeCalls = @$cursession.TTD.Calls("ntdll!RtlFreeHeap")

# 3. Compare allocations vs deallocations
dx @$allocCalls.Count()
dx @$freeCalls.Count()

# 4. Track specific allocations
dx @$allocCalls.Where(c => c.Parameters[2] > 0x1000)  # Large allocations
```

### Scenario 3: Logic Error Investigation

**Problem:** Program produces incorrect results
```c
// Example problematic function
int calculate_average(int* values, int count) {
    int sum = 0;
    for (int i = 0; i <= count; i++) {  // Bug: should be i < count
        sum += values[i];
    }
    return sum / count;
}
```

**Debugging approach:**
```windbg
# 1. Set breakpoint on function
bp myprogram!calculate_average

# 2. Find all calls to function
dx @$cursession.TTD.Calls("myprogram!calculate_average")

# 3. Navigate to specific call
!tt <call_position>

# 4. Step through function execution
p    # Step through each instruction
p    # Continue stepping
# Watch for array bounds violation

# 5. Examine memory corruption
dv   # Display variables
dd esp, 10  # Display stack memory
```

## Performance Considerations

### TTD Recording Overhead

**Performance Impact:**
- **CPU Overhead**: 10x - 20x slower execution
- **Memory Usage**: 2x - 5x more memory required
- **Disk Space**: Trace files can be very large (GBs)
- **Recording Time**: Proportional to execution complexity

**Optimization Strategies:**
```windbg
# 1. Record only specific modules
ttd.exe -launch myapp.exe -include myapp.exe,mylib.dll

# 2. Limit recording duration
ttd.exe -launch myapp.exe -maxFile 1000MB

# 3. Use conditional recording
# Start/stop recording programmatically
```

### Trace File Management

**File Size Considerations:**
```cmd
# Check trace file size
dir *.run

# Compress trace files
compact /c /s *.run

# Archive old traces
move old_traces.run archive\
```

**Best Practices:**
- Record only when necessary
- Use specific module filtering
- Clean up old trace files regularly
- Consider network storage for large traces

## Integration with Development Workflow

### Continuous Integration Integration

**Automated TTD Recording:**
```yaml
# Azure DevOps Pipeline example
- task: CmdLine@2
  displayName: 'Record TTD Trace'
  inputs:
    script: |
      ttd.exe -launch $(Build.BinariesDirectory)\myapp.exe -out $(Build.ArtifactStagingDirectory)\trace
  condition: failed()  # Only record on test failures
```

**Post-mortem Analysis:**
```powershell
# PowerShell script for automated analysis
param($tracePath)

$windbg = "C:\Program Files\Windows Kits\10\Debuggers\x64\windbg.exe"
$script = @"
.opendump $tracePath
!tt 100
.exr -1
.cxr -1
k
q
"@

& $windbg -c $script -logo analysis.log
```

### Team Collaboration

**Sharing Traces:**
- Store traces in shared network location
- Include symbol files and source code
- Document reproduction steps
- Create analysis templates

**Review Process:**
1. Record trace on failure
2. Share trace with team
3. Collaborative analysis session
4. Document findings
5. Implement fix
6. Verify with new trace

## Learning Objectives

By the end of this section, you should be able to:

- **Install and configure** WinDbg Preview with proper symbol settings
- **Record TTD traces** using multiple methods (launch, attach)
- **Navigate through traces** using timeline commands and position references
- **Analyze memory access patterns** and track data corruption over time
- **Debug complex issues** like buffer overflows, memory leaks, and race conditions
- **Use advanced features** like JavaScript extensions and data model queries
- **Optimize TTD usage** for performance and storage considerations
- **Integrate TTD** into development and CI/CD workflows

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Successfully install and configure WinDbg Preview  
□ Record a TTD trace of a simple program  
□ Navigate backwards and forwards through a trace  
□ Set breakpoints in the past and analyze execution  
□ Find all calls to a specific function in a trace  
□ Track memory writes to a specific address  
□ Analyze an exception or crash using TTD  
□ Write basic JavaScript queries for trace analysis  
□ Understand the performance implications of TTD recording  
□ Clean up and manage trace files effectively  

### Practical Exercises

**Exercise 1: Basic TTD Recording**
```c
// TODO: Create a program with a subtle bug and record its execution
// Then use TTD to find and fix the bug
#include <stdio.h>
#include <stdlib.h>

int buggy_function(int n) {
    int* arr = malloc(n * sizeof(int));
    
    for (int i = 0; i <= n; i++) {  // Bug here!
        arr[i] = i * i;
    }
    
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    
    free(arr);
    return sum;
}

int main() {
    printf("Result: %d\n", buggy_function(10));
    return 0;
}
```

**Exercise 2: Memory Corruption Investigation**
```c
// TODO: Use TTD to trace memory corruption in this program
#include <stdio.h>
#include <string.h>

void corrupt_memory() {
    char buffer[10];
    strcpy(buffer, "This string is way too long for the buffer!");
    printf("Buffer: %s\n", buffer);
}

int main() {
    corrupt_memory();
    return 0;
}
```

**Exercise 3: Multi-threaded Race Condition**
```c
// TODO: Record and analyze a race condition using TTD
#include <stdio.h>
#include <windows.h>

int shared_counter = 0;

DWORD WINAPI worker_thread(LPVOID param) {
    for (int i = 0; i < 1000; i++) {
        shared_counter++;  // Race condition!
    }
    return 0;
}

int main() {
    HANDLE threads[4];
    
    for (int i = 0; i < 4; i++) {
        threads[i] = CreateThread(NULL, 0, worker_thread, NULL, 0, NULL);
    }
    
    WaitForMultipleObjects(4, threads, TRUE, INFINITE);
    
    printf("Final counter: %d (should be 4000)\n", shared_counter);
    return 0;
}
```

## Study Materials

### Recommended Reading
- **Primary:** [Time Travel Debugging Documentation](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/time-travel-debugging-overview) - Microsoft Docs
- **Advanced:** "Windows Internals" by Russinovich & Solomon - Debugging chapters
- **Reference:** [WinDbg Preview Documentation](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/debugging-using-windbg-preview)

### Video Resources
- "Introduction to Time Travel Debugging" - Microsoft Channel 9
- "Advanced WinDbg Techniques" - DefCon presentations
- "Windows Debugging Fundamentals" - Microsoft Virtual Academy

### Hands-on Labs
- **Lab 1:** Set up WinDbg Preview and record your first TTD trace
- **Lab 2:** Debug a buffer overflow using TTD timeline navigation
- **Lab 3:** Analyze memory leaks with TTD memory tracking
- **Lab 4:** Create custom JavaScript extensions for trace analysis

### Practice Scenarios

**Beginner Scenarios:**
1. Simple crash investigation
2. Variable value tracking
3. Function call analysis

**Intermediate Scenarios:**
1. Memory corruption detection
2. Multi-threaded debugging
3. Performance bottleneck analysis

**Advanced Scenarios:**
1. Kernel-mode debugging with TTD
2. Large-scale trace analysis
3. Custom debugging extensions

### Development Environment Setup

**Required Software:**
```cmd
# Install Windows SDK with Debugging Tools
# Download from: https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/

# Verify installation
windbg -version
ttd.exe -help
```

**Environment Configuration:**
```cmd
# Set symbol path
set _NT_SYMBOL_PATH=srv*C:\Symbols*https://msdl.microsoft.com/download/symbols

# Set source path
set _NT_SOURCE_PATH=C:\Source

# Configure debugging privilege
# Run as Administrator when needed
```

**Useful Scripts and Extensions:**
```javascript
// Save as ttd_helpers.js
function showTraceInfo() {
    var session = host.currentSession;
    host.diagnostics.debugLog("Trace duration: ", session.TTD.TimeEnd.toString(), "\n");
    host.diagnostics.debugLog("Position count: ", session.TTD.Positions.Count(), "\n");
}

function findLargeAllocations(minSize) {
    var calls = host.currentSession.TTD.Calls("*Alloc*");
    for (var call of calls) {
        if (call.Parameters.length > 0 && call.Parameters[0] >= minSize) {
            host.diagnostics.debugLog("Large allocation: ", call.TimeStart.toString(), 
                                      " Size: ", call.Parameters[0].toString(), "\n");
        }
    }
}
```

## Troubleshooting Common Issues

### TTD Recording Problems

**Issue: Cannot record trace**
```
Solutions:
1. Run WinDbg Preview as Administrator
2. Check if process is already being debugged
3. Verify TTD is enabled on the system
4. Check available disk space
```

**Issue: Trace file corruption**
```
Solutions:
1. Ensure adequate disk space during recording
2. Don't interrupt recording process
3. Check for filesystem errors
4. Use file system with large file support
```

### Performance Issues

**Issue: Recording too slow**
```
Solutions:
1. Use module filtering: -include module.exe
2. Record shorter durations
3. Use faster storage (SSD)
4. Increase system memory
```

**Issue: Large trace files**
```
Solutions:
1. Compress traces after recording
2. Use selective recording
3. Clean up old traces regularly
4. Archive to network storage
```

