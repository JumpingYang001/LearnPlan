# WinDbg Fundamentals

*Duration: 1 week*

## Overview

WinDbg is Microsoft's flagship debugger for Windows applications, drivers, and operating system components. It's an essential tool for advanced Windows development, system debugging, crash dump analysis, and performance investigation. This comprehensive guide covers everything from basic usage to advanced debugging techniques.

### What is WinDbg?

WinDbg (Windows Debugger) is a multipurpose debugger that can:
- Debug user-mode applications (live debugging)
- Analyze crash dumps (post-mortem debugging)
- Debug kernel-mode drivers and the Windows kernel
- Perform remote debugging across networks
- Analyze memory dumps from blue screens (BSOD)
- Profile application performance and memory usage

### Why Learn WinDbg?

**For Developers:**
- Deep debugging capabilities beyond Visual Studio
- Advanced memory analysis and heap debugging
- Performance bottleneck identification
- Production crash analysis

**For System Administrators:**
- Windows system troubleshooting
- Blue screen analysis
- Memory leak detection in services
- Performance monitoring

**For Security Researchers:**
- Malware analysis
- Exploit development and mitigation
- Reverse engineering
- Vulnerability research

## WinDbg Architecture and Components

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    WinDbg Frontend                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────┐ │
│  │   Command Line  │  │   GUI Interface │  │  VS Code│ │
│  │    (cdb.exe)    │  │   (windbg.exe)  │  │Extension│ │
│  └─────────────────┘  └─────────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                 Debug Engine (DbgEng)                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────┐ │
│  │   Symbol Engine │  │  Target Control │  │ Memory  │ │
│  │   (symsrv.dll)  │  │                 │  │ Engine  │ │
│  └─────────────────┘  └─────────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Debug Targets                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────┐ │
│  │   Live Process  │  │   Crash Dump    │  │ Kernel  │ │
│  │                 │  │   (.dmp file)   │  │ Target  │ │
│  └─────────────────┘  └─────────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
```

### WinDbg Variants

| Tool | Purpose | Use Case |
|------|---------|----------|
| **WinDbg (GUI)** | Full-featured graphical debugger | Interactive debugging, learning |
| **WinDbg Preview** | Modern UWP version | Enhanced UI, better performance |
| **CDB** | Console debugger | Scripting, automation, CI/CD |
| **KD** | Kernel debugger | Driver debugging, BSOD analysis |
| **NTSD** | NT Symbolic Debugger | Service debugging, minimal UI |

## Installation and Setup

### Installing WinDbg

**Method 1: Windows SDK**
```powershell
# Download Windows SDK from Microsoft
# WinDbg is included in Windows SDK
```

**Method 2: Microsoft Store (WinDbg Preview)**
```powershell
# Search for "WinDbg Preview" in Microsoft Store
# Or use winget
winget install Microsoft.WinDbg
```

**Method 3: Standalone Download**
```powershell
# Download from Microsoft Developer Tools
# Extract to desired location
```

### Essential Configuration

**1. Symbol Path Setup**
```
# Set symbol path to Microsoft Symbol Server
.sympath srv*C:\symbols*https://msdl.microsoft.com/download/symbols

# Or set environment variable
set _NT_SYMBOL_PATH=srv*C:\symbols*https://msdl.microsoft.com/download/symbols
```

**2. Source Path Configuration**
```
# Set source code path
.srcpath C:\Source\MyProject;C:\Source\Libraries

# Or environment variable
set _NT_SOURCE_PATH=C:\Source\MyProject;C:\Source\Libraries
```

**3. WinDbg Workspace Setup**
```
# Save workspace for project
.writemem workspace.wds

# Load workspace
File -> Recent Workspaces -> workspace.wds
```

## Getting Started: Your First Debugging Session

Let's start with a comprehensive hands-on example that demonstrates core WinDbg concepts.

### Example Program: Multi-threaded Application with Issues

```c
// debug_example.c - Compile with: cl /Zi /Od debug_example.c
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

// Global variables for demonstration
int g_counter = 0;
HANDLE g_mutex = NULL;
volatile BOOL g_shutdown = FALSE;

// Structure for thread data
typedef struct {
    int thread_id;
    int iterations;
    BOOL use_synchronization;
} ThreadData;

// Thread function that may have race conditions
DWORD WINAPI WorkerThread(LPVOID param) {
    ThreadData* data = (ThreadData*)param;
    
    printf("Thread %d started (PID: %lu, TID: %lu)\n", 
           data->thread_id, GetCurrentProcessId(), GetCurrentThreadId());
    
    for (int i = 0; i < data->iterations && !g_shutdown; i++) {
        if (data->use_synchronization) {
            // Thread-safe version
            WaitForSingleObject(g_mutex, INFINITE);
            g_counter++;
            ReleaseMutex(g_mutex);
        } else {
            // Potential race condition
            g_counter++;
        }
        
        // Simulate work
        Sleep(1);
    }
    
    printf("Thread %d completed\n", data->thread_id);
    return 0;
}

// Function that demonstrates memory allocation
void* AllocateMemory(size_t size) {
    void* ptr = malloc(size);
    if (ptr) {
        printf("Allocated %zu bytes at address: %p\n", size, ptr);
        // Intentionally don't free to demonstrate heap analysis
    }
    return ptr;
}

// Function that may cause access violation
void DemonstrateAccessViolation(BOOL trigger_av) {
    if (trigger_av) {
        int* null_ptr = NULL;
        *null_ptr = 42;  // This will cause access violation
    }
}

int main(int argc, char* argv[]) {
    printf("WinDbg Debug Example\n");
    printf("Process ID: %lu\n", GetCurrentProcessId());
    printf("Main Thread ID: %lu\n", GetCurrentThreadId());
    
    // Create mutex for synchronization
    g_mutex = CreateMutex(NULL, FALSE, NULL);
    if (!g_mutex) {
        printf("Failed to create mutex\n");
        return 1;
    }
    
    // Parse command line arguments
    BOOL use_sync = TRUE;
    BOOL trigger_av = FALSE;
    int num_threads = 3;
    
    if (argc > 1) {
        if (strcmp(argv[1], "nosync") == 0) use_sync = FALSE;
        if (strcmp(argv[1], "crash") == 0) trigger_av = TRUE;
    }
    
    printf("Press ENTER to start debugging session...\n");
    getchar();  // Pause for debugger attachment
    
    // Demonstrate memory allocation
    for (int i = 0; i < 5; i++) {
        AllocateMemory(1024 * (i + 1));
    }
    
    // Create worker threads
    HANDLE threads[3];
    ThreadData thread_data[3];
    
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i + 1;
        thread_data[i].iterations = 1000;
        thread_data[i].use_synchronization = use_sync;
        
        threads[i] = CreateThread(
            NULL,                   // Security attributes
            0,                      // Stack size (default)
            WorkerThread,           // Thread function
            &thread_data[i],        // Thread parameter
            0,                      // Creation flags
            NULL                    // Thread ID
        );
        
        if (!threads[i]) {
            printf("Failed to create thread %d\n", i + 1);
            return 1;
        }
    }
    
    printf("Created %d worker threads\n", num_threads);
    printf("Expected final counter value: %d\n", num_threads * 1000);
    
    // Wait for threads to complete
    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);
    
    // Clean up thread handles
    for (int i = 0; i < num_threads; i++) {
        CloseHandle(threads[i]);
    }
    
    printf("Final counter value: %d\n", g_counter);
    
    // Demonstrate access violation if requested
    DemonstrateAccessViolation(trigger_av);
    
    // Clean up
    CloseHandle(g_mutex);
    
    printf("Press ENTER to exit...\n");
    getchar();
    
    return 0;
}
```

### Compilation Instructions

```batch
REM Compile with debug symbols
cl /Zi /Od /MTd debug_example.c /link /DEBUG

REM Or with GCC (MinGW)
gcc -g -O0 debug_example.c -o debug_example.exe

REM Create PDB file for Visual Studio
cl /Zi /Od debug_example.c /link /DEBUG /PDB:debug_example.pdb
```

## WinDbg Interface and Basic Commands

### WinDbg GUI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ File  Edit  View  Debug  Window  Help                    [_][□][×]  │
├─────────────────────────────────────────────────────────────────────┤
│ Toolbar: [▶] [⏸] [⏹] [📂] [💾] [🔍] [⚙]                          │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─── Command Window ────────────────┐ ┌─── Disassembly ─────────────┐ │
│ │ 0:000> g                          │ │ ntdll!NtDelayExecution+0x14:│ │
│ │ ModLoad: 00007ff9`a2340000        │ │ 00007ff9`a235c4f4 c3  ret   │ │
│ │ Break instruction exception       │ │ 00007ff9`a235c4f5 cc  int 3 │ │
│ │                                   │ │                               │ │
│ └───────────────────────────────────┘ └───────────────────────────────┘ │
│ ┌─── Call Stack ────────────────────┐ ┌─── Locals/Watch ─────────────┐ │
│ │ # ChildEBP RetAddr                │ │ Name      Value    Type       │ │
│ │ 00 0019ff7c 00401234              │ │ g_counter 1523     int        │ │
│ │ 01 0019ff8c 7c816d4f              │ │ data      0x005... ThreadData*│ │
│ │                                   │ │                               │ │
│ └───────────────────────────────────┘ └───────────────────────────────┘ │
│ ┌─── Memory Window ─────────────────────────────────────────────────── │ │
│ │ 0x00401000  4d 5a 90 00 03 00 00 00-04 00 00 00 ff ff 00 00     MZ│ │
│ │ 0x00401010  b8 00 00 00 00 00 00 00-40 00 00 00 00 00 00 00       │ │
│ └─────────────────────────────────────────────────────────────────── │ │
└─────────────────────────────────────────────────────────────────────┘
```

### Essential WinDbg Commands

#### Basic Navigation Commands

```
Command         Description                           Example
───────────────────────────────────────────────────────────────────────
g               Go (continue execution)               g
gh              Go with exception handled             gh
gu              Go up (step out of function)          gu
p               Step over (one instruction)           p
t               Trace into (one instruction)          t
pc              Step over (one C++ source line)       pc
tc              Trace into (one C++ source line)      tc
bp              Set breakpoint                        bp main
bl              List breakpoints                      bl
bc              Clear breakpoint                      bc 1
bd              Disable breakpoint                    bd 1
be              Enable breakpoint                     be 1
q               Quit debugger                         q
```

#### Information and Analysis Commands

```
Command         Description                           Example
───────────────────────────────────────────────────────────────────────
~               Show threads                          ~
~*k             Show stack trace for all threads     ~*k
!analyze        Automatic crash analysis              !analyze -v
lm              List loaded modules                   lm
x               Examine symbols                       x kernel32!*Create*
dt              Display type                          dt _PEB
?               Evaluate expression                   ? poi(esp)
r               Show/set registers                    r eax
d*              Display memory                        db 401000
u               Unassemble                           u main
k               Stack trace                          k
.frame          Set local context                    .frame 2
```

#### Symbol and Module Commands

```
Command         Description                           Example
───────────────────────────────────────────────────────────────────────
.sympath        Set symbol path                       .sympath srv*c:\symbols*...
.reload         Reload symbols                        .reload /f
ld              Load symbols for module               ld kernel32
!sym            Symbol server commands                !sym noisy
ln              List nearest symbols                  ln 401234
```

## Debugging Modes and Scenarios

### 1. Live Process Debugging

**Attaching to Running Process:**
```powershell
# Method 1: Launch WinDbg and attach
windbg.exe -p <PID>

# Method 2: Attach from command line
cdb.exe -p <PID>

# Method 3: Launch program under debugger
windbg.exe -o debug_example.exe

# Method 4: Non-invasive attach (read-only)
windbg.exe -pv <PID>
```

**Step-by-Step Live Debugging Session:**

1. **Start the debug example:**
```batch
REM Terminal 1: Run the program
debug_example.exe
Process ID: 1234
Press ENTER to start debugging session...
```

2. **Attach WinDbg:**
```batch
REM Terminal 2: Attach debugger
windbg.exe -p 1234
```

3. **Initial setup in WinDbg:**
```
0:000> .sympath srv*C:\symbols*https://msdl.microsoft.com/download/symbols
0:000> .reload /f
0:000> x debug_example!*
debug_example!main
debug_example!WorkerThread
debug_example!g_counter
```

4. **Set breakpoints:**
```
0:000> bp debug_example!WorkerThread
0:000> bp debug_example!main+0x123  ; Specific offset
0:000> ba w4 debug_example!g_counter  ; Hardware breakpoint on write
```

5. **Continue execution:**
```
0:000> g
Breakpoint 0 hit
debug_example!WorkerThread:
00007ff7`12341234 mov rdi,rcx
```

6. **Analyze thread state:**
```
0:001> ~
   0  Id: 1234.5678 Suspend: 1 Teb: 00000012`34567890 Unfrozen
.  1  Id: 1234.9abc Suspend: 1 Teb: 00000012`34567def Unfrozen
   2  Id: 1234.1def Suspend: 1 Teb: 00000012`34567123 Unfrozen

0:001> dt ThreadData rcx
   +0x000 thread_id        : 0n1
   +0x004 iterations       : 0n1000  
   +0x008 use_synchronization : 0y1
```

### 2. Crash Dump Analysis

**Creating Dump Files:**
```powershell
# Create full dump
taskmgr.exe -> Right-click process -> Create dump file

# Using command line
procdump.exe -ma <PID> crash_dump.dmp

# Automatic crash dumps (configure in Registry)
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpType /t REG_DWORD /d 2
```

**Analyzing Crash Dump:**
```
# Open crash dump
windbg.exe -z crash_dump.dmp

# Automatic analysis
0:000> !analyze -v
EXCEPTION_RECORD:  (.exr -1)
ExceptionAddress: 00007ff712341234 (debug_example!DemonstrateAccessViolation+0x14)
   ExceptionCode: c0000005 (Access violation)
  ExceptionFlags: 00000000
NumberParameters: 2
   Parameter[0]: 0000000000000001
   Parameter[1]: 0000000000000000
Attempt to write to address 0000000000000000

STACK_TEXT:
debug_example!DemonstrateAccessViolation+0x14
debug_example!main+0x1a3
debug_example!__tmainCRTStartup+0x10f
kernel32!BaseThreadInitThunk+0x14
ntdll!RtlUserThreadStart+0x21

FAULT_INSTR_CODE:  c7042a

SYMBOL_NAME:  debug_example!DemonstrateAccessViolation+14

MODULE_NAME: debug_example

FOLLOWUP_NAME:  MachineOwner
```

### 3. Kernel Mode Debugging

**Setting up Kernel Debugging:**
```powershell
# Configure target machine (requires reboot)
bcdedit /debug on
bcdedit /dbgsettings serial debugport:1 baudrate:115200

# Or for network debugging
bcdedit /dbgsettings net hostip:192.168.1.100 port:50000 key:1.2.3.4

# Host machine: Connect kernel debugger
windbg.exe -k com:port=COM1,baud=115200
```

**Kernel Debugging Commands:**
```
lkd> .reboot          ; Reboot target machine
lkd> !process 0 0     ; List all processes
lkd> !thread          ; Show current thread
lkd> !irql            ; Show current IRQL
lkd> !locks           ; Show kernel locks
lkd> !vm              ; Virtual memory statistics
lkd> !poolused        ; Pool memory usage
```

### 4. Advanced Memory Analysis

**Heap Analysis:**
```
# Enable heap debugging
0:000> !gflags +hpa +htc +htd

# Analyze heap
0:000> !heap
0:000> !heap -s         ; Summary
0:000> !heap -a 001b0000 ; Analyze specific heap

# Find heap corruption
0:000> !heap -p -a 12345678  ; Analyze heap block
0:000> !heap -l          ; Check for leaks
```

**Memory Leak Detection:**
```
# Application Verifier + WinDbg
0:000> !avrf           ; Show App Verifier settings
0:000> !htrace -enable ; Enable heap tracing
0:000> g               ; Run application
0:000> !htrace -diff   ; Show memory differences
0:000> !heap -s        ; Summary after operation
```

**Virtual Memory Analysis:**
```
0:000> !address                    ; Virtual memory layout
0:000> !address -summary           ; Memory summary
0:000> !vprot 12345678             ; Virtual protection
0:000> !vadump                     ; VAD tree dump
```

## Practical Debugging Scenarios

### Scenario 1: Race Condition Detection

**Problem:** Multiple threads accessing shared counter without synchronization.

```
# Run with race condition
debug_example.exe nosync

# Attach debugger and analyze
0:000> bp debug_example!WorkerThread "da poi(rcx); g"  ; Conditional breakpoint
0:000> ba w4 debug_example!g_counter "k; g"            ; Break on counter write
0:000> g

# Analyze race condition
0:001> ~ ; Show all threads
0:001> ~*k ; Stack trace for all threads
0:001> ?? g_counter ; Check current value
```

### Scenario 2: Deadlock Analysis

**Creating a deadlock scenario:**
```c
// Add to debug_example.c
HANDLE g_mutex1, g_mutex2;

DWORD WINAPI DeadlockThread1(LPVOID param) {
    WaitForSingleObject(g_mutex1, INFINITE);
    printf("Thread 1: Got mutex 1\n");
    Sleep(100);
    WaitForSingleObject(g_mutex2, INFINITE);  // Will deadlock
    ReleaseMutex(g_mutex2);
    ReleaseMutex(g_mutex1);
    return 0;
}

DWORD WINAPI DeadlockThread2(LPVOID param) {
    WaitForSingleObject(g_mutex2, INFINITE);
    printf("Thread 2: Got mutex 2\n");
    Sleep(100);
    WaitForSingleObject(g_mutex1, INFINITE);  // Will deadlock
    ReleaseMutex(g_mutex1);
    ReleaseMutex(g_mutex2);
    return 0;
}
```

**Debugging deadlock:**
```
# Threads are hanging
0:000> ~
   0  Id: 1234.5678 Suspend: 1 Teb: 00000012`34567890 Unfrozen
   1  Id: 1234.9abc Suspend: 1 Teb: 00000012`34567def Unfrozen
   2  Id: 1234.1def Suspend: 1 Teb: 00000012`34567123 Unfrozen

# Check what each thread is waiting for
0:000> ~1s
0:001> k
ntdll!NtWaitForSingleObject+0x14
kernel32!WaitForSingleObjectEx+0x94
debug_example!DeadlockThread1+0x23

0:001> ~2s
0:002> k
ntdll!NtWaitForSingleObject+0x14
kernel32!WaitForSingleObjectEx+0x94
debug_example!DeadlockThread2+0x23

# Analyze synchronization objects
0:002> !locks
0:002> !handle 0 f  ; Show all handles
```

### Scenario 3: Performance Profiling

**Using WinDbg for performance analysis:**
```
# Set time-based breakpoints
0:000> bp debug_example!WorkerThread ".time; g"

# Profile function execution
0:000> bp debug_example!WorkerThread "r $t0 = @$teb->SystemReserved1[0xb]; g"
0:000> bp debug_example!WorkerThread+0x50 "? @$teb->SystemReserved1[0xb] - $t0; g"

# Memory usage tracking
0:000> !heap -s
0:000> g
0:000> !heap -s  ; Compare before/after
```

## Advanced WinDbg Features

### 1. JavaScript Scripting

**WinDbg supports JavaScript for automation:**
```javascript
// analyze_threads.js
"use strict";

function analyzeThreads() {
    let control = host.namespace.Debugger.Utility.Control;
    let output = host.namespace.Debugger.Utility.Output;
    
    // Get all threads
    for (let thread of host.currentProcess.Threads) {
        output.writeLine(`Thread ${thread.Id}:`);
        
        // Show stack
        for (let frame of thread.Stack.Frames) {
            output.writeLine(`  ${frame}`);
        }
    }
}

// Load script: .scriptload C:\scripts\analyze_threads.js
// Run: dx @$scriptContents.analyzeThreads()
```

### 2. Time Travel Debugging (TTD)

**Record and replay execution:**
```powershell
# Record execution
ttd.exe -out recording.run debug_example.exe

# Replay in WinDbg
windbg.exe -o recording.run
```
```
# TTD commands
0:000> !tt.start    ; Start recording
0:000> !tt.stop     ; Stop recording
0:000> !tt.step -1  ; Step backward
0:000> !tt.trace    ; Show trace
```

### 3. Custom Extensions

**Loading and using extensions:**
```
# Load extension
0:000> .load C:\extensions\myext.dll

# Common extensions
0:000> .load psscor4   ; .NET debugging
0:000> .load sosex     ; Enhanced .NET debugging
0:000> .load wow64exts ; WOW64 debugging
```

## Learning Objectives

By the end of this section, you should be able to:

- **Set up and configure WinDbg** with appropriate symbol and source paths
- **Attach to live processes** and perform interactive debugging
- **Analyze crash dumps** to identify root causes of application failures
- **Navigate the WinDbg interface** efficiently using both GUI and command line
- **Set and manage breakpoints** including conditional and hardware breakpoints
- **Examine memory, registers, and call stacks** to understand program state
- **Debug multi-threaded applications** and identify synchronization issues
- **Perform basic heap and memory analysis** to detect leaks and corruption
- **Use advanced features** like Time Travel Debugging and JavaScript automation
- **Debug kernel-mode components** and analyze system-level issues

### Self-Assessment Checklist

Before proceeding to advanced topics, ensure you can:

□ Successfully attach WinDbg to a running process  
□ Set breakpoints and examine program state when they're hit  
□ Navigate through call stacks and examine local variables  
□ Load symbols and understand symbol resolution  
□ Analyze a simple crash dump and identify the cause  
□ Identify and debug race conditions in multi-threaded code  
□ Use basic memory analysis commands to examine heap state  
□ Write and execute simple WinDbg scripts  
□ Configure WinDbg workspace for efficient debugging sessions  

### Practical Exercises

**Exercise 1: Basic Process Debugging**
```c
// TODO: Debug this program and find why it crashes
#include <windows.h>
#include <stdio.h>

int main() {
    char* buffer = (char*)malloc(100);
    strcpy(buffer, "Hello World");
    free(buffer);
    printf("%s\n", buffer);  // Use after free!
    return 0;
}
```

**Exercise 2: Multi-threading Issues**
```c
// TODO: Use WinDbg to identify why the final count is incorrect
int g_count = 0;
DWORD WINAPI CounterThread(LPVOID param) {
    for (int i = 0; i < 10000; i++) {
        g_count++;  // Race condition
    }
    return 0;
}
```

**Exercise 3: Memory Leak Detection**
```c
// TODO: Use heap analysis to find the memory leak
void LeakyFunction() {
    for (int i = 0; i < 100; i++) {
        char* leak = (char*)malloc(1024);
        // Intentionally not freeing memory
    }
}
```

## Study Materials

### Recommended Reading
- **Primary:** "Advanced Windows Debugging" by Mario Hewardt and Daniel Pravat
- **Reference:** "Windows Internals" by Mark Russinovich and David Solomon
- **Online:** [WinDbg Documentation](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/) - Microsoft Docs
- **Tutorials:** [Defrag Tools](https://channel9.msdn.com/Shows/Defrag-Tools) - Video series on debugging

### Video Resources
- "WinDbg Basics" - Channel 9 Microsoft
- "Advanced Debugging Techniques" - PluralSight
- "Crash Dump Analysis" - YouTube debugging tutorials

### Hands-on Labs
- **Lab 1:** Debug a multi-threaded race condition
- **Lab 2:** Analyze a heap corruption crash dump
- **Lab 3:** Set up kernel debugging environment
- **Lab 4:** Create custom debugging scripts

### Practice Scenarios

**Beginner Level:**
1. Attach to calculator.exe and examine its modules
2. Create a simple crash dump and analyze it
3. Debug a program with a null pointer dereference
4. Set conditional breakpoints on function entry

**Intermediate Level:**
5. Debug a deadlock between two threads
6. Analyze heap corruption in a C++ application
7. Use Time Travel Debugging to find an elusive bug
8. Profile memory usage of a long-running service

**Advanced Level:**
9. Debug a kernel driver loading issue
10. Analyze a complex multi-threaded race condition
11. Create custom WinDbg extension for specific debugging needs
12. Set up automated crash dump analysis pipeline

### Development Environment Setup

**Required Tools:**
```powershell
# Install Windows SDK (includes WinDbg)
winget install Microsoft.WindowsSDK

# Install WinDbg Preview
winget install Microsoft.WinDbg

# Install debugging tools
winget install Microsoft.ProcessMonitor
winget install Microsoft.ProcessExplorer
winget install Microsoft.Sysinternals.Suite
```

**Environment Configuration:**
```batch
REM Set up symbol server
set _NT_SYMBOL_PATH=srv*C:\symbols*https://msdl.microsoft.com/download/symbols

REM Set up source server
set _NT_SOURCE_PATH=srv*

REM Configure debugger
reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\AeDebug" /v Debugger /t REG_SZ /d "windbg.exe -p %ld -e %ld -g"
```

**Useful Registry Settings:**
```batch
REM Enable automatic dump creation
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpType /t REG_DWORD /d 2
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpFolder /t REG_SZ /d "C:\CrashDumps"
```

