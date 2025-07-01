# Windows Crash Dump Analysis

*Duration: 1-2 weeks*

## Overview

Windows crash dump analysis is a critical skill for developers and system administrators working with Windows systems. This comprehensive guide covers how to analyze crash dumps, understand bug check codes (Blue Screen of Death), resolve system failures, and use WinDbg effectively for post-mortem debugging.

### What You'll Learn
- Understanding different types of crash dumps
- Setting up crash dump generation
- Using WinDbg for crash dump analysis
- Interpreting bug check codes and stack traces
- Analyzing kernel and user-mode crashes
- Common crash patterns and their solutions

## Types of Windows Crash Dumps

### 1. User-Mode Crash Dumps

**Mini Dump**: Contains minimal information (stack traces, loaded modules)
**Full Dump**: Contains complete process memory
**Heap Dump**: Contains heap information for memory leak analysis

### 2. Kernel-Mode Crash Dumps

**Complete Memory Dump**: Contains entire physical memory
**Kernel Memory Dump**: Contains only kernel memory
**Small Memory Dump (Mini Dump)**: Contains minimal kernel information
**Automatic Memory Dump**: Dynamically sized based on available disk space

### Visual Representation
```
Windows System
├── User Mode Applications
│   ├── Process 1 ──────► Mini/Full Dump
│   ├── Process 2 ──────► Mini/Full Dump
│   └── Process N ──────► Mini/Full Dump
└── Kernel Mode
    ├── Drivers ─────────► Kernel Dump
    ├── System Services ──► Kernel Dump
    └── Hardware Layer ───► Complete Dump
```

## Setting Up Crash Dump Generation

### Configuring System Crash Dumps

**Method 1: System Properties (GUI)**
```
1. Right-click "This PC" → Properties
2. Advanced system settings
3. Startup and Recovery → Settings
4. Under "System failure":
   - Write debugging information: Select dump type
   - Dump file location: %SystemRoot%\MEMORY.DMP
   - Overwrite any existing file: Check if desired
```

**Method 2: Registry Configuration**
```batch
REM Configure crash dump settings via registry
reg add "HKLM\SYSTEM\CurrentControlSet\Control\CrashControl" /v CrashDumpEnabled /t REG_DWORD /d 1 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\CrashControl" /v DumpFile /t REG_EXPAND_SZ /d "%SystemRoot%\MEMORY.DMP" /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\CrashControl" /v NMICrashDump /t REG_DWORD /d 1 /f
```

**Method 3: PowerShell Configuration**
```powershell
# Configure crash dump settings
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl" -Name "CrashDumpEnabled" -Value 1
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl" -Name "DumpFile" -Value "%SystemRoot%\MEMORY.DMP"

# Verify settings
Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl"
```

### Crash Dump Types Configuration

| Registry Value | Dump Type | Description |
|----------------|-----------|-------------|
| 0 | None | No crash dump generated |
| 1 | Complete | Full physical memory |
| 2 | Kernel | Kernel memory only |
| 3 | Small (64KB) | Minimal information |
| 7 | Automatic | System-determined size |

### Enabling User-Mode Crash Dumps

**Windows Error Reporting Configuration:**
```batch
REM Enable local dump collection
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpFolder /t REG_EXPAND_SZ /d "C:\CrashDumps" /f
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpType /t REG_DWORD /d 2 /f
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpCount /t REG_DWORD /d 10 /f

REM Create dump directory
mkdir C:\CrashDumps
```

## Generating Test Crash Dumps

### Example 1: Simple Access Violation
```c
#include <windows.h>
#include <DbgHelp.h>
#include <stdio.h>

// Function to generate mini dump
BOOL CreateMiniDump(EXCEPTION_POINTERS* pExceptionInfo) {
    HANDLE hFile = CreateFile(L"crash_dump.dmp", 
                             GENERIC_WRITE, 
                             0, 
                             NULL, 
                             CREATE_ALWAYS, 
                             FILE_ATTRIBUTE_NORMAL, 
                             NULL);
    
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("Failed to create dump file\n");
        return FALSE;
    }
    
    MINIDUMP_EXCEPTION_INFORMATION mdei;
    mdei.ThreadId = GetCurrentThreadId();
    mdei.ExceptionPointers = pExceptionInfo;
    mdei.ClientPointers = FALSE;
    
    BOOL success = MiniDumpWriteDump(GetCurrentProcess(),
                                   GetCurrentProcessId(),
                                   hFile,
                                   MiniDumpNormal,
                                   &mdei,
                                   NULL,
                                   NULL);
    
    CloseHandle(hFile);
    
    if (success) {
        printf("Mini dump created successfully: crash_dump.dmp\n");
    } else {
        printf("Failed to create mini dump. Error: %lu\n", GetLastError());
    }
    
    return success;
}

// Exception filter for structured exception handling
LONG WINAPI ExceptionFilter(EXCEPTION_POINTERS* pExceptionInfo) {
    printf("Exception caught! Creating dump...\n");
    CreateMiniDump(pExceptionInfo);
    return EXCEPTION_EXECUTE_HANDLER;
}

int main() {
    // Set up structured exception handling
    SetUnhandledExceptionFilter(ExceptionFilter);
    
    printf("About to cause an access violation...\n");
    
    // Simulate different types of crashes
    int* p = NULL;
    *p = 42;  // Access violation - writing to NULL pointer
    
    return 0;
}
```

### Example 2: Stack Overflow Crash
```c
#include <windows.h>
#include <stdio.h>

// Recursive function to cause stack overflow
int StackOverflowFunction(int depth) {
    char largeArray[10000];  // Consume stack space
    
    printf("Recursion depth: %d\n", depth);
    
    // Fill array to prevent optimization
    for (int i = 0; i < 10000; i++) {
        largeArray[i] = (char)(depth % 256);
    }
    
    // Infinite recursion
    return StackOverflowFunction(depth + 1) + largeArray[0];
}

int main() {
    printf("Starting stack overflow test...\n");
    
    __try {
        StackOverflowFunction(0);
    }
    __except(EXCEPTION_EXECUTE_HANDLER) {
        printf("Stack overflow exception caught!\n");
        printf("Exception code: 0x%08X\n", GetExceptionCode());
    }
    
    return 0;
}
```

### Example 3: Heap Corruption
```c
#include <windows.h>
#include <stdio.h>

int main() {
    printf("Testing heap corruption...\n");
    
    // Enable heap debugging
    HANDLE heap = GetProcessHeap();
    
    // Allocate memory
    char* buffer = (char*)HeapAlloc(heap, 0, 100);
    if (!buffer) {
        printf("Failed to allocate memory\n");
        return 1;
    }
    
    printf("Buffer allocated at: %p\n", buffer);
    
    // Cause buffer overflow (heap corruption)
    for (int i = 0; i < 200; i++) {  // Write beyond allocated size
        buffer[i] = 'A';
    }
    
    printf("Buffer overflow completed\n");
    
    // This may crash during free or later heap operations
    HeapFree(heap, 0, buffer);
    
    printf("Memory freed\n");
    return 0;
}
```

### Compilation Instructions
```batch
REM Compile with debug information
cl /Zi /MTd crash_test.c /link /DEBUG Dbghelp.lib

REM Or using GCC/MinGW
gcc -g -o crash_test.exe crash_test.c -lDbghelp

REM Run the program
crash_test.exe
```

*Run these programs to generate crash dumps, then analyze the dump files in WinDbg.*

## WinDbg Crash Dump Analysis

### Installing and Setting Up WinDbg

**Download WinDbg:**
- **WinDbg Preview**: From Microsoft Store (recommended)
- **Classic WinDbg**: Part of Windows SDK
- **Command Line**: `winget install Microsoft.WinDbg`

**Setting Up Symbol Path:**
```batch
REM Set symbol path for Microsoft symbols
_NT_SYMBOL_PATH=srv*c:\symbols*https://msdl.microsoft.com/download/symbols

REM Or set in WinDbg
.sympath srv*c:\symbols*https://msdl.microsoft.com/download/symbols
.reload
```

### Opening and Initial Analysis

**Opening a Dump File:**
```
1. File → Open Dump File (Ctrl+D)
2. Or: windbg -z crash_dump.dmp
3. Or: windbg -z C:\Windows\MEMORY.DMP
```

**Initial Commands:**
```windbg
# Basic information about the dump
.dumpdebug

# Show loaded modules
lm

# Display exception information
.exr -1

# Show current thread's stack
k

# Show all threads
~*k

# Analyze the dump automatically
!analyze -v
```

### Essential WinDbg Commands for Crash Analysis

#### 1. Basic Information Commands

```windbg
# Display dump file information
.dumpdebug

# Show process information
|

# Show current thread
~

# Show all threads
~*

# Switch to thread (e.g., thread 0)
~0s

# Show registers
r

# Show exception record
.exr -1

# Show exception context
.cxr -1
```

#### 2. Stack Analysis Commands

```windbg
# Call stack for current thread
k

# Call stack with parameters
kp

# Call stack with source lines (if available)
kn

# Call stack for all threads
~*k

# Call stack with frame numbers and addresses
kf

# Set stack trace depth
.kframes 100
```

#### 3. Memory Analysis Commands

```windbg
# Display memory at address
d <address>

# Display memory as characters
da <address>

# Display memory as Unicode
du <address>

# Display memory as DWORDs
dd <address>

# Display memory as QWORDs
dq <address>

# Search for pattern in memory
s -a 0 L?80000000 "error"

# Display heap information
!heap

# Display heap statistics
!heap -s
```

#### 4. Module and Symbol Commands

```windbg
# List loaded modules
lm

# List modules with details
lmv

# Load symbols for a module
.reload /f module_name

# Show symbol information
x module_name!*

# Display symbol at address
ln <address>

# Set source path
.srcpath C:\source\path
```

### Step-by-Step Crash Analysis Procedure

#### Phase 1: Initial Assessment

```windbg
# 1. Open the dump file
File → Open Dump File

# 2. Set symbol path (critical!)
.sympath srv*c:\symbols*https://msdl.microsoft.com/download/symbols
.reload

# 3. Get automatic analysis
!analyze -v
```

**Sample Output Analysis:**
```
FAULTING_IP: 
crash_test!main+0x1f
00007ff6`8b7c101f 89 10            mov     dword ptr [rax],edx

EXCEPTION_RECORD:  (.exr -1)
ExceptionAddress: 00007ff68b7c101f (crash_test!main+0x0000000000000020)
   ExceptionCode: c0000005 (Access violation)
  ExceptionFlags: 00000000
NumberParameters: 2
   Parameter[0]: 0000000000000001
   Parameter[1]: 0000000000000000
Attempt to write to address 0000000000000000
```

#### Phase 2: Exception Analysis

```windbg
# Display exception record
.exr -1

# Set context to exception
.cxr -1

# Show stack at time of exception
k

# Show faulting instruction
u @rip L5
```

**Understanding Exception Codes:**
- `0xC0000005`: Access Violation
- `0xC00000FD`: Stack Overflow
- `0xC0000096`: Privileged Instruction
- `0x80000003`: Breakpoint
- `0xC000001D`: Illegal Instruction

#### Phase 3: Thread and Stack Analysis

```windbg
# Show all threads and their states
!runaway

# Analyze each thread's stack
~*kv

# Look for interesting threads
~*e !clrstack

# Check for deadlocks
!locks

# Display critical sections
!cs -l
```

**Stack Trace Interpretation:**
```
Child-SP          RetAddr           Call Site
00000000`0018ff38 00007ffc`c5d0257d ntdll!NtTerminateProcess+0x14
00000000`0018ff40 00007ff6`8b7c1046 KERNEL32!ExitProcess+0x5d
00000000`0018ff70 00007ffc`c5d02d4d crash_test!main+0x46
00000000`0018ffa0 00007ffc`c6b0e58b KERNEL32!BaseThreadInitThunk+0x1d
00000000`0018ffd0 00000000`00000000 ntdll!RtlUserThreadStart+0x2b
```

#### Phase 4: Memory and Heap Analysis

```windbg
# Check heap state
!heap -s

# Look for heap corruption
!heap -x

# Display heap entries
!heap -h <heap_address>

# Check for memory leaks
!heap -stat

# Analyze virtual memory
!vm

# Check memory usage
!address
```

#### Phase 5: Module and Driver Analysis

```windbg
# List all modules
lmv

# Check for unsigned drivers
!verifier

# Display driver information
!drvobj

# Check system integrity
!chkimg -d <module_name>

# Analyze crash in driver
!irp
!devobj
```

### Common Crash Patterns and Analysis

#### 1. Access Violation (0xC0000005)

**Analysis Steps:**
```windbg
# Check faulting address
.exr -1

# Look at the instruction
u @rip L5

# Check register values
r

# Examine memory around the fault
db @rax L50

# Check if it's a NULL pointer dereference
dt @rax
```

**Example Analysis:**
```windbg
0:000> .exr -1
ExceptionAddress: 00401020 (crash_test!main+0x20)
   ExceptionCode: c0000005 (Access violation)
   Parameter[0]: 0000000000000001  # Write attempt
   Parameter[1]: 0000000000000000  # Address 0 (NULL)

0:000> u @rip L3
crash_test!main+0x20:
00401020 89 10    mov     dword ptr [rax],edx  # Writing to [rax]
00401022 33 c0    xor     eax,eax
00401024 c3       ret

0:000> r rax
rax=0000000000000000  # RAX is NULL!
```

#### 2. Stack Overflow (0xC00000FD)

**Analysis Steps:**
```windbg
# Check stack limits
!teb

# Look at stack usage
kf

# Check for infinite recursion
kv 100

# Examine stack pattern
dps @rsp L50
```

#### 3. Heap Corruption

**Analysis Steps:**
```windbg
# Enable heap debugging
!gflag +hpa

# Check heap integrity
!heap -x

# Look for corruption patterns
!heap -p -a <address>

# Check heap statistics
!heap -s
```

### Advanced Analysis Techniques

#### 1. Time Travel Debugging (TTD)

```batch
REM Record execution for analysis
ttd.exe -out recording.run myapp.exe

REM Open recording in WinDbg
windbg -ttd recording.run
```

**TTD Commands:**
```windbg
# Navigate to specific position
!tt 0:1A4

# Search for events
!tt 100:0

# Go to exception
!tt.ex

# Step backwards
p-

# Run backwards to previous call
gu-
```

#### 2. JavaScript Debugging in WinDbg

```javascript
// WinDbg JavaScript example
"use strict";

function findCrashCause() {
    let control = host.namespace.Debugger.Utility.Control;
    let output = control.ExecuteCommand("!analyze -v");
    
    for (let line of output) {
        if (line.includes("FAULTING_IP")) {
            host.diagnostics.debugLog("Found faulting instruction: " + line);
        }
    }
}

// Usage: dx @$scriptContents.findCrashCause()
```

#### 3. Custom Analysis Scripts

**PowerShell Integration:**
```powershell
# Automate WinDbg analysis
$windbgPath = "C:\Program Files\WindowsApps\Microsoft.WinDbg_*\windbg.exe"
$dumpFile = "C:\crash_dump.dmp"
$scriptFile = "analysis_script.txt"

# Create analysis script
@"
.sympath srv*c:\symbols*https://msdl.microsoft.com/download/symbols
.reload
!analyze -v
q
"@ | Out-File $scriptFile

# Run automated analysis
& $windbgPath -z $dumpFile -cf $scriptFile
```

### Practical Lab Exercises

#### Exercise 1: Basic Crash Analysis
```c
// Create this program and analyze its crash
#include <stdio.h>
#include <stdlib.h>

int main() {
    char* ptr = (char*)malloc(100);
    free(ptr);
    
    // Use after free bug
    strcpy(ptr, "This will crash!");
    
    return 0;
}
```

**Tasks:**
1. Compile and run the program
2. Collect the crash dump
3. Use WinDbg to identify the use-after-free bug
4. Document the analysis steps

#### Exercise 2: Multi-threaded Crash
```c
// Multi-threaded crash scenario
#include <windows.h>
#include <stdio.h>

int shared_data = 0;

DWORD WINAPI thread_function(LPVOID param) {
    for (int i = 0; i < 1000000; i++) {
        shared_data++;  // Race condition
    }
    return 0;
}

int main() {
    HANDLE threads[4];
    
    for (int i = 0; i < 4; i++) {
        threads[i] = CreateThread(NULL, 0, thread_function, NULL, 0, NULL);
    }
    
    WaitForMultipleObjects(4, threads, TRUE, INFINITE);
    
    printf("Final value: %d\n", shared_data);
    return 0;
}
```

**Tasks:**
1. Run and observe inconsistent results
2. Add crash-inducing code
3. Analyze the multi-threaded crash dump
4. Identify synchronization issues

### Best Practices for Crash Dump Analysis

#### 1. Environment Setup
- ✅ Always set up proper symbol paths
- ✅ Keep symbols and binaries in sync
- ✅ Use source server when available
- ✅ Enable full dumps for critical issues

#### 2. Analysis Methodology
- ✅ Start with `!analyze -v`
- ✅ Verify symbol loading
- ✅ Check all threads, not just the faulting one
- ✅ Look for patterns in recurring crashes
- ✅ Document your analysis steps

#### 3. Common Mistakes to Avoid
- ❌ Analyzing without proper symbols
- ❌ Focusing only on the faulting thread
- ❌ Ignoring module versions
- ❌ Not checking heap state
- ❌ Overlooking environmental factors

### Tools and Resources

#### Essential Tools
- **WinDbg Preview**: Modern debugging interface
- **Application Verifier**: Runtime verification
- **CDB**: Console debugger
- **DebugDiag**: Automated analysis tool
- **ProcDump**: Dump generation utility

#### Symbol Management
```batch
REM Download symbols manually
symchk /r c:\myapp *.exe /s srv*c:\symbols*https://msdl.microsoft.com/download/symbols

REM Verify symbols
symchk myapp.exe /s c:\symbols

REM Symbol server setup
set _NT_SYMBOL_PATH=srv*c:\localsymbols*\\server\symbols*https://msdl.microsoft.com/download/symbols
```

#### Automation Scripts
```batch
REM Automated dump analysis
@echo off
set DUMP_FILE=%1
set OUTPUT_FILE=%2

windbg -z %DUMP_FILE% -c ".sympath srv*c:\symbols*https://msdl.microsoft.com/download/symbols; .reload; !analyze -v; q" -logo %OUTPUT_FILE%

echo Analysis complete. Results in %OUTPUT_FILE%
```

## Learning Objectives

By the end of this section, you should be able to:
- **Configure crash dump generation** for both user-mode and kernel-mode applications
- **Use WinDbg effectively** to analyze crash dumps and identify root causes
- **Interpret exception codes** and understand common crash patterns
- **Analyze stack traces** and identify the sequence of events leading to crashes
- **Examine memory and heap state** to detect corruption and leaks
- **Set up proper debugging environments** with symbols and source code
- **Create automated analysis workflows** for recurring crash investigations
- **Apply advanced debugging techniques** like Time Travel Debugging

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Generate different types of crash dumps programmatically  
□ Configure Windows crash dump settings via registry and GUI  
□ Open and perform initial analysis of dump files in WinDbg  
□ Set up symbol paths and verify symbol loading  
□ Interpret exception records and identify crash causes  
□ Analyze stack traces across multiple threads  
□ Use memory analysis commands to detect heap corruption  
□ Identify common crash patterns (access violations, stack overflows)  
□ Create and use custom analysis scripts  
□ Document analysis findings for team collaboration  

## Study Materials

### Recommended Reading
- **"Advanced Windows Debugging"** by Mario Hewardt and Daniel Pravat
- **"Windows Internals"** (Part 1 & 2) by Mark Russinovich and David Solomon
- **Microsoft Docs**: [Crash Dump Analysis](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/)
- **"Debugging Applications"** by John Robbins

### Video Resources
- **Channel 9**: WinDbg tutorials and case studies
- **Microsoft Learn**: Debugging Windows applications
- **Pluralsight**: Advanced debugging techniques

### Hands-on Labs
- **Lab 1**: User-mode application crash analysis
- **Lab 2**: Kernel-mode driver crash investigation
- **Lab 3**: Multi-threaded synchronization issues
- **Lab 4**: Memory corruption detection and analysis

### Practice Scenarios

**Scenario 1: Production Web Server Crash**
```
- Random crashes during high load
- Multiple threads involved
- Possible memory leak
- Intermittent network issues
```

**Analysis Tasks:**
- Collect crash dumps during peak hours
- Identify memory usage patterns
- Check for resource leaks
- Analyze thread synchronization

**Scenario 2: Device Driver Blue Screen**
```
- BSOD on specific hardware
- Bug check 0x1E (KMODE_EXCEPTION_NOT_HANDLED)
- Third-party driver suspected
- Occurs during system resume
```

**Analysis Tasks:**
- Examine kernel crash dump
- Identify faulting driver
- Check driver signatures and versions
- Analyze hardware interaction patterns

### Development Environment Setup

**Required Software:**
```batch
REM Install debugging tools
winget install Microsoft.WinDbg
winget install Microsoft.WindowsSDK

REM Set up environment variables
setx _NT_SYMBOL_PATH "srv*c:\symbols*https://msdl.microsoft.com/download/symbols"
setx _NT_SOURCE_PATH "srv*c:\source*https://source.server.com"

REM Create directories
mkdir C:\symbols
mkdir C:\dumps
mkdir C:\analysis
```

**Compiler Settings for Better Debugging:**
```batch
REM MSVC with full debug info
cl /Zi /Od /MDd /RTC1 myapp.c /link /DEBUG:FULL

REM GCC with debug symbols
gcc -g -O0 -rdynamic myapp.c -o myapp.exe
```

**Registry Settings for Enhanced Crash Analysis:**
```batch
REM Enable user-mode crash dumps
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpType /t REG_DWORD /d 2 /f
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpFolder /t REG_EXPAND_SZ /d "C:\dumps" /f

REM Enable automatic crash dump analysis
reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting" /v DontShowUI /t REG_DWORD /d 1 /f
```

This comprehensive crash dump analysis guide provides the foundation for effective Windows debugging and troubleshooting skills essential for professional software development and system administration.
