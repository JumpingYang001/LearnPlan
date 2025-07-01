# Advanced WinDbg Debugging

*Duration: 2 weeks*

## Overview

WinDbg is Microsoft's flagship debugging tool for Windows applications, drivers, and system-level components. This comprehensive guide covers advanced debugging techniques including stack tracing, thread analysis, exception handling, memory analysis, and performance profiling. Whether you're debugging user-mode applications, kernel drivers, or analyzing crash dumps, WinDbg provides powerful capabilities for deep system analysis.

## Prerequisites

- Basic understanding of C/C++ programming
- Familiarity with Windows operating system concepts
- Basic knowledge of assembly language (helpful but not required)
- Understanding of memory management and pointers

## WinDbg Setup and Configuration

### Installation
```powershell
# Install Windows SDK (includes WinDbg)
# Download from: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/

# Or install WinDbg Preview from Microsoft Store (recommended)
# Provides modern UI and enhanced features
```

### Essential Configuration
```windbg
# Set symbol path for Microsoft symbols
.sympath srv*c:\symbols*https://msdl.microsoft.com/download/symbols

# Set source path
.srcpath c:\source;c:\projects

# Enable verbose mode
.verbose

# Load all symbols
.reload /f

# Set up for C++ debugging
.prefer_dml 1

# Configure automatic source loading
.lines
```

## Advanced Stack Tracing Techniques

### Basic Stack Trace Analysis

**Sample Program for Stack Tracing:**
```c
#include <stdio.h>
#include <windows.h>

void funcD() {
    int d = 4;
    printf("In funcD: d = %d\n", d);
    
    // Intentional access violation for debugging
    int* null_ptr = NULL;
    *null_ptr = 42;  // This will cause an exception
}

void funcC() {
    int c = 3;
    printf("In funcC: c = %d\n", c);
    funcD();
}

void funcB() {
    int y = 2;
    printf("In funcB: y = %d\n", y);
    funcC();
}

void funcA() {
    int x = 1;
    printf("In funcA: x = %d\n", x);
    funcB();
}

int main() {
    printf("Starting program...\n");
    funcA();
    return 0;
}
```

### Comprehensive Stack Analysis Commands

**Basic Stack Commands:**
```windbg
# Display call stack
k

# Display call stack with parameters
kp

# Display call stack with local variables
kv

# Display call stack with source lines
kL

# Display call stack for all threads
~*k

# Display detailed stack with frame numbers
kn

# Display stack with full module names
kf
```

**Advanced Stack Analysis:**
```windbg
# Display stack with memory addresses
kb

# Show stack for specific thread (thread 2)
~2k

# Display stack with calling convention
kp

# Limit stack display to 10 frames
k 10

# Display stack with DML (Debugger Markup Language) links
kM

# Show stack backtrace with function offsets
k =b

Example Output:
0:000> kv
 # ChildEBP RetAddr  Args to Child              
00 0019ff88 00401234 00000001 00427f44 00427f54 stacktrace!funcD+0x1c
01 0019ff98 00401245 00000002 00000000 00000000 stacktrace!funcC+0x14
02 0019ffa8 00401256 00000000 00000000 00000000 stacktrace!funcB+0x15
03 0019ffb8 00401267 00000000 00000000 00000000 stacktrace!funcA+0x16
04 0019ffc8 76e0336a 00000000 00000000 00000000 stacktrace!main+0x17
05 0019ffdc 77bb9902 fffffffe 77bb98c5 fffffffe kernel32!BaseThreadInitThunk+0xe
06 0019ff1c 77bb98d5 00401250 fffffffe 00000000 ntdll!__RtlUserThreadStart+0x70
07 0019ff34 00000000 00401250 00000000 00000000 ntdll!_RtlUserThreadStart+0x1b
```

### Stack Frame Analysis

**Examining Individual Stack Frames:**
```windbg
# Switch to frame 0 (current frame)
.frame 0

# Switch to frame 1 (caller)
.frame 1

# Display current frame info
.frame

# Display local variables for current frame
dv

# Display parameters for current frame
dv /t /v

# Display local variables with types and values
dv /t /v

Example:
0:000> .frame 1
01 0019ff98 00401245 stacktrace!funcC+0x14
0:000> dv /t /v
@ebp+0x08 @ecx          int c = 0n3
```

### Advanced Stack Corruption Detection

**Detecting Stack Corruption:**
```windbg
# Check stack integrity
!analyze -hang

# Verify stack cookies (if enabled)
!chkimg ntdll

# Display stack limits
!teb

# Check for stack overflow
!analyze -v

# Manual stack walking
dps esp L20

Example Stack Corruption Detection:
0:000> !analyze -v
EXCEPTION_RECORD:  (.exr -1)
ExceptionAddress: 00401234
   ExceptionCode: c0000005 (Access violation)
  ExceptionFlags: 00000000
NumberParameters: 2
   Parameter[0]: 00000001
   Parameter[1]: 00000000

STACK_TEXT:
0019ff88 00401234 stacktrace!funcD+0x1c [CORRUPTED STACK]
WARNING: Stack unwind information not available
```

## Advanced Thread Analysis

### Multi-threaded Application Example

**Sample Multi-threaded Program:**
```c
#include <stdio.h>
#include <windows.h>
#include <process.h>

CRITICAL_SECTION g_cs;
int g_shared_counter = 0;
HANDLE g_events[3];

// Thread that increments counter
unsigned __stdcall WorkerThread(void* pArguments) {
    int thread_id = (int)(uintptr_t)pArguments;
    
    for (int i = 0; i < 1000; i++) {
        EnterCriticalSection(&g_cs);
        g_shared_counter++;
        printf("Thread %d: Counter = %d\n", thread_id, g_shared_counter);
        LeaveCriticalSection(&g_cs);
        
        Sleep(10);
    }
    
    SetEvent(g_events[thread_id - 1]);
    return 0;
}

// Thread that waits for an event
unsigned __stdcall WaitingThread(void* pArguments) {
    printf("Waiting thread started...\n");
    
    // This thread will wait indefinitely
    WaitForSingleObject(g_events[2], INFINITE);
    
    printf("Waiting thread signaled!\n");
    return 0;
}

// Thread that causes a deadlock
unsigned __stdcall DeadlockThread(void* pArguments) {
    printf("Deadlock thread acquiring critical section...\n");
    
    EnterCriticalSection(&g_cs);
    
    // Sleep while holding the lock to cause contention
    Sleep(30000);  // 30 seconds
    
    LeaveCriticalSection(&g_cs);
    return 0;
}

int main() {
    InitializeCriticalSection(&g_cs);
    
    // Create events
    for (int i = 0; i < 3; i++) {
        g_events[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
    }
    
    // Create worker threads
    HANDLE hThread1 = (HANDLE)_beginthreadex(NULL, 0, WorkerThread, (void*)1, 0, NULL);
    HANDLE hThread2 = (HANDLE)_beginthreadex(NULL, 0, WorkerThread, (void*)2, 0, NULL);
    
    // Create waiting thread
    HANDLE hWaitingThread = (HANDLE)_beginthreadex(NULL, 0, WaitingThread, NULL, 0, NULL);
    
    // Create deadlock thread after some delay
    Sleep(5000);
    HANDLE hDeadlockThread = (HANDLE)_beginthreadex(NULL, 0, DeadlockThread, NULL, 0, NULL);
    
    // Wait for worker threads
    WaitForSingleObject(hThread1, INFINITE);
    WaitForSingleObject(hThread2, INFINITE);
    
    // Cleanup
    CloseHandle(hThread1);
    CloseHandle(hThread2);
    CloseHandle(hWaitingThread);
    CloseHandle(hDeadlockThread);
    
    for (int i = 0; i < 3; i++) {
        CloseHandle(g_events[i]);
    }
    
    DeleteCriticalSection(&g_cs);
    return 0;
}
```

### Thread Analysis Commands

**Basic Thread Commands:**
```windbg
# List all threads
~

# Display detailed thread information
~*

# Switch to thread 0
~0s

# Display current thread
~.

# Display thread 2's stack
~2k

# Display all thread stacks
~*k

# Show thread-specific information
!thread

# Display thread environment block (TEB)
!teb
```

**Advanced Thread Analysis:**
```windbg
# Display thread states
~*e !thread @$thread

# Show thread wait analysis
!analyze -hang

# Display critical section information
!cs -l

# Show all locks held by threads
!locks

# Display thread pool information
!threadpool

# Show runaway threads (CPU usage)
!runaway

Example Output:
0:000> ~
   0  Id: 1234.5678 Suspend: 1 Teb: 7efdd000 Unfrozen
   1  Id: 1234.5679 Suspend: 0 Teb: 7efdc000 Unfrozen
   2  Id: 1234.567a Suspend: 0 Teb: 7efdb000 Unfrozen
.  3  Id: 1234.567b Suspend: 0 Teb: 7efda000 Unfrozen

0:003> ~*k
   0  Id: 1234.5678 Suspend: 1 Teb: 7efdd000 Unfrozen
# Child-SP          RetAddr           Call Site
00 00000048`0006f8c8 00007ff8`4e8d1fe4 ntdll!ZwWaitForSingleObject+0x14
01 00000048`0006f8d0 00007ff8`4d8e1398 KERNELBASE!WaitForSingleObjectEx+0x94
02 00000048`0006f970 00007ff8`4d8e13ee threadtest!WaitingThread+0x28
```

### Deadlock Detection and Analysis

**Detecting Deadlocks:**
```windbg
# Comprehensive deadlock analysis
!analyze -hang -v

# Display critical section details
!cs -l -o

# Show blocked threads
!locks -v

# Display wait chain analysis
!wct

# Manual deadlock investigation
~*e .echo "Thread"; ~ ; k 5

Example Deadlock Analysis:
0:000> !analyze -hang
HANG_ANALYSIS: 

BLOCKED_THREAD:  00001234

BLOCKING_THREAD: 00005678

CRITICAL_SECTION: MyApp!g_cs (00401000)
-----------------------------------------
LOCKED BY: Thread 1 (00005678)
WAITERS: Thread 0 (00001234), Thread 2 (00009abc)

WAIT_CHAIN:
Thread 0 -> Critical Section -> Thread 1 -> Event -> Never signaled
```

### Thread Synchronization Analysis

**Analyzing Synchronization Objects:**
```windbg
# Display all handles in process
!handle

# Show specific handle information
!handle 0x1234 f

# Display critical section details
!cs address

# Show event object information
!object address

# Display mutex information
!mutex address

# Analyze wait chains
!wct -t thread_id

Example Synchronization Analysis:
0:000> !cs 00401000
-----------------------------------------
Critical section   = 0x00401000 (MyApp!g_cs)
DebugInfo          = 0x77bb0520
LOCKED
LockCount          = 0x2
WaiterWoken        = No
OwningThread       = 0x00001234
RecursionCount     = 0x1
LockSemaphore      = 0x0
SpinCount          = 0x00000000
```

### Performance Analysis of Threads

**Thread Performance Commands:**
```windbg
# Display CPU usage by thread
!runaway 7

# Show thread times
!runaway 1

# Display context switches
!runaway 2

# Show kernel/user time breakdown
!runaway 3

Example Performance Output:
0:000> !runaway 7
 User Mode Time
  Thread       Time
   0:1234      0 days 0:00:01.234
   1:5678      0 days 0:00:00.567
   2:9abc      0 days 0:00:00.123

 Kernel Mode Time
  Thread       Time
   0:1234      0 days 0:00:00.045
   1:5678      0 days 0:00:00.023
   2:9abc      0 days 0:00:00.012
```

## Exception Handling and Analysis

### Exception Analysis Example

**Program with Various Exception Types:**
```c
#include <stdio.h>
#include <windows.h>
#include <signal.h>

// Custom exception handler
LONG WINAPI CustomExceptionHandler(PEXCEPTION_POINTERS pExceptionPointers) {
    printf("Custom exception handler called!\n");
    printf("Exception Code: 0x%lx\n", pExceptionPointers->ExceptionRecord->ExceptionCode);
    printf("Exception Address: 0x%p\n", pExceptionPointers->ExceptionRecord->ExceptionAddress);
    
    return EXCEPTION_CONTINUE_SEARCH;  // Let the debugger handle it
}

void AccessViolationExample() {
    printf("Testing access violation...\n");
    
    int* null_ptr = NULL;
    *null_ptr = 42;  // Access violation
}

void DivideByZeroExample() {
    printf("Testing divide by zero...\n");
    
    int a = 10;
    int b = 0;
    int result = a / b;  // Division by zero
    
    printf("Result: %d\n", result);
}

void StackOverflowExample() {
    printf("Testing stack overflow...\n");
    
    // Recursive function without base case
    StackOverflowExample();
}

void IntegerOverflowExample() {
    printf("Testing integer overflow...\n");
    
    int max_int = 2147483647;  // Maximum 32-bit signed integer
    int overflow = max_int + 1;  // This will overflow
    
    printf("Overflow result: %d\n", overflow);
}

void CustomExceptionExample() {
    printf("Testing custom exception...\n");
    
    RaiseException(0xE1234567, 0, 0, NULL);  // Custom exception code
}

int main() {
    // Install custom exception handler
    SetUnhandledExceptionFilter(CustomExceptionHandler);
    
    printf("Exception Analysis Examples\n");
    printf("===========================\n");
    
    int choice;
    printf("Select exception type:\n");
    printf("1. Access Violation\n");
    printf("2. Divide by Zero\n");
    printf("3. Stack Overflow\n");
    printf("4. Integer Overflow\n");
    printf("5. Custom Exception\n");
    printf("Enter choice (1-5): ");
    
    scanf("%d", &choice);
    
    __try {
        switch (choice) {
            case 1:
                AccessViolationExample();
                break;
            case 2:
                DivideByZeroExample();
                break;
            case 3:
                StackOverflowExample();
                break;
            case 4:
                IntegerOverflowExample();
                break;
            case 5:
                CustomExceptionExample();
                break;
            default:
                printf("Invalid choice\n");
                break;
        }
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        printf("Exception caught in main!\n");
    }
    
    return 0;
}
```

### Exception Analysis Commands

**Basic Exception Commands:**
```windbg
# Display exception information
.exr -1

# Display exception context
.cxr -1

# Analyze the exception
!analyze -v

# Display structured exception handling (SEH) chain
!exchain

# Show exception record details
.exr address

# Display context record
.cxr address

Example Exception Analysis:
0:000> .exr -1
ExceptionAddress: 00401234 (myapp!AccessViolationExample+0x12)
   ExceptionCode: c0000005 (Access violation)
  ExceptionFlags: 00000000
NumberParameters: 2
   Parameter[0]: 00000001
   Parameter[1]: 00000000

0:000> !analyze -v
EXCEPTION_RECORD:  (.exr -1)
ExceptionAddress: 00401234 (myapp!AccessViolationExample+0x12)
   ExceptionCode: c0000005 (Access violation)
  ExceptionFlags: 00000000
NumberParameters: 2
   Parameter[0]: 00000001
   Parameter[1]: 00000000

FAULTING_IP: 
myapp!AccessViolationExample+12
00401234 c70500000000    mov     dword ptr [0],0x2a

EXCEPTION_ANALYSIS:
    NULL_POINTER_DEREFERENCE
    WRITE_ADDRESS: 00000000
    FOLLOWUP_IP: myapp!AccessViolationExample+12
```

### Advanced Exception Handling

**SEH (Structured Exception Handling) Analysis:**
```windbg
# Display SEH chain for current thread
!exchain

# Display SEH chain for specific thread
~2!exchain

# Show exception registration records
dt ntdll!_EXCEPTION_REGISTRATION_RECORD

# Display vectored exception handlers
!veh

Example SEH Chain:
0:000> !exchain
0019ff94: myapp!_except_handler4+0 (00401567)
  CRT scope  0, filter: myapp!main+0x89 (004013c9)
                func:   myapp!main+0x8f (004013cf)
0019ffb8: ntdll!_except_handler4+0 (77bb7428)
  CRT scope  0, filter: ntdll!__RtlUserThreadStart+0x62 (77bb9964)
                func:   ntdll!__RtlUserThreadStart+0x6f (77bb9971)
0019ffdc: ntdll!FinalExceptionHandler+0 (77bb98a5)
```

### Exception Breakpoints and Control

**Setting Exception Breakpoints:**
```windbg
# Break on all exceptions
sxe *

# Break on access violations
sxe av

# Break on specific exception code
sxe 0xc0000005

# Ignore specific exceptions
sxi dz

# Break on first chance exceptions
sxe eh

# Break on second chance exceptions
sxe *

# List current exception settings
sx

Example Exception Breakpoint Configuration:
0:000> sx
   ct - Create thread                - ignore
   et - Exit thread                  - ignore  
   cpr- Create process               - ignore
   epr- Exit process                 - ignore
   ld - Load module                  - output
   ud - Unload module                - ignore
   ser- System error                 - ignore
   ibp- Initial breakpoint           - ignore
   iml- Initial module load          - ignore
   
   av - Access violation             - break
   dz - Integer divide by zero       - break
   ii - Illegal instruction          - break
```

## Memory Analysis and Debugging

### Memory Leak Detection

**Memory Leak Example Program:**
```c
#include <stdio.h>
#include <windows.h>
#include <crtdbg.h>

typedef struct {
    int id;
    char data[1024];
    struct Node* next;
} Node;

Node* g_head = NULL;

void CreateMemoryLeak() {
    printf("Creating memory leaks...\n");
    
    // Leak 1: Simple malloc without free
    char* leaked_buffer = (char*)malloc(1000);
    sprintf(leaked_buffer, "This buffer will be leaked!");
    
    // Leak 2: Linked list with missing cleanup
    for (int i = 0; i < 100; i++) {
        Node* new_node = (Node*)malloc(sizeof(Node));
        new_node->id = i;
        sprintf(new_node->data, "Node %d data", i);
        new_node->next = g_head;
        g_head = new_node;
    }
    
    // Intentionally not freeing the linked list
    
    // Leak 3: Windows handle leak
    HANDLE hEvent = CreateEvent(NULL, TRUE, FALSE, L"TestEvent");
    // Intentionally not calling CloseHandle(hEvent)
    
    // Leak 4: GDI object leak
    HDC hdc = GetDC(NULL);
    HBRUSH hBrush = CreateSolidBrush(RGB(255, 0, 0));
    // Intentionally not calling DeleteObject(hBrush) or ReleaseDC
}

void PartialCleanup() {
    // Clean up only some nodes to demonstrate partial leaks
    Node* current = g_head;
    int count = 0;
    
    while (current && count < 50) {  // Only free half the nodes
        Node* next = current->next;
        free(current);
        current = next;
        count++;
    }
    
    g_head = current;  // Update head to remaining nodes
}

int main() {
    // Enable CRT memory leak detection
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    
    printf("Memory Leak Analysis Example\n");
    printf("============================\n");
    
    CreateMemoryLeak();
    PartialCleanup();
    
    printf("Program ending with memory leaks...\n");
    
    // CRT will automatically detect and report leaks on exit
    return 0;
}
```

### Memory Analysis Commands

**Basic Memory Commands:**
```windbg
# Display memory usage statistics
!address

# Show heap information
!heap -stat

# Display virtual memory layout
!vadump

# Check for heap leaks
!heap -l

# Display process memory info
!peb

# Show memory protection
!address address

Example Memory Analysis:
0:000> !address
        BaseAddr EndAddr+1 RgnSize     Type       State                 Protect             Usage
-------------------------------------------------------------------------------------------
           10000    11000     1000 MEM_IMAGE   MEM_COMMIT  PAGE_EXECUTE_READ                  Image
           11000    12000     1000 MEM_IMAGE   MEM_COMMIT  PAGE_READONLY                      Image  
           12000    13000     1000 MEM_IMAGE   MEM_COMMIT  PAGE_READWRITE                     Image
```

**Advanced Memory Debugging:**
```windbg
# Enable page heap for detailed heap debugging
!gflag +hpa myapp.exe

# Display heap blocks
!heap -p -h handle

# Find memory leaks
!heap -l

# Display heap entry details
!heap -p -a address

# Check for heap corruption
!heap -x

# Display allocation stack trace (with page heap)
!heap -p -a address

Example Page Heap Output:
0:000> !heap -p -a 00340000
    address 00340000 found in
    _DPH_HEAP_ROOT @ 4fd1000
    in busy allocation (  DPH_HEAP_BLOCK:         UserAddr         UserSize -         VirtAddr         VirtSize)
                             0034fff0:         00340000             1000 -         0033f000             2000
    
    00007ff8`4d8e2590 verifier!AVrfDebugPageHeapAllocate+0x00000240
    00007ff8`4d8e1198 verifier!AVrfDebugPageHeapReAllocate+0x000002c8
    00007ff8`77bb1398 ntdll!RtlDebugAllocateHeap+0x00000039
    00007ff8`77bb2fe4 ntdll!RtlpAllocateHeap+0x000000c4
    00007ff8`4e8d1234 myapp!CreateMemoryLeak+0x00000024
```

### Application Verifier Integration

**Setting up Application Verifier:**
```cmd
# Enable Application Verifier for heap checking
appverif.exe

# Command line setup
appverif -enable Heaps -for myapp.exe
appverif -enable Handles -for myapp.exe
appverif -enable Locks -for myapp.exe
```

**WinDbg Configuration Script:**
```windbg
$$ Save this as windbg_setup.txt and run: $$>a< windbg_setup.txt

.echo Setting up WinDbg environment...

$$ Set symbol path
.sympath srv*c:\symbols*https://msdl.microsoft.com/download/symbols

$$ Enable source line support
.lines

$$ Set verbose mode
.verbose

$$ Enable DML (Debugger Markup Language)
.prefer_dml 1

$$ Load useful extensions
.load wow64exts
.load uext
.load umext

$$ Set up automatic commands
.echo WinDbg setup completed!
.echo Symbol path: srv*c:\symbols*https://msdl.microsoft.com/download/symbols
.echo Use '.help' for command help
.echo Use '!analyze -v' for automatic crash analysis
```

**Compilation for Debugging:**
```cmd
# Compile with debug information
cl /Zi /Od myapp.c /link /DEBUG

# For Release builds with symbols
cl /O2 /Zi myapp.c /link /DEBUG /OPT:REF /OPT:ICF

# Enable Application Verifier
appverif -enable Heaps Handles Locks -for myapp.exe

# Enable Page Heap for detailed heap debugging
gflags -p /enable myapp.exe /full
```

## Advanced WinDbg Techniques

### Custom Scripts and Automation

**1. WinDbg Scripting Basics:**
- WinDbg supports scripting via the `.cmd` and `.dox` file extensions
- Scripts can automate repetitive tasks, such as loading symbols, setting breakpoints, and analyzing dumps

**2. Example Script - Automated Crash Analysis:**
```cmd
.logopen crash_analysis.log
.sympath srv*c:\symbols*https://msdl.microsoft.com/download/symbols
.reload /f
!analyze -v
!dump

# Custom commands for specific analysis
!heap -l
!address
```
- To run the script: `.cmd /r crash_analysis.cmd`

**3. Advanced Automation with Python and Win32 Extensions:**
- Use Python scripts to control WinDbg via COM automation
- Example: Automatically attach to a process and dump stack traces
```python
import win32com.client

# Connect to WinDbg
dbg = win32com.client.Dispatch("WinDbg.Application")

# Attach to process by ID
dbg.Attach(1234)

# Run commands
dbg.Execute("!analyze -v")
dbg.Execute("kv")

# Detach and exit
dbg.Detach()
dbg.Quit()
```

### Kernel Debugging Basics

**1. Introduction to Kernel Debugging:**
- Kernel debugging is used for diagnosing issues in device drivers, kernel modules, and the Windows kernel itself
- Requires a second computer or virtual machine for debugging target

**2. Setting Up Kernel Debugging:**
```powershell
# Configure target machine (VM or physical)
bcdedit /debug on
bcdedit /set debug on
bcdedit /set debugport com1
bcdedit /set baudrate 115200

# Configure host machine (where WinDbg runs)
windbg -k com:port=COM1,baud=115200
```

**3. Basic Kernel Debugging Commands:**
```windbg
# Display loaded drivers
lm

# Display driver information
!drvobj driver_name

# Display kernel memory usage
!memusage

# Display system uptime
!time

# Display process list
!process

# Display thread list
!thread

# Display current CPU registers
r

# Display memory at address
db address
```
- Use `.sympath` to set symbol path for kernel debugging symbols

### Advanced Kernel Debugging Techniques

**1. Analyzing Crash Dumps:**
- Use `!analyze -v` for automatic analysis of crash dumps
- Examine the call stack, exception record, and thread context

**2. Debugging Device Drivers:**
- Set breakpoints in driver entry points (e.g., `DriverEntry`, `Unload`)
- Use `!devobj` and `!irp` to inspect device objects and I/O request packets

**3. Analyzing Deadlocks and Hangs:**
- Use `!analyze -hang` to diagnose deadlocks
- Examine the call stacks of all threads to identify circular wait conditions

**4. Memory Dump Analysis:**
- Use `!address` to analyze virtual memory layout
- Use `!heap` to analyze heap allocations and detect leaks

**5. Performance Analysis:**
- Use `!runaway` to identify CPU usage by threads
- Use `!process` and `!thread` to analyze context switches and wait chains

## Real-World Debugging Scenarios

### Case Study 1: E-commerce Application Crash

**Background:**
An e-commerce web application crashes intermittently during high-traffic periods, specifically during checkout processing.

**Crash Information:**
```
Application: OnlineStore.exe
Exception Code: 0xC0000005 (Access Violation)
Faulting Address: 0x00000008
Crash Frequency: 2-3 times per hour during peak traffic
```

**Investigation Process:**

**Step 1: Initial Analysis**
```windbg
# Load the crash dump
.opendump C:\CrashDumps\OnlineStore_20241201_143022.dmp

# Automatic analysis
!analyze -v

Output:
FAULTING_IP: 
OnlineStore!ProcessOrder+0x45
00401123 8b4808          mov     ecx,dword ptr [eax+8]

EXCEPTION_RECORD:
ExceptionAddress: 00401123 (OnlineStore!ProcessOrder+0x45)
   ExceptionCode: c0000005 (Access violation)
  ExceptionFlags: 00000000
NumberParameters: 2
   Parameter[0]: 00000000 (read access)
   Parameter[1]: 00000008 (address being read)

STACK_TEXT:
00a2f8c4 00401234 OnlineStore!ProcessOrder+0x45
00a2f8d8 00401456 OnlineStore!HandleCheckout+0x34
00a2f8ec 00401567 OnlineStore!ProcessRequest+0x67
00a2f900 77bb1234 OnlineStore!WorkerThread+0x123
```

**Step 2: Code Analysis**
```c
// Problematic code in ProcessOrder function
typedef struct {
    int order_id;
    char* customer_name;
    OrderItem* items;  // This pointer is NULL!
    double total;
} Order;

int ProcessOrder(Order* order) {
    if (order == NULL) return -1;
    
    // BUG: Not checking if order->items is NULL
    int item_count = order->items->count;  // Crash here when items is NULL
    
    for (int i = 0; i < item_count; i++) {
        // Process each item...
    }
    
    return 0;
}
```

**Step 3: Root Cause Analysis**
```windbg
# Examine the order structure
0:000> dt OnlineStore!Order @rax
   +0x000 order_id         : 0n12345
   +0x004 customer_name    : 0x00234567  "John Smith"
   +0x008 items            : (null)      <- NULL pointer!
   +0x00c total            : 299.99

# Check call stack for context
0:000> kv
00 00a2f8c4 00401234 OnlineStore!ProcessOrder+0x45
01 00a2f8d8 00401456 OnlineStore!HandleCheckout+0x34
02 00a2f8ec 00401567 OnlineStore!ProcessRequest+0x67
```

**Solution:**
```c
// Fixed version with proper NULL checking
int ProcessOrder(Order* order) {
    if (order == NULL) {
        LogError("ProcessOrder: order is NULL");
        return -1;
    }
    
    if (order->items == NULL) {
        LogError("ProcessOrder: order->items is NULL for order %d", order->order_id);
        return -2;  // Different error code for items being NULL
    }
    
    int item_count = order->items->count;
    
    for (int i = 0; i < item_count; i++) {
        // Process each item...
    }
    
    return 0;
}
```

### Case Study 2: Memory Leak in Document Processor

**Background:**
A document processing service gradually consumes more memory over time, eventually causing system instability.

**Symptoms:**
- Memory usage increases by ~50MB per hour
- Performance degrades after 24 hours of operation
- Eventually causes out-of-memory conditions

**Investigation Process:**

**Step 1: Enable Heap Debugging**
```cmd
# Enable Application Verifier
appverif -enable Heaps -for DocProcessor.exe

# Enable page heap for detailed tracking
gflags -p /enable DocProcessor.exe /full
```

**Step 2: Capture Heap State**
```windbg
# After running for several hours
.attach DocProcessor.exe

# Analyze heap usage
!heap -s

Output:
Heap     Flags   Reserv  Commit  Virt   Free  List   UCR  Virt  Lock  Fast 
                  (k)     (k)    (k)     (k) length      blocks cont. heap
00340000 00000002   64512  64124  64512   1234    58     4    0      0   LFH
00450000 00001002    1024   1024   1024    45     12     1    0      0   LFH
00560000 00041002   32768  32654  32768    123    89     2    0      0   

# Look for leaks
!heap -l

Output:
Searching the heap for memory leaks...
Heap 00340000
    Address: 00567890 Size: 1024 bytes - LEAKED
    Stack trace:
        DocProcessor!LoadDocument+0x45
        DocProcessor!ProcessFile+0x123
        DocProcessor!WorkerThread+0x67
        
    Address: 00567a90 Size: 2048 bytes - LEAKED
    Stack trace:
        DocProcessor!AllocateBuffer+0x23
        DocProcessor!ParseXML+0x89
        DocProcessor!LoadDocument+0x56
```

**Step 3: Analyze Leak Pattern**
```c
// Problematic code found through stack traces
Document* LoadDocument(const char* filename) {
    Document* doc = (Document*)malloc(sizeof(Document));
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        // BUG: Memory leak - doc is not freed on error
        return NULL;
    }
    
    char* buffer = (char*)malloc(MAX_BUFFER_SIZE);
    if (!buffer) {
        fclose(file);
        // BUG: Another memory leak - doc is not freed
        return NULL;
    }
    
    // Read and process file...
    fread(buffer, 1, MAX_BUFFER_SIZE, file);
    
    // BUG: buffer is never freed!
    
    fclose(file);
    return doc;
}
```

**Solution:**
```c
// Fixed version with proper cleanup
Document* LoadDocument(const char* filename) {
    Document* doc = (Document*)malloc(sizeof(Document));
    if (!doc) {
        return NULL;
    }
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        free(doc);  // Clean up doc on error
        return NULL;
    }
    
    char* buffer = (char*)malloc(MAX_BUFFER_SIZE);
    if (!buffer) {
        fclose(file);
        free(doc);  // Clean up doc on error
        return NULL;
    }
    
    // Read and process file...
    size_t bytes_read = fread(buffer, 1, MAX_BUFFER_SIZE, file);
    
    // Process the buffer content...
    ProcessDocumentData(doc, buffer, bytes_read);
    
    // Clean up temporary buffer
    free(buffer);
    fclose(file);
    
    return doc;
}

// Also need proper cleanup when document is no longer needed
void FreeDocument(Document* doc) {
    if (doc) {
        // Free any internal allocations
        if (doc->content) {
            free(doc->content);
        }
        free(doc);
    }
}
```

### Case Study 3: Deadlock in Multi-threaded Service

**Background:**
A multi-threaded Windows service occasionally hangs, becoming completely unresponsive.

**Investigation Process:**

**Step 1: Attach to Hung Process**
```windbg
.attach ServiceApp.exe

# Check all thread states
~*k

Output:
   0  Id: 1234.5678 Suspend: 0 Teb: 7efdd000 Unfrozen
# Child-SP          RetAddr           Call Site
00 `0019ff88 `77bb1234 ntdll!NtWaitForSingleObject+0x15
01 `0019ff98 `00401234 kernel32!WaitForSingleObjectEx+0x98
02 `0019ffa8 `00401345 ServiceApp!WorkerThread1+0x123

   1  Id: 1234.5679 Suspend: 0 Teb: 7efdc000 Unfrozen
# Child-SP          RetAddr           Call Site
00 `002aff88 `77bb5678 ntdll!NtWaitForCriticalSection+0x15
01 `002aff98 `00401456 ntdll!RtlEnterCriticalSection+0x45
02 `002affa8 `00401567 ServiceApp!WorkerThread2+0x234
```

**Step 2: Analyze Critical Sections**
```windbg
# Display critical section information
!cs -l

Output:
-----------------------------------------
Critical section   = 0x00403000 (ServiceApp!g_cs1)
DebugInfo          = 0x77bb0520
LOCKED
LockCount          = 0x1
WaiterWoken        = No
OwningThread       = 0x5678 (Thread 0)
RecursionCount     = 0x1
LockSemaphore      = 0x0
SpinCount          = 0x00000000

-----------------------------------------
Critical section   = 0x00403100 (ServiceApp!g_cs2)
DebugInfo          = 0x77bb0620
LOCKED
LockCount          = 0x1
WaiterWoken        = No
OwningThread       = 0x5679 (Thread 1)
RecursionCount     = 0x1
LockSemaphore      = 0x0
SpinCount          = 0x00000000

# Analyze the hang
!analyze -hang

Output:
BLOCKING_THREAD: 00005678
    Owns critical section ServiceApp!g_cs1
    Waiting for critical section ServiceApp!g_cs2

BLOCKED_THREAD: 00005679
    Owns critical section ServiceApp!g_cs2
    Waiting for critical section ServiceApp!g_cs1

DEADLOCK DETECTED:
Thread 0 -> CS1 -> waits for CS2 -> Thread 1
Thread 1 -> CS2 -> waits for CS1 -> Thread 0
```

**Step 3: Examine Source Code**
```c
// Problematic code causing deadlock
CRITICAL_SECTION g_cs1, g_cs2;

DWORD WINAPI WorkerThread1(LPVOID param) {
    while (running) {
        EnterCriticalSection(&g_cs1);
        // Do some work...
        
        // Need data protected by cs2
        EnterCriticalSection(&g_cs2);  // Potential deadlock here
        // Access shared data...
        LeaveCriticalSection(&g_cs2);
        
        LeaveCriticalSection(&g_cs1);
        Sleep(100);
    }
    return 0;
}

DWORD WINAPI WorkerThread2(LPVOID param) {
    while (running) {
        EnterCriticalSection(&g_cs2);
        // Do some work...
        
        // Need data protected by cs1
        EnterCriticalSection(&g_cs1);  // Potential deadlock here
        // Access shared data...
        LeaveCriticalSection(&g_cs1);
        
        LeaveCriticalSection(&g_cs2);
        Sleep(100);
    }
    return 0;
}
```

**Solution:**
```c
// Fixed version - always acquire locks in same order
DWORD WINAPI WorkerThread1(LPVOID param) {
    while (running) {
        // Always acquire cs1 first, then cs2
        EnterCriticalSection(&g_cs1);
        EnterCriticalSection(&g_cs2);
        
        // Do work with both locks held
        // Access shared data...
        
        // Release in reverse order
        LeaveCriticalSection(&g_cs2);
        LeaveCriticalSection(&g_cs1);
        
        Sleep(100);
    }
    return 0;
}

DWORD WINAPI WorkerThread2(LPVOID param) {
    while (running) {
        // Same lock ordering as Thread1
        EnterCriticalSection(&g_cs1);
        EnterCriticalSection(&g_cs2);
        
        // Do work with both locks held
        // Access shared data...
        
        // Release in reverse order
        LeaveCriticalSection(&g_cs2);
        LeaveCriticalSection(&g_cs1);
        
        Sleep(100);
    }
    return 0;
}
```

### Performance Debugging Best Practices

**1. Systematic Approach:**
- Always start with `!analyze -v` for automatic analysis
- Examine call stacks of all threads
- Check for obvious patterns (NULL pointers, heap corruption)
- Use symbols and source code when available

**2. Memory Debugging:**
- Enable Application Verifier early in development
- Use Page Heap for detailed allocation tracking
- Monitor heap growth over time
- Look for common leak patterns (missing cleanup in error paths)

**3. Concurrency Debugging:**
- Map out lock ordering in your application
- Use consistent locking patterns across all threads
- Consider timeout-based locks to detect potential deadlocks
- Document locking dependencies

**4. Production Debugging:**
- Set up automatic crash dump collection
- Use debug symbols even in release builds
- Implement comprehensive logging
- Consider remote debugging for difficult issues

**5. Prevention:**
- Use static analysis tools (PREfast, PC-Lint)
- Implement unit tests for error conditions
- Regular stress testing with debugging tools enabled
- Code reviews focusing on resource management and threading

## Next Steps

After mastering these advanced WinDbg techniques, consider exploring:

- **Kernel Debugging:** Debug device drivers and system-level issues
- **Performance Analysis:** Integrate with Perfetto and other profiling tools  
- **Automated Analysis:** Create custom debugging scripts and extensions
- **Cloud Debugging:** Debug applications running in cloud environments
- **Advanced Tools:** ETW (Event Tracing for Windows) and custom instrumentation

