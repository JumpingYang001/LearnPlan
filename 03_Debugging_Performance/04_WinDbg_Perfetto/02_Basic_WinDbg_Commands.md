# Basic WinDbg Commands

*Duration: 1 week*

## Overview

WinDbg is Microsoft's flagship debugger for Windows applications, drivers, and system debugging. This comprehensive guide covers essential WinDbg commands for navigation, breakpoints, memory examination, and advanced debugging scenarios. Whether you're debugging user-mode applications or kernel-mode drivers, mastering these commands is crucial for effective Windows debugging.

### What You'll Learn
- Essential navigation and control commands
- Breakpoint management and types
- Memory examination and manipulation
- Register and stack inspection
- Symbol handling and module analysis
- Advanced debugging techniques

## Getting Started with WinDbg

### Sample Program for Practice

Let's use a more comprehensive example program that demonstrates various debugging scenarios:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// Structure for demonstration
typedef struct {
    int id;
    char name[50];
    double value;
} DataRecord;

// Global variables for debugging practice
int g_counter = 0;
DataRecord g_records[3];

// Function prototypes
void initialize_data(void);
void process_record(DataRecord* record);
void demonstrate_memory_operations(void);
void create_thread_example(void);

int main() {
    printf("WinDbg Debugging Example\n");
    printf("========================\n");
    
    // Initialize some data
    initialize_data();
    
    // Process records
    for (int i = 0; i < 3; i++) {
        printf("Processing record %d\n", i);
        process_record(&g_records[i]);      // Breakpoint location 1
        g_counter++;
    }
    
    // Demonstrate memory operations
    demonstrate_memory_operations();        // Breakpoint location 2
    
    // Create a thread for advanced debugging
    create_thread_example();                // Breakpoint location 3
    
    printf("Program completed. Counter: %d\n", g_counter);
    return 0;
}

void initialize_data(void) {
    strcpy(g_records[0].name, "Record_One");
    g_records[0].id = 1001;
    g_records[0].value = 123.45;
    
    strcpy(g_records[1].name, "Record_Two");
    g_records[1].id = 1002;
    g_records[1].value = 678.90;
    
    strcpy(g_records[2].name, "Record_Three");
    g_records[2].id = 1003;
    g_records[2].value = 999.99;
}

void process_record(DataRecord* record) {
    if (record == NULL) return;
    
    printf("ID: %d, Name: %s, Value: %.2f\n", 
           record->id, record->name, record->value);
    
    // Simulate some processing
    record->value *= 1.1;  // Increase by 10%
}

void demonstrate_memory_operations(void) {
    // Allocate dynamic memory
    char* buffer = (char*)malloc(256);
    if (buffer != NULL) {
        strcpy(buffer, "Dynamic memory content for debugging");
        printf("Buffer content: %s\n", buffer);
        free(buffer);
    }
    
    // Stack variable for memory examination
    int local_array[5] = {10, 20, 30, 40, 50};
    printf("Local array values: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", local_array[i]);
    }
    printf("\n");
}

DWORD WINAPI thread_function(LPVOID param) {
    int thread_id = *(int*)param;
    printf("Thread %d is running\n", thread_id);
    Sleep(1000);
    printf("Thread %d completed\n", thread_id);
    return 0;
}

void create_thread_example(void) {
    int thread_param = 42;
    HANDLE thread_handle = CreateThread(
        NULL,                    // Security attributes
        0,                       // Stack size
        thread_function,         // Thread function
        &thread_param,          // Parameter
        0,                       // Creation flags
        NULL                     // Thread ID
    );
    
    if (thread_handle != NULL) {
        WaitForSingleObject(thread_handle, INFINITE);
        CloseHandle(thread_handle);
    }
}
```

### Compilation Instructions

```batch
:: Compile with debug information
cl /Zi /Od /MDd windbg_example.c /Fe:windbg_example.exe

:: Alternative with GCC (MinGW)
gcc -g -O0 -o windbg_example.exe windbg_example.c

:: Create PDB file for better debugging
cl /Zi /Od /MDd windbg_example.c /Fe:windbg_example.exe /Fd:windbg_example.pdb
```

## Essential WinDbg Command Categories

### 1. Program Control Commands

#### Starting and Attaching to Processes

**Launch a new process:**
```
windbg -o windbg_example.exe          # Launch with debugger attached
windbg -g windbg_example.exe          # Launch and run (go)
windbg -c "g; q" windbg_example.exe   # Run command and quit
```

**Attach to running process:**
```
.attach <PID>                          # Attach to process by ID
.attach -p <process_name>              # Attach by process name
.detach                                # Detach from process
```

**Process control:**
```
g                                      # Go/Continue execution
gc                                     # Go from conditional breakpoint
gh                                     # Go with exception handled
gn                                     # Go with exception not handled
q                                      # Quit debugger
qd                                     # Quit and detach
```

#### Execution Control

**Step commands:**
```
t                                      # Trace (step into)
p                                      # Step over
tc                                     # Trace to next call
pc                                     # Step over to next call
tt                                     # Trace to next return
pt                                     # Step over to next return
gu                                     # Go up (step out of current function)
```

**Practical Example:**
```
0:000> bp main                         # Set breakpoint at main
0:000> g                               # Run to main
Breakpoint 0 hit
windbg_example!main:
00007ff6`9e5e1180 48 89 5c 24 08     mov qword ptr [rsp+8],rbx

0:000> t                               # Step into next instruction
windbg_example!main+0x5:
00007ff6`9e5e1185 48 89 74 24 10     mov qword ptr [rsp+10h],rsi

0:000> p                               # Step over next instruction
0:000> gu                              # Step out of current function
```

### 2. Breakpoint Commands

#### Setting Breakpoints

**Function breakpoints:**
```
bp main                                # Break at function entry
bp windbg_example!main                 # Module-qualified breakpoint
bp windbg_example!process_record       # Break at specific function
bp `windbg_example.c:25`               # Source line breakpoint
```

**Address breakpoints:**
```
bp 00401000                            # Break at specific address
bp main+0x10                           # Break at offset from function
bp @eip+5                              # Break 5 bytes from current instruction
```

**Conditional breakpoints:**
```
bp main ".if (poi(@esp+4) == 1) {} .else {gc}"  # Break if argument equals 1
bp process_record "j (@$arg1 != 0) ''; 'gc'"    # Break if pointer not null
```

**Data breakpoints (watchpoints):**
```
ba r4 g_counter                        # Break on read of 4-byte variable
ba w4 g_counter                        # Break on write of 4-byte variable
ba e1 00401000                         # Break on execute at address
ba rw8 @rsp                            # Break on read/write of stack pointer
```

#### Managing Breakpoints

**List and control:**
```
bl                                     # List all breakpoints
bd 0                                   # Disable breakpoint 0
be 0                                   # Enable breakpoint 0
bc 0                                   # Clear breakpoint 0
bc *                                   # Clear all breakpoints
```

**Breakpoint information:**
```
bp /1 main                             # One-shot breakpoint
bp /p <PID> main                       # Process-specific breakpoint
bp /t <TID> main                       # Thread-specific breakpoint
```

**Example Breakpoint Session:**
```
0:000> bp main
0:000> bp process_record
0:000> bp `windbg_example.c:45`
0:000> bl
     0 e Disable Clear  00007ff6`9e5e1180     0001 (0001)  0:**** windbg_example!main
     1 e Disable Clear  00007ff6`9e5e11f0     0001 (0001)  0:**** windbg_example!process_record
     2 e Disable Clear  00007ff6`9e5e1220     0001 (0001)  0:**** windbg_example!main+0x40

0:000> g
Breakpoint 0 hit
windbg_example!main:
00007ff6`9e5e1180 48 89 5c 24 08     mov qword ptr [rsp+8],rbx
```

### 3. Memory Examination Commands

#### Display Memory

**Basic memory display:**
```
d <address>                            # Display memory (default format)
da <address>                           # Display as ASCII string
du <address>                           # Display as Unicode string
dw <address>                           # Display as words (2 bytes)
dd <address>                           # Display as dwords (4 bytes)
dq <address>                           # Display as qwords (8 bytes)
dp <address>                           # Display as pointers
```

**Advanced memory display:**
```
db <address> L<length>                 # Display bytes with length
dds <address>                          # Display dwords with symbols
dqs <address>                          # Display qwords with symbols
dt <type> <address>                    # Display typed data
```

**Practical Examples:**
```
0:000> dd g_counter L1                 # Display counter variable
00007ff6`9e607000  00000003

0:000> da g_records                    # Display records as ASCII
00007ff6`9e607014  "Record_One"

0:000> dt DataRecord g_records         # Display structured data
windbg_example!DataRecord
   +0x000 id               : 0n1001
   +0x004 name             : [50] "Record_One"
   +0x036 value            : 123.45

0:000> dq @rsp L4                      # Display stack
000000d7`9f8ff568  00007ff6`9e5e1220
000000d7`9f8ff570  00000000`00000000
```

#### Memory Search

**Search patterns:**
```
s -a <range> "string"                  # Search for ASCII string
s -u <range> "string"                  # Search for Unicode string
s -d <range> <dword>                   # Search for DWORD value
s -q <range> <qword>                   # Search for QWORD value
```

**Examples:**
```
0:000> s -a 0 L?80000000 "Record"      # Search for "Record" in memory
00007ff6`9e607014  52 65 63 6f 72 64 5f 4f-6e 65 00 00 00 00 00 00  Record_One......

0:000> s -d 0 L?80000000 1001          # Search for ID value 1001
00007ff6`9e607010  000003e9
```

### 4. Register and Stack Commands

#### Register Examination

**Display registers:**
```
r                                      # Display all registers
r eax                                  # Display specific register
r eax=123                              # Set register value
r @eax                                 # Register in expression
```

**x64 Register Display:**
```
0:000> r
rax=0000000000000000 rbx=0000000000000000 rcx=0000000000000000
rdx=0000000000000000 rsi=0000000000000000 rdi=0000000000000000
rip=00007ff69e5e1180 rsp=000000d79f8ff568 rbp=0000000000000000
 r8=0000000000000000  r9=0000000000000000 r10=0000000000000000
r11=0000000000000000 r12=0000000000000000 r13=0000000000000000
r14=0000000000000000 r15=0000000000000000
iopl=0         nv up ei pl zr na po nc
cs=0033  ss=002b  ds=002b  es=002b  fs=0053  gs=002b             efl=00000246
```

**Flag registers:**
```
r $flags                               # Display flags register
r $flag                                # Display individual flags
.formats 123                           # Display number in different formats
```

#### Stack Examination

**Stack display:**
```
k                                      # Display stack trace
kn                                     # Stack trace with frame numbers
kb                                     # Stack trace with first 3 parameters
kp                                     # Stack trace with full parameters
kv                                     # Stack trace with FPO information
```

**Stack trace example:**
```
0:000> kn
 # Child-SP          RetAddr           Call Site
00 000000d7`9f8ff568 00007ff6`9e5e1220 windbg_example!main
01 000000d7`9f8ff570 00007ff8`a5dd257d windbg_example!invoke_main+0x22
02 000000d7`9f8ff5c0 00007ff8`a7e6aa78 KERNEL32!BaseThreadInitThunk+0x1d
03 000000d7`9f8ff5f0 00000000`00000000 ntdll!RtlUserThreadStart+0x28
```

**Stack manipulation:**
```
dq @rsp                                # Display stack memory
.frame 1                               # Switch to stack frame 1
.frame /r 1                            # Switch to frame 1 and show registers
```

### 5. Symbol and Module Commands

#### Symbol Management

**Load symbols:**
```
.sympath srv*C:\Symbols*https://msdl.microsoft.com/download/symbols
.reload                                # Reload symbols
.reload /f                             # Force reload
ld *                                   # Load all module symbols
ld windbg_example                      # Load specific module symbols
```

**Symbol examination:**
```
x *!                                   # List all symbols
x windbg_example!*                     # List module symbols
x windbg_example!main                  # Find specific symbol
x windbg_example!g_*                   # Find global variables
```

**Symbol information:**
```
ln <address>                           # Find nearest symbol to address
u <address>                            # Unassemble at address
uf <function>                          # Unassemble entire function
```

#### Module Information

**List modules:**
```
lm                                     # List all modules
lm v                                   # Verbose module list
lm vm windbg_example                   # Verbose info for specific module
```

**Module details:**
```
!lmi windbg_example                    # Module information
.imgscan                               # Scan for images
.reload /f windbg_example.exe          # Force reload module
```

### 6. Advanced Examination Commands

#### Type and Structure Display

**Display types:**
```
dt                                     # List all types
dt DataRecord                          # Display type definition
dt DataRecord 00401000                 # Display instance at address
dt -r DataRecord 00401000              # Recursive display
```

**Pointer following:**
```
poi(address)                           # Dereference pointer
wo(address)                            # Dereference as word
by(address)                            # Dereference as byte
```

#### Expression Evaluation

**Evaluate expressions:**
```
? <expression>                         # Evaluate C++ expression
?? <expression>                        # C++ expression evaluator
.printf "%d %s", poi(g_counter), g_records
```

**Examples:**
```
0:000> ? g_counter
Evaluate expression: 2147319808 = 00007ff6`9e607000

0:000> ?? sizeof(DataRecord)
unsigned int64 0x3a

0:000> ? poi(g_counter)
Evaluate expression: 3 = 00000000`00000003
```

### 7. Thread and Process Commands

#### Thread Management

**Thread information:**
```
~                                      # List threads
~* k                                   # Stack trace for all threads
~<thread_id> s                         # Switch to thread
~<thread_id> f                         # Freeze thread
~<thread_id> u                         # Unfreeze thread
```

**Thread context:**
```
.thread <address>                      # Set thread context
!thread <address>                      # Display thread information
!runaway                               # Show thread times
```

#### Process Information

**Process details:**
```
|                                      # List processes
!peb                                   # Process Environment Block
!teb                                   # Thread Environment Block
!handle                                # List handles
```

### 8. Practical Debugging Scenarios

#### Scenario 1: Debugging a Crash

```
0:000> .ecxr                           # Display exception context
0:000> k                               # Get stack trace
0:000> !analyze -v                     # Analyze crash
0:000> r                               # Check registers
0:000> u @rip                          # Disassemble at crash location
```

#### Scenario 2: Finding Memory Corruption

```
0:000> !heap -a                        # Analyze heap
0:000> !address                        # Memory usage summary
0:000> s -d 0 L?80000000 deadbeef      # Search for corruption pattern
0:000> ba w4 <suspected_address>       # Set write breakpoint
```

#### Scenario 3: Performance Analysis

```
0:000> !runaway 7                      # Thread CPU usage
0:000> !locks                          # Display locks
0:000> !cs -l                          # Critical sections
0:000> wt                              # Trace execution time
```

## Command Quick Reference

### Essential Commands Cheat Sheet

| Category | Command | Description |
|----------|---------|-------------|
| **Control** | `g` | Go/Continue |
| | `t` | Trace into |
| | `p` | Step over |
| | `gu` | Go up (step out) |
| **Breakpoints** | `bp <func>` | Set breakpoint |
| | `bl` | List breakpoints |
| | `bc *` | Clear all breakpoints |
| **Memory** | `dd <addr>` | Display dwords |
| | `da <addr>` | Display ASCII |
| | `dt <type> <addr>` | Display typed data |
| **Registers** | `r` | Display registers |
| | `k` | Stack trace |
| **Symbols** | `x *!<pattern>` | Find symbols |
| | `ln <addr>` | Nearest symbol |
| **Threads** | `~` | List threads |
| | `~n s` | Switch thread |

### Common Debugging Workflows

**1. Initial Setup:**
```
.sympath srv*C:\Symbols*https://msdl.microsoft.com/download/symbols
.reload
bp main
g
```

**2. Examine Variables:**
```
dt DataRecord g_records
dd g_counter L1
da g_records[0].name
```

**3. Set Conditional Breakpoint:**
```
bp process_record ".if (poi(@rcx) == 1002) {} .else {gc}"
g
```

**4. Memory Analysis:**
```
s -a 0 L?80000000 "Record"
!heap -a
!address
```

## Hands-on Practice Exercises

### Exercise 1: Basic Navigation and Breakpoints

**Objective:** Master basic debugging navigation

**Steps:**
1. Compile the sample program with debug information
2. Start WinDbg and load the program
3. Set breakpoints and examine execution

```
windbg windbg_example.exe

0:000> bp main
0:000> bp process_record
0:000> g
Breakpoint 0 hit

0:000> t        # Step through initialization
0:000> p        # Step over function calls
0:000> g        # Continue to next breakpoint
```

**Practice Tasks:**
- [ ] Set a breakpoint at each function
- [ ] Step through the main function line by line
- [ ] Examine local variables at each step
- [ ] Continue execution and hit the next breakpoint

### Exercise 2: Memory Examination

**Objective:** Learn to examine different types of memory

**Practice Commands:**
```
0:000> dt DataRecord g_records[0]      # Examine structure
0:000> da g_records[0].name            # View string content
0:000> dd &g_records[0].id L1          # View integer value
0:000> dq &g_records[0].value L1       # View double value
```

**Advanced Memory Tasks:**
- [ ] Find all instances of "Record" in memory
- [ ] Examine the stack during function calls
- [ ] Watch memory changes during array processing
- [ ] Analyze heap allocations in `demonstrate_memory_operations`

### Exercise 3: Conditional Debugging

**Objective:** Use conditional breakpoints for targeted debugging

**Setup conditional breakpoints:**
```
0:000> bp process_record ".if (poi(@rcx) == 0) {.echo Null pointer detected} .else {gc}"
0:000> bp main+offset ".if (g_counter >= 2) {} .else {gc}"
```

**Tasks:**
- [ ] Set a breakpoint that only triggers on the second iteration
- [ ] Create a breakpoint that prints variable values without stopping
- [ ] Use watchpoints to detect when `g_counter` changes
- [ ] Set up logging breakpoints for function entry/exit

### Exercise 4: Advanced Analysis

**Objective:** Perform comprehensive program analysis

**Analysis Commands:**
```
0:000> x windbg_example!g_*           # Find all global variables
0:000> uf main                        # Disassemble main function
0:000> !heap -a                       # Analyze heap usage
0:000> wt                             # Trace execution with timing
```

**Analysis Tasks:**
- [ ] Map out all global variables and their addresses
- [ ] Analyze function call overhead using `wt`
- [ ] Examine thread creation and management
- [ ] Profile memory allocation patterns

## Troubleshooting Common Issues

### Symbol Loading Problems

**Problem:** Symbols not loading properly
```
0:000> .symfix
0:000> .sympath+ C:\MyProject\Debug
0:000> .reload /f
0:000> ld windbg_example
```

**Verification:**
```
0:000> lm v m windbg_example
0:000> x windbg_example!main          # Should show symbol
```

### Breakpoint Issues

**Problem:** Breakpoints not hitting
```
# Check if module is loaded
0:000> lm

# Verify breakpoint location
0:000> bl
0:000> u main L5                       # Disassemble to verify

# Use module qualification
0:000> bp windbg_example!main
```

### Memory Access Violations

**Problem:** Cannot access memory
```
# Check memory protection
0:000> !address <address>

# Use safe memory access
0:000> .if (poi(<address>) != 0) {dd <address>}
```

### Performance Tips

**Optimization strategies:**
1. **Use specific breakpoints** instead of stepping extensively
2. **Load symbols on demand** with `ld` instead of loading all
3. **Use conditional breakpoints** to avoid unnecessary breaks
4. **Employ logging breakpoints** for automated data collection

## Learning Objectives Assessment

### Self-Check Questions

**Basic Commands:**
1. What's the difference between `t` and `p` commands?
2. How do you set a breakpoint at a specific source line?
3. What command shows the current stack trace?
4. How do you examine a structure at a specific memory address?

**Intermediate Skills:**
5. How do you set a conditional breakpoint that only triggers when a variable equals a specific value?
6. What's the difference between `dd` and `dds` commands?
7. How do you search for a specific string in memory?
8. How do you switch between different threads in a multi-threaded program?

**Advanced Techniques:**
9. How do you set up symbol paths for Microsoft symbols?
10. What commands help analyze heap corruption?
11. How do you profile function execution time?
12. How do you debug a program that's already running?

### Practical Skills Checklist

Before proceeding to advanced topics, ensure you can:

□ **Launch WinDbg** and attach to a process  
□ **Set and manage breakpoints** of different types  
□ **Navigate through code** using step commands  
□ **Examine memory** in various formats  
□ **Display and modify registers**  
□ **Analyze stack traces** and function calls  
□ **Load and manage symbols** properly  
□ **Use conditional breakpoints** effectively  
□ **Search memory** for patterns and values  
□ **Debug multi-threaded applications**  
□ **Analyze crash dumps** and exceptions  
□ **Profile performance** characteristics  

## Next Steps and Advanced Topics

### Recommended Learning Path

1. **Master these basic commands** through hands-on practice
2. **Practice with real applications** beyond the sample program
3. **Learn kernel debugging** with WinDbg
4. **Explore crash dump analysis** techniques
5. **Study advanced extensions** and plugins

### Additional Resources

**Official Documentation:**
- [Microsoft WinDbg Documentation](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/)
- [Debugging Tools for Windows](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/debugger-download-tools)

**Community Resources:**
- WinDbg user forums and communities
- Advanced debugging blogs and tutorials
- Video tutorials and walkthroughs

**Recommended Books:**
- "Advanced Windows Debugging" by Mario Hewardt and Daniel Pravat
- "Inside Windows Debugging" by Tarik Soulami
- "Windows Internals" series by Mark Russinovich

### Practice Projects

**Project 1: Debug a Memory Leak**
Create a program with intentional memory leaks and use WinDbg to identify and fix them.

**Project 2: Analyze a Crash Dump**
Generate crash dumps and practice analyzing them with WinDbg commands.

**Project 3: Performance Profiling**
Use WinDbg to profile a CPU-intensive application and identify bottlenecks.

**Project 4: Kernel Debugging Setup**
Set up a virtual machine environment for kernel-mode debugging practice.

## Command Reference Summary

### Quick Command Reference Card

Print this section for easy reference during debugging sessions:

```
CONTROL:     g (go), t (trace), p (step), gu (go up), q (quit)
BREAKPOINTS: bp (set), bl (list), bc (clear), bd (disable), be (enable)
MEMORY:      dd/dw/db (display), da/du (strings), dt (types), s (search)
REGISTERS:   r (display), r reg=value (set), k (stack trace)
SYMBOLS:     x (examine), ln (nearest), uf (unassemble function)
THREADS:     ~ (list), ~n s (switch), ~* k (all stacks)
MODULES:     lm (list), ld (load symbols), .reload (reload symbols)
HELP:        .help, .hh (context help), ? (evaluate expression)
```

Remember: WinDbg is a powerful tool that requires practice to master. Start with simple programs and gradually work your way up to more complex debugging scenarios!
