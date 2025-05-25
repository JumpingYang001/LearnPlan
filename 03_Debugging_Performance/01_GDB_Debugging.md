# Debugging Tools: GDB

*Last Updated: May 25, 2025*

## Overview

The GNU Debugger (GDB) is a powerful debugging tool that allows developers to see what's happening inside a program while it executes or what it was doing at the moment it crashed. This learning track covers GDB from basic usage to advanced debugging techniques for C/C++ programs.

## Learning Path

### 1. GDB Fundamentals (1 week)
- Installing and configuring GDB
- Compiling programs with debugging information (-g flag)
- Starting GDB with a program
- Basic GDB commands
- GDB command syntax and help system
- Command completion and history

### 2. Basic Debugging Techniques (1 week)
- **Running Programs in GDB**
  - Starting, stopping, continuing execution
  - Running with arguments
  - Setting environment variables
  - Controlling terminal I/O
- **Breakpoints and Watchpoints**
  - Setting breakpoints at functions and lines
  - Conditional breakpoints
  - Temporary breakpoints
  - Watchpoints for data changes
  - Catch points for events
- **Examining Program State**
  - Printing variables and expressions
  - Examining memory
  - Displaying registers
  - Displaying types and structures
  - Pretty printing complex data structures

### 3. Controlling Program Execution (1 week)
- **Stepping Through Code**
  - Single stepping (step, next)
  - Stepping into functions
  - Stepping out of functions
  - Continuing until a specified location
  - Jumping to different code locations
- **Managing Execution Context**
  - Call stack navigation
  - Frame selection
  - Thread selection
  - Inferior (process) selection
- **Signal Handling**
  - Catching signals
  - Sending signals
  - Configuring signal actions

### 4. Advanced GDB Features (2 weeks)
- **Scripting and Automation**
  - GDB command files
  - User-defined commands
  - Python scripting in GDB
  - Pretty printers
  - Custom GDB commands
- **Reverse Debugging**
  - Record and replay execution
  - Reverse stepping
  - Reverse continue
  - Limitations and overhead
- **Remote Debugging**
  - Setting up a debug server (gdbserver)
  - Connecting to remote targets
  - Cross-platform debugging
  - Embedded system debugging
- **Core Dump Analysis**
  - Generating core dumps
  - Loading core dumps in GDB
  - Extracting information from core dumps
  - Post-mortem debugging techniques

### 5. Debugging Multi-threaded Programs (2 weeks)
- Thread creation and termination detection
- Listing and selecting threads
- Thread-specific breakpoints
- Examining thread-local storage
- Deadlock detection
- Race condition analysis
- Non-stop mode debugging
- All-stop mode debugging
- Scheduler locking

### 6. Memory Debugging with GDB (1 week)
- Finding memory leaks
- Detecting buffer overflows
- Tracking heap allocations
- Inspecting memory regions
- Integration with Valgrind
- Integration with AddressSanitizer

### 7. C++ Specific Debugging (1 week)
- Dealing with templates
- Setting breakpoints in overloaded functions
- Inspecting STL containers
- Handling exceptions
- Pretty printing STL types
- Debugging virtual function calls
- Name demangling

### 8. GDB TUI and GUIs (1 week)
- **Text User Interface (TUI)**
  - Source window
  - Assembly window
  - Register window
  - Navigation in TUI mode
- **GDB Frontends**
  - DDD (Data Display Debugger)
  - Nemiver
  - Eclipse CDT debugging
  - Visual Studio Code with GDB
  - CLion debugging interface

## Projects

1. **Custom GDB Script Library**
   - Create a collection of useful GDB scripts
   - Implement pretty printers for common data structures

2. **Automated Debugging Tool**
   - Build a tool that automates common debugging tasks with GDB
   - Support batch analysis of programs

3. **Memory Leak Detector**
   - Create a GDB-based tool to identify memory leaks
   - Generate reports of allocation sites

4. **Thread Analysis Tool**
   - Implement a GDB script for analyzing thread interactions
   - Detect potential deadlocks or race conditions

5. **GDB Frontend Extension**
   - Extend an existing GDB frontend with new features
   - Improve visualization of complex data structures

## Resources

### Books
- "Debugging with GDB: The GNU Source-Level Debugger" (GDB Documentation)
- "The Art of Debugging with GDB, DDD, and Eclipse" by Norman Matloff and Peter Jay Salzman
- "Advanced C/C++ Debugging" (various authors)

### Online Resources
- [Official GDB Documentation](https://sourceware.org/gdb/current/onlinedocs/gdb/)
- [GDB Cheat Sheet](https://darkdust.net/files/GDB%20Cheat%20Sheet.pdf)
- [GDB Tutorial by Stanford CS](https://web.stanford.edu/class/cs107/guide/gdb.html)
- [GDB Wiki](https://sourceware.org/gdb/wiki/)

### Video Courses
- "GDB Debugging Fundamentals" on Pluralsight
- "Advanced Debugging with GDB" on Linux Foundation Training

## Assessment Criteria

You should be able to:
- Debug complex C/C++ programs effectively with GDB
- Analyze program crashes and generate meaningful bug reports
- Create custom GDB scripts for specialized debugging tasks
- Debug multi-threaded applications and identify concurrency issues
- Use GDB for remote debugging and core dump analysis
- Integrate GDB with IDEs and build systems

## Next Steps

After mastering GDB, consider exploring:
- LLDB (LLVM Debugger)
- WinDbg for Windows debugging
- Dynamic analysis tools (Valgrind, AddressSanitizer)
- Static analysis tools
- Automated test generation and fuzzing techniques
