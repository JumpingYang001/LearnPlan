# GDB TUI and GUIs: Enhanced Debugging Experience

*Duration: 2-3 hours*

## Overview

The traditional command-line GDB interface can be challenging for complex debugging sessions. GDB's Text User Interface (TUI) and various graphical frontends provide visual debugging capabilities that make the debugging process more intuitive and efficient. This section covers how to leverage these tools for enhanced debugging productivity.

### Why Use Visual Debugging Interfaces?

**Benefits of Visual Debugging:**
- **Source code visibility** while debugging
- **Multiple information panes** (source, assembly, registers, stack)
- **Visual breakpoint management**
- **Real-time variable inspection**
- **Easier navigation** through code and call stack
- **Reduced context switching** between terminal and editor

## GDB Text User Interface (TUI)

### What is GDB TUI?

GDB TUI is a built-in text-based interface that divides the terminal into multiple windows, allowing you to see source code, assembly, and command output simultaneously.

### TUI Layout Components

```
┌─────────────────────────────────────────────────────────┐
│                    Source Window                        │
│  1  #include <stdio.h>                                  │
│  2  #include <stdlib.h>                                 │
│  3                                                      │
│  4  int main() {                                        │
│  5►     int x = 10;        ← Current execution line     │
│  6      int y = 20;                                     │
│  7      printf("Sum: %d\n", x + y);                     │
│  8      return 0;                                       │
│  9  }                                                   │
├─────────────────────────────────────────────────────────┤
│                   Command Window                        │
│ (gdb) break main                                        │
│ Breakpoint 1 at 0x1149: file test.c, line 5.          │
│ (gdb) run                                               │
│ Starting program: /path/to/program                      │
│ Breakpoint 1, main () at test.c:5                      │
│ 5           int x = 10;                                 │
│ (gdb) _                                                 │
└─────────────────────────────────────────────────────────┘
```

### Starting GDB TUI

**Method 1: Start with TUI mode**
```bash
# Start GDB directly in TUI mode
gdb -tui ./your_program

# With additional arguments
gdb -tui --args ./your_program arg1 arg2
```

**Method 2: Enable TUI from within GDB**
```bash
# Start regular GDB
gdb ./your_program

# Enable TUI mode
(gdb) tui enable
# or use Ctrl+X then Ctrl+A
```

**Method 3: Toggle TUI on/off**
```bash
# Toggle TUI mode
(gdb) tui disable    # Disable TUI
(gdb) tui enable     # Enable TUI

# Keyboard shortcut: Ctrl+X, then Ctrl+A
```

### TUI Window Layouts

**Available TUI Layouts:**
```bash
# Show available layouts
(gdb) help layout

# Source code layout (default)
(gdb) layout src

# Assembly layout
(gdb) layout asm

# Source and assembly split
(gdb) layout split

# Registers layout
(gdb) layout regs

# Next/previous layout
(gdb) layout next
(gdb) layout prev
```

### TUI Navigation and Controls

**Essential TUI Commands:**
```bash
# Window management
(gdb) focus src      # Focus on source window
(gdb) focus cmd      # Focus on command window
(gdb) focus asm      # Focus on assembly window
(gdb) focus regs     # Focus on registers window

# Scroll in source window
(gdb) up             # Scroll up
(gdb) down           # Scroll down

# Refresh display (if corrupted)
(gdb) refresh
# or Ctrl+L
```

**TUI Keyboard Shortcuts:**
| Key Combination | Action |
|----------------|--------|
| `Ctrl+X, Ctrl+A` | Toggle TUI mode |
| `Ctrl+X, Ctrl+1` | Single window layout |
| `Ctrl+X, Ctrl+2` | Two window layout |
| `Ctrl+X, O` | Change active window |
| `Ctrl+P` | Previous command |
| `Ctrl+N` | Next command |
| `Page Up/Down` | Scroll source window |
| `Ctrl+L` | Refresh screen |

### Practical TUI Debugging Session

Let's create a sample program and debug it with TUI:

**Sample Program (debug_example.c):**
```c
#include <stdio.h>
#include <stdlib.h>

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    int numbers[] = {3, 5, 7, 0};
    int size = sizeof(numbers) / sizeof(numbers[0]);
    
    printf("Factorial calculations:\n");
    
    for (int i = 0; i < size; i++) {
        int num = numbers[i];
        int result = factorial(num);
        printf("factorial(%d) = %d\n", num, result);
    }
    
    return 0;
}
```

**Complete TUI Debugging Session:**
```bash
# Compile with debug information
gcc -g -o debug_example debug_example.c

# Start GDB with TUI
gdb -tui ./debug_example

# In GDB TUI mode:
(gdb) break main
(gdb) break factorial
(gdb) run

# Use different layouts
(gdb) layout split     # See source and assembly
(gdb) layout regs      # Add registers view

# Step through code
(gdb) next             # Next line
(gdb) step             # Step into functions
(gdb) continue         # Continue execution

# Examine variables
(gdb) print num
(gdb) print result
(gdb) info locals

# Watch expressions
(gdb) watch n          # Watch variable 'n' in factorial

# Navigate call stack
(gdb) backtrace
(gdb) frame 1          # Switch to different frame
```

### TUI Customization

**Configure TUI behavior:**
```bash
# In .gdbinit file
set tui border-kind ascii     # Use ASCII characters for borders
set tui active-border-mode bold-standout
set tui border-mode reverse

# Custom TUI startup
define tui-start
    tui enable
    layout split
    focus src
end
```

## Graphical GDB Frontends

### DDD (Data Display Debugger)

DDD is a popular graphical frontend for GDB that provides a comprehensive visual debugging environment.

#### Installing DDD

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ddd
```

**CentOS/RHEL:**
```bash
sudo yum install ddd
# or
sudo dnf install ddd
```

**macOS:**
```bash
brew install ddd
```

#### Using DDD

**Starting DDD:**
```bash
# Basic usage
ddd ./your_program

# With arguments
ddd --args ./your_program arg1 arg2

# Attach to running process
ddd -p <process_id>
```

**DDD Interface Overview:**
```
┌─────────────────────────────────────────────────────────────┐
│ File  Edit  View  Program  Commands  Status  Source  Help  │
├─────────────────────────────────────────────────────────────┤
│ [Run] [Break] [Step] [Next] [Finish] [Up] [Down]          │
├─────────────────────────────────────────────────────────────┤
│                    Source Code Pane                        │
│  1  #include <stdio.h>                                     │
│  2  #include <stdlib.h>                                    │
│  3                                                         │
│  4  int main() {                                           │
│  5►     int x = 10;    ← Breakpoint indicator              │
│  6      int y = 20;                                        │
│  7      printf("Sum: %d\n", x + y);                        │
├─────────────────────────────────────────────────────────────┤
│              Data Display Window                           │
│  x = 10                                                    │
│  y = <not accessible>                                      │
├─────────────────────────────────────────────────────────────┤
│              Console/Command Window                        │
│ (gdb) break main                                           │
│ Breakpoint 1 at 0x1149: file test.c, line 5.             │
└─────────────────────────────────────────────────────────────┘
```

**DDD Key Features:**

1. **Visual Breakpoint Management**
```bash
# Click on line numbers to set breakpoints
# Right-click for breakpoint options:
# - Temporary breakpoint
# - Conditional breakpoint
# - Breakpoint properties
```

2. **Data Visualization**
```bash
# Display variables graphically
(gdb) display x
(gdb) display array[0]@10    # Display array elements

# Create data displays by right-clicking variables
# DDD shows pointers, structures, and arrays visually
```

3. **Machine Code View**
```bash
# View → Machine Code Window
# Shows assembly alongside source code
# Useful for low-level debugging
```

### VS Code with GDB Integration

VS Code provides excellent GDB integration through extensions.

#### Setting up VS Code for GDB

**Required Extensions:**
- C/C++ (Microsoft)
- C/C++ Extension Pack
- Native Debug (optional, for enhanced debugging)

**Launch Configuration (.vscode/launch.json):**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "GDB Debug",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/debug_example",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "C/C++: gcc build active file",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

**Build Configuration (.vscode/tasks.json):**
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "type": "cppbuild",
            "label": "C/C++: gcc build active file",
            "command": "/usr/bin/gcc",
            "args": [
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "options": {
                "cwd": "${workspaceFolder}"
            },
            "problemMatcher": ["$gcc"],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

### CLion Integration

CLion provides built-in GDB integration with advanced debugging features.

**CLion Debugging Features:**
```cpp
// Example: Debugging with CLion
#include <iostream>
#include <vector>
#include <memory>

class DebugExample {
private:
    std::vector<int> data;
    std::unique_ptr<int> ptr;
    
public:
    DebugExample() : ptr(std::make_unique<int>(42)) {
        data = {1, 2, 3, 4, 5};
    }
    
    void process() {
        // Set breakpoint here - CLion shows:
        // - Variable values in tooltips
        // - Memory view for pointers
        // - STL container contents
        for (auto& item : data) {
            item *= 2;  // Watch this transformation
        }
        
        *ptr += 10;     // Examine smart pointer content
    }
};

int main() {
    DebugExample example;
    example.process();
    return 0;
}
```

### Advanced GUI Debugging Techniques

#### Memory Visualization

**In DDD:**
```bash
# Memory window
(gdb) x/10x &array     # Examine memory as hex
# DDD shows this in a graphical memory viewer

# Structure visualization
(gdb) print *struct_ptr
# DDD displays structure members graphically
```

**Complex Data Structure Debugging:**
```c
// Example: Linked list debugging
typedef struct Node {
    int data;
    struct Node* next;
} Node;

Node* create_list() {
    Node* head = malloc(sizeof(Node));
    head->data = 1;
    head->next = malloc(sizeof(Node));
    head->next->data = 2;
    head->next->next = NULL;
    return head;
}

// In GUI debuggers:
// 1. Set breakpoint after list creation
// 2. Display 'head' variable
// 3. GUI shows linked structure visually
// 4. Can follow pointers graphically
```

#### Multi-threaded Debugging

**Thread Management in GDB TUI:**
```bash
# In TUI mode
(gdb) info threads
(gdb) thread 2          # Switch to thread 2
(gdb) layout split      # See source for current thread

# Set thread-specific breakpoints
(gdb) break thread_function thread 2
```

**GUI Frontend Thread Debugging:**
- **DDD**: Thread menu shows all threads
- **VS Code**: Thread panel in debug view
- **CLion**: Threads tool window

## Comparison of Debugging Interfaces

| Feature | GDB CLI | GDB TUI | DDD | VS Code | CLion |
|---------|---------|---------|-----|---------|--------|
| **Learning Curve** | Steep | Moderate | Easy | Easy | Easy |
| **Source Visibility** | No | Yes | Yes | Yes | Yes |
| **Variable Inspection** | Text only | Text only | Graphical | Graphical | Graphical |
| **Memory View** | Commands | Commands | Visual | Limited | Advanced |
| **Multi-file Projects** | Manual | Manual | Good | Excellent | Excellent |
| **Remote Debugging** | Yes | Yes | Yes | Yes | Yes |
| **Customization** | High | Medium | Medium | High | Medium |
| **Performance** | Fast | Fast | Slower | Medium | Medium |

## Best Practices and Tips

### Choosing the Right Interface

**Use GDB CLI when:**
- Remote debugging over SSH
- Scripting debug sessions
- Memory-constrained environments
- Learning GDB fundamentals

**Use GDB TUI when:**
- Need source visibility
- Working in terminal environments
- Quick debugging sessions
- Learning to transition from CLI

**Use GUI frontends when:**
- Complex data structures
- Multi-file projects
- Visual learners
- Long debugging sessions
- Team collaboration

### Productivity Tips

**TUI Efficiency:**
```bash
# Create custom GDB commands
define debug-start
    set confirm off
    file $arg0
    break main
    run
    layout split
end

# Usage: debug-start ./program
```

**GUI Setup:**
```bash
# DDD startup script
#!/bin/bash
# Save as ddd-debug.sh
if [ $# -eq 0 ]; then
    echo "Usage: $0 <program> [args...]"
    exit 1
fi

PROGRAM=$1
shift
ddd --gdb --args $PROGRAM "$@"
```

### Common Issues and Solutions

**TUI Display Problems:**
```bash
# Terminal size issues
export TERM=xterm-256color
export LINES=24
export COLUMNS=80

# If display gets corrupted
(gdb) refresh
# or restart with proper terminal size
```

**GUI Performance Issues:**
```bash
# For large programs, limit symbol loading
(gdb) set auto-solib-add off
(gdb) sharedlibrary <specific_lib>

# Reduce update frequency in GUI
(gdb) set update-frequency 10
```

## Learning Exercises

### Exercise 1: TUI Mastery
```c
// debug_practice.c
#include <stdio.h>
#include <string.h>

void string_reverse(char* str) {
    int len = strlen(str);
    for (int i = 0; i < len/2; i++) {
        char temp = str[i];
        str[i] = str[len-1-i];
        str[len-1-i] = temp;
    }
}

int main() {
    char text[] = "Hello, World!";
    printf("Original: %s\n", text);
    string_reverse(text);
    printf("Reversed: %s\n", text);
    return 0;
}
```

**Tasks:**
1. Compile and debug with GDB TUI
2. Use different layouts (src, split, regs)
3. Step through the string_reverse function
4. Watch variables change in real-time
5. Examine memory layout

### Exercise 2: GUI Debugging
Use the same program with DDD or VS Code:
1. Set conditional breakpoints
2. Visualize the string array
3. Use data display windows
4. Examine assembly code view

### Exercise 3: Complex Data Structures
```c
// tree_debug.c
typedef struct TreeNode {
    int value;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

TreeNode* create_tree() {
    TreeNode* root = malloc(sizeof(TreeNode));
    root->value = 10;
    root->left = malloc(sizeof(TreeNode));
    root->left->value = 5;
    root->left->left = NULL;
    root->left->right = NULL;
    root->right = malloc(sizeof(TreeNode));
    root->right->value = 15;
    root->right->left = NULL;
    root->right->right = NULL;
    return root;
}

// Debug tree traversal and visualization
```

## Summary

Visual debugging interfaces significantly enhance the debugging experience by providing:
- **Immediate visual feedback** on code execution
- **Simplified variable inspection** and data structure visualization
- **Integrated breakpoint management**
- **Multi-pane views** for comprehensive debugging context

Choose the interface that best fits your workflow, environment, and the complexity of your debugging tasks.
