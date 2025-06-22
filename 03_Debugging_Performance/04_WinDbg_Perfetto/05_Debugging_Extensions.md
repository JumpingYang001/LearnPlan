# Debugging Extensions in WinDbg

## Overview
Understand and use WinDbg extensions like SOS and SOSEX for enhanced debugging.

## Example: Using SOS Extension
```csharp
// Example for .NET application
using System;

class Program {
    static void Main() {
        Console.WriteLine("Hello from .NET");
        Console.ReadLine(); // Attach WinDbg and use !clrstack
    }
}
```

*Attach WinDbg, load SOS with `.loadby sos clr`, and use `!clrstack` to view the .NET stack trace.*
