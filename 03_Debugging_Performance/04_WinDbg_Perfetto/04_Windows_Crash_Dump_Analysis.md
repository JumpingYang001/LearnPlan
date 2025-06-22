# Windows Crash Dump Analysis

## Overview
Learn how to analyze Windows crash dumps, understand bug check codes, and resolve system failures.

## Example: Generating a Crash Dump
```c
#include <windows.h>
#include <DbgHelp.h>
#include <stdio.h>

int main() {
    // Simulate a crash
    int* p = NULL;
    *p = 1;
    return 0;
}
```

*Run this program to generate a crash, then analyze the dump file in WinDbg.*
