# Project: Profile-Guided Optimization Framework

## Description
Build a framework for automating PGO workflows, including instrumentation, profile collection, and visualization of hot paths.

## Example (C/C++)
```c
// Example: Instrumented function
int compute(int x) {
    if (x > 0) return x * 2;
    else return -x;
}
```
// Compile with:
// gcc -fprofile-generate -o prog prog.c
// Run the program to generate profile data
// Recompile with:
// gcc -fprofile-use -o prog_opt prog.c
// Visualize hot paths using gprof or similar tools.
