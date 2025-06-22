# Project: Custom GCC/Clang Plugin

## Goal
Implement a compiler plugin for static analysis to detect a specific class of bugs.

## Example: GCC Plugin (C)
```c
// Example: Simple GCC plugin skeleton
#include <gcc-plugin.h>
#include <plugin-version.h>
int plugin_init(struct plugin_name_args *plugin_info, struct plugin_gcc_version *version) {
    // Plugin logic here
    return 0;
}
```

## Example: Clang Tool (C++)
```cpp
// Example: Clang LibTooling skeleton
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
int main(int argc, const char **argv) {
    // Tool logic here
    return 0;
}
```
