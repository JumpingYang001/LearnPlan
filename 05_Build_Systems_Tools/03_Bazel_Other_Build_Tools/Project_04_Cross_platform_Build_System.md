# Project: Cross-platform Build System

## Description
Build a system that works across multiple operating systems, implement conditional compilation and platform-specific code, and create unified testing and packaging.

## C++ Example: Platform-specific Source
```python
# BUILD file
cc_binary(
    name = "platform_app",
    srcs = select({
        "@platforms//os:windows": ["main_win.cpp"],
        "@platforms//os:linux": ["main_linux.cpp"],
        "//conditions:default": ["main.cpp"],
    }),
)
```
