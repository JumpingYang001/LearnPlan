# Project 3: Custom Build Tool Integration

## Goal
Integrate code generation tools, create custom build steps, and ensure proper dependency tracking in CMake.

### Example: CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)
project(CustomBuildTool)
add_custom_command(
  OUTPUT generated.c
  COMMAND python gen.py > generated.c
  DEPENDS gen.py
)
add_executable(mygenapp generated.c)
```

```python
# gen.py
print("#include <stdio.h>\nint main() { printf(\"Generated!\\n\"); return 0; }")
```
