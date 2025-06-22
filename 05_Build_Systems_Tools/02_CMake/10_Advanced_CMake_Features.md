# Advanced CMake Features

## Custom Commands and Targets
- add_custom_command, add_custom_target

### Example: Custom Command
```cmake
add_custom_command(
  OUTPUT generated.c
  COMMAND python gen.py > generated.c
  DEPENDS gen.py
)
add_executable(myexe generated.c)
```
