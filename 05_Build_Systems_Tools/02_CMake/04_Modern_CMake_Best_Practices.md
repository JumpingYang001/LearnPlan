# Modern CMake Best Practices

## Target-Based Design
- Use targets as self-contained units
- Prefer target properties over global settings

### Example: Target Properties
```cmake
add_library(mylib STATIC mylib.c)
target_compile_features(mylib PUBLIC c_std_11)
```

## Generator Expressions
- Conditional expressions
- Platform/config selection

### Example: Generator Expression
```cmake
target_compile_definitions(mylib PUBLIC $<$<CONFIG:Debug>:DEBUG_BUILD>)
```
