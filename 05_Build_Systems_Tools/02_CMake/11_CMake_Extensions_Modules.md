# CMake Extensions and Modules

## Writing CMake Modules
- Module structure
- Namespacing

### Example: Custom Module
```cmake
# MyModule.cmake
function(print_hello)
  message(STATUS "Hello from module!")
endfunction()
```
