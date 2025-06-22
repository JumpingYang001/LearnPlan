# CMake Language Basics

## Variables and Properties
- Setting and accessing variables
- Cache variables
- Environment variables
- Target properties

### Example: Setting Variables
```cmake
set(MY_VAR "Hello")
message(STATUS "MY_VAR = ${MY_VAR}")
```

## Control Structures
- if/else/endif
- foreach/endforeach
- while/endwhile

### Example: if/else
```cmake
set(VALUE 1)
if(VALUE EQUAL 1)
    message(STATUS "Value is 1")
else()
    message(STATUS "Value is not 1")
endif()
```
