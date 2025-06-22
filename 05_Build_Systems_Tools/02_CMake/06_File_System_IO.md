# File System and I/O Operations in CMake

## File System Commands
- file(GLOB), file(READ/WRITE), file(DOWNLOAD)

### Example: GLOB Sources
```cmake
file(GLOB SOURCES "src/*.c")
add_executable(myexe ${SOURCES})
```

## File Generation
- configure_file, file(GENERATE)
