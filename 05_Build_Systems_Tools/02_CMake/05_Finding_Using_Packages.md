# Finding and Using Packages

## find_package Command
- Module mode vs. Config mode
- VERSION and COMPONENTS

### Example: Find Threads
```cmake
find_package(Threads REQUIRED)
target_link_libraries(myexe PRIVATE Threads::Threads)
```

## Creating Config Packages
- PackageConfig.cmake files
- Exporting targets
