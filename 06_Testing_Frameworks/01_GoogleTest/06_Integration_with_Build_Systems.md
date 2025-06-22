# Integration with Build Systems

## Explanation
This section explains how to integrate GoogleTest with CMake, set up test discovery, configure output formats, and implement test filtering and sharding.

## Example Code
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(MyTests)
find_package(GTest REQUIRED)
add_executable(runTests test.cpp)
target_link_libraries(runTests GTest::gtest_main)
enable_testing()
add_test(NAME runTests COMMAND runTests)
```
