# Project 2: Legacy Code Testing

## Description
Take an existing codebase with minimal testing. Implement a test suite with GoogleTest. Refactor as needed to improve testability.

## Example Code
```cpp
// legacy_code.h
#pragma once
int legacy_sum(int* arr, int size);

// legacy_code.cpp
#include "legacy_code.h"
int legacy_sum(int* arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) sum += arr[i];
    return sum;
}

// test_legacy_code.cpp
#include <gtest/gtest.h>
#include "legacy_code.h"

TEST(LegacyTest, SumArray) {
    int arr[] = {1, 2, 3};
    EXPECT_EQ(legacy_sum(arr, 3), 6);
}
```
