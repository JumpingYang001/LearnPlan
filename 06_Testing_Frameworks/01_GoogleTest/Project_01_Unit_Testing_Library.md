# Project 1: Unit Testing Library

## Description
Create a small utility library with comprehensive GoogleTest coverage. Implement fixtures, parameterized tests, and custom matchers. Generate test reports and analyze coverage.

## Example Code
```cpp
// math_utils.h
#pragma once
int add(int a, int b);

// math_utils.cpp
#include "math_utils.h"
int add(int a, int b) { return a + b; }

// test_math_utils.cpp
#include <gtest/gtest.h>
#include "math_utils.h"

class MathTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(MathTest, Add) {
    int a, b, expected;
    std::tie(a, b, expected) = GetParam();
    EXPECT_EQ(add(a, b), expected);
}
INSTANTIATE_TEST_SUITE_P(AddTests, MathTest, ::testing::Values(
    std::make_tuple(1, 2, 3),
    std::make_tuple(-1, 1, 0),
    std::make_tuple(0, 0, 0)
));
```
