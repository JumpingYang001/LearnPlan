# GoogleTest Basics

## Explanation
This section covers the basic structure of GoogleTest, including simple assertions and the difference between fatal and non-fatal assertions. You will learn how to create and run basic test cases.

## Example Code
```cpp
#include <gtest/gtest.h>

TEST(SampleTest, BasicAssertions) {
    EXPECT_EQ(1, 1); // Non-fatal assertion
    ASSERT_TRUE(true); // Fatal assertion
}
```
