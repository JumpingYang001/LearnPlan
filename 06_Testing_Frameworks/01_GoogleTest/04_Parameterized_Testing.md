# Parameterized Testing

## Explanation
Parameterized tests allow running the same test logic with different values. This section covers value-parameterized and type-parameterized tests, as well as test generators.

## Example Code
```cpp
#include <gtest/gtest.h>

class MyParamTest : public ::testing::TestWithParam<int> {};

TEST_P(MyParamTest, IsEven) {
    EXPECT_EQ(GetParam() % 2, 0);
}

INSTANTIATE_TEST_SUITE_P(EvenNumbers, MyParamTest, ::testing::Values(2, 4, 6));
```
