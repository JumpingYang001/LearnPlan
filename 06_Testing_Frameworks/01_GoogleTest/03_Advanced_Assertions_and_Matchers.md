# Advanced Assertions and Matchers

## Explanation
Learn to use advanced assertions and GoogleMock matchers for complex validations. This section also covers creating custom matchers and using string/container-specific assertions.

## Example Code
```cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::HasSubstr;

TEST(StringTest, Substring) {
    EXPECT_THAT("Hello, world!", HasSubstr("world"));
}
```
