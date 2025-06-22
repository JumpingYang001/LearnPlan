# Death Tests and Exception Testing

## Explanation
Death tests check code that is expected to crash, while exception tests verify thrown exceptions and their messages.

## Example Code
```cpp
#include <gtest/gtest.h>

void Crash() {
    abort();
}

TEST(DeathTest, Crashes) {
    ASSERT_DEATH(Crash(), "");
}

TEST(ExceptionTest, Throws) {
    EXPECT_THROW(throw std::runtime_error("error"), std::runtime_error);
}
```
