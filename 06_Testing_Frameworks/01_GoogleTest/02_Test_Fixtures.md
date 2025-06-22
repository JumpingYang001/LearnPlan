# Test Fixtures

## Explanation
Test fixtures help in reusing code for multiple tests by setting up common objects and cleaning them up after tests. This section explains SetUp() and TearDown() methods and how to create test suites with shared resources.

## Example Code
```cpp
#include <gtest/gtest.h>

class MyFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Code here will be called before each test
    }
    void TearDown() override {
        // Code here will be called after each test
    }
};

TEST_F(MyFixture, Test1) {
    EXPECT_TRUE(true);
}
```
