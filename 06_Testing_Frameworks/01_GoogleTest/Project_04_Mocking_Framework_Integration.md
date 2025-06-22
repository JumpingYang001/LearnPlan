# Project 4: Mocking Framework Integration

## Description
Build a system with external dependencies. Use GoogleMock to isolate components for testing. Create a comprehensive test suite with mocks and stubs.

## Example Code
```cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>

class IDatabase {
public:
    virtual ~IDatabase() = default;
    virtual int getData() = 0;
};

class MockDatabase : public IDatabase {
public:
    MOCK_METHOD(int, getData, (), (override));
};

int process(IDatabase* db) {
    return db->getData() * 2;
}

TEST(MockTest, Process) {
    MockDatabase mock;
    EXPECT_CALL(mock, getData()).WillOnce(::testing::Return(21));
    EXPECT_EQ(process(&mock), 42);
}
```
