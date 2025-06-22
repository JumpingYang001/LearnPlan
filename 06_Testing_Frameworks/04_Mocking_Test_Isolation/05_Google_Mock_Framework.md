# Google Mock (gmock) Framework

## Description
Google Mock is a C++ framework for creating mock classes and setting expectations on their behavior in tests.

## Example (C++)
```cpp
#include <gmock/gmock.h>
class MockCalculator : public ICalculator {
public:
    MOCK_METHOD(int, add, (int, int), (override));
};
```
