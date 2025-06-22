# Project: Legacy Code Refactoring with Mocks

## Description
Take existing hard-to-test code, apply seam techniques and dependency breaking, and create tests using appropriate mocking strategies.

## Example (C++)
```cpp
// Before refactoring
int calculate() { return getValueFromSystem(); }

// After introducing seam
int calculate(std::function<int()> getValue) { return getValue(); }

// In test
int mockValue() { return 42; }
assert(calculate(mockValue) == 42);
```
