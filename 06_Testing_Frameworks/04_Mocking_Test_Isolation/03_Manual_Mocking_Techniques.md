# Manual Mocking Techniques

## Description
Manual mocking involves creating your own test doubles without a framework. This is useful for understanding the mechanics of mocking and for simple cases.

## Example (C++)
```cpp
class ICalculator {
public:
    virtual int add(int a, int b) = 0;
};
class MockCalculator : public ICalculator {
public:
    int add(int a, int b) override { return 42; }
};
```
