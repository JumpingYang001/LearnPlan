# Test Isolation Fundamentals

## Description
Test isolation ensures that each unit test runs independently, without relying on or affecting other tests. This is achieved by isolating the code under test from its dependencies, making tests more reliable and easier to debug.

## Example (Python)
```python
class Calculator:
    def add(self, a, b):
        return a + b

def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5
```
