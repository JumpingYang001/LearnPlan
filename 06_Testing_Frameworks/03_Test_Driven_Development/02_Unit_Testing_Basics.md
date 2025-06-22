# Unit Testing Basics for TDD

## Description
Master unit test structure and organization. Learn about test assertions, matchers, naming conventions, and readability. Implement basic unit tests for TDD.

## Example
```python
import unittest
class TestMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(1 + 1, 2)
    def test_subtract(self):
        self.assertEqual(5 - 3, 2)
```
