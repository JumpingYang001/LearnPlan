# TDD for Object-Oriented Design

## Description
Apply TDD for class design, interfaces, responsibility-driven design, interface segregation, and dependency inversion. Implement TDD for class hierarchies and relationships.

## Example
```python
# Example: TDD for a simple class
import unittest
class Calculator:
    def add(self, a, b):
        return a + b
class TestCalculator(unittest.TestCase):
    def test_add(self):
        calc = Calculator()
        self.assertEqual(calc.add(2, 3), 5)
```
