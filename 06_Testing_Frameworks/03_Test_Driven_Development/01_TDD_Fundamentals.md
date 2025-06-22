# TDD Fundamentals

## Description
Understand the TDD cycle: Red-Green-Refactor. Learn about TDD principles, benefits, and the mindset shift required for effective TDD. Compare TDD with traditional development.

## Example
```python
# Example: Simple TDD Cycle in Python
# Step 1: Write a failing test
import unittest
class TestAdder(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

# Step 2: Implement minimal code
def add(a, b):
    return a + b

# Step 3: Refactor (if needed)
# Code is already clean in this simple case.
```
