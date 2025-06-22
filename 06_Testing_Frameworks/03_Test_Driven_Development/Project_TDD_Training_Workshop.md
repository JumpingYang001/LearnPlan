# Project: TDD Training Workshop Materials

## Description
Develop materials for teaching TDD. Implement example projects with step-by-step TDD guidance. Create exercises and solutions for TDD practice.

## Example: Step-by-Step TDD Exercise (Python)
```python
# Step 1: Write a failing test
import unittest
class TestAdder(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
# Step 2: Implement minimal code
def add(a, b):
    return a + b
# Step 3: Refactor (if needed)
```
