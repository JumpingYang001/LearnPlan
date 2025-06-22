# Project: Legacy Code Transformation

## Description
Take an existing codebase with minimal testing. Apply TDD techniques to add tests and refactor. Document the transformation process and improvements.

## Example: Adding Tests to Legacy Code (Python)
```python
# legacy.py
def add(a, b):
    return a + b
# test_legacy.py
import unittest
from legacy import add
class TestLegacy(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
# Refactor legacy.py as needed, keeping tests green.
```
