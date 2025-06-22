# TDD for Legacy Code

## Description
Add tests to untested code, use characterization tests, and refactor legacy code safely. Apply TDD in brownfield projects.

## Example
```python
# Example: Characterization test for legacy function
import unittest
def legacy_sum(a, b):
    return a + b  # Legacy code
class TestLegacySum(unittest.TestCase):
    def test_legacy_sum(self):
        self.assertEqual(legacy_sum(2, 3), 5)
```
