# TDD in Practice - Basic Examples

## Description
Write the first test, implement minimal code, and refactor while preserving behavior. Practice TDD for simple algorithms and functions.

## Example
```python
# Example: FizzBuzz with TDD
import unittest
class TestFizzBuzz(unittest.TestCase):
    def test_fizzbuzz(self):
        self.assertEqual(fizzbuzz(3), 'Fizz')
        self.assertEqual(fizzbuzz(5), 'Buzz')
        self.assertEqual(fizzbuzz(15), 'FizzBuzz')
        self.assertEqual(fizzbuzz(2), '2')

def fizzbuzz(n):
    if n % 15 == 0:
        return 'FizzBuzz'
    if n % 3 == 0:
        return 'Fizz'
    if n % 5 == 0:
        return 'Buzz'
    return str(n)
```
