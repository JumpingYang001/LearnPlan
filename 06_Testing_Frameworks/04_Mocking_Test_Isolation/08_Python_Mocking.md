# Python Mocking

## Description
Python's unittest.mock and pytest-mock provide tools for creating mocks, stubs, and spies in Python tests.

## Example (Python)
```python
from unittest.mock import Mock
calc = Mock()
calc.add.return_value = 5
assert calc.add(2, 3) == 5
```
