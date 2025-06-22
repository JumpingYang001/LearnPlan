# Advanced Mocking Patterns

## Description
Advanced patterns include interaction-based testing, designing for mockability, and using different schools of TDD (London vs. Chicago).

## Example (Python)
```python
from unittest.mock import Mock
service = Mock()
service.process.return_value = 'done'
assert service.process('input') == 'done'
```
