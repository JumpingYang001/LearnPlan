# Advanced TDD Patterns and Practices

## Description
Understand test-driven architectural patterns, outside-in development (London school), Detroit/classicist approach, and advanced TDD techniques.

## Example
```python
# Example: Outside-in TDD (London school)
# Start with a high-level test and use mocks for dependencies
from unittest.mock import Mock
class PaymentProcessor:
    def __init__(self, gateway):
        self.gateway = gateway
    def pay(self, amount):
        return self.gateway.charge(amount)
def test_payment_processor():
    mock_gateway = Mock()
    mock_gateway.charge.return_value = True
    processor = PaymentProcessor(mock_gateway)
    assert processor.pay(100) is True
```
