# Test Doubles and Isolation

## Description
Understand stubs, mocks, fakes, and spies. Learn about dependency injection and isolation techniques for unit testing. Implement TDD with test doubles.

## Example
```python
# Example: Using unittest.mock for test doubles
from unittest.mock import Mock
class Service:
    def fetch(self):
        pass
class Client:
    def __init__(self, service):
        self.service = service
    def get_data(self):
        return self.service.fetch()

def test_client_uses_service():
    mock_service = Mock()
    mock_service.fetch.return_value = 'data'
    client = Client(mock_service)
    assert client.get_data() == 'data'
```
