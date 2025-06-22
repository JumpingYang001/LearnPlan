# Project: Microservices Testing Suite

## Description
Build tests for a microservices architecture, mock dependencies, and implement contract tests.

## Example
```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def service_b():
    mock = Mock()
    mock.get_data.return_value = {"id": 1, "value": "test"}
    return mock

def test_service_a(service_b):
    result = service_b.get_data()
    assert result["value"] == "test"
```
