# Project: External API Client with Tests

## Description
Develop a client for an external API with comprehensive tests. Use mocking to simulate HTTP responses and errors, ensuring the test suite does not require the actual API.

## Example (Python)
```python
import requests
from unittest.mock import patch

def get_data():
    return requests.get('https://api.example.com/data').json()

with patch('requests.get') as mock_get:
    mock_get.return_value.json.return_value = {'key': 'value'}
    assert get_data() == {'key': 'value'}
```
