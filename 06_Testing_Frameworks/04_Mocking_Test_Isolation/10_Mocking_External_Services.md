# Mocking External Services

## Description
Mocking external services like HTTP APIs or databases allows tests to run without real network or database dependencies.

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
