# Project: API Testing Framework

## Description
Build a test suite for a REST API using PyTest, with fixtures for authentication and parameterized endpoint tests.

## Example
```python
import pytest
import requests

@pytest.fixture
def auth_token():
    return "fake-token"

def test_get_users(auth_token):
    response = requests.get("https://api.example.com/users", headers={"Authorization": f"Bearer {auth_token}"})
    assert response.status_code == 200

@pytest.mark.parametrize("endpoint", ["users", "posts"])
def test_endpoints(auth_token, endpoint):
    url = f"https://api.example.com/{endpoint}"
    response = requests.get(url, headers={"Authorization": f"Bearer {auth_token}"})
    assert response.ok
```
