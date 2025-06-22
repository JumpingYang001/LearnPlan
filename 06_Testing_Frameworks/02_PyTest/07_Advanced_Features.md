# Advanced PyTest Features

## Description
Explores fixture scopes, parallel execution, custom collection, and reporters.

## Example
```python
import pytest

@pytest.fixture(scope="session")
def db():
    return "db_connection"
```
