# Fixtures

## Description
Details the fixture concept, creation, usage, and built-in fixtures like capsys, monkeypatch, and tmpdir.

## Example
```python
import pytest

@pytest.fixture
def sample_data():
    return [1, 2, 3]

def test_sum(sample_data):
    assert sum(sample_data) == 6
```
