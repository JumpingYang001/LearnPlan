# Mocking and Patching

## Description
Covers monkeypatch, unittest.mock integration, spy objects, and mocking complex behaviors.

## Example
```python
def test_monkeypatch(monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: "/tmp")
    import os
    assert os.getcwd() == "/tmp"
```
