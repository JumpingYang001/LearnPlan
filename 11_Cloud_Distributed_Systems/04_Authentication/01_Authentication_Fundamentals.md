# Authentication Fundamentals

## Concepts and Principles
- Authentication verifies identity using credentials.
- Common factors: something you know (password), have (token), or are (biometrics).

## Example: Basic Username/Password Authentication (Python)
```python
def authenticate(username, password):
    # Example: hardcoded credentials
    return username == 'admin' and password == 'secret'

print(authenticate('admin', 'secret'))  # True
```
