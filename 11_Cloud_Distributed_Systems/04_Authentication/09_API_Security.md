# API Security

## Best Practices
- Use strong authentication (OAuth 2.0, API keys)
- Rate limiting and throttling

## Example: API Key Check (Python)
```python
API_KEYS = {'key1', 'key2'}
def is_valid_key(key):
    return key in API_KEYS
print(is_valid_key('key1'))  # True
```
